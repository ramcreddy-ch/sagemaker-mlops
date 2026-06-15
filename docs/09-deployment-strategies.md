# Section 09 — Deployment Strategies

> **Production Context**: Capital One serves 500M fraud inferences/day. JPMorgan scores 50M customers in nightly batch. Goldman Sachs LLM handles async research queries. Each use case needs a different deployment strategy.

---

## 9.1 Deployment Strategy Selection Guide

```
DECISION TREE — WHICH ENDPOINT TYPE TO USE?

Is the use case REAL-TIME? (< 1 second response needed)
├── YES: Use Real-Time Endpoint
│   ├── Traffic < 10 TPS and sporadic? → Serverless Inference
│   ├── Traffic consistent > 10 TPS?   → Real-Time Endpoint
│   └── Need to host 50+ models?       → Multi-Model Endpoint (MME)
│
└── NO: Is response needed within minutes?
    ├── YES (1 min - 15 min):  → Async Endpoint
    │   ├── LLM inference (70B model)
    │   ├── Document processing
    │   └── Complex pipelines
    └── NO (hours/overnight):  → Batch Transform
        ├── Nightly credit scoring (50M customers)
        ├── Weekly churn prediction
        └── Monthly portfolio risk assessment
```

---

## 9.2 Real-Time Endpoints — Production Setup

```python
# code/deployment/realtime_endpoint.py
"""
Production real-time endpoint deployment for fraud detection.
Target: P99 latency < 10ms, 99.99% uptime.
"""

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import json
import time
import logging

logger = logging.getLogger(__name__)
session = sagemaker.Session()
sm = boto3.client('sagemaker', region_name='us-east-1')
region = 'us-east-1'
account_id = '123456789012'
role = 'arn:aws:iam::123456789012:role/SageMakerExecutionRole-Inference'


def deploy_fraud_detection_endpoint(
    model_package_arn: str,
    endpoint_name: str = 'fraud-detection-prod',
    instance_type: str = 'ml.c5.2xlarge',
    initial_instance_count: int = 3,
) -> Predictor:
    """
    Deploy fraud detection model to real-time endpoint.
    """
    
    image_uri = sagemaker.image_uris.retrieve(
        framework='xgboost',
        region=region,
        version='1.7-1',
    )
    
    # Create model from registry
    model = Model(
        image_uri=image_uri,
        model_data=model_package_arn,
        role=role,
        sagemaker_session=session,
        name=f"fraud-detection-model-{int(time.time())}",
        
        # VPC configuration — REQUIRED in production
        vpc_config={
            'SecurityGroupIds': ['sg-sagemaker-inference'],
            'Subnets': ['subnet-ml-a', 'subnet-ml-b', 'subnet-ml-c'],
        },
        
        # Environment variables for the inference container
        env={
            'SAGEMAKER_PROGRAM': 'inference.py',
            'FRAUD_THRESHOLD': '0.85',
            'FEATURE_STORE_GROUP': 'fraud-features-v3',
            'LOG_LEVEL': 'INFO',
        },
    )
    
    # Endpoint configuration with data capture (for Model Monitor)
    endpoint_config_name = f"fraud-detection-config-{int(time.time())}"
    
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model.name,
            'InstanceType': instance_type,
            'InitialInstanceCount': initial_instance_count,
            'InitialVariantWeight': 1.0,
            'RoutingConfig': {
                'RoutingStrategy': 'LEAST_OUTSTANDING_REQUESTS',  # better than round-robin
            },
        }],
        DataCaptureConfig={
            'EnableCapture': True,
            'InitialSamplingPercentage': 20,  # capture 20% of traffic for monitoring
            'DestinationS3Uri': f's3://fintech-datalake-prod/monitoring/data-capture/fraud-detection/',
            'KmsKeyId': f'arn:aws:kms:{region}:{account_id}:key/mrk-monitoring-prod',
            'CaptureOptions': [
                {'CaptureMode': 'Input'},
                {'CaptureMode': 'Output'},
            ],
            'CaptureContentTypeHeader': {
                'CsvContentTypes': ['text/csv'],
                'JsonContentTypes': ['application/json'],
            },
        },
        KmsKeyId=f'arn:aws:kms:{region}:{account_id}:key/mrk-inference-prod',
        Tags=[
            {'Key': 'environment', 'Value': 'production'},
            {'Key': 'model', 'Value': 'fraud-detection'},
            {'Key': 'cost-center', 'Value': 'fraud-prevention'},
        ],
    )
    
    # Create or update endpoint
    existing_endpoints = sm.list_endpoints(NameContains=endpoint_name)['Endpoints']
    
    if existing_endpoints:
        logger.info(f"Updating existing endpoint: {endpoint_name}")
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
    else:
        logger.info(f"Creating new endpoint: {endpoint_name}")
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
    
    # Wait for endpoint to be in service
    waiter = sm.get_waiter('endpoint_in_service')
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={'Delay': 15, 'MaxAttempts': 80},  # 20 min max
    )
    
    logger.info(f"✅ Endpoint {endpoint_name} is in service")
    
    # Set up auto-scaling
    _configure_autoscaling(endpoint_name, 'AllTraffic', min_capacity=3, max_capacity=50)
    
    return Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=session,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
    )


def _configure_autoscaling(
    endpoint_name: str,
    variant_name: str,
    min_capacity: int = 3,
    max_capacity: int = 50,
):
    """Configure auto-scaling for the endpoint."""
    
    aas = boto3.client('application-autoscaling')
    
    resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
    
    # Register scalable target
    aas.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity,
    )
    
    # Scale-out policy: add instances when invocations per instance > 1000
    aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scale-out",
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 1000.0,    # target invocations per instance
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance',
            },
            'ScaleInCooldown': 300,    # 5 min cool-down before scale-in
            'ScaleOutCooldown': 60,    # 1 min cool-down before scale-out
        },
    )
    
    logger.info(f"Auto-scaling configured: {min_capacity}-{max_capacity} instances")
    logger.info(f"Target: 1000 invocations/instance")


def test_endpoint(endpoint_name: str):
    """Smoke test the deployed endpoint."""
    runtime = boto3.client('sagemaker-runtime')
    
    # Sample transaction features
    test_payload = "250.00,0,14,0,1200.50,720,365,0"
    
    import time
    start = time.time()
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Accept='application/json',
        Body=test_payload,
    )
    latency_ms = (time.time() - start) * 1000
    
    result = json.loads(response['Body'].read())
    
    print(f"✅ Endpoint test passed")
    print(f"   Latency: {latency_ms:.1f}ms")
    print(f"   Response: {result}")
    
    return latency_ms, result
```

---

## 9.3 Async Endpoints — LLM & Heavy Processing

```python
# code/deployment/async_endpoint.py
"""
Async endpoint for LLM inference (70B+ models).
Requests are queued, processed asynchronously, output written to S3.
"""

import boto3
import sagemaker
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.model import Model
import json, time, uuid

session = sagemaker.Session()
sm = boto3.client('sagemaker')
sagemaker_runtime = boto3.client('sagemaker-runtime')

def deploy_llm_async_endpoint(
    model_data_s3: str,
    endpoint_name: str = 'llm-financial-assistant-async',
    instance_type: str = 'ml.g5.48xlarge',
):
    """
    Deploy LLM to async endpoint.
    ml.g5.48xlarge = 8x NVIDIA A10G (192GB total GPU memory)
    Suitable for Llama 3 70B with FP16 quantization.
    """
    
    async_config = AsyncInferenceConfig(
        output_path='s3://fintech-datalake-prod/llm-outputs/',
        max_concurrent_invocations_per_instance=4,   # 4 concurrent requests per GPU
        
        # SNS notifications on completion/failure
        notification_config={
            'SuccessTopic': 'arn:aws:sns:us-east-1:123456789:llm-inference-success',
            'ErrorTopic': 'arn:aws:sns:us-east-1:123456789:llm-inference-error',
        },
    )
    
    model = Model(
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu22.04",
        model_data=model_data_s3,
        role='arn:aws:iam::123456789:role/SageMakerExecutionRole-LLM',
        env={
            'HF_MODEL_ID': 'meta-llama/Meta-Llama-3-70B-Instruct',
            'HF_TASK': 'text-generation',
            'SM_NUM_GPUS': '8',
            'MAX_INPUT_LENGTH': '8000',
            'MAX_TOTAL_TOKENS': '10000',
            'MAX_BATCH_TOTAL_TOKENS': '50000',
            'QUANTIZE': 'bitsandbytes',   # 4-bit quantization to fit in memory
            'DTYPE': 'float16',
        },
    )
    
    predictor = model.deploy(
        instance_type=instance_type,
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        async_inference_config=async_config,
    )
    
    return predictor


def invoke_async(endpoint_name: str, prompt: str, request_id: str = None) -> str:
    """
    Invoke async endpoint and return the output location.
    Caller polls S3 or waits for SNS notification.
    """
    if not request_id:
        request_id = str(uuid.uuid4())
    
    input_location = f"s3://fintech-datalake-prod/llm-inputs/{request_id}.json"
    
    # Write input to S3
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket='fintech-datalake-prod',
        Key=f'llm-inputs/{request_id}.json',
        Body=json.dumps({
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': 2048,
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True,
                'return_full_text': False,
            }
        }).encode()
    )
    
    # Invoke async endpoint
    response = sagemaker_runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        ContentType='application/json',
        InputLocation=input_location,
        RequestTTLSeconds=900,    # 15 min TTL
        InvocationTimeoutSeconds=600,  # 10 min timeout
        InferenceId=request_id,
    )
    
    output_location = response['OutputLocation']
    print(f"Request submitted. Output will be at: {output_location}")
    print(f"Inference ID: {request_id}")
    
    return output_location, request_id


def poll_for_result(output_location: str, timeout_seconds: int = 600) -> dict:
    """Poll S3 for async inference result."""
    s3 = boto3.client('s3')
    bucket = output_location.split('/')[2]
    key = '/'.join(output_location.split('/')[3:])
    
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            result = json.loads(response['Body'].read())
            return result
        except s3.exceptions.NoSuchKey:
            time.sleep(5)
    
    raise TimeoutError(f"Async inference did not complete within {timeout_seconds}s")
```

---

## 9.4 Batch Transform — Nightly Scoring

```python
# code/deployment/batch_transform.py
"""
Batch Transform for nightly customer scoring.
Scores 50 million customer records overnight.
"""

import boto3
import sagemaker
from sagemaker.transformer import Transformer
import json

session = sagemaker.Session()
sm = boto3.client('sagemaker')

def run_nightly_credit_scoring(
    model_name: str,
    input_s3: str,           # s3://fintech-datalake-prod/batch-input/2024-06-15/
    output_s3: str,          # s3://fintech-datalake-prod/batch-output/2024-06-15/
    instance_type: str = 'ml.m5.4xlarge',
    instance_count: int = 20,   # 20 instances for 50M records overnight
):
    """
    Run nightly batch scoring for all customers.
    Completes 50M records in ~4 hours with 20 instances.
    """
    
    transformer = Transformer(
        model_name=model_name,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=output_s3,
        base_transform_job_name='nightly-credit-scoring',
        sagemaker_session=session,
        
        # Processing strategy
        strategy='MultiRecord',    # batch multiple records per request
        max_payload=6,             # MB per batch request
        max_concurrent_transforms=instance_count * 10,
        
        assemble_with='Line',      # output one result per line
        accept='text/csv',
        
        # VPC
        volume_kms_key='arn:aws:kms:us-east-1:123456789:key/mrk-batch-prod',
    )
    
    transformer.transform(
        data=input_s3,
        data_type='S3Prefix',
        content_type='text/csv',
        split_type='Line',         # each line is one record
        
        input_filter='$[1:]',      # skip first column (customer_id, not a feature)
        output_filter='$[0,-1]',   # output: customer_id + score
        join_source='Input',       # join input to output (for customer_id)
        
        wait=False,                # non-blocking for overnight jobs
        logs=False,
    )
    
    job_name = transformer.latest_transform_job.name
    print(f"Batch transform started: {job_name}")
    print(f"Input: {input_s3}")
    print(f"Output: {output_s3}")
    print(f"Instances: {instance_count}x {instance_type}")
    
    return job_name


def estimate_batch_cost(
    instance_type: str,
    instance_count: int,
    duration_hours: float,
) -> float:
    """Estimate batch transform cost."""
    PRICING = {
        'ml.m5.xlarge': 0.23,
        'ml.m5.4xlarge': 0.922,
        'ml.m5.12xlarge': 2.765,
        'ml.c5.4xlarge': 0.748,
    }
    hourly = PRICING.get(instance_type, 1.0)
    total = hourly * instance_count * duration_hours
    
    print(f"\nBatch Transform Cost Estimate:")
    print(f"  Instance: {instance_type} @ ${hourly}/hr")
    print(f"  Count: {instance_count} instances")
    print(f"  Duration: {duration_hours:.1f} hours")
    print(f"  Total: ${total:.2f}")
    
    return total
```

---

## 9.5 Multi-Model Endpoints (MME) — Cost Optimization

```python
# code/deployment/multi_model_endpoint.py
"""
Multi-Model Endpoint: host 100+ models on a single fleet.
Used for regional risk models, customer segment models, etc.
"""

import boto3
import sagemaker
from sagemaker.multidatamodel import MultiDataModel
import json

session = sagemaker.Session()
sm_runtime = boto3.client('sagemaker-runtime')

# MME: one endpoint, many models
# Models are loaded/unloaded dynamically (LRU cache in container)
def deploy_multi_model_endpoint(
    endpoint_name: str = 'risk-models-mme',
    model_data_prefix: str = 's3://fintech-models-prod/mme-models/',
    instance_type: str = 'ml.c5.4xlarge',
    instance_count: int = 3,
):
    """
    Deploy Multi-Model Endpoint for 100+ regional risk models.
    Each region/segment has its own model, but they share compute.
    """
    
    image_uri = sagemaker.image_uris.retrieve(
        framework='xgboost',
        region='us-east-1',
        version='1.7-1',
    )
    
    mme = MultiDataModel(
        name='risk-models-mme',
        model_data_prefix=model_data_prefix,
        image_uri=image_uri,
        role='arn:aws:iam::123456789:role/SageMakerExecutionRole-Inference',
        sagemaker_session=session,
    )
    
    predictor = mme.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    
    print(f"MME deployed: {endpoint_name}")
    print(f"Model store: {model_data_prefix}")
    print(f"Instances: {instance_count}x {instance_type}")
    
    return predictor, mme


def invoke_specific_model(
    endpoint_name: str,
    model_name: str,     # e.g., 'risk-model-northeast.tar.gz'
    payload: str,
) -> dict:
    """
    Invoke a specific model on the MME.
    SageMaker loads the model on first request, caches it afterward.
    """
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        TargetModel=model_name,   # which model to invoke
        ContentType='text/csv',
        Accept='application/json',
        Body=payload,
    )
    
    return json.loads(response['Body'].read())


# Cost comparison
print("""
MME vs. Individual Endpoints Cost Comparison:

Scenario: 100 regional risk models, each serving 10 TPS average

Option A: 100 individual endpoints
  100 endpoints × 1 instance (ml.c5.xlarge) × $0.23/hr
  = $23/hr = $552/day = $16,560/month

Option B: Multi-Model Endpoint  
  1 MME × 5 instances (ml.c5.4xlarge) × $0.922/hr
  = $4.61/hr = $110/day = $3,312/month
  
SAVING: $13,248/month (80% reduction)
""")
```

---

## 9.6 Serverless Inference — Dev/Test Models

```python
# code/deployment/serverless_endpoint.py
"""
Serverless Inference: zero instances when idle.
Perfect for dev/test models or low-traffic use cases.
NOT suitable for production latency-sensitive models.
"""

import boto3
import sagemaker
from sagemaker.serverless import ServerlessInferenceConfig

session = sagemaker.Session()

def deploy_serverless(
    model_package_arn: str,
    endpoint_name: str,
    memory_size_mb: int = 4096,
    max_concurrency: int = 10,
):
    """
    Deploy model to serverless inference.
    Cold start: 1-3 seconds (not suitable for real-time fraud)
    """
    
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=memory_size_mb,    # 1024, 2048, 3072, 4096, 5120, 6144
        max_concurrency=max_concurrency,      # max simultaneous requests
    )
    
    from sagemaker.model import Model
    model = Model(
        model_data=model_package_arn,
        role='arn:aws:iam::123456789:role/SageMakerExecutionRole-Inference',
        sagemaker_session=session,
    )
    
    predictor = model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=endpoint_name,
    )
    
    print(f"Serverless endpoint deployed: {endpoint_name}")
    print(f"Memory: {memory_size_mb}MB, Max concurrency: {max_concurrency}")
    
    return predictor

# Cost comparison
print("""
Serverless vs. Real-Time Endpoint:

Dev/Test model (5000 requests/day × 50ms per request):

Real-Time (ml.t3.medium, 24/7):
  $0.056/hr × 24 = $1.34/day

Serverless (4096MB):
  5000 requests × 50ms × $0.0000001/ms/GB × 4GB
  = $0.10/day

Serverless wins for < 100K requests/day.
Above that, real-time endpoint is cheaper.
""")
```

---

*Next: [Section 10 — Advanced Deployments →](10-advanced-deployments.md)*
