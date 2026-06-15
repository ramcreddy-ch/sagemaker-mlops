# Section 13 — CI/CD for MLOps

> **Production Context**: At Netflix and Uber, data scientists don't deploy models from Jupyter notebooks. All models reach production through automated CI/CD pipelines.

---

## 13.1 MLOps Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE MLOPS CI/CD WORKFLOW                          │
│                                                                             │
│  1. CODE REPOSITORY (GitHub)                                                │
│     ├── /model-code        (Training scripts, feature eng)                  │
│     ├── /pipeline-code     (SageMaker Pipeline definition)                  │
│     └── /infrastructure    (Terraform, IaC)                                 │
│           ↓                                                                 │
│  2. CONTINUOUS INTEGRATION (GitHub Actions / Jenkins)                       │
│     ├── Linting & Unit Tests (pytest)                                       │
│     ├── Build custom training Docker container                              │
│     └── Push container to Amazon ECR                                        │
│           ↓                                                                 │
│  3. CONTINUOUS TRAINING (CT)                                                │
│     ├── Trigger SageMaker Pipeline (dev account)                            │
│     ├── Train → Evaluate → Conditional Registration                         │
│     └── Register to Model Registry (Pending Approval)                       │
│           ↓                                                                 │
│  4. HUMAN IN THE LOOP                                                       │
│     └── Model Risk Team reviews Model Registry metadata → APPROVES          │
│           ↓                                                                 │
│  5. CONTINUOUS DEPLOYMENT (CD) (AWS CodePipeline)                           │
│     ├── EventBridge detects "Model Approved" event                          │
│     ├── Deploy to Staging (integration tests)                               │
│     └── Deploy to Production (Blue/Green or Canary)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13.2 Continuous Integration (GitHub Actions)

```yaml
# .github/workflows/ml-ci-pipeline.yml
name: MLOps CI Pipeline

on:
  push:
    branches: [ main ]
    paths:
      - 'code/**'
      - 'docker/**'
      - '.github/workflows/**'
  pull_request:
    branches: [ main ]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: fintech-mlops-containers
  PIPELINE_NAME: fraud-detection-pipeline

jobs:
  test-and-lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest sagemaker pandas scikit-learn
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        
    - name: Lint with flake8
      run: |
        flake8 code/ --count --select=E9,F63,F7,F82 --show-source --statistics
        
    - name: Run unit tests
      run: |
        pytest tests/unit/

  build-and-push:
    needs: test-and-lint
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        role-to-assume: arn:aws:iam::123456789:role/GitHubActions-MLOps
        aws-region: ${{ env.AWS_REGION }}
        
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
      
    - name: Build, tag, and push image to ECR
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        cd docker/custom-training
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

  update-sagemaker-pipeline:
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        role-to-assume: arn:aws:iam::123456789:role/GitHubActions-MLOps
        aws-region: ${{ env.AWS_REGION }}
        
    - name: Install SageMaker SDK
      run: pip install sagemaker boto3
        
    - name: Update and Execute Pipeline
      env:
        IMAGE_TAG: ${{ github.sha }}
      run: |
        # Script parses pipeline definition, upserts it, and starts execution
        python code/pipelines/run_pipeline.py \
          --pipeline-name ${{ env.PIPELINE_NAME }} \
          --image-tag $IMAGE_TAG \
          --execute
```

---

## 13.3 Continuous Deployment (Event-Driven)

```python
# code/cicd/event_driven_deployment.py
"""
EventBridge + Lambda setup for Continuous Deployment.
Triggers when a model is Approved in the Model Registry.
"""

import boto3
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# This runs as an AWS Lambda function triggered by EventBridge
def lambda_handler(event, context):
    """
    EventBridge triggers this when ModelApprovalStatus changes to 'Approved'.
    Deploys the approved model to production (Blue/Green).
    """
    sm = boto3.client('sagemaker')
    
    # 1. Parse EventBridge event
    detail = event['detail']
    model_package_arn = detail['ModelPackageArn']
    group_name = model_package_arn.split('/')[-2]
    
    logger.info(f"CD Pipeline triggered for approved model: {model_package_arn}")
    
    # Check if this is the fraud detection model
    if group_name != 'fraud-detection-xgboost':
        logger.info("Not a fraud model, ignoring.")
        return
        
    endpoint_name = 'fraud-detection-prod'
    timestamp = int(time.time())
    
    try:
        # 2. Create Model
        model_name = f"fraud-model-{timestamp}"
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={'ModelPackageName': model_package_arn},
            ExecutionRoleArn=os.environ['INFERENCE_ROLE_ARN'],
            VpcConfig={
                'SecurityGroupIds': [os.environ['SECURITY_GROUP_ID']],
                'Subnets': json.loads(os.environ['SUBNET_IDS']),
            }
        )
        
        # 3. Create Endpoint Config (for Blue/Green deployment)
        config_name = f"fraud-config-{timestamp}"
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InstanceType': 'ml.c5.2xlarge',
                'InitialInstanceCount': 3,
            }],
            # Enable Data Capture for Model Monitor
            DataCaptureConfig={
                'EnableCapture': True,
                'InitialSamplingPercentage': 20,
                'DestinationS3Uri': f"s3://fintech-datalake-prod/monitoring/{endpoint_name}/",
                'CaptureOptions': [{'CaptureMode': 'Input'}, {'CaptureMode': 'Output'}],
            }
        )
        
        # 4. Trigger Deployment (Update Endpoint)
        logger.info(f"Updating endpoint {endpoint_name} with config {config_name}")
        
        # SageMaker automatically performs a blue/green deployment natively
        # when update_endpoint is called. It spins up the new fleet, ensures
        # it's healthy, switches traffic, and terminates the old fleet.
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        # 5. Notify Success
        sns = boto3.client('sns')
        sns.publish(
            TopicArn=os.environ['NOTIFICATION_TOPIC_ARN'],
            Subject="🚀 CD Pipeline: Deployment Started",
            Message=f"Deploying approved model {model_package_arn} to {endpoint_name}."
        )
        
        return {
            'statusCode': 200,
            'body': f"Deployment initiated for {endpoint_name}"
        }
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        # Notify Failure
        sns = boto3.client('sns')
        sns.publish(
            TopicArn=os.environ['NOTIFICATION_TOPIC_ARN'],
            Subject="❌ CD Pipeline: Deployment Failed",
            Message=f"Failed to deploy {model_package_arn}.\nError: {str(e)}"
        )
        raise e

# ---------------------------------------------------------
# EventBridge Rule Definition (IaC/Terraform equivalent)
# ---------------------------------------------------------
"""
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Package State Change"],
  "detail": {
    "ModelApprovalStatus": ["Approved"]
  }
}
"""
```

---

## 13.4 Automated Retraining (Data Drift Trigger)

```python
# code/cicd/automated_retraining.py
"""
Triggers retraining when SageMaker Model Monitor detects data drift.
"""
import boto3
import json
import logging

logger = logging.getLogger(__name__)

def trigger_retraining_on_drift(event, context):
    """
    EventBridge triggers this when CloudWatch Alarm (Model Monitor Drift) fires.
    """
    sm = boto3.client('sagemaker')
    
    logger.warning("🚨 Drift detected! Triggering automated retraining pipeline.")
    
    # Start the SageMaker Pipeline
    # The pipeline will pull the latest data from the Feature Store,
    # retrain, evaluate, and register the new model automatically.
    response = sm.start_pipeline_execution(
        PipelineName='fraud-detection-pipeline',
        PipelineExecutionDescription='Automated retraining triggered by data drift',
        PipelineParameters=[
            {'Name': 'TrainingInstanceType', 'Value': 'ml.m5.4xlarge'},
            # Optionally pass the drift report location
            {'Name': 'TriggerReason', 'Value': 'DataDriftDetected'},
        ]
    )
    
    logger.info(f"Pipeline started: {response['PipelineExecutionArn']}")
    return response['PipelineExecutionArn']
```

---

*Next: [Section 14 — MLOps Automation →](14-mlops-automation.md)*
