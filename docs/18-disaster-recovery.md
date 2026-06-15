# Section 18 — Disaster Recovery

> **Production Context**: An entire AWS region goes down (e.g., us-east-1). Does the ML platform survive? At enterprise scale, multi-region Active-Active or Active-Passive architectures are mandatory.

---

## 18.1 Multi-Region ML Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-REGION DISASTER RECOVERY                           │
│                                                                             │
│               Route53 (Health Checks & Routing)                             │
│                      ↙            ↘                                         │
│          US-EAST-1 (Active)      US-WEST-2 (Passive/Active)                 │
│          ──────────────────      ──────────────────────────                 │
│  Data    S3 Datalake        ──▶  S3 Cross-Region Replication (CRR)          │
│          Feature Store (DDB)──▶  DynamoDB Global Tables                     │
│                                                                             │
│  Models  Model Registry     ──▶  EventBridge → Lambda → Copy Model to West  │
│                                                                             │
│  Serve   SageMaker Endpoint      SageMaker Endpoint (Scaled down or Auto)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 18.2 Cross-Region Model Replication

```python
# code/dr/cross_region_model_copy.py
"""
Lambda function to copy approved models to the DR region.
Ensures that if us-east-1 goes down, us-west-2 has the exact same model artifacts ready to deploy.
"""

import boto3
import json
import logging

logger = logging.getLogger(__name__)

def lambda_handler(event, context):
    """
    Triggered by EventBridge when a Model is Approved in the primary region.
    Copies the S3 artifact and registers the model in the secondary region.
    """
    primary_region = 'us-east-1'
    dr_region = 'us-west-2'
    
    sm_primary = boto3.client('sagemaker', region_name=primary_region)
    sm_dr = boto3.client('sagemaker', region_name=dr_region)
    s3 = boto3.client('s3')
    
    model_package_arn = event['detail']['ModelPackageArn']
    
    try:
        # 1. Get Model Details
        model_info = sm_primary.describe_model_package(ModelPackageName=model_package_arn)
        group_name = model_info['ModelPackageGroupName']
        
        # S3 URI of the model artifact (e.g., s3://primary-bucket/models/model.tar.gz)
        primary_s3_uri = model_info['InferenceSpecification']['Containers'][0]['ModelDataUrl']
        
        # 2. Copy S3 Artifact to DR Region
        # (Assuming cross-region replication is too slow or we want explicit control)
        source_bucket = primary_s3_uri.split('/')[2]
        source_key = '/'.join(primary_s3_uri.split('/')[3:])
        dr_bucket = f"fintech-models-dr-{dr_region}"
        dr_key = source_key
        
        logger.info(f"Copying artifact from {source_bucket} to {dr_bucket}")
        s3.copy_object(
            CopySource={'Bucket': source_bucket, 'Key': source_key},
            Bucket=dr_bucket,
            Key=dr_key
        )
        dr_s3_uri = f"s3://{dr_bucket}/{dr_key}"
        
        # 3. Create Model Package Group in DR Region (if it doesn't exist)
        try:
            sm_dr.describe_model_package_group(ModelPackageGroupName=group_name)
        except sm_dr.exceptions.ClientError:
            sm_dr.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription=f"DR Replica for {group_name}"
            )
            
        # 4. Register Model in DR Region
        inference_spec = model_info['InferenceSpecification']
        inference_spec['Containers'][0]['ModelDataUrl'] = dr_s3_uri
        
        dr_response = sm_dr.create_model_package(
            ModelPackageGroupName=group_name,
            InferenceSpecification=inference_spec,
            ModelApprovalStatus='Approved',
            CustomerMetadataProperties=model_info.get('CustomerMetadataProperties', {})
        )
        
        logger.info(f"✅ Successfully replicated model to DR region: {dr_response['ModelPackageArn']}")
        
    except Exception as e:
        logger.error(f"Failed to replicate model: {e}")
        raise e
```

---

*Next: [Section 19 — Cost Optimization →](19-cost-optimization.md)*
