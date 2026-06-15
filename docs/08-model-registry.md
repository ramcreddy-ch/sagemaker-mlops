# Section 08 — Model Registry & Approval Workflows

---

## 8.1 Model Registry Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SAGEMAKER MODEL REGISTRY                                 │
│                                                                             │
│  Model Package Groups (one per use case / model family):                   │
│  ├── fraud-detection-xgboost                                                │
│  │   ├── v47 [Approved] ← current champion in production                   │
│  │   ├── v48 [PendingManualApproval] ← challenger awaiting review          │
│  │   ├── v46 [Approved] ← previous champion (kept for rollback)            │
│  │   └── v45 [Rejected]                                                    │
│  ├── credit-risk-lightgbm                                                  │
│  │   └── v12 [Approved]                                                    │
│  └── llm-financial-assistant                                                │
│      └── llama3-70b-v3 [Approved]                                          │
│                                                                             │
│  APPROVAL WORKFLOW:                                                         │
│  Training Job → Evaluate → Submit to Registry [Pending] →                  │
│  Model Risk Review → MLOps Approval → [Approved] → CD Pipeline             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.2 Registering Models — Production Code

```python
# code/model-registry/register_model.py
"""
Production model registration with full metadata.
Registers a trained model into the SageMaker Model Registry.
"""

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.sklearn.model import SKLearnModel
import json
from datetime import datetime

session = sagemaker.Session()
sm = boto3.client('sagemaker', region_name='us-east-1')
role = 'arn:aws:iam::123456789:role/SageMakerExecutionRole-Pipelines'
region = 'us-east-1'
account_id = '123456789012'

def register_fraud_model(
    model_artifact_s3: str,
    evaluation_s3: str,
    training_job_name: str,
    data_version: str,
    approval_status: str = "PendingManualApproval",
) -> str:
    """
    Register a trained fraud detection model with complete metadata.
    Returns the model package ARN.
    """
    
    # Load evaluation metrics
    s3 = boto3.client('s3')
    bucket, key = evaluation_s3.replace('s3://', '').split('/', 1)
    eval_data = json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
    
    auc_roc = eval_data['classification_metrics']['auc_roc']['value']
    avg_precision = eval_data['classification_metrics']['avg_precision']['value']
    f1 = eval_data['classification_metrics']['f1_score']['value']
    fnr = eval_data['business_metrics']['false_negative_rate']['value']
    threshold = eval_data['business_metrics']['optimal_threshold']['value']
    
    # Model image URI
    image_uri = sagemaker.image_uris.retrieve(
        framework='xgboost',
        region=region,
        version='1.7-1',
    )
    
    # Register the model package
    response = sm.create_model_package(
        ModelPackageGroupName='fraud-detection-xgboost',
        
        ModelPackageDescription=(
            f"Fraud detection XGBoost model trained on {data_version} data. "
            f"AUC-ROC: {auc_roc:.4f}, Avg Precision: {avg_precision:.4f}"
        ),
        
        InferenceSpecification={
            'Containers': [{
                'Image': image_uri,
                'ModelDataUrl': model_artifact_s3,
                'Framework': 'XGBOOST',
                'FrameworkVersion': '1.7',
            }],
            'SupportedContentTypes': ['text/csv', 'application/json'],
            'SupportedResponseMIMETypes': ['application/json'],
            'SupportedTransformInstanceTypes': ['ml.m5.xlarge', 'ml.m5.4xlarge'],
            'SupportedRealtimeInferenceInstanceTypes': [
                'ml.c5.large', 'ml.c5.xlarge', 'ml.c5.2xlarge', 'ml.c5.4xlarge'
            ],
        },
        
        ModelApprovalStatus=approval_status,
        
        ModelMetrics={
            'ModelQuality': {
                'Statistics': {
                    'ContentType': 'application/json',
                    'S3Uri': evaluation_s3,
                },
            },
        },
        
        MetadataProperties={
            'GeneratedBy': f"arn:aws:sagemaker:{region}:{account_id}:training-job/{training_job_name}",
            'ProjectId': 'fintech-fraud-detection',
            'CommitId': 'abc123def456',           # git commit SHA
            'Repository': 'github.com/fintech/ml-platform',
        },
        
        CustomerMetadataProperties={
            'data_version': data_version,
            'training_date': datetime.utcnow().date().isoformat(),
            'training_job_name': training_job_name,
            'auc_roc': str(round(auc_roc, 6)),
            'avg_precision': str(round(avg_precision, 6)),
            'f1_score': str(round(f1, 6)),
            'false_negative_rate': str(round(fnr, 6)),
            'optimal_threshold': str(round(threshold, 4)),
            'model_type': 'XGBoostClassifier',
            'framework_version': '1.7-1',
            'use_case': 'fraud_detection',
            'champion_status': 'CHALLENGER',     # becomes CHAMPION after approval
            'risk_tier': 'MEDIUM',
            'approved_by': 'pending',
        },
        
        Tags=[
            {'Key': 'environment', 'Value': 'production'},
            {'Key': 'use-case', 'Value': 'fraud-detection'},
            {'Key': 'team', 'Value': 'fraud-ml'},
            {'Key': 'data-classification', 'Value': 'confidential'},
            {'Key': 'pci-scope', 'Value': 'true'},
        ],
    )
    
    model_package_arn = response['ModelPackageArn']
    print(f"✅ Model registered: {model_package_arn}")
    print(f"   Status: {approval_status}")
    print(f"   AUC-ROC: {auc_roc:.6f}")
    
    # Notify model risk team via SNS
    sns = boto3.client('sns')
    sns.publish(
        TopicArn='arn:aws:sns:us-east-1:123456789:model-review-requests',
        Subject=f"Model Review Required: fraud-detection-xgboost",
        Message=json.dumps({
            'model_package_arn': model_package_arn,
            'auc_roc': auc_roc,
            'avg_precision': avg_precision,
            'training_job': training_job_name,
            'review_link': f"https://studio.sagemaker.aws/...",
            'requested_by': 'mlops-pipeline',
            'data_version': data_version,
        }, indent=2)
    )
    
    return model_package_arn
```

---

## 8.3 Approval Workflow Automation

```python
# code/model-registry/approval_workflow.py
"""
Automated model approval workflow with quality gates.
In production, this is the gatekeeper between training and deployment.
"""

import boto3
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)
sm = boto3.client('sagemaker')
sns = boto3.client('sns')


class ModelApprovalWorkflow:
    """
    Implements the model approval gate with automated quality checks
    and human review integration.
    """
    
    # Quality thresholds — change requires JIRA ticket + approval
    QUALITY_GATES = {
        'fraud-detection-xgboost': {
            'auc_roc': 0.94,           # minimum AUC-ROC
            'avg_precision': 0.70,     # minimum average precision
            'false_negative_rate': 0.08,   # maximum FNR (miss rate)
            'max_regression_pct': -0.01,   # max allowed regression vs champion
        },
        'credit-risk-lightgbm': {
            'auc_roc': 0.82,
            'avg_precision': 0.60,
            'false_negative_rate': 0.12,
            'max_regression_pct': -0.005,
        },
    }
    
    def automated_quality_gate(self, model_package_arn: str) -> dict:
        """
        Run automated quality checks before human review.
        Returns: {passed: bool, failures: list, champion_comparison: dict}
        """
        # Get model metadata
        model_info = sm.describe_model_package(ModelPackageName=model_package_arn)
        metadata = model_info.get('CustomerMetadataProperties', {})
        group_name = model_info['ModelPackageGroupName']
        
        gates = self.QUALITY_GATES.get(group_name, {})
        failures = []
        
        # Check absolute thresholds
        auc_roc = float(metadata.get('auc_roc', 0))
        if auc_roc < gates.get('auc_roc', 0):
            failures.append(f"AUC-ROC {auc_roc:.4f} < threshold {gates['auc_roc']}")
        
        avg_precision = float(metadata.get('avg_precision', 0))
        if avg_precision < gates.get('avg_precision', 0):
            failures.append(f"Avg Precision {avg_precision:.4f} < threshold {gates['avg_precision']}")
        
        fnr = float(metadata.get('false_negative_rate', 1.0))
        if fnr > gates.get('false_negative_rate', 1.0):
            failures.append(f"False Negative Rate {fnr:.4f} > threshold {gates['false_negative_rate']}")
        
        # Compare vs champion (regression check)
        champion_metrics = self._get_champion_metrics(group_name)
        champion_comparison = {}
        
        if champion_metrics:
            regression = auc_roc - champion_metrics['auc_roc']
            max_regression = gates.get('max_regression_pct', -0.01)
            
            if regression < max_regression:
                failures.append(
                    f"AUC-ROC regression: {regression:.4f} < allowed {max_regression}"
                )
            
            champion_comparison = {
                'champion_auc_roc': champion_metrics['auc_roc'],
                'challenger_auc_roc': auc_roc,
                'delta': regression,
                'status': 'IMPROVEMENT' if regression > 0 else 'REGRESSION',
            }
        
        result = {
            'passed': len(failures) == 0,
            'failures': failures,
            'champion_comparison': champion_comparison,
            'model_package_arn': model_package_arn,
            'metrics': {
                'auc_roc': auc_roc,
                'avg_precision': avg_precision,
                'false_negative_rate': fnr,
            }
        }
        
        logger.info(f"Quality gate result: {'PASSED' if result['passed'] else 'FAILED'}")
        if failures:
            for f in failures:
                logger.warning(f"  ❌ {f}")
        
        return result
    
    def approve_model(
        self,
        model_package_arn: str,
        approved_by: str,
        approval_comment: str,
    ) -> bool:
        """Approve a model for deployment."""
        
        # Final quality check
        gate_result = self.automated_quality_gate(model_package_arn)
        if not gate_result['passed']:
            logger.error(f"Cannot approve — quality gates failed: {gate_result['failures']}")
            return False
        
        # Update approval status
        sm.update_model_package(
            ModelPackageName=model_package_arn,
            ModelApprovalStatus='Approved',
            CustomerMetadataProperties={
                'approved_by': approved_by,
                'approval_comment': approval_comment,
                'approval_timestamp': str(pd.Timestamp.now()),
                'champion_status': 'CHAMPION',
            }
        )
        
        logger.info(f"✅ Model approved by {approved_by}: {model_package_arn}")
        
        # Trigger deployment pipeline via EventBridge
        events = boto3.client('events')
        events.put_events(
            Entries=[{
                'Source': 'fintech.model-registry',
                'DetailType': 'ModelApproved',
                'Detail': json.dumps({
                    'model_package_arn': model_package_arn,
                    'approved_by': approved_by,
                    'group_name': model_package_arn.split('/')[-2],
                }),
                'EventBusName': 'fintech-mlops-events',
            }]
        )
        
        return True
    
    def reject_model(self, model_package_arn: str, rejected_by: str, reason: str):
        """Reject a model with reason documented."""
        sm.update_model_package(
            ModelPackageName=model_package_arn,
            ModelApprovalStatus='Rejected',
            CustomerMetadataProperties={
                'rejected_by': rejected_by,
                'rejection_reason': reason,
                'rejection_timestamp': str(pd.Timestamp.now()),
            }
        )
        logger.info(f"Model rejected by {rejected_by}: {reason}")
    
    def rollback_to_version(self, group_name: str, version: int):
        """
        Emergency rollback to a specific model version.
        Updates the production endpoint to use an older approved model.
        """
        # Get the specific model package version
        response = sm.list_model_packages(
            ModelPackageGroupName=group_name,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
        )
        
        packages = response['ModelPackageSummaryList']
        target = next(
            (p for p in packages if p['ModelPackageVersion'] == version), None
        )
        
        if not target:
            raise ValueError(f"Version {version} not found or not approved")
        
        logger.info(f"🔄 Rolling back to {group_name} version {version}")
        
        # Trigger rollback deployment
        events = boto3.client('events')
        events.put_events(
            Entries=[{
                'Source': 'fintech.model-registry',
                'DetailType': 'ModelRollback',
                'Detail': json.dumps({
                    'group_name': group_name,
                    'target_version': version,
                    'model_package_arn': target['ModelPackageArn'],
                    'rollback_reason': 'Emergency rollback triggered by MLOps',
                }),
                'EventBusName': 'fintech-mlops-events',
            }]
        )
    
    def _get_champion_metrics(self, group_name: str) -> Optional[dict]:
        """Get current champion model metrics."""
        response = sm.list_model_packages(
            ModelPackageGroupName=group_name,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1,
        )
        
        if not response['ModelPackageSummaryList']:
            return None
        
        champion = sm.describe_model_package(
            ModelPackageName=response['ModelPackageSummaryList'][0]['ModelPackageArn']
        )
        
        meta = champion.get('CustomerMetadataProperties', {})
        return {
            'auc_roc': float(meta.get('auc_roc', 0)),
            'avg_precision': float(meta.get('avg_precision', 0)),
        }
```

---

## 8.4 Multi-Account Model Promotion

```python
# Cross-account model promotion: dev → staging → production

def promote_model_cross_account(
    model_package_arn: str,   # source (dev account)
    target_account: str,
    target_region: str = 'us-east-1',
):
    """
    Share model package from dev account to prod account.
    Used in enterprise multi-account setups.
    """
    sm = boto3.client('sagemaker')
    
    # Grant cross-account access to model package group
    policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AllowProdAccountAccess",
            "Effect": "Allow",
            "Principal": {
                "AWS": f"arn:aws:iam::{target_account}:root"
            },
            "Action": [
                "sagemaker:DescribeModelPackage",
                "sagemaker:ListModelPackages",
                "sagemaker:UpdateModelPackage",
                "sagemaker:CreateModel",
            ],
            "Resource": model_package_arn,
        }]
    }
    
    group_name = model_package_arn.split('/')[-2]
    
    sm.put_model_package_group_policy(
        ModelPackageGroupName=group_name,
        ResourcePolicy=json.dumps(policy),
    )
    
    print(f"Cross-account access granted to {target_account}")
    print(f"In prod account, reference: {model_package_arn}")
```

---

*Next: [Section 09 — Deployment Strategies →](09-deployment-strategies.md)*
