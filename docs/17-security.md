# Section 17 — Security

> **Production Context**: At banks like Capital One and JPMorgan, security is not an afterthought — it's the primary reason they chose SageMaker over raw EC2/EKS.

---

## 17.1 SageMaker Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE SECURITY ARCHITECTURE                         │
│                                                                             │
│  1. NETWORK ISOLATION (VPC)                                                 │
│     ├── No internet access for Training/Inference instances                 │
│     ├── Traffic flows exclusively through VPC Endpoints (PrivateLink)       │
│     └── Security Groups restrict traffic (e.g., Studio ↔ Training only)     │
│                                                                             │
│  2. DATA ENCRYPTION (KMS)                                                   │
│     ├── At Rest: EBS volumes, S3 buckets, Feature Store all use CMKs        │
│     └── In Transit: TLS 1.2+ mandatory, Inter-container encryption          │
│                                                                             │
│  3. IDENTITY & ACCESS (IAM)                                                 │
│     ├── Least privilege execution roles for every job/endpoint              │
│     ├── SageMaker Studio uses IAM Identity Center (SSO)                     │
│     └── Resource-based policies (Bucket policies restrict access)           │
│                                                                             │
│  4. GOVERNANCE & COMPLIANCE                                                 │
│     ├── CloudTrail logging for all Control Plane APIs                       │
│     ├── Data Capture for Model Monitor (auditability)                       │
│     └── Model Registry for complete lineage                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 17.2 VPC Isolation (Network Security)

```python
# code/security/vpc_config.py
"""
Example of configuring a SageMaker Training Job for strict VPC isolation.
This prevents the container from calling out to the public internet
(e.g., downloading a malicious package or exfiltrating data).
"""

from sagemaker.estimator import Estimator

def create_secure_estimator(role, image_uri, instance_type):
    return Estimator(
        role=role,
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=1,
        
        # 1. VPC Configuration (Mandatory in Enterprise)
        # Prevents internet access. All AWS API calls must go through VPC Endpoints.
        subnets=['subnet-private-1a', 'subnet-private-1b'],
        security_group_ids=['sg-sagemaker-training-isolated'],
        
        # 2. Network Isolation (Double down on security)
        # Even within the VPC, the container cannot make outbound network calls.
        # It can ONLY read/write to the S3 buckets specified in the input/output channels.
        enable_network_isolation=True,
        
        # 3. Inter-container Encryption
        # If using distributed training (multi-node), encrypt traffic between nodes.
        # Slight performance hit (~5%), but required for compliance.
        encrypt_inter_container_traffic=True,
        
        # 4. EBS Volume Encryption
        volume_kms_key='arn:aws:kms:us-east-1:123456789:key/mrk-ebs-training',
        
        # ... other config
    )
```

---

## 17.3 Least Privilege IAM Execution Roles

```json
// Example of a highly restricted SageMaker Training Execution Role
// This role can ONLY read from the input bucket and write to the output bucket.
// It cannot delete S3 objects, create new buckets, or access other services.

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowSagemakerAssume",
            "Effect": "Allow",
            "Action": "sts:AssumeRole",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            }
        },
        {
            "Sid": "AllowS3ReadInput",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::fintech-datalake-prod",
                "arn:aws:s3:::fintech-datalake-prod/training-data/*"
            ]
        },
        {
            "Sid": "AllowS3WriteOutput",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::fintech-datalake-prod/training-output/*"
            ]
        },
        {
            "Sid": "AllowKMSDecrypt",
            "Effect": "Allow",
            "Action": [
                "kms:Decrypt",
                "kms:GenerateDataKey"
            ],
            "Resource": "arn:aws:kms:us-east-1:123456789:key/mrk-datalake-prod"
        },
        {
            "Sid": "AllowCloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:us-east-1:123456789:log-group:/aws/sagemaker/*"
        },
        {
            "Sid": "AllowECRPull",
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*" 
        }
    ]
}
```

---

*Next: [Section 18 — Disaster Recovery →](18-disaster-recovery.md)*
