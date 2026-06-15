# Section 03 — SageMaker Architecture Deep Dive

> **Production Architecture used at Capital One, JPMorgan, Goldman Sachs**
> Focus: Enterprise Financial Intelligence Platform — Fraud Detection + Risk + LLM

---

## 3.1 Complete Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE FINANCIAL INTELLIGENCE PLATFORM                       │
│                         AWS SageMaker Production Architecture                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  USERS & CONSUMERS                                                               │
│                                                                                  │
│  Data Scientists    ML Engineers    MLOps Engineers    Business Analysts         │
│       │                  │                │                   │                  │
│  SageMaker Studio   Code Editor    CloudWatch/Grafana    QuickSight/Tableau      │
└────────────────────────────────┬─────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────────────┐
│  DATA SOURCES                                                                    │
│                                                                                  │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ Core Banking │  │ Kafka/MSK  │  │ Third-Party  │  │ Market Data        │     │
│  │ Systems      │  │ (events)   │  │ Credit APIs  │  │ (Bloomberg/Refinitiv│     │
│  │ (Oracle/DB2) │  │ 500K msg/s │  │ FICO Scores  │  │ feeds)             │     │
│  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘  └──────────┬─────────┘     │
└─────────│────────────────│──────────────────│───────────────────────────────────┘
          │                │                  │
┌─────────▼────────────────▼──────────────────▼──────────────────────────────────┐
│  DATA INGESTION LAYER                                                           │
│                                                                                  │
│  ┌─────────────┐   ┌────────────────┐   ┌──────────────────┐                   │
│  │ AWS DMS     │   │ Kinesis Data   │   │ AWS Glue          │                   │
│  │ (CDC from   │   │ Streams        │   │ (batch ETL)       │                   │
│  │ Oracle)     │   │ (real-time)    │   │                   │                   │
│  └──────┬──────┘   └───────┬────────┘   └────────┬──────────┘                   │
└─────────│──────────────────│───────────────────────────────────────────────────┘
          │                  │                      │
┌─────────▼──────────────────▼──────────────────────▼──────────────────────────┐
│  DATA LAKE (S3)                                                               │
│                                                                               │
│  s3://fintech-datalake-prod/                                                  │
│  ├── raw/          (immutable, encrypted, versioned)                          │
│  │   ├── transactions/year=2024/month=06/day=15/hour=12/                      │
│  │   ├── customer-profiles/                                                   │
│  │   └── market-data/                                                         │
│  ├── processed/    (cleaned, validated, schema-enforced)                      │
│  ├── features/     (computed features, partitioned)                           │
│  ├── training/     (train/val/test splits per model)                          │
│  ├── models/       (serialized model artifacts)                               │
│  └── inference/    (input/output for batch transform)                         │
└─────────────────────┬──────────────────────────────────────────────────────┘
                      │
     ┌────────────────┼────────────────────────┐
     │                │                        │
┌────▼────┐    ┌──────▼──────┐          ┌──────▼──────┐
│  Glue   │    │  Redshift   │          │  Athena     │
│  Catalog│    │  (analytics)│          │  (ad-hoc)   │
└────┬────┘    └──────┬──────┘          └─────────────┘
     │                │
┌────▼────────────────▼────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING                                                          │
│                                                                               │
│  ┌──────────────────┐     ┌──────────────────────────────────────────────┐   │
│  │ EMR Spark        │     │ SageMaker Processing Jobs                    │   │
│  │ (large-scale     │────▶│ (feature computation, data validation)       │   │
│  │ feature compute) │     └───────────────┬──────────────────────────────┘   │
│  └──────────────────┘                     │                                   │
└───────────────────────────────────────────┼───────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼───────────────────────────────────┐
│  SAGEMAKER FEATURE STORE                                                      │
│                                                                               │
│  ┌─────────────────────────────┐   ┌─────────────────────────────────────┐   │
│  │ OFFLINE STORE               │   │ ONLINE STORE                        │   │
│  │ (S3 + Glue Catalog)         │   │ (DynamoDB — sub-5ms reads)          │   │
│  │                             │   │                                     │   │
│  │ fraud_features_v3           │   │ fraud_features_v3                   │   │
│  │ customer_profile_v2         │   │ customer_profile_v2                 │   │
│  │ transaction_history_v4      │   │ transaction_history_v4              │   │
│  │                             │   │                                     │   │
│  │ Retention: 2 years          │   │ TTL: 30 days                        │   │
│  └─────────────────────────────┘   └─────────────────────────────────────┘   │
└───────────────────────────────────────────┬───────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼───────────────────────────────────┐
│  TRAINING PLATFORM                                                             │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  SageMaker Pipelines (Orchestration)                                     │  │
│  │                                                                          │  │
│  │  Preprocess → Train → Evaluate → Register → (Conditional) → Deploy      │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │ Training Jobs   │  │ HPO (Tuning)    │  │ Distributed Training        │   │
│  │                 │  │                 │  │                             │   │
│  │ XGBoost Fraud   │  │ Bayesian opt.   │  │ PyTorch DDP (4x A10G)      │   │
│  │ LightGBM Risk   │  │ 100 candidates  │  │ Deepspeed ZeRO-3 (LLM)    │   │
│  │ LSTM Forecast   │  │                 │  │                             │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘   │
└───────────────────────────────────────────┬───────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼───────────────────────────────────┐
│  SAGEMAKER MODEL REGISTRY                                                      │
│                                                                                │
│  Model Package Groups:                                                          │
│  ├── fraud-detection-xgboost        (champion: v47, challenger: v48)           │
│  ├── credit-risk-lightgbm           (champion: v12)                            │
│  ├── customer-churn-pytorch         (champion: v8)                             │
│  ├── market-forecast-lstm           (champion: v31)                            │
│  └── llm-financial-assistant        (champion: llama3-70b-finetuned-v3)        │
│                                                                                │
│  Approval states: PendingManualApproval → Approved → Rejected                  │
└───────────────────────────────────────────┬───────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼───────────────────────────────────┐
│  DEPLOYMENT LAYER                                                              │
│                                                                                │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐              │
│  │ Real-Time        │ │ Async Endpoint   │ │ Batch Transform   │              │
│  │ Endpoints        │ │                  │ │                   │              │
│  │                  │ │ LLM (70B)        │ │ Nightly Churn     │              │
│  │ fraud-detection  │ │ Document Intel.  │ │ Credit Scoring    │              │
│  │ credit-risk      │ │ Timeout: 15 min  │ │ 50M customers     │              │
│  │ P99 < 10ms       │ │                  │ │                   │              │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘              │
│                                                                                │
│  ┌──────────────────┐ ┌──────────────────┐                                    │
│  │ Multi-Model      │ │ Serverless        │                                    │
│  │ Endpoint         │ │ Inference         │                                    │
│  │                  │ │                   │                                    │
│  │ 50 risk models   │ │ dev/test models   │                                    │
│  │ per instance     │ │ low traffic       │                                    │
│  └──────────────────┘ └──────────────────┘                                    │
└───────────────────────────────────────────┬───────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼───────────────────────────────────┐
│  MONITORING & OBSERVABILITY                                                    │
│                                                                                │
│  ┌──────────────────┐ ┌───────────────────┐ ┌────────────────────────────┐   │
│  │ Model Monitor    │ │ CloudWatch        │ │ Grafana + Prometheus       │   │
│  │                  │ │                   │ │                            │   │
│  │ Data Quality     │ │ Latency P50/P99   │ │ Custom ML Dashboards       │   │
│  │ Model Quality    │ │ Error Rates       │ │ Drift Visualization        │   │
│  │ Feature Drift    │ │ GPU Utilization   │ │ Business KPIs              │   │
│  │ Bias Monitor     │ │ Cost Metrics      │ │                            │   │
│  └──────────────────┘ └───────────────────┘ └────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.2 VPC Architecture for Production SageMaker

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VPC: vpc-ml-prod (10.0.0.0/16)                                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Private Subnets (NO internet access)                               │   │
│  │                                                                     │   │
│  │  subnet-ml-a (10.0.1.0/24) — AZ us-east-1a — Training + Inference  │   │
│  │  subnet-ml-b (10.0.2.0/24) — AZ us-east-1b — Training + Inference  │   │
│  │  subnet-ml-c (10.0.3.0/24) — AZ us-east-1c — Inference only        │   │
│  │                                                                     │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  SageMaker Training Cluster                                  │   │   │
│  │  │  ├── ml.p3dn.24xlarge (8x V100) — fraud model training      │   │   │
│  │  │  └── ml.g5.48xlarge (8x A10G) — LLM fine-tuning             │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  SageMaker Inference Fleet                                   │   │   │
│  │  │  ├── fraud-detection-prod: 8x ml.c5.2xlarge (auto-scaling)  │   │   │
│  │  │  ├── credit-risk-prod: 4x ml.m5.4xlarge                     │   │   │
│  │  │  └── llm-assistant-async: 2x ml.g5.48xlarge                 │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  VPC Endpoints (PrivateLink — NO traffic to internet)               │   │
│  │  ├── com.amazonaws.us-east-1.sagemaker.api                          │   │
│  │  ├── com.amazonaws.us-east-1.sagemaker.runtime                      │   │
│  │  ├── com.amazonaws.us-east-1.s3                                     │   │
│  │  ├── com.amazonaws.us-east-1.ecr.api                                │   │
│  │  ├── com.amazonaws.us-east-1.ecr.dkr                                │   │
│  │  ├── com.amazonaws.us-east-1.logs                                   │   │
│  │  ├── com.amazonaws.us-east-1.kms                                    │   │
│  │  └── com.amazonaws.us-east-1.sts                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Security Groups                                                    │   │
│  │  ├── sg-sagemaker-training: egress to S3, ECR endpoints             │   │
│  │  ├── sg-sagemaker-inference: ingress 8443 from API Gateway SG       │   │
│  │  └── sg-sagemaker-studio: egress to all SageMaker VPC endpoints     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.3 IAM Architecture for SageMaker

```
IAM Roles:
├── SageMakerExecutionRole-Training
│   ├── AmazonSageMakerFullAccess (scoped)
│   ├── S3: GetObject, PutObject on s3://fintech-datalake-prod/
│   ├── ECR: GetAuthorizationToken, BatchGetImage
│   ├── KMS: Decrypt (training data keys)
│   └── CloudWatch: PutMetricData, CreateLogGroup
│
├── SageMakerExecutionRole-Inference
│   ├── S3: GetObject on s3://fintech-models-prod/
│   ├── ECR: GetAuthorizationToken, BatchGetImage
│   ├── CloudWatch: PutMetricData
│   └── Feature Store: GetRecord (online store reads)
│
├── SageMakerExecutionRole-Pipelines
│   ├── All above roles (assumed during pipeline execution)
│   ├── SageMaker: CreateTrainingJob, CreateModel, CreateEndpoint
│   ├── Lambda: InvokeFunction (for pipeline steps)
│   └── EventBridge: PutRule (for scheduled pipelines)
│
├── SageMakerStudioRole-DataScientist
│   ├── S3: Read/Write on dev prefixes only
│   ├── SageMaker: Start/Stop Studio, training jobs
│   ├── NO production endpoint access
│   └── NO model approval permissions
│
└── SageMakerStudioRole-MLOpsEngineer
    ├── Full SageMaker access (scoped to account)
    ├── S3: All buckets
    ├── Model Registry: Approve/Reject model packages
    └── Endpoint: Create/Update/Delete production endpoints
```

---

## 3.4 Network Flow for Real-Time Inference

```
Client (API Gateway) 
  → API Gateway (public endpoint with auth)
  → VPC Link
  → Application Load Balancer (private subnet)
  → Lambda (feature enrichment from online store)
      ↓ Feature Store lookup (DynamoDB, <3ms)
  → SageMaker Runtime VPC Endpoint
  → Inference Container (ml.c5.2xlarge)
      ↓ Model inference (XGBoost, ~2ms)
  → Response returned upstream
  
Total expected latency breakdown:
  - API Gateway overhead: ~1ms
  - Lambda feature enrichment: ~3ms  
  - SageMaker Runtime: ~0.5ms
  - Inference container: ~2ms
  - Network overhead: ~1ms
  
  Total P50: ~7.5ms ✅ (target < 10ms P99)
```

---

## 3.5 SageMaker Pipelines — The Orchestration Backbone

```python
# Production SageMaker Pipeline for Fraud Detection
# This is what a Staff MLOps Engineer designs and owns

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep, TrainingStep, TransformStep
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.parameters import (
    ParameterInteger, ParameterFloat, ParameterString
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import MetricsSource, ModelMetrics

session = sagemaker.Session()
role = "arn:aws:iam::123456789:role/SageMakerExecutionRole-Pipelines"
region = "us-east-1"
bucket = "fintech-datalake-prod"
prefix = "fraud-detection"

# Pipeline Parameters (configurable at runtime)
training_instance_count = ParameterInteger(
    name="TrainingInstanceCount", default_value=1
)
training_instance_type = ParameterString(
    name="TrainingInstanceType", default_value="ml.m5.4xlarge"
)
accuracy_mse_threshold = ParameterFloat(
    name="AucRocThreshold", default_value=0.94
)
model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="PendingManualApproval"
)

# Step 1: Data Preprocessing
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type="ml.m5.4xlarge",
    instance_count=1,
    base_job_name="fraud-preprocessing",
    role=role,
    sagemaker_session=session,
)

preprocessing_step = ProcessingStep(
    name="FraudDataPreprocessing",
    processor=sklearn_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=f"s3://{bucket}/raw/transactions/",
            destination="/opt/ml/processing/input/transactions"
        ),
        sagemaker.processing.ProcessingInput(
            source=f"s3://{bucket}/raw/customer-profiles/",
            destination="/opt/ml/processing/input/profiles"
        ),
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=f"s3://{bucket}/{prefix}/processed/train"
        ),
        sagemaker.processing.ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/output/validation",
            destination=f"s3://{bucket}/{prefix}/processed/validation"
        ),
        sagemaker.processing.ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
            destination=f"s3://{bucket}/{prefix}/processed/test"
        ),
    ],
    code="code/preprocessing.py",
)

# Step 2: Model Training
xgboost_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1"
    ),
    instance_type=training_instance_type,
    instance_count=training_instance_count,
    output_path=f"s3://{bucket}/{prefix}/models/",
    base_job_name="fraud-training",
    role=role,
    sagemaker_session=session,
    use_spot_instances=True,            # Cost optimization
    max_run=3600,
    max_wait=7200,
    checkpoint_s3_uri=f"s3://{bucket}/{prefix}/checkpoints/",
    enable_sagemaker_metrics=True,
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": 500,
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "eval_metric": "auc",
        "scale_pos_weight": 99,         # handle class imbalance (1% fraud)
    },
)

training_step = TrainingStep(
    name="FraudModelTraining",
    estimator=xgboost_estimator,
    inputs={
        "train": sagemaker.TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig\
                .Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": sagemaker.TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig\
                .Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv"
        ),
    },
)

# Step 3: Model Evaluation
evaluation_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, "1.2-1"),
    command=["python3"],
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="fraud-evaluation",
    role=role,
    sagemaker_session=session,
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

evaluation_step = ProcessingStep(
    name="FraudModelEvaluation",
    processor=evaluation_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        sagemaker.processing.ProcessingInput(
            source=preprocessing_step.properties.ProcessingOutputConfig\
                .Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        ),
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{bucket}/{prefix}/evaluation/"
        ),
    ],
    code="code/evaluation.py",
    property_files=[evaluation_report],
)

# Step 4: Register Model (conditional on AUC-ROC)
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json",
    )
)

from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.model_step import ModelStep

register_step = ModelStep(
    name="RegisterFraudModel",
    step_args=xgboost_estimator.register(
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.c5.2xlarge", "ml.c5.4xlarge"],
        transform_instances=["ml.m5.4xlarge"],
        model_package_group_name="fraud-detection-xgboost",
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
)

# Step 5: Conditional Step — only register if AUC-ROC threshold met
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

auc_condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path="classification_metrics.auc_roc.value",
    ),
    right=accuracy_mse_threshold,
)

condition_step = ConditionStep(
    name="CheckAucRocThreshold",
    conditions=[auc_condition],
    if_steps=[register_step],
    else_steps=[],  # alert via EventBridge if threshold not met
)

# Assemble the Pipeline
pipeline = Pipeline(
    name="fraud-detection-training-pipeline",
    parameters=[
        training_instance_count,
        training_instance_type,
        accuracy_mse_threshold,
        model_approval_status,
    ],
    steps=[
        preprocessing_step,
        training_step,
        evaluation_step,
        condition_step,
    ],
    sagemaker_session=session,
)

# Upsert the pipeline definition
pipeline.upsert(role_arn=role)
print("Pipeline upserted successfully")

# Execute
execution = pipeline.start(
    parameters={
        "TrainingInstanceType": "ml.m5.12xlarge",
        "AucRocThreshold": 0.94,
    }
)
print(f"Pipeline execution started: {execution.arn}")
```

---

## 3.6 Multi-Account Architecture (Enterprise)

Production enterprises do NOT run everything in a single AWS account:

```
AWS Organization
├── Management Account (billing, SCPs)
│
├── ML Platform Accounts
│   ├── ml-dev-account      (data scientists, experiments)
│   ├── ml-staging-account  (integration testing, canary)
│   └── ml-prod-account     (production endpoints, feature store)
│
├── Data Platform Accounts
│   ├── data-lake-account   (S3 data lake, Glue catalog)
│   └── streaming-account   (Kafka/MSK, Kinesis)
│
└── Shared Services Accounts
    ├── security-account    (GuardDuty, Security Hub, Config)
    └── logging-account     (CloudTrail, VPC Flow Logs)
```

**Cross-Account Model Promotion Flow**:

```
ml-dev-account: Train & register model
    ↓ (cross-account Model Registry read)
ml-staging-account: Integration tests, shadow deployment
    ↓ (cross-account Model Registry approval)
ml-prod-account: Production deployment
```

---

*Next: [Section 04 — Data Platform →](04-data-platform.md)*
