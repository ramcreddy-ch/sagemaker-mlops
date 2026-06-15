# Section 01 — AWS SageMaker Fundamentals

> **Production Context**: Every concept here is framed around the Enterprise Financial Intelligence Platform use case — Fraud Detection, Risk Scoring, LLM Financial Assistant.

---

## 1.1 What Is Amazon SageMaker?

Amazon SageMaker is a **fully managed Machine Learning platform** that provides every tool needed to build, train, deploy, and monitor ML models at production scale — without managing the underlying infrastructure.

But that's the marketing definition. Here's the **real production definition**:

> SageMaker is AWS's answer to Uber's Michelangelo, Google's Vertex AI, and Meta's FBLearner — a unified control plane for the entire ML lifecycle, designed to remove the infrastructure burden from data scientists and ML engineers.

### What SageMaker Actually Provides

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAGEMAKER CAPABILITY MAP                            │
│                                                                             │
│  DATA PREP          TRAINING              DEPLOYMENT           OPS          │
│  ──────────         ────────              ──────────           ───          │
│  Data Wrangler      Training Jobs         Real-Time Endpoints  Model Monitor│
│  Processing Jobs    Distributed Train     Async Endpoints      Experiments  │
│  Feature Store      HPO (Tuning)          Batch Transform      Pipelines    │
│  Ground Truth       Spot Training         Serverless Inference Model Registry│
│  Clarify            Managed Warm Pools    Multi-Model Endpoint Debugger     │
│                                           Shadow Testing                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.2 Why SageMaker Exists

Before SageMaker (circa 2017), production ML required:

1. **Provisioning EC2 clusters** — manually spinning up p3.16xlarge instances
2. **Installing CUDA drivers** — hours of driver/CUDA/cuDNN version conflicts
3. **Managing training storage** — EFS mounts, S3 sync scripts, checkpoint logic
4. **Custom serving infrastructure** — Flask/FastAPI on EC2, manual scaling
5. **No experiment tracking** — Excel spreadsheets with model parameters
6. **No model versioning** — S3 folders named `model_v2_final_FINAL.tar.gz`

SageMaker solves all of this by providing:

- **Managed compute** — spin up 100-GPU clusters in < 3 minutes, terminated on job completion
- **Pre-built containers** — optimized XGBoost, PyTorch, TensorFlow DLCs
- **Integrated storage** — automatic S3 input/output channel management
- **Managed serving** — auto-scaling inference fleets with health checks
- **Experiment tracking** — built-in lineage from data → training → model → endpoint

### The Real Reason Fortune 500s Use SageMaker

At Goldman Sachs or JPMorgan, the actual value proposition is:

| Pain Point | SageMaker Solution |
|---|---|
| Regulatory audit trails | Full model lineage in Model Registry |
| Data governance | S3 + Lake Formation + Glue Data Catalog |
| Security/compliance | VPC isolation, KMS encryption, CloudTrail |
| Cost predictability | Spot training, savings plans, auto-scaling |
| Talent scalability | Data scientists use Studio; MLOps manages platform |

---

## 1.3 When To Use SageMaker

### ✅ Use SageMaker When:

| Scenario | Why SageMaker Wins |
|---|---|
| **Your team is AWS-first** | Tight integration with S3, RDS, Kinesis, Redshift, EMR |
| **You need managed compute** | No cluster management, automatic provisioning |
| **Regulatory compliance matters** | SOC2, HIPAA BAA, PCI-DSS controls built-in |
| **You want MLOps out of the box** | Pipelines, Experiments, Model Registry all native |
| **Cost predictability** | Pay-per-use, no idle infra costs when training stops |
| **Feature store is important** | Native offline+online store with sub-5ms lookups |
| **Batch inference at scale** | Batch Transform can process TB of data efficiently |
| **LLM deployment** | JumpStart has 300+ foundation models ready to deploy |

### ❌ Do NOT Use SageMaker When:

| Scenario | Better Alternative |
|---|---|
| **Multi-cloud strategy** | Kubeflow on EKS / GKE; MLflow on AKS |
| **Team is heavily GCP-native** | Vertex AI with GKE |
| **Sub-millisecond inference** | Custom Triton on EKS with direct GPU access |
| **Massive Spark workloads** | EMR + Delta Lake (SageMaker Processing = overhead) |
| **Vendor lock-in concerns** | Metaflow + S3 + Kubernetes |
| **SageMaker cost is prohibitive** | Self-managed K8s + KServe |
| **Real-time streaming ML** | Flink ML, custom Kafka consumers |

---

## 1.4 SageMaker vs. Competing Platforms

### SageMaker vs. Databricks

```
┌──────────────────┬─────────────────────────────┬──────────────────────────────┐
│ Dimension        │ AWS SageMaker               │ Databricks                   │
├──────────────────┼─────────────────────────────┼──────────────────────────────┤
│ Training         │ Managed jobs (containers)   │ Spark clusters + MLflow      │
│ Feature Store    │ Native (online + offline)   │ Feature Store (Unity Catalog)│
│ Data Transform   │ Processing Jobs, Wrangler   │ Spark SQL, Delta Live Tables │
│ Model Serving    │ Native endpoints            │ MLflow Model Serving         │
│ Experiment Track │ SageMaker Experiments       │ MLflow (native)              │
│ Model Registry   │ SageMaker Model Registry    │ MLflow Model Registry        │
│ LLMs             │ JumpStart, SageMaker Domain │ Mosaic AI, DBRX              │
│ Cost             │ Pay per training job        │ DBU pricing (complex)        │
│ Best For         │ AWS-native, compliance      │ Spark-heavy, analytics teams │
└──────────────────┴─────────────────────────────┴──────────────────────────────┘
```

**Real-World Verdict**: At Capital One, they use Databricks for data engineering (Spark + Delta Lake) and SageMaker for model training/deployment. The two platforms complement each other.

---

### SageMaker vs. Google Vertex AI

```
┌──────────────────┬─────────────────────────────┬──────────────────────────────┐
│ Dimension        │ AWS SageMaker               │ Google Vertex AI             │
├──────────────────┼─────────────────────────────┼──────────────────────────────┤
│ TPU Support      │ Trainium (custom)           │ TPU v4/v5 (best in class)    │
│ Model Garden     │ JumpStart (300+ models)     │ Model Garden (200+ models)   │
│ Pipelines        │ SageMaker Pipelines         │ Vertex Pipelines (KFP based) │
│ Feature Store    │ Native, mature              │ Vertex Feature Store         │
│ LLM Fine-tuning  │ Good (Llama, Mistral)       │ Best (Gemini tuning native)  │
│ Multi-tenancy    │ SageMaker Domains           │ Vertex AI User Stores        │
│ MLflow           │ Supported (managed)         │ Not native                   │
│ Best For         │ AWS-native orgs             │ GCP-native, LLM/TPU workloads│
└──────────────────┴─────────────────────────────┴──────────────────────────────┘
```

---

### SageMaker vs. Azure ML

```
┌──────────────────┬─────────────────────────────┬──────────────────────────────┐
│ Dimension        │ AWS SageMaker               │ Azure ML                     │
├──────────────────┼─────────────────────────────┼──────────────────────────────┤
│ MLOps Maturity   │ Very mature (2017)          │ Mature (2018)                │
│ Responsible AI   │ Clarify (bias detection)    │ Responsible AI Dashboard     │
│ AutoML           │ Autopilot                   │ AutoML (very strong)         │
│ Pipelines        │ SageMaker Pipelines         │ Azure ML Pipelines           │
│ Designer (GUI)   │ Canvas (no-code)            │ Designer (drag-and-drop)     │
│ Kubernetes       │ SageMaker on EKS            │ Azure Arc + AKS              │
│ Active Directory │ IAM + SSO                   │ Azure AD (enterprise auth)   │
│ Best For         │ AWS-native, financial svcs  │ Microsoft enterprises        │
└──────────────────┴─────────────────────────────┴──────────────────────────────┘
```

---

### SageMaker vs. EKS (for ML)

This is the most important comparison because many organizations debate this internally.

```
┌──────────────────┬─────────────────────────────┬──────────────────────────────┐
│ Dimension        │ AWS SageMaker               │ EKS + KServe/Kubeflow        │
├──────────────────┼─────────────────────────────┼──────────────────────────────┤
│ Setup Time       │ Minutes                     │ Days/Weeks                   │
│ Ops Overhead     │ Very Low                    │ High (k8s expertise needed)  │
│ Cost             │ Higher per unit             │ Lower (spot nodes)           │
│ Flexibility      │ Container-based, limited    │ Full control                 │
│ Multi-framework  │ Good (DLCs)                 │ Excellent (any framework)    │
│ Latency          │ ~1-5ms overhead             │ Near-bare-metal              │
│ Vendor Lock-in   │ High                        │ Low                          │
│ Feature Store    │ Native                      │ Feast (open source)          │
│ Best For         │ Teams without k8s expertise │ Platform teams, low latency  │
└──────────────────┴─────────────────────────────┴──────────────────────────────┘
```

**Staff Engineer Decision Framework**:

```
IF team_has_kubernetes_expertise AND latency_requirement < 1ms:
    → Use EKS + KServe
ELIF organization_is_aws_native AND compliance_required:
    → Use SageMaker
ELIF team_is_data_science_heavy AND ops_capacity_low:
    → Use SageMaker (lower ops burden)
ELIF multi_cloud_strategy:
    → Use Kubeflow on EKS (portable)
```

---

## 1.5 SageMaker Control Plane vs. Data Plane

This is a critical distinction that every Staff MLOps Engineer must understand.

### Control Plane

The **Control Plane** manages the LIFECYCLE of SageMaker resources. It handles:

- Creating/deleting training jobs
- Creating/updating endpoints
- Registering models in the Model Registry
- Managing SageMaker Pipelines definitions
- Feature Group schema management

```
Control Plane APIs (via boto3 / SageMaker SDK):
  ┌─────────────────────────────────────────────┐
  │  CreateTrainingJob()                        │
  │  CreateModel()                              │
  │  CreateEndpointConfig()                     │
  │  CreateEndpoint()                           │
  │  UpdateEndpoint()         ← these are       │
  │  DeleteEndpoint()           CONTROL PLANE   │
  │  CreateFeatureGroup()       operations      │
  │  PutRecord()                                │
  └─────────────────────────────────────────────┘
          ↓
  SageMaker Control Plane (us-east-1)
  (Runs in AWS-managed account, not yours)
```

**Key Production Insight**: Control Plane API calls are **eventually consistent** and subject to **API throttling**. At scale (100+ concurrent training jobs), you WILL hit throttling limits. Always implement exponential backoff.

### Data Plane

The **Data Plane** handles actual inference requests to your deployed endpoints:

```
Client Application
      ↓
SageMaker Runtime Endpoint (your VPC)
      ↓
InvokeEndpoint() — THIS IS DATA PLANE
      ↓
Container (your model) running on ml.c5.xlarge, ml.g5.2xlarge, etc.
```

**Critical Production Insight**: 
- Control Plane = `sagemaker.us-east-1.amazonaws.com`
- Data Plane = `runtime.sagemaker.us-east-1.amazonaws.com`

They have **separate throttle limits** and **separate quotas**. A Control Plane outage does NOT affect running endpoints (Data Plane stays up).

---

## 1.6 Real-World Usage Patterns

### Pattern 1: Fraud Detection Pipeline (Capital One Style)

```
Kafka (transaction events)
    ↓
Kinesis Data Streams
    ↓
Lambda (feature lookup from Feature Store Online)
    ↓
SageMaker Real-Time Endpoint (XGBoost fraud model)
    ↓
Result → Risk Engine → Transaction Approved/Blocked
    ↓
S3 → Training Data Lake → Weekly Retraining Pipeline
```

### Pattern 2: Credit Risk Scoring (JPMorgan Style)

```
Loan Application (batch, nightly)
    ↓
EMR Spark (feature computation from Redshift/S3)
    ↓
SageMaker Feature Store (offline ingestion)
    ↓
SageMaker Training Job (LightGBM on ml.m5.12xlarge)
    ↓
SageMaker Batch Transform (score 50M customers overnight)
    ↓
Results → Redshift → Risk Dashboard
```

### Pattern 3: LLM Financial Research Assistant (Goldman Sachs Style)

```
Analyst Query (via internal API)
    ↓
RAG Retrieval (OpenSearch + Titan Embeddings)
    ↓
SageMaker Async Endpoint (Llama 3 70B on ml.g5.48xlarge)
    ↓
Streamed Response via S3 presigned URL
    ↓
Analyst Dashboard
```

---

## 1.7 SageMaker Service Limits (Production Reality)

Every Senior MLOps Engineer should know these limits cold:

| Resource | Default Limit | Typical Enterprise Limit |
|---|---|---|
| Training jobs in parallel | 20 | 500+ |
| Endpoints per account | 10 | 500+ |
| Endpoint invocations/sec per endpoint | 10,000 | 100,000+ |
| Feature Groups | 100 | 2,000+ |
| Model packages in Registry | 1,000 | 10,000+ |
| ml.p3.16xlarge instances | 0 (!) | 16-32 |
| ml.g5.48xlarge instances | 0 (!) | 4-16 |
| Processing job max runtime | 5 days | 5 days (fixed) |

> ⚠️ **Production Gotcha**: GPU instances have a default limit of **ZERO** in a new AWS account. You must pre-request capacity weeks in advance. At Goldman Sachs, there is a dedicated AWS capacity manager who handles this.

---

## 1.8 SageMaker Studio — The Developer Environment

SageMaker Studio is the IDE for ML development. In production:

```
SageMaker Studio
├── Notebooks (JupyterLab 3.0 based)
├── Studio Classic (JupyterLab 1.0 based, being deprecated)
├── Code Editor (VS Code-based, 2023+)
├── JupyterLab (standalone)
├── RStudio (for R users)
└── Canvas (no-code AutoML)
```

### Production Studio Configuration

```python
# Production Studio Domain configuration
# THIS IS WHAT A STAFF ENGINEER SETS UP
studio_config = {
    "DomainName": "fintech-mlplatform-prod",
    "AuthMode": "SSO",  # NOT IAM — always use SSO in enterprise
    "DefaultUserSettings": {
        "ExecutionRole": "arn:aws:iam::123456789:role/SageMakerStudioRole",
        "SecurityGroups": ["sg-studio-prod"],
        "SharingSettings": {
            "NotebookOutputOption": "Disabled",  # security — no output sharing
        },
        "JupyterServerAppSettings": {
            "DefaultResourceSpec": {
                "InstanceType": "system",  # lightweight notebook server
            }
        },
        "KernelGatewayAppSettings": {
            "DefaultResourceSpec": {
                "InstanceType": "ml.t3.medium",  # default kernel instance
                "SageMakerImageArn": "arn:aws:sagemaker:us-east-1:...",
            },
            "CustomImages": [
                {
                    "ImageName": "fintech-ml-custom-image",  # company image
                    "AppImageConfigName": "fintech-image-config",
                }
            ]
        }
    },
    "SubnetIds": ["subnet-private-a", "subnet-private-b"],
    "VpcId": "vpc-prod-ml",
    "AppNetworkAccessType": "VpcOnly",  # NEVER use PublicInternetOnly
    "KmsKeyId": "arn:aws:kms:us-east-1:...:key/...",
}
```

---

## Key Takeaways — SageMaker Fundamentals

1. **SageMaker is a managed control plane** — it abstracts infrastructure, not ML logic
2. **Control plane ≠ Data plane** — endpoint failures don't mean SageMaker is down
3. **GPU quotas start at ZERO** — request capacity well in advance
4. **SageMaker wins on compliance** — VPC isolation, KMS, CloudTrail built-in
5. **EKS beats SageMaker on latency and cost** at scale — know when to use which
6. **Always use SSO in Studio** — IAM auth is only for small teams / dev accounts
7. **Throttling is real** — implement retries on all Control Plane API calls

---

*Next: [Section 02 — MLOps Engineer Role in Production →](02-mlops-engineer-role.md)*
