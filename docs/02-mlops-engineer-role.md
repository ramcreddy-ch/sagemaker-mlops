# Section 02 — MLOps Engineer Role in Production

> **Context**: What MLOps Engineers actually DO at Goldman Sachs, Capital One, Uber, and Netflix. Not job descriptions — real workflows.

---

## 2.1 The Reality of the MLOps Engineer Role

The job description says: *"Build and maintain ML pipelines."*

The reality at 9:00 AM on a Monday:

```
09:02 AM — PagerDuty alert: fraud_detection_endpoint P99 latency > 50ms
09:05 AM — Check CloudWatch: memory utilization at 94% on ml.c5.2xlarge fleet
09:12 AM — Scale out endpoint from 3 → 8 instances
09:18 AM — P99 back to 8ms, alert resolved
09:20 AM — Root cause: weekend batch job loaded 40GB of new features into online store
09:25 AM — Write incident ticket, schedule post-mortem
09:35 AM — Data scientist pings: "my training job failed after 6 hours"
09:40 AM — Check CloudWatch Logs: CUDA OOM on ml.p3.2xlarge
09:45 AM — Recommend switching to ml.p3.8xlarge or reducing batch size
10:00 AM — Weekly platform review meeting begins...
```

That's the real job.

---

## 2.2 Daily Responsibilities by Phase

### Development Phase Responsibilities

When data scientists are developing models, MLOps engineers:

| Responsibility | Concrete Action |
|---|---|
| **Studio environment management** | Ensure JupyterLab instances have correct IAM roles, VPC access, KMS keys |
| **Data access provisioning** | Grant Lake Formation permissions for dev datasets |
| **Container image management** | Build and push custom training containers to ECR |
| **Development infrastructure** | Maintain SageMaker Domains, user profiles, lifecycle configs |
| **Code review** | Review training scripts for distributed training compatibility, S3 I/O patterns |
| **Cost controls** | Set SageMaker budget alerts; kill zombie notebook instances |

```bash
# Daily task: Find zombie Studio kernel sessions > 8 hours old
aws sagemaker list-apps \
  --domain-id d-xxxxxxxx \
  --query 'Apps[?Status==`InService` && AppType==`KernelGateway`].[AppName,CreationTime]' \
  --output table

# Kill expensive idle instances
aws sagemaker delete-app \
  --domain-id d-xxxxxxxx \
  --user-profile-name researcher-john \
  --app-type KernelGateway \
  --app-name instance/ml.g5.2xlarge/xxxxxx
```

---

### Training Phase Responsibilities

| Responsibility | Concrete Action |
|---|---|
| **Training job monitoring** | CloudWatch dashboards for GPU utilization, throughput, loss curves |
| **Spot interruption handling** | Configure checkpoint/resume logic for spot training jobs |
| **Distributed training setup** | Configure SageMaker distributed training library or Horovod |
| **Cost optimization** | Route eligible jobs to Spot, Managed Warm Pools, Savings Plans |
| **Training quota management** | Request EC2 capacity increases before large experiments |
| **Failed job triage** | Root cause: OOM, CUDA errors, data loading bottlenecks, S3 throttling |

**Common Training Failures Triage**:

```python
# MLOps Engineer's mental model for training failures

FAILURE_TAXONOMY = {
    "ResourceExhaustedError": {
        "symptoms": ["CUDA out of memory", "OOM killed"],
        "causes": ["Batch size too large", "Model too big for instance", "Memory leak"],
        "resolution": "Reduce batch size, upgrade instance, enable gradient checkpointing"
    },
    "DataLoadingBottleneck": {
        "symptoms": ["GPU utilization < 30%", "High DataLoader wait time"],
        "causes": ["S3 GET throttling", "Single-threaded dataloader", "Small files"],
        "resolution": "Increase DataLoader workers, use S3FS, enable S3 prefix sharding"
    },
    "SpotInterruption": {
        "symptoms": ["Job failed with status Stopped", "CheckpointConfig exists"],
        "causes": ["EC2 spot capacity reclaimed"],
        "resolution": "Job auto-resumes from last checkpoint (verify checkpoint logic)"
    },
    "NetworkTimeout": {
        "symptoms": ["Connection timeout in distributed training"],
        "causes": ["Security group blocking port 7777 (NCCL)", "EFA not configured"],
        "resolution": "Check SG rules, verify EFA enabled on instance type"
    }
}
```

---

### Validation Phase Responsibilities

| Responsibility | Concrete Action |
|---|---|
| **Model evaluation** | Run evaluation pipelines against holdout sets, OOT validation |
| **Bias detection** | Run SageMaker Clarify for demographic parity, disparate impact |
| **Model comparison** | Compare new model vs. champion model on production traffic samples |
| **Registry submission** | Submit model to Model Registry with metadata, metrics, lineage |
| **Approval workflow** | Notify model risk team, trigger review process |
| **A/B test design** | Define traffic split, success criteria, rollback triggers |

---

### Deployment Phase Responsibilities

| Responsibility | Concrete Action |
|---|---|
| **Endpoint configuration** | Instance type selection, auto-scaling policies, model config |
| **Traffic management** | Configure Blue/Green or Canary deployment |
| **Smoke testing** | Run synthetic load tests against new endpoint |
| **Canary monitoring** | Watch error rates, latency, prediction distribution for first 30 min |
| **Rollback readiness** | Ensure previous model version is ready for instant rollback |
| **DNS/routing update** | Update API Gateway or internal routing to new endpoint |

---

### Monitoring Phase Responsibilities

| Responsibility | Concrete Action |
|---|---|
| **Drift detection** | Daily data quality and model quality monitoring jobs |
| **Performance dashboards** | Maintain CloudWatch/Grafana dashboards for all production models |
| **Alert tuning** | Reduce false positive alerts; refine drift thresholds |
| **SLA tracking** | Weekly P50/P99/P999 latency reports per model |
| **Feature freshness** | Monitor feature store ingestion lag (< 60s target) |
| **Cost monitoring** | Daily cost per model report; flag budget anomalies |

---

## 2.3 Hourly Workflow Examples

### A Typical Production Day (Senior MLOps Engineer)

```
08:30 — Review overnight CloudWatch alarms (typically 3-10 resolved)
08:45 — Check training job status board (20-40 jobs running overnight)
09:00 — Standup with ML team (5 min: blockers, infra status)
09:15 — Incident triage from overnight PagerDuty history
09:30 — Model deployment review: approve 2 models for canary
10:00 — Review PR: data scientist added new feature to training script
10:30 — Capacity planning: request 4 ml.g5.48xlarge for next LLM experiment
11:00 — Debug: training job failing on EMR feature extraction step
12:00 — Lunch / async Slack responses
13:00 — Write runbook for new endpoint auto-scaling policy
14:00 — Security review: IAM role audit for SageMaker execution roles
15:00 — 1:1 with ML Engineer on distributed training optimization
16:00 — Cost review: identify $12K of wasted Notebook instance spending
17:00 — Update monitoring dashboards for new model version
17:30 — Respond to on-call questions; handoff to Asia-Pacific team
```

---

## 2.4 Weekly Operational Activities

### Monday — Infrastructure Health Review

```bash
#!/bin/bash
# Weekly health check script run every Monday 09:00

echo "=== SageMaker Endpoint Health ==="
aws sagemaker list-endpoints \
  --status-equals InService \
  --query 'Endpoints[*].[EndpointName,CreationTime]' \
  --output table

echo "=== Training Job Success Rate (last 7 days) ==="
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name TrainingJobsSucceededCount \
  --start-time $(date -d '7 days ago' --iso-8601=seconds) \
  --end-time $(date --iso-8601=seconds) \
  --period 604800 \
  --statistics Sum

echo "=== Feature Store Ingestion Lag ==="
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name IngestionToStoreTime \
  --dimensions Name=FeatureGroupName,Value=fraud-features-v3 \
  --start-time $(date -d '1 hour ago' --iso-8601=seconds) \
  --end-time $(date --iso-8601=seconds) \
  --period 3600 \
  --statistics Maximum
```

### Tuesday — Model Performance Review

- Review prediction quality metrics for all production models
- Compare against SLA thresholds (e.g., fraud AUC-ROC > 0.94)
- Review data drift reports from SageMaker Model Monitor
- Schedule retraining if drift detected

### Wednesday — Cost Optimization Review

```python
# Weekly cost analysis
import boto3
from datetime import datetime, timedelta

ce = boto3.client('ce')

response = ce.get_cost_and_usage(
    TimePeriod={
        'Start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        'End': datetime.now().strftime('%Y-%m-%d')
    },
    Granularity='DAILY',
    Filter={
        'Dimensions': {
            'Key': 'SERVICE',
            'Values': ['Amazon SageMaker']
        }
    },
    GroupBy=[
        {'Type': 'TAG', 'Key': 'model-name'},
        {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
    ],
    Metrics=['UnblendedCost']
)

# Typical output:
# fraud-detection-endpoint: $2,340/week
# risk-scoring-training: $8,100/week  ← investigate spot usage
# llm-research-assistant: $15,200/week ← high, check utilization
```

### Thursday — Reliability & SLA Review

- P99 latency trends per endpoint
- Error rate analysis (5xx, timeout rates)
- SLA breach reports to stakeholders
- Capacity planning adjustments

### Friday — Platform Improvements

- PR reviews for MLOps tooling changes
- Documentation updates
- Runbook improvements based on week's incidents
- Backlog grooming for MLOps platform roadmap

---

## 2.5 Monthly Platform Reviews

### Monthly Business Review (MBR) Items for MLOps

| Metric | Target | Typical Actual |
|---|---|---|
| **Model SLA Uptime** | 99.99% | 99.97% |
| **Deployment Frequency** | 20/month | 18/month |
| **MTTD (detection)** | < 5 min | 4.2 min |
| **MTTR (recovery)** | < 30 min | 22 min |
| **Training Cost** | Budget ±10% | +8% |
| **Inference Cost/1M requests** | < $50 | $43 |
| **P99 Latency (fraud endpoint)** | < 10ms | 7.8ms |
| **Drift Incidents** | < 2/month | 1 |

---

## 2.6 Incident Response Responsibilities

### Severity Levels and Response

```
SEV-1: Production endpoint down / fraud detection offline
  Response Time: < 5 minutes
  Actions: Page on-call engineer, escalate to Staff within 10 min
  Resolution Target: < 30 minutes
  
SEV-2: Latency degraded > 3x baseline / error rate > 1%
  Response Time: < 15 minutes
  Actions: Engineer investigates, page Staff if no resolution in 20 min
  Resolution Target: < 2 hours

SEV-3: Data drift detected / model quality degraded
  Response Time: < 1 hour  
  Actions: Investigate during business hours
  Resolution Target: < 24 hours (can retrain)

SEV-4: Cost anomaly / quota warning
  Response Time: < 4 hours
  Actions: Investigate root cause, optimize or request quota increase
  Resolution Target: < 3 days
```

### Incident Response Runbook Template

```markdown
## Incident: [Title]
**Severity**: SEV-1 / SEV-2 / SEV-3
**Date**: YYYY-MM-DD HH:MM UTC
**Responder**: [name]

### Detection
- Alert source: PagerDuty / CloudWatch / Grafana
- First detected: HH:MM UTC
- Symptom: [description]

### Investigation Commands
```bash
# 1. Check endpoint status
aws sagemaker describe-endpoint --endpoint-name fraud-detection-prod

# 2. Check CloudWatch metrics
aws cloudwatch get-metric-data ...

# 3. Check container logs
aws logs tail /aws/sagemaker/Endpoints/fraud-detection-prod
```

### Timeline
- HH:MM — Alert triggered
- HH:MM — Investigation started
- HH:MM — Root cause identified
- HH:MM — Fix applied
- HH:MM — Resolution confirmed

### Root Cause
[Description]

### Resolution
[Steps taken]

### Prevention
[What to change to prevent recurrence]
```

---

## 2.7 Disaster Recovery Responsibilities

MLOps Engineers own the DR plan for the ML platform:

| Component | DR Strategy | RTO | RPO |
|---|---|---|---|
| **Production Endpoints** | Multi-region active-active | 5 min | 0 |
| **Model Registry** | Cross-region replication | 1 hour | 15 min |
| **Feature Store (online)** | DynamoDB Global Tables | 5 min | 1 min |
| **Feature Store (offline)** | S3 Cross-Region Replication | 4 hours | 1 hour |
| **Training Pipelines** | CodePipeline cross-region | 2 hours | N/A |
| **Monitoring Data** | CloudWatch cross-region | 1 hour | 5 min |

---

## 2.8 Staff vs. Senior vs. Lead MLOps Engineer

Understanding the differences helps with interview preparation:

```
SENIOR MLOPS ENGINEER (5-8 years):
  - Executes well-defined MLOps patterns
  - Owns specific components (training pipeline, monitoring)
  - Responds to incidents, follows runbooks
  - Writes solid automation scripts
  - Reviews PRs from junior engineers

LEAD MLOPS ENGINEER (8-12 years):
  - Designs new platform components
  - Sets technical direction for a team
  - Creates runbooks and standards
  - Cross-team collaboration (DS, DE, Security)
  - Leads post-mortems
  - Owns platform SLAs

STAFF MLOPS ENGINEER (12+ years):
  - Platform architecture across business units
  - Technology selection and vendor evaluation
  - Long-term roadmap (6-18 month horizon)
  - Drive organizational MLOps maturity
  - Executive reporting on platform health
  - Design for failure (chaos engineering mindset)
  - Define ML governance frameworks
  - Influence without authority across teams
```

---

## 2.9 Tools the MLOps Engineer Uses Daily

```
AWS Tools:
  ├── SageMaker Studio (development, debugging)
  ├── CloudWatch (metrics, logs, alarms)
  ├── AWS Cost Explorer (daily cost monitoring)
  ├── IAM Access Analyzer (security posture)
  └── AWS Service Quotas (capacity management)

Observability:
  ├── Grafana (dashboards)
  ├── Prometheus (metrics collection)
  ├── OpenSearch (log aggregation)
  └── PagerDuty (alerting, incident management)

Collaboration:
  ├── JIRA (incident and project tracking)
  ├── Confluence (runbooks, architecture docs)
  ├── Slack (ops alerts, team communication)
  └── GitHub (code, IaC, pipeline definitions)

Development:
  ├── VS Code / PyCharm
  ├── boto3 + sagemaker Python SDK
  ├── Terraform (infrastructure)
  └── Docker (container builds)
```

---

*Next: [Section 03 — SageMaker Architecture Deep Dive →](03-sagemaker-architecture.md)*
