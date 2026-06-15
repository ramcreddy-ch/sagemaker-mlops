# Section 15 — Monitoring & Observability

> **Production Context**: At enterprise scale, models fail silently. A model's accuracy doesn't throw a stack trace when it degrades. You need proactive monitoring for Data Drift, Concept Drift, and Operational Metrics.

---

## 15.1 Observability Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ML OBSERVABILITY ARCHITECTURE                            │
│                                                                             │
│  1. OPERATIONAL METRICS (CloudWatch / Prometheus)                           │
│     ├── Latency (P50, P90, P99)                                             │
│     ├── Error Rates (4xx, 5xx)                                              │
│     ├── Throughput (Invocations/sec)                                        │
│     └── Resource Utilization (CPU, GPU, Memory)                             │
│                                                                             │
│  2. DATA QUALITY & DRIFT (SageMaker Model Monitor)                          │
│     ├── Data Capture (S3) ← Endpoint captures 20% of traffic               │
│     ├── Baseline Data (S3) ← Training dataset statistics                    │
│     └── Monitoring Schedule ← Hourly/Daily Processing Job comparing data    │
│                                                                             │
│  3. MODEL QUALITY (Concept Drift)                                           │
│     ├── Ground Truth Ingestion ← Wait for actual fraud labels (days/weeks) │
│     └── Model Quality Monitor ← Compares predictions vs. ground truth       │
│                                                                             │
│  4. BIAS & EXPLAINABILITY (Clarify)                                         │
│     └── Feature Attribution ← SHAP values for why a decision was made       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 15.2 SageMaker Model Monitor (Data Drift)

```python
# code/monitoring/model_monitor_setup.py
"""
Setup Data Quality monitoring for a production endpoint.
Detects if the incoming inference data distribution drifts from the training data.
"""

import boto3
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

session = sagemaker.Session()
role = 'arn:aws:iam::123456789:role/SageMakerExecutionRole-Monitor'

def setup_data_drift_monitor(
    endpoint_name: str = 'fraud-detection-prod',
    baseline_dataset_uri: str = 's3://fintech-datalake-prod/fraud-detection/processed/train/train.csv',
    baseline_results_uri: str = 's3://fintech-datalake-prod/monitoring/baselines/fraud/',
):
    """
    1. Create a baseline from training data.
    2. Schedule a monitor to compare endpoint traffic against this baseline.
    """
    
    # 1. Create Data Quality Monitor
    my_monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=session
    )
    
    # 2. Suggest Baseline (Runs a Processing Job)
    print("Running baseline suggestion job...")
    my_monitor.suggest_baseline(
        baseline_dataset=baseline_dataset_uri,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=baseline_results_uri,
        wait=True
    )
    
    print("Baseline generated at:", my_monitor.latest_baselining_job.outputs[0].destination)
    
    # 3. Schedule the Monitor
    # Runs daily at midnight UTC
    from sagemaker.model_monitor import CronExpressionGenerator
    
    schedule_name = f"{endpoint_name}-daily-monitor"
    
    print("Creating monitoring schedule...")
    my_monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=endpoint_name,
        output_s3_uri=f"s3://fintech-datalake-prod/monitoring/reports/{endpoint_name}/",
        statistics=my_monitor.baseline_statistics(),
        constraints=my_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=True  # Important: Emits drift metrics to CloudWatch
    )
    
    print(f"Monitoring schedule {schedule_name} created successfully.")
```

---

## 15.3 Creating CloudWatch Dashboards

```python
# code/monitoring/dashboard_creation.py
"""
Programmatically create a CloudWatch Dashboard for the ML Platform.
Includes both operational and drift metrics.
"""

import boto3
import json

def create_ml_dashboard(endpoint_name: str = 'fraud-detection-prod'):
    cw = boto3.client('cloudwatch')
    
    # Define dashboard widgets
    widgets = [
        # 1. Operational Metrics: Latency
        {
            "type": "metric",
            "x": 0, "y": 0, "width": 12, "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "ModelLatency", "EndpointName", endpoint_name, {"stat": "p99", "color": "#d62728"}],
                    [".", ".", ".", ".", {"stat": "p50", "color": "#1f77b4"}]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": "us-east-1",
                "title": f"Endpoint Latency (P50/P99) - {endpoint_name}",
                "period": 300
            }
        },
        # 2. Operational Metrics: Errors & Invocations
        {
            "type": "metric",
            "x": 12, "y": 0, "width": 12, "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "Invocations", "EndpointName", endpoint_name, {"stat": "Sum"}],
                    ["AWS/SageMaker", "Invocation5XXErrors", "EndpointName", endpoint_name, {"stat": "Sum", "color": "#d62728"}]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": "us-east-1",
                "title": "Traffic & Errors",
                "period": 300
            }
        },
        # 3. Model Monitor: Feature Drift (PSI)
        {
            "type": "metric",
            "x": 0, "y": 6, "width": 24, "height": 6,
            "properties": {
                "metrics": [
                    ["aws/sagemaker/Endpoints/data-metrics", "feature_baseline_drift_amount", "Endpoint", endpoint_name, "Feature", "amount"],
                    [".", ".", ".", ".", ".", "transaction_velocity_1h"],
                    [".", ".", ".", ".", ".", "distance_from_home"]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": "us-east-1",
                "title": "Data Drift (Feature Violations)",
                "period": 86400, # Daily
                "annotations": {
                    "horizontal": [{"value": 0.1, "label": "Warning", "color": "#ff7f0e"},
                                   {"value": 0.25, "label": "Critical", "color": "#d62728"}]
                }
            }
        }
    ]
    
    dashboard_body = {"widgets": widgets}
    
    cw.put_dashboard(
        DashboardName=f"{endpoint_name}-Operations",
        DashboardBody=json.dumps(dashboard_body)
    )
    print(f"Dashboard {endpoint_name}-Operations created.")

# Run
create_ml_dashboard()
```

---

*Next: [Section 16 — Incident Management →](16-incident-management.md)*
