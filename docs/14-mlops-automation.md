# Section 14 — MLOps Automation & Event-Driven Architecture

> **Production Context**: Scale requires automation. Enterprise MLOps platforms use EventBridge to decouple systems, automatically reacting to training completions, model approvals, and infrastructure failures.

---

## 14.1 Event-Driven MLOps Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVENT-DRIVEN MLOPS PLATFORM                              │
│                                                                             │
│  PRODUCERS (Emitting Events)          EVENT BUS (AWS EventBridge)           │
│  ───────────────────────────          ───────────────────────────           │
│  SageMaker Training ────────────────▶                                       │
│  SageMaker Pipelines ───────────────▶  Rule: Job Failed                     │
│  SageMaker Model Registry ──────────▶  Rule: Model Approved                 │
│  SageMaker Model Monitor ───────────▶  Rule: Drift Detected                 │
│  CloudWatch Alarms ─────────────────▶                                       │
│                                            │                                │
│                                            ▼                                │
│                                       CONSUMERS (Taking Action)             │
│                                       ─────────────────────────             │
│                                       ──▶ Lambda (Trigger CI/CD Pipeline)   │
│                                       ──▶ SNS (Slack/PagerDuty Alert)       │
│                                       ──▶ Lambda (Auto-Rollback)            │
│                                       ──▶ Step Functions (Complex Workflow) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 14.2 Automating Endpoint Rollbacks

```python
# code/automation/auto_rollback.py
"""
Automatically roll back an endpoint to the previous known good model
if the new model trips CloudWatch alarms (e.g., latency spikes, 5xx errors).
"""

import boto3
import logging
import json

logger = logging.getLogger(__name__)

def lambda_handler(event, context):
    """
    Triggered by CloudWatch Alarm via EventBridge.
    Alarm: "Fraud-Endpoint-P99-Latency-High" or "Fraud-Endpoint-Error-Rate-High"
    """
    sm = boto3.client('sagemaker')
    
    # 1. Parse Alarm Details
    alarm_name = event['detail']['alarmName']
    
    # Extract endpoint name from alarm description or tags (simplified here)
    endpoint_name = 'fraud-detection-prod'
    logger.error(f"🚨 Alarm {alarm_name} triggered for endpoint {endpoint_name}. Initiating auto-rollback.")
    
    try:
        # 2. Get current endpoint configuration
        endpoint_info = sm.describe_endpoint(EndpointName=endpoint_name)
        current_config_name = endpoint_info['EndpointConfigName']
        
        # 3. Find the previous configuration (rollback target)
        # We list configs containing the endpoint name, sorted by creation time
        configs = sm.list_endpoint_configs(
            NameContains=endpoint_name,
            SortBy='CreationTime',
            SortOrder='Descending'
        )['EndpointConfigs']
        
        # The first config is likely the current one. We want the one before that.
        # Ensure we don't pick the current config
        previous_configs = [c['EndpointConfigName'] for c in configs if c['EndpointConfigName'] != current_config_name]
        
        if not previous_configs:
            logger.critical("❌ Rollback failed: No previous configuration found.")
            _notify_pagerduty(endpoint_name, "Rollback failed: No previous configuration.")
            return
            
        rollback_config_name = previous_configs[0]
        logger.info(f"Rolling back from {current_config_name} to {rollback_config_name}")
        
        # 4. Execute the Rollback
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=rollback_config_name
        )
        
        # 5. Notify Team
        _notify_pagerduty(endpoint_name, f"Auto-rollback initiated to {rollback_config_name} due to {alarm_name}.")
        
        # Optional: Mark the bad model as 'Rejected' in the Model Registry
        _reject_bad_model(current_config_name, alarm_name)
        
    except Exception as e:
        logger.critical(f"❌ Rollback execution failed: {e}")
        _notify_pagerduty(endpoint_name, f"URGENT: Auto-rollback failed: {e}")


def _notify_pagerduty(endpoint_name, message):
    """Send alert to PagerDuty/Slack via SNS."""
    sns = boto3.client('sns')
    sns.publish(
        TopicArn='arn:aws:sns:us-east-1:123456789:mlops-critical-alerts',
        Subject=f"ENDPOINT ROLLBACK: {endpoint_name}",
        Message=message
    )

def _reject_bad_model(bad_config_name, reason):
    """Find the model in the bad config and reject it in the registry."""
    # (Implementation details omitted for brevity: DescribeEndpointConfig -> get ModelName -> DescribeModel -> get ModelPackageName -> UpdateModelPackage)
    pass
```

---

## 14.3 Automated Cost Cleanup (Zombie Resource Killer)

```python
# code/automation/zombie_killer.py
"""
Runs nightly to terminate idle SageMaker resources that are wasting money.
Essential for large teams where data scientists forget to shut down instances.
"""

import boto3
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

def lambda_handler(event, context):
    """Nightly cron job to clean up expensive idle resources."""
    sm = boto3.client('sagemaker')
    
    total_savings_estimated = 0
    
    # 1. Clean up idle Studio Apps (JupyterLab kernels)
    # ------------------------------------------------
    logger.info("Checking for idle Studio Apps...")
    domains = sm.list_domains()['Domains']
    
    for domain in domains:
        apps = sm.list_apps(DomainIdEquals=domain['DomainId'])['Apps']
        for app in apps:
            if app['Status'] == 'InService' and app['AppType'] == 'KernelGateway':
                # Check how long it's been running
                creation_time = app['CreationTime']
                running_hours = (datetime.now(timezone.utc) - creation_time).total_seconds() / 3600
                
                # If running for > 12 hours (likely left overnight)
                if running_hours > 12:
                    logger.info(f"Terminating idle app {app['AppName']} (User: {app['UserProfileName']}) - Running {running_hours:.1f}h")
                    
                    sm.delete_app(
                        DomainId=domain['DomainId'],
                        UserProfileName=app['UserProfileName'],
                        AppType=app['AppType'],
                        AppName=app['AppName']
                    )
                    # Rough estimate: ml.t3.medium is ~$0.05/hr
                    total_savings_estimated += 0.05 * 12 
                    
    # 2. Clean up old Endpoint Configurations
    # ---------------------------------------
    # SageMaker retains endpoint configs forever. 
    # Delete configs older than 30 days that are NOT currently in use.
    logger.info("Checking for old Endpoint Configs...")
    active_endpoints = [e['EndpointName'] for e in sm.list_endpoints()['Endpoints']]
    
    # Get all active configs
    active_configs = []
    for ep in active_endpoints:
        active_configs.append(sm.describe_endpoint(EndpointName=ep)['EndpointConfigName'])
        
    all_configs = sm.list_endpoint_configs()['EndpointConfigs']
    for config in all_configs:
        config_name = config['EndpointConfigName']
        creation_time = config['CreationTime']
        age_days = (datetime.now(timezone.utc) - creation_time).total_seconds() / 86400
        
        if age_days > 30 and config_name not in active_configs:
            logger.info(f"Deleting unused endpoint config: {config_name}")
            sm.delete_endpoint_config(EndpointConfigName=config_name)
            
    logger.info(f"Cleanup complete. Estimated monthly savings: ${total_savings_estimated * 30:.2f}")
```

---

*Next: [Section 15 — Monitoring & Observability →](15-monitoring-observability.md)*
