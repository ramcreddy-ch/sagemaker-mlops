# Section 12 — GPU Management & Capacity Planning

> **Production Context**: GPU capacity is the most constrained resource in modern ML. Large enterprises have dedicated MLOps engineers just managing GPU pools, avoiding limits, and optimizing cost.

---

## 12.1 AWS GPU Instance Types (2024 Reality)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  GPU INSTANCE SELECTION GUIDE                                                  │
│                                                                                │
│  Instance Family  | Use Case                          | Cost/hr | Memory     │
│  ─────────────────|───────────────────────────────────|─────────|──────────  │
│  ml.g4dn (T4)     | Inference (Light), BERT, Tabular  | $0.73   | 16GB       │
│  ml.g5 (A10G)     | Inference (LLM 8B-70B), Fine-tune | $1.51   | 24GB       │
│  ml.p3 (V100)     | Training (Legacy CNN/RNN)         | $3.82   | 16GB       │
│  ml.p4d (A100)    | LLM Pre-training/Fine-tuning      | $32.77  | 320GB (8x) │
│  ml.p5 (H100)     | Trillion-param LLM Training       | $98.32  | 640GB (8x) │
│  ml.inf2          | LLM Inference (Custom Silicon)    | $0.75   | 32GB       │
│  ml.trn1          | LLM Training (Custom Silicon)     | $21.50  | 512GB      │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Staff Engineer Secret**: Default quota for `ml.p4d` and `ml.p5` in a new AWS account is **ZERO**. You must work with an AWS Technical Account Manager (TAM) months in advance to secure capacity.

---

## 12.2 GPU Capacity Planner Dashboard

```python
# code/gpu/capacity_planner.py
"""
Production GPU capacity and utilization dashboard.
Used by Staff MLOps Engineers to track expensive resources.
"""

import boto3
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class GPUCapacityManager:
    """Manages GPU capacity, usage, and limits across regions."""
    
    # Standard GPU instances we track
    GPU_INSTANCES = [
        'ml.g4dn.xlarge', 'ml.g4dn.12xlarge', 
        'ml.g5.2xlarge', 'ml.g5.12xlarge', 'ml.g5.48xlarge',
        'ml.p3.2xlarge', 'ml.p3dn.24xlarge',
        'ml.p4d.24xlarge'
    ]
    
    def __init__(self, regions: list = ['us-east-1', 'us-west-2']):
        self.regions = regions
        
    def get_gpu_quotas(self) -> pd.DataFrame:
        """Fetch current SageMaker service quotas for GPUs."""
        quotas = []
        
        for region in self.regions:
            sq = boto3.client('service-quotas', region_name=region)
            
            # Service code for SageMaker is 'sagemaker'
            try:
                response = sq.list_service_quotas(ServiceCode='sagemaker')
                
                for q in response['Quotas']:
                    name = q['QuotaName']
                    if 'ml.g' in name or 'ml.p' in name:
                        quotas.append({
                            'Region': region,
                            'Instance Family': name,
                            'Limit': q['Value'],
                            'Usage': q.get('UsageMetric', {}).get('MetricStatisticRecommendation', 'Unknown')
                        })
            except Exception as e:
                logger.error(f"Failed to get quotas for {region}: {e}")
                
        return pd.DataFrame(quotas)
    
    def check_gpu_utilization(self, hours: int = 24) -> dict:
        """
        Check CloudWatch to see if we are actually USING the GPUs we pay for.
        Low utilization = wasted money.
        """
        results = {}
        for region in self.regions:
            cw = boto3.client('cloudwatch', region_name=region)
            sm = boto3.client('sagemaker', region_name=region)
            
            # Find all endpoints using GPUs
            endpoints = sm.list_endpoints(StatusEquals='InService')['Endpoints']
            gpu_endpoints = []
            
            for ep in endpoints:
                config = sm.describe_endpoint(EndpointName=ep['EndpointName'])
                for variant in config.get('ProductionVariants', []):
                    if 'ml.g' in variant['InstanceType'] or 'ml.p' in variant['InstanceType']:
                        gpu_endpoints.append(ep['EndpointName'])
                        
            # Get utilization metrics
            for ep_name in gpu_endpoints:
                response = cw.get_metric_statistics(
                    Namespace='/aws/sagemaker/Endpoints',
                    MetricName='GPUUtilization',
                    Dimensions=[{'Name': 'EndpointName', 'Value': ep_name}],
                    StartTime=datetime.utcnow() - timedelta(hours=hours),
                    EndTime=datetime.utcnow(),
                    Period=3600,
                    Statistics=['Average', 'Maximum']
                )
                
                if response['Datapoints']:
                    avg_util = sum(d['Average'] for d in response['Datapoints']) / len(response['Datapoints'])
                    max_util = max(d['Maximum'] for d in response['Datapoints'])
                    results[ep_name] = {'Region': region, 'Avg Util %': avg_util, 'Max Util %': max_util}
                    
        return results
    
    def recommend_optimizations(self):
        """Analyze usage and recommend cost optimizations."""
        utilization = self.check_gpu_utilization(hours=72)
        
        recommendations = []
        for ep, stats in utilization.items():
            if stats['Max Util %'] < 30:
                recommendations.append(
                    f"⚠️ Endpoint '{ep}' peak GPU util is {stats['Max Util %']:.1f}%. "
                    f"Action: Scale down instance size or enable auto-scaling."
                )
            elif stats['Avg Util %'] < 10 and stats['Max Util %'] > 80:
                recommendations.append(
                    f"💡 Endpoint '{ep}' has bursty traffic (Avg <10%, Max >80%). "
                    f"Action: Consider Async Endpoint or Serverless to reduce idle costs."
                )
                
        if not recommendations:
            return ["✅ All GPU endpoints are well-utilized."]
        return recommendations
```

---

## 12.3 Distributed Training GPU Strategy (DeepSpeed)

```python
# code/gpu/deepspeed_config.py
"""
Optimizing multi-GPU training for Large Language Models using DeepSpeed on SageMaker.
"""

# deepspeed_config.json
deepspeed_config = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    # ZeRO Stage 3 is required for 70B+ models on A10G/A100
    # Partitions optimizer states, gradients, AND model parameters
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",     # Offload to CPU memory when GPU is full
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}
```

---

*Next: [Section 13 — CI/CD for MLOps →](13-cicd-mlops.md)*
