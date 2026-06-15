# Section 19 — Cost Optimization (FinOps for ML)

> **Production Context**: "We trained a great model, but the AWS bill just jumped by $50,000 this month." FinOps is a core Staff MLOps responsibility.

---

## 19.1 ML Cost Optimization Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINOPS STRATEGIES FOR SAGEMAKER                          │
│                                                                             │
│  PHASE       | STRATEGY                          | POTENTIAL SAVINGS        │
│  ────────────|───────────────────────────────────|──────────────────────────│
│  Notebooks   | Auto-shutdown idle Studio apps    | 30% - 60%                │
│              | Use Local Mode for prototyping    | 90% (Zero EC2 cost)      │
│  ────────────|───────────────────────────────────|──────────────────────────│
│  Training    | Spot Instances                    | Up to 90%                │
│              | Right-size instances              | 20% - 40%                │
│              | Distributed training (faster)     | Indirect (engineer time) │
│  ────────────|───────────────────────────────────|──────────────────────────│
│  Inference   | Multi-Model Endpoints (MME)       | 80% (consolidation)      │
│              | Serverless Inference              | >90% (for low traffic)   │
│              | Auto-scaling (Scale to zero?)     | 40% - 60%                │
│              | Inferentia (Inf2) instances       | 30% - 40% vs GPU         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 19.2 Managed Spot Training

```python
# code/finops/spot_training.py
"""
Use Managed Spot Instances for training.
SageMaker bids for spare EC2 capacity, saving up to 90%.
If the instance is reclaimed, SageMaker automatically resumes from the last checkpoint.
"""

from sagemaker.xgboost.estimator import XGBoost

def run_cost_optimized_training(role, input_data_s3, output_s3):
    
    # Enable checkpointing - MANDATORY for Spot training
    # If the instance is interrupted, it needs to know where to resume.
    checkpoint_s3_uri = f"{output_s3}/checkpoints/"
    
    xgb_estimator = XGBoost(
        entry_point="train.py",
        role=role,
        instance_count=1,
        instance_type='ml.m5.4xlarge',
        framework_version="1.7-1",
        output_path=output_s3,
        
        # --- SPOT INSTANCE CONFIGURATION ---
        use_spot_instances=True,
        max_run=3600,          # Max time the job is allowed to run (1 hr)
        max_wait=7200,         # Max time to wait for Spot capacity (2 hrs)
        checkpoint_s3_uri=checkpoint_s3_uri,
        # -----------------------------------
    )
    
    xgb_estimator.fit({'train': input_data_s3})
    
    # After training, SageMaker prints the actual savings:
    # "Training seconds: 1200"
    # "Billable seconds: 360"
    # "Managed Spot Training savings: 70.0%"
```

---

## 19.3 Right-Sizing Inference Compute (Inferentia)

For transformer models, AWS Inferentia (`ml.inf2`) provides the lowest cost-per-inference.

```python
# code/finops/inferentia_deployment.py
"""
Deploying a Hugging Face model to AWS Inferentia2 instead of GPUs.
Requires compiling the model with AWS Neuron, but cuts inference costs significantly.
"""

import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def deploy_to_inferentia(role, model_data_s3):
    # Use the Neuron container optimized for Inferentia
    image_uri = sagemaker.image_uris.retrieve(
        framework='huggingface',
        region='us-east-1',
        version='4.28',
        image_scope='inference',
        base_framework_version='pytorch1.13',
        instance_type='ml.inf2.xlarge' # Inferentia instance
    )
    
    huggingface_model = HuggingFaceModel(
        model_data=model_data_s3,
        role=role,
        image_uri=image_uri,
        env={
            'HF_MODEL_ID': 'distilbert-base-uncased-finetuned-sst-2-english',
            'HF_TASK': 'text-classification',
            # Required for Neuron compilation
            'NEURON_CORE_GUEST_IDS': '0' 
        }
    )
    
    # ml.inf2.xlarge costs ~$0.76/hr compared to ml.g5.xlarge at ~$1.01/hr,
    # but provides significantly higher throughput for compiled models.
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type='ml.inf2.xlarge'
    )
```

---

*Next: [Section 20 — Compliance & Audit →](20-compliance-audit.md)*
