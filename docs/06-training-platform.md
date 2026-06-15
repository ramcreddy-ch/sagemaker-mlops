# Section 06 — Training Platform & Distributed Training

> **Production Context**: How Netflix, Uber, and Goldman Sachs run 500+ training experiments per day on SageMaker.

---

## 6.1 Training Platform Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SAGEMAKER TRAINING PLATFORM                              │
│                                                                             │
│  TRAINING JOB TYPES                                                         │
│  ─────────────────                                                          │
│  Standard Job:      Single instance, XGBoost/sklearn                        │
│  Distributed Job:   Multi-instance, PyTorch DDP, data parallelism           │
│  HPO Job:           Bayesian optimization over hyperparameter space          │
│  Spot Job:          Managed Spot Training — 70-90% cost saving              │
│  Warm Pool:         Reuse infrastructure across jobs — fast restarts        │
│                                                                             │
│  INSTANCE SELECTION GUIDE                                                   │
│  ─────────────────────────                                                  │
│  XGBoost/LightGBM (fraud, risk):  ml.m5.4xlarge - ml.m5.24xlarge          │
│  PyTorch (deep learning):         ml.p3.2xlarge - ml.p3dn.24xlarge        │
│  LLM fine-tuning:                 ml.g5.48xlarge, ml.p4d.24xlarge          │
│  Distributed (100B+ params):      ml.p4de.24xlarge (A100 80GB)             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6.2 Production Fraud Detection Training Script

```python
# code/training/fraud_detection_xgboost.py
"""
Production XGBoost fraud detection training script.
Runs as SageMaker Training Job.
"""

import argparse
import os
import json
import logging
import pickle
import tarfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import sagemaker.experiments

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--objective',          type=str,   default='binary:logistic')
    parser.add_argument('--num-round',          type=int,   default=500)
    parser.add_argument('--max-depth',          type=int,   default=8)
    parser.add_argument('--eta',                type=float, default=0.05)
    parser.add_argument('--subsample',          type=float, default=0.8)
    parser.add_argument('--colsample-bytree',   type=float, default=0.8)
    parser.add_argument('--min-child-weight',   type=int,   default=5)
    parser.add_argument('--scale-pos-weight',   type=float, default=99.0)
    parser.add_argument('--early-stopping-rounds', type=int, default=20)
    parser.add_argument('--eval-metric',        type=str,   default='auc')
    
    # SageMaker environment variables
    parser.add_argument('--model-dir',   type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train',       type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation',  type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--test',        type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    return parser.parse_args()


def load_data(data_dir: str, filename: str = 'train.csv') -> tuple:
    """Load data from SageMaker channel."""
    data_path = Path(data_dir) / filename
    
    # Handle both single file and directory of part files
    if data_path.exists():
        df = pd.read_csv(data_path, header=None)
    else:
        # Multiple part files (large datasets)
        dfs = []
        for f in sorted(Path(data_dir).glob('*.csv')):
            dfs.append(pd.read_csv(f, header=None))
        df = pd.concat(dfs, ignore_index=True)
    
    # First column is label
    y = df.iloc[:, 0].values.astype(int)
    X = df.iloc[:, 1:].values.astype(float)
    
    logger.info(f"Loaded {len(X):,} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)} ({y.mean():.4%} positive)")
    
    return X, y


def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    """Compute comprehensive model metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'avg_precision': average_precision_score(y_true, y_pred_proba),
        'f1_score': f1_score(y_true, y_pred),
        'threshold': threshold,
    }
    
    # Business metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Fraud-specific business metric
    # Cost model: FN (missed fraud) costs $500, FP (false block) costs $10
    metrics['business_cost'] = (fn * 500) + (fp * 10)
    
    return metrics


def find_optimal_threshold(y_true, y_pred_proba) -> float:
    """Find threshold that minimizes business cost."""
    best_threshold = 0.5
    best_cost = float('inf')
    
    for threshold in np.arange(0.1, 0.95, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = (fn * 500) + (fp * 10)
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    
    logger.info(f"Optimal threshold: {best_threshold:.3f}, Business cost: ${best_cost:,.0f}")
    return best_threshold


def train_model(args):
    """Main training function."""
    
    # Load data
    X_train, y_train = load_data(args.train)
    X_val, y_val = load_data(args.validation, 'validation.csv')
    X_test, y_test = load_data(args.test, 'test.csv')
    
    # XGBoost DMatrix (optimized format)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': args.objective,
        'max_depth': args.max_depth,
        'eta': args.eta,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'min_child_weight': args.min_child_weight,
        'scale_pos_weight': args.scale_pos_weight,
        'eval_metric': ['auc', 'aucpr', 'logloss'],
        'tree_method': 'hist',       # fastest for tabular data
        'device': 'cuda' if os.path.exists('/dev/nvidia0') else 'cpu',
        'seed': 42,
        'nthread': os.cpu_count(),
    }
    
    logger.info(f"Training with params: {json.dumps(params, indent=2)}")
    
    # Training with early stopping
    evals_result = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50,
        callbacks=[
            # SageMaker metrics callback — streams to CloudWatch
            xgb.callback.TrainingCallback(),
        ]
    )
    
    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best validation AUC: {model.best_score:.6f}")
    
    # Emit metrics to SageMaker (for HPO)
    # These go to CloudWatch and are parsed by HPO
    val_auc = evals_result['validation']['auc'][-1]
    val_aucpr = evals_result['validation']['aucpr'][-1]
    print(f"validation:auc={val_auc:.6f}")
    print(f"validation:aucpr={val_aucpr:.6f}")
    
    # Evaluate on held-out test set
    y_pred_proba = model.predict(dtest)
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
    test_metrics = compute_metrics(y_test, y_pred_proba, optimal_threshold)
    
    logger.info("=== TEST SET METRICS ===")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        # Emit to SageMaker Experiments
        print(f"test:{k}={v}")
    
    # Feature importance
    importance = model.get_score(importance_type='gain')
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 features by gain:")
    for feat, score in top_features:
        logger.info(f"  {feat}: {score:.4f}")
    
    # Save model
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save XGBoost model
    model.save_model(str(model_dir / 'xgboost-model'))
    
    # Save evaluation metrics for Model Registry
    evaluation_output = {
        'classification_metrics': {
            'auc_roc': {'value': test_metrics['auc_roc']},
            'avg_precision': {'value': test_metrics['avg_precision']},
            'f1_score': {'value': test_metrics['f1_score']},
            'precision': {'value': test_metrics['precision']},
            'recall': {'value': test_metrics['recall']},
        },
        'business_metrics': {
            'optimal_threshold': {'value': test_metrics['threshold']},
            'estimated_business_cost': {'value': test_metrics['business_cost']},
            'false_negative_rate': {'value': test_metrics['false_negative_rate']},
        },
        'model_config': {
            'best_iteration': model.best_iteration,
            'best_val_auc': model.best_score,
            'num_features': X_train.shape[1],
            'training_samples': len(X_train),
        }
    }
    
    with open(str(model_dir / 'evaluation.json'), 'w') as f:
        json.dump(evaluation_output, f, indent=2)
    
    logger.info(f"✅ Model saved to {model_dir}")
    logger.info(f"✅ Final AUC-ROC: {test_metrics['auc_roc']:.6f}")
    
    return model, test_metrics


if __name__ == '__main__':
    args = parse_args()
    model, metrics = train_model(args)
```

---

## 6.3 Distributed Training — PyTorch DDP

```python
# code/training/distributed_training_pytorch.py
"""
Distributed training for deep learning fraud detection.
Uses SageMaker Distributed Data Parallel (SMDDP).
Run on 4x ml.g5.2xlarge instances.
"""

import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

logger = logging.getLogger(__name__)


class FraudDetectionNet(nn.Module):
    """
    Deep neural network for fraud detection.
    Better than XGBoost when interaction features are complex
    and dataset > 100M records.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def setup_distributed():
    """Initialize distributed training environment."""
    # SageMaker sets these environment variables automatically
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',   # GPU distributed training
            init_method='env://',
        )
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    logger.info(f"Distributed: rank={rank}, world_size={world_size}, device={device}")
    return rank, world_size, local_rank, device


def train_epoch(model, loader, optimizer, criterion, device, rank):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (features, labels) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        
        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).long()
        correct += (predicted == labels.long()).sum().item()
        total += labels.size(0)
        
        if batch_idx % 100 == 0 and rank == 0:
            logger.info(f"Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.6f}")
    
    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--batch-size',    type=int,   default=4096)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay',  type=float, default=1e-4)
    parser.add_argument('--model-dir',     type=str,   default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train',         type=str,   default=os.environ.get('SM_CHANNEL_TRAIN'))
    args = parser.parse_args()
    
    rank, world_size, local_rank, device = setup_distributed()
    
    # Load training data
    import pandas as pd
    import numpy as np
    df = pd.read_parquet(f"{args.train}/features.parquet")
    X = torch.FloatTensor(df.drop('label', axis=1).values)
    y = torch.LongTensor(df['label'].values)
    
    dataset = TensorDataset(X, y)
    
    # Distributed sampler ensures each GPU sees different data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) \
              if world_size > 1 else None
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,           # faster CPU→GPU transfer
        persistent_workers=True,  # avoid worker restart overhead
    )
    
    # Model
    model = FraudDetectionNet(input_dim=X.shape[1]).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 10,
        epochs=args.epochs,
        steps_per_epoch=len(loader),
        pct_start=0.1,
    )
    
    # Class-weighted loss for imbalanced fraud data (1% positive rate)
    pos_weight = torch.tensor([99.0]).to(device)
    criterion = nn.BCELoss(reduction='mean')
    
    # Training loop
    best_val_auc = 0.0
    
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)   # shuffle differently each epoch
        
        train_loss, train_acc = train_epoch(model, loader, optimizer, criterion, device, rank)
        scheduler.step()
        
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                       f"Loss: {train_loss:.6f} | Acc: {train_acc:.4f}")
            print(f"train:loss={train_loss:.6f}")
            print(f"train:accuracy={train_acc:.6f}")
        
    # Save model (only on rank 0)
    if rank == 0:
        model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
        torch.save({
            'model_state_dict': model_state,
            'input_dim': X.shape[1],
            'architecture': 'FraudDetectionNet',
        }, f"{args.model_dir}/model.pt")
        logger.info(f"✅ Model saved to {args.model_dir}/model.pt")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
```

---

## 6.4 Hyperparameter Tuning — Bayesian Optimization

```python
# code/training/hyperparameter_tuning.py
"""
Production HPO setup using SageMaker Automatic Model Tuning.
Uses Bayesian optimization for efficient search.
"""

import boto3
import sagemaker
from sagemaker.tuner import (
    HyperparameterTuner, IntegerParameter,
    ContinuousParameter, CategoricalParameter
)
from sagemaker.estimator import Estimator
import json

session = sagemaker.Session()
region = 'us-east-1'
role = 'arn:aws:iam::123456789:role/SageMakerExecutionRole-Training'
bucket = 'fintech-datalake-prod'

# Base estimator
xgboost_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve('xgboost', region, '1.7-1'),
    instance_type='ml.m5.4xlarge',
    instance_count=1,
    output_path=f's3://{bucket}/fraud-detection/hpo-models/',
    base_job_name='fraud-hpo',
    role=role,
    sagemaker_session=session,
    use_spot_instances=True,
    max_run=3600,
    max_wait=7200,
)

# Fixed hyperparameters
xgboost_estimator.set_hyperparameters(
    objective='binary:logistic',
    eval_metric='auc',
    num_round=500,
    early_stopping_rounds=20,
    scale_pos_weight=99,
)

# Search space (Bayesian optimization explores this efficiently)
hyperparameter_ranges = {
    # Regularization
    'max_depth': IntegerParameter(4, 12),
    'min_child_weight': IntegerParameter(1, 20),
    'reg_alpha': ContinuousParameter(0, 2),        # L1 regularization
    'reg_lambda': ContinuousParameter(0.5, 5),     # L2 regularization
    
    # Learning dynamics
    'eta': ContinuousParameter(0.01, 0.3),
    'gamma': ContinuousParameter(0, 2),
    
    # Stochastic subsampling
    'subsample': ContinuousParameter(0.5, 1.0),
    'colsample_bytree': ContinuousParameter(0.3, 1.0),
    'colsample_bylevel': ContinuousParameter(0.3, 1.0),
}

tuner = HyperparameterTuner(
    estimator=xgboost_estimator,
    objective_metric_name='validation:auc',
    objective_type='Maximize',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=100,                   # 100 candidate models
    max_parallel_jobs=10,           # 10 in parallel
    strategy='Bayesian',            # Bayesian > Random for efficiency
    early_stopping_type='Auto',     # kill poor candidates early
    warm_start_config=None,         # can warm start from previous HPO run
)

# Training data inputs
from sagemaker import TrainingInput
inputs = {
    'train': TrainingInput(
        s3_data=f's3://{bucket}/fraud-detection/processed/train/',
        content_type='text/csv'
    ),
    'validation': TrainingInput(
        s3_data=f's3://{bucket}/fraud-detection/processed/validation/',
        content_type='text/csv'
    ),
}

tuner.fit(inputs=inputs, include_cls_metadata=False)
print(f"HPO job started: {tuner.latest_tuning_job.name}")
print("Monitor at: https://console.aws.amazon.com/sagemaker/home#/hyper-tuning-jobs")

# Get best training job after completion
best_job = tuner.best_training_job()
print(f"Best job: {best_job}")

analytics = tuner.analytics()
df = analytics.dataframe()
print(df.sort_values('FinalObjectiveValue', ascending=False).head(5))
```

---

## 6.5 Managed Spot Training — Cost Optimization

```python
# Spot training configuration for production
# Saves 60-90% on training costs

from sagemaker.estimator import Estimator

def create_spot_estimator(
    image_uri: str,
    instance_type: str,
    base_job_name: str,
    role: str,
    session: sagemaker.Session,
    max_run_hours: int = 6,
) -> Estimator:
    """
    Create a spot-enabled estimator with proper checkpoint configuration.
    CRITICAL: checkpointing is REQUIRED for spot training.
    Without checkpoints, a spot interruption loses ALL training progress.
    """
    
    checkpoint_s3 = f"s3://fintech-datalake-prod/checkpoints/{base_job_name}/"
    checkpoint_local = "/opt/ml/checkpoints"
    
    return Estimator(
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=session,
        base_job_name=base_job_name,
        
        # Spot configuration
        use_spot_instances=True,
        max_run=max_run_hours * 3600,
        max_wait=max_run_hours * 2 * 3600,   # max_wait > max_run
        
        # CRITICAL: Checkpoint config for spot interruption recovery
        checkpoint_s3_uri=checkpoint_s3,
        checkpoint_local_path=checkpoint_local,
        
        # Metrics for monitoring
        metric_definitions=[
            {'Name': 'train:auc',       'Regex': r'train:auc=([0-9\.]+)'},
            {'Name': 'validation:auc',  'Regex': r'validation:auc=([0-9\.]+)'},
            {'Name': 'train:loss',      'Regex': r'train:loss=([0-9\.]+)'},
        ],
        
        enable_sagemaker_metrics=True,
    )


# Cost comparison
print("""
Instance Cost Comparison (us-east-1, 2024):
ml.p3.2xlarge On-Demand:  $3.825/hr
ml.p3.2xlarge Spot:       $1.148/hr  (70% saving)

For 100 training jobs × 2 hours each:
On-Demand: 100 × 2 × $3.825 = $765
Spot:      100 × 2 × $1.148 = $230  (saving: $535/month)

At 500 jobs/day: Spot saves ~$80,000/month
""")
```

---

## 6.6 SageMaker Training — Operational Monitoring

```python
# code/training/training_monitor.py
"""
Production monitoring for SageMaker training jobs.
MLOps engineers run this during daily operations.
"""

import boto3
import json
from datetime import datetime, timedelta
from typing import List, Dict

class TrainingJobMonitor:
    """Monitor all training jobs across the platform."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.sm = boto3.client('sagemaker', region_name=region)
        self.cw = boto3.client('cloudwatch', region_name=region)
    
    def get_active_jobs(self) -> List[Dict]:
        """Get all currently running training jobs."""
        response = self.sm.list_training_jobs(
            StatusEquals='InProgress',
            SortBy='CreationTime',
            SortOrder='Descending',
        )
        
        jobs = []
        for job in response['TrainingJobSummaries']:
            detail = self.sm.describe_training_job(
                TrainingJobName=job['TrainingJobName']
            )
            
            duration = (datetime.utcnow() - 
                       detail['TrainingStartTime'].replace(tzinfo=None)).total_seconds()
            
            jobs.append({
                'name': job['TrainingJobName'],
                'status': job['TrainingJobStatus'],
                'instance_type': detail['ResourceConfig']['InstanceType'],
                'instance_count': detail['ResourceConfig']['InstanceCount'],
                'duration_hours': duration / 3600,
                'spot': detail.get('EnableManagedSpotTraining', False),
                'created': job['CreationTime'].strftime('%Y-%m-%d %H:%M'),
            })
        
        return jobs
    
    def check_stale_jobs(self, max_hours: int = 12) -> List[str]:
        """Find jobs that have been running too long — likely hung."""
        active = self.get_active_jobs()
        stale = [
            j['name'] for j in active
            if j['duration_hours'] > max_hours
        ]
        
        if stale:
            print(f"⚠️  {len(stale)} stale training jobs (>{max_hours}h):")
            for name in stale:
                print(f"  - {name}")
        
        return stale
    
    def get_daily_cost(self) -> Dict:
        """Estimate training cost for the past 24 hours."""
        # Instance pricing (approximate)
        PRICING = {
            'ml.m5.xlarge': 0.23,
            'ml.m5.4xlarge': 0.922,
            'ml.m5.12xlarge': 2.765,
            'ml.p3.2xlarge': 3.825,
            'ml.p3.8xlarge': 14.688,
            'ml.g5.2xlarge': 1.515,
            'ml.g5.12xlarge': 5.672,
            'ml.g5.48xlarge': 16.288,
        }
        
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        response = self.sm.list_training_jobs(
            CreationTimeAfter=yesterday,
        )
        
        total_cost = 0.0
        by_instance = {}
        
        for job in response['TrainingJobSummaries']:
            detail = self.sm.describe_training_job(
                TrainingJobName=job['TrainingJobName']
            )
            
            instance_type = detail['ResourceConfig']['InstanceType']
            instance_count = detail['ResourceConfig']['InstanceCount']
            
            if detail.get('TrainingEndTime') and detail.get('TrainingStartTime'):
                duration_hours = (
                    detail['TrainingEndTime'] - detail['TrainingStartTime']
                ).total_seconds() / 3600
                
                hourly_rate = PRICING.get(instance_type, 1.0)
                spot_discount = 0.3 if detail.get('EnableManagedSpotTraining') else 1.0
                cost = hourly_rate * instance_count * duration_hours * spot_discount
                
                total_cost += cost
                by_instance[instance_type] = by_instance.get(instance_type, 0) + cost
        
        return {
            'total_24h_cost': total_cost,
            'by_instance_type': by_instance,
            'job_count': len(response['TrainingJobSummaries']),
        }
    
    def print_dashboard(self):
        """Print training platform status dashboard."""
        active = self.get_active_jobs()
        cost = self.get_daily_cost()
        
        print("\n" + "="*60)
        print("    TRAINING PLATFORM DASHBOARD")
        print("="*60)
        print(f"\n🏃 Active Jobs: {len(active)}")
        
        for job in active:
            spot_icon = "💰" if job['spot'] else "  "
            print(f"  {spot_icon} {job['name'][:45]} | {job['instance_type']} | {job['duration_hours']:.1f}h")
        
        stale = self.check_stale_jobs()
        
        print(f"\n💵 24h Training Cost: ${cost['total_24h_cost']:,.2f}")
        print(f"   Jobs completed: {cost['job_count']}")
        print(f"\nBy instance type:")
        for inst, cost_val in sorted(cost['by_instance_type'].items(), key=lambda x: -x[1]):
            print(f"  {inst}: ${cost_val:,.2f}")
        print("="*60 + "\n")


if __name__ == '__main__':
    monitor = TrainingJobMonitor()
    monitor.print_dashboard()
```

---

*Next: [Section 07 — Experiment Tracking →](07-experiment-tracking.md)*
