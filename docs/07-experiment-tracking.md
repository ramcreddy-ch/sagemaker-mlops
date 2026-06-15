# Section 07 — Experiment Tracking & MLflow Integration

---

## 7.1 SageMaker Experiments — Production Usage

```python
# code/experiment-tracking/sagemaker_experiments.py
"""
Production experiment tracking with SageMaker Experiments.
Tracks every training run, enables model comparison and lineage.
"""

import boto3
import sagemaker
from sagemaker.experiments.run import Run
from sagemaker.utils import unique_name_from_base
import json, logging
from datetime import datetime

logger = logging.getLogger(__name__)
session = sagemaker.Session()

# ──────────────────────────────────────────────────────────
# EXPERIMENT STRUCTURE:
#   Experiment → Runs (each training job)
#   Experiments group logically related runs.
# ──────────────────────────────────────────────────────────

EXPERIMENT_NAME = "fraud-detection-xgboost-2024"

def run_tracked_experiment(hyperparams: dict, data_version: str):
    """
    Run a training experiment with full tracking.
    Everything logged here is searchable and comparable in Studio.
    """
    run_name = unique_name_from_base(f"fraud-xgb-{data_version}")
    
    with Run(
        experiment_name=EXPERIMENT_NAME,
        run_name=run_name,
        sagemaker_session=session,
    ) as run:
        
        # Log hyperparameters
        for k, v in hyperparams.items():
            run.log_parameter(k, v)
        
        # Log dataset info
        run.log_parameter("data_version", data_version)
        run.log_parameter("training_date", datetime.utcnow().date().isoformat())
        
        # Log file artifact (data schema, config)
        run.log_file(
            file_path="configs/training_config.json",
            name="training-config",
            is_output=False,
        )
        
        # ── Run training ──
        # (in real usage, you'd call your training function here)
        # Simulating training results:
        
        # Log metrics (streaming — appears live in Studio)
        for epoch in range(1, 11):
            run.log_metric("train:auc", 0.85 + epoch * 0.01, step=epoch)
            run.log_metric("validation:auc", 0.84 + epoch * 0.009, step=epoch)
            run.log_metric("train:loss", 0.5 - epoch * 0.04, step=epoch)
        
        # Final metrics
        final_auc = 0.97
        run.log_metric("test:auc_roc", final_auc)
        run.log_metric("test:avg_precision", 0.82)
        run.log_metric("test:f1_score", 0.79)
        run.log_metric("test:false_negative_rate", 0.03)
        run.log_metric("business:estimated_monthly_loss_prevented", 2_400_000.0)
        
        # Log confusion matrix artifact
        run.log_confusion_matrix(
            actual=[0]*9900 + [1]*100,
            predicted=[0]*9870 + [1]*30 + [0]*3 + [1]*97,
            title="Fraud Detection Confusion Matrix",
        )
        
        # Tag the run
        run.log_parameter("champion_challenger", "challenger")
        run.log_parameter("experiment_type", "hyperparameter_variation")
        
        logger.info(f"Experiment run '{run_name}' completed. AUC: {final_auc}")
        return run_name, final_auc


def compare_experiments(experiment_name: str, metric: str = "test:auc_roc"):
    """Compare all runs in an experiment."""
    from sagemaker.experiments import Experiment
    
    experiment = Experiment.load(experiment_name=experiment_name)
    
    runs_df = experiment.list_runs(
        sort_by="CreationTime",
        sort_order="Descending",
        max_results=50,
    )
    
    print(f"\nExperiment: {experiment_name}")
    print(f"Total runs: {len(runs_df)}")
    print(f"\nTop 5 runs by {metric}:")
    
    # Sort by objective metric
    if not runs_df.empty:
        top_runs = runs_df.nlargest(5, metric) if metric in runs_df.columns else runs_df.head(5)
        print(top_runs[['run_name', metric, 'max_depth', 'eta', 'subsample']].to_string())
    
    return runs_df
```

---

## 7.2 MLflow on SageMaker — Production Integration

```python
# code/experiment-tracking/mlflow_sagemaker.py
"""
MLflow integrated with SageMaker for experiment tracking.
MLflow is the industry standard — many teams use it alongside
or instead of SageMaker Experiments.
"""

import mlflow
import mlflow.xgboost
import mlflow.sklearn
import mlflow.pytorch
import boto3
import os
import logging
from pathlib import Path
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# MLflow Tracking Server Options on AWS:
#   Option 1: MLflow on ECS/EC2 with S3 artifact store + RDS backend
#   Option 2: SageMaker Managed MLflow (2024 feature)
#   Option 3: Databricks MLflow (if using Databricks + SageMaker)
# ──────────────────────────────────────────────────────────────────────

# Option 2: SageMaker Managed MLflow (recommended for AWS-native teams)
MLFLOW_TRACKING_URI = "arn:aws:sagemaker:us-east-1:123456789:mlflow-tracking-server/fintech-mlflow-prod"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class MLflowFraudExperiment:
    """
    Production MLflow experiment management for fraud detection.
    """
    
    def __init__(self, experiment_name: str = "fraud-detection-production"):
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
    
    def log_training_run(
        self,
        model: xgb.Booster,
        params: dict,
        train_metrics: dict,
        test_metrics: dict,
        feature_names: list,
        dataset_info: dict,
        tags: dict = None,
    ) -> str:
        """
        Log a complete training run to MLflow.
        Returns the run ID for model registration.
        """
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Tags for filtering/querying
            default_tags = {
                "model_type": "xgboost",
                "use_case": "fraud_detection",
                "environment": "production",
                "data_version": dataset_info.get("version"),
                "git_commit": os.environ.get("GIT_COMMIT_SHA", "unknown"),
                "triggered_by": os.environ.get("TRIGGERED_BY", "manual"),
            }
            if tags:
                default_tags.update(tags)
            
            mlflow.set_tags(default_tags)
            
            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("training_samples", dataset_info["train_size"])
            mlflow.log_param("feature_count", len(feature_names))
            mlflow.log_param("positive_rate", dataset_info["positive_rate"])
            mlflow.log_param("best_iteration", model.best_iteration)
            
            # Log training metrics
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v)
            
            # Log test metrics
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            
            # Log the model with feature schema
            signature = mlflow.models.infer_signature(
                model_input=pd.DataFrame(
                    np.zeros((1, len(feature_names))),
                    columns=feature_names
                ),
                model_output=np.array([0.5])
            )
            
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name="fraud-detection-xgboost",
                input_example=pd.DataFrame(
                    np.zeros((3, len(feature_names))),
                    columns=feature_names
                ),
            )
            
            # Log feature importance
            importance = model.get_score(importance_type='gain')
            importance_df = pd.DataFrame(
                list(importance.items()),
                columns=['feature', 'importance_gain']
            ).sort_values('importance_gain', ascending=False)
            
            importance_df.to_csv("/tmp/feature_importance.csv", index=False)
            mlflow.log_artifact("/tmp/feature_importance.csv", "analysis")
            
            # Log training config
            mlflow.log_dict(params, "configs/training_params.json")
            mlflow.log_dict(dataset_info, "configs/dataset_info.json")
            
            logger.info(f"MLflow run logged: {run_id}")
            logger.info(f"Test AUC-ROC: {test_metrics.get('auc_roc', 'N/A')}")
            
            return run_id
    
    def promote_to_production(self, run_id: str, model_name: str = "fraud-detection-xgboost"):
        """
        Promote a model run to production stage in MLflow Model Registry.
        Equivalent to 'Approve' in SageMaker Model Registry.
        """
        client = mlflow.tracking.MlflowClient()
        
        # Get all versions for this run
        versions = client.search_model_versions(f"run_id='{run_id}'")
        
        for version in versions:
            # Archive current production model
            current_prod = client.get_latest_versions(
                model_name, stages=["Production"]
            )
            for prod_model in current_prod:
                client.transition_model_version_stage(
                    name=model_name,
                    version=prod_model.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
                logger.info(f"Archived version {prod_model.version}")
            
            # Promote new version
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Production",
            )
            
            # Add production annotation
            client.update_model_version(
                name=model_name,
                version=version.version,
                description=f"Promoted to production on {pd.Timestamp.now().date()}. "
                           f"AUC-ROC > 0.97 threshold met.",
            )
            
            logger.info(f"✅ Model {model_name} v{version.version} promoted to Production")
            
            return version.version
    
    def compare_with_champion(
        self,
        challenger_run_id: str,
        metric: str = "test_auc_roc",
        model_name: str = "fraud-detection-xgboost",
    ) -> dict:
        """
        Statistical comparison of challenger vs. champion model.
        """
        client = mlflow.tracking.MlflowClient()
        
        # Get champion run
        champion_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not champion_versions:
            return {"recommendation": "PROMOTE", "reason": "No champion exists"}
        
        champion_run_id = champion_versions[0].run_id
        champion_run = client.get_run(champion_run_id)
        challenger_run = client.get_run(challenger_run_id)
        
        champion_metric = float(champion_run.data.metrics.get(metric, 0))
        challenger_metric = float(challenger_run.data.metrics.get(metric, 0))
        
        improvement = challenger_metric - champion_metric
        improvement_pct = (improvement / champion_metric) * 100
        
        result = {
            "champion_version": champion_versions[0].version,
            "champion_run_id": champion_run_id,
            "champion_metric": champion_metric,
            "challenger_run_id": challenger_run_id,
            "challenger_metric": challenger_metric,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "recommendation": "PROMOTE" if improvement > 0.005 else "KEEP_CHAMPION",
            "reason": f"Improvement of {improvement_pct:.2f}% {'exceeds' if improvement > 0.005 else 'below'} 0.5% threshold",
        }
        
        logger.info(f"Champion vs Challenger comparison:")
        logger.info(f"  Champion {metric}: {champion_metric:.6f}")
        logger.info(f"  Challenger {metric}: {challenger_metric:.6f}")
        logger.info(f"  Recommendation: {result['recommendation']}")
        
        return result
```

---

## 7.3 Model Lineage Tracking

```python
# Automatic lineage tracking in SageMaker
# Shows: data → processing → training → model → endpoint

import boto3
from sagemaker.lineage.context import Context
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association
from sagemaker.lineage.query import LineageFilter, LineageSourceType

sm = boto3.client('sagemaker')

def get_model_lineage(model_package_arn: str):
    """
    Trace complete lineage: endpoint → model → training → data.
    Critical for regulatory audit trails (SOX, Basel III).
    """
    
    from sagemaker.lineage.query import LineageEntityEnum, LineageQueryDirectionEnum
    
    # Start from model package and trace upstream
    response = sm.query_lineage(
        StartArns=[model_package_arn],
        Direction='Ascendants',   # look backwards (data, training jobs)
        IncludeEdges=True,
        Filters={
            'Types': ['DataSet', 'TrainingJob', 'ProcessingJob'],
        },
        MaxDepth=10,
    )
    
    lineage = {
        'model_package': model_package_arn,
        'vertices': [],
        'edges': [],
    }
    
    for vertex in response.get('Vertices', []):
        lineage['vertices'].append({
            'arn': vertex['Arn'],
            'type': vertex['Type'],
            'lineage_type': vertex['LineageType'],
        })
    
    print("Model Lineage:")
    print(f"  Model Package: {model_package_arn}")
    for v in lineage['vertices']:
        print(f"  ↑ [{v['type']}] {v['arn'][-50:]}")
    
    return lineage

# Usage
model_arn = "arn:aws:sagemaker:us-east-1:123456789:model-package/fraud-detection-xgboost/47"
lineage = get_model_lineage(model_arn)
```

---

*Next: [Section 08 — Model Registry →](08-model-registry.md)*
