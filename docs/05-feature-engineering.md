# Section 05 — Feature Engineering & SageMaker Feature Store

> **Production Context**: How Uber's Michelangelo, Capital One, and JPMorgan structure feature stores for real-time fraud detection.

---

## 5.1 Feature Store Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SAGEMAKER FEATURE STORE ARCHITECTURE                     │
│                                                                             │
│  WRITE PATH (Feature Ingestion)                                             │
│  ─────────────────────────────                                              │
│  Streaming:  Kafka → Lambda → PutRecord() → Online Store (DynamoDB)         │
│                                          → Offline Store (S3 + Glue)       │
│                                                                             │
│  Batch:      EMR Spark → BatchIngestion → Offline Store only                │
│                                                                             │
│  Processing: SageMaker Processing → Ingest → Both stores                   │
│                                                                             │
│  READ PATH (Feature Retrieval)                                              │
│  ─────────────────────────────                                              │
│  Real-time inference: GetRecord() → Online Store → <5ms                    │
│  Training:            Athena query → Offline Store → S3 → Training Job      │
│  Batch inference:     Athena → Offline Store → Batch Transform              │
│                                                                             │
│  FEATURE GROUPS (our Financial Platform):                                   │
│  ├── fraud-features-v3          (transaction-level, real-time)              │
│  ├── customer-profile-v2        (customer-level, daily refresh)             │
│  ├── merchant-risk-v1           (merchant-level, weekly refresh)            │
│  ├── device-fingerprint-v1      (device-level, per-event)                  │
│  └── market-regime-v1           (market-level, hourly)                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.2 Creating Feature Groups — Production Pattern

```python
# code/feature-store/feature_group_fraud.py
"""
Production Feature Group setup for Fraud Detection.
This code is typically run once during platform setup,
then managed via IaC (Terraform).
"""

import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition, FeatureTypeEnum
)
import pandas as pd
import time

session = sagemaker.Session()
region = "us-east-1"
account_id = boto3.client('sts').get_caller_identity()['Account']
role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole-FeatureStore"
bucket = "fintech-datalake-prod"

# ─────────────────────────────────────────────────────────────────
# FRAUD TRANSACTION FEATURES (Real-time, Online + Offline)
# ─────────────────────────────────────────────────────────────────
fraud_feature_definitions = [
    FeatureDefinition(feature_name="transaction_id",     feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="customer_id",        feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="event_time",         feature_type=FeatureTypeEnum.STRING),
    # Transaction features
    FeatureDefinition(feature_name="amount",             feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="amount_log",         feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="merchant_category",  feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="country_code",       feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="is_international",   feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="hour_of_day",        feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="day_of_week",        feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="is_weekend",         feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="is_night",           feature_type=FeatureTypeEnum.INTEGRAL),
    # Aggregated features (rolling windows)
    FeatureDefinition(feature_name="txn_count_1h",       feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="txn_count_24h",      feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="txn_count_7d",       feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="spend_1h",           feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="spend_24h",          feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="spend_7d",           feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="avg_txn_amount_30d", feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="std_txn_amount_30d", feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="unique_merchants_7d", feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="unique_countries_30d", feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="velocity_ratio",     feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="amount_vs_avg_ratio", feature_type=FeatureTypeEnum.FRACTIONAL),
    # Device features
    FeatureDefinition(feature_name="device_fingerprint", feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="device_seen_before", feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="device_age_days",    feature_type=FeatureTypeEnum.INTEGRAL),
]

fraud_feature_group = FeatureGroup(
    name="fraud-features-v3",
    sagemaker_session=session,
    feature_definitions=fraud_feature_definitions,
)

fraud_feature_group.create(
    s3_uri=f"s3://{bucket}/feature-store/fraud-features-v3",
    record_identifier_name="transaction_id",
    event_time_feature_name="event_time",
    role_arn=role,
    enable_online_store=True,
    online_store_config={
        "EnableOnlineStore": True,
        "SecurityConfig": {
            "KmsKeyId": f"arn:aws:kms:{region}:{account_id}:key/mrk-featurestore-prod"
        },
        "TtlDuration": {
            "Unit": "Days",
            "Value": 30,  # online records expire after 30 days
        }
    },
    offline_store_config={
        "S3StorageConfig": {
            "S3Uri": f"s3://{bucket}/feature-store/fraud-features-v3",
            "KmsKeyId": f"arn:aws:kms:{region}:{account_id}:key/mrk-featurestore-prod"
        },
        "DisableGlueTableCreation": False,  # auto-create Glue table
        "DataCatalogConfig": {
            "TableName": "fraud_features_v3",
            "Catalog": "AwsDataCatalog",
            "Database": "fintech_feature_store",
        }
    },
    description="Real-time fraud detection features - v3 - 25 features",
    tags=[
        {"Key": "environment", "Value": "production"},
        {"Key": "model", "Value": "fraud-detection"},
        {"Key": "owner", "Value": "mlops-team"},
        {"Key": "pii", "Value": "false"},
        {"Key": "data-classification", "Value": "confidential"},
    ]
)

print("Waiting for Feature Group to be created...")
while True:
    status = fraud_feature_group.describe()["FeatureGroupStatus"]
    print(f"Status: {status}")
    if status == "Created":
        break
    elif status == "CreateFailed":
        raise Exception(f"Feature Group creation failed!")
    time.sleep(5)

print("✅ fraud-features-v3 Feature Group created successfully")
```

---

## 5.3 Feature Ingestion Pipeline

```python
# code/feature-store/feature_ingestion_pipeline.py
"""
Production feature ingestion pipeline.
Runs as SageMaker Processing Job on schedule.
"""

import pandas as pd
import numpy as np
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FraudFeatureEngineer:
    """
    Computes fraud detection features from raw transaction data.
    Mirrors Uber Michelangelo's FeatureSet concept.
    """
    
    def __init__(self, session: sagemaker.Session):
        self.session = session
        self.featurestore_runtime = boto3.client(
            'sagemaker-featurestore-runtime',
            region_name='us-east-1'
        )
        self.feature_group = FeatureGroup(
            name="fraud-features-v3",
            sagemaker_session=session
        )
    
    def compute_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling window transaction velocity features.
        These are the most predictive features for fraud.
        """
        df = df.sort_values(['customer_id', 'transaction_timestamp'])
        
        # Convert timestamp
        df['ts'] = pd.to_datetime(df['transaction_timestamp'])
        df = df.set_index('ts').sort_index()
        
        # Rolling window aggregations per customer
        grouped = df.groupby('customer_id')
        
        # 1-hour velocity
        df['txn_count_1h'] = grouped['transaction_id'].transform(
            lambda x: x.rolling('1h').count()
        )
        df['spend_1h'] = grouped['amount'].transform(
            lambda x: x.rolling('1h').sum()
        )
        
        # 24-hour velocity
        df['txn_count_24h'] = grouped['transaction_id'].transform(
            lambda x: x.rolling('24h').count()
        )
        df['spend_24h'] = grouped['amount'].transform(
            lambda x: x.rolling('24h').sum()
        )
        
        # 7-day velocity
        df['txn_count_7d'] = grouped['transaction_id'].transform(
            lambda x: x.rolling('7D').count()
        )
        df['spend_7d'] = grouped['amount'].transform(
            lambda x: x.rolling('7D').sum()
        )
        
        return df.reset_index()
    
    def compute_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute statistical features relative to customer history."""
        
        # 30-day averages and std devs
        customer_stats = df.groupby('customer_id').agg(
            avg_txn_amount_30d=('amount', 'mean'),
            std_txn_amount_30d=('amount', 'std'),
            unique_merchants_7d=('merchant_id', 'nunique'),
            unique_countries_30d=('country_code', 'nunique'),
        ).reset_index()
        
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Velocity ratio: current hour velocity vs 30-day daily average
        daily_avg = df.groupby('customer_id')['amount'].transform('mean')
        df['velocity_ratio'] = df['spend_1h'] / (daily_avg / 24 + 1e-8)
        
        # Amount vs customer average ratio
        df['amount_vs_avg_ratio'] = df['amount'] / (df['avg_txn_amount_30d'] + 1e-8)
        
        return df
    
    def compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temporal features."""
        df['ts'] = pd.to_datetime(df['transaction_timestamp'])
        df['hour_of_day'] = df['ts'].dt.hour
        df['day_of_week'] = df['ts'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        df['amount_log'] = np.log1p(df['amount'])
        return df
    
    def ingest_to_feature_store(self, df: pd.DataFrame, batch_size: int = 500):
        """
        Ingest computed features to SageMaker Feature Store.
        Uses the FeatureGroup.ingest() method for batch ingestion.
        """
        df['event_time'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Select only feature group columns
        feature_columns = [
            'transaction_id', 'customer_id', 'event_time',
            'amount', 'amount_log', 'merchant_category', 'country_code',
            'is_international', 'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
            'txn_count_1h', 'txn_count_24h', 'txn_count_7d',
            'spend_1h', 'spend_24h', 'spend_7d',
            'avg_txn_amount_30d', 'std_txn_amount_30d',
            'unique_merchants_7d', 'unique_countries_30d',
            'velocity_ratio', 'amount_vs_avg_ratio',
            'device_fingerprint', 'device_seen_before', 'device_age_days',
        ]
        
        feature_df = df[feature_columns].fillna(0)
        
        # Handle string NaN
        string_cols = ['merchant_category', 'country_code', 'device_fingerprint']
        for col in string_cols:
            feature_df[col] = feature_df[col].fillna('UNKNOWN').astype(str)
        
        logger.info(f"Ingesting {len(feature_df):,} records to Feature Store...")
        
        # SageMaker SDK batch ingest (handles retries internally)
        ingestion_manager = self.feature_group.ingest(
            data_frame=feature_df,
            max_workers=8,           # parallel ingestion workers
            max_processes=4,
            wait=True,
        )
        
        if ingestion_manager.failed_rows:
            failed_count = len(ingestion_manager.failed_rows)
            logger.error(f"Failed to ingest {failed_count} rows")
            # Write failures to S3 for reprocessing
            failed_df = feature_df.iloc[ingestion_manager.failed_rows]
            failed_df.to_parquet(
                f"s3://fintech-datalake-prod/feature-store/failed-ingestion/{datetime.utcnow().date()}/failed.parquet"
            )
        
        logger.info(f"✅ Feature ingestion complete. "
                   f"Success: {len(feature_df) - len(ingestion_manager.failed_rows or [])}, "
                   f"Failed: {len(ingestion_manager.failed_rows or [])}")
    
    def run(self, transactions_df: pd.DataFrame):
        """Full feature engineering pipeline."""
        logger.info(f"Starting feature engineering for {len(transactions_df):,} transactions")
        
        df = self.compute_time_features(transactions_df)
        df = self.compute_velocity_features(df)
        df = self.compute_statistical_features(df)
        
        self.ingest_to_feature_store(df)
        return df
```

---

## 5.4 Online Feature Lookup (Real-Time Inference)

```python
# code/feature-store/online_store_lookup.py
"""
Real-time feature lookup from Feature Store Online Store.
Used by Lambda function during real-time fraud scoring.
Latency target: < 5ms P99
"""

import boto3
import json
import logging
from typing import Dict, List, Optional
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class OnlineFeatureStore:
    """
    High-performance online feature store client.
    Optimized for sub-5ms lookups in Lambda.
    """
    
    def __init__(self, region: str = 'us-east-1'):
        # Use boto3 with connection pooling
        self.client = boto3.client(
            'sagemaker-featurestore-runtime',
            region_name=region,
            config=boto3.session.Config(
                max_pool_connections=50,    # connection pooling
                connect_timeout=2,
                read_timeout=5,
                retries={'max_attempts': 2, 'mode': 'adaptive'},
            )
        )
        self._cache = {}   # in-memory cache for Lambda warm instances
        self._cache_ttl = 300  # 5 minute cache for slowly-changing features
    
    def get_customer_features(self, customer_id: str) -> Optional[Dict]:
        """
        Retrieve customer-level features from online store.
        Cache is used for cold-start optimization in Lambda.
        """
        cache_key = f"customer:{customer_id}"
        cached = self._cache.get(cache_key)
        
        if cached and (time.time() - cached['ts']) < self._cache_ttl:
            return cached['data']
        
        try:
            response = self.client.get_record(
                FeatureGroupName='customer-profile-v2',
                RecordIdentifierValueAsString=customer_id,
                FeatureNames=[
                    'customer_id', 'account_age_days', 'credit_score',
                    'avg_monthly_spend', 'preferred_merchant_categories',
                    'has_fraud_history', 'risk_tier'
                ]
            )
            
            features = {
                item['FeatureName']: item['ValueAsString']
                for item in response.get('Record', [])
            }
            
            self._cache[cache_key] = {'data': features, 'ts': time.time()}
            return features
            
        except self.client.exceptions.ResourceNotFound:
            logger.warning(f"No customer profile found for: {customer_id}")
            return self._default_customer_features()
        except Exception as e:
            logger.error(f"Feature Store lookup failed: {e}")
            return self._default_customer_features()
    
    def get_transaction_features(self, transaction_id: str) -> Optional[Dict]:
        """Retrieve transaction-level features (velocity, etc.)."""
        try:
            response = self.client.get_record(
                FeatureGroupName='fraud-features-v3',
                RecordIdentifierValueAsString=transaction_id,
            )
            return {
                item['FeatureName']: item['ValueAsString']
                for item in response.get('Record', [])
            }
        except Exception as e:
            logger.error(f"Transaction feature lookup failed: {e}")
            return {}
    
    def batch_get_features(self, identifiers: List[Dict]) -> List[Dict]:
        """
        Batch feature retrieval for multiple records.
        More efficient than sequential GetRecord calls.
        Used in near-real-time scoring (e.g., 100 records at once).
        """
        try:
            response = self.client.batch_get_record(
                Identifiers=identifiers
            )
            
            results = {}
            for record in response.get('Records', []):
                key = record['RecordIdentifierValueAsString']
                results[key] = {
                    item['FeatureName']: item['ValueAsString']
                    for item in record.get('Record', [])
                }
            
            # Handle unprocessed identifiers (retry)
            if response.get('UnprocessedIdentifiers'):
                logger.warning(f"{len(response['UnprocessedIdentifiers'])} unprocessed records")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch feature lookup failed: {e}")
            return {}
    
    @staticmethod
    def _default_customer_features() -> Dict:
        """Safe defaults when customer profile not found (new customer)."""
        return {
            'account_age_days': '0',
            'credit_score': '0',
            'avg_monthly_spend': '0',
            'has_fraud_history': '0',
            'risk_tier': 'UNKNOWN',
        }


# Lambda handler for real-time fraud scoring
def lambda_handler(event, context):
    """
    Lambda function for real-time fraud scoring.
    Enriches transaction with features then calls SageMaker endpoint.
    """
    import json
    import boto3
    
    feature_store = OnlineFeatureStore()
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    transaction = event['transaction']
    customer_id = transaction['customer_id']
    
    # Feature enrichment (< 5ms target)
    customer_features = feature_store.get_customer_features(customer_id)
    
    # Build inference payload
    inference_row = [
        float(transaction.get('amount', 0)),
        float(transaction.get('is_international', 0)),
        int(transaction.get('hour_of_day', 12)),
        int(transaction.get('is_weekend', 0)),
        float(customer_features.get('avg_monthly_spend', 0)),
        float(customer_features.get('credit_score', 0)),
        int(customer_features.get('account_age_days', 0)),
        int(customer_features.get('has_fraud_history', 0)),
    ]
    
    # Call SageMaker endpoint (< 3ms target)
    payload = ','.join(str(x) for x in inference_row)
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName='fraud-detection-prod',
        ContentType='text/csv',
        Accept='application/json',
        Body=payload,
    )
    
    result = json.loads(response['Body'].read())
    fraud_probability = result['predictions'][0]['score']
    
    return {
        'transaction_id': transaction['transaction_id'],
        'fraud_probability': fraud_probability,
        'decision': 'BLOCK' if fraud_probability > 0.85 else 'ALLOW',
        'risk_score': int(fraud_probability * 1000),
    }
```

---

## 5.5 Feature Governance & Reuse

```python
# Feature catalog for governance and reuse
# Mirrors what Airbnb's Zipline and Uber's Michelangelo implement

FEATURE_CATALOG = {
    "fraud_features_v3": {
        "description": "Real-time transaction features for fraud detection",
        "owner": "fraud-ml-team",
        "consumers": ["fraud-detection-model-v47", "risk-scoring-model-v12"],
        "freshness_sla": "60 seconds",
        "pii_contains": False,
        "data_classification": "confidential",
        "regulatory_flags": ["SOX", "PCI-DSS"],
        "feature_definitions": {
            "velocity_ratio": {
                "description": "Ratio of 1h spend vs 30d daily average",
                "formula": "spend_1h / (avg_daily_spend_30d / 24)",
                "type": "float",
                "expected_range": [0, 50],
                "importance_rank": 1,  # most important feature
            },
            "amount_vs_avg_ratio": {
                "description": "Current txn amount vs customer average",
                "formula": "amount / avg_txn_amount_30d",
                "type": "float",
                "expected_range": [0, 100],
                "importance_rank": 2,
            },
        },
        "monitoring": {
            "drift_detection": True,
            "drift_threshold": 0.2,  # PSI threshold
            "alert_channel": "slack:#mlops-alerts",
        }
    }
}
```

---

## 5.6 Offline Store for Training Data

```python
# code/feature-store/offline_store_training_query.py
"""
Query offline feature store to build training datasets.
This is how you create point-in-time correct features for training.
"""

import sagemaker
import boto3
from sagemaker.feature_store.feature_group import FeatureGroup
import pandas as pd

session = sagemaker.Session()

def build_training_dataset(
    label_table: str,           # Athena table with fraud labels
    feature_group_names: list,  # Feature groups to join
    start_date: str,
    end_date: str,
    output_s3_uri: str,
) -> str:
    """
    Build point-in-time correct training dataset.
    CRITICAL: Uses AS-OF join to prevent data leakage.
    """
    
    # Get offline store S3 locations for each feature group
    feature_groups = {}
    for fg_name in feature_group_names:
        fg = FeatureGroup(name=fg_name, sagemaker_session=session)
        desc = fg.describe()
        s3_uri = desc['OfflineStoreConfig']['S3StorageConfig']['ResolvedOutputS3Uri']
        feature_groups[fg_name] = s3_uri
    
    # Build point-in-time correct Athena query
    # This prevents future data leakage in training
    query = f"""
    WITH labeled_events AS (
        SELECT 
            transaction_id,
            customer_id,
            label_timestamp,
            is_fraud
        FROM "fintech_feature_store"."fraud_labels"
        WHERE label_date BETWEEN '{start_date}' AND '{end_date}'
    ),
    -- Point-in-time correct feature join
    -- Only use features available BEFORE the label timestamp
    fraud_features AS (
        SELECT 
            ff.*,
            ROW_NUMBER() OVER (
                PARTITION BY ff.transaction_id 
                ORDER BY ff.event_time DESC
            ) as rn
        FROM "fintech_feature_store"."fraud_features_v3" ff
        INNER JOIN labeled_events le 
            ON ff.transaction_id = le.transaction_id
            AND ff.event_time <= le.label_timestamp  -- point-in-time correctness!
    ),
    customer_features AS (
        SELECT 
            cf.*,
            ROW_NUMBER() OVER (
                PARTITION BY cf.customer_id 
                ORDER BY cf.event_time DESC
            ) as rn
        FROM "fintech_feature_store"."customer_profile_v2" cf
        INNER JOIN labeled_events le 
            ON cf.customer_id = le.customer_id
            AND cf.event_time <= le.label_timestamp
    )
    SELECT 
        le.transaction_id,
        le.is_fraud as label,
        -- Transaction features
        ff.amount, ff.amount_log, ff.is_international, 
        ff.hour_of_day, ff.is_weekend, ff.is_night,
        ff.txn_count_1h, ff.txn_count_24h, ff.txn_count_7d,
        ff.spend_1h, ff.spend_24h,
        ff.velocity_ratio, ff.amount_vs_avg_ratio,
        ff.unique_merchants_7d, ff.unique_countries_30d,
        -- Customer features  
        cf.account_age_days, cf.credit_score,
        cf.avg_monthly_spend, cf.has_fraud_history,
        cf.risk_tier
    FROM labeled_events le
    INNER JOIN fraud_features ff ON le.transaction_id = ff.transaction_id AND ff.rn = 1
    LEFT JOIN customer_features cf ON le.customer_id = cf.customer_id AND cf.rn = 1
    """
    
    # Execute via Athena
    athena = boto3.client('athena')
    
    query_execution = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': 'fintech_feature_store'},
        ResultConfiguration={
            'OutputLocation': output_s3_uri,
            'EncryptionConfiguration': {
                'EncryptionOption': 'SSE_KMS',
            }
        },
        WorkGroup='ml-training-workgroup',
    )
    
    execution_id = query_execution['QueryExecutionId']
    print(f"Athena query started: {execution_id}")
    print(f"Training dataset will be at: {output_s3_uri}")
    
    return execution_id
```

---

## 5.7 Feature Drift Detection

```python
# Continuous feature drift monitoring
# Runs as scheduled Lambda or SageMaker Processing Job

def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) for feature drift detection.
    PSI < 0.10: No significant change
    PSI 0.10-0.25: Moderate change — investigate
    PSI > 0.25: Significant change — potential data issue
    """
    import numpy as np
    
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


class FeatureDriftMonitor:
    """Monitor feature distribution drift for production models."""
    
    def __init__(self):
        self.cw = boto3.client('cloudwatch')
        self.sns = boto3.client('sns')
        self.alert_topic = "arn:aws:sns:us-east-1:123456789:mlops-alerts"
    
    def check_all_features(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame):
        """Check drift for all features and emit to CloudWatch."""
        
        numeric_features = baseline_df.select_dtypes(include='number').columns
        drift_report = {}
        
        for feature in numeric_features:
            psi = compute_psi(baseline_df[feature], current_df[feature])
            drift_report[feature] = psi
            
            # Emit to CloudWatch
            self.cw.put_metric_data(
                Namespace='FinTech/FeatureDrift',
                MetricData=[{
                    'MetricName': 'PSI',
                    'Value': psi,
                    'Unit': 'None',
                    'Dimensions': [
                        {'Name': 'FeatureName', 'Value': feature},
                        {'Name': 'FeatureGroup', 'Value': 'fraud-features-v3'},
                    ]
                }]
            )
            
            if psi > 0.25:
                self._send_critical_alert(feature, psi)
            elif psi > 0.10:
                self._send_warning_alert(feature, psi)
        
        return drift_report
    
    def _send_critical_alert(self, feature: str, psi: float):
        self.sns.publish(
            TopicArn=self.alert_topic,
            Subject=f"CRITICAL: Feature Drift Detected — {feature}",
            Message=f"Feature '{feature}' PSI={psi:.4f} exceeds critical threshold 0.25. "
                   f"Immediate investigation required. Model retraining may be needed."
        )
```

---

*Next: [Section 06 — Training Platform →](06-training-platform.md)*
