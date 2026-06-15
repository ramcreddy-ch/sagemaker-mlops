# Section 04 — Data Platform

> **Production Context**: How Capital One, JPMorgan, and Goldman Sachs structure their data platforms for ML workloads.

---

## 4.1 Enterprise Data Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE DATA PLATFORM                                 │
│                    Financial Intelligence Use Case                          │
│                                                                             │
│  SOURCES                 INGESTION              STORAGE          CONSUME    │
│  ────────                ─────────              ───────          ───────    │
│  Core Banking  ────────▶ AWS DMS       ────────▶              ──▶ Athena   │
│  (Oracle/DB2)            (CDC)                  S3 DATA LAKE  ──▶ Redshift │
│  Transactions  ────────▶ Kinesis Data  ────────▶ (partitioned ──▶ EMR Spark│
│  (real-time)             Streams                by date/type) ──▶ Feature  │
│  Market Data   ────────▶ AWS Glue      ────────▶              ──▶   Store  │
│  (Bloomberg)             (ETL batch)                                       │
│  FICO/Equifax  ────────▶ SFN + Lambda  ────────▶                           │
│  (3rd party)             (API pull)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.2 S3 Data Lake Structure

```
s3://fintech-datalake-prod/
├── raw/                          # Raw, immutable data
│   ├── transactions/
│   │   └── year=2024/month=06/day=15/hour=12/
│   │       └── txn_20240615_120000_part001.parquet
│   ├── customer-profiles/
│   │   └── snapshot_date=2024-06-15/
│   └── market-data/
│       └── source=bloomberg/year=2024/month=06/day=15/
├── processed/                    # Cleaned, validated data
│   ├── transactions/
│   └── customers/
├── features/                     # ML features (partitioned)
│   ├── fraud-features/
│   │   └── feature_date=2024-06-15/
│   └── risk-features/
├── training-datasets/            # Train/val/test splits
│   ├── fraud-detection/
│   │   ├── v20240615/
│   │   │   ├── train.csv
│   │   │   ├── validation.csv
│   │   │   └── test.csv
│   │   └── latest -> v20240615/   # symlink pattern via metadata
├── models/                       # Training artifacts
│   └── fraud-detection-xgboost/
│       └── 2024-06-15-training-job-xyz/
│           └── output/model.tar.gz
├── inference/                    # Batch transform I/O
│   ├── input/
│   └── output/
└── monitoring/                   # Model Monitor baseline/reports
    ├── baselines/
    └── reports/
```

### S3 Bucket Configuration (Production)

```python
# infrastructure/s3_data_lake_setup.py
import boto3
import json

s3 = boto3.client('s3')
region = 'us-east-1'
account_id = '123456789012'

BUCKET_CONFIG = {
    'fintech-datalake-prod': {
        'versioning': True,
        'encryption': 'aws:kms',
        'kms_key': f'arn:aws:kms:{region}:{account_id}:key/mrk-datalake-prod',
        'lifecycle_rules': [
            {
                'id': 'raw-data-tiering',
                'prefix': 'raw/',
                'transitions': [
                    {'days': 30, 'storage_class': 'STANDARD_IA'},
                    {'days': 90, 'storage_class': 'GLACIER'},
                    {'days': 365, 'storage_class': 'DEEP_ARCHIVE'},
                ],
            },
            {
                'id': 'processed-data-tiering',
                'prefix': 'processed/',
                'transitions': [
                    {'days': 60, 'storage_class': 'STANDARD_IA'},
                    {'days': 180, 'storage_class': 'GLACIER'},
                ],
            },
            {
                'id': 'training-datasets-retention',
                'prefix': 'training-datasets/',
                'expiration_days': 730,  # 2 years
            },
        ],
        'replication': {
            'destination_bucket': 'fintech-datalake-dr-us-west-2',
            'destination_region': 'us-west-2',
        },
        'block_public_access': True,
        'access_logging': True,
        'logging_bucket': 'fintech-access-logs-prod',
    }
}

def setup_production_bucket(bucket_name: str, config: dict):
    """Setup S3 bucket with all production configurations."""
    
    # Create bucket
    if region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)
    else:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': region}
        )
    
    # Block all public access (non-negotiable in production)
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            'BlockPublicAcls': True,
            'IgnorePublicAcls': True,
            'BlockPublicPolicy': True,
            'RestrictPublicBuckets': True,
        }
    )
    
    # Enable versioning (required for data lake)
    if config.get('versioning'):
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
    
    # KMS encryption
    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            'Rules': [{
                'ApplyServerSideEncryptionByDefault': {
                    'SSEAlgorithm': 'aws:kms',
                    'KMSMasterKeyID': config['kms_key'],
                },
                'BucketKeyEnabled': True,  # cost optimization
            }]
        }
    )
    
    # Enforce SSL-only
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DenyHTTP",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:*",
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ],
                "Condition": {
                    "Bool": {"aws:SecureTransport": "false"}
                }
            }
        ]
    }
    s3.put_bucket_policy(
        Bucket=bucket_name,
        Policy=json.dumps(bucket_policy)
    )
    
    print(f"Bucket {bucket_name} configured successfully")
```

---

## 4.3 AWS Glue ETL — Production Patterns

```python
# code/data-platform/glue_etl_job.py
# Production Glue ETL for transaction data processing

import sys
import boto3
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timedelta

# Initialize
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'source_database',
    'source_table',
    'target_bucket',
    'processing_date',
    'bookmark_enabled',
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Enable job bookmarks (critical for incremental processing)
# Job bookmarks ensure we don't reprocess already-processed data

# Read source data with bookmark
raw_dyf = glueContext.create_dynamic_frame.from_catalog(
    database=args['source_database'],
    table_name=args['source_table'],
    transformation_ctx="raw_transactions",  # bookmark key
    additional_options={
        "useS3ListImplementation": True,
        "groupFiles": "inPartition",
        "groupSize": "104857600",  # 100MB group size
    }
)

raw_df = raw_dyf.toDF()

# Data Quality Checks (Great Expectations style inline)
def validate_transactions(df):
    """Production data quality validation."""
    
    total_count = df.count()
    
    # Check 1: No null transaction IDs
    null_txn_ids = df.filter(F.col('transaction_id').isNull()).count()
    assert null_txn_ids == 0, f"Found {null_txn_ids} null transaction IDs!"
    
    # Check 2: Amount must be positive
    negative_amounts = df.filter(F.col('amount') <= 0).count()
    if negative_amounts > 0:
        print(f"WARNING: {negative_amounts} negative/zero amounts found")
        # In production, we don't fail — we flag and quarantine
    
    # Check 3: Timestamp within expected range
    stale_threshold = datetime.now() - timedelta(hours=25)
    stale_records = df.filter(
        F.col('transaction_timestamp') < stale_threshold.isoformat()
    ).count()
    
    if stale_records / total_count > 0.05:  # >5% stale is anomalous
        raise Exception(f"Data staleness alert: {stale_records}/{total_count} records stale")
    
    # Check 4: Referential integrity — customer must exist
    # (simplified — in production, join with customer dim table)
    null_customers = df.filter(F.col('customer_id').isNull()).count()
    quarantine_rate = null_customers / total_count
    
    print(f"Data quality summary:")
    print(f"  Total records: {total_count:,}")
    print(f"  Null txn IDs: {null_txn_ids}")
    print(f"  Negative amounts: {negative_amounts}")
    print(f"  Stale records: {stale_records}")
    print(f"  Null customers: {null_customers} ({quarantine_rate:.2%})")
    
    return df, quarantine_rate

df, quarantine_rate = validate_transactions(raw_df)

# Quarantine bad records
quarantine_df = df.filter(F.col('customer_id').isNull())
good_df = df.filter(F.col('customer_id').isNotNull())

# Write quarantine records for human review
if quarantine_df.count() > 0:
    quarantine_df.write \
        .mode('append') \
        .parquet(f"s3://fintech-datalake-prod/quarantine/transactions/{args['processing_date']}/")

# Feature Engineering in ETL
processed_df = good_df \
    .withColumn('hour_of_day', F.hour('transaction_timestamp')) \
    .withColumn('day_of_week', F.dayofweek('transaction_timestamp')) \
    .withColumn('is_weekend', (F.col('day_of_week').isin([1, 7])).cast('int')) \
    .withColumn('is_night', ((F.col('hour_of_day') >= 22) | (F.col('hour_of_day') <= 6)).cast('int')) \
    .withColumn('amount_log', F.log1p(F.col('amount'))) \
    .withColumn('processing_date', F.lit(args['processing_date'])) \
    .withColumn('_ingestion_timestamp', F.current_timestamp())

# Write processed data (partitioned by date)
processed_dyf = DynamicFrame.fromDF(processed_df, glueContext, "processed")

glueContext.write_dynamic_frame.from_options(
    frame=processed_dyf,
    connection_type="s3",
    format="parquet",
    format_options={"compression": "snappy"},
    connection_options={
        "path": f"s3://fintech-datalake-prod/processed/transactions/",
        "partitionKeys": ["processing_date"],
    },
    transformation_ctx="write_processed",
)

# Update Glue Data Catalog (MSCK REPAIR equivalent)
glue_client = boto3.client('glue')
glue_client.batch_create_partition(
    DatabaseName='fintech_processed',
    TableName='transactions',
    PartitionInputList=[{
        'Values': [args['processing_date']],
        'StorageDescriptor': {
            'Location': f"s3://fintech-datalake-prod/processed/transactions/processing_date={args['processing_date']}/",
            'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
            'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
        }
    }]
)

job.commit()
print(f"ETL completed successfully. Processed {processed_df.count():,} records")
```

---

## 4.4 Kafka / MSK — Real-Time Transaction Ingestion

```python
# code/data-platform/kafka_consumer_fraud.py
# Real-time transaction consumer for fraud detection pipeline

from kafka import KafkaConsumer
from kafka.errors import KafkaError
import boto3
import json
import logging
import time
from typing import Generator

logger = logging.getLogger(__name__)

class TransactionConsumer:
    """
    Production Kafka consumer for real-time transaction processing.
    Feeds the SageMaker Feature Store for real-time fraud detection.
    """
    
    def __init__(self, bootstrap_servers: list, topic: str, group_id: str):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset='latest',       # real-time only
            enable_auto_commit=False,          # manual commit for reliability
            max_poll_records=500,
            max_poll_interval_ms=300000,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            # MSK IAM auth
            security_protocol='SASL_SSL',
            sasl_mechanism='OAUTHBEARER',
        )
        
        self.featurestore_runtime = boto3.client(
            'sagemaker-featurestore-runtime',
            region_name='us-east-1'
        )
        self.kinesis = boto3.client('kinesis', region_name='us-east-1')
        self.batch_size = 100
        self.feature_group_name = 'fraud-features-v3'
    
    def process_transaction(self, txn: dict) -> dict:
        """Transform raw transaction into ML features."""
        return {
            'transaction_id': {'Value': str(txn['txn_id']), 'ValueType': 'String'},
            'customer_id': {'Value': str(txn['customer_id']), 'ValueType': 'String'},
            'amount': {'Value': str(txn['amount']), 'ValueType': 'FractionalNumber'},
            'merchant_category': {'Value': txn['mcc_code'], 'ValueType': 'String'},
            'is_international': {'Value': str(int(txn.get('is_intl', False))), 'ValueType': 'Integral'},
            'hour_of_day': {'Value': str(txn['hour_of_day']), 'ValueType': 'Integral'},
            'is_night': {'Value': str(int(txn['hour_of_day'] in range(22, 24) or txn['hour_of_day'] < 6)), 'ValueType': 'Integral'},
            'amount_log': {'Value': str(float(txn['amount']) ** 0.5), 'ValueType': 'FractionalNumber'},
            'event_time': {'Value': txn['timestamp'], 'ValueType': 'String'},
        }
    
    def ingest_to_feature_store(self, records: list):
        """Batch ingest records to SageMaker Feature Store online store."""
        # Feature Store supports batch ingestion up to 10 records
        for i in range(0, len(records), 10):
            batch = records[i:i+10]
            try:
                self.featurestore_runtime.batch_get_record(
                    Identifiers=[{
                        'FeatureGroupName': self.feature_group_name,
                        'RecordIdentifiersValueAsString': [r['transaction_id']['Value'] for r in batch]
                    }]
                )
                for record in batch:
                    self.featurestore_runtime.put_record(
                        FeatureGroupName=self.feature_group_name,
                        Record=list(record.values())
                    )
            except Exception as e:
                logger.error(f"Feature Store ingestion error: {e}")
                # Forward to DLQ
                self._send_to_dlq(batch, str(e))
    
    def _send_to_dlq(self, records: list, error: str):
        """Send failed records to Kinesis DLQ for reprocessing."""
        self.kinesis.put_record(
            StreamName='fraud-feature-dlq',
            Data=json.dumps({'records': records, 'error': error}),
            PartitionKey='dlq'
        )
    
    def run(self):
        """Main consumer loop."""
        logger.info(f"Starting consumer for topic, group: {self.consumer.subscription()}")
        
        pending_records = []
        last_flush = time.time()
        
        try:
            for message in self.consumer:
                txn = message.value
                features = self.process_transaction(txn)
                pending_records.append(features)
                
                # Flush every 100 records OR every 1 second
                if len(pending_records) >= self.batch_size or (time.time() - last_flush) > 1.0:
                    self.ingest_to_feature_store(pending_records)
                    self.consumer.commit()  # commit after successful ingestion
                    
                    logger.info(f"Flushed {len(pending_records)} records to Feature Store")
                    pending_records = []
                    last_flush = time.time()
                    
        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
            raise
        finally:
            self.consumer.close()


if __name__ == '__main__':
    consumer = TransactionConsumer(
        bootstrap_servers=['b-1.msk-cluster.xxx.kafka.us-east-1.amazonaws.com:9098'],
        topic='financial-transactions-prod',
        group_id='fraud-feature-store-consumer'
    )
    consumer.run()
```

---

## 4.5 Amazon Athena — Production Queries

```sql
-- Ad-hoc analysis: Find suspicious transaction patterns
-- Used by data scientists during model development

WITH transaction_stats AS (
    SELECT 
        customer_id,
        COUNT(*) as txn_count_30d,
        SUM(amount) as total_spend_30d,
        AVG(amount) as avg_txn_amount,
        STDDEV(amount) as stddev_amount,
        COUNT(DISTINCT merchant_category) as unique_merchants,
        COUNT(DISTINCT country_code) as unique_countries,
        SUM(CASE WHEN is_night = 1 THEN 1 ELSE 0 END) as night_txn_count,
        MAX(amount) as max_single_txn
    FROM "fintech_processed"."transactions"
    WHERE processing_date >= DATE_FORMAT(DATE_ADD('day', -30, CURRENT_DATE), '%Y-%m-%d')
    GROUP BY customer_id
),
fraud_labels AS (
    SELECT DISTINCT customer_id, 1 as is_fraud
    FROM "fintech_processed"."confirmed_fraud"
    WHERE fraud_date >= DATE_FORMAT(DATE_ADD('day', -30, CURRENT_DATE), '%Y-%m-%d')
)
SELECT 
    t.*,
    COALESCE(f.is_fraud, 0) as fraud_label,
    -- Risk indicators
    CASE WHEN unique_countries > 3 THEN 1 ELSE 0 END as multi_country_flag,
    CASE WHEN stddev_amount > avg_txn_amount * 3 THEN 1 ELSE 0 END as high_variance_flag,
    CASE WHEN night_txn_count * 1.0 / txn_count_30d > 0.5 THEN 1 ELSE 0 END as night_heavy_flag
FROM transaction_stats t
LEFT JOIN fraud_labels f ON t.customer_id = f.customer_id
ORDER BY fraud_label DESC, total_spend_30d DESC;
```

---

## 4.6 Data Quality Framework

```python
# code/data-platform/data_quality_checks.py
# Production data quality framework using AWS Deequ + custom checks

import boto3
import json
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class DataQualityResult:
    check_name: str
    passed: bool
    actual_value: float
    threshold: float
    severity: str  # CRITICAL, WARNING, INFO

class ProductionDataQualityChecker:
    """
    Data quality framework used before training jobs.
    Prevents training on corrupted or drifted data.
    """
    
    def __init__(self, s3_client, cloudwatch_client):
        self.s3 = s3_client
        self.cw = cloudwatch_client
        self.namespace = 'FinTech/DataQuality'
    
    def check_completeness(self, df, column: str, threshold: float = 0.99) -> DataQualityResult:
        """Column completeness check."""
        total = df.count()
        non_null = df[df[column].notna()].shape[0]
        completeness = non_null / total if total > 0 else 0
        
        return DataQualityResult(
            check_name=f'completeness_{column}',
            passed=completeness >= threshold,
            actual_value=completeness,
            threshold=threshold,
            severity='CRITICAL' if completeness < 0.95 else 'WARNING'
        )
    
    def check_freshness(self, df, timestamp_col: str, max_age_hours: int = 25) -> DataQualityResult:
        """Data freshness check — critical for real-time fraud detection."""
        import pandas as pd
        max_ts = df[timestamp_col].max()
        age_hours = (datetime.utcnow() - pd.Timestamp(max_ts).to_pydatetime()).total_seconds() / 3600
        
        return DataQualityResult(
            check_name='data_freshness',
            passed=age_hours <= max_age_hours,
            actual_value=age_hours,
            threshold=max_age_hours,
            severity='CRITICAL'
        )
    
    def check_distribution_drift(self, df, column: str, baseline_stats: dict) -> DataQualityResult:
        """
        Statistical drift detection using Population Stability Index (PSI).
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change — investigate
        PSI > 0.2: Significant change — potential data issue
        """
        import numpy as np
        
        current_mean = df[column].mean()
        current_std = df[column].std()
        
        # Simplified PSI based on mean shift
        expected_mean = baseline_stats['mean']
        z_score = abs(current_mean - expected_mean) / baseline_stats['std']
        
        return DataQualityResult(
            check_name=f'distribution_drift_{column}',
            passed=z_score < 3.0,  # within 3 sigma
            actual_value=z_score,
            threshold=3.0,
            severity='WARNING' if z_score < 5.0 else 'CRITICAL'
        )
    
    def emit_metrics(self, results: List[DataQualityResult], dataset_name: str):
        """Emit data quality metrics to CloudWatch."""
        metric_data = []
        
        for result in results:
            metric_data.append({
                'MetricName': result.check_name,
                'Value': 1.0 if result.passed else 0.0,
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'Dataset', 'Value': dataset_name},
                    {'Name': 'Severity', 'Value': result.severity},
                ]
            })
        
        # CloudWatch accepts max 20 metrics per call
        for i in range(0, len(metric_data), 20):
            self.cw.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data[i:i+20]
            )
    
    def run_all_checks(self, df, dataset_name: str, baseline_stats: dict) -> bool:
        """Run full quality suite. Returns True if training should proceed."""
        results = []
        
        critical_columns = ['transaction_id', 'customer_id', 'amount', 'transaction_timestamp']
        for col in critical_columns:
            if col in df.columns:
                results.append(self.check_completeness(df, col))
        
        results.append(self.check_freshness(df, 'transaction_timestamp'))
        
        if baseline_stats:
            for col, stats in baseline_stats.items():
                if col in df.columns:
                    results.append(self.check_distribution_drift(df, col, stats))
        
        self.emit_metrics(results, dataset_name)
        
        # Fail fast on any CRITICAL failure
        critical_failures = [r for r in results if not r.passed and r.severity == 'CRITICAL']
        
        if critical_failures:
            print(f"CRITICAL data quality failures — BLOCKING TRAINING:")
            for r in critical_failures:
                print(f"  ❌ {r.check_name}: {r.actual_value:.4f} (threshold: {r.threshold})")
            return False
        
        warnings = [r for r in results if not r.passed and r.severity == 'WARNING']
        for r in warnings:
            print(f"  ⚠️  {r.check_name}: {r.actual_value:.4f} (threshold: {r.threshold})")
        
        print(f"✅ Data quality checks passed ({len(results)} checks, {len(warnings)} warnings)")
        return True
```

---

## 4.7 Schema Evolution Strategies

In production, data schemas change. Here's how to handle this safely:

```python
# Schema evolution handler for production data pipelines

class SchemaEvolutionHandler:
    """
    Handles backward/forward compatible schema changes.
    Critical for Feature Store and training data consistency.
    """
    
    SCHEMA_REGISTRY = {
        'transactions_v1': {
            'columns': ['transaction_id', 'customer_id', 'amount', 'timestamp'],
            'deprecated': False,
        },
        'transactions_v2': {
            'columns': ['transaction_id', 'customer_id', 'amount', 'timestamp',
                       'merchant_category', 'country_code'],  # added columns
            'deprecated': False,
            'backward_compatible': True,  # v1 data still works with defaults
        },
        'transactions_v3': {
            'columns': ['transaction_id', 'customer_id', 'amount', 'timestamp',
                       'merchant_category', 'country_code', 'device_fingerprint'],
            'deprecated': False,
            'backward_compatible': True,
        }
    }
    
    def validate_schema(self, df, expected_schema_version: str) -> bool:
        """Validate DataFrame matches expected schema."""
        schema = self.SCHEMA_REGISTRY.get(expected_schema_version)
        if not schema:
            raise ValueError(f"Unknown schema version: {expected_schema_version}")
        
        required_columns = set(schema['columns'])
        actual_columns = set(df.columns)
        
        missing = required_columns - actual_columns
        if missing:
            if schema.get('backward_compatible'):
                # Add missing columns with defaults
                print(f"Schema migration: adding default values for {missing}")
                return True
            else:
                raise ValueError(f"Schema mismatch: missing columns {missing}")
        
        return True
```

---

## 4.8 Data Lineage Tracking

```python
# Data lineage using AWS Lake Formation + Glue Data Catalog

class DataLineageTracker:
    """Track data lineage for regulatory compliance (GDPR, SOX)."""
    
    def __init__(self):
        self.glue = boto3.client('glue')
        self.lakeformation = boto3.client('lakeformation')
    
    def record_transformation(self, source_table: str, target_table: str,
                               job_name: str, row_count: int):
        """Record ETL transformation in Glue Data Catalog."""
        
        # Update table with lineage metadata
        self.glue.update_table(
            DatabaseName='fintech_processed',
            TableInput={
                'Name': target_table,
                'Parameters': {
                    'source_table': source_table,
                    'transformation_job': job_name,
                    'last_updated': datetime.utcnow().isoformat(),
                    'row_count': str(row_count),
                    'lineage_version': '1.0',
                }
            }
        )
    
    def get_lineage(self, table_name: str) -> dict:
        """Get upstream lineage for a table."""
        response = self.glue.get_table(
            DatabaseName='fintech_processed',
            Name=table_name
        )
        params = response['Table'].get('Parameters', {})
        return {
            'table': table_name,
            'source': params.get('source_table'),
            'transform_job': params.get('transformation_job'),
            'last_updated': params.get('last_updated'),
        }
```

---

*Next: [Section 05 — Feature Engineering & Feature Store →](05-feature-engineering.md)*
