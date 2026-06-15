# Section 10 — Advanced Deployments: Blue/Green, Canary, Shadow, A/B

---

## 10.1 Blue/Green Deployment

```python
# code/deployment/blue_green_deployment.py
"""
Blue/Green deployment for zero-downtime model updates.
Traffic is switched atomically: 0% → 100% to new model.
"""

import boto3
import time
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
sm = boto3.client('sagemaker', region_name='us-east-1')
sm_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
cw = boto3.client('cloudwatch', region_name='us-east-1')


class BlueGreenDeployment:
    """
    Blue/Green deployment for SageMaker endpoints.
    
    Architecture:
      Blue: Current production endpoint config (old model)
      Green: New endpoint config (new model)
      
    Process:
      1. Create new "green" endpoint config
      2. Run smoke tests against green config
      3. Switch traffic atomically to green
      4. Monitor for 30 min (auto-rollback if issues)
      5. Delete old blue config
    """
    
    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        self.rollback_window_minutes = 30
    
    def deploy(self, new_model_package_arn: str, instance_type: str = 'ml.c5.2xlarge') -> bool:
        """Execute blue/green deployment."""
        
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        
        # Get current (blue) config
        endpoint_info = sm.describe_endpoint(EndpointName=self.endpoint_name)
        blue_config = endpoint_info['EndpointConfigName']
        
        logger.info(f"Blue config: {blue_config}")
        logger.info(f"Starting blue/green deployment...")
        
        # Step 1: Create green config
        green_config = f"fraud-detection-green-{timestamp}"
        
        sm.create_endpoint_config(
            EndpointConfigName=green_config,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': self._create_model_from_package(new_model_package_arn, timestamp),
                'InstanceType': instance_type,
                'InitialInstanceCount': 3,
                'InitialVariantWeight': 1.0,
            }],
            DataCaptureConfig={
                'EnableCapture': True,
                'InitialSamplingPercentage': 20,
                'DestinationS3Uri': f's3://fintech-datalake-prod/monitoring/data-capture/{self.endpoint_name}/',
                'CaptureOptions': [{'CaptureMode': 'Input'}, {'CaptureMode': 'Output'}],
            },
        )
        
        # Step 2: Smoke test green config (before switching traffic)
        # We can't test the config before applying it in standard SageMaker
        # But we can test on a staging endpoint first
        smoke_passed = self._run_smoke_tests_on_staging(new_model_package_arn)
        
        if not smoke_passed:
            sm.delete_endpoint_config(EndpointConfigName=green_config)
            logger.error("Smoke tests failed — deployment aborted")
            return False
        
        # Step 3: Atomic traffic switch (blue → green)
        logger.info(f"Switching traffic to green config: {green_config}")
        
        sm.update_endpoint(
            EndpointName=self.endpoint_name,
            EndpointConfigName=green_config,
            # NO traffic-shifting — this is full atomic switch
        )
        
        # Wait for update to complete
        waiter = sm.get_waiter('endpoint_in_service')
        waiter.wait(
            EndpointName=self.endpoint_name,
            WaiterConfig={'Delay': 15, 'MaxAttempts': 80}
        )
        
        logger.info(f"Traffic switched to green ✅")
        
        # Step 4: Monitor for rollback window
        rollback_needed = self._monitor_for_issues(self.rollback_window_minutes)
        
        if rollback_needed:
            logger.error(f"Issues detected — rolling back to blue config")
            self._rollback(blue_config)
            sm.delete_endpoint_config(EndpointConfigName=green_config)
            return False
        
        # Step 5: Cleanup blue config
        logger.info(f"Deployment successful — cleaning up blue config: {blue_config}")
        sm.delete_endpoint_config(EndpointConfigName=blue_config)
        
        return True
    
    def _monitor_for_issues(self, duration_minutes: int) -> bool:
        """
        Monitor endpoint for errors/latency degradation.
        Returns True if rollback is needed.
        """
        logger.info(f"Monitoring for {duration_minutes} minutes...")
        
        end_time = time.time() + (duration_minutes * 60)
        check_interval = 30  # seconds
        
        while time.time() < end_time:
            time.sleep(check_interval)
            
            # Check error rate
            error_rate = self._get_error_rate_pct(minutes=5)
            if error_rate > 1.0:  # > 1% errors
                logger.error(f"Error rate {error_rate:.2f}% > 1% threshold")
                return True
            
            # Check P99 latency
            p99_ms = self._get_p99_latency_ms(minutes=5)
            if p99_ms > 25.0:  # > 25ms (2.5x normal baseline of 10ms)
                logger.error(f"P99 latency {p99_ms:.1f}ms > 25ms threshold")
                return True
            
            remaining = (end_time - time.time()) / 60
            logger.info(f"  {remaining:.0f}m remaining | Error: {error_rate:.3f}% | P99: {p99_ms:.1f}ms ✅")
        
        return False
    
    def _rollback(self, blue_config: str):
        """Rollback to blue config."""
        sm.update_endpoint(
            EndpointName=self.endpoint_name,
            EndpointConfigName=blue_config,
        )
        waiter = sm.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=self.endpoint_name)
        logger.info(f"🔄 Rolled back to: {blue_config}")
    
    def _get_error_rate_pct(self, minutes: int) -> float:
        """Get endpoint error rate from CloudWatch."""
        import math
        response = cw.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName='Invocation5XXErrors',
            Dimensions=[{'Name': 'EndpointName', 'Value': self.endpoint_name}],
            StartTime=datetime.utcnow().__class__.utcnow().__class__.utcfromtimestamp(
                time.time() - minutes * 60
            ),
            EndTime=datetime.utcnow(),
            Period=minutes * 60,
            Statistics=['Sum'],
        )
        
        total_errors = sum(d['Sum'] for d in response['Datapoints'])
        
        inv_response = cw.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName='Invocations',
            Dimensions=[{'Name': 'EndpointName', 'Value': self.endpoint_name}],
            StartTime=datetime.utcnow().__class__.utcfromtimestamp(time.time() - minutes * 60),
            EndTime=datetime.utcnow(),
            Period=minutes * 60,
            Statistics=['Sum'],
        )
        
        total_invocations = sum(d['Sum'] for d in inv_response['Datapoints'])
        
        if total_invocations == 0:
            return 0.0
        
        return (total_errors / total_invocations) * 100
    
    def _get_p99_latency_ms(self, minutes: int) -> float:
        """Get P99 latency from CloudWatch."""
        response = cw.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName='ModelLatency',
            Dimensions=[{'Name': 'EndpointName', 'Value': self.endpoint_name}],
            StartTime=datetime.utcnow().__class__.utcfromtimestamp(time.time() - minutes * 60),
            EndTime=datetime.utcnow(),
            Period=minutes * 60,
            Statistics=['p99'],
        )
        
        if not response['Datapoints']:
            return 0.0
        
        return response['Datapoints'][0]['p99'] / 1000  # microseconds → ms
    
    def _create_model_from_package(self, model_package_arn: str, timestamp: str) -> str:
        """Create SageMaker Model resource from model package."""
        model_name = f"fraud-detection-{timestamp}"
        
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'ModelPackageName': model_package_arn,
            },
            ExecutionRoleArn='arn:aws:iam::123456789:role/SageMakerExecutionRole-Inference',
            VpcConfig={
                'SecurityGroupIds': ['sg-sagemaker-inference'],
                'Subnets': ['subnet-ml-a', 'subnet-ml-b'],
            },
        )
        
        return model_name
    
    def _run_smoke_tests_on_staging(self, model_package_arn: str) -> bool:
        """Run smoke tests on staging endpoint before production switch."""
        # In real production: deploy to staging endpoint, run test suite
        # Simplified for this guide
        logger.info("Running smoke tests on staging...")
        time.sleep(5)  # simulate test execution
        return True
```

---

## 10.2 Canary Deployment

```python
# code/deployment/canary_deployment.py
"""
Canary deployment: gradually shift traffic from old model to new.
5% → 10% → 25% → 50% → 100% with health checks at each step.
"""

import boto3
import time
import logging

logger = logging.getLogger(__name__)
sm = boto3.client('sagemaker')


class CanaryDeployment:
    """
    Canary deployment using SageMaker endpoint variants.
    
    Traffic routing:
      Variant A (champion/blue): gets (100 - canary_weight)% traffic
      Variant B (challenger/canary): gets canary_weight% traffic
    """
    
    CANARY_STEPS = [5, 10, 25, 50, 100]  # traffic percentages
    STEP_DURATION_MINUTES = 10            # time at each step before advancing
    
    def __init__(self, endpoint_name: str, champion_model: str, challenger_model: str):
        self.endpoint_name = endpoint_name
        self.champion_model = champion_model
        self.challenger_model = challenger_model
    
    def start_canary(self) -> bool:
        """
        Start canary deployment with two variants.
        Initial: champion=99%, canary=1%
        """
        endpoint_config_name = f"canary-{self.endpoint_name}-{int(time.time())}"
        
        sm.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'champion',
                    'ModelName': self.champion_model,
                    'InstanceType': 'ml.c5.2xlarge',
                    'InitialInstanceCount': 3,
                    'InitialVariantWeight': 99,   # 99% traffic
                },
                {
                    'VariantName': 'canary',
                    'ModelName': self.challenger_model,
                    'InstanceType': 'ml.c5.2xlarge',
                    'InitialInstanceCount': 1,
                    'InitialVariantWeight': 1,    # 1% traffic
                },
            ],
        )
        
        sm.update_endpoint(
            EndpointName=self.endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        
        waiter = sm.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=self.endpoint_name)
        
        logger.info("Canary started: champion=99%, canary=1%")
        return self._progress_canary()
    
    def _progress_canary(self) -> bool:
        """
        Progressively increase canary traffic.
        Returns True if canary successfully reached 100%.
        """
        
        for canary_pct in self.CANARY_STEPS:
            champion_pct = 100 - canary_pct
            
            logger.info(f"\nProgressing canary to {canary_pct}% (champion: {champion_pct}%)")
            
            # Update variant weights
            sm.update_endpoint_weights_and_capacities(
                EndpointName=self.endpoint_name,
                DesiredWeightsAndCapacities=[
                    {'VariantName': 'champion', 'DesiredWeight': float(champion_pct)},
                    {'VariantName': 'canary',   'DesiredWeight': float(canary_pct)},
                ]
            )
            
            # Wait for update
            time.sleep(30)
            
            # Monitor at this traffic level
            for minute in range(self.STEP_DURATION_MINUTES):
                time.sleep(60)
                
                healthy = self._check_canary_health(canary_pct)
                
                if not healthy:
                    logger.error(f"Canary unhealthy at {canary_pct}% — rolling back!")
                    self._rollback_to_champion()
                    return False
                
                logger.info(f"  Minute {minute+1}/{self.STEP_DURATION_MINUTES}: canary healthy ✅")
            
            if canary_pct == 100:
                # Canary is now 100% — it becomes the new champion
                logger.info("✅ Canary fully promoted to production!")
                self._promote_canary()
                return True
        
        return True
    
    def _check_canary_health(self, canary_traffic_pct: float) -> bool:
        """
        Compare canary vs champion metrics.
        Canary fails if error rate or latency is significantly worse.
        """
        cw = boto3.client('cloudwatch')
        
        def get_variant_metric(variant: str, metric: str, stat: str = 'Average') -> float:
            response = cw.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName=metric,
                Dimensions=[
                    {'Name': 'EndpointName', 'Value': self.endpoint_name},
                    {'Name': 'VariantName', 'Value': variant},
                ],
                StartTime=time.time() - 300,  # last 5 min
                EndTime=time.time(),
                Period=300,
                Statistics=[stat],
            )
            datapoints = response.get('Datapoints', [])
            return datapoints[0][stat] if datapoints else 0.0
        
        champion_error = get_variant_metric('champion', 'Invocation5XXErrors', 'Sum')
        canary_error = get_variant_metric('canary', 'Invocation5XXErrors', 'Sum')
        
        champion_latency = get_variant_metric('champion', 'ModelLatency', 'p99')
        canary_latency = get_variant_metric('canary', 'ModelLatency', 'p99')
        
        # Fail if canary error rate is 2x champion
        if champion_error > 0 and canary_error > champion_error * 2:
            logger.warning(f"Canary error rate ({canary_error}) > 2x champion ({champion_error})")
            return False
        
        # Fail if canary P99 latency is 1.5x champion
        if champion_latency > 0 and canary_latency > champion_latency * 1.5:
            logger.warning(f"Canary P99 ({canary_latency:.0f}µs) > 1.5x champion ({champion_latency:.0f}µs)")
            return False
        
        return True
    
    def _rollback_to_champion(self):
        """Move all traffic back to champion immediately."""
        sm.update_endpoint_weights_and_capacities(
            EndpointName=self.endpoint_name,
            DesiredWeightsAndCapacities=[
                {'VariantName': 'champion', 'DesiredWeight': 100.0},
                {'VariantName': 'canary',   'DesiredWeight': 0.0},
            ]
        )
        logger.info("🔄 Traffic rolled back to champion")
    
    def _promote_canary(self):
        """Replace champion variant with canary (now the new champion)."""
        timestamp = int(time.time())
        new_config = f"post-canary-{self.endpoint_name}-{timestamp}"
        
        sm.create_endpoint_config(
            EndpointConfigName=new_config,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': self.challenger_model,
                'InstanceType': 'ml.c5.2xlarge',
                'InitialInstanceCount': 3,
                'InitialVariantWeight': 1,
            }],
        )
        
        sm.update_endpoint(
            EndpointName=self.endpoint_name,
            EndpointConfigName=new_config,
        )
        
        logger.info("Canary promoted to single-variant production endpoint")
```

---

## 10.3 Shadow Deployment

```python
# code/deployment/shadow_deployment.py
"""
Shadow testing: new model receives mirrored real traffic
but results are NOT returned to users.
Lets you validate the challenger with real traffic before going live.
"""

import boto3
import time
import logging

logger = logging.getLogger(__name__)
sm = boto3.client('sagemaker')


class ShadowDeployment:
    """
    Shadow deployment: challenger processes all real requests
    but results are discarded. No user impact, real-world validation.
    
    Important: SageMaker doesn't natively support shadow mode routing.
    We implement it at the application layer via Lambda fan-out.
    """
    
    def __init__(self, production_endpoint: str, shadow_endpoint: str):
        self.production = production_endpoint
        self.shadow = shadow_endpoint
        self.sm_runtime = boto3.client('sagemaker-runtime')
        self.s3 = boto3.client('s3')
        self.comparison_bucket = 'fintech-model-comparisons'
    
    def invoke_with_shadow(self, payload: str, request_id: str) -> dict:
        """
        Invoke both endpoints simultaneously.
        Return production result to caller.
        Log both results for comparison analysis.
        """
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Both invocations happen concurrently
            prod_future = executor.submit(
                self._invoke_endpoint, self.production, payload
            )
            shadow_future = executor.submit(
                self._invoke_endpoint, self.shadow, payload
            )
            
            prod_result = prod_future.result(timeout=5)
            
            try:
                shadow_result = shadow_future.result(timeout=10)
                self._log_comparison(request_id, payload, prod_result, shadow_result)
            except Exception as e:
                logger.warning(f"Shadow invocation failed (non-critical): {e}")
        
        # Return ONLY production result to caller
        return prod_result
    
    def _invoke_endpoint(self, endpoint_name: str, payload: str) -> dict:
        """Invoke a single endpoint."""
        import json
        response = self.sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Accept='application/json',
            Body=payload,
        )
        return json.loads(response['Body'].read())
    
    def _log_comparison(self, request_id: str, payload: str,
                        prod: dict, shadow: dict):
        """Log production vs shadow comparison for analysis."""
        import json
        
        comparison = {
            'request_id': request_id,
            'payload': payload,
            'production_score': prod.get('predictions', [{}])[0].get('score', 0),
            'shadow_score': shadow.get('predictions', [{}])[0].get('score', 0),
            'timestamp': time.time(),
        }
        
        self.s3.put_object(
            Bucket=self.comparison_bucket,
            Key=f"shadow-comparisons/{request_id}.json",
            Body=json.dumps(comparison),
        )
    
    def analyze_shadow_results(self, days: int = 7) -> dict:
        """
        Analyze shadow vs production discrepancies.
        High discrepancy = the two models disagree significantly.
        """
        # Query S3 Select or Athena for comparison data
        # (simplified here)
        print(f"Analyzing {days} days of shadow comparisons...")
        
        # Key metrics to compare:
        # 1. Prediction correlation (should be high, > 0.95)
        # 2. Decision agreement rate (same BLOCK/ALLOW decision)
        # 3. Score distribution similarity (PSI)
        # 4. Latency distribution (shadow should not be slower)
        
        print("""
Shadow Analysis Summary (example):
  Prediction correlation: 0.97
  Decision agreement: 94.2% (5.8% disagree)
  Score PSI: 0.08 (acceptable)
  Shadow P99 latency: 8.2ms vs Production 7.8ms (+5%)
  
  Recommendation: SAFE TO CANARY (correlation > 0.95)
        """)
```

---

## 10.4 A/B Testing

```python
# code/deployment/ab_testing.py
"""
A/B Testing: two models serve different user segments.
Used to measure business impact, not just technical metrics.
"""

import boto3
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ABTestManager:
    """
    A/B test management for ML models.
    
    Use case: Test whether new fraud model catches more fraud
    without increasing false positives (customer friction).
    """
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.sm = boto3.client('sagemaker')
        self.sm_runtime = boto3.client('sagemaker-runtime')
        self.kinesis = boto3.client('kinesis')
        self.results_stream = 'ab-test-results'
    
    def get_variant(self, customer_id: str) -> str:
        """
        Deterministic variant assignment based on customer_id hash.
        50% control (A), 50% treatment (B).
        """
        hash_val = hash(customer_id) % 100
        return 'B' if hash_val < 50 else 'A'
    
    def score_transaction(self, transaction: dict) -> dict:
        """
        Score transaction with assigned variant model.
        Log result for A/B analysis.
        """
        customer_id = transaction['customer_id']
        variant = self.get_variant(customer_id)
        
        endpoint = (
            'fraud-detection-variant-b' if variant == 'B'
            else 'fraud-detection-prod'
        )
        
        payload = self._build_payload(transaction)
        response = self.sm_runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType='text/csv',
            Body=payload,
        )
        score = json.loads(response['Body'].read())['predictions'][0]['score']
        
        # Log for analysis
        self._log_result({
            'experiment_id': self.experiment_id,
            'variant': variant,
            'customer_id': customer_id,
            'transaction_id': transaction['transaction_id'],
            'fraud_score': score,
            'decision': 'BLOCK' if score > 0.85 else 'ALLOW',
            'amount': transaction['amount'],
        })
        
        return {'fraud_score': score, 'decision': 'BLOCK' if score > 0.85 else 'ALLOW', 'variant': variant}
    
    def _log_result(self, data: dict):
        """Stream results to Kinesis for real-time analysis."""
        self.kinesis.put_record(
            StreamName=self.results_stream,
            Data=json.dumps(data),
            PartitionKey=data['experiment_id'],
        )
    
    def compute_experiment_results(self) -> Dict:
        """
        Compute A/B test statistical significance.
        Returns: winner, confidence interval, business impact.
        """
        # In production: query Kinesis → Firehose → S3 → Athena
        # Statistical analysis using chi-squared test or t-test
        
        results = {
            'experiment_id': self.experiment_id,
            'variant_a': {
                'fraud_catch_rate': 0.923,    # % of actual fraud caught
                'false_positive_rate': 0.012,  # % of legit txns blocked
                'samples': 500_000,
            },
            'variant_b': {
                'fraud_catch_rate': 0.941,    # +1.8% improvement
                'false_positive_rate': 0.013,  # +0.1% more FP (acceptable)
                'samples': 498_000,
            },
            'statistical_significance': 0.99,  # 99% confident
            'winner': 'B',
            'recommendation': 'PROMOTE_B',
            'estimated_monthly_savings': 180_000,  # $180K less fraud
        }
        
        print(json.dumps(results, indent=2))
        return results
    
    def _build_payload(self, transaction: dict) -> str:
        features = [
            transaction.get('amount', 0),
            transaction.get('is_international', 0),
            transaction.get('hour_of_day', 12),
            transaction.get('is_weekend', 0),
        ]
        return ','.join(str(f) for f in features)
```

---

## 10.5 Deployment Decision Matrix

```
┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Scenario         │ Blue/Green   │ Canary       │ Shadow       │ A/B Test     │
├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Risk level       │ Medium       │ Low          │ Very Low     │ Low          │
│ Rollback speed   │ Instant      │ < 1 minute   │ N/A          │ Instant      │
│ User impact      │ Zero         │ Minimal      │ Zero         │ Controlled   │
│ Validation       │ Pre-deploy   │ Live traffic │ Live traffic │ Live traffic │
│ Time to 100%     │ Minutes      │ Hours        │ Never (test) │ Weeks        │
│ When to use      │ Low-risk     │ High-risk    │ Risky models │ Measure biz  │
│                  │ model swaps  │ production   │ in prod      │ impact       │
└──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

---

*Next: [Section 11 — LLMOps on SageMaker →](11-llmops-on-sagemaker.md)*
