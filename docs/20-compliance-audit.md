# Section 20 — Compliance & Audit (SR11-7)

> **Production Context**: In financial services (and healthcare), models are heavily regulated. In the US, the Federal Reserve's SR 11-7 guidance dictates Model Risk Management. You cannot deploy a model without an audit trail.

---

## 20.1 Regulatory Requirements (SR 11-7)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL RISK MANAGEMENT (SR 11-7)                          │
│                                                                             │
│  Requirement                  | SageMaker Implementation                    │
│  ─────────────────────────────|──────────────────────────────────────────── │
│  Model Inventory              | SageMaker Model Registry                    │
│  Reproducibility              | SageMaker Pipelines + MLflow / Experiments  │
│  Model Lineage                | SageMaker Lineage Tracking                  │
│  Explainability               | SageMaker Clarify (SHAP values)             │
│  Bias Detection               | SageMaker Clarify (Disparate Impact)        │
│  Ongoing Monitoring           | SageMaker Model Monitor                     │
│  Independent Review           | IAM Role Separation (Data Scientist vs.     │
│                               | Model Risk Validator)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 20.2 SageMaker Clarify (Explainability & Bias)

```python
# code/compliance/clarify_analysis.py
"""
Run SageMaker Clarify to generate explainability reports (SHAP) and bias metrics.
These reports are mandatory for Model Risk Review before production deployment.
"""

import sagemaker
from sagemaker import clarify

session = sagemaker.Session()
role = 'arn:aws:iam::123456789:role/SageMakerExecutionRole-Compliance'

def run_compliance_report(
    model_name: str,
    train_data_uri: str,
    report_output_uri: str
):
    
    # 1. Setup Clarify Processor
    clarify_processor = clarify.SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        sagemaker_session=session
    )
    
    # 2. Define Data Configuration
    data_config = clarify.DataConfig(
        s3_data_input_path=train_data_uri,
        s3_output_path=report_output_uri,
        label='is_fraud',
        headers=['is_fraud', 'amount', 'age', 'is_international', 'customer_zip'],
        dataset_type='text/csv'
    )
    
    # 3. Define Model Configuration
    # Clarify spins up a shadow endpoint to query the model and calculate SHAP values
    model_config = clarify.ModelConfig(
        model_name=model_name,
        instance_type='ml.c5.xlarge',
        instance_count=1,
        accept_type='text/csv',
        content_type='text/csv'
    )
    
    # 4. Define Bias Configuration
    # Checking if the model is biased against certain age groups
    bias_config = clarify.BiasConfig(
        label_values_or_threshold=[1], # 1 = fraud
        facet_name='age',
        facet_values_or_threshold=[[18, 25], [65, 100]], # Check for age bias
        group_name='customer_zip'
    )
    
    # 5. Define SHAP Configuration (Explainability)
    shap_config = clarify.SHAPConfig(
        baseline=[[0, 50, 40, 0, 90210]], # Baseline for comparison
        num_samples=15,
        agg_method='mean_abs'
    )
    
    # 6. Run the Analysis
    print("Starting Clarify Analysis. This generates PDFs for the Risk team.")
    clarify_processor.run_explainability(
        data_config=data_config,
        model_config=model_config,
        explainability_config=shap_config
    )
    
    clarify_processor.run_bias(
        data_config=data_config,
        bias_config=bias_config,
        model_config=model_config,
        pre_training_methods='all',
        post_training_methods='all'
    )
    
    print(f"Reports available at: {report_output_uri}")
```

---

## 20.3 CloudTrail Audit Logging

For an MLOps platform, you must prove *who* did *what*. AWS CloudTrail logs every SageMaker API call.

**Example Query (Athena on CloudTrail logs):**
*"Who approved the fraud model v47 on Tuesday?"*

```sql
SELECT 
    useridentity.arn,
    eventname,
    requestparameters,
    eventtime
FROM cloudtrail_logs
WHERE eventsource = 'sagemaker.amazonaws.com'
AND eventname = 'UpdateModelPackage'
AND requestparameters LIKE '%fraud-detection-xgboost%'
AND requestparameters LIKE '%Approved%';
```

---

*Next: [Section 21 — Conclusion & Future Trends →](21-conclusion.md)*
