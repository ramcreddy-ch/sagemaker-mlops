# Section 16 — Incident Management (50 Real Production Incidents)

> **Production Context**: Knowing how to build a model is ML Engineering. Knowing how to fix it when it breaks at 3 AM is MLOps. Here is a sample of real-world incidents.

---

## 16.1 Incident Response Playbook

### SEV-1: Endpoint Latency Spike (>50ms P99)

**Symptoms:**
*   CloudWatch Alarm `High-Latency-P99` fires.
*   Upstream services (API Gateway) start returning 504 timeouts.
*   Auto-scaling is thrashing (scaling up and down rapidly).

**Investigation (The 5-Minute Drill):**

1.  **Check CloudWatch Metrics for the Endpoint:** Is it `ModelLatency` (time spent inside the container) or `OverheadLatency` (SageMaker infrastructure)?
    *   *If ModelLatency is high:* The container is struggling.
    *   *If OverheadLatency is high:* SageMaker itself is struggling (rare, but happens during massive traffic spikes).
2.  **Check Container Logs:**
    ```bash
    aws logs tail /aws/sagemaker/Endpoints/fraud-detection-prod --follow
    ```
    Look for out-of-memory (OOM) errors, connection timeouts to external services (like Feature Store), or garbage collection pauses.
3.  **Check CPU/Memory/GPU Utilization:**
    Are we hitting 100% CPU? If so, the requests are queuing up inside the container.
4.  **Check Upstream Traffic:**
    Did invocations spike 10x?

**Root Causes & Resolutions:**

*   **Root Cause 1: Traffic Spike.** Traffic doubled instantly, auto-scaling couldn't catch up (takes ~3-5 mins to spin up new EC2 instances).
    *   *Resolution:* Immediately manually update the endpoint to max capacity.
    *   *Prevention:* Pre-scale before known events; use Step Scaling instead of Target Tracking for faster reaction.
*   **Root Cause 2: Feature Store Latency.** The Lambda enriching the request is slow because DynamoDB is throttling.
    *   *Resolution:* Increase DynamoDB provisioned capacity or enable On-Demand.
*   **Root Cause 3: Large Payload.** A bug upstream started sending 10MB payloads instead of 10KB.
    *   *Resolution:* Roll back upstream service or add payload size validation.

---

### SEV-2: Training Job OOM (Out Of Memory)

**Symptoms:**
*   Training job fails after 4 hours with `AlgorithmError: exit code 137`.
*   CloudWatch metrics show memory hitting 100% right before failure.

**Investigation:**
1.  Check if it's System Memory (RAM) or GPU Memory (VRAM) that OOM'd.
    *   Exit code 137 = Linux OOM Killer (System RAM).
    *   CUDA out of memory = GPU VRAM.

**Root Causes & Resolutions:**

*   **Root Cause 1 (System RAM): Large Dataset in Memory.** Using pandas `read_csv` on a 50GB file on an `ml.m5.xlarge` (16GB RAM).
    *   *Resolution:* Switch to streaming (Pipe mode), use chunking, or upgrade to `ml.m5.4xlarge`. Use Dask or PySpark.
*   **Root Cause 2 (GPU VRAM): Batch Size Too Large.**
    *   *Resolution:* Reduce `batch_size`. Enable gradient accumulation to maintain effective batch size.
*   **Root Cause 3: Memory Leak.** Appending tensors to a list without calling `.detach()` or `.item()` in PyTorch, keeping the entire computation graph in memory.

---

### SEV-2: Data Drift Alarm Triggers

**Symptoms:**
*   SageMaker Model Monitor emits `feature_baseline_drift_amount > 0.1` for the `transaction_amount` feature.
*   PagerDuty alert triggers for Data Quality.

**Investigation:**
1.  Download the constraints violations report from S3:
    ```bash
    aws s3 cp s3://fintech-datalake-prod/monitoring/reports/fraud-detection/violations.json .
    ```
2.  Check the specific feature. What changed? Mean shifted? Distribution changed? New categorical values appeared?

**Root Causes & Resolutions:**

*   **Root Cause 1: Real-world shift.** Inflation happened, holiday season started, average transaction amounts actually went up.
    *   *Resolution:* Trigger a retraining pipeline on the new data. Update the monitor baseline.
*   **Root Cause 2: Upstream pipeline bug.** The data engineering team changed the unit from dollars to cents (value increased 100x).
    *   *Resolution:* Block retraining! Fix the upstream ETL pipeline. Replay the affected data.
*   **Root Cause 3: Missing data.** The feature store stopped receiving updates for a specific feature, so it defaults to 0.
    *   *Resolution:* Fix the ingestion pipeline.

---

*(More incidents in the full playbook...)*

---

*Next: [Section 17 — Security →](17-security.md)*
