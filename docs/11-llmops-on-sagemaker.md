# Section 11 — LLMOps on SageMaker

> **Production Context**: Goldman Sachs GS AI Platform, JPMorgan LLM Suite, Bloomberg GPT — all running large language models for financial research, document intelligence, and code generation.

---

## 11.1 LLMOps Architecture on SageMaker

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLMOPS ARCHITECTURE — Financial Research Assistant       │
│                                                                             │
│  USER: "What are the key risks in Goldman Sachs Q3 2024 earnings report?"  │
│         ↓                                                                   │
│  API Gateway → Lambda (auth + routing)                                     │
│         ↓                                                                   │
│  RAG Retrieval:                                                             │
│    Titan Embeddings → OpenSearch → Top-5 relevant documents                │
│         ↓                                                                   │
│  Prompt Construction:                                                       │
│    System prompt + retrieved context + user query                          │
│         ↓                                                                   │
│  SageMaker Async Endpoint (Llama 3 70B on ml.g5.48xlarge)                  │
│         ↓                                                                   │
│  Response → S3 → Presigned URL → Client                                    │
│         ↓                                                                   │
│  Monitoring:                                                                │
│    Latency, token count, hallucination rate, response quality               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11.2 Deploying Llama 3 70B on SageMaker

```python
# code/llmops/llama3_deployment.py
"""
Production deployment of Llama 3 70B on SageMaker.
Using HuggingFace Text Generation Inference (TGI) container.
"""

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
import json
import time

session = sagemaker.Session()
region = 'us-east-1'
account_id = '123456789012'
role = 'arn:aws:iam::123456789012:role/SageMakerExecutionRole-LLM'

def deploy_llama3_70b(
    endpoint_name: str = 'llm-financial-assistant-prod',
    instance_type: str = 'ml.g5.48xlarge',  # 8x A10G, 192GB GPU
):
    """
    Deploy Llama 3 70B with TGI for financial research assistant.
    
    Instance options:
      ml.g5.48xlarge: 8x A10G (192GB) — Llama 3 70B with 4-bit quantization
      ml.p4d.24xlarge: 8x A100 (320GB) — Llama 3 70B in FP16
      ml.p4de.24xlarge: 8x A100 (640GB) — Llama 3 70B in BF16 + full context
    """
    
    # TGI container — optimized for transformer inference
    llm_image = get_huggingface_llm_image_uri(
        backend='huggingface',   # or 'lmi' for SageMaker LMI
        region=region,
    )
    
    # Environment configuration for TGI
    env = {
        'HF_MODEL_ID': 'meta-llama/Meta-Llama-3-70B-Instruct',
        'HF_TASK': 'text-generation',
        'SM_NUM_GPUS': '8',                    # use all 8 A10G GPUs
        'HUGGING_FACE_HUB_TOKEN': '{{resolve:secretsmanager:hf-token}}',
        'MAX_INPUT_LENGTH': '8000',
        'MAX_TOTAL_TOKENS': '10000',
        'MAX_BATCH_PREFILL_TOKENS': '32000',
        'MAX_CONCURRENT_REQUESTS': '128',
        'QUANTIZE': 'bitsandbytes',            # 4-bit quantization
        'DTYPE': 'float16',
        'SHARDED': 'true',                     # tensor parallelism across 8 GPUs
        'NUM_SHARD': '8',
        # Financial-specific settings
        'TRUST_REMOTE_CODE': 'false',
        'WATERMARK_GAMMA': '0',
        'WATERMARK_DELTA': '0',
    }
    
    llm_model = HuggingFaceModel(
        role=role,
        image_uri=llm_image,
        env=env,
        sagemaker_session=session,
        name=f"llama3-70b-{int(time.time())}",
    )
    
    from sagemaker.async_inference import AsyncInferenceConfig
    
    async_config = AsyncInferenceConfig(
        output_path=f's3://fintech-datalake-prod/llm-outputs/',
        max_concurrent_invocations_per_instance=4,
        notification_config={
            'SuccessTopic': f'arn:aws:sns:{region}:{account_id}:llm-inference-complete',
            'ErrorTopic': f'arn:aws:sns:{region}:{account_id}:llm-inference-error',
        },
    )
    
    predictor = llm_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        async_inference_config=async_config,
        volume_size=200,  # 200GB EBS for model weights cache
        
        # Container startup timeout (70B model takes 8-12 min to load)
        container_startup_health_check_timeout=900,
        model_data_download_timeout=1800,
    )
    
    print(f"✅ Llama 3 70B deployed: {endpoint_name}")
    print(f"   Instance: {instance_type}")
    print(f"   Quantization: 4-bit (bitsandbytes)")
    print(f"   Model load time: ~10 minutes")
    
    return predictor


def deploy_llama3_8b_realtime(
    endpoint_name: str = 'llm-query-classifier',
    instance_type: str = 'ml.g5.2xlarge',  # 1x A10G, 24GB GPU
):
    """
    Llama 3 8B for real-time classification tasks.
    Fits comfortably on a single A10G GPU.
    Use for: query classification, intent detection, PII detection.
    """
    
    llm_image = get_huggingface_llm_image_uri(backend='huggingface', region=region)
    
    env = {
        'HF_MODEL_ID': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'HF_TASK': 'text-generation',
        'SM_NUM_GPUS': '1',
        'HUGGING_FACE_HUB_TOKEN': '{{resolve:secretsmanager:hf-token}}',
        'MAX_INPUT_LENGTH': '4096',
        'MAX_TOTAL_TOKENS': '6144',
        'MAX_CONCURRENT_REQUESTS': '64',
        'DTYPE': 'float16',
    }
    
    model = HuggingFaceModel(
        role=role,
        image_uri=llm_image,
        env=env,
        sagemaker_session=session,
    )
    
    predictor = model.deploy(
        initial_instance_count=2,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=300,
    )
    
    return predictor
```

---

## 11.3 RAG Pipeline — Financial Research Assistant

```python
# code/llmops/rag_pipeline.py
"""
Production RAG pipeline for Financial Research Assistant.
Sources: SEC filings, earnings reports, market research, internal research notes.
"""

import boto3
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RetrievedDocument:
    content: str
    source: str
    relevance_score: float
    document_type: str  # 'SEC_10K', 'EARNINGS_CALL', 'ANALYST_REPORT', etc.


class FinancialRAGPipeline:
    """
    Production RAG pipeline for financial document Q&A.
    Architecture:
      1. Query → Titan Embeddings → 1536-dim vector
      2. OpenSearch kNN → top-k relevant chunks
      3. Reranking with Cohere Rerank
      4. Prompt construction with retrieved context
      5. Llama 3 70B → response generation
    """
    
    def __init__(self, region: str = 'us-east-1'):
        self.bedrock = boto3.client('bedrock-runtime', region_name=region)
        self.opensearch_client = self._init_opensearch()
        self.sm_runtime = boto3.client('sagemaker-runtime', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        
        self.embedding_model_id = 'amazon.titan-embed-text-v2:0'
        self.llm_endpoint = 'llm-financial-assistant-prod'
        self.index_name = 'financial-documents-prod'
        self.top_k = 10
        self.rerank_top_n = 5
    
    def _init_opensearch(self):
        """Initialize OpenSearch client with AWS auth."""
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            'us-east-1',
            'es',
            session_token=credentials.token,
        )
        
        return OpenSearch(
            hosts=[{'host': 'search-fintech-docs.us-east-1.es.amazonaws.com', 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
        )
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for user query using Amazon Titan."""
        response = self.bedrock.invoke_model(
            modelId=self.embedding_model_id,
            body=json.dumps({
                'inputText': query,
                'dimensions': 1536,
                'normalize': True,
            }),
            contentType='application/json',
        )
        
        result = json.loads(response['body'].read())
        return result['embedding']
    
    def retrieve_documents(self, query: str, document_filter: dict = None) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents using kNN search.
        Supports filtering by document type, date range, company.
        """
        query_embedding = self.embed_query(query)
        
        # Build OpenSearch query
        knn_query = {
            'size': self.top_k,
            'query': {
                'bool': {
                    'must': [{
                        'knn': {
                            'embedding': {
                                'vector': query_embedding,
                                'k': self.top_k,
                            }
                        }
                    }],
                    'filter': [],
                }
            },
            '_source': ['content', 'source', 'document_type', 'company', 'date', 'chunk_id'],
        }
        
        # Apply filters
        if document_filter:
            if 'company' in document_filter:
                knn_query['query']['bool']['filter'].append({
                    'term': {'company': document_filter['company']}
                })
            if 'doc_types' in document_filter:
                knn_query['query']['bool']['filter'].append({
                    'terms': {'document_type': document_filter['doc_types']}
                })
            if 'date_range' in document_filter:
                knn_query['query']['bool']['filter'].append({
                    'range': {
                        'date': {
                            'gte': document_filter['date_range']['start'],
                            'lte': document_filter['date_range']['end'],
                        }
                    }
                })
        
        response = self.opensearch_client.search(
            index=self.index_name,
            body=knn_query,
        )
        
        documents = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            documents.append(RetrievedDocument(
                content=source['content'],
                source=source['source'],
                relevance_score=hit['_score'],
                document_type=source['document_type'],
            ))
        
        return documents
    
    def rerank_documents(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Rerank documents using Cohere Rerank for better relevance.
        Reranking improves retrieval quality significantly.
        """
        try:
            response = self.bedrock.invoke_model(
                modelId='cohere.rerank-v3-5:0',
                body=json.dumps({
                    'query': query,
                    'documents': [{'text': doc.content} for doc in documents],
                    'top_n': self.rerank_top_n,
                }),
                contentType='application/json',
            )
            
            result = json.loads(response['body'].read())
            
            # Reorder documents based on rerank scores
            reranked = []
            for item in result['results']:
                doc = documents[item['index']]
                doc.relevance_score = item['relevance_score']
                reranked.append(doc)
            
            return reranked
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return documents[:self.rerank_top_n]
    
    def construct_prompt(self, query: str, documents: List[RetrievedDocument]) -> str:
        """
        Build the final prompt with system instructions and retrieved context.
        """
        context = "\n\n---\n\n".join([
            f"[{doc.document_type}] Source: {doc.source}\n{doc.content}"
            for doc in documents
        ])
        
        system_prompt = """You are a senior financial analyst assistant with expertise in 
analyzing financial reports, regulatory filings, and market data. 
You provide accurate, detailed responses based only on the provided context.

CRITICAL INSTRUCTIONS:
- Only use information from the provided context
- If the context doesn't contain enough information, clearly state what is unknown
- Always cite your sources using [Source Name] notation
- For numerical data, always include the time period and currency
- Do NOT make up financial figures or forecast without explicit data
- Flag any regulatory or compliance considerations when relevant"""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT FROM FINANCIAL DOCUMENTS:
{context}

USER QUESTION:
{query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def query(
        self,
        user_query: str,
        document_filter: dict = None,
        max_tokens: int = 2048,
    ) -> dict:
        """
        Main RAG query pipeline.
        Returns response with citations and confidence score.
        """
        import uuid
        request_id = str(uuid.uuid4())
        
        logger.info(f"RAG query [{request_id}]: {user_query[:100]}...")
        
        # Step 1: Retrieve
        documents = self.retrieve_documents(user_query, document_filter)
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Step 2: Rerank
        documents = self.rerank_documents(user_query, documents)
        logger.info(f"Reranked to top {len(documents)} documents")
        
        # Step 3: Construct prompt
        prompt = self.construct_prompt(user_query, documents)
        
        # Step 4: Generate response (async endpoint)
        input_s3_key = f"llm-inputs/{request_id}.json"
        self.s3.put_object(
            Bucket='fintech-datalake-prod',
            Key=input_s3_key,
            Body=json.dumps({
                'inputs': prompt,
                'parameters': {
                    'max_new_tokens': max_tokens,
                    'temperature': 0.3,        # low temperature for factual accuracy
                    'top_p': 0.9,
                    'do_sample': True,
                    'return_full_text': False,
                    'stop': ['<|eot_id|>', '<|end_of_text|>'],
                }
            }).encode()
        )
        
        sm_runtime = boto3.client('sagemaker-runtime')
        response = sm_runtime.invoke_endpoint_async(
            EndpointName=self.llm_endpoint,
            ContentType='application/json',
            InputLocation=f's3://fintech-datalake-prod/{input_s3_key}',
            InferenceId=request_id,
        )
        
        return {
            'request_id': request_id,
            'output_location': response['OutputLocation'],
            'retrieved_sources': [
                {'source': d.source, 'type': d.document_type, 'score': d.relevance_score}
                for d in documents
            ],
        }
    
    def index_document(self, document: dict):
        """
        Index a new financial document into OpenSearch.
        Called by the document ingestion pipeline.
        """
        # Chunk document
        chunks = self._chunk_document(document['content'], chunk_size=500, overlap=50)
        
        for i, chunk in enumerate(chunks):
            embedding = self.embed_query(chunk)
            
            doc = {
                'content': chunk,
                'embedding': embedding,
                'source': document['source'],
                'document_type': document['document_type'],
                'company': document.get('company', 'unknown'),
                'date': document.get('date'),
                'chunk_id': f"{document['id']}-chunk-{i}",
            }
            
            self.opensearch_client.index(
                index=self.index_name,
                body=doc,
                id=doc['chunk_id'],
            )
    
    @staticmethod
    def _chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
```

---

## 11.4 LLM Fine-Tuning on SageMaker

```python
# code/llmops/llm_finetuning.py
"""
Fine-tune Llama 3 8B on financial domain data.
Using LoRA (Low-Rank Adaptation) to reduce GPU memory requirements.
"""

import sagemaker
from sagemaker.huggingface import HuggingFace
import json

session = sagemaker.Session()
role = 'arn:aws:iam::123456789:role/SageMakerExecutionRole-Training'

def finetune_llama3_financial(
    training_data_s3: str,
    output_s3: str,
    instance_type: str = 'ml.g5.12xlarge',  # 4x A10G
):
    """
    Fine-tune Llama 3 8B with LoRA for financial domain.
    Dataset: financial Q&A pairs, SEC filing summaries, earning call transcripts.
    """
    
    hyperparameters = {
        # Model config
        'model_id': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'max_seq_length': 4096,
        
        # LoRA config (efficient fine-tuning)
        'use_peft': True,
        'lora_r': 64,              # rank of LoRA matrices
        'lora_alpha': 128,
        'lora_dropout': 0.1,
        'lora_target_modules': 'q_proj,v_proj,k_proj,o_proj',  # which layers to adapt
        
        # Training
        'num_train_epochs': 3,
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 4,   # effective batch = 4×4 = 16
        'learning_rate': 2e-4,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.05,
        'weight_decay': 0.01,
        'fp16': True,
        'gradient_checkpointing': True,     # save GPU memory
        
        # Quantization (reduces memory 4x)
        'use_4bit': True,
        'bnb_4bit_compute_dtype': 'float16',
        'bnb_4bit_quant_type': 'nf4',
        
        # Output
        'merge_weights': True,     # merge LoRA into base model
        'output_dir': '/tmp/output',
    }
    
    estimator = HuggingFace(
        entry_point='finetune.py',
        source_dir='code/llmops/finetuning/',
        instance_type=instance_type,
        instance_count=1,
        role=role,
        transformers_version='4.36',
        pytorch_version='2.1',
        py_version='py310',
        hyperparameters=hyperparameters,
        output_path=output_s3,
        base_job_name='llama3-financial-finetune',
        use_spot_instances=False,   # don't use spot for LLM training — expensive restarts
        max_run=24 * 3600,
        volume_size=200,
        environment={
            'HUGGING_FACE_HUB_TOKEN': '{{resolve:secretsmanager:hf-token}}',
        },
    )
    
    estimator.fit({
        'train': sagemaker.TrainingInput(
            s3_data=training_data_s3,
            content_type='application/json'
        )
    })
    
    return estimator.model_data


# Cost estimate
print("""
LLM Fine-Tuning Cost Estimate:
  ml.g5.12xlarge: $5.672/hr
  Estimated time (3 epochs, 100K samples): ~8 hours
  Cost: 8 × $5.672 = $45.38
  
Vs. GPT-4 fine-tuning API:
  100K samples × $0.0080/1K tokens ≈ $800+ (20x more expensive)
""")
```

---

## 11.5 LLMOps Monitoring

```python
# code/llmops/llm_monitoring.py
"""
Production monitoring for LLM endpoints.
Tracks: latency, token usage, hallucination rate, response quality.
"""

import boto3
import json
import re
from typing import Optional

class LLMMonitor:
    """Monitor LLM endpoint performance and quality."""
    
    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        self.cw = boto3.client('cloudwatch')
        self.namespace = 'FinTech/LLMMetrics'
    
    def log_inference_metrics(
        self,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        response_text: str,
    ):
        """Log per-inference metrics to CloudWatch."""
        
        # Quality checks
        contains_hallucination = self._detect_hallucination(response_text)
        pii_detected = self._detect_pii(response_text)
        has_citations = '[' in response_text and ']' in response_text
        
        self.cw.put_metric_data(
            Namespace=self.namespace,
            MetricData=[
                {
                    'MetricName': 'InputTokens',
                    'Value': float(input_tokens),
                    'Unit': 'Count',
                    'Dimensions': [{'Name': 'Endpoint', 'Value': self.endpoint_name}]
                },
                {
                    'MetricName': 'OutputTokens',
                    'Value': float(output_tokens),
                    'Unit': 'Count',
                    'Dimensions': [{'Name': 'Endpoint', 'Value': self.endpoint_name}]
                },
                {
                    'MetricName': 'InferenceLatencyMs',
                    'Value': latency_ms,
                    'Unit': 'Milliseconds',
                    'Dimensions': [{'Name': 'Endpoint', 'Value': self.endpoint_name}]
                },
                {
                    'MetricName': 'HallucinationDetected',
                    'Value': 1.0 if contains_hallucination else 0.0,
                    'Unit': 'Count',
                    'Dimensions': [{'Name': 'Endpoint', 'Value': self.endpoint_name}]
                },
                {
                    'MetricName': 'PIIInResponse',
                    'Value': 1.0 if pii_detected else 0.0,
                    'Unit': 'Count',
                    'Dimensions': [{'Name': 'Endpoint', 'Value': self.endpoint_name}]
                },
                {
                    'MetricName': 'HasCitations',
                    'Value': 1.0 if has_citations else 0.0,
                    'Unit': 'Count',
                    'Dimensions': [{'Name': 'Endpoint', 'Value': self.endpoint_name}]
                },
                {
                    'MetricName': 'CostPerRequest',
                    'Value': self._estimate_cost(input_tokens, output_tokens),
                    'Unit': 'None',  # USD, but CW doesn't have USD unit
                    'Dimensions': [{'Name': 'Endpoint', 'Value': self.endpoint_name}]
                },
            ]
        )
    
    def _detect_hallucination(self, text: str) -> bool:
        """
        Basic hallucination detection for financial responses.
        In production: use a specialized classifier model.
        """
        # Red flags for financial hallucinations
        hallucination_patterns = [
            r'\$[\d,]+\s*(billion|million|trillion)',  # suspicious large numbers
            r'[\d]+%\s+(?:growth|return|yield)',        # specific percentage claims
        ]
        
        for pattern in hallucination_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # This is a simplification — in production, validate against source docs
                pass
        
        return False  # simplified
    
    def _detect_pii(self, text: str) -> bool:
        """Detect PII in LLM output — critical for compliance."""
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',    # SSN
            r'\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b',  # credit card
            r'\b[A-Z]{2}\d{6}\b',          # passport
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for this inference request."""
        # ml.g5.48xlarge: ~$16.29/hr
        # At 4 concurrent requests, cost per request:
        # tokens processed per second: ~100 tokens/s
        # time = (input + output) / 100 seconds
        # cost = time × ($16.29 / 3600)
        
        tokens = input_tokens + output_tokens
        time_seconds = tokens / 100
        cost = time_seconds * (16.29 / 3600)
        return cost
```

---

*Next: [Section 12 — GPU Management →](12-gpu-management.md)*
