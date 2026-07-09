# RAG-Based Fault Diagnosis Assistant for Microservice Systems

A Retrieval-Augmented Generation (RAG) based system for Root Cause Analysis (RCA) in microservice environments. This project compares **Vector Retrieval** and **Vectorless Retrieval** to determine which retrieval strategy provides more accurate and explainable fault diagnosis.

---

## Overview

Modern microservice applications generate large amounts of telemetry data, including logs, metrics, and distributed traces. Identifying the root cause of failures from this data is challenging and time-consuming.

This project develops a RAG-based diagnosis assistant that:

- Detects anomalies from telemetry data.
- Retrieves relevant historical incidents.
- Uses a Large Language Model (LLM) to diagnose the root cause.
- Compares vector and vectorless retrieval strategies using the RCAEval benchmark.

---

## Objectives

- Develop a RAG-based fault diagnosis assistant.
- Compare Vector Retrieval and Vectorless Retrieval.
- Generate explainable root-cause diagnoses.
- Evaluate performance using standard RCAEval metrics.

---

## Project Architecture

```
Telemetry
    │
    ▼
Dataset Loader
    │
    ▼
Anomaly Detection
    │
    ▼
Incident Summary
    │
    ▼
Knowledge Base
    │
    ├──────────────┐
    ▼              ▼
Vector         Vectorless
Retrieval      Retrieval
    │              │
    └──────┬───────┘
           ▼
      Prompt Builder
           ▼
      Llama 3.3 (Ollama)
           ▼
Root Cause + Explanation
           ▼
      Performance Evaluation
```

---

## Features

- RCAEval Dataset Support
- Rolling Z-Score Anomaly Detection
- Incident Formation
- Retrieval Window Generation
- Vector Retrieval using FAISS
- Vectorless Hierarchical Retrieval
- Llama 3.3 Integration via Ollama
- Citation-Based Explanations
- AC@1, AC@3 and Avg@5 Evaluation

---

## Tech Stack

### Programming Language

- Python 3.11+

### Libraries

- Pandas
- NumPy
- FAISS
- LangChain
- Sentence Transformers
- Ollama
- Hugging Face Transformers

### Dataset

- RCAEval Benchmark

### LLM

- Llama 3.3 8B (Local via Ollama)

---

## Repository Structure

```
rag-rca-assistant/
│
├── data/
│
├── docs/
│
├── src/
│   ├── data_loader.py
│   ├── anomaly_detection.py
│   ├── run_anomaly_detection.py
│   ├── knowledge_base.py
│   ├── vector_index.py
│   ├── vectorless_index.py
│   ├── retriever.py
│   ├── prompt_builder.py
│   ├── llm.py
│   ├── evaluation.py
│   └── main.py
│
├── tests/
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

## Workflow

1. Load RCAEval dataset
2. Detect anomalies using rolling Z-score
3. Form incidents
4. Create retrieval windows
5. Build knowledge base
6. Retrieve similar historical incidents
7. Generate diagnosis using Llama 3.3
8. Evaluate using RCAEval metrics

---

## Evaluation Metrics

- AC@1
- AC@3
- Avg@5
- Explanation Quality
- Retrieval Latency

---

## Team Members

| Name | Responsibility |
|------|----------------|
| Member 1 | Dataset Preparation & Anomaly Detection |
| Member 2 | Knowledge Base & Vector Retrieval |
| Member 3 | Vectorless Retrieval |
| Member 4 | LLM Integration & Evaluation |

---

## Future Improvements

- Fine-tuned embedding models
- Hybrid retrieval
- Real-time streaming support
- Knowledge Graph integration
- Multi-agent diagnosis
- Production deployment with Kubernetes

---

## License

This project is developed for academic research and educational purposes.
