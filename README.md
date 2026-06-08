# FNVMA: Fallout New Vegas Modding Assistant

> A production-grade, decoupled Retrieval-Augmented Generation (RAG) application designed to diagnose unstable load orders and answer highly specific technical modding questions for Fallout: New Vegas.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![D3.js](https://img.shields.io/badge/d3.js-F9A03C?style=flat&logo=d3.js&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RAG-FF6F00)

## Executive Summary
FNVMA transitions the traditional approach to game modding diagnostics from manual text-parsing to a modern, AI-driven workflow. Originally built as a monolithic prototype, the system has been entirely re-architected into a scalable client-server application. It demonstrates full-stack machine learning infrastructure, interactive data visualization, and applied statistical analysis to resolve complex modding conflicts and prevent game instability.

## Core Architecture & AI Pipeline

The backend is engineered for zero cold-start latency and mathematical precision in document retrieval:

* **Decoupled FastAPI Backend:** Utilizes application lifespan managers to cache embedding models directly in memory, serving requests rapidly and reliably.
* **Hybrid Search Ensemble:** Combines semantic meaning and exact-keyword frequencies—crucial for identifying specific `.esp` and `.dll` files.
  * *Dense Retriever:* Powered by ChromaDB for k-NN semantic similarity.
  * *Sparse Retriever:* Utilizes BM25 for exact token matching.
* **Cross-Encoder Re-ranking:** The ensemble output is routed through a HuggingFace Cross-Encoder. By executing the cross-encoding pipeline manually, the system preserves raw mathematical match scores and compresses candidate document pools to maximize the LLM's context window relevance.
* **Optimized Inference Batching:** Engineered a batch-processing endpoint that compresses the diagnostic workload. Analyzing a 15-mod load order now requires only **1 LLM inference call** instead of 15, preventing `RESOURCE_EXHAUSTED` timeouts and drastically reducing network overhead.

## MLOps, Telemetry, & Observability

Observability is built into the core of FNVMA to prove the efficacy of the reranking pipeline and monitor system health.

* **Interactive Semantic Visualization (D3.js):** The custom vanilla frontend features a real-time, force-directed network graph. This maps the mathematical distance between user queries and high-dimensional document chunks, making the vector space visible and interactive.
* **Asynchronous Telemetry Engine:** A persistent SQLite database silently logs pipeline metrics for every query. Captured data includes candidate pool sizes, selected context chunks, latency metrics, and average rerank scores.
* **Statistical Performance Profiling:** Utilizes `pandas` and `scikit-learn` to conduct Ordinary Least Squares (OLS) regression on logged telemetry. This isolates vector retrieval overhead from API network constraints, allowing us to mathematically optimize the `k` value in the ensemble retriever based on baseline LLM generation latency.

## Strategic Roadmap (Future Deployments)

- [ ] **Containerization:** Wrap the FastAPI backend, custom CRT-styled frontend, and vector database into a unified `docker-compose.yml` for fully reproducible builds.
- [ ] **Cloud Deployment:** Push the containerized stack to a public cloud provider (e.g., Render/AWS) to provide a live, zero-friction interactive portfolio link.
- [ ] **CI/CD Pipeline:** Implement GitHub Actions with a cron job to automatically ingest the latest community documentation, ensuring the vector database remains cutting-edge without manual updates.
- [ ] **Quantitative A/B Testing:** Build a testing framework to programmatically measure the statistical significance of different Ensemble Retriever weightings (e.g., evaluating 50/50 vs. 70/30 Dense/Sparse splits for context precision).

## Tech Stack

* **Backend & API:** FastAPI, Python
* **AI & Retrieval:** ChromaDB, HuggingFace (Cross-Encoders), BM25, LangChain
* **Frontend & Visualization:** Vanilla JavaScript, D3.js, HTML/CSS
* **Data Science & Telemetry:** SQLite, Pandas, Scikit-learn (OLS Regression)

---
*Patrolling the Mojave almost makes you wish for a nuclear winter.*
