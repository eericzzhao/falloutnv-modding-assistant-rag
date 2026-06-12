---
title: FNV RAG Backend
emoji: ☢️
colorFrom: yellow
colorTo: pink
sdk: docker
app_port: 7860
---

# FNVMA: Fallout New Vegas Modding Assistant

> A production-grade, decoupled Retrieval-Augmented Generation (RAG) application designed to diagnose unstable load orders and answer highly specific technical modding questions for Fallout: New Vegas.
>
> Feel free to test the service.
> **Live Application:** [fnvma.vercel.app](https://fnvma.vercel.app)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![D3.js](https://img.shields.io/badge/d3.js-F9A03C?style=flat&logo=d3.js&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RAG-FF6F00)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=github-actions&logoColor=white)

## Executive Summary
FNVMA transitions the traditional approach to game modding diagnostics from manual text-parsing to a modern, AI-driven workflow. Originally built as a monolithic prototype, the system has been entirely re-architected into a scalable client-server application. It demonstrates full-stack machine learning infrastructure, interactive data visualization, and applied statistical analysis to resolve complex modding conflicts and prevent game instability.

## Core Architecture & AI Pipeline

The backend is engineered for zero cold-start latency and mathematical precision in document retrieval:

* **Decoupled FastAPI Backend:** Containerized and hosted on Hugging Face Spaces. Utilizes application lifespan managers to cache embedding models directly in system RAM, serving requests rapidly and reliably.
* **Hybrid Search Ensemble:** Combines semantic meaning and exact-keyword frequencies—crucial for identifying specific `.esp` and `.dll` files.
  * *Dense Retriever:* Powered by **Qdrant Cloud** for scalable, stateless k-NN semantic similarity.
  * *Sparse Retriever:* Utilizes BM25 for exact token matching.
* **Dynamic Knowledge Expansion:** Integrates directly with the Nexus Mods API to pull live mod descriptions, version histories, and endorsement metrics. This ensures the LLM weights community standards heavily and avoids recommending outdated patches.
* **Cross-Encoder Re-ranking:** The ensemble output is routed through a HuggingFace Cross-Encoder (`BAAI/bge-reranker-base`). By executing the cross-encoding pipeline manually, the system preserves raw mathematical match scores and compresses candidate document pools to maximize the LLM's context window relevance.
* **Optimized Inference Batching:** Engineered a batch-processing endpoint that compresses the diagnostic workload. Analyzing a 15-mod load order now requires only **1 LLM inference call** instead of 15, preventing `RESOURCE_EXHAUSTED` timeouts and drastically reducing network overhead.

## MLOps, Telemetry, & CI/CD

Observability and automation are built into the core of FNVMA to prove the efficacy of the reranking pipeline and ensure data remains up-to-date without infrastructure costs.

* **GitOps Deployment Automation:** A dedicated GitHub Actions workflow (`sync_to_hf.yml`) automatically mirrors the GitHub repository to the Hugging Face production server upon every push, ensuring continuous delivery.
* **Automated Data Ingestion (CI/CD):** Implemented a scheduled GitHub Actions workflow (`update_kb.yml`) that executes dual ingestion scripts. It parses baseline modding guides and safely navigates Nexus Mods API rate limits to fetch trending mods, embeds the combined data, and upserts it directly to Qdrant Cloud without human intervention.
* **Interactive Semantic Visualization (D3.js):** The custom vanilla frontend (deployed on Vercel) features a real-time, force-directed network graph. This maps the mathematical distance between user queries and high-dimensional document chunks, making the vector space visible and interactive.
* **Asynchronous Telemetry Engine:** A persistent SQLite database silently logs pipeline metrics for every query. Captured data includes candidate pool sizes, selected context chunks, latency metrics, and average rerank scores.
* **Statistical Performance Profiling:** Utilizes `pandas` and `scikit-learn` to conduct Ordinary Least Squares (OLS) regression on logged telemetry. This isolates vector retrieval overhead from API network constraints, allowing us to mathematically optimize the `k` value in the ensemble retriever based on baseline LLM generation latency.

## Strategic Roadmap (Future Deployments)

- [x] **Cloud Vector DB Migration:** Transitioned the local ChromaDB instance to **Qdrant Cloud** to completely decouple storage from the application logic, enabling fully stateless, ephemeral deployments.
- [x] **Containerization & Deployment:** Wrapped the FastAPI backend into a `Dockerfile` hosted on Hugging Face Spaces, and deployed the custom CRT-styled frontend to a Vercel edge network for a live, zero-friction interactive portfolio link.
- [ ] **Quantitative A/B Testing:** Build a testing framework to programmatically measure the statistical significance of different Ensemble Retriever weightings (e.g., evaluating 50/50 vs. 70/30 Dense/Sparse splits for context precision).

## Tech Stack

* **Frontend & Visualization:** Vanilla JavaScript, D3.js, HTML/CSS, Vercel
* **Backend & API:** FastAPI, Python, Docker, Nexus Mods API, Hugging Face Spaces
* **AI & Retrieval:** Qdrant Cloud, HuggingFace (Sentence Transformers & Cross-Encoders), BM25, LangChain
* **Automation:** GitHub Actions (GitOps & ETL workflows)
* **Data Science & Telemetry:** SQLite, Pandas, Scikit-learn, BeautifulSoup4

---
*Patrolling the Mojave almost makes you wish for a nuclear winter.*
