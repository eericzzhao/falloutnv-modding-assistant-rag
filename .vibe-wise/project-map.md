# Project Map

## Purpose
FNVMA (Fallout: New Vegas Modding Assistant). A RAG app that answers modding questions and
flags known-bad mods in an uploaded load order. Live at fnvma.vercel.app.

## Requirements
- Answer modding questions grounded in a knowledge base (Nexus Mods data + older guide content).
- Flag known-problematic mods in an uploaded load-order `.txt`.
- Expose retrieval internals (candidate pool, rerank scores) so the frontend can visualize them.
- Test suite must stay offline (pytest-socket `--disable-socket`).

## Components (verified in code, 2026-10-06)
- `backend/main.py` — FastAPI routes: `GET /api/v1/health`, `POST /api/v1/query`,
  `POST /api/v1/analyze-load-order`. `lifespan` builds the engine once into `server_state`.
- `backend/services.py` — `FalloutRAGEngine` (retrieval + rerank + Gemini), telemetry
  (SQLite + S3 sync), `qdrant_status`, `parse_load_order`, `detect_problematic_mods`,
  `KNOWN_BAD_MODS`.
- `s3_utils.py` — optional S3 persistence (no-ops without `AWS_S3_BUCKET`).
- `frontend/` — `index.html`, `style.css`, `app.js` (328 lines, vanilla JS + D3 graph).
- `build_pipeline_NM.py` — live ingestion: Nexus -> Qdrant `fnvma` + `chunks.pkl`.
- `build_pipeline.py` — orphaned (writes local Chroma, unused by backend).
- `analyze_telemetry.py` — OLS latency analysis over telemetry DB.
- `tests/` — `conftest.py`, `test_harness.py` (5 tests), `fixtures/`.

## Main Flow
```
Browser (app.js) --POST /query--> main.py --> FalloutRAGEngine.run_query
   run_query: EnsembleRetriever[ Qdrant dense k=15 | BM25 over chunks.pkl k=15 ]
              --> cross-encoder score each pair --> top 5 --> Gemini 2.5 Flash
              --> log_telemetry (SQLite, S3 upload on daemon thread)
   <-- answer + candidate pool + selected context -- (D3 graph draws pool)

Browser --upload .txt--> /analyze-load-order --> parse_load_order --> detect_problematic_mods
        --> one batched run_query(route="load_order") --> diagnostics prose
```

## Data and Trust Boundaries
- External: Qdrant Cloud (vectors), Gemini API (generation), Nexus API (ingest only),
  AWS S3 (telemetry DB, chunks.pkl, ingest trackers), HuggingFace (model downloads at boot).
- `chunks.pkl` (BM25) and Qdrant `fnvma` (dense) must stay in sync; only the NM pipeline writes both.
- Uploaded load-order files are untrusted user input.
- CORS: hardcoded origin allowlist in `main.py`.

## Build and Deployment
- Backend: `uvicorn backend.main:app --reload --port 7860` from repo root.
- Tests: `pytest` from repo root.
- Deploy: push to `master` -> `sync_to_hf.yml` (gated on tests) force-pushes to HF Space.
- Root `Dockerfile` is production; `backend/Dockerfile` is stale.

## Unknowns / discrepancies found during mapping (2026-10-06)
- CLAUDE.md says `parse_load_order` strips one leading `*`/`-` marker. The code
  (`backend/services.py:330`) does not — it only strips whitespace and `#` lines.
- CLAUDE.md references `tests/test_api.py` and `tests/test_analyze_telemetry.py`; neither exists.
  Only `conftest.py` and `test_harness.py` are present.
- Uncommitted work in progress: `tests/`, `pytest.ini`, `requirements-dev.txt`, `tests.yml`,
  edits to `sync_to_hf.yml` and `.gitignore`.
