import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from backend.services import (
    FalloutRAGEngine,
    parse_load_order,
    detect_problematic_mods,
    telemetry_status,
)

#dictionary object to main cross-route global server
server_state = {}

# sync_to_hf.yml overwrites VERSION with the master SHA it deployed, so /health reports
# which commit is actually serving traffic. The committed placeholder reads "dev" and
# keeps local Docker builds working, since COPY of a missing file is a hard build error.
_VERSION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION"
)

def _read_version() -> str:
    try:
        with open(_VERSION_FILE) as f:
            return f.read().strip() or "unknown"
    except OSError:
        return "unknown"

APP_VERSION = _read_version()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Fallout New Vegas AI Models and DB Context...")
    # instantiate models and data structures only once during bootstrap sequence
    server_state["rag_engine"] = FalloutRAGEngine()
    yield
    print("clearing active server allocations...")
    server_state.clear()

app = FastAPI(
    title="FNVMA",
    version = "1.0.0",
    lifespan=lifespan
)

# important for custom frontends to fetch from this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://eericzzhao.github.io", # production
        "https://fnvma.vercel.app", # vercel server
        "http://127.0.0.1:5500", # local live server
        "http://localhost:5500",
        "http://127.0.0.1:8000", # fallback local ports
        "http://localhost:8000"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    candidate_pool_size: int
    candidates: List[Dict[str, Any]]
    selected_context: List[Dict[str, Any]]

@app.get("/api/v1/health")
async def health_check():
    """Reports what is actually running, not just that something is.

    A 200 from /docs only proves the container is up. This distinguishes "the Space
    is alive" from "the deployed commit is the one I pushed, the telemetry migration
    applied, rows are being logged, and S3 persistence is on".
    """
    engine_ready = bool(server_state.get("rag_engine"))
    return {
        "status": "ok" if engine_ready else "initializing",
        "version": APP_VERSION,
        "engine_ready": engine_ready,
        "telemetry": telemetry_status(),
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag_pipeline(request: QueryRequest):
    engine: FalloutRAGEngine = server_state.get("rag_engine")
    if not engine:
        raise HTTPException(status_code=503, detail="RAG Engine service is currently uninitialized.")
    
    try:
        results = engine.run_query(request.question)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")
    
@app.post("/api/v1/analyze-load-order")
async def analyze_load_order_file(file: UploadFile = File(...)):
    engine: FalloutRAGEngine = server_state.get("rag_engine")
    if not engine:
        raise HTTPException(status_code=503, detail="RAG Engine service is uninitialized.")
    
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Invalid format. Please supply a standard text configuration file.")
    
    try:
        content = await file.read()
        user_plugins = parse_load_order(content)
        bad_mods_found = detect_problematic_mods(user_plugins)

        diagnostics = []

        # reminder: previous method was way too expensive (API for every bad mod)
        # Batch every bad mod --> LLM only needs to run API once rather than every single time it saw it
        if bad_mods_found:
            # combine all bad mods into 1 comma-separeated string
            mods_list_str = ", ".join(bad_mods_found)
            query = f"The user's load order contains the following outdated/broken mods: {mods_list_str}. Briefly explain why each is broken and list the modern alternative."

            # tagged so these synthetic prompts don't skew the latency regression
            rag_result = engine.run_query(query, route="load_order")
            diagnostics.append({
                "mod_name": "Multiple Issues Detected",
                "issue_description": rag_result["answer"]
            })

        return {
            "status": "success",
            "plugins_parsed": len(user_plugins), 
            "issues_detected": len(bad_mods_found),
            "diagnostics": diagnostics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract data payload: {str(e)}")
    

