from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from backend.services import FalloutRAGEngine, parse_load_order, detect_problematic_mods

#dictionary object to main cross-route global server
server_state = {}

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
    allow_origins=["*"],
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
    
@app.post("/api/v1/analyze-load-rder")
async def analyze_load_order_file(file: UploadFile = File(...)):
    engine: FalloutRAGEngine = server_state.get("rag_engine")
    if not engine:
        raise HTTPException(status_code=503, detail="RAG Engine service is uninitialized.")
    
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detaul="Invalid format. Please supply a standard text configuration file.")
    
    try:
        content = await file.read()
        user_plugins = parse_load_order(content)
        bad_mods_found = detect_problematic_mods(user_plugins)

        diagnostics = []
        for mod in bad_mods_found:
            # give users a solution for any broken mod that got flagged
            query = f"Explain why the mod '{mod}' is broken or outdated, and tell me the modern alternative to use instead."
            rag_result = engine.run_query(query)
            diagnostics.append({
                "mod_name": mod,
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
    

