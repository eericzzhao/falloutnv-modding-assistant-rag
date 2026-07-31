# the RAG pipeline logic
import os
import pickle
import sqlite3
import threading
import time
from typing import Dict, List, Any
from dotenv import load_dotenv

import s3_utils

#from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
# re-ranking imports
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# this will load the environment variable from .env automaticlly
load_dotenv()

#DB_DIR = "./vnv_chroma_db"
#TELEMETRY_DB_PATH = "telemetry.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_DB_PATH = os.path.join(BASE_DIR, "telemetry.db")
CHUNKS_PATH = "chunks.pkl"
QDRANT_COLLECTION = "fnvma"

# S3 keys (see s3_utils.py) — bucket/artifact store for chunks.pkl + telemetry backups
S3_CHUNKS_KEY = "chunks.pkl"
S3_TELEMETRY_KEY = "telemetry/telemetry.db"
# Upload telemetry.db to S3 every N logged queries. This defaults to 1 because the
# counter below is module-level and resets on every process start, while HF Spaces
# sleep after 30 minutes idle and lose their filesystem -- so at N=20 nothing ever
# persisted unless 20 queries landed inside a single wake window, which at this
# traffic volume never happened. Raise it only if the DB grows enough that a
# per-query upload actually costs something.
TELEMETRY_SYNC_INTERVAL = int(os.environ.get("TELEMETRY_SYNC_INTERVAL", "1"))

_telemetry_log_count = 0

# Dictionary of known horrible, outdated mods (unformatted bc it doesn't matter what's in here)
KNOWN_BAD_MODS = {"New Vegas Stutter Remover": ["NVSR.esp", "nvse_stutter_remover.dll"], "Project Nevada": ["Project Nevada - Core.esm", "Project Nevada - Cyberware.esp", "Project Nevada - Equipment.esm"], "Zan AutoPurge": ["Zan_AutoPurge_SmartAgro_NV.esp"], "Unlimited Companions": ["UnlimitedCompanions.esp"], "Solid Project": ["SolidProject.esm"]}

def init_telemetry_db():
    # pull down prior telemetry history (if any) so a redeploy doesn't wipe it
    s3_utils.download_file(S3_TELEMETRY_KEY, TELEMETRY_DB_PATH)
    with sqlite3.connect(TELEMETRY_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        query TEXT,
                        pool_size INTEGER,
                        context_size INTEGER,
                        avg_rerank_score REAL,
                        latency_ms REAL,
                        route TEXT DEFAULT 'query',
                        llm_ms REAL
                        )
        """)
        # this runs on every boot against a DB restored from S3, so migrations have to
        # be conditional -- an unguarded ALTER TABLE would fail the second startup.
        # llm_ms gets no default on purpose: rows logged before it existed genuinely
        # have no measurement, and NULL says that honestly where a 0 would lie.
        existing_columns = [row[1] for row in conn.execute("PRAGMA table_info(query_logs)")]
        for column, ddl in (("route", "TEXT DEFAULT 'query'"), ("llm_ms", "REAL")):
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE query_logs ADD COLUMN {column} {ddl}")

def log_telemetry(query: str, pool_size: int, context_size: int, avg_score:float, latency: float, route: str = "query", llm_ms: float = None):
    global _telemetry_log_count
    with sqlite3.connect(TELEMETRY_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO query_logs (query, pool_size, context_size, avg_rerank_score, latency_ms, route, llm_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query, pool_size, context_size, avg_score, latency, route, llm_ms)
        )

    # periodically back up telemetry.db to S3 in the background so a redeploy
    # doesn't lose history; doesn't block the request thread
    _telemetry_log_count += 1
    if _telemetry_log_count % TELEMETRY_SYNC_INTERVAL == 0:
        threading.Thread(
            target=s3_utils.upload_file,
            args=(TELEMETRY_DB_PATH, S3_TELEMETRY_KEY),
            daemon=True
        ).start()

def qdrant_status(timeout: float = 5.0) -> Dict[str, Any]:
    """Actively probes Qdrant Cloud for the /health endpoint.

    The engine is built once at boot, so a cluster that dies *after* boot leaves
    engine_ready true while every query 500s -- observed in production. A cluster
    that is dead *at* boot fails differently and worse: QdrantVectorStore calls
    _validate_collection_config in __init__, so FalloutRAGEngine raises, lifespan
    never completes, and the Space exits into RUNTIME_ERROR. Uses its own
    short-timeout client so a hanging cluster stalls health rather than real queries.
    """
    status: Dict[str, Any] = {"collection": QDRANT_COLLECTION}
    try:
        client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=timeout,
        )
        info = client.get_collection(QDRANT_COLLECTION)
        status["reachable"] = True
        status["points"] = info.points_count
    except Exception as e:
        # "no available server" here means the cluster is suspended or mid-upgrade
        status["reachable"] = False
        status["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return status

def flush_telemetry() -> bool:
    """Uploads telemetry.db to S3 synchronously. Called on graceful shutdown.

    Belt-and-braces only: a sleeping Space may be killed outright rather than shut
    down cleanly, so this cannot be the primary persistence mechanism -- that is the
    per-query upload in log_telemetry.
    """
    return s3_utils.upload_file(TELEMETRY_DB_PATH, S3_TELEMETRY_KEY)

def telemetry_status() -> Dict[str, Any]:
    """Reports what the telemetry DB actually contains, for the /health endpoint.

    Exists because none of this is observable from outside: a Space can serve stale
    code, skip a migration, or lose every row to an unconfigured S3 bucket while the
    API keeps returning perfectly good answers.
    """
    status: Dict[str, Any] = {
        "db_path": TELEMETRY_DB_PATH,
        "s3_configured": s3_utils.is_configured(),
    }
    try:
        with sqlite3.connect(TELEMETRY_DB_PATH) as conn:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(query_logs)")]
            status["columns"] = columns
            if not columns:
                status["error"] = "query_logs table does not exist"
                return status

            if "route" in columns:
                status["rows_by_route"] = {
                    route: count
                    for route, count in conn.execute(
                        "SELECT route, COUNT(*) FROM query_logs GROUP BY route"
                    )
                }
            else:
                # pre-migration database; report the total rather than pretending
                total = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
                status["rows_by_route"] = {"(no route column)": total}

            if "llm_ms" in columns:
                status["llm_ms_populated"] = conn.execute(
                    "SELECT COUNT(*) FROM query_logs WHERE llm_ms IS NOT NULL"
                ).fetchone()[0]
    except Exception as e:
        # health must answer even when the DB is unreadable -- that IS the finding
        status["error"] = f"{type(e).__name__}: {e}"
    return status

class FalloutRAGEngine:
    def __init__(self):
        init_telemetry_db()

        # 1. Embeddings & Vector DB
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # qdrant cloud vector db connection
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.vector_db = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embedding=self.embeddings
        )
        self.dense_retriever = self.vector_db.as_retriever(search_kwargs={"k": 15})

        # 2. Sparse Retriever (BM25)

        # pull the latest BM25 chunk store from S3 (no-op if AWS_S3_BUCKET unset)
        s3_utils.download_file(S3_CHUNKS_KEY, CHUNKS_PATH)

        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, "rb") as f:
                raw_chunks = pickle.load(f)
            self.sparse_retriever = BM25Retriever.from_documents(raw_chunks)
            self.sparse_retriever.k = 15
        else:
            raise FileNotFoundError("chunks.pkl missing. Run build_pipeline.py first.")

        # 3. Create Ensemble Retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[0.5, 0.5]
        )

        # 4. Cross-Encoder Re-ranking Setup
        self.cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        self.reranker = CrossEncoderReranker(model=self.cross_encoder_model, top_n=5)

        # 5. LLM Setup
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    def run_query(self, query: str, route: str = "query") -> Dict[str, Any]:
        """Runs the query through the pipeline, exposing telemetry for D3.js

        `route` tags the telemetry row so analyze_telemetry.py can separate real user
        questions from the synthetic prompts the load-order endpoint sends through here.
        """
        start_time = time.time()

        # Step A: Get base candidates from Hybrid Retrieval
        initial_docs = self.ensemble_retriever.invoke(query)
        
        # Format candidate details for visual tracking
        candidate_pool = []
        for doc in initial_docs:
            candidate_pool.append({
                "text": doc.page_content,
                "source_file": doc.metadata.get("source", "unknown")
            })

        # Step B: Pass candidates through the Cross-Encoder compressor
        query_doc_pairs = [[query, doc.page_content] for doc in initial_docs]

        raw_scores = self.cross_encoder_model.score(query_doc_pairs)
        scored_docs = []
        for doc,score in zip(initial_docs, raw_scores):
            # Safely capture the score added by the reranker middleware
            source = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
            scored_docs.append({
                "text": doc.page_content,
                "source_file": source,
                "rerank_score": float(score)
            })
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        final_context_chunks = scored_docs[:5]

        # Step C: Synthesize final output context block
        context_str = "\n\n".join([d["text"] for d in final_context_chunks])
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        # timed separately from the total: retrieval params can't explain latency swings
        # that happen entirely inside this call, and without its own column the two are
        # indistinguishable in the telemetry
        llm_start = time.time()
        response = self.llm.invoke(prompt)
        llm_ms = (time.time() - llm_start) * 1000

        latency_ms = (time.time() - start_time) * 1000
        avg_score = sum(c["rerank_score"] for c in final_context_chunks) / len(final_context_chunks) if final_context_chunks else 0.0

        # log to SQLite
        log_telemetry(query, len(candidate_pool), len(final_context_chunks), avg_score, latency_ms, route, llm_ms)

        return {
            "answer": response.content,
            "candidate_pool_size": len(candidate_pool),
            "candidates": candidate_pool,
            "selected_context": final_context_chunks
        }

def parse_load_order(file_content: bytes) -> List[str]:
    """Decodes and parses lines from loadorder.txt files safely"""
    lines = file_content.decode("utf-8-sig").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

def detect_problematic_mods(plugins: List[str]) -> List[str]:
    """Cross-references plugin list with known unstable items"""
    detected = []
    for mod_name, plugin_files in KNOWN_BAD_MODS.items():
        if any(p.lower() in [pl.lower() for pl in plugins] for p in plugin_files):
            detected.append(mod_name)
    return detected