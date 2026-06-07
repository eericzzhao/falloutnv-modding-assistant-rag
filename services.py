# the RAG pipeline logic 
import os
import pickle
from typing import Dict, List, Any
from dotenv import load_dotenv
 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
# re-ranking imports
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# this will load the environment variable from .env automaticlly
load_dotenv()

DB_DIR = "./vnv_chroma_db"

# Dictionary of known horrible, outdated mods (unformatted bc it doesn't matter what's in here)
KNOWN_BAD_MODS = {"New Vegas Stutter Remover": ["NVSR.esp", "nvse_stutter_remover.dll"], "Project Nevada": ["Project Nevada - Core.esm", "Project Nevada - Cyberware.esp", "Project Nevada - Equipment.esm"], "Zan AutoPurge": ["Zan_AutoPurge_SmartAgro_NV.esp"], "Unlimited Companions": ["UnlimitedCompanions.esp"], "Solid Project": ["SolidPorject.esm"]}

class FalloutRAGEngine:
    def __init__(self):
        # 1. Embeddings & Vector DB
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        self.dense_retriever = self.vector_db.as_retriever(search_kwargs={"k": 15})

        # 2. Sparse Retriever (BM25)
        chunks_path = os.path.join(DB_DIR, "chunks.pkl")
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
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

    def run_query(self, query: str) -> Dict[str, Any]:
        """Runs the query through the pipeline, exposing telemetry for D3.js"""
        # Step A: Get base candidates from Hybrid Retrieval
        initial_docs = self.ensemble_retriever.invoke(query)
        
        # Format candidate details for visual tracking
        candidate_pool = []
        for doc in initial_docs:
            candidate_pool.append({
                "text": doc.page_content[:120] + "...",
                "source_file": doc.metadata.get("source", "unknown")
            })

        # Step B: Pass candidates through the Cross-Encoder compressor
        compressed_docs = self.reranker.compress_documents(initial_docs, query)
        
        final_context_chunks = []
        for doc in compressed_docs:
            # Safely capture the score added by the reranker middleware
            score = getattr(doc, "state", {}).get("relevance_score", 0.0)
            final_context_chunks.append({
                "text": doc.page_content,
                "source_file": doc.metadata.get("source", "unknown"),
                "rerank_score": float(score)
            })

        # Step C: Synthesize final output context block
        context_str = "\n\n".join([d.page_content for d in compressed_docs])
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        response = self.llm.invoke(prompt)

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