# persistent way to log analyzed data
import sqlite3
import time

DB_PATH = "telemetry.db"

def init_telemetry_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        query TEXT,
                        pool_size INTEGER,
                        context_size INTEGER,
                        avg_rerank_score REAL,
                        latency_ms REAL
                        )
        """)

def log_telemetry(query: str, pool_size: int, context_size: int, avg_score:float, latency: float):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO query_logs (query, pool_size, context_size, avg_rerank_score, latency_ms) VALUES (?, ?, ?, ?, ?)",
            (query, pool_size, context_size, avg_score, latency)
        )