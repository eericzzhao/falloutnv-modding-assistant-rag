import os
import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# services.py anchors the telemetry DB to backend/, so resolve the same absolute path
# here. Connecting to a bare "telemetry.db" resolves against the current working
# directory instead, which is how this script spent a while analyzing a stale copy at
# the repo root while the live app logged to backend/telemetry.db.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("TELEMETRY_DB_PATH", os.path.join(BASE_DIR, "backend", "telemetry.db"))

CANDIDATE_FEATURES = ["pool_size", "context_size"]
TARGET = "latency_ms"


def load_query_logs(db_path):
    """Reads query_logs, restricted to real user queries where possible."""
    conn = sqlite3.connect(db_path)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(query_logs)")]
        if not columns:
            raise RuntimeError(f"no query_logs table in {db_path}")

        # the load-order endpoint runs its synthetic prompts through the same pipeline,
        # so those rows have to be excluded to keep the latency model honest
        if "route" in columns:
            return pd.read_sql_query(
                "SELECT * FROM query_logs WHERE route = 'query'", conn
            )

        print("Note: this database predates the 'route' column, so synthetic")
        print("      load-order calls cannot be filtered out and are included below.")
        return pd.read_sql_query("SELECT * FROM query_logs", conn)
    finally:
        conn.close()


def run_performance_study():
    # 1. Extract the pipeline data
    print(f"Loading telemetry from {DB_PATH}...")
    try:
        df = load_query_logs(DB_PATH)
    except Exception as e:
        print(f"Error reading database: {e}")
        return

    if len(df) < 5:
        print(f"Only {len(df)} usable queries logged. Run a few more through the UI!")
        return

    print(f"Total queries analyzed: {len(df)}")
    print("-" * 40)

    # 2. Descriptive Statistics
    print("--- Core Metrics ---")
    print(f"Average Generation Latency:  {df[TARGET].mean():.2f} ms")
    print(f"Average Cross-Encoder Score: {df['avg_rerank_score'].mean():.4f}")
    print("-" * 40)

    # 3. OLS Regression: What causes latency spikes?
    # Filter out any weird anomalies or 0 latency rows just in case
    clean_df = df[df[TARGET] > 0].copy()

    # A feature that never changes carries no information about latency: it is perfectly
    # collinear with the intercept, so the split between "flat cost" and "cost per unit"
    # is unidentifiable and lstsq just returns one of infinitely many equivalent fits.
    # context_size is currently constant at 5 because run_query hardcodes scored_docs[:5],
    # so it normally gets dropped here rather than contributing a meaningless coefficient.
    features = [f for f in CANDIDATE_FEATURES if clean_df[f].nunique() > 1]
    for f in CANDIDATE_FEATURES:
        if f not in features:
            print(f"Skipping '{f}': constant at {clean_df[f].iloc[0]} in every row "
                  f"-- no variance to regress on.")

    if not features:
        print("No varying inputs left to model. Vary k or the rerank top_n to get signal.")
        return

    X = clean_df[features]
    y = clean_df[TARGET]

    model = LinearRegression()
    model.fit(X, y)
    r_squared = r2_score(y, model.predict(X))

    # 4. Output the Findings
    print()
    print("--- Latency Regression Analysis ---")
    print(f"Base Network/LLM Latency (Intercept): {model.intercept_:.2f} ms")
    labels = {
        "pool_size": "Time added per candidate fetched",
        "context_size": "Time added per chunk sent to LLM",
    }
    for name, coef in zip(features, model.coef_):
        print(f"{labels[name] + ':':<38}{coef:.2f} ms")
    print(f"{'Model R-squared value:':<38}{r_squared:.2f}")

    # 5. Strategic Conclusion Engine
    print("-" * 40)
    print("Engineering Takeaway:")

    if r_squared < 0.2:
        print(f"-> These inputs explain almost none of the latency variance (R^2={r_squared:.2f}).")
        print("   Latency is dominated by something not logged here -- most likely Gemini")
        print("   response time and network jitter. Tuning k will not move the needle.")
        return

    coefs = dict(zip(features, model.coef_))

    if "pool_size" in coefs and "context_size" in coefs:
        # both vary, so the comparison the old version tried to make is finally valid
        if coefs["pool_size"] * clean_df["pool_size"].mean() > coefs["context_size"] * clean_df["context_size"].mean():
            print("-> Retrieval dominates. Consider lowering 'k' in services.py.")
        else:
            print("-> The LLM context window dominates. Compress chunks before generation.")
        return

    # only one input varies -- compare its share of mean latency against the fixed baseline
    name = features[0]
    attributable = coefs[name] * clean_df[name].mean()
    share = attributable / clean_df[TARGET].mean()
    print(f"-> '{name}' accounts for roughly {share:.0%} of average latency "
          f"({attributable:.0f} ms of {clean_df[TARGET].mean():.0f} ms).")
    if share < 0.25:
        print("   The fixed baseline (LLM generation + network) dominates; lowering 'k'")
        print("   would buy little. Look at the generation step instead.")
    else:
        print("   Retrieval is a real cost center. Lowering 'k' in services.py should help.")


if __name__ == "__main__":
    run_performance_study()
