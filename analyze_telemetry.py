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
LABELS = {
    "pool_size": "Time added per candidate fetched",
    "context_size": "Time added per chunk sent to LLM",
}
MIN_ROWS = 5
MIN_R2 = 0.2


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


def describe_split(df):
    """Splits total latency into LLM generation vs everything else.

    Returns the subset of rows carrying an llm_ms measurement, or None when the
    column is absent (old database) or never populated (no traffic since the
    column was added). Rows predating the column hold NULL rather than 0, so they
    have to be dropped rather than treated as zero-cost generation.
    """
    if "llm_ms" not in df.columns:
        return None
    timed = df[df["llm_ms"].notna()].copy()
    if timed.empty:
        return None

    timed["pipeline_ms"] = timed["latency_ms"] - timed["llm_ms"]
    return timed


def fit_model(clean_df, target):
    """Fits latency against whichever retrieval inputs actually vary.

    A feature that never changes carries no information: it is perfectly collinear
    with the intercept, so the split between "flat cost" and "cost per unit" is
    unidentifiable and lstsq just returns one of infinitely many equivalent fits.
    context_size is constant at 5 whenever run_query hardcodes scored_docs[:5], so
    it normally gets dropped here rather than contributing a meaningless coefficient.
    """
    features = [f for f in CANDIDATE_FEATURES if clean_df[f].nunique() > 1]
    for f in CANDIDATE_FEATURES:
        if f not in features:
            print(f"Skipping '{f}': constant at {clean_df[f].iloc[0]} in every row "
                  f"-- no variance to regress on.")

    if not features:
        print("No varying inputs left to model. Vary k or the rerank top_n to get signal.")
        return None

    X = clean_df[features]
    y = clean_df[target]
    model = LinearRegression()
    model.fit(X, y)
    return model, features, r2_score(y, model.predict(X))


def report_model(model, features, r_squared, target_label):
    print()
    print(f"--- Regression: {target_label} ---")
    print(f"{'Fixed baseline (Intercept):':<38}{model.intercept_:.2f} ms")
    for name, coef in zip(features, model.coef_):
        print(f"{LABELS[name] + ':':<38}{coef:.2f} ms")
    print(f"{'Model R-squared value:':<38}{r_squared:.2f}")


def print_takeaway(model, features, r_squared, clean_df, target, pipeline_share=None):
    print("-" * 40)
    print("Engineering Takeaway:")

    # Without this the next few lines read as advice about total latency, which they
    # are not -- when the target is pipeline_ms they only describe the slice of the
    # request that retrieval actually controls.
    if pipeline_share is not None:
        print(f"(Scope: the {pipeline_share:.0%} of each request that is not LLM generation.)")

    if r_squared < MIN_R2:
        print(f"-> These inputs explain almost none of the variance (R^2={r_squared:.2f}).")
        print("   Whatever drives this number is not being measured here.")
        return

    coefs = dict(zip(features, model.coef_))
    means = {f: clean_df[f].mean() for f in features}

    if "pool_size" in coefs and "context_size" in coefs:
        # both inputs vary, so the comparison the original version attempted is valid
        if coefs["pool_size"] * means["pool_size"] > coefs["context_size"] * means["context_size"]:
            print("-> Fetching candidates costs more than assembling context.")
            print("   Lowering 'k' in services.py is the lever within this scope.")
        else:
            print("-> Assembling context costs more than fetching candidates.")
            print("   Compressing chunks is the lever within this scope.")
        return

    # only one input varies -- weigh its share against the fixed baseline
    name = features[0]
    attributable = coefs[name] * means[name]
    share = attributable / clean_df[target].mean()
    print(f"-> '{name}' accounts for roughly {share:.0%} of the average "
          f"({attributable:.0f} ms of {clean_df[target].mean():.0f} ms).")
    if share < 0.25:
        print("   The fixed baseline dominates; lowering 'k' would buy little.")
    else:
        print("   This is the dominant cost within this scope; lowering 'k' would cut it.")


def run_performance_study():
    # 1. Extract the pipeline data
    print(f"Loading telemetry from {DB_PATH}...")
    try:
        df = load_query_logs(DB_PATH)
    except Exception as e:
        print(f"Error reading database: {e}")
        return

    if len(df) < MIN_ROWS:
        print(f"Only {len(df)} usable queries logged. Run a few more through the UI!")
        return

    print(f"Total queries analyzed: {len(df)}")
    print("-" * 40)

    # 2. Descriptive Statistics
    print("--- Core Metrics ---")
    print(f"Average Total Latency:       {df['latency_ms'].mean():.2f} ms")
    print(f"Average Cross-Encoder Score: {df['avg_rerank_score'].mean():.4f}")

    # 3. Split the total before modelling it. Retrieval parameters cannot explain time
    # spent inside llm.invoke(), so mixing the two guarantees a weak, misleading fit.
    timed = describe_split(df)
    if timed is None:
        print()
        print("No llm_ms measurements in this database, so generation time cannot be")
        print("separated from retrieval. Falling back to modelling total latency, which")
        print("conflates the two. Log some traffic on a current build for a clean split.")
        target, target_label, model_df = "latency_ms", "total latency", df
    else:
        llm_mean = timed["llm_ms"].mean()
        pipe_mean = timed["pipeline_ms"].mean()
        total_mean = timed["latency_ms"].mean()
        print()
        print(f"--- Latency Decomposition ({len(timed)} timed queries) ---")
        print(f"{'LLM generation:':<26}{llm_mean:>10.0f} ms  ({llm_mean / total_mean:.0%} of total)")
        print(f"{'Retrieval + reranking:':<26}{pipe_mean:>10.0f} ms  ({pipe_mean / total_mean:.0%} of total)")
        print(f"{'LLM spread (min-max):':<26}{timed['llm_ms'].min():>10.0f} - {timed['llm_ms'].max():.0f} ms")
        print(f"{'Pipeline spread (min-max):':<26}{timed['pipeline_ms'].min():>10.0f} - {timed['pipeline_ms'].max():.0f} ms")

        # the retrieval knobs only control the non-LLM portion, so that is what we model
        target, target_label, model_df = "pipeline_ms", "retrieval + reranking cost", timed

    print("-" * 40)

    # Filter out any weird anomalies or 0 latency rows just in case
    clean_df = model_df[model_df[target] > 0].copy()
    if len(clean_df) < MIN_ROWS:
        print(f"Only {len(clean_df)} rows with positive {target}; not enough to model.")
        return

    fitted = fit_model(clean_df, target)
    if fitted is None:
        return

    model, features, r_squared = fitted
    report_model(model, features, r_squared, target_label)
    pipeline_share = None
    if timed is not None:
        pipeline_share = timed["pipeline_ms"].mean() / timed["latency_ms"].mean()
    print_takeaway(model, features, r_squared, clean_df, target, pipeline_share)

    # 4. The decomposition answers the practical question even when the model cannot
    if timed is not None:
        print("-" * 40)
        print("Where the time actually goes:")
        share = timed["llm_ms"].mean() / timed["latency_ms"].mean()
        if share > 0.7:
            print(f"-> {share:.0%} of every request is Gemini generating tokens. Retrieval")
            print("   tuning cannot meaningfully move total latency; streaming the response")
            print("   or a faster model is the lever that would.")
        else:
            print(f"-> Generation is {share:.0%} of total; the pipeline itself is a real")
            print("   share of the cost and is worth optimizing.")


if __name__ == "__main__":
    run_performance_study()
