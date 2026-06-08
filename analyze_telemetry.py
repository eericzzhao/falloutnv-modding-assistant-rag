import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def run_performance_study():
    # 1. Extract the pipeline data
    print("Loading telemetry from SQLite...")
    try:
        conn = sqlite3.connect("telemetry.db")
        df = pd.read_sql_query("SELECT * FROM query_logs", conn)
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")
        return

    if df.empty or len(df) < 5:
        print("Not enough data logged yet. Run a few more queries through the UI!")
        return

    print(f"Total queries analyzed: {len(df)}")
    print("-" * 40)

    # 2. Descriptive Statistics
    print("--- Core Metrics ---")
    print(f"Average Generation Latency:  {df['latency_ms'].mean():.2f} ms")
    print(f"Average Cross-Encoder Score: {df['avg_rerank_score'].mean():.4f}")
    print("-" * 40)

    # 3. OLS Regression: What causes latency spikes?
    # We want to measure how the size of the initial candidate pool and 
    # the final context size mathematically impact processing time.
    
    features = ['pool_size', 'context_size']
    target = 'latency_ms'

    # Filter out any weird anomalies or 0 latency rows just in case
    clean_df = df[df['latency_ms'] > 0].copy()
    
    X = clean_df[features]
    y = clean_df[target]

    # Fit the linear model
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    r_squared = r2_score(y, predictions)

    # 4. Output the Findings
    print("--- Latency Regression Analysis ---")
    print(f"Base Network/LLM Latency (Intercept): {model.intercept_:.2f} ms")
    print(f"Time added per candidate fetched:     {model.coef_[0]:.2f} ms")
    print(f"Time added per chunk sent to LLM:     {model.coef_[1]:.2f} ms")
    print(f"Model R-squared value:                {r_squared:.2f}")
    
    # 5. Strategic Conclusion Engine
    print("-" * 40)
    print("Engineering Takeaway:")
    if model.coef_[0] > model.coef_[1]:
        print("-> The Dense/Sparse retrieval phase is your bottleneck. Consider lowering your 'k' value in services.py.")
    else:
        print("-> The LLM context window is the bottleneck. Consider compressing chunks further before generation.")

if __name__ == "__main__":
    run_performance_study()