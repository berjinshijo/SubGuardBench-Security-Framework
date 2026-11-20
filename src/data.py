from pathlib import Path
import pandas as pd

def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Put the downloaded dataset under data/raw/ and update config.yaml.")
    suffix = path.suffix.lower()
    if suffix == ".csv": return pd.read_csv(path)
    if suffix in (".parquet", ".pq"): return pd.read_parquet(path)
    if suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    if suffix in (".xlsx", ".xls"): return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")

def load_and_validate(cfg):
    df = read_table(cfg["data"]["input_path"])
    if df.empty: raise ValueError("Dataset is empty.")
    ts = cfg["data"].get("timestamp_col")
    if ts and ts in df.columns:
        df[ts] = pd.to_datetime(df[ts], errors="coerce")
        df = df.dropna(subset=[ts]).sort_values(ts).reset_index(drop=True)
    print(f"[Stage 1] Loaded {len(df):,} rows and {len(df.columns)} columns")
    print("[Stage 1] Columns:", list(df.columns))
    return df
