import json
import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_FILE, LOCATION, FEATURES, TARGETS, CONTEXT_HOURS, TRAIN_SPLIT, RANDOM_SEED

FEATURE_SHORT = ["t", "h", "p", "w", "P25", "P10", "CO", "NO2", "AQI"]
FEATURE_ORDER = list(zip(FEATURES, FEATURE_SHORT))

def load_and_filter(path):
    df = pd.read_csv(path)
    assert not df.empty, "Dataset is empty"
    df["datetime"] = pd.to_datetime(df["date_ist"] + " " + df["time_ist"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df = df[df["location"] == LOCATION].reset_index(drop=True)
    assert len(df) > CONTEXT_HOURS + 2
    return df

def compute_norm_params(df, features):
    stats = {}
    for f in features:
        lo, hi = df[f].min(), df[f].max()
        stats[f] = {"min": float(lo), "max": float(hi), "range": float(hi - lo) if hi != lo else 1.0}
    return stats

def normalize(val, stats, f):
    return int(round((val - stats[f]["min"]) / stats[f]["range"] * 100))

def denormalize(scaled, stats, f):
    return scaled / 100.0 * stats[f]["range"] + stats[f]["min"]

def make_token(row, stats):
    return "_".join(str(normalize(row[f], stats, f)) for f, _ in FEATURE_ORDER)

def create_sequences(df, stats, context):
    X, Y = [], []
    for i in range(len(df) - context):
        inp_hours = [make_token(df.iloc[i + j], stats) for j in range(context)]
        tgt_hour = make_token(df.iloc[i + context], stats)
        X.append(f"In:{context} {' '.join(inp_hours)}")
        Y.append(tgt_hour)
    return X, Y

def run():
    print("[data_prep] Loading dataset...")
    df = load_and_filter(DATA_FILE)
    print(f"  Rows after filter: {len(df)}")

    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split_idx]
    params = compute_norm_params(train_df, FEATURES)
    print(f"  Normalization stats computed from training split")

    X_full, y_full = create_sequences(df, params, CONTEXT_HOURS)
    X_train, X_val = X_full[:split_idx - CONTEXT_HOURS], X_full[split_idx - CONTEXT_HOURS:]
    y_train, y_val = y_full[:split_idx - CONTEXT_HOURS], y_full[split_idx - CONTEXT_HOURS:]
    print(f"  Train sequences: {len(X_train)}, Val sequences: {len(X_val)}")

    out_dir = Path(__file__).parent.parent / "data_prep_output"
    out_dir.mkdir(exist_ok=True)
    def save_lines(path, lines):
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    save_lines(out_dir / "X_train.txt", X_train)
    save_lines(out_dir / "X_val.txt", X_val)
    save_lines(out_dir / "y_train.txt", y_train)
    save_lines(out_dir / "y_val.txt", y_val)
    with open(out_dir / "norm_params.json", "w") as f:
        json.dump(params, f)

    print(f"  Saved to {out_dir}/")
    return str(out_dir)

if __name__ == "__main__":
    run()
