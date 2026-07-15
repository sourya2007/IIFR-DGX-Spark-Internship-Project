import json
import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_FILE, LOCATION, FEATURES, TARGETS, CONTEXT_HOURS, TRAIN_SPLIT, RANDOM_SEED

def load_and_filter(path):
    df = pd.read_csv(path)
    assert not df.empty, "Dataset is empty"
    df["datetime"] = pd.to_datetime(df["date_ist"] + " " + df["time_ist"], dayfirst=False, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df = df[df["location"] == LOCATION].reset_index(drop=True)
    assert len(df) > CONTEXT_HOURS + 2, f"Not enough rows for {LOCATION}"
    return df

def compute_norm_params(df, features):
    stats = {}
    for f in features:
        lo, hi = df[f].min(), df[f].max()
        stats[f] = {"min": float(lo), "max": float(hi), "range": float(hi - lo) if hi != lo else 1.0}
    return stats

def normalize(val, stats, f):
    scaled = (val - stats[f]["min"]) / stats[f]["range"]
    return int(round(scaled * 100))

def denormalize(scaled, stats, f):
    return scaled / 100.0 * stats[f]["range"] + stats[f]["min"]

def make_token(row, stats, features):
    parts = []
    for f in features:
        short = {"temp_c": "t", "humidity": "h", "pressure_mb": "p", "windspeed_kph": "w",
                 "pm2_5": "P25", "pm10": "P10", "co": "CO", "no2": "NO2", "aqi_index": "AQI"}[f]
        v = normalize(row[f], stats, f)
        parts.append(f"{short}:{v}")
    return " ".join(parts)

def create_sequences(df, stats, features, context):
    texts, targets = [], []
    for i in range(len(df) - context):
        inp_rows = df.iloc[i:i + context]
        tgt_row = df.iloc[i + context]
        inp_tokens = [make_token(r, stats, features) for r in inp_rows.to_dict("records")]
        tgt_tokens = make_token(tgt_row, stats, features)
        texts.append(" ".join(inp_tokens))
        targets.append(tgt_tokens)
    return texts, targets

def run():
    print("[data_prep] Loading dataset...")
    df = load_and_filter(DATA_FILE)
    print(f"  Rows after filter: {len(df)}")

    params = compute_norm_params(df.iloc[:int(len(df) * TRAIN_SPLIT)], FEATURES)
    print(f"  Normalization stats computed from training split")

    texts, targets = create_sequences(df, params, FEATURES, CONTEXT_HOURS)
    X_full = texts
    y_full = targets

    split_idx = int(len(X_full) * TRAIN_SPLIT)
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    y_train, y_val = y_full[:split_idx], y_full[split_idx:]
    print(f"  Train sequences: {len(X_train)}, Val sequences: {len(X_val)}")

    out_dir = Path(__file__).parent.parent / "data_prep_output"
    out_dir.mkdir(exist_ok=True)
    np.savetxt(out_dir / "X_train.txt", X_train, fmt="%s", encoding="utf-8")
    np.savetxt(out_dir / "X_val.txt", X_val, fmt="%s", encoding="utf-8")
    np.savetxt(out_dir / "y_train.txt", y_train, fmt="%s", encoding="utf-8")
    np.savetxt(out_dir / "y_val.txt", y_val, fmt="%s", encoding="utf-8")
    with open(out_dir / "norm_params.json", "w") as f:
        json.dump(params, f)

    print(f"  Saved to {out_dir}/")
    return str(out_dir)

if __name__ == "__main__":
    run()
