import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from config import DATA_FILE, LOCATION, FEATURES, CONTEXT_HOURS, TRAIN_SPLIT, \
    TOKEN_START, TOKEN_SEP, TOKEN_PAD, MAX_SEQ_LENGTH, N_FEATURES, DATA_OUT_DIR

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

def row_to_tokens(row, stats):
    return [normalize(row[f], stats, f) for f in FEATURES]

def create_sequences(df, stats, context):
    input_ids_list, labels_list = [], []
    for i in range(len(df) - context):
        tokens = [TOKEN_START]
        for j in range(context):
            tokens.extend(row_to_tokens(df.iloc[i + j], stats))
        tokens.append(TOKEN_SEP)
        tgt = row_to_tokens(df.iloc[i + context], stats)
        tokens.extend(tgt)

        seq = torch.full((MAX_SEQ_LENGTH,), TOKEN_PAD, dtype=torch.long)
        n = min(len(tokens), MAX_SEQ_LENGTH)
        seq[:n] = torch.tensor(tokens[:n], dtype=torch.long)

        lbl = torch.full((MAX_SEQ_LENGTH,), -100, dtype=torch.long)
        input_end = 1 + context * N_FEATURES
        lbl[input_end:n] = seq[input_end:n]

        input_ids_list.append(seq)
        labels_list.append(lbl)
    return torch.stack(input_ids_list), torch.stack(labels_list)

def run():
    print("[data_prep] Loading dataset...")
    df = load_and_filter(DATA_FILE)
    print(f"  Rows after filter: {len(df)}")

    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split_idx]
    params = compute_norm_params(train_df, FEATURES)
    print(f"  Normalization stats computed from training split")

    X_full, y_full = create_sequences(df, params, CONTEXT_HOURS)
    train_sz = split_idx - CONTEXT_HOURS
    X_train, X_val = X_full[:train_sz], X_full[train_sz:]
    y_train, y_val = y_full[:train_sz], y_full[train_sz:]
    print(f"  Train sequences: {len(X_train)}, Val sequences: {len(X_val)}")

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(X_train, DATA_OUT_DIR / "X_train.pt")
    torch.save(y_train, DATA_OUT_DIR / "y_train.pt")
    torch.save(X_val, DATA_OUT_DIR / "X_val.pt")
    torch.save(y_val, DATA_OUT_DIR / "y_val.pt")
    with open(DATA_OUT_DIR / "norm_params.json", "w") as f:
        json.dump(params, f)

    print(f"  Saved to {DATA_OUT_DIR}/")
    return str(DATA_OUT_DIR)

if __name__ == "__main__":
    run()
