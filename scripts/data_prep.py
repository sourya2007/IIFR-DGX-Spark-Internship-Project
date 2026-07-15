import json
import pandas as pd
import torch
from pathlib import Path

from config import DATA_FILE, INPUT_POLLUTANTS, ALL_FEATURES, TRAIN_SPLIT, \
    TOKEN_START, TOKEN_SEP, TOKEN_PAD, MAX_SEQ_LENGTH, DATA_OUT_DIR

LOCATIONS_CSV = None

def get_locations():
    global LOCATIONS_CSV
    if LOCATIONS_CSV is None:
        LOCATIONS_CSV = pd.read_csv(DATA_FILE)
    locs = LOCATIONS_CSV.groupby("location")[["lat", "lon"]].first().reset_index()
    return locs

def load_and_filter(location):
    df = pd.read_csv(DATA_FILE)
    assert not df.empty, "Dataset is empty"
    df = df[df["location"] == location].reset_index(drop=True)
    assert len(df) > 10, f"Not enough rows for {location}"
    df = df[ALL_FEATURES].dropna()
    return df

def compute_norm_params(df):
    params = {}
    for f in ALL_FEATURES:
        lo, hi = float(df[f].min()), float(df[f].max())
        params[f] = {"min": lo, "max": hi, "range": hi - lo if hi != lo else 1.0}
    return params

def normalize(val, stats, f):
    return int(round((val - stats[f]["min"]) / stats[f]["range"] * 100))

def create_pairs(df, stats, features_in, features_out):
    input_ids_list, labels_list = [], []
    for _, row in df.iterrows():
        tokens = [TOKEN_START]
        for f in features_in:
            tokens.append(normalize(row[f], stats, f))
        tokens.append(TOKEN_SEP)
        for f in features_out:
            tokens.append(normalize(row[f], stats, f))

        seq = torch.full((MAX_SEQ_LENGTH,), TOKEN_PAD, dtype=torch.long)
        n = min(len(tokens), MAX_SEQ_LENGTH)
        seq[:n] = torch.tensor(tokens[:n], dtype=torch.long)

        lbl = torch.full((MAX_SEQ_LENGTH,), -100, dtype=torch.long)
        input_end = 1 + len(features_in)
        for p in range(input_end - 1, min(n - 1, MAX_SEQ_LENGTH - 1)):
            lbl[p] = seq[p + 1]

        input_ids_list.append(seq)
        labels_list.append(lbl)
    return torch.stack(input_ids_list), torch.stack(labels_list)

def run(location):
    print(f"[data_prep] Loading data for {location}...")
    df = load_and_filter(location)
    print(f"  Clean rows: {len(df)}")

    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split_idx]
    params = compute_norm_params(train_df)

    X_full, y_full = create_pairs(df, params, INPUT_POLLUTANTS, ["aqi_index"])
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    y_train, y_val = y_full[:split_idx], y_full[split_idx:]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    loc_dir = DATA_OUT_DIR / location.replace(" ", "_")
    loc_dir.mkdir(parents=True, exist_ok=True)
    torch.save(X_train, loc_dir / "X_train.pt")
    torch.save(y_train, loc_dir / "y_train.pt")
    torch.save(X_val, loc_dir / "X_val.pt")
    torch.save(y_val, loc_dir / "y_val.pt")
    with open(loc_dir / "norm_params.json", "w") as f:
        json.dump(params, f)
    print(f"  Saved to {loc_dir}/")
    return str(loc_dir)
