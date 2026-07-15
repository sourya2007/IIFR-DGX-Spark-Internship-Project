import json
import torch
from pathlib import Path

from .config import CONTEXT, HORIZON, TRAIN_SPLIT, TOKEN_START, TOKEN_SEP, \
    TOKEN_PAD, MAX_SEQ_LENGTH, DATA_OUT_DIR

def prepare_data(df, context=CONTEXT, horizon=HORIZON):
    prices = df["Close"].values.astype(float)
    assert len(prices) > context + horizon, "Not enough data"

    params = {"min": float(prices.min()), "max": float(prices.max()),
              "range": float(prices.max() - prices.min()) if prices.max() != prices.min() else 1.0}
    old_params = params.copy()

    input_ids_list, labels_list = [], []
    norm_params_list = []

    for i in range(len(prices) - context - horizon + 1):
        ctx = prices[i:i + context]
        tgt = prices[i + context:i + context + horizon]
        lo, hi = ctx.min(), ctx.max()
        rng = hi - lo if hi != lo else 1.0
        norm_params_list.append({"min": float(lo), "max": float(hi), "range": float(rng)})

        def _norm(v):
            return max(0, min(100, int(round((v - lo) / rng * 100))))
        tokens = [TOKEN_START]
        tokens.extend(_norm(v) for v in ctx)
        tokens.append(TOKEN_SEP)
        tokens.extend(_norm(v) for v in tgt)

        seq = torch.full((MAX_SEQ_LENGTH,), TOKEN_PAD, dtype=torch.long)
        n = min(len(tokens), MAX_SEQ_LENGTH)
        seq[:n] = torch.tensor(tokens[:n], dtype=torch.long)

        lbl = torch.full((MAX_SEQ_LENGTH,), -100, dtype=torch.long)
        input_end = 1 + context
        for p in range(input_end - 1, min(n - 1, MAX_SEQ_LENGTH - 1)):
            lbl[p] = seq[p + 1]

        input_ids_list.append(seq)
        labels_list.append(lbl)

    X = torch.stack(input_ids_list)
    y = torch.stack(labels_list)
    return X, y, norm_params_list, old_params

def run(df):
    print("[stock_prep] Preparing data...")
    X_full, y_full, norm_params_list, old_params = prepare_data(df)

    split = int(len(X_full) * TRAIN_SPLIT)
    X_train, X_val = X_full[:split], X_full[split:]
    y_train, y_val = y_full[:split], y_full[split:]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(X_train, DATA_OUT_DIR / "X_train.pt")
    torch.save(y_train, DATA_OUT_DIR / "y_train.pt")
    torch.save(X_val, DATA_OUT_DIR / "X_val.pt")
    torch.save(y_val, DATA_OUT_DIR / "y_val.pt")
    with open(DATA_OUT_DIR / "norm_params.json", "w") as f:
        json.dump({"global": old_params, "per_window": norm_params_list}, f)

    print(f"  Saved to {DATA_OUT_DIR}/")
    return str(DATA_OUT_DIR)
