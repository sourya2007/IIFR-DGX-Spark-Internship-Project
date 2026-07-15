import os
import json
import numpy as np
import torch

from config import MODEL_DIR, VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, \
    MAX_SEQ_LENGTH, DROPOUT, MAX_GEN_TOKENS, N_FEATURES, FEATURES, \
    TOKEN_START, TOKEN_SEP, TOKEN_PAD
from tinygpt import TinyGPT

def load_model(weights_path, device):
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, DROPOUT)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def parse_tokens(tokens, params):
    start = 0
    while start < len(tokens) and tokens[start] >= 100:
        start += 1
    if start + N_FEATURES <= len(tokens):
        vals = tokens[start:start + N_FEATURES]
    else:
        vals = []
    if len(vals) != N_FEATURES:
        return None
    result = {}
    for f, v in zip(FEATURES, vals):
        result[f] = v / 100.0 * params[f]["range"] + params[f]["min"]
    return result

def run(data_dir):
    print("[evaluate] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(os.path.join(MODEL_DIR, "best.pt"), device)

    with open(os.path.join(data_dir, "norm_params.json")) as f:
        params = json.load(f)

    X_val = torch.load(os.path.join(data_dir, "X_val.pt"))
    y_val = torch.load(os.path.join(data_dir, "y_val.pt"))

    pred_list, actual_list = [], []
    print(f"  Running inference on {len(X_val)} validation samples...")

    context_len = 1 + 12 * N_FEATURES
    for i in range(len(X_val)):
        inp = X_val[i:i+1, :context_len].to(device)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=N_FEATURES + 2, temperature=1.0)
        gen_ids = out[0, context_len:].tolist()
        gen_ids = [t for t in gen_ids if t < 100]
        pred = parse_tokens(gen_ids[:N_FEATURES], params)

        lbl = y_val[i].tolist()
        actual_ids = [t for t in lbl if t >= 0 and t < 100]
        actual = parse_tokens(actual_ids[:N_FEATURES], params)

        if pred and actual:
            pred_list.append(pred)
            actual_list.append(actual)

    if not pred_list:
        print("  No valid predictions generated")
        return {"mae_aqi": -1, "rmse_aqi": -1, "mae_pm": -1, "rmse_pm": -1}

    aqi_pred = np.array([p["aqi_index"] for p in pred_list])
    aqi_act = np.array([a["aqi_index"] for a in actual_list])
    pm_pred = np.array([p["pm2_5"] for p in pred_list])
    pm_act = np.array([a["pm2_5"] for a in actual_list])

    mae_aqi = float(np.mean(np.abs(aqi_pred - aqi_act)))
    rmse_aqi = float(np.sqrt(np.mean((aqi_pred - aqi_act) ** 2)))
    mae_pm = float(np.mean(np.abs(pm_pred - pm_act)))
    rmse_pm = float(np.sqrt(np.mean((pm_pred - pm_act) ** 2)))

    print(f"\n  {'='*42}")
    print(f"  {'Metric':<15} {'AQI':<12} {'PM2.5':<12}")
    print(f"  {'-'*39}")
    print(f"  {'MAE':<15} {mae_aqi:<12.2f} {mae_pm:<12.2f}")
    print(f"  {'RMSE':<15} {rmse_aqi:<12.2f} {rmse_pm:<12.2f}")

    print(f"\n  {'='*62}")
    print(f"  {'#':<4} {'Actual AQI':<12} {'Pred AQI':<12} {'Actual PM2.5':<14} {'Pred PM2.5':<12}")
    print(f"  {'-'*54}")
    for i in range(min(5, len(pred_list))):
        print(f"  {i:<4} {aqi_act[i]:<12.1f} {aqi_pred[i]:<12.1f} {pm_act[i]:<14.1f} {pm_pred[i]:<12.1f}")
    print()

    return {"mae_aqi": mae_aqi, "rmse_aqi": rmse_aqi, "mae_pm": mae_pm, "rmse_pm": rmse_pm}

if __name__ == "__main__":
    run("data_prep_output")
