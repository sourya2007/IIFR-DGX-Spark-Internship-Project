import json
import os
import torch

from config import VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, \
    DROPOUT, INPUT_POLLUTANTS, TOKEN_SEP, TOKEN_START, TEMPERATURE, N_GENERATIONS
from tinygpt import TinyGPT

def load_model(weights_path, device):
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, DROPOUT)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def predict_aqi(model, device, params, pm25, pm10, co, no2):
    raw_vals = {"pm2_5": pm25, "pm10": pm10, "co": co, "no2": no2}
    scaled = []
    for f in INPUT_POLLUTANTS:
        v = int(round((raw_vals[f] - params[f]["min"]) / params[f]["range"] * 100))
        scaled.append(max(0, min(100, v)))
    tokens = [TOKEN_START] + scaled + [TOKEN_SEP]
    inp = torch.tensor([tokens], dtype=torch.long, device=device)

    all_preds = []
    with torch.no_grad():
        for _ in range(N_GENERATIONS):
            out = model.generate(inp, max_new_tokens=3, temperature=TEMPERATURE)
            aqi_tok = out[0, len(tokens)].item()
            if 0 <= aqi_tok <= 100:
                aqi = aqi_tok / 100.0 * params["aqi_index"]["range"] + params["aqi_index"]["min"]
                all_preds.append(aqi)

    if not all_preds:
        return None
    all_preds.sort()
    return all_preds[len(all_preds) // 2]  # median

def predict_batch(model, device, params, pm25_vals, pm10_vals, co_vals, no2_vals):
    results = []
    for i in range(len(pm25_vals)):
        aqi = predict_aqi(model, device, params, pm25_vals[i], pm10_vals[i], co_vals[i], no2_vals[i])
        results.append(aqi)
    return results
