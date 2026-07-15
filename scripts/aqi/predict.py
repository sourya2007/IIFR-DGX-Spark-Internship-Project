import json
import os
import torch

from config import VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, \
    DROPOUT, INPUT_POLLUTANTS, TOKEN_SEP, TOKEN_START, N_GENERATIONS
from tinygpt import TinyGPT

def load_model(weights_path, device):
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, DROPOUT)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def generate_greedy(model, input_ids, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = model.forward(input_ids[:, -model.max_seq_len:])
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids

def predict_aqi(model, device, params, pm25, pm10, co, no2):
    raw_vals = {"pm2_5": pm25, "pm10": pm10, "co": co, "no2": no2}
    scaled = []
    for f in INPUT_POLLUTANTS:
        v = int(round((raw_vals[f] - params[f]["min"]) / params[f]["range"] * 100))
        scaled.append(max(0, min(100, v)))
    tokens = [TOKEN_START] + scaled + [TOKEN_SEP]
    inp = torch.tensor([tokens], dtype=torch.long, device=device)
    gen_len = 5

    all_preds = []
    with torch.no_grad():
        for _ in range(N_GENERATIONS):
            out = model.generate(inp, max_new_tokens=gen_len, temperature=0.7)
            for offset in range(gen_len):
                tok = out[0, len(tokens) + offset].item()
                if 0 <= tok <= 100:
                    aqi = tok / 100.0 * params["aqi_index"]["range"] + params["aqi_index"]["min"]
                    all_preds.append(aqi)
                    break

    if not all_preds:
        out = generate_greedy(model, inp, gen_len)
        for offset in range(gen_len):
            tok = out[0, len(tokens) + offset].item()
            if 0 <= tok <= 100:
                aqi = tok / 100.0 * params["aqi_index"]["range"] + params["aqi_index"]["min"]
                return aqi
        return None

    all_preds.sort()
    return all_preds[len(all_preds) // 2]

def predict_batch(model, device, params, pm25_vals, pm10_vals, co_vals, no2_vals):
    return [predict_aqi(model, device, params, pm25_vals[i], pm10_vals[i],
                        co_vals[i], no2_vals[i]) for i in range(len(pm25_vals))]
