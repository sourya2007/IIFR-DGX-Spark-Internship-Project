import os
import re
import json
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ADAPTER_DIR, MODEL_NAME, MAX_SEQ_LENGTH, TEMPERATURE, MAX_GEN_TOKENS, FEATURES, TARGETS

def parse_generated(text):
    aqi = re.search(r"AQI:(\d+)", text)
    p25 = re.search(r"P25:(\d+)", text)
    return {"aqi_index": int(aqi.group(1)) if aqi else None,
            "pm2_5": int(p25.group(1)) if p25 else None}

def denormalize_val(scaled, params, f):
    s = params[f]
    if scaled is None:
        return None
    return scaled / 100.0 * s["range"] + s["min"]

def run(data_dir):
    print("[evaluate] Loading model and adapter...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with open(os.path.join(data_dir, "norm_params.json")) as f:
        params = json.load(f)

    X_val = np.loadtxt(os.path.join(data_dir, "X_val.txt"), dtype=str, encoding="utf-8").tolist()
    y_val = np.loadtxt(os.path.join(data_dir, "y_val.txt"), dtype=str, encoding="utf-8").tolist()

    preds_list, actuals_list = [], []
    print(f"  Running inference on {len(X_val)} validation samples...")

    for i, (inp, actual_str) in enumerate(zip(X_val, y_val)):
        prompt = f"Input: {inp}\nTarget:"
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_GEN_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0], skip_special_tokens=True)
        # strip the input prompt
        gen_part = generated[len(prompt):].strip()
        pred = parse_generated(gen_part)

        actual = parse_generated(actual_str)
        if pred["aqi_index"] is not None and actual["aqi_index"] is not None:
            preds_list.append({
                "aqi_index": denormalize_val(pred["aqi_index"], params, "aqi_index"),
                "pm2_5": denormalize_val(pred["pm2_5"], params, "pm2_5"),
            })
            actuals_list.append({
                "aqi_index": denormalize_val(actual["aqi_index"], params, "aqi_index"),
                "pm2_5": denormalize_val(actual["pm2_5"], params, "pm2_5"),
            })

    pred_aqi = np.array([p["aqi_index"] for p in preds_list])
    act_aqi = np.array([a["aqi_index"] for a in actuals_list])
    pred_pm = np.array([p["pm2_5"] for p in preds_list])
    act_pm = np.array([a["pm2_5"] for a in actuals_list])

    mae_aqi = np.mean(np.abs(pred_aqi - act_aqi))
    rmse_aqi = np.sqrt(np.mean((pred_aqi - act_aqi) ** 2))
    mae_pm = np.mean(np.abs(pred_pm - act_pm))
    rmse_pm = np.sqrt(np.mean((pred_pm - act_pm) ** 2))

    print(f"\n  === Validation Metrics ===")
    print(f"  {'Metric':<15} {'AQI':<12} {'PM2.5':<12}")
    print(f"  {'-'*39}")
    print(f"  {'MAE':<15} {mae_aqi:<12.2f} {mae_pm:<12.2f}")
    print(f"  {'RMSE':<15} {rmse_aqi:<12.2f} {rmse_pm:<12.2f}")

    print(f"\n  === Sample Predictions (first 5) ===")
    print(f"  {'#':<4} {'Actual AQI':<12} {'Pred AQI':<12} {'Actual PM2.5':<14} {'Pred PM2.5':<12}")
    print(f"  {'-'*54}")
    for i in range(min(5, len(preds_list))):
        print(f"  {i:<4} {act_aqi[i]:<12.1f} {pred_aqi[i]:<12.1f} {act_pm[i]:<14.1f} {pred_pm[i]:<12.1f}")

    return {"mae_aqi": mae_aqi, "rmse_aqi": rmse_aqi, "mae_pm": mae_pm, "rmse_pm": rmse_pm}

if __name__ == "__main__":
    run("data_prep_output")
