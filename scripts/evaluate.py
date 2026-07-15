import os
import re
import json
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ADAPTER_DIR, MODEL_NAME, MAX_SEQ_LENGTH, TEMPERATURE, MAX_GEN_TOKENS, FEATURES

FEATURE_ORDER = list(zip(FEATURES, ["t", "h", "p", "w", "P25", "P10", "CO", "NO2", "AQI"]))

def parse_hour(text):
    nums = re.findall(r'\d+', text)
    if len(nums) >= 9:
        return [int(n) for n in nums[-9:]]
    return None

def denormalize_vals(scaled_list, params):
    return {f: scaled / 100.0 * params[f]["range"] + params[f]["min"]
            for f, scaled in zip(FEATURES, scaled_list)}

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

    def load_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    X_val = load_lines(os.path.join(data_dir, "X_val.txt"))
    y_val = load_lines(os.path.join(data_dir, "y_val.txt"))

    pred_list, actual_list = [], []
    print(f"  Running inference on {len(X_val)} validation samples...")

    for i, (inp, actual_str) in enumerate(zip(X_val, y_val)):
        prompt = f"{inp} | "
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_GEN_TOKENS, temperature=TEMPERATURE,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0], skip_special_tokens=True)
        gen_part = gen[len(prompt):].strip()

        pred_nums = parse_hour(gen_part)
        actual_nums = parse_hour(actual_str)
        if pred_nums and actual_nums:
            pred_list.append(denormalize_vals(pred_nums, params))
            actual_list.append(denormalize_vals(actual_nums, params))

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
