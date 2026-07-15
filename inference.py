import re
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, MAX_SEQ_LENGTH, TEMPERATURE, MAX_GEN_TOKENS, FEATURES

def load():
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, "aqi_lora_adapter")
    tokenizer = AutoTokenizer.from_pretrained("aqi_lora_adapter")
    model.eval()
    with open("data_prep_output/norm_params.json") as f:
        params = json.load(f)
    return model, tokenizer, params

def make_token(row, stats, features):
    parts = []
    for f in features:
        short = {"temp_c": "t", "humidity": "h", "pressure_mb": "p", "windspeed_kph": "w",
                 "pm2_5": "P25", "pm10": "P10", "co": "CO", "no2": "NO2", "aqi_index": "AQI"}[f]
        scaled = int(round((row[f] - stats[f]["min"]) / stats[f]["range"] * 100))
        parts.append(f"{short}:{scaled}")
    return " ".join(parts)

def predict(model, tokenizer, params, input_rows):
    inp = " ".join(make_token(r, params, FEATURES) for r in input_rows)
    prompt = f"Input: {inp}\nTarget:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_GEN_TOKENS,
                             temperature=TEMPERATURE, do_sample=True,
                             pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    gen_part = gen[len(prompt):].strip()
    aqi_m = re.search(r"AQI:(\d+)", gen_part)
    p25_m = re.search(r"P25:(\d+)", gen_part)
    aqi_scaled = int(aqi_m.group(1)) if aqi_m else None
    p25_scaled = int(p25_m.group(1)) if p25_m else None
    aqi = aqi_scaled / 100.0 * params["aqi_index"]["range"] + params["aqi_index"]["min"] if aqi_scaled else None
    pm = p25_scaled / 100.0 * params["pm2_5"]["range"] + params["pm2_5"]["min"] if p25_scaled else None
    return {"aqi_index": aqi, "pm2_5": pm, "raw_generated": gen_part}

if __name__ == "__main__":
    model, tokenizer, params = load()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print("Inference ready. Pass 12-row list of dicts with feature keys to predict().")
