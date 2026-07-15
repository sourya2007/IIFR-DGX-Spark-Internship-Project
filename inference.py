import re
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, MAX_SEQ_LENGTH, TEMPERATURE, MAX_GEN_TOKENS, FEATURES

FEATURE_SHORT = ["t", "h", "p", "w", "P25", "P10", "CO", "NO2", "AQI"]

def load():
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, "aqi_lora_adapter")
    tokenizer = AutoTokenizer.from_pretrained("aqi_lora_adapter")
    model.eval()
    with open("data_prep_output/norm_params.json") as f:
        params = json.load(f)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer, params

def make_token(row, stats):
    vals = []
    for f in FEATURES:
        scaled = int(round((row[f] - stats[f]["min"]) / stats[f]["range"] * 100))
        vals.append(str(scaled))
    return "_".join(vals)

def predict(model, tokenizer, params, input_rows):
    tokens = [make_token(r, params) for r in input_rows]
    prompt = f"In:{len(input_rows)} {' '.join(tokens)} | "
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_GEN_TOKENS, temperature=TEMPERATURE,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    gen_part = gen[len(prompt):].strip()
    nums = re.findall(r'\d+', gen_part)
    if len(nums) >= 9:
        nums = [int(n) for n in nums[-9:]]
        result = {}
        for i, f in enumerate(FEATURES):
            result[f] = nums[i] / 100.0 * params[f]["range"] + params[f]["min"]
        result["raw"] = gen_part
        return result
    return {"error": "could not parse", "raw": gen_part}

if __name__ == "__main__":
    model, tokenizer, params = load()
    print("Inference ready. Pass 12-row list of dicts with feature keys to predict().")
