import json
import torch
from tinygpt import TinyGPT
from config import VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, \
    DROPOUT, N_FEATURES, FEATURES, TOKEN_START, TOKEN_SEP, MODEL_DIR

def load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, DROPOUT)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    with open("data_prep_output/norm_params.json") as f:
        params = json.load(f)
    return model, params, device

def predict(model, params, device, input_rows):
    stats = params
    tokens = [TOKEN_START]
    for row in input_rows:
        for f in FEATURES:
            scaled = int(round((row[f] - stats[f]["min"]) / stats[f]["range"] * 100))
            tokens.append(scaled)
    inp = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=N_FEATURES + 2, temperature=1.0)
    gen = out[0, len(tokens):].tolist()
    gen = [t for t in gen if t < 100]
    result = {}
    for i, f in enumerate(FEATURES):
        if i < len(gen):
            result[f] = gen[i] / 100.0 * stats[f]["range"] + stats[f]["min"]
        else:
            result[f] = None
    return result

if __name__ == "__main__":
    model, params, device = load()
    print("Ready. Call predict(model, params, device, [12 dicts with feature keys])")
