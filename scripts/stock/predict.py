import numpy as np
import torch

from .config import CONTEXT, TOKEN_START, TOKEN_SEP, N_GENERATIONS, TEMPERATURE

def predict_next(model, device, context_prices):
    lo, hi = context_prices.min(), context_prices.max()
    rng = hi - lo if hi != lo else 1.0
    norm = [int(round((v - lo) / rng * 100)) for v in context_prices]
    norm = [max(0, min(100, v)) for v in norm]

    tokens = [TOKEN_START] + norm + [TOKEN_SEP]
    inp = torch.tensor([tokens], dtype=torch.long, device=device)

    all_preds = []
    with torch.no_grad():
        for _ in range(N_GENERATIONS):
            out = model.generate(inp, max_new_tokens=3, temperature=TEMPERATURE)
            for offset in range(3):
                tok = out[0, len(tokens) + offset].item()
                if 0 <= tok <= 100:
                    price = tok / 100.0 * rng + lo
                    all_preds.append(price)
                    break

    if not all_preds:
        out = generate_greedy(model, inp, 3)
        for offset in range(3):
            tok = out[0, len(tokens) + offset].item()
            if 0 <= tok <= 100:
                return tok / 100.0 * rng + lo
        return context_prices[-1]

    return float(np.median(sorted(all_preds)))

def generate_greedy(model, input_ids, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = model.forward(input_ids[:, -model.max_seq_len:])
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids
