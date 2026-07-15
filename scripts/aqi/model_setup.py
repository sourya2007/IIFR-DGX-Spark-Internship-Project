import torch

from tinygpt import TinyGPT
from config import VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, DROPOUT

def setup():
    print("[model_setup] Building TinyGPT from scratch...")
    model = TinyGPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params:,}")

    torch.set_float32_matmul_precision("high")
    return model

if __name__ == "__main__":
    m = setup()
    print(m)
