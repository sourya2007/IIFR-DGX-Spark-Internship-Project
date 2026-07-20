"""
DGX Spark Demonstration — TinyGPT Text Generation
===================================================
Trains TinyGPT at 3 scales (Small / Medium / Large) using mixed
precision, torch.compile, and maximum GPU throughput.  After training
it opens an interactive REPL where you type a prompt and watch all
three models generate side-by-side.

Usage:
    python dgx_demo.py                # full demo
    python dgx_demo.py --quick        # 5 epochs each (sanity check)
"""

import sys, time, re, urllib.request, argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from tinygpt import TinyGPT

# ═══════════════════════════════════════════════════════════════
#  Config — one dataclass per model variant
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    name: str
    d_model: int
    n_layer: int
    n_head: int
    d_ff: int
    max_seq_len: int
    epochs: int
    batch_size: int
    lr: float

CONFIGS = [
    ModelConfig("Small",   128,  6,  4,  512,   256,  80, 256, 3e-4),
    ModelConfig("Medium",  256,  8,  8, 1024,   512,  60, 128, 3e-4),
    ModelConfig("Large",   512, 12,  8, 2048,   512,  40,  64, 2e-4),
]

PROMPTS = [
    "ROMEO:",
    "The king",
    "It was a dark",
    "Once upon a time",
    "The meaning of life is",
]

DATASETS = {
    "shakespeare.txt": "https://www.gutenberg.org/cache/epub/1513/pg1513.txt",
    "sherlock.txt":    "https://www.gutenberg.org/cache/epub/244/pg244.txt",
}

# ═══════════════════════════════════════════════════════════════
#  ASCII tokenizer (shared by all models)
# ═══════════════════════════════════════════════════════════════

CHARS = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
C2I = {c: i + 4 for i, c in enumerate(CHARS)}
I2C = {i + 4: c for i, c in enumerate(CHARS)}
PAD, BOS, EOS, UNK = 0, 1, 2, 3
VOCAB_SIZE = len(CHARS) + 4

def encode(text):
    ids = [BOS]
    for c in text:
        ids.append(C2I.get(c, UNK))
    ids.append(EOS)
    return torch.tensor(ids, dtype=torch.long)

def encode_no_eos(text):
    ids = [BOS]
    for c in text:
        ids.append(C2I.get(c, UNK))
    return torch.tensor(ids, dtype=torch.long)

def decode(ids):
    out = []
    for i in ids:
        if i == EOS: break
        if i in (PAD, BOS): continue
        out.append(I2C.get(i, "?"))
    return "".join(out)

# ═══════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    def __init__(self, filepath, max_seq):
        self.max_seq = max_seq
        raw = Path(filepath).read_text(encoding="utf-8", errors="replace")
        raw = raw.encode("ascii", "replace").decode("ascii")
        ids = encode(raw)
        self.seqs = []
        for i in range(0, len(ids) - 1, max_seq - 1):
            seg = ids[i:i + max_seq]
            if len(seg) >= 20:
                self.seqs.append(seg)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = self.seqs[idx]
        pad = self.max_seq - len(x)
        x = torch.cat([x, torch.full((pad,), PAD, dtype=torch.long)])
        y = torch.cat([x[1:], torch.tensor([PAD])])
        y[x == PAD] = -100
        return x, y

# ═══════════════════════════════════════════════════════════════
#  System info
# ═══════════════════════════════════════════════════════════════

def system_info():
    lines = []
    lines.append(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        lines.append(f"GPU    {torch.cuda.get_device_name(i)}")
        free, total = torch.cuda.mem_get_info(i)
        lines.append(f"VRAM   {total / 1e9:.1f} GB total  ({free / 1e9:.1f} GB free)")
        lines.append(f"SM     {torch.cuda.get_device_capability(i)}")
        cc = torch.cuda.get_device_capability(i)
        if cc >= (8, 0):
            lines.append("Tensor Cores: FP16/BF16 ✓")
        else:
            lines.append("Tensor Cores: no")
        lines.append(f"BF16   {'✓' if torch.cuda.is_bf16_supported() else '✗'}")
        lines.append(f"torch.compile: {'✓' if hasattr(torch, 'compile') else '✗'}")
        if hasattr(torch, 'compile'):
            try:
                torch.compile(lambda x: x, backend="inductor")
                lines.append("  inductor backend available")
            except Exception:
                pass
    else:
        lines.append("GPU    NONE — running on CPU")
    return "\n".join(f"  {l}" for l in lines)

# ═══════════════════════════════════════════════════════════════
#  Dataset helpers
# ═══════════════════════════════════════════════════════════════

def _strip_gutenberg(text):
    for marker in ["*** START OF THE PROJECT GUTENBERG EBOOK",
                   "*** START OF THIS PROJECT GUTENBERG EBOOK"]:
        if marker in text:
            text = text.split(marker, 1)[1]
    for marker in ["*** END OF THE PROJECT GUTENBERG EBOOK",
                   "*** END OF THIS PROJECT GUTENBERG EBOOK"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()

def download_datasets():
    dest = BASE / "dgx_data"
    dest.mkdir(exist_ok=True)
    combined_path = dest / "combined.txt"
    if combined_path.exists():
        print(f"  combined dataset exists ({combined_path.stat().st_size // 1000}K chars)")
        return combined_path

    texts = []
    for name, url in DATASETS.items():
        path = dest / name
        if not path.exists():
            print(f"  downloading {name} ...", end=" ", flush=True)
            data = urllib.request.urlopen(url, timeout=30).read()
            text = data.decode("utf-8", errors="replace")
            text = _strip_gutenberg(text)
            path.write_text(text, encoding="utf-8")
            print(f"{len(text):,} chars")
        else:
            text = path.read_text(encoding="utf-8")
        texts.append(text)

    combined = "\n\n".join(texts)
    combined_path.write_text(combined, encoding="utf-8")
    print(f"  combined: {len(combined):,} chars -> {combined_path.name}")
    return combined_path

# ═══════════════════════════════════════════════════════════════
#  Training engine
# ═══════════════════════════════════════════════════════════════

def train_model(cfg, dataset_path, device, amp_dtype, quick=False):
    epochs = 5 if quick else cfg.epochs
    bs = cfg.batch_size
    if quick:
        bs = min(bs, 64)

    print(f"\n  [data] loading ...", end=" ", flush=True)
    ds = TextDataset(str(dataset_path), cfg.max_seq_len)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    print(f"{len(ds):,} sequences  |  batch={bs}")

    print(f"  [model] building ...", end=" ", flush=True)
    model = TinyGPT(VOCAB_SIZE, cfg.d_model, cfg.n_head, cfg.n_layer,
                    cfg.d_ff, cfg.max_seq_len, 0.1)
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    print(f" {n_params:,} params")

    if hasattr(torch, "compile"):
        print(f"  [model] torch.compile ...", end=" ", flush=True)
        try:
            model = torch.compile(model, mode="default")
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = GradScaler("cuda") if amp_dtype == torch.float16 else None

    losses = []
    tok_speeds = []
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t_ep = time.perf_counter()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            if scaler:
                with autocast("cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                with autocast("cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)

        elapsed = time.perf_counter() - t_ep
        toks_per_sec = n_batches * bs * cfg.max_seq_len / elapsed
        tok_speeds.append(toks_per_sec)

        total_elapsed = time.perf_counter() - t_start
        mem = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  E{epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
              f"{toks_per_sec:,.0f} tok/s  "
              f"GPU {mem:.1f}GB  "
              f"{total_elapsed:.0f}s")

    total_time = time.perf_counter() - t_start
    avg_tok_speed = sum(tok_speeds) / len(tok_speeds)

    return model, {
        "name": cfg.name,
        "params": n_params,
        "losses": losses,
        "final_loss": losses[-1],
        "avg_tok_speed": avg_tok_speed,
        "total_time": total_time,
        "max_gpu_mem": mem,
        "config": cfg,
    }

# ═══════════════════════════════════════════════════════════════
#  Generation
# ═══════════════════════════════════════════════════════════════

def generate_text(model, prompt, temp, length, device):
    ids = encode_no_eos(prompt).to(device)
    if len(ids) > model.max_seq_len:
        ids = torch.cat([ids[:1], ids[-(model.max_seq_len - 1):]])
    model.eval()
    with torch.no_grad():
        out = model.generate(ids.unsqueeze(0), length, temperature=temp)
    full = decode(out[0].tolist())
    idx = full.find(prompt)
    return full[idx + len(prompt):].strip() if idx >= 0 else full.strip()

# ═══════════════════════════════════════════════════════════════
#  Display helpers
# ═══════════════════════════════════════════════════════════════

def print_header(text):
    n = (70 - len(text)) // 2
    pad = "═" * max(n, 1)
    print(f"\n{pad}  {text}  {pad}")

def print_sep():
    print("─" * 70)

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DGX Spark TinyGPT Demo")
    parser.add_argument("--quick", action="store_true", help="5 epochs each")
    args = parser.parse_args()

    # ══════════════════════════════════════════════════════════
    #  Welcome
    # ══════════════════════════════════════════════════════════
    print_header("DGX Spark — TinyGPT Text Generation Demo")
    print_sep()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(system_info())
    print_sep()

    if device.type == "cpu":
        print("  ⚠  DGX Spark has a GPU — this will be slow on CPU!")
        print_sep()

    amp_dtype = torch.bfloat16 if (torch.cuda.is_available()
                                   and torch.cuda.is_bf16_supported()) else torch.float16

    # ══════════════════════════════════════════════════════════
    #  Data
    # ══════════════════════════════════════════════════════════
    print_header("Downloading Datasets")
    data_path = download_datasets()
    print_sep()

    # ══════════════════════════════════════════════════════════
    #  Train all model variants
    # ══════════════════════════════════════════════════════════
    trained = []
    for cfg in CONFIGS:
        print_header(f"Training {cfg.name}")
        print(f"  d_model={cfg.d_model}  layers={cfg.n_layer}  "
              f"heads={cfg.n_head}  context={cfg.max_seq_len}")
        print(f"  epochs={cfg.epochs}  batch={cfg.batch_size}  "
              f"lr={cfg.lr}  precision={str(amp_dtype).split('.')[-1]}")
        print_sep()

        t0 = time.perf_counter()
        model, stats = train_model(cfg, data_path, device, amp_dtype, quick=args.quick)
        stats["train_time"] = time.perf_counter() - t0

        print(f"\n  ▶  final loss: {stats['final_loss']:.4f}")
        print(f"  ▶  throughput: {stats['avg_tok_speed']:,.0f} tok/s")
        print(f"  ▶  GPU memory: {stats['max_gpu_mem']:.1f} GB")
        print_sep()

        # Quick sample right after training
        temp = 0.8
        for prompt in PROMPTS[:1]:
            text = generate_text(model, prompt, temp, 200, device)
            words = text.split()
            print(f"\n  prompt: \"{prompt}\"  (temp={temp})")
            print(f"  output: \"{' '.join(words[:40])}...\"")
        print_sep()

        trained.append((cfg.name, model, stats))

    # ══════════════════════════════════════════════════════════
    #  Comparison table
    # ══════════════════════════════════════════════════════════
    print_header("Performance Comparison")
    print(f"  {'Model':<10} {'Params':<10} {'Final Loss':<12} "
          f"{'Tok/s':<12} {'Time':<8} {'GPU Mem':<8}")
    print(f"  {'─'*8 :<10} {'─'*8 :<10} {'─'*10 :<12} "
          f"{'─'*8 :<12} {'─'*6 :<8} {'─'*6 :<8}")
    for name, _, s in trained:
        print(f"  {s['name']:<10} {s['params']:<10,} {s['final_loss']:<12.4f} "
              f"{s['avg_tok_speed']:<12,.0f} {s['total_time']:<8.1f}s "
              f"{s['max_gpu_mem']:<8.1f}GB")
    print_sep()

    # ══════════════════════════════════════════════════════════
    #  Side-by-side generation (fixed prompts)
    # ══════════════════════════════════════════════════════════
    print_header("Side-by-Side Generation")
    for prompt in PROMPTS:
        print(f"\n  Prompt: \"{prompt}\"  (temp=0.8)\n")
        for name, model, stats in trained:
            text = generate_text(model, prompt, 0.8, 200, device)
            words = text.split()
            print(f"  {name:<8} ({stats['params']//1000}K) → "
                  f"\"{' '.join(words[:30])}...\"")
        print()
    print_sep()

    # ══════════════════════════════════════════════════════════
    #  Interactive demo
    # ══════════════════════════════════════════════════════════
    print_header("Interactive Demo — Type a Prompt")
    print("  All 3 models generate from your prompt.")
    print("  T=0.5  → set temperature  (default: 0.8)")
    print("  <Enter>  → quit\n")

    temperature = 0.8
    while True:
        try:
            line = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            break
        if line.startswith("T="):
            try:
                temperature = float(line[2:])
                print(f"  → temperature = {temperature}")
            except ValueError:
                print("  → usage: T=0.8")
            continue

        for name, model, stats in trained:
            text = generate_text(model, line, temperature, 200, device)
            print(f"\n  {name:<8} ({stats['params']//1000}K): "
                  f"{text[:200]}")
        print()

    print_header("Demo Complete")
    print("  All models cleared from GPU memory.\n")


if __name__ == "__main__":
    main()
