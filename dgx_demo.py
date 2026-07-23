"""
DGX Spark — TinyGPT Word-Level Language Model
==============================================
Downloads 5 classic books, builds a word-level vocabulary,
trains 3 TinyGPT models (35M / 75M / 180M params) using
BF16 mixed precision + torch.compile, then launches a
Gradio web UI for interactive text generation and live AQI.

Usage:
    python dgx_demo.py                   # full training + web UI
    python dgx_demo.py --quick           # 5 epochs each (test)
    python dgx_demo.py --train-only      # train & save, no web UI
    python dgx_demo.py --web-only        # skip training, load saved models
"""

import sys, time, re, urllib.request, argparse, json, math
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from tinygpt import TinyGPT
from scripts.dgx.text_utils import (
    build_vocab, vocab_size, encode, encode_no_eos, decode,
    save_vocab, load_vocab, TextDataset, search_passages,
)

# ═══════════════════════════════════════════════════════════════
#  Configuration — model sizes scaled for DGX Spark
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
    ModelConfig("Medium",  384,  8,  6, 1536,  512,  80, 128, 3e-4),
    ModelConfig("Large",   512, 12,  8, 2048,  768,  60,  64, 2e-4),
    ModelConfig("X-Large", 768, 16,  8, 3072, 1024,  40,  32, 1e-4),
]

BOOKS = {
    "shakespeare.txt": "https://www.gutenberg.org/cache/epub/1513/pg1513.txt",
    "sherlock.txt":    "https://www.gutenberg.org/cache/epub/244/pg244.txt",
    "pride.txt":       "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "tale_of_two.txt": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "alice.txt":       "https://www.gutenberg.org/cache/epub/11/pg11.txt",
}

PROMPTS = [
    "ROMEO:",
    "The king",
    "It was a dark",
    "Once upon a time",
    "The meaning of life is",
]

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
        cc = torch.cuda.get_device_capability(i)
        lines.append(f"SM     {cc}")
        if cc >= (8, 0):
            lines.append("Tensor Cores: FP16/BF16 ✓")
        if torch.cuda.is_bf16_supported():
            lines.append("BF16   ✓")
        if hasattr(torch, "compile"):
            try:
                torch.compile(lambda x: x, backend="inductor")
                lines.append("torch.compile: ✓ (inductor)")
            except Exception:
                lines.append("torch.compile: available (backend?)")
    else:
        lines.append("GPU    NONE — running on CPU")
    return "\n".join(f"  {l}" for l in lines)

# ═══════════════════════════════════════════════════════════════
#  Dataset helpers
# ═══════════════════════════════════════════════════════════════

def _strip_gutenberg(text):
    for m in ["*** START OF THE PROJECT GUTENBERG EBOOK",
              "*** START OF THIS PROJECT GUTENBERG EBOOK"]:
        if m in text:
            text = text.split(m, 1)[1]
    for m in ["*** END OF THE PROJECT GUTENBERG EBOOK",
              "*** END OF THIS PROJECT GUTENBERG EBOOK"]:
        if m in text:
            text = text.split(m, 1)[0]
    return text.strip()


def download_datasets():
    dest = BASE / "dgx_data"
    dest.mkdir(exist_ok=True)
    manifest = dest / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print(f"  {len(m)} books cached ({m.get('total_chars', 0):,} chars)")
        return dest

    info = {}
    for name, url in BOOKS.items():
        path = dest / name
        if not path.exists():
            print(f"  downloading {name} ...", end=" ", flush=True)
            data = urllib.request.urlopen(url, timeout=30).read()
            text = data.decode("utf-8", errors="replace")
            text = _strip_gutenberg(text)
            path.write_text(text, encoding="utf-8")
            chars = len(text)
            print(f"{chars:,} chars")
        else:
            chars = len(path.read_text(encoding="utf-8"))
        info[name] = {"path": str(path), "chars": chars}
    info["total_chars"] = sum(v["chars"] for v in info.values() if isinstance(v, dict))
    manifest.write_text(json.dumps(info, indent=2))
    print(f"  total: {info['total_chars']:,} chars across {len(BOOKS)} books")
    return dest

# ═══════════════════════════════════════════════════════════════
#  Vocabulary
# ═══════════════════════════════════════════════════════════════

def build_vocabulary(data_dir, min_freq=2):
    print(f"\n  scanning text (min_freq={min_freq}) ...", end=" ", flush=True)
    texts = []
    for f in sorted(data_dir.iterdir()):
        if f.suffix == ".txt" and f.name != "combined.txt":
            texts.append(f.read_text(encoding="utf-8", errors="replace"))
    word2idx, idx2word = build_vocab(texts, min_freq=min_freq)
    vs = vocab_size(word2idx)
    print(f"vocab: {vs:,} tokens")
    return word2idx, idx2word

# ═══════════════════════════════════════════════════════════════
#  Training engine
# ═══════════════════════════════════════════════════════════════

def train_model(cfg, data_dir, word2idx, idx2word, device, amp_dtype, quick=False):
    epochs = 5 if quick else cfg.epochs
    bs = cfg.batch_size
    if quick:
        bs = min(bs, 64)

    # Build dataset from all .txt files
    all_seqs = []
    for f in sorted(data_dir.iterdir()):
        if f.suffix == ".txt" and f.name != "manifest.json":
            ds = TextDataset(str(f), cfg.max_seq_len, word2idx)
            if len(ds) > 0:
                all_seqs.extend(ds.seqs)
    if not all_seqs:
        raise RuntimeError("No training data found")

    class ConcatDataset(torch.utils.data.Dataset):
        def __init__(self, seqs, max_seq):
            self.seqs = seqs
            self.max_seq = max_seq
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, idx):
            x = self.seqs[idx]
            pad = self.max_seq - len(x)
            x = torch.cat([x, torch.full((pad,), 0, dtype=torch.long)])
            y = torch.cat([x[1:], torch.tensor([0])])
            y[x == 0] = -100
            return x, y

    ds = ConcatDataset(all_seqs, cfg.max_seq_len)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    vs = vocab_size(word2idx)
    print(f"  [data] {len(ds):,} sequences  |  batch={bs}  |  {len(loader):,} steps/epoch")

    print(f"  [model] building ...", end=" ", flush=True)
    model = TinyGPT(vs, cfg.d_model, cfg.n_head, cfg.n_layer,
                    cfg.d_ff, cfg.max_seq_len, 0.1)
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    print(f" {n_params:,} params")

    if hasattr(torch, "compile") and not quick:
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
              f"{toks_per_sec:,.0f} tok/s  GPU {mem:.1f}GB  {total_elapsed:.0f}s")

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

def generate_text(model, prompt, temp, length, word2idx, idx2word, device, context=None):
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    ids = encode_no_eos(full_prompt, word2idx).to(device)
    if len(ids) > model.max_seq_len:
        ids = torch.cat([ids[:1], ids[-(model.max_seq_len - 1):]])
    model.eval()
    with torch.no_grad():
        out = model.generate(ids.unsqueeze(0), length, temperature=temp)
    result = decode(out[0].tolist(), idx2word)
    idx = result.find(prompt)
    if idx >= 0:
        result = result[idx + len(prompt):].strip()
    elif context and result.find(context.split("\n")[0]) >= 0:
        idx = result.find(context.split("\n")[0])
        result = result[idx + len(context):].strip()
    return result

# ═══════════════════════════════════════════════════════════════
#  Save / Load models
# ═══════════════════════════════════════════════════════════════

MODELS_DIR = BASE / "dgx_models"

def save_trained(model, stats, word2idx, idx2word, device):
    MODELS_DIR.mkdir(exist_ok=True)
    name = stats["name"]
    model_path = MODELS_DIR / f"{name.lower()}.pt"
    vocab_path = MODELS_DIR / "vocab.pt"
    config_path = MODELS_DIR / "configs.json"

    if hasattr(model, "_orig_mod"):
        state = model._orig_mod.state_dict()
    else:
        state = model.state_dict()
    torch.save(state, model_path)
    save_vocab(word2idx, idx2word, vocab_path)

    configs = {}
    if config_path.exists():
        configs = json.loads(config_path.read_text())
    configs[name] = {
        "d_model": stats["config"].d_model,
        "n_layer": stats["config"].n_layer,
        "n_head": stats["config"].n_head,
        "d_ff": stats["config"].d_ff,
        "max_seq_len": stats["config"].max_seq_len,
        "vocab_size": vocab_size(word2idx),
        "params": stats["params"],
        "final_loss": stats["final_loss"],
    }
    config_path.write_text(json.dumps(configs, indent=2))
    print(f"  saved {name} -> {model_path.name} / vocab.pt")


def load_trained(device):
    if not MODELS_DIR.exists():
        return None, None, None, None
    vocab_path = MODELS_DIR / "vocab.pt"
    config_path = MODELS_DIR / "configs.json"
    if not vocab_path.exists() or not config_path.exists():
        return None, None, None, None

    word2idx, idx2word = load_vocab(vocab_path)
    configs = json.loads(config_path.read_text())
    vs = vocab_size(word2idx)
    trained = []

    for name, cfg in configs.items():
        model_path = MODELS_DIR / f"{name.lower()}.pt"
        if not model_path.exists():
            continue
        model = TinyGPT(vs, cfg["d_model"], cfg["n_head"], cfg["n_layer"],
                        cfg["d_ff"], cfg["max_seq_len"], 0.1)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        trained.append((name, model, {
            "name": name,
            "params": cfg["params"],
            "final_loss": cfg["final_loss"],
        }))

    if not trained:
        return None, None, None, None
    return trained, word2idx, idx2word, configs

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
    parser = argparse.ArgumentParser(description="DGX Spark TinyGPT — Word-Level Demo")
    parser.add_argument("--quick", action="store_true", help="5 epochs each")
    parser.add_argument("--train-only", action="store_true", help="train & exit (no web UI)")
    parser.add_argument("--web-only", action="store_true", help="skip training, load saved models")
    args = parser.parse_args()

    print_header("DGX Spark — TinyGPT Word-Level Demo")
    print_sep()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(system_info())
    print_sep()

    if device.type == "cpu":
        print("  ⚠ DGX Spark has a GPU — this will be slow on CPU! Use --quick for testing.\n")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available()
                                   and torch.cuda.is_bf16_supported()) else torch.float16

    data_dir = MODELS_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "manifest.json"

    # ── Download ─────────────────────────────────────────────
    if not manifest.exists() or not args.web_only:
        print_header("Downloading Datasets")
        data_dir = download_datasets()
        print_sep()

    # ── Vocabulary ───────────────────────────────────────────
    trained = None
    word2idx = idx2word = None

    if args.web_only:
        print("  Loading saved models from dgx_models/ ...")
        trained, word2idx, idx2word, _ = load_trained(device)
        if trained is None:
            print("  No saved models found. Run without --web-only first.\n")
            sys.exit(1)
        print(f"  Loaded {len(trained)} models")
    else:
        print_header("Building Vocabulary")
        word2idx, idx2word = build_vocabulary(data_dir, min_freq=2)
        vs = vocab_size(word2idx)
        print(f"  Vocab: {vs:,} words, vectors: {vs*384:,} tokens in data")
        print_sep()

        # ── Train each config ────────────────────────────────
        trained = []
        all_configs = CONFIGS
        if args.quick:
            all_configs = CONFIGS[:2]
        for cfg in all_configs:
            print_header(f"Training {cfg.name}")
            print(f"  d_model={cfg.d_model}  layers={cfg.n_layer}  "
                  f"heads={cfg.n_head}  context={cfg.max_seq_len}")
            print(f"  epochs={cfg.epochs}  batch={cfg.batch_size}  "
                  f"lr={cfg.lr}  precision={str(amp_dtype).split('.')[-1]}")
            print_sep()
            t0 = time.perf_counter()
            model, stats = train_model(cfg, data_dir, word2idx, idx2word,
                                       device, amp_dtype, quick=args.quick)
            stats["train_time"] = time.perf_counter() - t0
            print(f"\n  ▶ final loss: {stats['final_loss']:.4f}")
            print(f"  ▶ throughput: {stats['avg_tok_speed']:,.0f} tok/s")
            print(f"  ▶ GPU memory: {stats['max_gpu_mem']:.1f} GB")
            print_sep()
            save_trained(model, stats, word2idx, idx2word, device)

            temp = 0.8
            text = generate_text(model, PROMPTS[0], temp, 100,
                                 word2idx, idx2word, device)
            print(f"\n  prompt: \"{PROMPTS[0]}\"  (temp={temp})")
            print(f"  output: {text[:200]}\n")
            trained.append((cfg.name, model, stats))
            print_sep()

        # ── Comparison ───────────────────────────────────────
        print_header("Performance Comparison")
        header = f"  {'Model':<10} {'Params':<10} {'Loss':<10} {'Tok/s':<12} {'Time':<8} {'GPU':<8}"
        print(header)
        print("  " + "─" * 58)
        for name, _, s in trained:
            mem_str = f"{s['max_gpu_mem']:.1f}GB"
            print(f"  {s['name']:<10} {s['params']:<10,} {s['final_loss']:<10.4f} "
                  f"{s['avg_tok_speed']:<12,.0f} {s['total_time']:<8.1f}s {mem_str:<8}")
        print_sep()

        # ── Side-by-side generation ──────────────────────────
        print_header("Side-by-Side Generation")
        for prompt in PROMPTS[:3]:
            print(f"\n  Prompt: \"{prompt}\"\n")
            for name, model, stats in trained:
                text = generate_text(model, prompt, 0.8, 100,
                                     word2idx, idx2word, device)
                print(f"  {name:<8} ({stats['params']//1000_000}M) → {text[:150]}")
            print()
        print_sep()

    if args.train_only:
        print("  Training complete. Models saved to dgx_models/")
        return

    # ── Launch Web UI ────────────────────────────────────────
    print_header("Launching Web UI")
    try:
        import gradio as gr
    except ImportError:
        print("  gradio not installed. Run: pip install gradio")
        print("  Then launch manually: python -c 'import app; app.launch()'")
        return

    sys.path.insert(0, str(BASE))
    from app import create_ui
    print("  Starting Gradio server on http://localhost:7860")
    print("  Press Ctrl+C to stop.\n")
    demo = create_ui(trained, word2idx, idx2word, str(data_dir), device, amp_dtype)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
