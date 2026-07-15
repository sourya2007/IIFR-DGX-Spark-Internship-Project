"""
TinyGPT Text Generation Project — Single script for the assignment.

Usage:
    python run_project.py

What it does:
    1. Downloads Shakespeare + Sherlock Holmes datasets
    2. Runs a baseline experiment + 3 config changes
    3. For each: trains TinyGPT, records loss & time, saves model & loss plot
    4. Generates 5 text samples per experiment
    5. Prints a comparison table

Edit EXPERIMENTS below to customize. Edit PROMPTS to change generation prompts.
"""

import sys
import time
import json
import re
import urllib.request
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from tinygpt import TinyGPT

# ──────────────────────────────────────────────────────────────────────
# Configuration — edit these to match your assignment plan
# ──────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    # (name, file, context, layers, dmodel, heads, epochs, batch, lr, device)
    ("Baseline",           "shakespeare.txt", 256, 6, 128, 4, 200, 32, 3e-4, "auto"),
    ("Change 1: dataset",  "sherlock.txt",    256, 6, 128, 4, 200, 32, 3e-4, "auto"),
    ("Change 2: context",  "shakespeare.txt", 128, 6, 128, 4, 200, 32, 3e-4, "auto"),
    ("Change 3: layers",   "shakespeare.txt", 256, 4, 128, 4, 200, 32, 3e-4, "auto"),
    # Uncomment for CPU vs GPU (slow, use fewer epochs):
    # ("Change 4: CPU",   "shakespeare.txt", 256, 6, 128, 4, 50,  16, 3e-4, "cpu"),
]

PROMPTS = [
    "ROMEO:",
    "The king",
    "It was a dark",
    "Once upon a time",
    "The meaning of life is",
]

DATASETS = {
    "shakespeare.txt":
        "https://www.gutenberg.org/cache/epub/1513/pg1513.txt",
    "sherlock.txt":
        "https://www.gutenberg.org/cache/epub/244/pg244.txt",
}

# ──────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────

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
        if i == EOS:
            break
        if i in (PAD, BOS):
            continue
        out.append(I2C.get(i, "?"))
    return "".join(out)


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, filepath, max_seq):
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
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
        pad = MAX_SEQ - len(x)
        x = torch.cat([x, torch.full((pad,), PAD, dtype=torch.long)])
        y = torch.cat([x[1:], torch.tensor([PAD])])
        y[x == PAD] = -100
        return x, y


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_model(model, loader, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr)
    losses = []
    for epoch in range(epochs):
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:>3}/{epochs}  loss {avg:.4f}")
    return losses


# ──────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Download datasets
# ──────────────────────────────────────────────────────────────────────

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
    for name, url in DATASETS.items():
        path = BASE / name
        if path.exists():
            print(f"  {name} exists, skipping")
            continue
        print(f"  downloading {name} ...", end=" ", flush=True)
        try:
            data = urllib.request.urlopen(url, timeout=30).read()
            text = data.decode("utf-8", errors="replace")
            text = _strip_gutenberg(text)
            path.write_text(text, encoding="utf-8")
            print(f"{len(text)} chars")
        except Exception as e:
            print(f"FAILED: {e}")
            sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Save loss plot
# ──────────────────────────────────────────────────────────────────────

def save_loss_plot(losses, name, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss — {name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  (loss plot skipped: {e})")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs for testing")
    args = parser.parse_args()

    print("=" * 60)
    print("  TinyGPT — Text Generation Project")
    print("=" * 60)

    if args.quick:
        print("  QUICK MODE — reduced epochs\n")

    # Step 1: Download datasets
    print("\n[1] Downloading datasets ...")
    download_datasets()

    # Step 2: Run experiments
    print(f"\n[2] Running {len(EXPERIMENTS)} experiments ...\n")

    all_results = []
    output_dir = BASE / "output"
    output_dir.mkdir(exist_ok=True)

    for exp_name, filename, context, layers, dmodel, heads, epochs, batch, lr, dev in EXPERIMENTS:
        if args.quick:
            epochs = min(epochs, 20)
            batch = min(batch, 16)
        print(f"{'='*60}")
        print(f"  Experiment: {exp_name}")
        print(f"{'='*60}")
        print(f"  file={filename}  context={context}  layers={layers}  "
              f"d_model={dmodel}  heads={heads}")
        print(f"  epochs={epochs}  batch={batch}  lr={lr}  device={dev}")

        # Resolve device
        if dev == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(dev)

        # Global MAX_SEQ for TextDataset
        global MAX_SEQ
        MAX_SEQ = context

        # Load data
        data_path = BASE / filename
        if not data_path.exists():
            print(f"  [SKIP] {filename} not found")
            all_results.append((exp_name, "N/A", "N/A", []))
            continue

        print(f"\n  [data] loading ...")
        ds = TextDataset(str(data_path), context)
        loader = DataLoader(ds, batch_size=batch, shuffle=True)
        print(f"  [data] {len(ds)} sequences")

        # Build model
        d_ff = 4 * dmodel
        model = TinyGPT(VOCAB_SIZE, dmodel, heads, layers, d_ff, context, 0.1).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  [model] params: {total_params:,}")

        # Train
        print(f"  [train] starting ...")
        t0 = time.perf_counter()
        losses = train_model(model, loader, epochs, lr, device)
        train_time = time.perf_counter() - t0
        final_loss = losses[-1]
        print(f"  [train] final loss: {final_loss:.4f}  time: {train_time:.1f}s")

        # Save model
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", exp_name)
        model_path = output_dir / f"{safe_name}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "config": {"vocab_size": VOCAB_SIZE, "d_model": dmodel,
                       "n_head": heads, "n_layer": layers,
                       "d_ff": d_ff, "max_seq_len": context, "dropout": 0.1},
            "losses": losses,
            "name": exp_name,
        }, model_path)
        print(f"  [save] model -> {model_path.name}")

        # Save loss plot
        plot_path = output_dir / f"{safe_name}_loss.png"
        save_loss_plot(losses, exp_name, plot_path)
        print(f"  [save] loss plot -> {plot_path.name}")

        # Generate samples
        print(f"  [gen] generating {len(PROMPTS)} samples ...")
        samples = []
        for prompt in PROMPTS:
            for temp in [0.8]:
                text = generate_text(model, prompt, temp, 200, device)
                samples.append((prompt, temp, text))
                print(f"    prompt={repr(prompt)}  temp={temp}")
                print(f"    output={text[:120]}...")
                print()

        all_results.append((exp_name, final_loss, train_time, samples))

    # ──────────────────────────────────────────────────────────────────
    # Step 3: Results table
    # ──────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  RESULTS TABLE")
    print("=" * 60)
    print(f"{'Experiment':<25} {'Final Loss':<12} {'Time (s)':<10}")
    print("-" * 50)
    for name, loss, t, _ in all_results:
        loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
        time_str = f"{t:.1f}" if isinstance(t, float) else str(t)
        print(f"{name:<25} {loss_str:<12} {time_str:<10}")

    # ──────────────────────────────────────────────────────────────────
    # Step 4: Generated samples report
    # ──────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  GENERATED SAMPLES")
    print("=" * 60)
    for name, loss, t, samples in all_results:
        print(f"\n--- {name} (loss={loss:.4f}, time={t:.1f}s) ---" if isinstance(loss, float)
              else f"\n--- {name} ---")
        for prompt, temp, text in samples:
            print(f"\n  Prompt: {prompt}")
            print(f"  Temp:   {temp}")
            print(f"  Output: {text[:300]}")
            print()

    # ──────────────────────────────────────────────────────────────────
    # Step 5: Save full report
    # ──────────────────────────────────────────────────────────────────

    report = {"experiments": []}
    for name, loss, t, samples in all_results:
        report["experiments"].append({
            "name": name,
            "final_loss": loss,
            "train_time": t,
            "samples": [{"prompt": p, "temp": tmp, "text": txt} for p, tmp, txt in samples],
        })

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")

    print("\nDone. All outputs in:", output_dir)


if __name__ == "__main__":
    main()
