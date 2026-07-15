import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE, EPOCHS, FP16, MAX_SEQ_LENGTH, PLOTS_DIR, ADAPTER_DIR

class TextDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.input_ids, self.labels = [], []
        target_prefix_ids = tokenizer("Target:", add_special_tokens=False)["input_ids"]
        for inp, tgt in zip(texts, targets):
            txt = f"Input: {inp} Target: {tgt}"
            enc = tokenizer(txt, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"][0]
            labels = input_ids.clone()
            for j in range(len(input_ids) - len(target_prefix_ids)):
                if (input_ids[j:j+len(target_prefix_ids)] == torch.tensor(target_prefix_ids)).all():
                    labels[:j] = -100
                    break
            self.input_ids.append(input_ids)
            self.labels.append(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

def run(model, tokenizer, data_dir, max_epochs=None):
    if max_epochs is None:
        max_epochs = EPOCHS
    print(f"[train] Loading data (epochs={max_epochs})...")
    X_train = np.loadtxt(os.path.join(data_dir, "X_train.txt"), dtype=str, encoding="utf-8").tolist()
    y_train = np.loadtxt(os.path.join(data_dir, "y_train.txt"), dtype=str, encoding="utf-8").tolist()
    X_val = np.loadtxt(os.path.join(data_dir, "X_val.txt"), dtype=str, encoding="utf-8").tolist()
    y_val = np.loadtxt(os.path.join(data_dir, "y_val.txt"), dtype=str, encoding="utf-8").tolist()

    if isinstance(X_train, str):
        X_train, y_train, X_val, y_val = [X_train], [y_train], [X_val], [y_val]

    train_ds = TextDataset(X_train, y_train, tokenizer, MAX_SEQ_LENGTH)
    val_ds = TextDataset(X_val, y_val, tokenizer, MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_steps = max_epochs * len(train_loader) // GRAD_ACCUM_STEPS
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=0, num_training_steps=num_steps)
    scaler = torch.cuda.amp.GradScaler() if FP16 and torch.cuda.is_available() else None

    all_losses, all_val_losses = [], []
    epoch_times, gpu_memory_log = [], []
    best_val_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        total_loss = 0
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 50 == 0:
                print(f"  E{epoch} S{step} loss={loss.item()*GRAD_ACCUM_STEPS:.4f}")

        avg_loss = total_loss / len(train_loader)
        all_losses.append(avg_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        all_val_losses.append(avg_val_loss)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        if torch.cuda.is_available() and device.type == "cuda":
            gpu_memory_log.append(torch.cuda.memory_allocated(device) / 1e9)
        print(f"  >>> Epoch {epoch}: train_loss={avg_loss:.4f} val_loss={avg_val_loss:.4f} time={epoch_time:.1f}s")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ADAPTER_DIR)
            tokenizer.save_pretrained(ADAPTER_DIR)
            print(f"  >>> New best model saved to {ADAPTER_DIR}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(all_losses)+1), all_losses, label="Train Loss")
    plt.plot(range(1, len(all_val_losses)+1), all_val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "loss_curve.png", dpi=150)
    plt.close()
    print(f"  Loss plot saved to {PLOTS_DIR / 'loss_curve.png'}")
    return {"train_losses": all_losses, "val_losses": all_val_losses,
            "epoch_times": epoch_times, "gpu_memory": gpu_memory_log}

if __name__ == "__main__":
    from model_setup import setup
    model, tokenizer = setup()
    run(model, tokenizer, "data_prep_output")
