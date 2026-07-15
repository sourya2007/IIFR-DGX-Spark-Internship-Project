import os
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE, EPOCHS, FP16, \
    MAX_SEQ_LENGTH, PLOTS_DIR, MODEL_DIR, VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, DROPOUT

def run(model, data_dir, max_epochs=None):
    if max_epochs is None:
        max_epochs = EPOCHS
    print(f"[train] Loading data (epochs={max_epochs})...")
    X_train = torch.load(os.path.join(data_dir, "X_train.pt"))
    y_train = torch.load(os.path.join(data_dir, "y_train.pt"))
    X_val = torch.load(os.path.join(data_dir, "X_val.pt"))
    y_val = torch.load(os.path.join(data_dir, "y_val.pt"))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda") if FP16 and torch.cuda.is_available() else None

    all_losses, all_val_losses = [], []
    epoch_times, gpu_memory_log = [], []
    best_val_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        total_loss = 0
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", unit="batch", leave=False)
        for step, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            if scaler:
                with torch.amp.autocast("cuda"):
                    _, loss = model(inputs, labels)
                    loss = loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
            else:
                _, loss = model(inputs, labels)
                loss = loss / GRAD_ACCUM_STEPS
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
            pbar.set_postfix(loss=f"{loss.item()*GRAD_ACCUM_STEPS:.4f}")

        avg_loss = total_loss / len(train_loader)
        all_losses.append(avg_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"  Val", unit="batch", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                _, loss = model(inputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        all_val_losses.append(avg_val_loss)

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        if torch.cuda.is_available() and device.type == "cuda":
            gpu_memory_log.append(torch.cuda.memory_allocated(device) / 1e9)

        print(f"  >>> Epoch {epoch}: train_loss={avg_loss:.4f} val_loss={avg_val_loss:.4f} time={epoch_time:.1f}s")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / "best.pt")
            print(f"  >>> New best model saved to {MODEL_DIR / 'best.pt'}")

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
    model = setup()
    run(model, "data_prep_output")
