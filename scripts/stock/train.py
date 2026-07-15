import os
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tinygpt import TinyGPT
from .config import VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, \
    DROPOUT, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, \
    EARLY_STOP_PATIENCE, GRAD_CLIP, FP16, PLOTS_DIR, MODEL_DIR

def build_model():
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, DROPOUT)
    print(f"  TinyGPT params: {sum(p.numel() for p in model.parameters()):,}")
    return model

def run(model, data_dir, device):
    print(f"[stock_train] Loading data...")
    X_train = torch.load(os.path.join(data_dir, "X_train.pt"))
    y_train = torch.load(os.path.join(data_dir, "y_train.pt"))
    X_val = torch.load(os.path.join(data_dir, "X_val.pt"))
    y_val = torch.load(os.path.join(data_dir, "y_val.pt"))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda") if FP16 and device.type == "cuda" else None

    all_losses, all_val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()
        total_loss = 0
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"E{epoch}/{MAX_EPOCHS}", unit="batch", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            if scaler:
                with torch.amp.autocast("cuda"):
                    _, loss = model(inputs, labels)
                scaler.scale(loss).backward()
            else:
                _, loss = model(inputs, labels)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        all_losses.append(avg_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, loss = model(inputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        all_val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        elapsed = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(f"  E{epoch:2d}: train={avg_loss:.4f} val={avg_val_loss:.4f} time={elapsed:.1f}s lr={lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(all_losses)+1), all_losses, label="Train")
    plt.plot(range(1, len(all_val_losses)+1), all_val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "stock_loss_curve.png", dpi=150)
    plt.close()
    print(f"  Loss plot saved")
    return {"train_losses": all_losses, "val_losses": all_val_losses}

def run_retrain(model, data_dir, device, max_epochs=10):
    print(f"[retrain] Fine-tuning on new data...")
    return run(model, data_dir, device)
