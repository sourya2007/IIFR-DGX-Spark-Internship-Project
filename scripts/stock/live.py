import os
import json
import time
import threading
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import deque

from .config import CONTEXT, UPDATE_INTERVAL, RETRAIN_EVERY, LIVE_INTERVAL, \
    VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, DROPOUT, \
    BATCH_SIZE, MODEL_DIR, DATA_OUT_DIR
from .fetch import fetch_live
from .predict import predict_next
from .train import build_model, run_retrain

_model_lock = threading.Lock()
_current_model = None
_current_device = None

def _retrain_worker(data_df, data_dir, device):
    global _current_model
    print("\n[retrain] Starting background retraining...")
    new_model = build_model().to(device)
    run_retrain(new_model, data_dir, device)
    with _model_lock:
        _current_model = new_model
        _current_device = device
    print("[retrain] Model updated.")

def _seed_graph(prices, model, device, actuals, preds, max_steps=120):
    n = min(len(prices) - CONTEXT, max_steps)
    if n < 1:
        return
    for i in range(n):
        ctx = prices[i:i+CONTEXT]
        preds.append(predict_next(model, device, ctx))
        actuals.append(prices[i+CONTEXT])

def _draw(fig, ax, actuals, preds, symbol, current_price, predicted):
    ax.clear()
    x_act = list(range(len(actuals)))
    ax.plot(x_act, list(actuals), "b-", lw=2, label="Actual")
    if len(preds) > 1:
        x_pred = list(range(len(preds)))
        ax.plot(x_pred, list(preds), "r--", lw=1.5, alpha=0.7, label="Predicted")
    ax.scatter(len(actuals) - 1, predicted, c="r", s=150, marker="*",
               edgecolors="black", zorder=5)
    ax.axvline(x=len(actuals) - 1, color="gray", linestyle=":", alpha=0.4)
    change = ((predicted - current_price) / current_price) * 100
    ax.set_title(f"{symbol} — Live: ${current_price:.2f}  "
                 f"Predicted: ${predicted:.2f}  ({change:+.2f}%)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Time steps (minutes)")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

def run_live(initial_model, params, symbol, data_dir, device, initial_df=None):
    global _current_model, _current_device
    _current_model = initial_model
    _current_device = device

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title(f"{symbol} - Live Stock Prediction")

    actual_prices = deque(maxlen=120)
    pred_prices = deque(maxlen=120)
    pred_count = 0
    accumulated_data = None if initial_df is None else initial_df.copy()

    print(f"\n[Live] Opening graph for {symbol}. Updates every {UPDATE_INTERVAL}s.")
    print("[Live] Close the graph window to stop.\n")

    # Seed graph with initial training data so it shows immediately
    if initial_df is not None and len(initial_df) > CONTEXT:
        prices = initial_df["Close"].values.astype(float)
        _seed_graph(prices, initial_model, device, actual_prices, pred_prices)
        pred_count = len(actual_prices)
        predicted = pred_prices[-1]
        current_price = actual_prices[-1]
        _draw(fig, ax, actual_prices, pred_prices, symbol, current_price, predicted)
        plt.draw()
        plt.pause(0.5)
        print(f"  Seeded with {pred_count} predictions from training data.")

    try:
        while plt.fignum_exists(fig.number):
            df = fetch_live(symbol, LIVE_INTERVAL)
            if df is None or len(df) < 1:
                plt.pause(UPDATE_INTERVAL)
                continue

            prices = df["Close"].values.astype(float)
            context = prices[-min(len(prices), CONTEXT):]
            current_price = context[-1]

            with _model_lock:
                model = _current_model
                dev = _current_device

            predicted = predict_next(model, dev, context)

            actual_prices.append(current_price)
            pred_prices.append(predicted)
            pred_count += 1

            if accumulated_data is None:
                accumulated_data = df.copy()
            else:
                new_rows = df[~df["datetime"].isin(accumulated_data["datetime"])]
                accumulated_data = pd.concat([accumulated_data, new_rows]).drop_duplicates(subset=["datetime"]).sort_values("datetime")

            _draw(fig, ax, actual_prices, pred_prices, symbol, current_price, predicted)
            plt.draw()
            plt.pause(UPDATE_INTERVAL)

            if pred_count % RETRAIN_EVERY == 0 and accumulated_data is not None:
                from .prep import prepare_data
                X_new, y_new, norm_list, old_p = prepare_data(accumulated_data)
                split = int(len(X_new) * 0.8)
                torch.save(X_new[:split], DATA_OUT_DIR / "X_train.pt")
                torch.save(y_new[:split], DATA_OUT_DIR / "y_train.pt")
                torch.save(X_new[split:], DATA_OUT_DIR / "X_val.pt")
                torch.save(y_new[split:], DATA_OUT_DIR / "y_val.pt")
                t = threading.Thread(target=_retrain_worker,
                                     args=(accumulated_data, str(DATA_OUT_DIR), device),
                                     daemon=True)
                t.start()

    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.close(fig)
