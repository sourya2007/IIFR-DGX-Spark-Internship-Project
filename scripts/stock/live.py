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
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
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
_chart_style = "candle"

def _draw_candle(ax, x, o, h, l, c, w=0.6):
    color = "#26a69a" if c >= o else "#ef5350"
    ax.add_line(Line2D([x, x], [l, h], color=color, lw=1))
    body = Rectangle((x - w/2, min(o, c)), w, abs(c - o) or 0.01,
                     facecolor=color, edgecolor=color)
    ax.add_patch(body)

def _draw_ohlc(ax, x, o, h, l, c, w=0.6):
    color = "#26a69a" if c >= o else "#ef5350"
    ax.add_line(Line2D([x, x], [l, h], color=color, lw=1))
    ax.add_line(Line2D([x - w/2, x], [o, o], color=color, lw=1.5))
    ax.add_line(Line2D([x, x + w/2], [c, c], color=color, lw=1.5))

def _on_key(event):
    global _chart_style
    if event.key == 'c':
        _chart_style = "candle"
    elif event.key == 'o':
        _chart_style = "ohlc"
    elif event.key == 'l':
        _chart_style = "line"

def _draw(fig, ax1, ax2, data, actuals, preds, symbol, style, last_pred):
    ax1.clear()
    ax2.clear()
    n = len(data)
    if n < 1:
        return
    o = data["Open"].values.astype(float)
    h = data["High"].values.astype(float)
    l = data["Low"].values.astype(float)
    c = data["Close"].values.astype(float)
    v = data["Volume"].values.astype(float)
    x = range(n)
    if style == "line":
        ax1.plot(x, c, "b-", lw=2, label="Price")
    elif style == "ohlc":
        for i in range(n):
            _draw_ohlc(ax1, i, o[i], h[i], l[i], c[i])
    else:
        for i in range(n):
            _draw_candle(ax1, i, o[i], h[i], l[i], c[i])

    if preds:
        n_pred = len(preds)
        px = list(range(n - n_pred, n))
        py = list(preds)
        ax1.plot(px, py, "r-", lw=3, alpha=0.85, label="Prediction")
        ax1.scatter(px[-1], py[-1], c="r", s=200, marker="*",
                    edgecolors="black", zorder=5)
        ax1.axvline(x=px[-1], color="gray", linestyle=":", alpha=0.3)

    bar_colors = ["#26a69a" if c[i] >= o[i] else "#ef5350" for i in range(n)]
    ax2.bar(x, v, color=bar_colors, alpha=0.5, width=0.8)
    ax2.set_ylabel("Volume", fontsize=9)
    ax2.set_xlabel("Time steps (minutes)", fontsize=9)
    current_price = c[-1] if len(c) > 0 else 0
    ax1.set_title(f"{symbol} — [{style.upper()}] "
                  f"${current_price:.2f}  →  Pred: ${last_pred:.2f}" if last_pred else
                  f"{symbol} — [{style.upper()}]  ${current_price:.2f}",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.15)
    ax2.grid(True, alpha=0.15)
    fig.tight_layout()

def _seed_graph(prices, model, device, actuals, preds, max_steps=120):
    n = min(len(prices) - CONTEXT, max_steps)
    if n < 1:
        return
    for i in range(n):
        ctx = prices[i:i+CONTEXT]
        preds.append(predict_next(model, device, ctx))
        actuals.append(prices[i+CONTEXT])

def _retrain_worker(data_df, data_dir, device):
    global _current_model
    print("\n[retrain] Starting background retraining...")
    new_model = build_model().to(device)
    run_retrain(new_model, data_dir, device)
    with _model_lock:
        _current_model = new_model
        _current_device = device
    print("[retrain] Model updated.")

def run_live(initial_model, params, symbol, data_dir, device, initial_df=None):
    global _current_model, _current_device, _chart_style
    _current_model = initial_model
    _current_device = device
    _chart_style = "candle"

    plt.ion()
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title(f"{symbol} - Live Stock Prediction")
    fig.canvas.mpl_connect("key_press_event", _on_key)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    plt.subplots_adjust(hspace=0.05)

    actual_prices = deque(maxlen=120)
    pred_prices = deque(maxlen=120)
    pred_count = 0
    accumulated_data = None if initial_df is None else initial_df.copy()

    print(f"\n[Live] Opening graph for {symbol}. Updates every {UPDATE_INTERVAL}s.")
    print("  Keys: [c]andle  [o]hlc  [l]ine  |  Close graph to stop.\n")

    if initial_df is not None and len(initial_df) > CONTEXT:
        prices = initial_df["Close"].values.astype(float)
        _seed_graph(prices, initial_model, device, actual_prices, pred_prices)
        pred_count = len(actual_prices)
        _draw(fig, ax1, ax2, initial_df.iloc[-120:],
              actual_prices, pred_prices, symbol,
              _chart_style, pred_prices[-1])
        plt.draw()
        plt.pause(0.5)
        print(f"  Seeded with {pred_count} predictions from training data.")

    try:
        while plt.fignum_exists(fig.number):
            df = fetch_live(symbol, LIVE_INTERVAL)
            if df is not None and len(df) > 0:
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

            plot_data = accumulated_data.iloc[-120:] if accumulated_data is not None and len(accumulated_data) > 0 else None
            if plot_data is not None and len(actual_prices) > 0:
                _draw(fig, ax1, ax2, plot_data,
                      actual_prices, pred_prices, symbol,
                      _chart_style, pred_prices[-1])
            plt.draw()
            plt.pause(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.close(fig)
