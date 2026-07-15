import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PLOTS_DIR, MODEL_DIR, INPUT_POLLUTANTS
from scripts.aqi.predict import load_model, predict_aqi

LABELS = {"pm2_5": "PM2.5 (\u03bcg/m\u00b3)", "pm10": "PM10 (\u03bcg/m\u00b3)",
           "co": "CO (\u03bcg/m\u00b3)", "no2": "NO2 (\u03bcg/m\u00b3)"}

def extract_data(seq_tensor, params):
    vals = {f: [] for f in INPUT_POLLUTANTS}
    aqis = []
    for i in range(len(seq_tensor)):
        seq = seq_tensor[i]
        if seq[0].item() != 101:
            continue
        inp = []
        for j, f in enumerate(INPUT_POLLUTANTS):
            t = seq[1 + j].item()
            if 0 <= t <= 100:
                inp.append(t / 100.0 * params[f]["range"] + params[f]["min"])
            else:
                inp.append(None)
        aqi_tok = seq[6].item() if len(seq) > 6 else None
        if 0 <= aqi_tok <= 100 and all(v is not None for v in inp):
            for j, f in enumerate(INPUT_POLLUTANTS):
                vals[f].append(inp[j])
            aqis.append(aqi_tok / 100.0 * params["aqi_index"]["range"] + params["aqi_index"]["min"])
    return vals, aqis

def run(data_dir, user_input, predicted_aqi):
    with open(os.path.join(data_dir, "norm_params.json")) as f:
        params = json.load(f)

    X_train = torch.load(os.path.join(data_dir, "X_train.pt"))
    X_val = torch.load(os.path.join(data_dir, "X_val.pt"))
    X_all = torch.cat([X_train, X_val])
    vals, aqis = extract_data(X_all, params)
    if not vals[INPUT_POLLUTANTS[0]]:
        print("  No valid data points to plot")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(os.path.join(MODEL_DIR, "best.pt"), device)

    medians = {f: np.median(vals[f]) for f in INPUT_POLLUTANTS}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, pollutant in enumerate(INPUT_POLLUTANTS):
        ax = axes[idx]
        ax.scatter(vals[pollutant], aqis, c="#cccccc", alpha=0.4,
                   s=8, label="Training data", edgecolors="none")
        x_range = np.linspace(params[pollutant]["min"], params[pollutant]["max"], 30)
        trend_aqis = []
        for x in x_range:
            inp = {f: medians[f] for f in INPUT_POLLUTANTS}
            inp[pollutant] = x
            a = predict_aqi(model, device, params,
                            inp["pm2_5"], inp["pm10"], inp["co"], inp["no2"])
            trend_aqis.append(a)
        valid = [(x, a) for x, a in zip(x_range, trend_aqis) if a is not None]
        if valid:
            tx, ty = zip(*valid)
            ax.plot(tx, ty, color="#2196F3", linewidth=2.5,
                    label="Model trend", zorder=3)
        ax.scatter([user_input[pollutant]], [predicted_aqi],
                   c="#F44336", s=200, marker="*", edgecolors="black",
                   linewidth=0.8, label="Your prediction", zorder=5)
        ax.set_xlabel(LABELS[pollutant], fontsize=11)
        ax.set_ylabel("AQI Index", fontsize=11)
        ax.set_title(f"{pollutant.upper()}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Pollutant-AQI Relationship", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / "pollutant_relationship.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {path}")
