import sys
import json
import time
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.aqi.data_prep import get_locations, run as run_data_prep
from scripts.aqi.model_setup import setup
from scripts.aqi.train import run as run_train
from scripts.aqi.predict import load_model, predict_aqi
from scripts.aqi.visualize import run as run_visualize
from config import MODEL_DIR, DATA_OUT_DIR

def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

def main():
    print_header("AQI Predictor — TinyGPT")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    locs = get_locations()
    print("\nAvailable locations:")
    print(f"  {'#':<3} {'Location':<22} {'Lat':<10} {'Lon':<10}")
    print(f"  {'-'*45}")
    for i, (_, row) in enumerate(locs.iterrows(), 1):
        print(f"  {i:<3} {row['location']:<22} {row['lat']:<10.4f} {row['lon']:<10.4f}")
    while True:
        try:
            choice = int(input("\nSelect location (1-6): "))
            if 1 <= choice <= len(locs):
                location = locs.iloc[choice - 1]["location"]
                break
        except ValueError:
            pass
        print("  Invalid choice.")
    print(f"  Selected: {location}")

    prompts = {"pm2_5": "PM2.5 (\u03bcg/m\u00b3): ", "pm10": "PM10 (\u03bcg/m\u00b3): ",
               "co": "CO (\u03bcg/m\u00b3): ", "no2": "NO2 (\u03bcg/m\u00b3): "}
    user_vals = {}
    for key, p in prompts.items():
        while True:
            try:
                v = float(input(f"  Enter {p}"))
                if v >= 0:
                    user_vals[key] = v
                    break
            except ValueError:
                pass
            print("    Enter a non-negative number.")

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")

    print_header("Phase 1: Data Preparation")
    try:
        data_dir = run_data_prep(location)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print_header("Phase 2: Building TinyGPT")
    try:
        model = setup()
        model = model.to(device)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print_header("Phase 3: Training")
    try:
        t0 = time.perf_counter()
        run_train(model, data_dir)
        print(f"  Training: {time.perf_counter() - t0:.1f}s")
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA OOM. Reduce BATCH_SIZE in config.py")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print_header("Phase 4: Prediction")
    try:
        model = load_model(MODEL_DIR / "best.pt", device)
        with open(f"{data_dir}/norm_params.json") as f:
            params = json.load(f)
        aqi = predict_aqi(model, device, params,
                          user_vals["pm2_5"], user_vals["pm10"],
                          user_vals["co"], user_vals["no2"])
        if aqi is None:
            print("  Prediction failed")
            sys.exit(1)
        print(f"\n  >>> Predicted AQI for {location}: {aqi:.1f}")
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print_header("Phase 5: Visualization")
    try:
        run_visualize(data_dir, user_vals, aqi)
    except Exception as e:
        print(f"  Plot failed: {e}")

    print_header("Done")
    print(f"  Model: {MODEL_DIR / 'best.pt'}")
    print(f"  Loss plot: plots/loss_curve.png")
    print(f"  Relationship plot: plots/pollutant_relationship.png")

if __name__ == "__main__":
    main()
