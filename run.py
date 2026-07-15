import sys
import json
import time
import torch
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from scripts.data_prep import get_locations, run as run_data_prep
from scripts.model_setup import setup
from scripts.train import run as run_train
from scripts.predict import load_model, predict_aqi
from scripts.visualize import run as run_visualize
from config import MODEL_DIR, DATA_OUT_DIR

def print_banner():
    print("=" * 55)
    print("  AQI Predictor — TinyGPT")
    print("  Predict AQI from PM2.5, PM10, CO, NO2")
    print("=" * 55)

def choose_location():
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
                return locs.iloc[choice - 1]["location"]
        except ValueError:
            pass
        print("  Invalid choice, try again.")

def get_pollutants():
    prompts = {"pm2_5": "PM2.5 (\u03bcg/m\u00b3): ",
               "pm10": "PM10 (\u03bcg/m\u00b3): ",
               "co": "CO (\u03bcg/m\u00b3): ",
               "no2": "NO2 (\u03bcg/m\u00b3): "}
    vals = {}
    print()
    for key, prompt in prompts.items():
        while True:
            try:
                v = float(input(f"  Enter {prompt}"))
                if v >= 0:
                    vals[key] = v
                    break
            except ValueError:
                pass
            print("    Enter a non-negative number.")
    return vals

def main():
    print_banner()
    location = choose_location()
    locs = get_locations()
    loc_row = locs[locs["location"] == location].iloc[0]
    print(f"\n  Selected: {location} ({loc_row['lat']}, {loc_row['lon']})")

    user_vals = get_pollutants()
    print(f"\n  Input: PM2.5={user_vals['pm2_5']}, PM10={user_vals['pm10']}, "
          f"CO={user_vals['co']}, NO2={user_vals['no2']}")

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    if has_cuda:
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 55)
    print("  Phase 1: Data Preparation")
    print("=" * 55)
    try:
        data_dir = run_data_prep(location)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  Phase 2: Building TinyGPT")
    print("=" * 55)
    try:
        model = setup()
        model = model.to(device)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  Phase 3: Training")
    print("=" * 55)
    try:
        t0 = time.perf_counter()
        run_train(model, data_dir)
        train_time = time.perf_counter() - t0
        print(f"  Training completed in {train_time:.1f}s")
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA OOM. Set BATCH_SIZE=16 in config.py.")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  Phase 4: Prediction")
    print("=" * 55)
    try:
        model = load_model(MODEL_DIR / "best.pt", device)
        with open(f"{data_dir}/norm_params.json") as f:
            params = json.load(f)
        aqi = predict_aqi(model, device, params,
                          user_vals["pm2_5"], user_vals["pm10"],
                          user_vals["co"], user_vals["no2"])
        if aqi is None:
            print("  Prediction failed (model generated invalid tokens)")
            sys.exit(1)
        print(f"\n  >>> Predicted AQI for {location}: {aqi:.1f}")
        print(f"      (PM2.5={user_vals['pm2_5']}, PM10={user_vals['pm10']}, "
              f"CO={user_vals['co']}, NO2={user_vals['no2']})")
    except Exception as e:
        print(f"[FATAL] Prediction failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  Phase 5: Visualization")
    print("=" * 55)
    try:
        run_visualize(data_dir, user_vals, aqi)
    except Exception as e:
        print(f"  Plot generation failed (non-fatal): {e}")

    print("\n" + "=" * 55)
    print("  Done!")
    print(f"  Model: {MODEL_DIR / 'best.pt'}")
    print(f"  Loss plot: {Path('plots') / 'loss_curve.png'}")
    print(f"  Relationship plot: {Path('plots') / 'pollutant_relationship.png'}")
    print("=" * 55)

if __name__ == "__main__":
    main()
