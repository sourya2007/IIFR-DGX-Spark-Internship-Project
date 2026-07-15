import sys
import time
import torch
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from scripts.stock.fetch import download_initial
from scripts.stock.prep import run as run_prep
from scripts.stock.train import build_model, run as run_train
from scripts.stock.live import run_live

def main():
    print("=" * 55)
    print("  StockGPT — Live Stock Price Prediction")
    print("=" * 55)

    symbol = input("\nStock symbol [AAPL]: ").strip().upper() or "AAPL"

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n{'='*55}")
    print(f"  Phase 1: Downloading {symbol} data")
    print(f"{'='*55}")
    try:
        df = download_initial(symbol)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Phase 2: Data Preparation")
    print(f"{'='*55}")
    try:
        data_dir = run_prep(df)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Phase 3: Building TinyGPT")
    print(f"{'='*55}")
    try:
        model = build_model().to(device)
        print(f"  Model on {device}")
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Phase 4: Training")
    print(f"{'='*55}")
    try:
        t0 = time.perf_counter()
        run_train(model, data_dir, device)
        print(f"  Training: {time.perf_counter() - t0:.1f}s")
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA OOM. Reduce BATCH_SIZE in scripts/stock/config.py")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Phase 5: Live Prediction Graph")
    print(f"{'='*55}")
    try:
        run_live(model, None, symbol, data_dir, device, initial_df=df)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print("\n  Done!")

if __name__ == "__main__":
    main()
