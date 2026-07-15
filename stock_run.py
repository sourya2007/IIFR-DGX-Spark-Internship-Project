import sys
import time
from datetime import datetime
import zoneinfo
import torch
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from scripts.stock.fetch import download_initial
from scripts.stock.prep import run as run_prep
from scripts.stock.train import build_model, run as run_train
from scripts.stock.live import run_live

_MARKETS = [
    ("NYSE/NASDAQ (US)",       "America/New_York",     (9,30), (16,0)),
    ("NSE/BSE (India)",        "Asia/Kolkata",         (9,15), (15,30)),
    ("LSE (UK)",               "Europe/London",        (8,0),  (16,30)),
    ("TSE (Japan)",            "Asia/Tokyo",           (9,0),  (15,0)),
    ("HKEX (Hong Kong)",       "Asia/Hong_Kong",       (9,30), (16,0)),
    ("ASX (Australia)",        "Australia/Sydney",     (10,0), (16,0)),
    ("XETRA/FRA (Germany)",    "Europe/Berlin",        (9,0),  (17,30)),
    ("EURONEXT (Paris)",       "Europe/Paris",         (9,0),  (17,30)),
]

def _print_open_markets():
    now_utc = datetime.now(zoneinfo.ZoneInfo("UTC"))
    today_utc = now_utc.date()
    # weekday 0=Mon..4=Fri, 5=Sat, 6=Sun
    if now_utc.weekday() >= 5:
        print("  Weekend — most markets closed.\n")
        return
    open_markets = []
    for label, tz_name, (oh, om), (ch, cm) in _MARKETS:
        tz = zoneinfo.ZoneInfo(tz_name)
        now_local = now_utc.astimezone(tz)
        # Local weekday check (e.g. Sunday in Israel etc)
        if now_local.weekday() >= 5:
            continue
        market_open = now_local.replace(hour=oh, minute=om, second=0, microsecond=0)
        market_close = now_local.replace(hour=ch, minute=cm, second=0, microsecond=0)
        if market_open <= now_local <= market_close:
            open_markets.append(f"  {label}")
    if open_markets:
        print("  Open now:")
        for m in open_markets:
            print(m)
    else:
        print("  No major markets open right now.")
    print()

def main():
    print("=" * 55)
    print("  StockGPT — Live Stock Price Prediction")
    print("=" * 55)
    _print_open_markets()

    symbol = input("Stock symbol [AAPL]: ").strip().upper() or "AAPL"

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    while True:
        print(f"\n{'='*55}")
        print(f"  Phase 1: Validating {symbol}")
        print(f"{'='*55}")
        try:
            df = download_initial(symbol, period="1d")
            print(f"  {symbol} OK — {len(df)} rows")
            break
        except Exception as e:
            print(f"  {e}")
            symbol = input("Try another symbol: ").strip().upper()
            if not symbol:
                print("  Aborted.")
                sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Phase 2: Downloading full 7d data")
    print(f"{'='*55}")
    try:
        df = download_initial(symbol, period="7d")
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Phase 3: Data Preparation")
    print(f"{'='*55}")
    try:
        data_dir = run_prep(df)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Phase 4: Building TinyGPT")
    print(f"{'='*55}")
    try:
        model = build_model().to(device)
        print(f"  Model on {device}")
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Phase 5: Training")
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
    print(f"  Phase 6: Live Prediction Graph")
    print(f"{'='*55}")
    try:
        run_live(model, None, symbol, data_dir, device, initial_df=df)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print("\n  Done!")

if __name__ == "__main__":
    main()
