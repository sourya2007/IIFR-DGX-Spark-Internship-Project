import sys
import time
import torch

from scripts.data_prep import run as run_data_prep
from scripts.model_setup import setup
from scripts.train import run as run_train
from scripts.evaluate import run as run_evaluate

def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

def main():
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    print_header("AQI Prediction — TinyGPT (DistilGPT2 + LoRA)")
    print(f"  Device: {device}")
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Phase 1
    print_header("Phase 1: Data Preparation")
    try:
        data_dir = run_data_prep()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    # Phase 2
    print_header("Phase 2: Model Setup")
    try:
        model, tokenizer = setup()
        if has_cuda:
            model = model.to("cuda")
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    # Phase 3: Training
    print_header("Phase 3: GPU Training")
    try:
        t0 = time.perf_counter()
        train_log = run_train(model, tokenizer, data_dir)
        print(f"  Total time: {time.perf_counter() - t0:.1f}s")
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA OOM. Set BATCH_SIZE=4 in config.py.")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    # Phase 4: Evaluation
    print_header("Phase 4: Evaluation")
    try:
        metrics = run_evaluate(data_dir)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print_header("Summary")
    print(f"  LoRA adapter: aqi_lora_adapter/")
    print(f"  Loss plot: plots/loss_curve.png")
    print(f"  AQI   MAE={metrics['mae_aqi']:.2f}  RMSE={metrics['rmse_aqi']:.2f}")
    print(f"  PM2.5 MAE={metrics['mae_pm']:.2f}  RMSE={metrics['rmse_pm']:.2f}")
    if train_log["epoch_times"]:
        print(f"  Avg epoch: {sum(train_log['epoch_times'])/len(train_log['epoch_times']):.1f}s")
    if train_log["gpu_memory"]:
        print(f"  Peak GPU mem: {max(train_log['gpu_memory']):.2f} GB")
    print("  Done!")

if __name__ == "__main__":
    main()
