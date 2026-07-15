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
    print_header("AQI Prediction System — TinyGPT (DistilGPT2 + LoRA)")
    has_cuda = torch.cuda.is_available()
    print(f"  Device: {'CUDA' if has_cuda else 'CPU'}")
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Phase 1
    print_header("Phase 1: Data Preparation")
    try:
        data_dir = run_data_prep()
    except Exception as e:
        print(f"[FATAL] Data preparation failed: {e}")
        sys.exit(1)

    # Phase 2
    print_header("Phase 2: Model Setup")
    try:
        model, tokenizer = setup()
        if has_cuda:
            model = model.to("cuda")
    except Exception as e:
        print(f"[FATAL] Model setup failed: {e}")
        sys.exit(1)

    # Phase 3a: CPU benchmark (1 epoch) — test on CPU first
    if has_cuda:
        print_header("Phase 3a: CPU Benchmark (1 epoch)")
        model_cpu = model.to("cpu")
        t0 = time.perf_counter()
        try:
            run_train(model_cpu, tokenizer, data_dir, max_epochs=1)
            cpu_time = time.perf_counter() - t0
            print(f"  CPU 1 epoch: {cpu_time:.1f}s")
        except Exception as e:
            print(f"  CPU benchmark note: {e}")
        model = model.to("cuda")

    # Phase 3b: GPU Training
    print_header("Phase 3b: GPU Training (full)")
    try:
        t0 = time.perf_counter()
        train_log = run_train(model, tokenizer, data_dir)
        gpu_total = time.perf_counter() - t0
        print(f"  Total GPU training: {gpu_total:.1f}s")
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA OOM. Reduce batch_size in config.py to 4 or 2, or enable gradient checkpointing.")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Training failed: {e}")
        sys.exit(1)

    # Phase 4: Evaluation
    print_header("Phase 4: Evaluation & Metrics")
    try:
        metrics = run_evaluate(data_dir)
    except Exception as e:
        print(f"[FATAL] Evaluation failed: {e}")
        sys.exit(1)

    # Summary
    print_header("Summary")
    print(f"  LoRA adapter: aqi_lora_adapter/")
    print(f"  Loss plot: plots/loss_curve.png")
    print(f"  AQI   MAE={metrics['mae_aqi']:.2f}  RMSE={metrics['rmse_aqi']:.2f}")
    print(f"  PM2.5 MAE={metrics['mae_pm']:.2f}  RMSE={metrics['rmse_pm']:.2f}")
    if train_log["epoch_times"]:
        print(f"  Avg epoch time: {sum(train_log['epoch_times'])/len(train_log['epoch_times']):.1f}s")
    if train_log["gpu_memory"]:
        print(f"  Peak GPU mem: {max(train_log['gpu_memory']):.2f} GB")
    print("  Done!")

if __name__ == "__main__":
    main()
