import urllib.request, torch

print("=== GPU ===")
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(i)
    print(f"Device: {torch.cuda.get_device_name(i)}")
    print(f"SM: {cc[0]}.{cc[1]}")
    print(f"BF16: {torch.cuda.is_bf16_supported()}")
    free, total = torch.cuda.mem_get_info(i)
    print(f"VRAM: {total/1e9:.1f}GB total, {free/1e9:.1f}GB free")
    for dt in ["float8_e4m3fn", "float8_e5m2"]:
        try:
            torch.empty(1, dtype=getattr(torch, dt), device="cuda")
            print(f"FP8 {dt}: supported")
        except: print(f"FP8 {dt}: not supported")
    print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 cudnn: {torch.backends.cudnn.allow_tf32}")
    print(f"torch.compile: {hasattr(torch, 'compile')}")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("NO GPU")

print("\n=== Network ===")
urls = [
    ("HF model.pt", "https://huggingface.co/bmeyer2025/tiny-gpt-shakespeare/resolve/main/model.pt"),
    ("Tiny Shakespeare", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"),
]
for name, url in urls:
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        data = resp.read()
        print(f"{name}: {len(data):,} bytes [{resp.status}]")
    except Exception as e:
        print(f"{name}: FAILED - {e}")
