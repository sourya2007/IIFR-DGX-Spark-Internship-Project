#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
#  setup.sh  —  DGX Spark environment install
#  Run ONCE on the DGX Spark before the demo.
#  Usage:  bash setup.sh
# ──────────────────────────────────────────────────────────────

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-tinygpt-env}"

echo "════════════════════════════════════════════════════════"
echo "  TinyGPT — DGX Spark Setup"
echo "════════════════════════════════════════════════════════"

# ── 1. Check Python ─────────────────────────────────────────
echo ""
echo "── [1/4] Checking Python ──"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "  ✗ $PYTHON not found. Install Python 3.10+ first."
    exit 1
fi
echo "  Python: $($PYTHON --version)"

# ── 2. Create virtual environment ───────────────────────────
echo ""
echo "── [2/4] Creating virtual environment ──"
if [ -d "$VENV_DIR" ]; then
    echo "  ${VENV_DIR} exists, activating."
else
    $PYTHON -m venv "$VENV_DIR"
    echo "  Created ${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "  $(python --version)  @ $(which python)"

# ── 3. Install PyTorch (CUDA) ───────────────────────────────
echo ""
echo "── [3/4] Installing PyTorch (CUDA) ──"
# DGX Spark / Grace Blackwell — use latest stable CUDA build
pip install --upgrade pip wheel

# Install PyTorch with CUDA.  If this fails (e.g. newer CUDA),
# replace the --index-url with the correct tag from pytorch.org.
# Versions: cu118 (11.8), cu121 (12.1), cu124 (12.4)
python -c "
import urllib.request, json
# Fetch the latest stable PyTorch version
url = 'https://pypi.org/pypi/torch/json'
data = json.loads(urllib.request.urlopen(url).read())
stable = [v for v in data['releases'] if not any(x in v for x in ['dev','rc','a','b','post','cpu'])]
latest = sorted(stable, key=lambda s: [int(x) for x in s.split('.')])[-1]
print(f'PyTorch stable: {latest}')
" 2>/dev/null || echo "  (version check skipped)"

pip install torch --index-url https://download.pytorch.org/whl/cu124

# Verify GPU
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print(f'  GPU: {torch.cuda.get_device_name(i)}')
    free, total = torch.cuda.mem_get_info(i)
    print(f'  VRAM: {total/1e9:.1f} GB')
    print(f'  BF16: {\"yes\" if torch.cuda.is_bf16_supported() else \"no\"}')
else:
    print('  ⚠ CUDA not available — running on CPU')
"

# ── 4. Install remaining dependencies ───────────────────────
echo ""
echo "── [4/4] Installing dependencies ──"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Setup complete."
echo ""
echo "  Activate:  source ${VENV_DIR}/bin/activate"
echo "  Run demo:  python dgx_demo.py"
echo "  Quick:     python dgx_demo.py --quick"
echo "════════════════════════════════════════════════════════"
