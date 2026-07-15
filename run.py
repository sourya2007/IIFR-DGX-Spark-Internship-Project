"""Run the full AQI prediction pipeline.
Usage: python run.py
"""
import os
import sys

os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONPATH"] = os.path.dirname(__file__)

from main import main

if __name__ == "__main__":
    main()
