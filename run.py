"""Run the full AQI prediction pipeline.
Usage: python run.py
"""
import os
os.environ["PYTHONPATH"] = os.path.dirname(__file__)
from main import main

if __name__ == "__main__":
    main()
