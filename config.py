from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "delhi-weather-aqi-2025.csv"
ADAPTER_DIR = BASE_DIR / "aqi_lora_adapter"
PLOTS_DIR = BASE_DIR / "plots"

LOCATION = "Anand Vihar"
CONTEXT_HOURS = 12
HORIZON = 1
MAX_SEQ_LENGTH = 256

FEATURES = ["temp_c", "humidity", "pressure_mb", "windspeed_kph",
            "pm2_5", "pm10", "co", "no2", "aqi_index"]
TARGETS = ["aqi_index", "pm2_5"]

MODEL_NAME = "distilbert/distilgpt2"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["c_attn"]

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 3e-4
EPOCHS = 7
FP16 = True
TEMPERATURE = 0.7
MAX_GEN_TOKENS = 30

TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
