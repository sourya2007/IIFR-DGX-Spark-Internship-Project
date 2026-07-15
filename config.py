from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "delhi-weather-aqi-2025.csv"
MODEL_DIR = BASE_DIR / "aqi_model"
PLOTS_DIR = BASE_DIR / "plots"
DATA_OUT_DIR = BASE_DIR / "data_prep_output"

LOCATION = "Anand Vihar"
CONTEXT_HOURS = 12
HORIZON = 1
MAX_SEQ_LENGTH = 256

FEATURES = ["temp_c", "humidity", "pressure_mb", "windspeed_kph",
            "pm2_5", "pm10", "co", "no2", "aqi_index"]
TARGETS = ["aqi_index", "pm2_5"]
N_FEATURES = len(FEATURES)

TOKEN_PAD = 103
TOKEN_SEP = 102
TOKEN_START = 101
VOCAB_SIZE = 104
VALUE_RANGE = 101

D_MODEL = 128
N_HEAD = 4
N_LAYER = 4
D_FF = 512
DROPOUT = 0.1

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 3e-4
EPOCHS = 7
FP16 = True
TEMPERATURE = 0.7
MAX_GEN_TOKENS = 20

TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
