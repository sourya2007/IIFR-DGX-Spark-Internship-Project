# TinyGPT — Text Generation & Mini-Transformer Projects

A collection of projects built around **TinyGPT**, a miniature GPT-like decoder-only transformer implemented from scratch in PyTorch. Also includes experimental forecasting pipelines for AQI and stock prices using the same model.

## Projects

### 1. Text Generation (main)

`run_project.py` — assignment project that trains TinyGPT on Shakespeare and Sherlock Holmes, comparing model variants:

| Experiment | Dataset | Context | Layers | Final Loss |
|-----------|---------|---------|--------|-----------|
| Baseline | Shakespeare | 256 | 6 | 1.15 |
| Change 1 | Sherlock Holmes | 256 | 6 | 1.10 |
| Change 2 | Shakespeare | 128 | 6 | 0.66 |
| Change 3 | Shakespeare | 256 | 4 | 1.40 |

```
python run_project.py
```

Datasets are downloaded automatically from Project Gutenberg.

### 2. AQI Predictor

`aqi_run.py` — predicts Air Quality Index from pollutant levels (PM2.5, PM10, CO, NO2) using a tiny transformer trained on Delhi weather data.

```
python aqi_run.py
```

Requires `delhi-weather-aqi-2025.csv` in the project root.

### 3. Stock Price Predictor

`stock_run.py` — downloads live stock data, trains a TinyGPT to predict price movements, and opens an interactive live prediction graph.

```
python stock_run.py
```

Requires `yfinance`.

## Model Architecture — TinyGPT

`tinygpt.py` — a decoder-only transformer with:

- Multi-head self-attention with causal masking
- Position-wise feed-forward networks (ReLU)
- Pre-layer normalization with residual connections
- Learned positional embeddings
- Configurable context length, layers, heads, and model dimension

```
TinyGPT(vocab_size, d_model, n_head, n_layer, d_ff, max_seq_len, dropout)
```

### Usage

```python
from tinygpt import TinyGPT

model = TinyGPT(vocab_size=5204, d_model=128, n_head=4,
                n_layer=6, d_ff=512, max_seq_len=256, dropout=0.1)

logits, loss = model(input_ids, labels)
output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.8)
```

## Project Structure

```
├── tinygpt.py              # Core transformer model
├── run_project.py          # Text generation experiments
├── pdf_chat.py             # Q&A on "Attention Is All You Need" paper
├── aqi_run.py              # AQI prediction pipeline
├── stock_run.py            # Stock price prediction + live graph
├── config.py               # AQI pipeline shared config
├── scripts/
│   ├── aqi/                # AQI data prep, training, prediction, visualization
│   └── stock/              # Stock data fetch, prep, training, prediction, live view
├── requirements.txt
└── output/                 # Generated experiment results (models, plots, reports)
```

## Requirements

- Python 3.9+
- PyTorch 2.1+
- pandas, numpy, matplotlib

Optional: `tqdm` (for AQI/stock training), `yfinance` (for stock pipeline), `pypdf` (for pdf_chat.py).
