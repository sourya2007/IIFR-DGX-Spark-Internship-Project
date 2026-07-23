import re, json, urllib.request, os
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent.parent
TOKEN_START, TOKEN_SEP = 101, 102
INPUT_POLLUTANTS = ["pm2_5", "pm10", "co", "no2"]

FALLBACK = {
    "delhi":   {"pm25": 180, "pm10": 320, "co": 2.1, "no2": 85},
    "mumbai":  {"pm25": 95,  "pm10": 190, "co": 1.2, "no2": 45},
    "beijing": {"pm25": 150, "pm10": 280, "co": 1.8, "no2": 70},
    "london":  {"pm25": 25,  "pm10": 42,  "co": 0.3, "no2": 38},
    "new york":{"pm25": 12,  "pm10": 22,  "co": 0.2, "no2": 28},
    "tokyo":   {"pm25": 20,  "pm10": 35,  "co": 0.4, "no2": 30},
    "paris":   {"pm25": 18,  "pm10": 30,  "co": 0.3, "no2": 35},
    "sydney":  {"pm25": 8,   "pm10": 18,  "co": 0.1, "no2": 15},
    "singapore":{"pm25": 22, "pm10": 40,  "co": 0.5, "no2": 22},
    "dubai":   {"pm25": 55,  "pm10": 120, "co": 0.8, "no2": 50},
    "seoul":   {"pm25": 45,  "pm10": 75,  "co": 0.6, "no2": 40},
    "bangkok": {"pm25": 65,  "pm10": 110, "co": 0.9, "no2": 38},
    "moscow":  {"pm25": 35,  "pm10": 55,  "co": 0.5, "no2": 42},
}

AQI_BREAKPOINTS = [
    (0, 50, "Good", "\U0001f7e2"),
    (51, 100, "Moderate", "\U0001f7e1"),
    (101, 150, "Unhealthy for Sensitive Groups", "\U0001f7e0"),
    (151, 200, "Unhealthy", "\U0001f534"),
    (201, 300, "Very Unhealthy", "\U0001f7e5"),
    (301, 500, "Hazardous", "\U0001f7e4"),
]


def get_aqi_category(aqi):
    for lo, hi, label, emoji in AQI_BREAKPOINTS:
        if lo <= aqi <= hi:
            return f"{emoji} {label}"
    return "\U00002620 Hazardous"


def fetch_waqi(city, token):
    if not token or token == "":
        return None
    city_enc = urllib.request.quote(city.lower().strip())
    url = f"https://api.waqi.info/feed/{city_enc}/?token={token}"
    try:
        resp = urllib.request.urlopen(url, timeout=10).read()
        data = json.loads(resp)
        if data.get("status") != "ok":
            return None
        iaqi = data["data"].get("iaqi", {})
        return {
            "pm25": (iaqi.get("pm25") or {}).get("v"),
            "pm10": (iaqi.get("pm10") or {}).get("v"),
            "co":   (iaqi.get("co") or {}).get("v"),
            "no2":  (iaqi.get("no2") or {}).get("v"),
            "o3":   (iaqi.get("o3") or {}).get("v"),
            "so2":  (iaqi.get("so2") or {}).get("v"),
            "aqi":  data["data"].get("aqi"),
            "city": data["data"].get("city", {}).get("name", city),
            "time": (data["data"].get("time") or {}).get("s", ""),
        }
    except Exception:
        return None


def get_fallback(city):
    key = city.lower().strip()
    exact = FALLBACK.get(key)
    if exact:
        return exact
    for k, v in FALLBACK.items():
        if k in key or key in k:
            return v
    return FALLBACK["delhi"]


def compute_aqi(pm25, pm10):
    def _pm_sub(value, breakpoints):
        for lo, hi, *_ in breakpoints:
            if lo <= value <= hi:
                return ((value - lo) / (hi - lo)) * 50 + (breakpoints[0][0] if lo == breakpoints[0][0] else 0)
        return None
    aqi25 = _pm_sub(pm25, [(0,12),(12,35.4),(35.4,55.4),(55.4,150.4),(150.4,250.4),(250.4,500)])
    aqi10 = _pm_sub(pm10, [(0,54),(54,154),(154,254),(254,354),(354,424),(424,604)])
    vals = [v for v in [aqi25, aqi10] if v is not None]
    return round(max(vals)) if vals else None


def predict_aqi_live(model, device, params, data):
    scaled = []
    for f in INPUT_POLLUTANTS:
        raw = data.get(f.replace("_", "")) or data.get(f) or 0
        v = int(round((raw - params[f]["min"]) / params[f]["range"] * 100))
        scaled.append(max(0, min(100, v)))
    tokens = [TOKEN_START] + scaled + [TOKEN_SEP]
    inp = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        preds = []
        for _ in range(5):
            out = model.generate(inp, max_new_tokens=3, temperature=0.7)
            for offset in range(3):
                tok = out[0, len(tokens) + offset].item()
                if 0 <= tok <= 100:
                    aqi = tok / 100.0 * params["aqi_index"]["range"] + params["aqi_index"]["min"]
                    preds.append(aqi)
                    break
        if not preds:
            return None
        preds.sort()
        return round(preds[len(preds) // 2], 1)


def get_pollutant_data(city, token):
    live = fetch_waqi(city, token) if token else None
    if live and all(live.get(k) for k in ["pm25", "pm10", "no2"]):
        return {
            "source": "WAQI live",
            "city": live.get("city", city),
            "pm25": live["pm25"],
            "pm10": live["pm10"],
            "co": live.get("co") or 0,
            "no2": live["no2"],
            "aqi": live.get("aqi"),
            "time": live.get("time", ""),
        }
    fb = get_fallback(city)
    return {
        "source": "simulated (no API key)",
        "city": city.title(),
        "pm25": fb["pm25"],
        "pm10": fb["pm10"],
        "co": fb["co"],
        "no2": fb["no2"],
        "aqi": compute_aqi(fb["pm25"], fb["pm10"]),
        "time": "now (simulated)",
    }
