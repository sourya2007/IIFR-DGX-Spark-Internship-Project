"""
DGX Spark — TinyGPT Web UI
============================
Gradio-based interface for interactive text generation with
TinyGPT word-level models, live AQI prediction, and system info.

Usage (after training):
    python dgx_demo.py --web-only
    # or imported by dgx_demo.py after training
"""

import os, sys, time, json, re, math
from pathlib import Path

import torch
import torch.nn as nn

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from scripts.dgx.text_utils import decode, encode_no_eos, search_passages
from scripts.dgx.aqi_live import get_pollutant_data, get_aqi_category, predict_aqi_live


def create_ui(trained_models, word2idx, idx2word, data_dir, device, amp_dtype):
    import gradio as gr
    from dgx_demo import system_info

    data_dir = Path(data_dir)
    full_text = ""
    for f in sorted(data_dir.iterdir()):
        if f.suffix == ".txt":
            full_text += f.read_text(encoding="utf-8", errors="replace") + "\n\n"

    aqi_model = aqi_params = None
    aqi_model_path = BASE / "aqi_model" / "best.pt"
    if aqi_model_path.exists():
        try:
            from config import VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, DROPOUT
            from tinygpt import TinyGPT
            aqi_model = TinyGPT(VOCAB_SIZE, D_MODEL, N_HEAD, N_LAYER, D_FF, MAX_SEQ_LENGTH, DROPOUT)
            aqi_model.load_state_dict(torch.load(aqi_model_path, map_location=device, weights_only=True))
            aqi_model = aqi_model.to(device)
            aqi_model.eval()
            import json
            params_path = BASE / "data_prep_output" / "Anand_Vihar" / "norm_params.json"
            if params_path.exists():
                aqi_params = json.loads(params_path.read_text())
        except Exception:
            aqi_model = aqi_params = None

    def generate_fn(prompt, model_choice, temperature, use_rag, max_length):
        if not prompt or not prompt.strip():
            return "", "", "", ""
        context = ""
        if use_rag and full_text:
            passages = search_passages(prompt, full_text, top_k=1)
            if passages:
                ctx = passages[0][:400]
                context = ctx

        outputs = {}
        for name, model, stats in trained_models:
            if model_choice != "All" and model_choice != name:
                continue
            try:
                text = _generate(model, prompt, temperature, max_length, word2idx, idx2word, device, context)
                outputs[name] = text
            except Exception as e:
                outputs[name] = f"[error: {e}]"

        src = context if context else "(no context found)"
        return src, outputs.get("Medium", ""), outputs.get("Large", ""), outputs.get("X-Large", "")

    def aqi_fn(city, api_token):
        try:
            data = get_pollutant_data(city, api_token)
            aqi = None
            if aqi_model and aqi_params:
                aqi = predict_aqi_live(aqi_model, device, aqi_params, data)
            display = (
                f"**Location:** {data['city']}  |  **Source:** {data['source']}\n\n"
                f"| Pollutant | Value |\n"
                f"|-----------|-------|\n"
                f"| PM2.5 | {data['pm25']} µg/m³ |\n"
                f"| PM10  | {data['pm10']} µg/m³ |\n"
                f"| CO    | {data['co']} µg/m³ |\n"
                f"| NO2   | {data['no2']} µg/m³ |\n\n"
            )
            if data.get("aqi"):
                cat = get_aqi_category(data["aqi"])
                display += f"**WAQI AQI:** {data['aqi']} — {cat}\n"
            if aqi is not None:
                cat = get_aqi_category(aqi)
                display += f"**TinyGPT Predicted AQI:** {aqi} — {cat}\n"
            display += f"\n_Last updated: {data.get('time', 'unknown')}_"
            return display
        except Exception as e:
            return f"Error: {e}"

    def dashboard_fn():
        info = system_info()
        table = "| Model | Params | Final Loss |\n|-------|--------|------------|\n"
        for name, _, stats in trained_models:
            loss = stats.get("final_loss", "—")
            params = stats.get("params", "—")
            if isinstance(params, int):
                params = f"{params:,}"
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            table += f"| {name} | {params} | {loss} |\n"
        return f"```\n{info}\n```\n\n{table}"

    with gr.Blocks(
        title="DGX Spark — TinyGPT",
        theme="soft",
        css="""
        .model-output { border-left: 3px solid #6366f1; padding-left: 12px; margin: 8px 0; }
        .source-box { background: #f3f4f6; border-radius: 6px; padding: 12px; font-size: 0.9em; }
        .aqi-value { font-size: 2.5em; font-weight: bold; text-align: center; }
        footer { visibility: hidden; }
        """
    ) as demo:
        gr.Markdown(
            "# DGX Spark — TinyGPT\n"
            "Word-level language model trained on 5 classic books. "
            "Powered by Blackwell GPU with BF16 + torch.compile."
        )

        with gr.Tab("✍ Text Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="Your prompt",
                        placeholder="e.g. Tell me about Romeo, or: It was a dark...",
                        lines=3,
                    )
                    with gr.Row():
                        model_choice = gr.Radio(
                            choices=["All", "Medium", "Large", "X-Large"],
                            value="All",
                            label="Model",
                        )
                        temperature = gr.Slider(
                            0.1, 2.0, value=0.8, step=0.05, label="Temperature"
                        )
                    with gr.Row():
                        use_rag = gr.Checkbox(value=True, label="Search books for context")
                        max_length = gr.Slider(50, 500, value=200, step=10, label="Max tokens")
                    generate_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=1):
                    source_box = gr.Markdown("**Source context:** _(none yet)_", elem_classes="source-box")
                    with gr.Tabs():
                        with gr.TabItem("Medium (35M)"):
                            out_medium = gr.Textbox(label="", lines=6, elem_classes="model-output")
                        with gr.TabItem("Large (75M)"):
                            out_large = gr.Textbox(label="", lines=6, elem_classes="model-output")
                        with gr.TabItem("X-Large (180M)"):
                            out_xl = gr.Textbox(label="", lines=6, elem_classes="model-output")

            generate_btn.click(
                fn=generate_fn,
                inputs=[prompt, model_choice, temperature, use_rag, max_length],
                outputs=[source_box, out_medium, out_large, out_xl],
            )
            prompt.submit(
                fn=generate_fn,
                inputs=[prompt, model_choice, temperature, use_rag, max_length],
                outputs=[source_box, out_medium, out_large, out_xl],
            )

        with gr.Tab("🌍 Live AQI"):
            gr.Markdown(
                "Enter a city name to fetch live air quality data "
                "and predict the AQI using TinyGPT.\n\n"
                "Get a free WAQI API token at [aqicn.org/api](https://aqicn.org/api/) "
                "(or leave blank for simulated data)."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    aqi_city = gr.Textbox(
                        label="City", placeholder="e.g. Delhi, London, Tokyo", value="Delhi"
                    )
                    aqi_token = gr.Textbox(
                        label="WAQI API token (optional)",
                        placeholder="your token or leave blank",
                        type="password",
                    )
                    aqi_btn = gr.Button("Fetch & Predict", variant="primary")
                with gr.Column(scale=1):
                    aqi_output = gr.Markdown("Click **Fetch & Predict** to start.")

            aqi_btn.click(fn=aqi_fn, inputs=[aqi_city, aqi_token], outputs=[aqi_output])

        with gr.Tab("📊 System Dashboard"):
            gr.Markdown("### DGX Spark System Info")
            dash_output = gr.Markdown("_Loading..._")
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(fn=dashboard_fn, outputs=[dash_output])
            demo.load(fn=dashboard_fn, outputs=[dash_output])

    return demo


def _generate(model, prompt, temperature, max_length, word2idx, idx2word, device, context):
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    ids = encode_no_eos(full_prompt, word2idx).to(device)
    if len(ids) > model.max_seq_len:
        ids = torch.cat([ids[:1], ids[-(model.max_seq_len - 1):]])
    model.eval()
    with torch.no_grad():
        out = model.generate(ids.unsqueeze(0), max_length, temperature=temperature)
    result = decode(out[0].tolist(), idx2word)
    idx = result.find(prompt)
    if idx >= 0:
        result = result[idx + len(prompt):].strip()
    elif context:
        first_line = context.split("\n")[0].strip()
        if first_line and first_line in result:
            idx = result.find(first_line)
            result = result[idx + len(context):].strip()
    return result if result else "(no output generated)"


def launch():
    """Launch the web UI with saved models (no training)."""
    from dgx_demo import load_trained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    trained, word2idx, idx2word, configs = load_trained(device)
    if trained is None:
        print("No saved models in dgx_models/. Run: python dgx_demo.py")
        return
    data_dir = Path(__file__).parent / "dgx_data"
    demo = create_ui(trained, word2idx, idx2word, str(data_dir), device, amp_dtype)
    print("Starting DGX Spark TinyGPT UI at http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    launch()
