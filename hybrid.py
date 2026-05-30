# hybrid.py
# Hybrid HF model + Feature-based classifier + Flask API (inference only)
# Robust: tokenizer->model->softmax inference (no pipeline) to avoid token_type_ids errors.

import os
import re
import json
import textwrap
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import red, black

# HuggingFace / PyTorch
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# feature classifier (your file)
from features_classifier import fit_human_stats, classify

# -------------------------
# Paths / Config
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HF_MODEL_DIR = os.path.join(BASE_DIR, "hf_model")        # put your HF export here
FEATURE_STATS_FILE = os.path.join(BASE_DIR, "feature_stats.json")
DATASET_CSV = os.path.join(BASE_DIR, "dataset.csv")
HUMAN_SAMPLES_TXT = os.path.join(BASE_DIR, "human_samples.txt")

# -------------------------
# Load HF model + tokenizer
# -------------------------
if not os.path.exists(HF_MODEL_DIR):
    raise FileNotFoundError(f"HF model folder not found: {HF_MODEL_DIR}. Put your config/model/tokenizer files there.")

print(f"[init] Loading HF model from: {HF_MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_DIR, local_files_only=True)
# ensure model on CPU/GPU appropriately
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("[init] model.config.id2label:", getattr(model.config, "id2label", None))
print("[init] model.config.num_labels:", getattr(model.config, "num_labels", None))

# -------------------------
# Helpers: decide AI index from id2label
# -------------------------
def find_ai_index_from_id2label():
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        return None
    lbls = {}
    # normalize possible key types
    for k, v in id2label.items():
        try:
            ki = int(k)
        except Exception:
            try:
                ki = int(str(k))
            except Exception:
                continue
        lbls[ki] = str(v).lower()
    for idx, name in lbls.items():
        if name in ("ai", "generated", "machine", "fake", "bot", "automated"):
            return idx
    return None

AI_INDEX = find_ai_index_from_id2label()
print("[init] detected AI_INDEX from id2label:", AI_INDEX)

# -------------------------
# Robust direct inference (tokenizer -> model -> softmax)
# -------------------------
def ml_predict_probs(texts, batch_size=32, max_length=1024):
    """
    texts: list[str]
    returns: numpy array of AI probabilities (shape len(texts), values 0..1)
    """
    probs_out = []
    device_local = next(model.parameters()).device

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        # If token_type_ids is present but model doesn't accept it, remove it.
        # (Some model classes like DistilBert don't accept token_type_ids.)
        if "token_type_ids" in enc:
            # Try to detect if model.forward accepts token_type_ids by inspecting signature.
            try:
                accepts = "token_type_ids" in model.forward.__code__.co_varnames
            except Exception:
                accepts = False
            if not accepts:
                enc.pop("token_type_ids", None)

        # move tensors to model device
        enc = {k: v.to(device_local) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            logits = getattr(out, "logits", None)
            if logits is None:
                # if no logits, fallback zeros
                batch_probs = [0.0] * len(batch)
            else:
                probs_all = F.softmax(logits, dim=-1).cpu().numpy()  # shape (B, num_labels)
                num_labels = probs_all.shape[1]
                # decide AI column index
                if AI_INDEX is not None and AI_INDEX < num_labels:
                    ai_col = AI_INDEX
                else:
                    ai_col = 1 if num_labels >= 2 else 0
                batch_probs = probs_all[:, ai_col].tolist()
        probs_out.extend(batch_probs)

    return np.array(probs_out)

def ml_predict_prob_single(text):
    return float(ml_predict_probs([text], batch_size=1)[0])

# -------------------------
# Simple cleaning (keep minimal)
# -------------------------
def clean_text(t):
    if t is None:
        return ""
    t = str(t).strip()
    return t

# -------------------------
# Load or compute feature_stats
# -------------------------
def load_feature_stats():
    # 1) load file if exists
    if os.path.exists(FEATURE_STATS_FILE):
        print(f"[init] Loading feature stats from {FEATURE_STATS_FILE}")
        with open(FEATURE_STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2) compute from dataset.csv if available
    if os.path.exists(DATASET_CSV):
        print(f"[init] Computing feature stats from {DATASET_CSV}")
        try:
            import pandas as pd
            df = pd.read_csv(DATASET_CSV)
            # heuristics to select human rows
            if "label" in df.columns:
                human_mask = df["label"].astype(str).str.lower().isin(["human","human-written","real","0"])
            elif "generated" in df.columns:
                try:
                    human_mask = df["generated"].astype(int) == 0
                except Exception:
                    human_mask = df["generated"].astype(str).str.lower().isin(["0","false","no"])
            else:
                human_mask = [True] * len(df)
            human_texts = df.loc[human_mask, "text"].astype(str).tolist()
            if len(human_texts) == 0:
                raise ValueError("No human samples found in dataset.csv")
            sample = human_texts[:300]
            mu, sigma = fit_human_stats(sample)
            stats = {"mu": mu, "sigma": sigma}
            with open(FEATURE_STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            print(f"[init] feature_stats saved to {FEATURE_STATS_FILE}")
            return stats
        except Exception as e:
            print("[init] failed to compute from dataset.csv:", e)

    # 3) compute from human_samples.txt
    if os.path.exists(HUMAN_SAMPLES_TXT):
        print(f"[init] Computing feature stats from {HUMAN_SAMPLES_TXT}")
        with open(HUMAN_SAMPLES_TXT, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        sample = lines[:300]
        mu, sigma = fit_human_stats(sample)
        stats = {"mu": mu, "sigma": sigma}
        with open(FEATURE_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        return stats

    # 4) fallback defaults
    print("[warning] No feature stats found. Using conservative default values.")
    default_stats = {"mu": {
                        "f1_lex_div": 0.58, "f2_sent_len_sigma": 4.10, "f3_ngram_rep": 0.008,
                        "f4_punct_ratio": 0.17, "f5_contraction_ratio": 0.045, "f6_stopword_ratio": 0.48,
                        "f7_flesch": 62.0},
                     "sigma": {
                        "f1_lex_div": 0.12, "f2_sent_len_sigma": 1.95, "f3_ngram_rep": 0.015,
                        "f4_punct_ratio": 0.08, "f5_contraction_ratio": 0.020, "f6_stopword_ratio": 0.10,
                        "f7_flesch": 14.5}}
    return default_stats

feature_stats = load_feature_stats()

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def home():
    return render_template("index.html") if os.path.exists(os.path.join(app.template_folder, "index.html")) else "Hybrid AI detector running."

@app.route("/analyze", methods=["POST"])
def analyze_text():
    data = request.get_json() or {}
    text = data.get("text", "")
    include_features = bool(data.get("include_features", False))

    if not text or not str(text).strip():
        return jsonify({"error": "No text provided"}), 400

    cleaned = clean_text(text)

    # ML inference (direct)
    try:
        ml_prob = float(ml_predict_prob_single(cleaned))
    except Exception as e:
        print("[error] ML inference failed:", e)
        ml_prob = 0.0
    ml_label = "AI" if ml_prob >= 0.5 else "Human"

    # Feature classifier
    try:
        feat_res = classify(cleaned, feature_stats, threshold=0.5, use_rttr=False)
        feat_score = float(feat_res.get("final_score", 0.0))
        feat_label = feat_res.get("classification", "Human")
    except Exception as e:
        print("[error] Feature classifier failed:", e)
        feat_res = {}
        feat_score = 0.0
        feat_label = "Human"

    resp = {
        "ml_label": ml_label,
        "ml_conf": round(ml_prob, 3),
        "feature_label": feat_label,
        "feature_score": round(feat_score, 3),
        "agreement": ml_label == feat_label
    }
    if include_features:
        resp["feature_details"] = feat_res

    return jsonify(resp)

@app.route("/analyze_pdf", methods=["POST"])
def analyze_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = None
        if page_text:
            text += page_text + "\n"

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return jsonify({"error": "No readable text in PDF"}), 400

    # ML probabilities (batch)
    try:
        probs = ml_predict_probs([clean_text(l) for l in lines], batch_size=64)
    except Exception as e:
        print("[error] batch ML inference failed:", e)
        probs = np.array([0.0] * len(lines))

    preds = np.where(probs >= 0.5, "AI", "Human")

    output_pdf = BytesIO()
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    y = height - 50

    for line, label, conf in zip(lines, preds, probs):
        c.setFillColor(red if label == "AI" else black)
        display_line = f"[{label} - {conf*100:.1f}%] {line}"
        for subline in textwrap.wrap(display_line, width=90):
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(50, y, subline)
            y -= 14

    c.save()
    output_pdf.seek(0)

    return send_file(output_pdf, as_attachment=True, download_name="analyzed_output.pdf", mimetype="application/pdf")

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    # debug True for dev, change to False in production
    app.run(host="0.0.0.0", port=8000, debug=True)
