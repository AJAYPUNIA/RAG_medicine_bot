# --- standard libs first ---
import os
import time
import json
import traceback
from typing import Optional

# --- third‑party ---
import requests
from flask import Flask, request, render_template
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---------------- Runtime safety knobs (optional) ----------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONFAULTHANDLER"] = "1"

# ---------------- RAG / Models Config ----------------
# MUST match the model you used when you built faiss_index.bin
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# HF Inference API (serverless) — correct casing + correct path
HF_MODEL = "microsoft/BioGPT-Large"          # you can switch to "microsoft/BioGPT"
HF_TOKEN = os.environ.get("HF_TOKEN") or ""
HF_URL   = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# Local fallback generator (works on CPU, small + fast)
LOCAL_GEN = "google/flan-t5-small"
local_tok = AutoTokenizer.from_pretrained(LOCAL_GEN)
local_gen = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_GEN)

# ---------------- Flask app ----------------
app = Flask(__name__)

# ---------------- Load FAISS + texts + embedder ----------------
index = faiss.read_index("faiss_index.bin")
texts = np.load("indexed_texts.npy", allow_pickle=True)
embedder = SentenceTransformer(EMBEDDER_NAME)

# ---------------- Helpers ----------------
def _truncate(txt: str, n: int = 600) -> str:
    return (txt[:n] + "…") if len(txt) > n else txt

def _key_tokens(q: str):
    import re
    toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", q)]
    return set(toks)

def retrieve(query: str, k: int = 8, keep: Optional[int] = None):
    """FAISS retrieve with simple keyword preference and dimension checks."""
    if index.ntotal == 0:
        raise RuntimeError("FAISS index is empty. Rebuild it first.")
    if keep is None:
        keep = min(2, k)

    # Encode → float32 → 2D
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32").reshape(1, -1)
    if qv.shape[1] != index.d:
        raise RuntimeError(
            f"Embedder dim ({qv.shape[1]}) != FAISS dim ({index.d}). "
            "Rebuild the index with the SAME embedder."
        )

    # Search top‑k
    D, I = index.search(qv, k)
    ids = [i for i in I[0] if i >= 0]
    sims = D[0][:len(ids)]
    cands = [(i, float(sims[j]), texts[i]) for j, i in enumerate(ids)]

    # Prefer chunks that mention a token from the question
    keys = _key_tokens(query)
    def mentions_key(t: str) -> bool:
        low = t.lower()
        return any(k in low for k in keys) if keys else True

    filtered = [(i, s, t) for (i, s, t) in cands if mentions_key(t)]
    chosen = filtered if filtered else cands
    chosen.sort(key=lambda x: x[1], reverse=True)

    docs = [t if len(t) <= 600 else t[:600] + "…" for (_, _, t) in chosen[:keep]]
    return docs

# ---------------- Generation (HF + fallback) ----------------
def _prompt(question: str, context: str) -> str:
    return (
        "You are a careful medical assistant.\n"
        "Answer ONLY using the context below in 1–3 sentences.\n"
        "If not present in the context, say: \"I don't know based on the provided context.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

def generate_with_hf(prompt: str) -> str:
    """Call HF Inference API serverless endpoint for BioGPT."""
    if not HF_TOKEN.strip():
        raise RuntimeError("HF_TOKEN not set")

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 120,
            "do_sample": False,
            "num_beams": 4,
            "early_stopping": True,
            "return_full_text": False
        }
    }

    # tiny retry loop for 503/429 cold start / rate limit
    last_err = None
    for _ in range(3):
        try:
            r = requests.post(HF_URL, headers=HF_HEADERS, data=json.dumps(payload), timeout=60)
            if r.status_code in (503, 504, 429):
                time.sleep(3); continue
            r.raise_for_status()
            data = r.json()
            # Responses can be list or dict
            if isinstance(data, list) and data and isinstance(data[0], dict):
                text = data[0].get("generated_text") or data[0].get("summary_text")
                if text:
                    return text.strip()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            # Fallback if shape is odd
            return "I couldn't generate an answer (unexpected HF response)."
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"HF generation failed: {last_err}")

def generate_with_local(prompt: str) -> str:
    """Local deterministic beam‑search with FLAN‑T5 Small (CPU‑friendly)."""
    inputs = local_tok(prompt, return_tensors="pt", truncation=True, max_length=768)
    with torch.no_grad():
        out = local_gen.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )
    return local_tok.decode(out[0], skip_special_tokens=True).strip()

def generate_answer(question: str, context: str) -> str:
    prompt = _prompt(question, context)

    # 1) Try HF BioGPT first
    try:
        txt = generate_with_hf(prompt)
        return txt.split("Answer:", 1)[-1].strip() if "Answer:" in txt else txt
    except Exception as e:
        print("⚠️ HF BioGPT failed — using local fallback:", repr(e))

    # 2) Local fallback
    txt = generate_with_local(prompt)
    return txt.split("Answer:", 1)[-1].strip() if "Answer:" in txt else txt

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    answer, sources, error = "", [], ""
    try:
        if request.method == "POST":
            question = (request.form.get("question") or "").strip()
            if question:
                sources = retrieve(question, k=6, keep=2)
                context = "\n\n".join(sources)
                answer = generate_answer(question, context)
    except Exception as e:
        traceback.print_exc()
        error = f"{type(e).__name__}: {e}"

    return render_template("index.html", answer=answer, sources=sources, error=error)

# ---------------- Main ----------------
if __name__ == "__main__":
    # run without reloader so the HF/local models don't reload twice
    app.run(host="0.0.0.0", port=5001, debug=False)
