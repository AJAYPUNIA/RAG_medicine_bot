# app.py
import os
import time
from typing import List, Optional

import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr

# ---- config ----
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL = "microsoft/BioGPT-Large"               # or "microsoft/biogpt"
HF_TOKEN = os.environ.get("HF_TOKEN")             # set this in your shell or Space Secrets
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONFAULTHANDLER"] = "1"

# ---- load RAG assets ----
index = faiss.read_index("faiss_index.bin")
texts = np.load("indexed_texts.npy", allow_pickle=True)
embedder = SentenceTransformer(EMBEDDER_NAME)

def _truncate(t: str, n: int = 600) -> str:
    return (t[:n] + "…") if len(t) > n else t

def _guard_index():
    if index.ntotal == 0:
        raise RuntimeError("FAISS index is empty. Rebuild it first (rebuild_index.py).")

def _guard_dim(qv: np.ndarray):
    if qv.shape[1] != index.d:
        raise RuntimeError(
            f"Embedder dim ({qv.shape[1]}) != FAISS dim ({index.d}). "
            "Rebuild the index with the SAME embedder."
        )

def retrieve(query: str, k: int = 6, keep: Optional[int] = 2) -> List[str]:
    _guard_index()
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32").reshape(1, -1)
    _guard_dim(qv)
    D, I = index.search(qv, k)
    ids = [i for i in I[0] if i >= 0]
    docs = [_truncate(str(texts[i])) for i in ids[: (keep or 2)]]
    return docs

def generate_answer(question: str, context: str) -> str:
    if not HF_TOKEN:
        return "HF token not set on server. Set HF_TOKEN and restart."

    prompt = (
        "You are a careful medical assistant.\n"
        "Answer ONLY using the context below in 1–3 sentences.\n"
        "If not present in the context, say: \"I don't know based on the provided context.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

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

    # small retry for cold starts/rate limits
    last_err = None
    for _ in range(3):
        try:
            r = requests.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=60)
            if r.status_code in (503, 429, 504):
                time.sleep(3)
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            return "No text generated—please try again."
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(2)
    return f"Generation error: {last_err}"

def rag_chat(question: str):
    question = (question or "").strip()
    if not question:
        return "", " ", ""
    try:
        sources = retrieve(question, k=6, keep=2)
        answer = generate_answer(question, "\n\n".join(sources))
        # pretty-print sources as bullets
        src_md = "\n".join([f"- {s}" for s in sources]) if sources else "_No sources_"
        return answer, src_md, ""
    except Exception as e:
        return "", "", f"Error: {type(e).__name__}: {e}"

with gr.Blocks(title="Medicine RAG Bot") as demo:
    gr.Markdown("## Medicine RAG Bot\n**Disclaimer:** This tool is not a substitute for professional medical advice.")
    q = gr.Textbox(label="Ask a question about a medicine", placeholder="e.g., What are the side effects of Metformin?")
    btn = gr.Button("Ask")
    ans = gr.Markdown(label="Answer")
    src = gr.Markdown(label="Top Sources")
    err = gr.Markdown(label="Errors", visible=True)
    btn.click(fn=rag_chat, inputs=q, outputs=[ans, src, err])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)