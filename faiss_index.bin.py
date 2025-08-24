cat > rebuild_index.py << 'PY'
import numpy as np, faiss, sys
from sentence_transformers import SentenceTransformer

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match app.py

def main():
    try:
        with open("medicine_abstracts.txt","r",encoding="utf-8") as f:
            texts = [t.strip() for t in f if t.strip()]
    except FileNotFoundError:
        print("❌ medicine_abstracts.txt not found."); sys.exit(1)

    print("Loaded texts:", len(texts))
    if not texts:
        print("❌ No texts found."); sys.exit(1)

    print("Loading embedder:", EMBEDDER_NAME, flush=True)
    m = SentenceTransformer(EMBEDDER_NAME)

    print("Encoding...", flush=True)
    X = m.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    print("Embeddings shape:", X.shape)  # expect (N, 384)

    idx = faiss.IndexFlatIP(X.shape[1])  # cosine via inner-product on normalized vectors
    idx.add(X)
    print(f"FAISS ntotal: {idx.ntotal}  dim: {idx.d}")

    faiss.write_index(idx, "faiss_index.bin")
    np.save("indexed_texts.npy", np.array(texts, dtype=object))
    print("✅ Wrote faiss_index.bin and indexed_texts.npy")

if __name__ == "__main__":
    main()
PY

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
python rebuild_index.py