from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np

# load abstracts from file

with open("medicine_abstracts.txt", "r", encoding = "utf-8") as f:
    texts = f.read().split("\n\n") # each abstract is seprated by 2 double lines

# removing empty chunks
texts = [t.strip() for t in texts if t.strip() != ""]
print("Number of texts loaded", len(texts))

# loading Biobert- based embedding model

model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

#embedding all abstracts
print("Genrating embeddings.......")
embeddings = model.encode(texts)

# creating FAISS index

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# save the FAISS index and texts

faiss.write_index(index, "faiss_index.bin")
np.save("indexed_texts.npy", np.array(texts))

print("FAISS index created and saved.")
