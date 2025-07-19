import faiss 
import pickle 
from sentence_transformer import SentenceTransformer
from transformer import AutoTokenizer , AutoModelForCausalLM
import torch

#loading faiss index and data
index = faiss.read_index("faiss_index.bin")

with open("indexed_texts.pkl","rb") as f:
    texts = pickle.load(f)

#loading embedding model and LLM
embedder = SentenceTransformer("sentence-transformer/all-MiniLm-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")

def generate_answer(query, k= 1):
    # embeding the question
    query_vector = embedder.encode([query])

    #Retrieve top-K closest abstracts
    distances, indices = index.search(query_vector,k)
    retrieved_texts = [texts[i] for i in indices[0]]

    #construct the prompt for BioGPT
    prompt  = "Context:\n" + "\n".join(retrieved_texts)+f"\n\nQuestion: {query}nAnswer:"

    #tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation = True, max_length= 1024)
    output = model.generate(**inputs, max_new_tokens= 100, do_sample = True)
    answer = tokenizer.decode(output[0], skip_special_tokens= True)

    # trim the prompt part from answer
    return answer.split("Answer:")[-1].strip()
