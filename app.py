from flask import Flask , request, render_template
import numpy as np
import faiss 
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer , AutoModelForCausalLM
import traceback

app = Flask(__name__)

# Loading FAISS index and saved texts 
index = faiss.read_index("faiss_index.bin")
texts = np.load("indexed_texts.npy", allow_pickle = True)

# Loading embedding model

embedder = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

# Loading bioGPT model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

@app.route("/", methods = ["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        try:
            question = request.form.get("question")
            print("Received question:", question)
            # step 1 : Embed user question
            query_vec = embedder.encode([question])

            #step 2 : Search FAISS for top 3 results
            D, I = index.search(np.array(query_vec), k=3)
            top_docs = [texts[i] for i in I[0]]
            context = "\n\n".join(top_docs)

            # step 3: ASk BioGPT
            # Prepare prompt
            prompt = f"{question} Context: {context}"

            # Tokenize input
            input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

            # Generate response
            output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)
            raw_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract final part of response
            answer = raw_answer.split("Context:")[-1].strip()

            #step 4 Extract only the final answer
            answer = raw_answer.split("Answer:")[-1].strip()

        except Exception as e:
            print("❌ ERROR OCCURRED")
            traceback.print_exc()
            answer = f"Sorry, something went wrong: {str(e)}"       

    return render_template("index.html",answer=answer)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)




