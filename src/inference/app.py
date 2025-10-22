# src/inference/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.utils.faiss_store import FaissStore
from src.utils.prompt import build_prompt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI(title="DocuMind RAG API", version="1.0")

# ---------------------------
# Pydantic model for request
# ---------------------------
class AskRequest(BaseModel):
    question: str
    top_k: int = 5

# ---------------------------
# Load FAISS and embedding model
# ---------------------------
print("Loading FAISS index and embeddings...")
faiss_store = FaissStore(index_dir="faiss_index")
print("FAISS loaded.")

# ---------------------------
# Load instruction-tuned LLM (Flan-T5 small)
# ---------------------------
print("Loading LLM model...")
model_name = "google/flan-t5-small"  # CPU-friendly, instruction-tuned
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}.")

# ---------------------------
# Health check endpoint
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# /ask endpoint
# ---------------------------
@app.post("/ask")
def ask(request: AskRequest):
    # Retrieve top_k contexts
    contexts = faiss_store.query(request.question, top_k=request.top_k)

    # Build prompt for LLM
    prompt = build_prompt(request.question, contexts)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate answer
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=4,
        early_stopping=True
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return response with sources
    return {
        "answer": answer,
        "sources": contexts
    }
