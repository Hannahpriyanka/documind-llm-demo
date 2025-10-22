# src/utils/faiss_store.py
import faiss
import json
from sentence_transformers import SentenceTransformer

class FaissStore:
    def __init__(self, index_dir):
        self.index = faiss.read_index(f"{index_dir}/index.faiss")
        with open(f"{index_dir}/meta.json", 'r') as f:
            self.metas = json.load(f)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def query(self, text, top_k=5):
        vec = self.model.encode([text])
        D, I = self.index.search(vec, top_k)
        results = []
        for idx in I[0]:
            # return full meta including text
            results.append(self.metas[idx])
        return results
