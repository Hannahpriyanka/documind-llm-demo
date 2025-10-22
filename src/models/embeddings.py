# src/models/embeddings.py
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import argparse

def build_index(dataset_file, index_dir):
    os.makedirs(index_dir, exist_ok=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = []
    metas = []

    with open(dataset_file, 'r') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            texts.append(obj['text'])
            # Store text + metadata in meta.json
            metas.append({
                "id": i,
                "text": obj['text'],
                "metadata": obj.get('metadata', {})
            })

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, 'index.faiss'))
    with open(os.path.join(index_dir, 'meta.json'), 'w') as f:
        json.dump(metas, f, indent=2)

    print(f"Built FAISS index with {len(texts)} entries at {index_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help="Input JSONL file")
    parser.add_argument('--index-dir', required=True, help="Output index directory")
    args = parser.parse_args()
    build_index(args.dataset, args.index_dir)
