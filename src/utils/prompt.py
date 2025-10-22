# src/utils/prompt.py

def build_prompt(question, contexts):
    context_text = "\n".join([c.get('text', '') for c in contexts])
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    return prompt
