# app_ui.py
import streamlit as st
import requests

# API endpoint
API_URL = "http://127.0.0.1:8080/ask"

st.title("📚 DocuMind RAG Demo")
st.write("Ask questions based on your uploaded knowledge base!")

# User input
question = st.text_input("Enter your question:")

top_k = st.slider("Number of context documents to use", min_value=1, max_value=5, value=3)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question!")
    else:
        payload = {"question": question, "top_k": top_k}
        try:
            response = requests.post(API_URL, json=payload)
            data = response.json()

            st.subheader("Answer:")
            st.write(data.get("answer", "No answer"))

            st.subheader("Sources:")
            for src in data.get("sources", []):
                st.markdown(f"- {src.get('text','')}")
        except Exception as e:
            st.error(f"Error contacting API: {e}")
