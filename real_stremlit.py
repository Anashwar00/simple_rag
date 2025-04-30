import streamlit as st
import openai
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import dotenv
from dotenv import load_dotenv
import os
import uuid
import re
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Class for embedding model
class EmbeddingModel:
    def __init__(self, model_type):
        if model_type == "openai":
            self.embd_model = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY")
            )

    def gen_model(self):
        return self.embd_model


# Class for LLM (Large Language Model)
class LLMType:
    def __init__(self, model):
        if model == "openai":
            self.client = OpenAI()
            self.model = "gpt-4"

    def generate_ans(self, message):
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=0.7
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"Error: {str(e)}")


# Function to process PDF and chunk it
def chunking(uploaded_file):
    document = []
    try:
        # Use PyPDF2 to read the PDF directly from the uploaded BytesIO object
        reader = PdfReader(uploaded_file)
        num_pages = len(reader.pages)
        for i in range(num_pages):
            page = reader.pages[i]
            page_text = page.extract_text()
            if not page_text:
                continue  # Skip pages with no text
            cleaned_text = re.sub(r'[\x00-\x1F\x7F]+', '', page_text)
            new_text = re.sub(r"Download from finelybook.*?www\.finelybook\.com", "", cleaned_text)
            words = new_text.split()
            st, end = 0, 100
            while st < len(words):
                chunk_words = words[st:end]
                sec_text = " ".join(chunk_words)
                document.append(sec_text.encode('utf-8', 'replace').decode('utf-8'))
                st = end - 20  # overlap
                end += 100
    except Exception as e:
        print(f"Error in chunking: {str(e)}")
    return document


# Function to set up the Chroma database
def database_setup(embd, document):
    client = chromadb.PersistentClient("./db/chroma")
    collection = client.get_or_create_collection(
        "ml_collection",
        embedding_function=embd
    )
    ids = [str(uuid.uuid4()) for _ in document]
    collection.add(
        ids=ids,
        documents=document
    )
    return collection


# Function to handle user queries and retrieve relevant chunks from the database
def user_query(query, collection):
    res = collection.query(
        query_texts=[query],
        n_results=10
    )
    return res


# Function to augment the prompt with context
def augment_prompt(query, context):
    flat_docs = []
    for doc in context['documents']:
        if isinstance(doc, list):
            flat_docs.extend(doc)
        else:
            flat_docs.append(doc)
    context_str = "\n".join(flat_docs)
    prompt = f"""You are a very helpful assistant who helps people to answer ML related questions by using
context below-
context:{context_str}

query:{query}
"""
    return prompt


# Function to handle the RAG pipeline
def rag_pipeline(query, collection, llm):
    chunk = user_query(query, collection)
    prompt = augment_prompt(query, chunk)
    message = [{"role": "system", "content": "You are a helpful assistant"}, {
        "role": "user", "content": prompt
    }]
    response = llm.generate_ans(message)
    return response


# Streamlit app
def main():
    st.title("RAG System with Streamlit")

    # Initialize models
    emb = EmbeddingModel("openai")
    embd = emb.gen_model()
    llm = LLMType("openai")

    # File upload
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file:
        # Process PDF and chunk it
        chunks = chunking(uploaded_file)
        collection = database_setup(embd, chunks)

        # Display query input
        query = st.text_input("Ask a question:")

        if query:
            # Run the RAG pipeline
            response = rag_pipeline(query, collection, llm)

            # Display response and sources
            st.write(response)
            st.sidebar.write("Sources:")
            for doc in collection.query(query_texts=[query], n_results=10)["documents"]:
                st.sidebar.write(doc)

    else:
        st.write("Please upload a PDF document to get started.")


if __name__ == "__main__":
    main()
