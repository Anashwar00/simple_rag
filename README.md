
# 🤖 Machine Learning RAG App (PDF-based)

This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to upload a **PDF book on machine learning**, ask questions about the content, and get intelligent answers based on **retrieved document chunks**. The project includes both a **Streamlit web interface** and a **command-line interface (CLI)** version.

---

## 🔍 Key Features

- 📄 Upload your own **PDF file** (e.g., ML textbooks).
- 📚 Text is automatically **chunked** and stored in **ChromaDB** with **OpenAI embeddings**.
- 💬 Ask natural language questions, get context-aware answers from **GPT-4**.
- ⚙️ Powered by **OpenAI**, **ChromaDB**, **Streamlit**, and **PyPDF2**.

---

## 🛠️ Tech Stack

- **Streamlit** – Interactive UI
- **ChromaDB** – Vector database for context retrieval
- **OpenAI** – GPT-4 for answering questions
- **PyPDF2** – PDF parsing and text extraction
- **dotenv** – Environment variable management

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Anashwar00/simple_rag/tree/main
cd simple_rag
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Set your OpenAI API Key->Create a .env file in the root directory:
```bash
OPENAI_API_KEY=your_openai_key_here
```
### 4. ▶️ Web App (Streamlit)
```bash
streamlit run main_streamlit.py
```

### Upload a PDF file (e.g., an ML textbook).

### Ask questions like:

"What is supervised learning?"

"Explain overfitting."

See contextual responses generated from the document.


