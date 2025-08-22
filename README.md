# 📘 LLM-Powered Query-Retrieval System  

A smart system that extracts text from PDF documents, performs semantic search, and answers user questions using **Groq LLMs**. Optimized for **insurance/legal documents** with caching, parallel processing, and high accuracy.  

---

## ✨ Features
- 🚀 FastAPI backend with async & caching  
- 📑 PDF text extraction (PyMuPDF)  
- 🔎 Semantic Search with FAISS + Sentence Transformers  
- 🧠 Smart Caching Engine (reduces redundant LLM API calls)  
- ⚡ Parallel question answering for speed & scale  
- 🔒 Secure API with Bearer token authentication  

---

## ✅ Prerequisites
- Python **3.10 or 3.11**  
- `git` installed  
- Internet connection (for downloading PDFs and LLM API calls)  
- (Optional) **Groq API key** for best answers. Without it, fallback extraction still works.  

---

## 🛠 Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

```
## 🛠 Step 2: Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
Activate it:

Linux / macOS:
source .venv/bin/activate

Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

## 🛠 Step 3: Install Dependencies
```bash
## pip install --upgrade pip
pip install -r requirements.txt

```
## 🛠 Step 4: Create .env file
## In the project root, create a file called .env and add:
```bash
BEARER_TOKEN=super-secret-token-123   # Your chosen API token
GROQ_API_KEY=your_groq_api_key_here   # Optional (for LLM answers)
# Or multiple keys:
# GROQ_API_KEYS=key1,key2,key3
```

## 🛠 Step 5: Run the App

```bash
uvicorn app:app --reload
```

