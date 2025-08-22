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

python -m venv .venv
.venv\Scripts\Activate.ps1


## pip install --upgrade pip
pip install -r requirements.txt
