import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
import tempfile
from typing import List, Dict, Any, Optional
import hashlib
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import logging

from extract import extract_text_from_pdf
from search import DocumentProcessor, SemanticSearch
from llm_processor import LLMProcessor
from decision_engine import DecisionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM-Powered Query-Retrieval System", version="1.0.0")

# Global instances with caching
document_processor = DocumentProcessor()
semantic_search = SemanticSearch()
llm_processor = LLMProcessor()
decision_engine = DecisionEngine(llm_processor)

# In-memory cache for processed documents
DOCUMENT_CACHE = {}
EMBEDDING_CACHE = {}

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def get_document_hash(url: str) -> str:
    """Generate hash for document URL for caching"""
    return hashlib.md5(url.encode()).hexdigest()

async def process_document_cached(document_url: str) -> tuple:
    """Process document with caching"""
    doc_hash = get_document_hash(document_url)
    
    # Check if already processed
    if doc_hash in DOCUMENT_CACHE:
        logger.info(f"Using cached document processing for {doc_hash}")
        return DOCUMENT_CACHE[doc_hash], EMBEDDING_CACHE[doc_hash]
    
    # Download and extract
    start_time = time.time()
    document_content = await download_and_extract_document(document_url)
    logger.info(f"Document extraction took {time.time() - start_time:.2f}s")
    
    # Process chunks
    start_time = time.time()
    chunks = document_processor.create_chunks(document_content)
    logger.info(f"Chunking took {time.time() - start_time:.2f}s")
    
    # Build search index
    start_time = time.time()
    search_index = semantic_search.build_index(chunks)
    logger.info(f"Embedding generation took {time.time() - start_time:.2f}s")
    
    # Cache results
    DOCUMENT_CACHE[doc_hash] = chunks
    EMBEDDING_CACHE[doc_hash] = search_index
    
    return chunks, search_index

async def process_questions_parallel(questions: List[str], chunks: List[Dict], search_index) -> List[str]:
    """Process questions in parallel for better performance"""
    
    async def process_single_question(question: str) -> str:
        try:
            # Parse query (fast operation)
            parsed_query = llm_processor.parse_query(question)
            
            # Retrieve relevant chunks
            relevant_chunks = semantic_search.search(search_index, question, chunks, top_k=3)  # Reduced from 5 to 3
            
            # Generate answer with timeout
            loop = asyncio.get_event_loop()
            answer_result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    decision_engine.generate_answer,
                    question,
                    parsed_query,
                    relevant_chunks
                ),
                timeout=5.0  # 5 second timeout per question
            )
            
            return answer_result["answer"]
            
        except asyncio.TimeoutError:
            logger.warning(f"Question processing timed out: {question[:50]}...")
            return "Unable to process question within time limit."
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"Error processing question: {str(e)[:100]}"
    
    # Process questions in parallel with limited concurrency
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent questions
    
    async def process_with_semaphore(question):
        async with semaphore:
            return await process_single_question(question)
    
    tasks = [process_with_semaphore(q) for q in questions]
    answers = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    processed_answers = []
    for i, answer in enumerate(answers):
        if isinstance(answer, Exception):
            logger.error(f"Question {i+1} failed: {str(answer)}")
            processed_answers.append("Error processing question.")
        else:
            processed_answers.append(answer)
    
    return processed_answers

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(req: QueryRequest, authorization: str = Header(None)):
    """
    Optimized main endpoint with caching and parallel processing
    """
    start_time = time.time()
    
    # Validate authorization
    expected_token = f"Bearer {os.getenv('BEARER_TOKEN', '16ca23504efb8f8b98b1d84b2516a4b6ccb69f3c955ac9a8107497f5d14d6dbb')}"
    if not authorization or authorization != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    try:
        logger.info(f"Processing {len(req.questions)} questions for document: {req.documents}")
        
        # Step 1: Process document with caching
        chunks, search_index = await process_document_cached(req.documents)
        
        processing_time = time.time() - start_time
        logger.info(f"Document processing completed in {processing_time:.2f}s")
        
        # Step 2: Process questions in parallel
        question_start = time.time()
        answers = await process_questions_parallel(req.questions, chunks, search_index)
        
        question_time = time.time() - question_start
        total_time = time.time() - start_time
        
        logger.info(f"Questions processed in {question_time:.2f}s, total time: {total_time:.2f}s")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in run_query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def download_and_extract_document(document_url: str) -> str:
    """
    Optimized document download and extraction
    """
    try:
        # Use async HTTP client for better performance
        import httpx
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(document_url)
            response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        try:
            # Extract text in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            document_content = await loop.run_in_executor(
                executor, 
                extract_text_from_pdf, 
                temp_path
            )
            
            if not document_content.strip():
                raise ValueError("No text content extracted from document")
            
            logger.info(f"Extracted {len(document_content)} characters from document")
            return document_content
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

# Add this to your existing app.py - just replace the health check endpoint

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with cache statistics"""
    cache_stats = decision_engine.get_cache_stats()
    
    return {
        "status": "healthy", 
        "service": "LLM Query-Retrieval System",
        "document_cache": {
            "documents_cached": len(DOCUMENT_CACHE),
            "embeddings_cached": len(EMBEDDING_CACHE)
        },
        "smart_cache_stats": cache_stats,
        "performance_metrics": {
            "cache_efficiency": f"{cache_stats.get('hit_rate_percent', 0):.1f}%",
            "api_calls_saved": cache_stats.get('cache_hits', 0)
        }
    }

# Also add a new endpoint to get detailed cache info
@app.get("/cache/stats")
async def get_cache_statistics():
    """Get detailed cache statistics"""
    return {
        "cache_stats": decision_engine.get_cache_stats(),
        "memory_cache_keys": len(decision_engine.cache.memory_cache),
        "cache_directory": decision_engine.cache.cache_dir,
        "cache_files": len([f for f in os.listdir(decision_engine.cache.cache_dir) if f.endswith('.json')]) if os.path.exists(decision_engine.cache.cache_dir) else 0
    }

# Optional: Add endpoint to find similar questions
@app.post("/cache/similar")
async def find_similar_questions(request: dict):
    """Find similar cached questions"""
    question = request.get("question", "")
    threshold = request.get("threshold", 0.7)
    
    similar = decision_engine.cache.find_similar_questions(question, threshold)
    return {
        "query": question,
        "similar_questions": similar,
        "count": len(similar)
    }

# Add startup event to preload models
@app.on_event("startup")
async def startup_event():
    """Preload models on startup"""
    logger.info("Preloading models...")
    # This will load the sentence transformer model
    try:
        semantic_search.model.encode(["test"], show_progress_bar=False)
        logger.info("Models preloaded successfully")
    except Exception as e:
        logger.warning(f"Model preloading failed: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting LLM Query-Retrieval System...")
    print("Server will be available at: http://localhost:8000")
    print("Health check endpoint: http://localhost:8000/health")
    print("API endpoint: http://localhost:8000/hackrx/run")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")