import hashlib
import json
import os
from typing import Dict, Any, List
import time
import logging
from llm_processor import LLMProcessor

logger = logging.getLogger(__name__)

class SmartCache:
    """Smart caching system to reduce API calls"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # In-memory cache for fast access
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info(f"Initialized SmartCache with directory: {cache_dir}")
    
    def _get_cache_key(self, question: str, context_chunks: List[Dict]) -> str:
        """Generate unique cache key for question + context combination"""
        # Create a hash of question + top chunks text
        content = question.lower().strip()
        
        # Add top 2 chunks to ensure context similarity
        if context_chunks:
            for chunk in context_chunks[:2]:
                content += chunk.get('text', '')[:200]  # First 200 chars of each chunk
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_answer(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Get cached answer if available"""
        cache_key = self._get_cache_key(question, context_chunks)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            cached_data = self.memory_cache[cache_key]
            cached_data['from_cache'] = True
            return cached_data
        
        # Check file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Add to memory cache
                self.memory_cache[cache_key] = cached_data
                self.cache_hits += 1
                cached_data['from_cache'] = True
                return cached_data
                
            except Exception as e:
                logger.error(f"Cache read error: {e}")
        
        self.cache_misses += 1
        return None
    
    def save_answer(self, question: str, context_chunks: List[Dict], answer_data: Dict[str, Any]):
        """Save answer to cache"""
        cache_key = self._get_cache_key(question, context_chunks)
        
        # Add metadata
        cache_data = answer_data.copy()
        cache_data['cached_at'] = time.time()
        cache_data['cache_key'] = cache_key
        
        # Save to memory cache
        self.memory_cache[cache_key] = cache_data
        
        # Save to file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_cache_size": len(self.memory_cache),
            "total_requests": total_requests
        }
    
    def find_similar_questions(self, question: str, threshold: float = 0.7) -> List[str]:
        """Find similar cached questions using simple word overlap"""
        question_words = set(question.lower().split())
        similar_questions = []
        
        for cache_key, cached_data in self.memory_cache.items():
            if 'original_question' in cached_data:
                cached_question = cached_data['original_question']
                cached_words = set(cached_question.lower().split())
                
                # Calculate Jaccard similarity
                intersection = question_words.intersection(cached_words)
                union = question_words.union(cached_words)
                similarity = len(intersection) / len(union) if union else 0
                
                if similarity >= threshold:
                    similar_questions.append({
                        'question': cached_question,
                        'similarity': similarity,
                        'answer': cached_data.get('answer', '')[:100] + '...'
                    })
        
        return sorted(similar_questions, key=lambda x: x['similarity'], reverse=True)


class DecisionEngine:
    """Enhanced Decision Engine with smart caching"""
    
    def __init__(self, llm_processor: LLMProcessor):
        self.llm_processor = llm_processor
        self.cache = SmartCache()
        logger.info("Initialized Decision Engine with smart caching")
    
    def generate_answer(self, question: str, parsed_query: Dict[str, Any], 
                       context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer with caching support"""
        
        # Check cache first
        cached_answer = self.cache.get_cached_answer(question, context_chunks)
        if cached_answer:
            logger.info(f"✓ Cache hit for question: {question[:50]}...")
            return cached_answer
        
        logger.info(f"✗ Cache miss, generating new answer for: {question[:50]}...")
        
        # Generate new answer
        try:
            # Your existing answer generation logic
            analysis = self._analyze_context_quality(question, context_chunks)
            
            # Try LLM first
            llm_result = None
            try:
                llm_result = self.llm_processor.generate_answer(
                    question=question, 
                    context="", 
                    retrieved_chunks=context_chunks
                )
            except Exception as e:
                logger.warning(f"LLM processing failed, using fallback: {str(e)}")
                llm_result = self._enhanced_fallback_answer(question, context_chunks)
            
            # Enhance the answer
            enhanced_answer = self._enhance_answer(
                question=question,
                llm_result=llm_result,
                parsed_query=parsed_query,
                context_chunks=context_chunks,
                analysis=analysis
            )
            
            # Add original question for similarity matching
            enhanced_answer['original_question'] = question
            enhanced_answer['from_cache'] = False
            
            # Save to cache
            self.cache.save_answer(question, context_chunks, enhanced_answer)
            
            return enhanced_answer
            
        except Exception as e:
            logger.error(f"Error in decision engine: {str(e)}")
            fallback_answer = self._enhanced_fallback_answer(question, context_chunks)
            fallback_answer['original_question'] = question
            return fallback_answer
    
    def get_cache_stats(self):
        """Get caching statistics"""
        return self.cache.get_stats()
    
    def _analyze_context_quality(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the quality and relevance of retrieved context"""
        if not chunks:
            return {
                "quality": "poor",
                "relevance": 0.0,
                "coverage": "incomplete",
                "missing_info": ["No relevant document sections found"]
            }
        
        # Calculate average relevance score
        avg_relevance = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
        
        # Analyze coverage
        total_length = sum(len(chunk['text']) for chunk in chunks)
        
        # Determine quality rating
        if avg_relevance > 0.8 and total_length > 1000:
            quality = "excellent"
        elif avg_relevance > 0.6 and total_length > 500:
            quality = "good"
        elif avg_relevance > 0.4:
            quality = "fair"
        else:
            quality = "poor"
        
        return {
            "quality": quality,
            "relevance": avg_relevance,
            "coverage": "complete" if total_length > 800 else "partial",
            "chunk_count": len(chunks),
            "total_context_length": total_length
        }
    
    def _enhanced_fallback_answer(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced fallback answer when LLM processing fails"""
        if not chunks:
            return {
                "answer": "I could not find relevant information in the document to answer your question.",
                "reasoning": "No relevant chunks retrieved from the document.",
                "confidence": "low",
                "supporting_chunks": [],
                "token_usage": 0,
                "metadata": {"fallback_used": True}
            }
        
        # Sort chunks by relevance
        sorted_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)
        best_chunk = sorted_chunks[0]
        
        # Try to extract answer using pattern matching and rules
        answer = self._extract_answer_with_rules(question, best_chunk['text'])
        
        if not answer:
            # Fallback to chunk summary
            answer = f"Based on the document: {best_chunk['text'][:300]}..."
        
        return {
            "answer": answer,
            "reasoning": f"Answer extracted from document section with {best_chunk.get('relevance_score', 0):.1%} relevance match using rule-based processing.",
            "confidence": self._assess_confidence_from_score(best_chunk.get('relevance_score', 0)),
            "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(sorted_chunks[:3])],
            "token_usage": 0,
            "metadata": {"fallback_used": True}
        }
    
    def _extract_answer_with_rules(self, question: str, text: str) -> str:
        """Extract answers using rule-based pattern matching"""
        import re
        
        question_lower = question.lower()
        text_lower = text.lower()
        
        # Common question patterns and extraction rules
        patterns = {
            'waiting period': r'waiting period[^\d]*(\d+)\s*(month|year|day)s?',
            'grace period': r'grace period[^\d]*(\d+)\s*(day|month)s?',
            'coverage': r'(cover|coverage)[^.]*[.!]',
            'percentage': r'(\d+)%',
            'amount': r'(\d+(?:,\d{3})*)\s*(rupee|dollar|\$|₹)',
            'definition': r'defined as[^.]*[.]',
            'benefit': r'benefit[^.]*[.]',
            'condition': r'condition[^.]*[.]'
        }
        
        # Check which pattern matches the question
        for pattern_name, pattern in patterns.items():
            if pattern_name in question_lower:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Find the sentence containing this match
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if match.group(0).lower() in sentence.lower():
                            return sentence.strip() + "."
        
        # If no specific pattern matches, look for direct question keywords
        question_keywords = re.findall(r'\b\w+\b', question_lower)
        best_sentence = ""
        max_keyword_matches = 0
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            keyword_matches = sum(1 for keyword in question_keywords if keyword in sentence_lower)
            if keyword_matches > max_keyword_matches:
                max_keyword_matches = keyword_matches
                best_sentence = sentence.strip()
        
        return best_sentence + "." if best_sentence else ""
    
    def _assess_confidence_from_score(self, relevance_score: float) -> str:
        """Assess confidence from relevance score"""
        if relevance_score > 0.8:
            return "high"
        elif relevance_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _enhance_answer(self, question: str, llm_result: Dict[str, Any], 
                       parsed_query: Dict[str, Any], context_chunks: List[Dict[str, Any]],
                       analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the LLM-generated answer with additional structured information"""
        # Extract key information from the answer
        if isinstance(llm_result, dict):
            answer_text = llm_result.get("answer", "")
            confidence = llm_result.get("confidence", "medium")
            token_usage = llm_result.get("token_usage", 0)
        else:
            answer_text = str(llm_result)
            confidence = "medium"
            token_usage = 0
        
        # Add structured metadata
        enhanced_result = {
            "answer": answer_text,
            "confidence": confidence,
            "reasoning": f"Answer generated using {analysis['quality']} quality context from {len(context_chunks)} document sections.",
            "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(context_chunks[:3])],
            "token_usage": token_usage,
            "metadata": {
                "question_intent": parsed_query.get("intent", "unknown"),
                "domain": parsed_query.get("domain", "unknown"),
                "context_quality": analysis["quality"],
                "confidence_level": confidence,
                "processing_stats": {
                    "chunks_analyzed": len(context_chunks),
                    "avg_relevance": analysis["relevance"],
                    "token_usage": token_usage
                },
                "fallback_used": False
            }
        }
        
        return enhanced_result