from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
import re
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processor with smarter chunking"""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Insurance/Legal domain keywords for better chunking
        self.domain_keywords = [
            'policy', 'coverage', 'premium', 'claim', 'benefit', 'waiting period',
            'grace period', 'exclusion', 'deductible', 'sum insured', 'hospital',
            'treatment', 'surgery', 'maternity', 'pre-existing', 'discount'
        ]
    
    def create_chunks(self, text: str) -> List[Dict[str, any]]:
        """
        Create semantic chunks optimized for insurance/legal documents
        """
        # Clean and preprocess text
        text = self._clean_text_enhanced(text)
        
        # Split by sections first (look for headers/titles)
        sections = self._split_by_sections(text)
        
        chunks = []
        chunk_id = 0
        
        for section_idx, section_text in enumerate(sections):
            if len(section_text.strip()) < 50:  # Skip very short sections
                continue
                
            # Create chunks within each section
            section_chunks = self._create_section_chunks(section_text, section_idx)
            
            for chunk_data in section_chunks:
                chunk_data['id'] = chunk_id
                chunks.append(chunk_data)
                chunk_id += 1
        
        # If no sections found, fallback to sentence-based chunking
        if not chunks:
            chunks = self._create_sentence_chunks(text)
        
        logger.info(f"Created {len(chunks)} optimized chunks from document")
        return chunks
    
    def _clean_text_enhanced(self, text: str) -> str:
        """Enhanced text cleaning for insurance documents"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Fix split numbers
        text = re.sub(r'([a-z])\s+([A-Z])', r'\1. \2', text)  # Fix sentence boundaries
        
        # Preserve important formatting
        text = re.sub(r'(\d+\.)\s*([A-Z])', r'\1 \2', text)  # Numbered lists
        
        return text.strip()
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by logical sections"""
        # Look for section headers
        section_patterns = [
            r'\n\s*(\d+\.?\s*[A-Z][^.]*?)[\n:]',  # Numbered sections
            r'\n\s*([A-Z][A-Z\s]{10,}?)[\n:]',    # ALL CAPS headers
            r'\n\s*(SECTION\s+\d+|CLAUSE\s+\d+|PART\s+[A-Z])',  # Formal sections
        ]
        
        sections = []
        current_section = ""
        
        lines = text.split('\n')
        for line in lines:
            is_header = False
            for pattern in section_patterns:
                if re.match(pattern, '\n' + line, re.IGNORECASE):
                    is_header = True
                    break
            
            if is_header and current_section.strip():
                sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        # Add final section
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections if len(sections) > 1 else [text]  # Fallback to full text
    
    def _create_section_chunks(self, section_text: str, section_idx: int) -> List[Dict]:
        """Create chunks within a section"""
        sentences = self._split_into_sentences_smart(section_text)
        chunks = []
        
        current_chunk = ""
        start_sentence = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_data = {
                    'text': current_chunk.strip(),
                    'section_idx': section_idx,
                    'start_sentence': start_sentence,
                    'end_sentence': i,
                    'length': len(current_chunk),
                    'domain_score': self._calculate_domain_relevance(current_chunk)
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_sentences = sentences[max(0, i - 2):i]  # Take last 2 sentences
                current_chunk = ' '.join(overlap_sentences) + ' ' if overlap_sentences else ""
                start_sentence = max(0, i - 2)
            
            current_chunk += sentence + ' '
        
        # Add final chunk
        if current_chunk.strip():
            chunk_data = {
                'text': current_chunk.strip(),
                'section_idx': section_idx,
                'start_sentence': start_sentence,
                'end_sentence': len(sentences),
                'length': len(current_chunk),
                'domain_score': self._calculate_domain_relevance(current_chunk)
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_sentence_chunks(self, text: str) -> List[Dict]:
        """Fallback sentence-based chunking"""
        sentences = self._split_into_sentences_smart(text)
        chunks = []
        chunk_id = 0
        
        current_chunk = ""
        start_idx = 0
        
        for i, sentence in enumerate(sentences):
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'start_sentence': start_idx,
                    'end_sentence': i,
                    'length': len(current_chunk),
                    'domain_score': self._calculate_domain_relevance(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_sentences = sentences[max(0, i - 1):i]
                current_chunk = ' '.join(overlap_sentences) + ' ' if overlap_sentences else ""
                start_idx = max(0, i - 1)
                chunk_id += 1
            
            current_chunk += sentence + ' '
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'start_sentence': start_idx,
                'end_sentence': len(sentences),
                'length': len(current_chunk),
                'domain_score': self._calculate_domain_relevance(current_chunk)
            })
        
        return chunks
    
    def _split_into_sentences_smart(self, text: str) -> List[str]:
        """Smarter sentence splitting for legal/insurance documents"""
        # Handle special cases first
        text = re.sub(r'(\d+)\.\s*(\d+)', r'\1DECIMAL\2', text)  # Protect decimals
        text = re.sub(r'([A-Z][a-z]+)\.\s*([A-Z][a-z]+)', r'\1ABBREV \2', text)  # Protect abbreviations
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore protected text
        sentences = [s.replace('DECIMAL', '.').replace('ABBREV', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _calculate_domain_relevance(self, text: str) -> float:
        """Calculate how relevant a chunk is to insurance/legal domain"""
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in self.domain_keywords if keyword in text_lower)
        
        # Bonus for numbers (often important in insurance)
        number_count = len(re.findall(r'\d+', text))
        
        # Normalize score
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        domain_score = (keyword_count * 2 + number_count * 0.5) / total_words
        return min(domain_score, 1.0)  # Cap at 1.0


class SemanticSearch:
    """Enhanced semantic search with better ranking"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 256  # Limit sequence length for speed
        logger.info(f"Loaded embedding model: {model_name}")
    
    def build_index(self, chunks: List[Dict[str, any]]) -> Tuple[faiss.Index, np.ndarray]:
        """
        Build optimized FAISS index with faster settings
        """
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings with progress tracking
        logger.info("Generating embeddings for chunks...")
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False,  # Disable for speed
            batch_size=32,  # Larger batch size for efficiency
            convert_to_numpy=True
        )
        
        # Build FAISS index with faster configuration
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for exact search (faster than other options)
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        logger.info(f"Built FAISS index with {index.ntotal} vectors of dimension {dimension}")
        return index, embeddings
    
    def search(self, index_data: Tuple[faiss.Index, np.ndarray], query: str, 
               chunks: List[Dict[str, any]], top_k: int = 3) -> List[Dict[str, any]]:
        """
        Enhanced search with re-ranking based on domain relevance
        """
        index, _ = index_data
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search for more candidates than needed
        search_k = min(top_k * 2, len(chunks))  # Search 2x candidates for re-ranking
        scores, indices = index.search(query_embedding.astype('float32'), search_k)
        
        # Re-rank results combining semantic similarity with domain relevance
        candidates = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(chunks):  # Valid index
                chunk = chunks[idx].copy()
                
                # Combine semantic score with domain relevance
                semantic_score = float(score)
                domain_score = chunk.get('domain_score', 0.0)
                
                # Enhanced scoring with query-specific boosting
                final_score = self._calculate_enhanced_score(
                    semantic_score, domain_score, query, chunk['text']
                )
                
                chunk['relevance_score'] = final_score
                chunk['semantic_score'] = semantic_score
                chunk['domain_score'] = domain_score
                chunk['rank'] = i + 1
                candidates.append(chunk)
        
        # Sort by enhanced score and return top_k
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        results = candidates[:top_k]
        
        # Fixed logging - avoid nested f-strings
        score_values = []
        for r in results:
            score_values.append(f"{r['relevance_score']:.3f}")
        
        logger.info(f"Retrieved {len(results)} relevant chunks (scores: {score_values})")
        return results
    
    def _calculate_enhanced_score(self, semantic_score: float, domain_score: float, 
                                 query: str, chunk_text: str) -> float:
        """Calculate enhanced relevance score"""
        
        # Base score from semantic similarity
        base_score = semantic_score
        
        # Domain relevance boost
        domain_boost = domain_score * 0.1
        
        # Query-specific boosting
        query_boost = self._calculate_query_boost(query, chunk_text)
        
        # Length penalty for very short chunks
        length_penalty = 0.0
        if len(chunk_text) < 100:
            length_penalty = -0.05
        
        # Combine scores
        final_score = base_score + domain_boost + query_boost + length_penalty
        
        return max(0.0, min(1.0, final_score))  # Clamp between 0 and 1
    
    def _calculate_query_boost(self, query: str, chunk_text: str) -> float:
        """Calculate query-specific boost based on exact matches"""
        query_lower = query.lower()
        chunk_lower = chunk_text.lower()
        
        boost = 0.0
        
        # Exact phrase matching
        query_phrases = self._extract_key_phrases(query_lower)
        for phrase in query_phrases:
            if phrase in chunk_lower:
                boost += 0.05
        
        # Number matching (important for insurance queries)
        query_numbers = re.findall(r'\d+', query)
        chunk_numbers = re.findall(r'\d+', chunk_text)
        common_numbers = set(query_numbers).intersection(set(chunk_numbers))
        boost += len(common_numbers) * 0.03
        
        # Time period matching
        time_patterns = ['month', 'year', 'day', 'period']
        for pattern in time_patterns:
            if pattern in query_lower and pattern in chunk_lower:
                boost += 0.02
        
        return min(boost, 0.2)  # Cap boost at 0.2
    
    def _extract_key_phrases(self, query: str) -> List[str]:
        """Extract key phrases from query"""
        # Common insurance/legal phrases
        key_phrases = []
        
        phrase_patterns = [
            r'waiting period',
            r'grace period',
            r'pre.?existing',
            r'room rent',
            r'sum insured',
            r'no claim discount',
            r'health check',
            r'organ donor',
            r'ayush treatment'
        ]
        
        for pattern in phrase_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            key_phrases.extend(matches)
        
        return key_phrases
    
    def get_query_keywords(self, query: str) -> List[str]:
        """
        Enhanced keyword extraction for insurance/legal queries
        """
        keywords = []
        
        # Domain-specific terms with higher weight
        domain_terms = re.findall(
            r'\b(?:policy|coverage|premium|claim|benefit|waiting|grace|exclusion|deductible|hospital|treatment|surgery|maternity|discount)\b', 
            query.lower()
        )
        keywords.extend(domain_terms)
        
        # Medical terms
        medical_terms = re.findall(
            r'\b(?:surgery|operation|treatment|condition|disease|therapy|medical|clinical)\b', 
            query.lower()
        )
        keywords.extend(medical_terms)
        
        # Numbers and time periods (crucial for insurance)
        numbers = re.findall(r'\b\d+\b', query)
        keywords.extend(numbers)
        
        time_units = re.findall(r'\b(?:month|year|day)s?\b', query.lower())
        keywords.extend(time_units)
        
        # Important entities (capitalized terms)
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        keywords.extend([term.lower() for term in entities])
        
        return list(set(keywords))