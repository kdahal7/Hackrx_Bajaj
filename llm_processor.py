from groq import Groq
import json
from typing import Dict, List, Any
import logging
import os
from pydantic import BaseModel
import time
import re

logger = logging.getLogger(__name__)

class QueryStructure(BaseModel):
    """Structured representation of a parsed query"""
    intent: str
    entities: Dict[str, Any]
    keywords: List[str]
    domain: str
    complexity: str

class GroqClientManager:
    """Manages multiple Groq API keys with rotation and fallback"""
    
    def __init__(self):
        # Load multiple API keys from environment
        api_keys_str = os.getenv("GROQ_API_KEYS", "")
        if api_keys_str:
            self.api_keys = [key.strip() for key in api_keys_str.split(",")]
        else:
            # Fallback to single key
            single_key = os.getenv("GROQ_API_KEY")
            self.api_keys = [single_key] if single_key else []
        
        self.current_key_index = 0
        self.failed_keys = set()
        self.clients = {}
        
        # IMPROVED: Better model selection strategy
        # Try the most capable models first, with fallbacks
        self.model_priority = [
            "meta-llama/llama-4-scout-17b-16e-instruct",  # TPM: 30K
            "gemma2-9b-it",                               # TPM: 15K
            "llama3-70b-8192",                            # TPM: 6K, high quality
            "llama-3.1-8b-instant",                       # Fast fallback
            "deepseek-r1-distill-llama-70b"              # Lowest quota, use last
        ]


        self.current_model = self.model_priority[0]
        
        # Initialize clients for each key
        for i, api_key in enumerate(self.api_keys):
            if api_key:
                try:
                    self.clients[i] = Groq(api_key=api_key)
                    logger.info(f"✓ Initialized API key {i+1}/{len(self.api_keys)}")
                except Exception as e:
                    logger.error(f"✗ Failed to initialize API key {i}: {e}")
        
        logger.info(f"🚀 Initialized with {len(self.api_keys)} API keys using model: {self.current_model}")

    def get_working_client(self):
        """Get a working client, rotating through available keys"""
        if not self.clients:
            return None
        
        # Try current key first
        if self.current_key_index not in self.failed_keys:
            return self.clients.get(self.current_key_index)
        
        # Find next working key
        for _ in range(len(self.api_keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            if self.current_key_index not in self.failed_keys:
                return self.clients.get(self.current_key_index)
        
        # If all keys failed, reset failed keys (they might work again later)
        if len(self.failed_keys) >= len(self.api_keys):
            logger.info("All keys failed, resetting and trying again...")
            self.failed_keys.clear()
            time.sleep(5)  # Wait 5 seconds before retrying
            return self.clients.get(0) if self.clients else None

    def mark_key_as_failed(self, key_index: int):
        """Mark a key as temporarily failed"""
        self.failed_keys.add(key_index)
        logger.warning(f"⚠️ API key {key_index + 1} quota exhausted, switching to next key")

    def try_fallback_model(self):
        """Try next available model in priority list"""
        current_idx = self.model_priority.index(self.current_model)
        if current_idx < len(self.model_priority) - 1:
            self.current_model = self.model_priority[current_idx + 1]
            logger.info(f"Switching to fallback model: {self.current_model}")
            return True
        return False

    def generate_content_with_fallback(self, messages: List[Dict], **kwargs):
        """Generate content with automatic key rotation and model fallback"""
        max_retries = len(self.api_keys) * len(self.model_priority)
        
        for attempt in range(max_retries):
            client = self.get_working_client()
            if not client:
                raise Exception("No working API keys available")
            
            try:
                response = client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', 2000),  # Increased for better coverage
                    temperature=kwargs.get('temperature', 0.1),  # LOWER for more consistent factual answers
                    top_p=kwargs.get('top_p', 0.9),
                    stop=kwargs.get('stop', None),  # Add stop sequences if needed
                )
                
                # Success - rotate to next key for load balancing
                logger.info(f"✓ Using API key {self.current_key_index + 1} with model {self.current_model}")
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                
                return response
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    logger.warning(f"API key {self.current_key_index + 1} quota exhausted, trying next...")
                    self.mark_key_as_failed(self.current_key_index)
                    continue
                elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                    # Model not available, try fallback
                    if self.try_fallback_model():
                        continue
                    else:
                        raise Exception(f"No available models work: {error_msg}")
                else:
                    # Non-quota error, propagate it
                    raise e
        
        raise Exception("All API keys and models exhausted or failed")

class LLMProcessor:
    """Enhanced LLM processor optimized for insurance document analysis"""
    
    def __init__(self, model: str = "deepseek-r1-distill-llama-70b", max_tokens: int = 2000):
        self.model = model
        self.max_tokens = max_tokens
        
        # Initialize multi-client manager
        self.client_manager = GroqClientManager()
        
        if not self.client_manager.api_keys:
            self.client_manager = None
            logger.warning("No GROQ_API_KEYS found. Using fallback processing.")
        else:
            logger.info(f"🚀 Initialized with {len(self.client_manager.api_keys)} API keys")

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query parsing for insurance documents"""
        return self._enhanced_fallback_parse_query(query)

    def generate_answer(self, question: str, context: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer with insurance-domain optimized prompting"""
        if not retrieved_chunks:
            return self._no_context_answer(question)

        # Prepare context with more generous limits for insurance documents
        context_text = self._prepare_insurance_context(retrieved_chunks)

        # Try LLM first with insurance-specific prompting
        try:
            if self.client_manager:
                return self._generate_with_groq_insurance(question, context_text, retrieved_chunks)
            else:
                return self._enhanced_fallback_answer(question, retrieved_chunks)
        except Exception as e:
            logger.warning(f"LLM generation failed: {str(e)}")
            return self._enhanced_fallback_answer(question, retrieved_chunks)

    def _prepare_insurance_context(self, chunks: List[Dict]) -> str:
        """Prepare context optimized for insurance documents"""
        context_parts = []
        total_length = 0
        max_context_length = 6000  # INCREASED for better coverage

        for i, chunk in enumerate(chunks[:5]):  # Use top 5 chunks instead of 4
            chunk_text = chunk['text']
            
            # More generous truncation for insurance content
            if len(chunk_text) > 1200:  # Increased from 1000
                # Smart truncation - try to keep complete sentences
                sentences = chunk_text.split('. ')
                truncated = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) > 1150:
                        break
                    truncated.append(sentence)
                    current_length += len(sentence)
                
                chunk_text = '. '.join(truncated) + '.'
            
            if total_length + len(chunk_text) > max_context_length:
                break
                
            # Add relevance score and metadata for better context
            relevance = chunk.get('relevance_score', 0)
            context_parts.append(f"[Document Section {i+1} - Relevance: {relevance:.3f}]\n{chunk_text}")
            total_length += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)

    def _generate_with_groq_insurance(self, question: str, context_text: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate with Groq API using insurance-domain optimized prompting"""
        
        # Analyze question type for better prompting
        question_type = self._classify_insurance_question(question)
        
        # IMPROVED: Better system message with more specific instructions
        system_message = {
            "role": "system",
            "content": """You are an expert insurance document analyst with deep knowledge of policy terms, conditions, and procedures.

CRITICAL INSTRUCTIONS:
1. ALWAYS quote exact text from the document when making factual claims
2. For numerical values (amounts, percentages, periods), be absolutely precise
3. If information is incomplete or ambiguous, clearly state what's missing
4. Distinguish clearly between what IS covered vs what is NOT covered
5. Pay special attention to conditions, waiting periods, exclusions, and limitations
6. Use proper insurance terminology accurately
7. Structure answers logically: Direct answer → Supporting details → Conditions/limitations

ANSWER QUALITY REQUIREMENTS:
- Start with a direct, clear answer to the question
- Provide specific quotes from the document to support your answer
- Include all relevant conditions or limitations that apply
- If multiple scenarios apply, address each one clearly
- If the answer depends on specific circumstances, explain those dependencies
- Never make assumptions - only state what is explicitly mentioned in the document

FORMATTING:
- Use quotes for exact text from documents
- Use bullet points for multiple conditions or requirements
- Be concise but complete - avoid unnecessary elaboration"""
        }
        
        # Create question-specific user message based on question type
        user_message = self._create_insurance_user_message(question, context_text, question_type)
        
        messages = [system_message, user_message]

        try:
            start_time = time.time()
            
            response = self.client_manager.generate_content_with_fallback(
                messages,
                max_tokens=self.max_tokens,
                temperature=0.1,  # VERY low temperature for maximum consistency
                top_p=0.8,  # Slightly more focused
            )
            
            processing_time = time.time() - start_time
            
            # Extract answer from Groq response
            answer_text = response.choices[0].message.content.strip()
            
            # Post-process the answer for insurance context
            cleaned_answer = self._clean_insurance_answer(answer_text)
            
            return {
                "answer": cleaned_answer,
                "reasoning": self._extract_insurance_reasoning(cleaned_answer, question_type),
                "confidence": self._assess_insurance_confidence(retrieved_chunks, cleaned_answer),
                "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(retrieved_chunks[:3])],
                "token_usage": response.usage.total_tokens if hasattr(response, 'usage') else self._estimate_tokens(' '.join([m['content'] for m in messages]) + answer_text),
                "processing_time": processing_time,
                "question_type": question_type
            }

        except Exception as e:
            logger.error(f"Groq generation failed: {str(e)}")
            return self._enhanced_fallback_answer(question, retrieved_chunks)

    def _classify_insurance_question(self, question: str) -> str:
        """Classify the type of insurance question for better prompting"""
        question_lower = question.lower()
        
        # More comprehensive classification
        patterns = {
            'waiting_period': ['waiting period', 'wait', 'cooling period'],
            'coverage': ['coverage', 'cover', 'included', 'benefit', 'reimburse'],
            'exclusion': ['exclude', 'exclusion', 'not covered', 'limitation', 'restrict'],
            'claim_process': ['claim', 'settlement', 'process', 'submit', 'document'],
            'condition': ['condition', 'requirement', 'eligib', 'criteria'],
            'definition': ['define', 'definition', 'mean', 'what is', 'explain'],
            'time_period': ['period', 'duration', 'time', 'day', 'month', 'year'],
            'amount': ['amount', 'limit', 'maximum', 'sum insured', 'rs', '$'],
        }
        
        for category, keywords in patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
                
        return "general"

    def _create_insurance_user_message(self, question: str, context_text: str, question_type: str) -> Dict[str, str]:
        """Create optimized user message based on question type"""
        
        type_specific_instructions = {
            "waiting_period": "Focus on finding exact waiting periods. Look for specific durations and any conditions that modify them.",
            "coverage": "Identify what is specifically covered. Look for positive statements about benefits and their scope.",
            "exclusion": "Identify what is specifically excluded. Look for limitation clauses and exclusion lists.",
            "claim_process": "Focus on claim procedures, required documents, and timelines.",
            "condition": "Identify specific conditions, requirements, or eligibility criteria.",
            "definition": "Look for explicit definitions or explanations of terms.",
            "time_period": "Focus on time-related information including periods and deadlines.",
            "amount": "Focus on specific amounts, limits, and numerical values.",
            "general": "Provide comprehensive information relevant to the question."
        }
        
        specific_instruction = type_specific_instructions.get(question_type, type_specific_instructions["general"])
        
        user_content = f"""QUESTION: {question}

ANALYSIS FOCUS: {specific_instruction}

DOCUMENT CONTEXT:
{context_text}

Based on the document sections above, provide a detailed and accurate answer. Include specific quotes where relevant and mention any important conditions, limitations, or requirements. If the information is not available in the provided context, clearly state this."""

        return {"role": "user", "content": user_content}

    def _clean_insurance_answer(self, answer_text: str) -> str:
        """Clean and format answer for insurance context"""
        # Remove common LLM prefixes that add no value
        prefixes_to_remove = [
            r'^(Answer:|ANSWER:|Response:|Based on the document[s]?:)\s*',
            r'^(According to the document[s]?:)\s*',
            r'^(The document states:)\s*'
        ]
        
        for prefix in prefixes_to_remove:
            answer_text = re.sub(prefix, '', answer_text, flags=re.IGNORECASE)
        
        # Clean up formatting
        answer_text = re.sub(r'\s+', ' ', answer_text).strip()
        
        # Ensure proper punctuation
        if answer_text and not answer_text.endswith(('.', '!', '?')):
            answer_text += '.'
            
        return answer_text

    def _extract_insurance_reasoning(self, answer_text: str, question_type: str) -> str:
        """Extract reasoning specific to insurance context"""
        # Look for quoted text which indicates document-based reasoning
        quotes = re.findall(r'"([^"]*)"', answer_text)
        if quotes:
            return f"Based on specific policy language: {quotes[0][:100]}..."
        
        # Look for specific insurance terms that indicate reasoning
        reasoning_indicators = [
            r'according to[^.]*\.',
            r'as stated[^.]*\.',
            r'the policy[^.]*\.',
            r'subject to[^.]*\.',
            r'provided[^.]*\.',
        ]
        
        for pattern in reasoning_indicators:
            matches = re.finditer(pattern, answer_text, re.IGNORECASE)
            for match in matches:
                return match.group(0).strip()
        
        return f"Analysis based on {question_type} provisions in the policy document."

    def _assess_insurance_confidence(self, chunks: List[Dict], answer: str) -> str:
        """Assess confidence specifically for insurance answers"""
        if not chunks:
            return "low"
            
        avg_score = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
        
        # Insurance-specific confidence indicators
        has_quotes = '"' in answer or "'" in answer
        has_specifics = bool(re.search(r'\d+\s*(day|month|year|%|₹|\$|rs)', answer, re.IGNORECASE))
        has_conditions = bool(re.search(r'(condition|subject to|provided|if|unless|except)', answer, re.IGNORECASE))
        has_policy_reference = bool(re.search(r'(policy|document|clause|section)', answer, re.IGNORECASE))
        
        confidence_score = avg_score
        if has_quotes: confidence_score += 0.15
        if has_specifics: confidence_score += 0.1
        if has_conditions: confidence_score += 0.05
        if has_policy_reference: confidence_score += 0.05
        
        if confidence_score > 0.8:
            return "high"
        elif confidence_score > 0.6:
            return "medium"
        else:
            return "low"

    # Keep the rest of your existing helper methods...
    def _enhanced_fallback_parse_query(self, query: str) -> Dict[str, Any]:
        """Enhanced fallback query parsing with better pattern recognition"""
        import re
        
        entities = {}
        keywords = []
        
        # Enhanced pattern matching for insurance domain
        patterns = {
            'waiting_period': r'waiting period|wait.*period|cooling.*period',
            'grace_period': r'grace period|grace.*day',
            'coverage': r'cover|coverage|covered|benefit|include',
            'surgery': r'surgery|operation|surgical|procedure',
            'maternity': r'maternity|pregnancy|childbirth|delivery',
            'pre_existing': r'pre.?existing|PED|pre.?condition',
            'claim': r'claim|discount|NCD|no.?claim',
            'hospital': r'hospital|medical facility|healthcare',
            'treatment': r'treatment|therapy|AYUSH|alternative',
            'room_rent': r'room rent|ICU|accommodation|hospital.*charge',
            'exclusion': r'exclusion|exclude|not.*cover|limitation',
            'deductible': r'deductible|co.?pay|excess',
            'sum_insured': r'sum insured|coverage.*amount|policy.*limit'
        }
        
        query_lower = query.lower()
        intent = "general_inquiry"
        domain = "insurance"
        
        # Determine intent based on patterns
        matched_patterns = []
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, query_lower):
                matched_patterns.append(pattern_name)
                entities[pattern_name] = True
                keywords.extend(re.findall(r'\b\w+\b', pattern))
        
        # Use the most specific pattern as intent
        if matched_patterns:
            intent = matched_patterns[0]
        
        # Extract numbers and time periods with context
        numbers = re.findall(r'\d+(?:,\d{3})*', query)
        time_units = re.findall(r'(month|year|day)s?', query_lower)
        percentages = re.findall(r'\d+%', query)
        
        if numbers:
            entities['numbers'] = numbers
        if time_units:
            entities['time_units'] = time_units
        if percentages:
            entities['percentages'] = percentages
        
        # Extract key insurance terms
        insurance_terms = re.findall(r'\b(?:policy|insurance|medical|health|premium|benefit|claim|coverage)\b', query_lower)
        keywords.extend(insurance_terms)
        
        # Determine complexity based on entities
        complexity = "simple"
        if len(matched_patterns) > 1 or len(numbers) > 1:
            complexity = "medium"
        if any(term in query_lower for term in ['condition', 'subject to', 'provided', 'except']):
            complexity = "complex"
        
        return {
            "intent": intent,
            "entities": entities,
            "keywords": list(set(keywords + query.lower().split())),
            "domain": domain,
            "complexity": complexity,
            "matched_patterns": matched_patterns
        }

    def _enhanced_fallback_answer(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Enhanced fallback answer generation with insurance-specific extraction"""
        if not chunks:
            return self._no_context_answer(question)

        # Sort chunks by relevance
        sorted_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)

        # Try insurance-specific rule-based extraction
        answer = self._insurance_rule_extraction(question, sorted_chunks)
        
        confidence = "medium" if sorted_chunks[0].get('relevance_score', 0) > 0.7 else "low"

        return {
            "answer": answer,
            "reasoning": f"Answer extracted using insurance-specific pattern matching from {len(chunks)} document sections.",
            "confidence": confidence,
            "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(sorted_chunks[:3])],
            "token_usage": 0,
            "processing_time": 0.1
        }

    def _insurance_rule_extraction(self, question: str, chunks: List[Dict]) -> str:
        """Insurance-specific rule-based answer extraction"""
        question_lower = question.lower()
        
        # Combine text from top chunks with more generous limits
        combined_text = " ".join([chunk['text'] for chunk in chunks[:4]])  # Use top 4 chunks
        
        # Enhanced extraction patterns for insurance
        extraction_rules = {
            'waiting period': [
                r'waiting period[^.]*?(\d+)[^.]*?(month|year|day)s?[^.]*?\.',
                r'(\d+)[^.]*?(month|year|day)s?[^.]*?waiting[^.]*?\.',
                r'after[^.]*?(\d+)[^.]*?(month|year|day)[^.]*?coverage[^.]*?\.'
            ],
            'grace period': [
                r'grace period[^.]*?(\d+)[^.]*?(day|month)s?[^.]*?\.',
                r'(\d+)[^.]*?(day|month)s?[^.]*?grace[^.]*?\.'
            ],
            'coverage': [
                r'(cover|covered|include)[^.]*?\.',
                r'(eligible|entitled|benefit)[^.]*?\.',
                r'(reimburse|reimbursement)[^.]*?\.'
            ],
            'exclusion': [
                r'(exclude|excluded|not covered)[^.]*?\.',
                r'(limitation|limit)[^.]*?\.',
                r'shall not[^.]*?\.'
            ],
            'percentage': [
                r'(\d+)%[^.]*?\.',
                r'(\d+)\s*percent[^.]*?\.'
            ],
            'define': [
                r'defined as[^.]*?\.',
                r'means[^.]*?\.',
                r'refers to[^.]*?\.',
                r'shall mean[^.]*?\.'
            ],
            'sum': [
                r'sum insured[^.]*?\.',
                r'coverage.*amount[^.]*?\.',
                r'maximum.*limit[^.]*?\.'
            ]
        }

        # Find best matching rule
        best_answer = ""
        best_score = 0
        
        for rule_type, patterns in extraction_rules.items():
            if any(keyword in question_lower for keyword in rule_type.split()):
                for pattern in patterns:
                    matches = re.finditer(pattern, combined_text, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        # Find the complete sentence containing this match
                        sentences = re.split(r'[.!?]+', combined_text)
                        for sentence in sentences:
                            if match.group(0).lower() in sentence.lower():
                                candidate = sentence.strip()
                                # Score based on length and keyword matches
                                score = len(candidate) + sum(1 for word in question_lower.split() if word in candidate.lower())
                                if score > best_score and len(candidate) > 20:
                                    best_score = score
                                    best_answer = candidate
                                break

        # If no specific pattern matched, use enhanced keyword-based extraction
        if not best_answer:
            best_answer = self._enhanced_keyword_extraction(question, combined_text)

        # Ensure proper ending
        if best_answer and not best_answer.endswith('.'):
            best_answer += '.'

        return best_answer or "The requested information could not be found in the available document sections. Please verify the information is present in the policy document."

    def _enhanced_keyword_extraction(self, question: str, text: str) -> str:
        """Enhanced keyword-based extraction for insurance documents"""
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        # Remove common stop words
        stop_words = {'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_words = question_words - stop_words
        
        sentences = re.split(r'[.!?]+', text)
        
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 30:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            sentence_words = set(re.findall(r'\b\w+\b', sentence_lower)) - stop_words
            
            # Calculate similarity score
            if question_words:
                common_words = question_words.intersection(sentence_words)
                base_score = len(common_words) / len(question_words)
                
                # Bonus for insurance-specific terms
                insurance_bonus = 0
                insurance_terms = ['policy', 'coverage', 'benefit', 'claim', 'premium', 'insured', 'condition']
                for term in insurance_terms:
                    if term in sentence_lower:
                        insurance_bonus += 0.1
                
                # Bonus for numbers (important in insurance)
                number_bonus = len(re.findall(r'\d+', sentence)) * 0.05
                
                final_score = base_score + insurance_bonus + number_bonus
                scored_sentences.append((final_score, sentence.strip()))

        # Return the sentence with highest score
        if scored_sentences:
            scored_sentences.sort(reverse=True)
            return scored_sentences[0][1]

        return ""

    def _no_context_answer(self, question: str) -> Dict[str, Any]:
        """Answer when no context is available"""
        return {
            "answer": "I could not find relevant information in the document to answer this question. Please ensure the document contains the requested information or try rephrasing your question.",
            "reasoning": "No relevant document sections were retrieved for this question.",
            "confidence": "low",
            "supporting_chunks": [],
            "token_usage": 0,
            "processing_time": 0.01
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text) // 3  # More accurate estimation for Llama models