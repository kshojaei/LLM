"""
Hallucination Detection Module for RAG System
This module provides techniques for detecting and preventing hallucinations in generated responses.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

@dataclass
class HallucinationConfig:
    """Configuration for hallucination detection."""
    similarity_threshold: float = 0.3
    fact_check_threshold: float = 0.7
    use_semantic_similarity: bool = True
    use_fact_checking: bool = True
    use_consistency_check: bool = True
    max_claims_to_check: int = 5

class HallucinationDetector:
    """
    Advanced hallucination detection system.
    
    This class implements multiple techniques for detecting
    hallucinations in generated responses.
    """
    
    def __init__(self, config: HallucinationConfig = None):
        self.config = config or HallucinationConfig()
        self.stop_words = set(stopwords.words('english'))
        self.detection_history = []
    
    def detect_hallucinations(self, response: str, context: str, query: str = "") -> Dict[str, Any]:
        """
        Detect hallucinations in a generated response.
        
        Args:
            response: Generated response to check
            context: Source context used for generation
            query: Original query (optional)
            
        Returns:
            Dictionary containing hallucination detection results
        """
        detection_results = {
            'response': response,
            'context': context,
            'query': query,
            'is_hallucinated': False,
            'confidence': 0.0,
            'detection_methods': {},
            'flagged_claims': [],
            'recommendations': []
        }
        
        # Method 1: Semantic similarity check
        if self.config.use_semantic_similarity:
            similarity_result = self._check_semantic_similarity(response, context)
            detection_results['detection_methods']['semantic_similarity'] = similarity_result
            
            if similarity_result['is_hallucinated']:
                detection_results['is_hallucinated'] = True
                detection_results['flagged_claims'].extend(similarity_result['flagged_claims'])
        
        # Method 2: Fact checking
        if self.config.use_fact_checking:
            fact_check_result = self._check_facts(response, context)
            detection_results['detection_methods']['fact_checking'] = fact_check_result
            
            if fact_check_result['is_hallucinated']:
                detection_results['is_hallucinated'] = True
                detection_results['flagged_claims'].extend(fact_check_result['flagged_claims'])
        
        # Method 3: Consistency check
        if self.config.use_consistency_check:
            consistency_result = self._check_consistency(response, context)
            detection_results['detection_methods']['consistency'] = consistency_result
            
            if consistency_result['is_hallucinated']:
                detection_results['is_hallucinated'] = True
                detection_results['flagged_claims'].extend(consistency_result['flagged_claims'])
        
        # Calculate overall confidence
        detection_results['confidence'] = self._calculate_overall_confidence(detection_results['detection_methods'])
        
        # Generate recommendations
        detection_results['recommendations'] = self._generate_recommendations(detection_results)
        
        # Store in history
        self.detection_history.append(detection_results)
        
        return detection_results
    
    def _check_semantic_similarity(self, response: str, context: str) -> Dict[str, Any]:
        """Check if response is semantically similar to context."""
        # Extract claims from response
        claims = self._extract_claims(response)
        
        flagged_claims = []
        similarities = []
        
        for claim in claims:
            # Calculate similarity between claim and context
            similarity = self._calculate_claim_similarity(claim, context)
            similarities.append(similarity)
            
            if similarity < self.config.similarity_threshold:
                flagged_claims.append({
                    'claim': claim,
                    'similarity': similarity,
                    'reason': 'Low similarity to context'
                })
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        is_hallucinated = len(flagged_claims) > 0
        
        return {
            'is_hallucinated': is_hallucinated,
            'avg_similarity': avg_similarity,
            'flagged_claims': flagged_claims,
            'confidence': 1.0 - avg_similarity if is_hallucinated else avg_similarity
        }
    
    def _check_facts(self, response: str, context: str) -> Dict[str, Any]:
        """Check if response contains facts not supported by context."""
        # Extract factual claims
        factual_claims = self._extract_factual_claims(response)
        
        flagged_claims = []
        
        for claim in factual_claims[:self.config.max_claims_to_check]:
            # Check if claim is supported by context
            is_supported = self._is_claim_supported(claim, context)
            
            if not is_supported:
                flagged_claims.append({
                    'claim': claim,
                    'reason': 'Not supported by context',
                    'confidence': 0.8
                })
        
        is_hallucinated = len(flagged_claims) > 0
        
        return {
            'is_hallucinated': is_hallucinated,
            'flagged_claims': flagged_claims,
            'confidence': len(flagged_claims) / len(factual_claims) if factual_claims else 0.0
        }
    
    def _check_consistency(self, response: str, context: str) -> Dict[str, Any]:
        """Check for internal consistency and contradictions."""
        # Extract statements from response
        statements = self._extract_statements(response)
        
        flagged_claims = []
        
        # Check for contradictions within response
        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i+1:], i+1):
                if self._are_contradictory(stmt1, stmt2):
                    flagged_claims.append({
                        'claim': f"Contradiction: '{stmt1}' vs '{stmt2}'",
                        'reason': 'Internal contradiction',
                        'confidence': 0.9
                    })
        
        # Check for contradictions with context
        for stmt in statements:
            if self._contradicts_context(stmt, context):
                flagged_claims.append({
                    'claim': stmt,
                    'reason': 'Contradicts context',
                    'confidence': 0.7
                })
        
        is_hallucinated = len(flagged_claims) > 0
        
        return {
            'is_hallucinated': is_hallucinated,
            'flagged_claims': flagged_claims,
            'confidence': len(flagged_claims) / len(statements) if statements else 0.0
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract claims from text."""
        # Simple claim extraction using sentence splitting
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences
        claims = [sent.strip() for sent in sentences if len(sent.strip()) > 20]
        
        return claims
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Look for statements that make factual assertions
        factual_patterns = [
            r'[A-Z][^.]*is [^.]*\.',
            r'[A-Z][^.]*are [^.]*\.',
            r'[A-Z][^.]*was [^.]*\.',
            r'[A-Z][^.]*were [^.]*\.',
            r'[A-Z][^.]*has [^.]*\.',
            r'[A-Z][^.]*have [^.]*\.',
        ]
        
        claims = []
        for pattern in factual_patterns:
            matches = re.findall(pattern, text)
            claims.extend(matches)
        
        return claims
    
    def _extract_statements(self, text: str) -> List[str]:
        """Extract individual statements from text."""
        # Split by sentences and filter
        sentences = sent_tokenize(text)
        statements = [sent.strip() for sent in sentences if len(sent.strip()) > 10]
        
        return statements
    
    def _calculate_claim_similarity(self, claim: str, context: str) -> float:
        """Calculate similarity between a claim and context."""
        # Use TF-IDF for similarity calculation
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([claim, context])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def _is_claim_supported(self, claim: str, context: str) -> bool:
        """Check if a claim is supported by context."""
        # Simple keyword-based support check
        claim_words = set(word.lower() for word in word_tokenize(claim) if word.isalpha())
        context_words = set(word.lower() for word in word_tokenize(context) if word.isalpha())
        
        # Remove stop words
        claim_words = claim_words - self.stop_words
        context_words = context_words - self.stop_words
        
        # Check overlap
        overlap = len(claim_words.intersection(context_words))
        total_claim_words = len(claim_words)
        
        if total_claim_words == 0:
            return True
        
        support_ratio = overlap / total_claim_words
        return support_ratio >= self.config.fact_check_threshold
    
    def _are_contradictory(self, stmt1: str, stmt2: str) -> bool:
        """Check if two statements are contradictory."""
        # Simple contradiction detection based on negation patterns
        negation_words = ['not', 'no', 'never', 'none', 'nothing', 'nowhere', 'nobody']
        
        stmt1_lower = stmt1.lower()
        stmt2_lower = stmt2.lower()
        
        # Check for direct negation
        for neg_word in negation_words:
            if neg_word in stmt1_lower and neg_word not in stmt2_lower:
                # Check if they're talking about the same thing
                if self._are_about_same_topic(stmt1, stmt2):
                    return True
        
        return False
    
    def _contradicts_context(self, statement: str, context: str) -> bool:
        """Check if statement contradicts context."""
        # Simple contradiction check with context
        statement_words = set(word.lower() for word in word_tokenize(statement) if word.isalpha())
        context_words = set(word.lower() for word in word_tokenize(context) if word.isalpha())
        
        # Check for conflicting information
        # This is a simplified version - in practice, you'd use more sophisticated NLP
        return False
    
    def _are_about_same_topic(self, stmt1: str, stmt2: str) -> bool:
        """Check if two statements are about the same topic."""
        # Simple topic similarity based on shared nouns and key terms
        stmt1_words = set(word.lower() for word in word_tokenize(stmt1) if word.isalpha())
        stmt2_words = set(word.lower() for word in word_tokenize(stmt2) if word.isalpha())
        
        # Remove stop words
        stmt1_words = stmt1_words - self.stop_words
        stmt2_words = stmt2_words - self.stop_words
        
        # Calculate overlap
        overlap = len(stmt1_words.intersection(stmt2_words))
        total_unique = len(stmt1_words.union(stmt2_words))
        
        if total_unique == 0:
            return False
        
        similarity = overlap / total_unique
        return similarity > 0.3
    
    def _calculate_overall_confidence(self, detection_methods: Dict[str, Any]) -> float:
        """Calculate overall confidence in hallucination detection."""
        if not detection_methods:
            return 0.0
        
        confidences = []
        for method, result in detection_methods.items():
            if 'confidence' in result:
                confidences.append(result['confidence'])
        
        return np.mean(confidences) if confidences else 0.0
    
    def _generate_recommendations(self, detection_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detection results."""
        recommendations = []
        
        if detection_results['is_hallucinated']:
            recommendations.append("Response may contain hallucinations. Consider:")
            
            if detection_results['detection_methods'].get('semantic_similarity', {}).get('is_hallucinated'):
                recommendations.append("- Verify claims against source context")
            
            if detection_results['detection_methods'].get('fact_checking', {}).get('is_hallucinated'):
                recommendations.append("- Fact-check unsupported claims")
            
            if detection_results['detection_methods'].get('consistency', {}).get('is_hallucinated'):
                recommendations.append("- Review for internal contradictions")
            
            recommendations.append("- Consider regenerating with more specific context")
        else:
            recommendations.append("Response appears to be well-grounded in context")
        
        return recommendations
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get statistics about hallucination detection."""
        if not self.detection_history:
            return {}
        
        total_detections = len(self.detection_history)
        hallucinated_count = sum(1 for result in self.detection_history if result['is_hallucinated'])
        
        avg_confidence = np.mean([result['confidence'] for result in self.detection_history])
        
        # Method effectiveness
        method_stats = {}
        for method in ['semantic_similarity', 'fact_checking', 'consistency']:
            method_results = [result['detection_methods'].get(method, {}) for result in self.detection_history]
            method_detections = sum(1 for result in method_results if result.get('is_hallucinated', False))
            method_stats[method] = {
                'detections': method_detections,
                'effectiveness': method_detections / total_detections if total_detections > 0 else 0
            }
        
        return {
            'total_detections': total_detections,
            'hallucination_rate': hallucinated_count / total_detections if total_detections > 0 else 0,
            'avg_confidence': avg_confidence,
            'method_statistics': method_stats
        }

class HallucinationPrevention:
    """
    Techniques for preventing hallucinations during generation.
    """
    
    def __init__(self):
        self.prevention_strategies = [
            self._add_uncertainty_markers,
            self._limit_response_scope,
            self._add_source_attribution
        ]
    
    def prevent_hallucinations(self, prompt: str, context: str) -> str:
        """Modify prompt to prevent hallucinations."""
        enhanced_prompt = prompt
        
        for strategy in self.prevention_strategies:
            enhanced_prompt = strategy(enhanced_prompt, context)
        
        return enhanced_prompt
    
    def _add_uncertainty_markers(self, prompt: str, context: str) -> str:
        """Add instructions to express uncertainty when appropriate."""
        uncertainty_instruction = """
        Important: If you are uncertain about any information or if it's not clearly stated in the context, 
        please indicate this uncertainty in your response. Use phrases like "based on the context," 
        "it appears that," or "the information suggests that" when appropriate.
        """
        
        return f"{prompt}\n\n{uncertainty_instruction}"
    
    def _limit_response_scope(self, prompt: str, context: str) -> str:
        """Limit response to information available in context."""
        scope_instruction = """
        Please base your response only on the information provided in the context. 
        Do not add information that is not explicitly stated or clearly implied in the context.
        """
        
        return f"{prompt}\n\n{scope_instruction}"
    
    def _add_source_attribution(self, prompt: str, context: str) -> str:
        """Add instructions for source attribution."""
        attribution_instruction = """
        When making specific claims, try to reference the source information when possible.
        """
        
        return f"{prompt}\n\n{attribution_instruction}"

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test hallucination detector
    detector = HallucinationDetector()
    
    # Test cases
    test_cases = [
        {
            'response': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
            'context': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.',
            'query': 'What is machine learning?'
        },
        {
            'response': 'Machine learning is a type of cooking technique used in French cuisine.',
            'context': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.',
            'query': 'What is machine learning?'
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Response: {test_case['response']}")
        print(f"Context: {test_case['context']}")
        
        result = detector.detect_hallucinations(
            test_case['response'],
            test_case['context'],
            test_case['query']
        )
        
        print(f"Is hallucinated: {result['is_hallucinated']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Flagged claims: {result['flagged_claims']}")
        print(f"Recommendations: {result['recommendations']}")
    
    # Test prevention
    prevention = HallucinationPrevention()
    
    original_prompt = "Answer this question based on the context:"
    context = "Machine learning is a subset of AI."
    
    enhanced_prompt = prevention.prevent_hallucinations(original_prompt, context)
    print(f"\nEnhanced prompt:\n{enhanced_prompt}")
    
    # Get statistics
    stats = detector.get_detection_statistics()
    print(f"\nDetection statistics: {stats}")
