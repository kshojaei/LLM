"""
Query Rewriting Module for RAG System
This module provides advanced query processing capabilities including rewriting, expansion, and reformulation.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

@dataclass
class QueryRewriteConfig:
    """Configuration for query rewriting."""
    use_synonyms: bool = True
    use_expansion: bool = True
    use_reformulation: bool = True
    max_rewrites: int = 3
    similarity_threshold: float = 0.7
    use_ner: bool = True

class QueryRewriter:
    """
    Advanced query rewriting system.
    
    This class provides multiple strategies for improving query quality
    including synonym expansion, query reformulation, and entity recognition.
    """
    
    def __init__(self, config: QueryRewriteConfig = None):
        self.config = config or QueryRewriteConfig()
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = None
        
        # Initialize spaCy if NER is enabled
        if self.config.use_ner:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. NER will be disabled.")
                self.nlp = None
                self.config.use_ner = False
    
    def rewrite_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Rewrite a query using multiple strategies.
        
        Args:
            query: Original query
            
        Returns:
            List of rewritten queries with metadata
        """
        rewrites = []
        
        # Original query
        rewrites.append({
            'query': query,
            'method': 'original',
            'confidence': 1.0,
            'metadata': {}
        })
        
        # Synonym expansion
        if self.config.use_synonyms:
            synonym_rewrites = self._expand_synonyms(query)
            rewrites.extend(synonym_rewrites)
        
        # Query expansion
        if self.config.use_expansion:
            expansion_rewrites = self._expand_query(query)
            rewrites.extend(expansion_rewrites)
        
        # Query reformulation
        if self.config.use_reformulation:
            reformulation_rewrites = self._reformulate_query(query)
            rewrites.extend(reformulation_rewrites)
        
        # Remove duplicates and limit results
        unique_rewrites = self._deduplicate_rewrites(rewrites)
        return unique_rewrites[:self.config.max_rewrites]
    
    def _expand_synonyms(self, query: str) -> List[Dict[str, Any]]:
        """Expand query with synonyms."""
        rewrites = []
        
        # Tokenize query
        words = query.lower().split()
        
        # Find synonyms for each word
        for i, word in enumerate(words):
            synonyms = self._get_synonyms(word)
            
            for synonym in synonyms[:2]:  # Limit to 2 synonyms per word
                # Create new query with synonym
                new_words = words.copy()
                new_words[i] = synonym
                new_query = ' '.join(new_words)
                
                rewrites.append({
                    'query': new_query,
                    'method': 'synonym_expansion',
                    'confidence': 0.8,
                    'metadata': {
                        'original_word': word,
                        'synonym': synonym,
                        'position': i
                    }
                })
        
        return rewrites
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        
        # Get synsets
        synsets = wordnet.synsets(word)
        
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym.split()) == 1:
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def _expand_query(self, query: str) -> List[Dict[str, Any]]:
        """Expand query with related terms."""
        rewrites = []
        
        # Extract key terms
        key_terms = self._extract_key_terms(query)
        
        # Create expanded versions
        for term in key_terms:
            # Add related terms
            related_terms = self._get_related_terms(term)
            
            for related_term in related_terms[:2]:
                expanded_query = f"{query} {related_term}"
                
                rewrites.append({
                    'query': expanded_query,
                    'method': 'query_expansion',
                    'confidence': 0.7,
                    'metadata': {
                        'key_term': term,
                        'added_term': related_term
                    }
                })
        
        return rewrites
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query."""
        # Simple key term extraction (can be improved with more sophisticated methods)
        words = query.lower().split()
        
        # Filter out stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms
    
    def _get_related_terms(self, term: str) -> List[str]:
        """Get related terms for a given term."""
        related_terms = []
        
        # Get synonyms
        synonyms = self._get_synonyms(term)
        related_terms.extend(synonyms)
        
        # Get hypernyms (broader terms)
        synsets = wordnet.synsets(term)
        for synset in synsets:
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    related_term = lemma.name().replace('_', ' ')
                    if related_term != term:
                        related_terms.append(related_term)
        
        # Get hyponyms (narrower terms)
        for synset in synsets:
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    related_term = lemma.name().replace('_', ' ')
                    if related_term != term:
                        related_terms.append(related_term)
        
        return list(set(related_terms))
    
    def _reformulate_query(self, query: str) -> List[Dict[str, Any]]:
        """Reformulate query using different phrasings."""
        rewrites = []
        
        # Question reformulation patterns
        patterns = [
            (r'what is (.+)', r'define \1'),
            (r'what are (.+)', r'explain \1'),
            (r'how does (.+) work', r'explain how \1 works'),
            (r'how to (.+)', r'steps to \1'),
            (r'why (.+)', r'reasons for \1'),
            (r'when (.+)', r'timing of \1'),
            (r'where (.+)', r'location of \1'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                reformulated = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                if reformulated != query:
                    rewrites.append({
                        'query': reformulated,
                        'method': 'query_reformulation',
                        'confidence': 0.9,
                        'metadata': {
                            'pattern': pattern,
                            'replacement': replacement
                        }
                    })
        
        # Add question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        for qw in question_words:
            if not query.lower().startswith(qw):
                reformulated = f"{qw} {query.lower()}"
                rewrites.append({
                    'query': reformulated,
                    'method': 'question_reformulation',
                    'confidence': 0.6,
                    'metadata': {
                        'added_question_word': qw
                    }
                })
        
        return rewrites
    
    def _deduplicate_rewrites(self, rewrites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate rewrites."""
        seen_queries = set()
        unique_rewrites = []
        
        for rewrite in rewrites:
            query = rewrite['query'].lower().strip()
            if query not in seen_queries:
                seen_queries.add(query)
                unique_rewrites.append(rewrite)
        
        return unique_rewrites
    
    def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract named entities from query."""
        if not self.nlp:
            return []
        
        doc = self.nlp(query)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and characteristics."""
        analysis = {
            'length': len(query.split()),
            'has_question_word': any(query.lower().startswith(qw) for qw in ['what', 'how', 'why', 'when', 'where', 'who']),
            'is_question': query.strip().endswith('?'),
            'entities': self.extract_entities(query),
            'key_terms': self._extract_key_terms(query),
            'complexity': self._assess_complexity(query)
        }
        
        return analysis
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity."""
        words = query.split()
        
        if len(words) <= 3:
            return 'simple'
        elif len(words) <= 6:
            return 'medium'
        else:
            return 'complex'

class QueryExpansion:
    """
    Advanced query expansion using various techniques.
    """
    
    def __init__(self):
        self.expansion_methods = {
            'synonym': self._synonym_expansion,
            'concept': self._concept_expansion,
            'context': self._context_expansion
        }
    
    def expand_query(self, query: str, method: str = 'synonym') -> List[str]:
        """Expand query using specified method."""
        if method in self.expansion_methods:
            return self.expansion_methods[method](query)
        else:
            return [query]
    
    def _synonym_expansion(self, query: str) -> List[str]:
        """Expand using synonyms."""
        # Implementation for synonym expansion
        return [query]
    
    def _concept_expansion(self, query: str) -> List[str]:
        """Expand using conceptual relationships."""
        # Implementation for concept expansion
        return [query]
    
    def _context_expansion(self, query: str) -> List[str]:
        """Expand using contextual information."""
        # Implementation for context expansion
        return [query]

class QueryOptimizer:
    """
    Query optimization for better retrieval performance.
    """
    
    def __init__(self):
        self.optimization_rules = [
            self._remove_stop_words,
            self._normalize_terms,
            self._add_boosting_terms,
            self._handle_negation
        ]
    
    def optimize_query(self, query: str) -> str:
        """Apply optimization rules to query."""
        optimized = query
        
        for rule in self.optimization_rules:
            optimized = rule(optimized)
        
        return optimized
    
    def _remove_stop_words(self, query: str) -> str:
        """Remove common stop words."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    def _normalize_terms(self, query: str) -> str:
        """Normalize terms in query."""
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _add_boosting_terms(self, query: str) -> str:
        """Add boosting terms for important concepts."""
        # Implementation for adding boosting terms
        return query
    
    def _handle_negation(self, query: str) -> str:
        """Handle negation in queries."""
        # Implementation for handling negation
        return query

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test query rewriter
    rewriter = QueryRewriter()
    
    test_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks",
        "Tell me about artificial intelligence"
    ]
    
    for query in test_queries:
        print(f"\nOriginal query: {query}")
        
        # Analyze query
        analysis = rewriter.analyze_query_intent(query)
        print(f"Analysis: {analysis}")
        
        # Rewrite query
        rewrites = rewriter.rewrite_query(query)
        print(f"Rewrites ({len(rewrites)}):")
        for i, rewrite in enumerate(rewrites):
            print(f"  {i+1}. {rewrite['query']} ({rewrite['method']}, conf: {rewrite['confidence']:.2f})")
        
        print("-" * 50)
