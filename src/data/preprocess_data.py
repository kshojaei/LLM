"""
Text Preprocessing Module for RAG System
This module handles cleaning, chunking, and preparing text data for embedding.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

# Import our configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import DATA_CONFIG, DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    chunk_by_sentences: bool = True
    preserve_structure: bool = True

class TextPreprocessor:
    """
    Handles text preprocessing for RAG systems.
    
    This class demonstrates:
    1. Text cleaning and normalization
    2. Different chunking strategies
    3. Quality filtering
    4. Metadata preservation
    """
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.output_dir = DATA_DIR / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Text preprocessor initialized. Output directory: {self.output_dir}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Normalize quotes and apostrophes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with more sophisticated methods)
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        sentences = [self.clean_text(s) for s in sentences]
        sentences = [s for s in sentences if len(s.strip()) > 10]  # Remove very short sentences
        
        return sentences
    
    def create_chunks_fixed_size(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Create chunks of fixed size with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Find the last space before the chunk end
                last_space = text.rfind(' ', start, end)
                if last_space > start + chunk_size * 0.7:  # Don't break too early
                    end = last_space
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def create_chunks_semantic(self, text: str, chunk_size: int = None) -> List[str]:
        """
        Create chunks based on semantic boundaries (sentences).
        
        Args:
            text: Text to chunk
            chunk_size: Target size in characters
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_chunks_hierarchical(self, text: str, chunk_size: int = None) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks with different levels of granularity.
        
        Args:
            text: Text to chunk
            chunk_size: Base chunk size
            
        Returns:
            List of hierarchical chunks with metadata
        """
        chunk_size = chunk_size or self.config.chunk_size
        
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        hierarchical_chunks = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            if len(paragraph) <= chunk_size:
                # Paragraph is small enough, use as-is
                hierarchical_chunks.append({
                    'text': paragraph,
                    'type': 'paragraph',
                    'level': 1,
                    'paragraph_id': para_idx,
                    'chunk_id': f"para_{para_idx}",
                    'word_count': len(paragraph.split()),
                    'char_count': len(paragraph)
                })
            else:
                # Paragraph is too long, split into sentences
                sentences = self.split_into_sentences(paragraph)
                current_chunk = ""
                chunk_idx = 0
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                        hierarchical_chunks.append({
                            'text': current_chunk.strip(),
                            'type': 'sentence_group',
                            'level': 2,
                            'paragraph_id': para_idx,
                            'chunk_id': f"para_{para_idx}_chunk_{chunk_idx}",
                            'word_count': len(current_chunk.split()),
                            'char_count': len(current_chunk)
                        })
                        current_chunk = sentence
                        chunk_idx += 1
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                
                # Add the last chunk
                if current_chunk.strip():
                    hierarchical_chunks.append({
                        'text': current_chunk.strip(),
                        'type': 'sentence_group',
                        'level': 2,
                        'paragraph_id': para_idx,
                        'chunk_id': f"para_{para_idx}_chunk_{chunk_idx}",
                        'word_count': len(current_chunk.split()),
                        'char_count': len(current_chunk)
                    })
        
        return hierarchical_chunks
    
    def chunk_document(self, document: Dict[str, Any], strategy: str = "semantic") -> List[Dict[str, Any]]:
        """
        Chunk a single document using the specified strategy.
        
        Args:
            document: Document dictionary with text content
            strategy: Chunking strategy ('fixed', 'semantic', 'hierarchical')
            
        Returns:
            List of chunk dictionaries
        """
        # Get text content
        text_key = 'text' if 'text' in document else 'abstract'
        text = document.get(text_key, '')
        
        if not text:
            logger.warning(f"Document {document.get('id', 'unknown')} has no text content")
            return []
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text) < self.config.min_chunk_size:
            logger.info(f"Document {document.get('id', 'unknown')} too short, skipping")
            return []
        
        # Apply chunking strategy
        if strategy == "fixed":
            chunks = self.create_chunks_fixed_size(cleaned_text)
            chunk_type = "fixed_size"
        elif strategy == "semantic":
            chunks = self.create_chunks_semantic(cleaned_text)
            chunk_type = "semantic"
        elif strategy == "hierarchical":
            hierarchical_chunks = self.create_chunks_hierarchical(cleaned_text)
            return hierarchical_chunks  # Already in the right format
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        # Convert chunks to dictionary format
        chunk_dicts = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                'text': chunk_text,
                'type': chunk_type,
                'chunk_id': f"{document.get('id', 'doc')}_chunk_{i}",
                'source_doc_id': document.get('id'),
                'source_title': document.get('title', ''),
                'source': document.get('source', ''),
                'chunk_index': i,
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text),
                'metadata': {
                    'original_length': len(text),
                    'cleaned_length': len(cleaned_text),
                    'chunking_strategy': strategy
                }
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts
    
    def process_documents(self, documents: List[Dict[str, Any]], strategy: str = "semantic") -> List[Dict[str, Any]]:
        """
        Process a list of documents.
        
        Args:
            documents: List of document dictionaries
            strategy: Chunking strategy to use
            
        Returns:
            List of processed chunks
        """
        logger.info(f"Processing {len(documents)} documents with {strategy} chunking strategy")
        
        all_chunks = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            try:
                chunks = self.chunk_document(doc, strategy)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Print statistics
        if all_chunks:
            word_counts = [chunk['word_count'] for chunk in all_chunks]
            char_counts = [chunk['char_count'] for chunk in all_chunks]
            
            logger.info(f"Chunk statistics:")
            logger.info(f"  Average words per chunk: {np.mean(word_counts):.1f}")
            logger.info(f"  Average chars per chunk: {np.mean(char_counts):.1f}")
            logger.info(f"  Word count range: {min(word_counts)} - {max(word_counts)}")
        
        return all_chunks
    
    def save_processed_data(self, chunks: List[Dict[str, Any]], filename: str) -> None:
        """
        Save processed chunks to file.
        
        Args:
            chunks: List of chunk dictionaries
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving {len(chunks)} chunks to {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed data saved successfully to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def load_and_process(self, input_file: str, output_file: str = None, strategy: str = "semantic") -> List[Dict[str, Any]]:
        """
        Load documents from file, process them, and save the results.
        
        Args:
            input_file: Path to input JSON file
            output_file: Output filename (optional)
            strategy: Chunking strategy
            
        Returns:
            List of processed chunks
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return []
        
        logger.info(f"Loading documents from {input_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Process the documents
            chunks = self.process_documents(documents, strategy)
            
            # Save if output file specified
            if output_file:
                self.save_processed_data(chunks, output_file)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading or processing data: {e}")
            return []

def main():
    """
    Main function to demonstrate text preprocessing.
    """
    print("Text Preprocessing for RAG System")
    print("=" * 40)
    
    # Initialize preprocessor
    config = ChunkingConfig(
        chunk_size=512,
        chunk_overlap=50,
        min_chunk_size=100,
        chunk_by_sentences=True
    )
    
    preprocessor = TextPreprocessor(config)
    
    # Process Wikipedia data
    wiki_file = DATA_DIR / "raw" / "wikipedia_sample.json"
    if wiki_file.exists():
        print(f"\nProcessing Wikipedia data from {wiki_file}")
        wiki_chunks = preprocessor.load_and_process(
            str(wiki_file), 
            "wikipedia_chunks.json", 
            "semantic"
        )
        print(f"Created {len(wiki_chunks)} Wikipedia chunks")
    else:
        print(f"Wikipedia file not found: {wiki_file}")
    
    # Process ArXiv data
    arxiv_file = DATA_DIR / "raw" / "arxiv_sample.json"
    if arxiv_file.exists():
        print(f"\nProcessing ArXiv data from {arxiv_file}")
        arxiv_chunks = preprocessor.load_and_process(
            str(arxiv_file), 
            "arxiv_chunks.json", 
            "semantic"
        )
        print(f"Created {len(arxiv_chunks)} ArXiv chunks")
    else:
        print(f"ArXiv file not found: {arxiv_file}")
    
    print("\nText preprocessing completed!")
    print(f"Processed data saved to: {preprocessor.output_dir}")
    print("\nNext steps:")
    print("1. Review the chunked data")
    print("2. Experiment with different chunking strategies")
    print("3. Move on to embedding generation")

if __name__ == "__main__":
    main()
