"""
Data Collection Script for Docs Copilot
This script downloads and prepares data from various sources.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
import logging
from tqdm import tqdm

# Import our configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import DATA_CONFIG, DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Handles data collection from various sources.
    
    This class demonstrates how to:
    1. Download datasets from HuggingFace
    2. Process and clean the data
    3. Save in a structured format
    4. Handle errors and progress tracking
    """
    
    def __init__(self, output_dir: Path = DATA_DIR / "raw"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data collector initialized. Output directory: {self.output_dir}")
    
    def collect_wikipedia_data(self, max_documents: int = None) -> List[Dict[str, Any]]:
        """
        Collect Wikipedia articles.
        
        Args:
            max_documents: Maximum number of documents to collect (for learning purposes)
            
        Returns:
            List of Wikipedia articles
        """
        logger.info("Collecting Wikipedia data...")
        
        # Get the max_documents from config if not specified
        if max_documents is None:
            max_documents = DATA_CONFIG["datasets"]["wikipedia"]["max_documents"]
        
        try:
            # Load Wikipedia dataset from HuggingFace
            # This is a large dataset, so we'll load it in streaming mode
            logger.info("Loading Wikipedia dataset from HuggingFace...")
            
            # For learning purposes, we'll use a smaller subset
            # In production, you might want the full dataset
            dataset = load_dataset(
                "wikipedia", 
                "20220301.en",
                split=f"train[:{max_documents}]",  # Take only first max_documents
                trust_remote_code=True
            )
            
            logger.info(f"Successfully loaded {len(dataset)} Wikipedia articles")
            
            # Convert to list of dictionaries for easier processing
            articles = []
            for i, article in enumerate(tqdm(dataset, desc="Processing Wikipedia articles")):
                # Clean and structure the article data
                article_data = {
                    "id": f"wiki_{i}",
                    "title": article["title"],
                    "text": article["text"],
                    "source": "wikipedia",
                    "length": len(article["text"]),
                    "word_count": len(article["text"].split())
                }
                
                # Filter out very short or very long articles
                if 100 <= article_data["word_count"] <= 5000:
                    articles.append(article_data)
            
            logger.info(f"Collected {len(articles)} Wikipedia articles after filtering")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting Wikipedia data: {e}")
            return []
    
    def collect_arxiv_data(self, max_documents: int = None) -> List[Dict[str, Any]]:
        """
        Collect ArXiv paper abstracts.
        
        Args:
            max_documents: Maximum number of documents to collect
            
        Returns:
            List of ArXiv abstracts
        """
        logger.info("Collecting ArXiv data...")
        
        if max_documents is None:
            max_documents = DATA_CONFIG["datasets"]["arxiv"]["max_documents"]
        
        try:
            # Load ArXiv dataset
            logger.info("Loading ArXiv dataset from HuggingFace...")
            
            dataset = load_dataset(
                "scientific_papers",
                "arxiv",
                split=f"train[:{max_documents}]",
                trust_remote_code=True
            )
            
            logger.info(f"Successfully loaded {len(dataset)} ArXiv papers")
            
            # Process the abstracts
            papers = []
            for i, paper in enumerate(tqdm(dataset, desc="Processing ArXiv papers")):
                paper_data = {
                    "id": f"arxiv_{i}",
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "source": "arxiv",
                    "length": len(paper["abstract"]),
                    "word_count": len(paper["abstract"].split()),
                    "authors": paper.get("authors", []),
                    "categories": paper.get("categories", [])
                }
                
                # Filter out very short abstracts
                if 50 <= paper_data["word_count"] <= 1000:
                    papers.append(paper_data)
            
            logger.info(f"Collected {len(papers)} ArXiv papers after filtering")
            return papers
            
        except Exception as e:
            logger.error(f"Error collecting ArXiv data: {e}")
            return []
    
    def save_data(self, data: List[Dict[str, Any]], filename: str) -> None:
        """
        Save collected data to JSON file.
        
        Args:
            data: List of documents to save
            filename: Name of the output file
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving {len(data)} documents to {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Data saved successfully to {output_path}")
            
            # Print some statistics
            if data:
                word_counts = [doc["word_count"] for doc in data]
                logger.info(f"Statistics:")
                logger.info(f"  - Total documents: {len(data)}")
                logger.info(f"  - Average words per document: {sum(word_counts) / len(word_counts):.1f}")
                logger.info(f"  - Min words: {min(word_counts)}")
                logger.info(f"  - Max words: {max(word_counts)}")
                
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def collect_all_data(self) -> Dict[str, int]:
        """
        Collect data from all sources.
        
        Returns:
            Dictionary with counts of collected documents
        """
        logger.info("Starting data collection from all sources...")
        
        results = {}
        
        # Collect Wikipedia data
        wiki_data = self.collect_wikipedia_data()
        if wiki_data:
            self.save_data(wiki_data, "wikipedia_articles.json")
            results["wikipedia"] = len(wiki_data)
        
        # Collect ArXiv data
        arxiv_data = self.collect_arxiv_data()
        if arxiv_data:
            self.save_data(arxiv_data, "arxiv_abstracts.json")
            results["arxiv"] = len(arxiv_data)
        
        logger.info("Data collection completed!")
        logger.info(f"Summary: {results}")
        
        return results

def main():
    """
    Main function to run data collection.
    This is what you'll call when you want to collect data.
    """
    print("Starting Docs Copilot Data Collection")
    print("=" * 50)
    
    # Initialize data collector
    collector = DataCollector()
    
    # Collect all data
    results = collector.collect_all_data()
    
    print("\nData Collection Summary:")
    for source, count in results.items():
        print(f"  {source}: {count} documents")
    
    print(f"\nData saved to: {collector.output_dir}")
    print("\nData collection completed successfully!")
    print("\nNext steps:")
    print("1. Check the collected data in the output directory")
    print("2. Run preprocessing: python src/data/preprocess_data.py")
    print("3. Continue with the next notebook in the learning path")

if __name__ == "__main__":
    main()
