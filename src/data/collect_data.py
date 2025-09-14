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
import logging
from tqdm import tqdm
import requests
import time
import xml.etree.ElementTree as ET

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
    
    def collect_wikipedia_data(self, max_documents: int = None, use_real_data: bool = True) -> List[Dict[str, Any]]:
        """
        Collect Wikipedia articles using Wikipedia API.
        
        Args:
            max_documents: Maximum number of documents to collect
            use_real_data: If True, fetch real data from Wikipedia API; if False, use sample data
            
        Returns:
            List of Wikipedia articles
        """
        logger.info("Collecting Wikipedia data...")
        
        # Get the max_documents from config if not specified
        if max_documents is None:
            max_documents = DATA_CONFIG["datasets"]["wikipedia"]["max_documents"]
        
        if not use_real_data:
            return self._get_sample_wikipedia_data(max_documents)
        
        try:
            # Use Wikipedia API to get real data
            logger.info("Fetching real Wikipedia data using Wikipedia API...")
            
            # Popular AI/ML related topics
            topics = [
                "Machine Learning", "Artificial Intelligence", "Deep Learning", 
                "Natural Language Processing", "Computer Vision", "Neural Network",
                "Reinforcement Learning", "Data Science", "Big Data", "Algorithm",
                "Supervised Learning", "Unsupervised Learning", "Feature Engineering",
                "Model Selection", "Cross Validation", "Overfitting", "Bias Variance Tradeoff"
            ]
            
            articles = []
            base_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
            
            for i, topic in enumerate(topics[:max_documents]):
                try:
                    # Get article summary with proper headers
                    headers = {
                        'User-Agent': 'RAG-Learning-System/1.0 (Educational Purpose)'
                    }
                    response = requests.get(f"{base_url}{topic.replace(' ', '_')}", 
                                          headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    if 'extract' in data and data['extract']:
                        # Get full article content
                        full_content = self._get_wikipedia_full_content(topic)
                        
                        article_data = {
                            "id": f"wiki_{i}",
                            "title": data.get('title', topic),
                            "text": full_content if full_content else data.get('extract', ''),
                            "source": "wikipedia",
                            "length": len(full_content) if full_content else len(data.get('extract', '')),
                            "word_count": len(full_content.split()) if full_content else len(data.get('extract', '').split()),
                            "url": data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            "description": data.get('extract', '')[:200] + "..." if len(data.get('extract', '')) > 200 else data.get('extract', '')
                        }
                        articles.append(article_data)
                        logger.info(f"Collected: {article_data['title']}")
                    
                    # Be respectful to Wikipedia API
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch {topic}: {e}")
                    continue
            
            logger.info(f"Successfully collected {len(articles)} Wikipedia articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting Wikipedia data: {e}")
            logger.info("Falling back to sample data...")
            return self._get_sample_wikipedia_data(max_documents)
    
    def _get_wikipedia_full_content(self, title: str) -> str:
        """Get full Wikipedia article content."""
        try:
            # Use Wikipedia API to get full content
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': False,
                'explaintext': True
            }
            
            headers = {
                'User-Agent': 'RAG-Learning-System/1.0 (Educational Purpose)'
            }
            response = requests.get("https://en.wikipedia.org/w/api.php", 
                                  params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            for page in pages.values():
                if 'extract' in page:
                    return page['extract']
            
            return ""
            
        except Exception as e:
            logger.warning(f"Failed to get full content for {title}: {e}")
            return ""
    
    def _get_sample_wikipedia_data(self, max_documents: int) -> List[Dict[str, Any]]:
        """Get sample Wikipedia data for fallback."""
        logger.info("Using sample Wikipedia data...")
        
        sample_articles = [
            {
                "title": "Machine Learning",
                "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It involves training models on large datasets to make predictions or decisions without being explicitly programmed for every task. Common types include supervised learning, unsupervised learning, and reinforcement learning. Machine learning is used in many applications including image recognition, natural language processing, recommendation systems, and autonomous vehicles."
            },
            {
                "title": "Artificial Intelligence",
                "text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI research has been highly successful in developing effective techniques for solving a wide range of problems, from game playing to medical diagnosis."
            },
            {
                "title": "Deep Learning",
                "text": "Deep learning is a subset of machine learning based on artificial neural networks with representation learning. It uses multiple layers of neural networks to progressively extract higher-level features from raw input. Deep learning has been particularly successful in computer vision, speech recognition, and natural language processing. Popular deep learning architectures include convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformer models."
            },
            {
                "title": "Natural Language Processing",
                "text": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. NLP combines computational linguistics with statistical, machine learning, and deep learning models to help computers understand, interpret, and manipulate human language. Applications include machine translation, sentiment analysis, chatbots, and text summarization."
            },
            {
                "title": "Computer Vision",
                "text": "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects and react to what they see. Computer vision is used in autonomous vehicles, medical image analysis, facial recognition systems, and augmented reality applications."
            }
        ]
        
        articles = []
        for i, article in enumerate(sample_articles[:max_documents]):
            article_data = {
                "id": f"wiki_{i}",
                "title": article["title"],
                "text": article["text"],
                "source": "wikipedia",
                "length": len(article["text"]),
                "word_count": len(article["text"].split())
            }
            articles.append(article_data)
        
        return articles
    
    def collect_arxiv_data(self, max_documents: int = None, use_real_data: bool = True) -> List[Dict[str, Any]]:
        """
        Collect ArXiv paper abstracts using ArXiv API.
        
        Args:
            max_documents: Maximum number of documents to collect
            use_real_data: If True, fetch real data from ArXiv API; if False, use sample data
            
        Returns:
            List of ArXiv abstracts
        """
        logger.info("Collecting ArXiv data...")
        
        if max_documents is None:
            max_documents = DATA_CONFIG["datasets"]["arxiv"]["max_documents"]
        
        if not use_real_data:
            return self._get_sample_arxiv_data(max_documents)
        
        try:
            # Use ArXiv API to get real data
            logger.info("Fetching real ArXiv data using ArXiv API...")
            
            # Search for recent AI/ML papers
            search_queries = [
                "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV",
                "machine learning",
                "deep learning", 
                "natural language processing",
                "computer vision",
                "neural networks"
            ]
            
            papers = []
            base_url = "http://export.arxiv.org/api/query"
            
            for query in search_queries[:2]:  # Limit queries to avoid too many requests
                try:
                    params = {
                        'search_query': query,
                        'start': 0,
                        'max_results': max_documents // 2,  # Split between queries
                        'sortBy': 'submittedDate',
                        'sortOrder': 'descending'
                    }
                    
                    headers = {
                        'User-Agent': 'RAG-Learning-System/1.0 (Educational Purpose)'
                    }
                    response = requests.get(base_url, params=params, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    # Parse XML response
                    root = ET.fromstring(response.content)
                    
                    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                        if len(papers) >= max_documents:
                            break
                            
                        paper_data = self._parse_arxiv_entry(entry, len(papers))
                        if paper_data:
                            papers.append(paper_data)
                            logger.info(f"Collected: {paper_data['title']}")
                    
                    # Be respectful to ArXiv API
                    time.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch papers for query '{query}': {e}")
                    continue
            
            logger.info(f"Successfully collected {len(papers)} ArXiv papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error collecting ArXiv data: {e}")
            logger.info("Falling back to sample data...")
            return self._get_sample_arxiv_data(max_documents)
    
    def _parse_arxiv_entry(self, entry, index: int) -> Dict[str, Any]:
        """Parse an ArXiv XML entry into our format."""
        try:
            # Extract title
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            title = title_elem.text.strip() if title_elem is not None else f"Paper {index}"
            
            # Extract abstract
            abstract_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            abstract = abstract_elem.text.strip() if abstract_elem is not None else ""
            
            if not abstract or len(abstract) < 50:  # Skip papers with very short abstracts
                return None
            
            # Extract authors
            authors = []
            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract categories
            categories = []
            for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Extract publication date
            published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
            published = published_elem.text if published_elem is not None else ""
            
            # Extract ArXiv ID
            id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
            arxiv_id = ""
            if id_elem is not None:
                arxiv_id = id_elem.text.split('/')[-1] if '/' in id_elem.text else id_elem.text
            
            return {
                "id": f"arxiv_{index}",
                "title": title,
                "abstract": abstract,
                "source": "arxiv",
                "length": len(abstract),
                "word_count": len(abstract.split()),
                "authors": authors if authors else ["Unknown Author"],
                "categories": categories if categories else ["cs.AI"],
                "published": published,
                "arxiv_id": arxiv_id
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse ArXiv entry: {e}")
            return None
    
    def _get_sample_arxiv_data(self, max_documents: int) -> List[Dict[str, Any]]:
        """Get sample ArXiv data for fallback."""
        logger.info("Using sample ArXiv data...")
        
        sample_papers = [
            {
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications."
            },
            {
                "title": "Generative Adversarial Networks",
                "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game."
            },
            {
                "title": "ResNet: Deep Residual Learning for Image Recognition",
                "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth."
            }
        ]
        
        papers = []
        for i, paper in enumerate(sample_papers[:max_documents]):
            paper_data = {
                "id": f"arxiv_{i}",
                "title": paper["title"],
                "abstract": paper["abstract"],
                "source": "arxiv",
                "length": len(paper["abstract"]),
                "word_count": len(paper["abstract"].split()),
                "authors": ["Sample Author"],
                "categories": ["cs.AI", "cs.LG"]
            }
            papers.append(paper_data)
        
        return papers
    
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
