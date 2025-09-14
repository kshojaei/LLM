"""
LLM Models Module for RAG System
This module provides various language models for text generation.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Generator
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MODELS, DATA_DIR

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

class LLMModel:
    """
    Base class for language models.
    
    This class provides a unified interface for different LLMs
    and demonstrates how to work with various model architectures.
    """
    
    def __init__(self, model_name: str, device: str = "auto", load_in_8bit: bool = False):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.tokenizer = None
        self.generation_config = GenerationConfig()
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the language model."""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def generate(self, prompt: str, config: GenerationConfig = None, **kwargs) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError("Subclasses must implement generate")
    
    def generate_stream(self, prompt: str, config: GenerationConfig = None, **kwargs) -> Generator[str, None, None]:
        """Generate text stream from a prompt."""
        raise NotImplementedError("Subclasses must implement generate_stream")

class LlamaModel(LLMModel):
    """
    Llama model implementation.
    
    Supports Llama-3-8B-Instruct and other Llama variants.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-8B-Instruct", device: str = "auto", load_in_8bit: bool = False):
        super().__init__(model_name, device, load_in_8bit)
        self.load_model()
    
    def load_model(self):
        """Load the Llama model."""
        try:
            logger.info(f"Loading Llama model: {self.model_name}")
            
            # Configure quantization if needed
            quantization_config = None
            if self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            logger.info("Llama model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise
    
    def generate(self, prompt: str, config: GenerationConfig = None, **kwargs) -> str:
        """Generate text from a prompt."""
        if config is None:
            config = self.generation_config
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    do_sample=config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate text with Llama: {e}")
            raise
    
    def generate_stream(self, prompt: str, config: GenerationConfig = None, **kwargs) -> Generator[str, None, None]:
        """Generate text stream from a prompt."""
        if config is None:
            config = self.generation_config
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with streaming
            with torch.no_grad():
                for output in self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    do_sample=config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                ):
                    # Decode new tokens
                    new_text = self.tokenizer.decode(
                        output[inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    yield new_text
                    
        except Exception as e:
            logger.error(f"Failed to generate stream with Llama: {e}")
            raise

class MistralModel(LLMModel):
    """
    Mistral model implementation.
    
    Supports Mistral-7B-Instruct and other Mistral variants.
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = "auto", load_in_8bit: bool = False):
        super().__init__(model_name, device, load_in_8bit)
        self.load_model()
    
    def load_model(self):
        """Load the Mistral model."""
        try:
            logger.info(f"Loading Mistral model: {self.model_name}")
            
            # Configure quantization if needed
            quantization_config = None
            if self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            logger.info("Mistral model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Mistral model: {e}")
            raise
    
    def generate(self, prompt: str, config: GenerationConfig = None, **kwargs) -> str:
        """Generate text from a prompt."""
        if config is None:
            config = self.generation_config
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    do_sample=config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate text with Mistral: {e}")
            raise
    
    def generate_stream(self, prompt: str, config: GenerationConfig = None, **kwargs) -> Generator[str, None, None]:
        """Generate text stream from a prompt."""
        if config is None:
            config = self.generation_config
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with streaming
            with torch.no_grad():
                for output in self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    do_sample=config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                ):
                    # Decode new tokens
                    new_text = self.tokenizer.decode(
                        output[inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    yield new_text
                    
        except Exception as e:
            logger.error(f"Failed to generate stream with Mistral: {e}")
            raise

class LLMModelFactory:
    """
    Factory class for creating language models.
    
    This demonstrates the factory pattern for creating different types
    of language models based on configuration.
    """
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> LLMModel:
        """Create a language model based on type."""
        model_configs = {
            "llama_3_8b": LlamaModel,
            "mistral_7b": MistralModel,
        }
        
        if model_type not in model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_configs[model_type]
        return model_class(**kwargs)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types."""
        return ["llama_3_8b", "mistral_7b"]

class PromptTemplate:
    """
    Template system for RAG prompts.
    
    This class helps create consistent and effective prompts for
    different RAG tasks and model types.
    """
    
    def __init__(self, template: str, model_type: str = "llama"):
        self.template = template
        self.model_type = model_type
    
    def format(self, query: str, context: str = "", **kwargs) -> str:
        """Format the template with provided variables."""
        return self.template.format(
            query=query,
            context=context,
            **kwargs
        )
    
    @classmethod
    def get_rag_template(cls, model_type: str = "llama") -> 'PromptTemplate':
        """Get a standard RAG template for the model type."""
        if model_type.lower() in ["llama", "llama_3_8b"]:
            template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions based on the provided context. Use the context to provide accurate and informative answers. If the context doesn't contain enough information to answer the question, say so.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Context: {context}

Question: {query}

Please provide a comprehensive answer based on the context above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        elif model_type.lower() in ["mistral", "mistral_7b"]:
            template = """<s>[INST] You are a helpful assistant that answers questions based on the provided context. Use the context to provide accurate and informative answers. If the context doesn't contain enough information to answer the question, say so.

Context: {context}

Question: {query} [/INST]"""
        else:
            # Generic template
            template = """Based on the following context, please answer the question.

Context: {context}

Question: {query}

Answer:"""
        
        return cls(template, model_type)

class RAGGenerator:
    """
    RAG-specific text generator.
    
    This class combines retrieval and generation for RAG systems,
    handling context formatting and response generation.
    """
    
    def __init__(self, llm_model: LLMModel, prompt_template: PromptTemplate = None):
        self.llm_model = llm_model
        self.prompt_template = prompt_template or PromptTemplate.get_rag_template()
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]], max_context_length: int = 4000) -> Dict[str, Any]:
        """Generate a response using retrieved documents as context."""
        # Format context from retrieved documents
        context = self._format_context(retrieved_docs, max_context_length)
        
        # Create prompt
        prompt = self.prompt_template.format(query=query, context=context)
        
        # Generate response
        start_time = time.time()
        response = self.llm_model.generate(prompt)
        generation_time = time.time() - start_time
        
        return {
            "response": response,
            "context": context,
            "generation_time": generation_time,
            "num_retrieved_docs": len(retrieved_docs),
            "context_length": len(context)
        }
    
    def _format_context(self, retrieved_docs: List[Dict[str, Any]], max_length: int) -> str:
        """Format retrieved documents into context."""
        context_parts = []
        current_length = 0
        
        for doc in retrieved_docs:
            doc_text = f"Title: {doc.get('title', 'Unknown')}\nContent: {doc.get('content', '')}\n\n"
            
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "".join(context_parts)

def load_llm_model(model_type: str = "llama_3_8b", **kwargs) -> LLMModel:
    """Convenience function to load a language model."""
    return LLMModelFactory.create_model(model_type, **kwargs)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test Llama model (if available)
    try:
        print("Testing Llama model...")
        llama_model = LlamaModel(load_in_8bit=True)  # Use 8-bit for memory efficiency
        
        sample_prompt = "What is machine learning?"
        response = llama_model.generate(sample_prompt)
        print(f"Llama response: {response}")
        
    except Exception as e:
        print(f"Llama model test failed: {e}")
    
    # Test Mistral model (if available)
    try:
        print("\nTesting Mistral model...")
        mistral_model = MistralModel(load_in_8bit=True)
        
        sample_prompt = "Explain quantum computing in simple terms."
        response = mistral_model.generate(sample_prompt)
        print(f"Mistral response: {response}")
        
    except Exception as e:
        print(f"Mistral model test failed: {e}")
    
    # Test RAG generator
    print("\nTesting RAG generator...")
    try:
        # Use any available model
        model = load_llm_model("llama_3_8b", load_in_8bit=True)
        rag_generator = RAGGenerator(model)
        
        # Mock retrieved documents
        mock_docs = [
            {"title": "Machine Learning Basics", "content": "Machine learning is a subset of artificial intelligence..."},
            {"title": "Deep Learning", "content": "Deep learning uses neural networks with multiple layers..."}
        ]
        
        response = rag_generator.generate_response("What is machine learning?", mock_docs)
        print(f"RAG response: {response['response']}")
        print(f"Generation time: {response['generation_time']:.2f}s")
        
    except Exception as e:
        print(f"RAG generator test failed: {e}")
