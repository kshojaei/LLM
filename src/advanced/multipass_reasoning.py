"""
Multipass Reasoning Module for RAG System
This module implements advanced reasoning techniques including multi-step reasoning, chain-of-thought, and iterative refinement.

Author: Kamran Shojaei - Physicist with background in AI/ML
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import time
from enum import Enum

logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    """Types of reasoning strategies."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    MULTI_STEP = "multi_step"
    SELF_CORRECTION = "self_correction"

@dataclass
class ReasoningConfig:
    """Configuration for multipass reasoning."""
    max_passes: int = 3
    confidence_threshold: float = 0.8
    use_self_correction: bool = True
    use_iterative_refinement: bool = True
    reasoning_type: ReasoningType = ReasoningType.CHAIN_OF_THOUGHT

class MultipassReasoner:
    """
    Advanced multipass reasoning system for RAG.
    
    This class implements various reasoning strategies to improve
    the quality and accuracy of generated responses.
    """
    
    def __init__(self, config: ReasoningConfig = None):
        self.config = config or ReasoningConfig()
        self.reasoning_history = []
    
    def reason(self, query: str, context: str, llm_generator) -> Dict[str, Any]:
        """
        Perform multipass reasoning on a query.
        
        Args:
            query: The question to answer
            context: Retrieved context documents
            llm_generator: LLM generator for response generation
            
        Returns:
            Dictionary containing the final answer and reasoning process
        """
        if self.config.reasoning_type == ReasoningType.CHAIN_OF_THOUGHT:
            return self._chain_of_thought_reasoning(query, context, llm_generator)
        elif self.config.reasoning_type == ReasoningType.ITERATIVE_REFINEMENT:
            return self._iterative_refinement_reasoning(query, context, llm_generator)
        elif self.config.reasoning_type == ReasoningType.MULTI_STEP:
            return self._multi_step_reasoning(query, context, llm_generator)
        elif self.config.reasoning_type == ReasoningType.SELF_CORRECTION:
            return self._self_correction_reasoning(query, context, llm_generator)
        else:
            raise ValueError(f"Unknown reasoning type: {self.config.reasoning_type}")
    
    def _chain_of_thought_reasoning(self, query: str, context: str, llm_generator) -> Dict[str, Any]:
        """Implement chain-of-thought reasoning."""
        reasoning_steps = []
        
        # Step 1: Analyze the question
        analysis_prompt = f"""
        Analyze this question and identify what information is needed to answer it:
        
        Question: {query}
        Context: {context[:1000]}...
        
        Provide a step-by-step analysis of what needs to be understood to answer this question.
        """
        
        analysis_response = llm_generator.llm_model.generate(analysis_prompt)
        reasoning_steps.append({
            'step': 'question_analysis',
            'prompt': analysis_prompt,
            'response': analysis_response
        })
        
        # Step 2: Extract relevant information
        extraction_prompt = f"""
        Based on the analysis, extract the most relevant information from the context:
        
        Question: {query}
        Analysis: {analysis_response}
        Context: {context}
        
        Extract only the information that directly relates to answering the question.
        """
        
        extraction_response = llm_generator.llm_model.generate(extraction_prompt)
        reasoning_steps.append({
            'step': 'information_extraction',
            'prompt': extraction_prompt,
            'response': extraction_response
        })
        
        # Step 3: Synthesize the answer
        synthesis_prompt = f"""
        Using the extracted information, provide a comprehensive answer:
        
        Question: {query}
        Extracted Information: {extraction_response}
        
        Provide a clear, well-structured answer that directly addresses the question.
        """
        
        final_answer = llm_generator.llm_model.generate(synthesis_prompt)
        reasoning_steps.append({
            'step': 'answer_synthesis',
            'prompt': synthesis_prompt,
            'response': final_answer
        })
        
        return {
            'answer': final_answer,
            'reasoning_steps': reasoning_steps,
            'reasoning_type': 'chain_of_thought',
            'confidence': self._calculate_confidence(reasoning_steps)
        }
    
    def _iterative_refinement_reasoning(self, query: str, context: str, llm_generator) -> Dict[str, Any]:
        """Implement iterative refinement reasoning."""
        current_answer = ""
        refinement_history = []
        
        for pass_num in range(self.config.max_passes):
            if pass_num == 0:
                # Initial answer
                prompt = f"""
                Answer this question based on the provided context:
                
                Question: {query}
                Context: {context}
                
                Provide a comprehensive answer.
                """
            else:
                # Refinement based on previous answer
                prompt = f"""
                Refine and improve this answer:
                
                Question: {query}
                Context: {context}
                Previous Answer: {current_answer}
                
                Identify any gaps, inaccuracies, or areas for improvement in the previous answer.
                Provide a refined and improved version.
                """
            
            response = llm_generator.llm_model.generate(prompt)
            
            refinement_history.append({
                'pass': pass_num + 1,
                'prompt': prompt,
                'response': response,
                'improvement': self._assess_improvement(current_answer, response) if pass_num > 0 else 0
            })
            
            current_answer = response
            
            # Check if we should stop early
            if pass_num > 0:
                improvement = refinement_history[-1]['improvement']
                if improvement < 0.1:  # Minimal improvement
                    break
        
        return {
            'answer': current_answer,
            'refinement_history': refinement_history,
            'reasoning_type': 'iterative_refinement',
            'confidence': self._calculate_confidence(refinement_history)
        }
    
    def _multi_step_reasoning(self, query: str, context: str, llm_generator) -> Dict[str, Any]:
        """Implement multi-step reasoning."""
        steps = []
        
        # Break down complex questions into sub-questions
        sub_questions = self._decompose_question(query)
        
        sub_answers = []
        for i, sub_q in enumerate(sub_questions):
            sub_prompt = f"""
            Answer this sub-question based on the context:
            
            Sub-question: {sub_q}
            Context: {context}
            
            Provide a focused answer to this specific sub-question.
            """
            
            sub_answer = llm_generator.llm_model.generate(sub_prompt)
            sub_answers.append(sub_answer)
            
            steps.append({
                'step': f'sub_question_{i+1}',
                'question': sub_q,
                'answer': sub_answer
            })
        
        # Synthesize final answer from sub-answers
        synthesis_prompt = f"""
        Combine these sub-answers to provide a comprehensive answer to the main question:
        
        Main Question: {query}
        Sub-answers:
        {chr(10).join([f"{i+1}. {answer}" for i, answer in enumerate(sub_answers)])}
        
        Provide a well-structured final answer that integrates all the sub-answers.
        """
        
        final_answer = llm_generator.llm_model.generate(synthesis_prompt)
        
        return {
            'answer': final_answer,
            'sub_questions': sub_questions,
            'sub_answers': sub_answers,
            'reasoning_steps': steps,
            'reasoning_type': 'multi_step',
            'confidence': self._calculate_confidence(steps)
        }
    
    def _self_correction_reasoning(self, query: str, context: str, llm_generator) -> Dict[str, Any]:
        """Implement self-correction reasoning."""
        # Generate initial answer
        initial_prompt = f"""
        Answer this question based on the provided context:
        
        Question: {query}
        Context: {context}
        
        Provide a comprehensive answer.
        """
        
        initial_answer = llm_generator.llm_model.generate(initial_prompt)
        
        # Self-evaluate the answer
        evaluation_prompt = f"""
        Evaluate the quality and accuracy of this answer:
        
        Question: {query}
        Context: {context}
        Answer: {initial_answer}
        
        Identify any potential issues, inaccuracies, or areas for improvement.
        Rate the answer on a scale of 1-10 and explain your reasoning.
        """
        
        evaluation = llm_generator.llm_model.generate(evaluation_prompt)
        
        # Generate corrected answer if needed
        correction_prompt = f"""
        Based on the evaluation, provide a corrected and improved answer:
        
        Question: {query}
        Context: {context}
        Original Answer: {initial_answer}
        Evaluation: {evaluation}
        
        Provide a corrected version that addresses any identified issues.
        """
        
        corrected_answer = llm_generator.llm_model.generate(correction_prompt)
        
        return {
            'answer': corrected_answer,
            'initial_answer': initial_answer,
            'evaluation': evaluation,
            'reasoning_type': 'self_correction',
            'confidence': self._calculate_confidence([{'response': corrected_answer}])
        }
    
    def _decompose_question(self, query: str) -> List[str]:
        """Decompose complex questions into sub-questions."""
        # Simple decomposition logic (can be improved with more sophisticated methods)
        decomposition_prompt = f"""
        Break down this complex question into 2-3 simpler sub-questions:
        
        Question: {query}
        
        Provide each sub-question on a new line, numbered 1, 2, 3, etc.
        """
        
        # For now, return a simple decomposition
        # In a real implementation, you would use an LLM to generate this
        if "what is" in query.lower():
            return [f"What is the definition of {query.split('what is')[1].strip('?')}?"]
        elif "how does" in query.lower():
            return [f"How does {query.split('how does')[1].strip('?')} work?"]
        else:
            return [query]
    
    def _assess_improvement(self, old_answer: str, new_answer: str) -> float:
        """Assess improvement between two answers."""
        # Simple improvement assessment based on length and content overlap
        if not old_answer or not new_answer:
            return 0.0
        
        # Calculate content similarity
        old_words = set(old_answer.lower().split())
        new_words = set(new_answer.lower().split())
        
        if not old_words or not new_words:
            return 0.0
        
        intersection = len(old_words.intersection(new_words))
        union = len(old_words.union(new_words))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Length difference as a factor
        length_ratio = len(new_answer) / len(old_answer) if len(old_answer) > 0 else 1.0
        
        # Combine similarity and length factors
        improvement = (1 - similarity) * 0.7 + abs(length_ratio - 1) * 0.3
        
        return min(improvement, 1.0)
    
    def _calculate_confidence(self, reasoning_steps: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on reasoning steps."""
        if not reasoning_steps:
            return 0.0
        
        # Simple confidence calculation based on number of steps and response quality
        num_steps = len(reasoning_steps)
        avg_length = sum(len(step.get('response', '')) for step in reasoning_steps) / num_steps
        
        # Normalize confidence based on steps and response quality
        step_confidence = min(num_steps / 3, 1.0)  # More steps = higher confidence
        length_confidence = min(avg_length / 200, 1.0)  # Longer responses = higher confidence
        
        return (step_confidence + length_confidence) / 2

class ReasoningAnalyzer:
    """
    Analyze reasoning patterns and effectiveness.
    """
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_reasoning(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality and effectiveness of reasoning."""
        analysis = {
            'reasoning_type': reasoning_result.get('reasoning_type', 'unknown'),
            'confidence': reasoning_result.get('confidence', 0.0),
            'num_steps': len(reasoning_result.get('reasoning_steps', [])),
            'answer_length': len(reasoning_result.get('answer', '')),
            'has_self_correction': 'evaluation' in reasoning_result,
            'has_iteration': 'refinement_history' in reasoning_result,
            'quality_score': self._calculate_quality_score(reasoning_result)
        }
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _calculate_quality_score(self, reasoning_result: Dict[str, Any]) -> float:
        """Calculate overall quality score for reasoning."""
        score = 0.0
        
        # Base score from confidence
        score += reasoning_result.get('confidence', 0.0) * 0.4
        
        # Length score (moderate length is better)
        answer_length = len(reasoning_result.get('answer', ''))
        if 100 <= answer_length <= 500:
            score += 0.3
        elif answer_length > 500:
            score += 0.2
        
        # Step count score
        num_steps = len(reasoning_result.get('reasoning_steps', []))
        if num_steps >= 2:
            score += 0.2
        
        # Self-correction bonus
        if 'evaluation' in reasoning_result:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning performance."""
        if not self.analysis_history:
            return {}
        
        confidences = [analysis['confidence'] for analysis in self.analysis_history]
        quality_scores = [analysis['quality_score'] for analysis in self.analysis_history]
        
        return {
            'total_reasoning_sessions': len(self.analysis_history),
            'avg_confidence': sum(confidences) / len(confidences),
            'avg_quality_score': sum(quality_scores) / len(quality_scores),
            'reasoning_types': list(set(analysis['reasoning_type'] for analysis in self.analysis_history))
        }

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test multipass reasoner
    config = ReasoningConfig(
        max_passes=2,
        reasoning_type=ReasoningType.CHAIN_OF_THOUGHT
    )
    
    reasoner = MultipassReasoner(config)
    analyzer = ReasoningAnalyzer()
    
    # Mock LLM generator for testing
    class MockLLMGenerator:
        def __init__(self):
            self.llm_model = MockLLMModel()
    
    class MockLLMModel:
        def generate(self, prompt):
            return f"Mock response to: {prompt[:50]}..."
    
    mock_llm = MockLLMGenerator()
    
    # Test reasoning
    query = "What is machine learning and how does it work?"
    context = "Machine learning is a subset of AI that focuses on algorithms..."
    
    result = reasoner.reason(query, context, mock_llm)
    print(f"Reasoning result: {result}")
    
    # Analyze reasoning
    analysis = analyzer.analyze_reasoning(result)
    print(f"Analysis: {analysis}")
    
    # Get statistics
    stats = analyzer.get_reasoning_statistics()
    print(f"Statistics: {stats}")
