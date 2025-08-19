"""
Patent Similarity Evaluation Module
Contains functions for evaluating similarity between patent documents using various methods
"""

import re
import torch
import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification

class PatentSimilarityEvaluator:
    """Evaluator for patent document similarity using multiple models"""
    
    def __init__(self):
        """Initialize the evaluator with required models"""
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.rerank_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
        self.rerank_model.eval()
    
    def evaluate_similarity(self, text_1: str, text_2: str) -> Dict[str, float]:
        """
        Evaluate similarity between two texts using multiple methods.
        
        Args:
            text_1: First text to compare
            text_2: Second text to compare
            
        Returns:
            Dictionary with similarity scores from different methods
        """
        return {
            "similarities_score": self._similarity_score(text_1, text_2),
            "rerank_score": self._rerank_score(text_1, text_2),
            "llm_score": self._llm_rerank_score(text_1, text_2)
        }
    
    def _similarity_score(self, text_1: str, text_2: str) -> float:
        """Calculate cosine similarity using sentence transformers"""
        text_1_encode = np.array(self.sentence_model.encode(text_1)).reshape(1, -1)
        text_2_encode = np.array(self.sentence_model.encode(text_2)).reshape(1, -1)
        similarities = cosine_similarity(text_1_encode, text_2_encode)
        return float(similarities[0][0])
    
    def _rerank_score(self, text_1: str, text_2: str) -> float:
        """Calculate rerank score using BGE reranker"""
        pairs = [text_1, text_2]
        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            )
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
        return float(scores[0])
    
    def _llm_rerank_score(self, text_1: str, text_2: str, instruction: str = None) -> float:
        """Calculate LLM-based rerank score using Qwen3 reranker"""
        # Initialize tokenizer and model
        tokenizer_llm = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
        model_llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
        token_false_id = tokenizer_llm.convert_tokens_to_ids("no")
        token_true_id = tokenizer_llm.convert_tokens_to_ids("yes")
        max_length = 8192
        
        prefix = "<|im_start|>system\nJudge whether Passage1 and Passage2 are related to each other based on the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer_llm.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer_llm.encode(suffix, add_special_tokens=False)
        
        # Format instruction
        if instruction is None:
            instruction = 'Judge whether the two passages are related to each other or not'
        formatted_instruction = "<Instruct>: {instruction}\n<Passage 1>: {query}\n<Passage 2>: {doc}".format(
            instruction=instruction, query=text_1, doc=text_2
        )
        
        # Process inputs
        pairs = [formatted_instruction]
        inputs = tokenizer_llm(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        
        inputs = tokenizer_llm.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model_llm.device)
        
        # Compute logits
        with torch.no_grad():
            batch_scores = model_llm(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        
        return scores[0]

class PatentPromptGenerator:
    """Generator for patent analysis prompts"""
    
    @staticmethod
    def generate_analysis_prompt(abstract: str, description: str, claims: str) -> str:
        """Generate prompt for patent analysis"""
        return f"""
Your task is to read the Abstract, Description, and Claims fields from a patent and extract the following two pieces of information:

1. **User Scenario: Describe the actual scenario in which a user would use the system or product described in the patent. Write a short paragraph using natural, easy-to-understand language that specifically describes who the user is, what they are doing, and under what circumstances. Avoid repeating information from the User Problem.

2. **User Problem: Identify the problem or difficulty that users have with previous systems or products that this invention is intended to solve. Write a short, clear paragraph that focuses on the actual problem, not the benefits of the new invention.

**Input**:
- Abstract: {abstract}
- Description: {description}
- Claims: {claims}

**Instruction**
- Analyze all three fields: Abstract, Description, and Claims.
- Simplify technical and legal language into language that is understandable to non-technical people.
- Look for User Problems primarily in the Background of the Invention or in the Description that mention limitations of the previous system (e.g., "disadvantage", "issue", "problem").
- Look for User Scenarios in the Abstract, Summary of the Invention, or Detailed Description, which describe the use or application of the invention.
- If the information is unclear, infer the context or problem based on the purpose of the invention.
- If the Description is too long, prioritize the paragraphs related to the problem or application.
- Your output must be a single JSON with two fields: user_scenario and user_problem. Do not include any additional text, analysis, or other explanation.

**Example**
**Input**:
- Abstract: "A method for improving telephone communication by using undulatory currents to transmit voice signals, enabling clearer audio."
- Description: "This invention addresses issues with existing telegraph systems that fail to transmit clear voice signals, especially in noisy environments. It uses undulatory currents to vibrate a receiver, ensuring accurate voice transmission."
- Claims: "1. A system for transmitting voice using undulatory currents. 2. A method for reducing signal interference in telephone systems."

**Output**:
```
{{
"User Scenario": "A user using a telephone to make a call in a noisy environment, such as a train station or street, needs to hear the other person's voice clearly.",
"User Problem": "With previous telegraph systems, users had difficulty with distorted or unclear voice sounds, especially when there was noise from the surrounding environment."
}}
```

Output need to follow instruction bellow:
```
{{
"User Scenario": "...",
"User Problem": "..."
}}
```
"""

class PatentTextParser:
    """Parser for extracting information from patent text"""
    
    @staticmethod
    def parse_idea_input(sample_text: str) -> Dict[str, str]:
        """Parse idea input text to extract structured information"""
        title_match = re.search(r'\*\*Idea title\*\*:\s*(.+)', sample_text)
        title = title_match.group(1).strip() if title_match else ""

        scenario_match = re.search(r'\*\*User scenario\*\*:\s*(.*?)(?=\*\*User problem\*\*)', sample_text, re.DOTALL)
        scenario = scenario_match.group(1).strip() if scenario_match else ""

        problem_match = re.search(r'\*\*User problem\*\*:\s*(.*)', sample_text, re.DOTALL)
        problem = problem_match.group(1).strip() if problem_match else ""

        return {
            "idea_title": title,
            "user_scenario": scenario,
            "user_problem": problem
        }
    
    @staticmethod
    def parse_idea_text(sample_text: str) -> Dict[str, str]:
        """Parse JSON-like text to extract user scenario and problem"""
        scenario_match = re.search(r'"User Scenario":\s*"([^"]*)"', sample_text)
        scenario = scenario_match.group(1).strip() if scenario_match else ""

        problem_match = re.search(r'"User Problem":\s*"([^"]*)"', sample_text)
        problem = problem_match.group(1).strip() if problem_match else ""

        return {
            "user_scenario": scenario,
            "user_problem": problem
        }
    
    @staticmethod
    def extract_user_info(data: Dict[str, Any]) -> Dict[str, str]:
        """Extract user information from parsed data with flexible key matching"""
        if "User Scenario" in data:
            user_scenario = data["User Scenario"]
        elif "user_scenario" in data:
            user_scenario = data["user_scenario"]
        else:
            user_scenario = "Not found"

        if "User Problem" in data:
            user_problem = data["User Problem"]
        elif "user_problem" in data:
            user_problem = data["user_problem"]
        else:
            user_problem = "Not found"

        return {
            "user_scenario": user_scenario,
            "user_problem": user_problem
        }

# Legacy function wrappers for backward compatibility
def eval_url(text_1: str, text_2: str) -> Dict[str, float]:
    """Legacy wrapper for similarity evaluation"""
    evaluator = PatentSimilarityEvaluator()
    return evaluator.evaluate_similarity(text_1, text_2)

def prompt(abstract: str, description: str, claims: str) -> str:
    """Legacy wrapper for prompt generation"""
    return PatentPromptGenerator.generate_analysis_prompt(abstract, description, claims)

def parse_idea_input(sample_text: str) -> Dict[str, str]:
    """Legacy wrapper for idea input parsing"""
    return PatentTextParser.parse_idea_input(sample_text)

def parse_idea_text(sample_text: str) -> Dict[str, str]:
    """Legacy wrapper for idea text parsing"""
    return PatentTextParser.parse_idea_text(sample_text)

def extract_user_info(data: Dict[str, Any]) -> Dict[str, str]:
    """Legacy wrapper for user info extraction"""
    return PatentTextParser.extract_user_info(data)
