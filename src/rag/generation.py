"""
Answer generation modules for RAG system.

Implements multiple generator backends including mT5, OpenAI API, and Ollama.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import numpy as np

from .logging_config import LoggerMixin


@dataclass
class GenerationResult:
    """Answer generation result."""
    answer: str
    confidence: float
    generation_time: float
    metadata: Dict[str, Any]


class AnswerGenerator(ABC, LoggerMixin):
    """Abstract base class for answer generators."""
    
    @abstractmethod
    def generate_answer(self,
                       question: str,
                       contexts: List[str],
                       max_length: int = 100,
                       **kwargs) -> GenerationResult:
        """
        Generate answer from question and contexts.
        
        Args:
            question: Question text
            contexts: List of context passages
            max_length: Maximum answer length
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult object
        """
        pass
    
    def _get_device(self, device: str) -> str:
        """
        Determine optimal device for inference.
        
        Args:
            device: Device preference ('auto', 'cpu', 'cuda', 'mps')
            
        Returns:
            Device string
        """
        if device == 'auto':
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            return 'cpu'
        return device


class MT5Generator(AnswerGenerator):
    """Pre-trained mT5 generator without fine-tuning."""
    
    def __init__(self,
                 model_name: str = 'google/mt5-base',
                 device: str = 'auto',
                 num_contexts: int = 3):
        """
        Initialize mT5 generator.
        
        Args:
            model_name: mT5 model name (mt5-base, mt5-large)
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            num_contexts: Number of contexts to use in prompt
        """
        super().__init__()
        
        try:
            from transformers import MT5ForConditionalGeneration, MT5Tokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.num_contexts = num_contexts
        
        # Determine device
        self.device = self._get_device(device)
        self.logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.logger.info(f"Loading mT5 model: {model_name}")
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        
        self.logger.info(f"mT5 model loaded successfully on {self.device}")
    
    def _construct_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Construct prompt for mT5.
        
        Args:
            question: Question text
            contexts: List of context passages
            
        Returns:
            Formatted prompt
        """
        # Use top N contexts to fit in context window
        selected_contexts = contexts[:self.num_contexts]
        context_text = " ".join(selected_contexts)
        
        # mT5 format: "question: {question} context: {context}"
        prompt = f"question: {question} context: {context_text}"
        return prompt
    
    def _calculate_confidence(self, scores: Optional[tuple]) -> float:
        """
        Calculate confidence from generation scores.
        
        Args:
            scores: Generation scores from model
            
        Returns:
            Confidence score (0-1)
        """
        if not scores:
            return 0.5
        
        # Average probability of generated tokens
        probs = []
        for score in scores:
            # Get max probability for each token
            prob = torch.softmax(score, dim=-1).max().item()
            probs.append(prob)
        
        return sum(probs) / len(probs) if probs else 0.5
    
    def generate_answer(self,
                       question: str,
                       contexts: List[str],
                       max_length: int = 100,
                       num_beams: int = 4,
                       temperature: float = 1.0,
                       **kwargs) -> GenerationResult:
        """
        Generate answer using mT5.
        
        Args:
            question: Question text
            contexts: List of context passages
            max_length: Maximum answer length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult object
        """
        start_time = time.time()
        
        # Construct prompt
        prompt = self._construct_prompt(question, contexts)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode answer
        answer = self.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(outputs.scores)
        
        generation_time = time.time() - start_time
        
        return GenerationResult(
            answer=answer,
            confidence=confidence,
            generation_time=generation_time,
            metadata={
                'model': self.model_name,
                'num_beams': num_beams,
                'temperature': temperature,
                'num_contexts_used': min(len(contexts), self.num_contexts),
                'device': self.device
            }
        )


class OpenAIGenerator(AnswerGenerator):
    """OpenAI API generator (GPT-3.5/GPT-4)."""
    
    def __init__(self,
                 api_key: str,
                 model: str = 'gpt-4',
                 num_contexts: int = 3,
                 max_retries: int = 3,
                 timeout: int = 30):
        """
        Initialize OpenAI generator.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-3.5-turbo, gpt-4, etc.)
            num_contexts: Number of contexts to use
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
        """
        super().__init__()
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required. "
                "Install with: pip install openai"
            )
        
        self.model = model
        self.num_contexts = num_contexts
        self.max_retries = max_retries
        self.timeout = timeout
        
        if not api_key:
            raise ValueError("OpenAI API key is required but was None!")
        
        self.logger.info(f"🚀 Initializing OpenAI client with key: {api_key[:20]}...")
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        
        self.logger.info(f"✅ OpenAI generator initialized with model: {model}")
        self.logger.info(f"💡 Expected confidence: 85-95%")
    
    def _construct_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Construct prompt for OpenAI.
        
        Args:
            question: Question text
            contexts: List of context passages
            
        Returns:
            Formatted prompt
        """
        selected_contexts = contexts[:self.num_contexts]
        context_text = "\n\n".join([
            f"Context {i+1}: {ctx}"
            for i, ctx in enumerate(selected_contexts)
        ])
        
        prompt = f"""Based on the following contexts, answer the question.

{context_text}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate_answer(self,
                       question: str,
                       contexts: List[str],
                       max_length: int = 100,
                       temperature: float = 0.7,
                       **kwargs) -> GenerationResult:
        """
        Generate answer using OpenAI API.
        
        Args:
            question: Question text
            contexts: List of context passages
            max_length: Maximum answer length (tokens)
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult object
        """
        start_time = time.time()
        
        # Construct prompt
        prompt = self._construct_prompt(question, contexts)
        
        # Call API with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based on provided contexts."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=max_length,
                    temperature=temperature
                )
                
                answer = response.choices[0].message.content
                
                # OpenAI doesn't provide confidence, use finish_reason as proxy
                confidence = 1.0 if response.choices[0].finish_reason == 'stop' else 0.7
                
                generation_time = time.time() - start_time
                
                return GenerationResult(
                    answer=answer,
                    confidence=confidence,
                    generation_time=generation_time,
                    metadata={
                        'model': self.model,
                        'tokens_used': response.usage.total_tokens,
                        'finish_reason': response.choices[0].finish_reason,
                        'num_contexts_used': min(len(contexts), self.num_contexts)
                    }
                )
            
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        raise RuntimeError("Failed to generate answer after max retries")


class OllamaGenerator(AnswerGenerator):
    """Local LLM generator using Ollama."""
    
    def __init__(self,
                 model: str = 'llama3',
                 base_url: str = 'http://localhost:11434',
                 num_contexts: int = 3):
        """
        Initialize Ollama generator.
        
        Args:
            model: Ollama model name (llama3, mistral, etc.)
            base_url: Ollama server URL
            num_contexts: Number of contexts to use
        """
        super().__init__()
        
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama is required. "
                "Install with: pip install ollama"
            )
        
        self.model = model
        self.base_url = base_url
        self.num_contexts = num_contexts
        self.ollama = ollama
        
        self.logger.info(f"Ollama generator initialized with model: {model}")
    
    def _construct_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Construct prompt for Ollama.
        
        Args:
            question: Question text
            contexts: List of context passages
            
        Returns:
            Formatted prompt
        """
        selected_contexts = contexts[:self.num_contexts]
        context_text = "\n\n".join(selected_contexts)
        
        prompt = f"""Answer the question based on the context.

Context: {context_text}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate_answer(self,
                       question: str,
                       contexts: List[str],
                       max_length: int = 100,
                       temperature: float = 0.7,
                       **kwargs) -> GenerationResult:
        """
        Generate answer using Ollama.
        
        Args:
            question: Question text
            contexts: List of context passages
            max_length: Maximum answer length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult object
        """
        start_time = time.time()
        
        # Construct prompt
        prompt = self._construct_prompt(question, contexts)
        
        # Generate
        response = self.ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': temperature,
                'num_predict': max_length
            }
        )
        
        answer = response['response']
        
        # Ollama doesn't provide confidence
        confidence = 0.8  # Default confidence
        
        generation_time = time.time() - start_time
        
        return GenerationResult(
            answer=answer,
            confidence=confidence,
            generation_time=generation_time,
            metadata={
                'model': self.model,
                'total_duration': response.get('total_duration', 0),
                'num_contexts_used': min(len(contexts), self.num_contexts)
            }
        )


class MBERTGenerator(AnswerGenerator):
    """Pre-trained mBERT generator for extractive QA."""
    
    def __init__(self,
                 model_name: str = 'bert-base-multilingual-cased',
                 device: str = 'auto',
                 num_contexts: int = 1):
        """
        Initialize mBERT generator.
        
        Args:
            model_name: mBERT model name
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            num_contexts: Number of contexts to use (mBERT works best with 1)
        """
        super().__init__()
        
        try:
            from transformers import BertForQuestionAnswering, BertTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.num_contexts = num_contexts
        
        # Determine device
        self.device = self._get_device(device)
        self.logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.logger.info(f"Loading mBERT model: {model_name}")
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        self.logger.info(f"mBERT model loaded successfully on {self.device}")
    
    def generate_answer(self,
                       question: str,
                       contexts: List[str],
                       max_length: int = 100,
                       **kwargs) -> GenerationResult:
        """
        Generate answer using mBERT (extractive).
        
        Args:
            question: Question text
            contexts: List of context passages
            max_length: Maximum answer length (not used for extractive)
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult object
        """
        start_time = time.time()
        
        # Use the first context (mBERT is extractive, works best with single context)
        context = contexts[0] if contexts else ""
        
        # Tokenize
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get start and end positions
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # Calculate confidence from logits
        start_prob = torch.softmax(start_scores, dim=-1)[0][start_idx].item()
        end_prob = torch.softmax(end_scores, dim=-1)[0][end_idx].item()
        confidence = (start_prob + end_prob) / 2
        
        # Extract answer
        input_ids = inputs['input_ids'][0]
        answer_tokens = input_ids[start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Handle empty or invalid answers
        if not answer or start_idx > end_idx:
            answer = ""
            confidence = 0.0
        
        generation_time = time.time() - start_time
        
        return GenerationResult(
            answer=answer,
            confidence=confidence,
            generation_time=generation_time,
            metadata={
                'model': self.model_name,
                'start_position': start_idx.item(),
                'end_position': end_idx.item(),
                'num_contexts_used': 1,
                'device': self.device,
                'type': 'extractive'
            }
        )


class GeneratorFactory:
    """Factory for creating generator instances."""
    
    _generators = {
        'mbert': MBERTGenerator,
        'mt5': MT5Generator,
        'openai': OpenAIGenerator,
        'ollama': OllamaGenerator,
    }
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> AnswerGenerator:
        """
        Create generator instance based on configuration.
        
        Args:
            config: Generator configuration with 'type' field
            
        Returns:
            AnswerGenerator instance
            
        Raises:
            ValueError: If generator type is not supported
        """
        gen_type = config.get('type', 'mt5').lower()
        
        if gen_type not in cls._generators:
            raise ValueError(
                f"Unsupported generator type: {gen_type}. "
                f"Supported types: {list(cls._generators.keys())}"
            )
        
        generator_class = cls._generators[gen_type]
        
        # Extract relevant config for each generator type
        if gen_type == 'mbert':
            return generator_class(
                model_name=config.get('model_name', 'bert-base-multilingual-cased'),
                device=config.get('device', 'auto'),
                num_contexts=config.get('num_contexts_to_use', 1)
            )
        
        elif gen_type == 'mt5':
            return generator_class(
                model_name=config.get('model_name', 'google/mt5-base'),
                device=config.get('device', 'auto'),
                num_contexts=config.get('num_contexts_to_use', 3)
            )
        
        elif gen_type == 'openai':
            api_key = config.get('api_key')
            if not api_key:
                # Try to get from environment
                import os
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv('OPENAI_API_KEY')
                
                if not api_key:
                    raise ValueError(
                        "OpenAI API key not found! "
                        "Set OPENAI_API_KEY in .env file or pass via config"
                    )
            
            print(f"✅ Creating OpenAI generator with key: {api_key[:20]}...")
            return generator_class(
                api_key=api_key,
                model=config.get('model_name', 'gpt-3.5-turbo'),
                num_contexts=config.get('num_contexts_to_use', 3),
                max_retries=config.get('max_retries', 3),
                timeout=config.get('timeout', 30)
            )
        
        elif gen_type == 'ollama':
            return generator_class(
                model=config.get('model_name', 'llama3'),
                base_url=config.get('base_url', 'http://localhost:11434'),
                num_contexts=config.get('num_contexts_to_use', 3)
            )
        
        raise ValueError(f"Unknown generator type: {gen_type}")
    
    @classmethod
    def list_generators(cls) -> List[str]:
        """Get list of available generator types."""
        return list(cls._generators.keys())
