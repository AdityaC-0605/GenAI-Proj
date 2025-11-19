"""Few-shot sampling utilities for cross-lingual QA training."""

import random
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

from src.data_models import QAExample

logger = logging.getLogger(__name__)


class FewShotSampler:
    """Sampler for few-shot learning examples."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize few-shot sampler.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def sample_examples(
        self,
        examples: List[QAExample],
        num_shots: int,
        balance_by_language: bool = True,
        balance_by_question_type: bool = True
    ) -> List[QAExample]:
        """
        Sample k examples per language pair with stratification.
        
        Args:
            examples: List of all available examples
            num_shots: Number of examples to sample per language pair
            balance_by_language: Whether to balance across language pairs
            balance_by_question_type: Whether to ensure diverse question types
            
        Returns:
            List of sampled examples
        """
        if balance_by_language:
            return self._sample_balanced_by_language(
                examples, 
                num_shots, 
                balance_by_question_type
            )
        else:
            return self._sample_random(examples, num_shots)
    
    def _sample_balanced_by_language(
        self,
        examples: List[QAExample],
        num_shots: int,
        balance_by_question_type: bool
    ) -> List[QAExample]:
        """
        Sample examples balanced across language pairs.
        
        Args:
            examples: List of all available examples
            num_shots: Number of examples per language pair
            balance_by_question_type: Whether to ensure diverse question types
            
        Returns:
            List of sampled examples
        """
        # Group examples by language pair
        language_pair_groups = defaultdict(list)
        for example in examples:
            lang_pair = (example.question_language, example.context_language)
            language_pair_groups[lang_pair].append(example)
        
        sampled_examples = []
        
        for lang_pair, pair_examples in language_pair_groups.items():
            if len(pair_examples) < num_shots:
                logger.warning(
                    f"Language pair {lang_pair} has only {len(pair_examples)} examples, "
                    f"requested {num_shots}. Using all available."
                )
                sampled_examples.extend(pair_examples)
            else:
                if balance_by_question_type:
                    sampled = self._sample_diverse_questions(pair_examples, num_shots)
                else:
                    sampled = random.sample(pair_examples, num_shots)
                
                sampled_examples.extend(sampled)
                logger.info(
                    f"Sampled {len(sampled)} examples for language pair {lang_pair}"
                )
        
        # Shuffle to mix language pairs
        random.shuffle(sampled_examples)
        
        logger.info(
            f"Total sampled examples: {len(sampled_examples)} "
            f"across {len(language_pair_groups)} language pairs"
        )
        
        return sampled_examples
    
    def _sample_diverse_questions(
        self,
        examples: List[QAExample],
        num_shots: int
    ) -> List[QAExample]:
        """
        Sample examples ensuring diverse question types.
        
        Args:
            examples: List of examples for a language pair
            num_shots: Number of examples to sample
            
        Returns:
            List of sampled examples with diverse question types
        """
        # Categorize by question type
        question_type_groups = defaultdict(list)
        
        for example in examples:
            q_type = self._get_question_type(example.question)
            question_type_groups[q_type].append(example)
        
        # Calculate samples per question type
        num_types = len(question_type_groups)
        if num_types == 0:
            return random.sample(examples, min(num_shots, len(examples)))
        
        samples_per_type = num_shots // num_types
        remainder = num_shots % num_types
        
        sampled = []
        question_types = list(question_type_groups.keys())
        random.shuffle(question_types)
        
        for i, q_type in enumerate(question_types):
            type_examples = question_type_groups[q_type]
            # Give remainder to first few types
            n_samples = samples_per_type + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(type_examples))
            
            sampled.extend(random.sample(type_examples, n_samples))
        
        # If we still need more examples (due to small groups), sample randomly
        if len(sampled) < num_shots:
            remaining = [ex for ex in examples if ex not in sampled]
            additional = min(num_shots - len(sampled), len(remaining))
            sampled.extend(random.sample(remaining, additional))
        
        return sampled[:num_shots]
    
    def _sample_random(
        self,
        examples: List[QAExample],
        num_shots: int
    ) -> List[QAExample]:
        """
        Sample examples randomly without stratification.
        
        Args:
            examples: List of all available examples
            num_shots: Total number of examples to sample
            
        Returns:
            List of randomly sampled examples
        """
        n_samples = min(num_shots, len(examples))
        return random.sample(examples, n_samples)
    
    def _get_question_type(self, question: str) -> str:
        """
        Determine question type based on question word.
        
        Args:
            question: Question text
            
        Returns:
            Question type (what, when, where, who, why, how, other)
        """
        question_lower = question.lower().strip()
        
        # English question words
        if any(question_lower.startswith(word) for word in ['what', 'qué', 'quoi', 'was', 'que']):
            return 'what'
        elif any(question_lower.startswith(word) for word in ['when', 'cuándo', 'quand', 'wann']):
            return 'when'
        elif any(question_lower.startswith(word) for word in ['where', 'dónde', 'où', 'wo']):
            return 'where'
        elif any(question_lower.startswith(word) for word in ['who', 'quién', 'qui', 'wer']):
            return 'who'
        elif any(question_lower.startswith(word) for word in ['why', 'por qué', 'pourquoi', 'warum']):
            return 'why'
        elif any(question_lower.startswith(word) for word in ['how', 'cómo', 'comment', 'wie']):
            return 'how'
        else:
            return 'other'
    
    def get_language_pair_distribution(
        self,
        examples: List[QAExample]
    ) -> Dict[Tuple[str, str], int]:
        """
        Get distribution of examples across language pairs.
        
        Args:
            examples: List of examples
            
        Returns:
            Dictionary mapping language pairs to counts
        """
        distribution = defaultdict(int)
        for example in examples:
            lang_pair = (example.question_language, example.context_language)
            distribution[lang_pair] += 1
        
        return dict(distribution)
    
    def get_question_type_distribution(
        self,
        examples: List[QAExample]
    ) -> Dict[str, int]:
        """
        Get distribution of question types.
        
        Args:
            examples: List of examples
            
        Returns:
            Dictionary mapping question types to counts
        """
        distribution = defaultdict(int)
        for example in examples:
            q_type = self._get_question_type(example.question)
            distribution[q_type] += 1
        
        return dict(distribution)
