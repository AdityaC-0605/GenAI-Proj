"""Core data models for the Cross-Lingual QA System."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class Answer:
    """Answer with position information."""
    text: str
    start_position: int
    end_position: int


@dataclass
class QAExample:
    """Single question-answering example."""
    id: str
    question: str
    context: str
    answers: List[Answer]
    question_language: str
    context_language: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAPrediction:
    """Model prediction."""
    answer_text: str
    confidence: float
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    alternative_answers: Optional[List[Tuple[str, float]]] = None


@dataclass
class EvaluationResult:
    """Evaluation results for a model."""
    model_name: str
    dataset_name: str
    language_pair: Tuple[str, str]
    exact_match: float
    f1_score: float
    num_examples: int
    predictions: List[QAPrediction]


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str
    model_type: str
    learning_mode: str  # zero-shot, few-shot
    num_shots: Optional[int]
    hyperparameters: Dict[str, Any]
    dataset_config: Dict[str, Any]
    random_seed: int
