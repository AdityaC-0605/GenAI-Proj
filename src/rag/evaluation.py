"""
Evaluation metrics for RAG system.

Implements retrieval and generation quality metrics.
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import re

from .logging_config import LoggerMixin


class RetrievalMetrics(LoggerMixin):
    """Calculate retrieval quality metrics."""
    
    def __init__(self):
        """Initialize retrieval metrics calculator."""
        super().__init__()
    
    def recall_at_k(self, 
                    retrieved_ids: List[str],
                    relevant_ids: Set[str],
                    k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0-1)
        """
        if not relevant_ids:
            return 0.0
        
        top_k = set(retrieved_ids[:k])
        hits = len(top_k & relevant_ids)
        
        return hits / len(relevant_ids)
    
    def mean_reciprocal_rank(self,
                            retrieved_ids: List[str],
                            relevant_ids: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs
            
        Returns:
            MRR score (0-1)
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg_at_k(self,
                  retrieved_ids: List[str],
                  relevant_ids: Set[str],
                  k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            NDCG@K score (0-1)
        """
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_ids:
                # Binary relevance: 1 if relevant, 0 otherwise
                rel = 1.0
                dcg += rel / np.log2(i + 2)  # i+2 because rank starts at 1
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_ids), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    def precision_at_k(self,
                      retrieved_ids: List[str],
                      relevant_ids: Set[str],
                      k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0-1)
        """
        if k == 0:
            return 0.0
        
        top_k = set(retrieved_ids[:k])
        hits = len(top_k & relevant_ids)
        
        return hits / k


class GenerationMetrics(LoggerMixin):
    """Calculate answer generation quality metrics."""
    
    def __init__(self):
        """Initialize generation metrics calculator."""
        super().__init__()
    
    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate Exact Match score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            EM score (0 or 1)
        """
        # Normalize strings
        pred_norm = self._normalize_answer(prediction)
        gt_norm = self._normalize_answer(ground_truth)
        
        return 1.0 if pred_norm == gt_norm else 0.0
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score (0-1)
        """
        pred_tokens = self._normalize_answer(prediction).split()
        gt_tokens = self._normalize_answer(ground_truth).split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        # Count common tokens
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def bleu_score(self, prediction: str, ground_truth: str, n: int = 4) -> float:
        """
        Calculate BLEU score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            n: Maximum n-gram order
            
        Returns:
            BLEU score (0-1)
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError:
            self.logger.warning("NLTK not available, returning 0.0 for BLEU")
            return 0.0
        
        pred_tokens = self._normalize_answer(prediction).split()
        gt_tokens = self._normalize_answer(ground_truth).split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        # Use smoothing to avoid zero scores
        smoothing = SmoothingFunction().method1
        
        return sentence_bleu(
            [gt_tokens],
            pred_tokens,
            smoothing_function=smoothing
        )
    
    def rouge_scores(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            self.logger.warning("rouge-score not available, returning zeros")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def _normalize_answer(self, text: str) -> str:
        """
        Normalize answer text for comparison.
        
        Args:
            text: Answer text
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text


class RAGEvaluator(LoggerMixin):
    """Complete RAG system evaluator."""
    
    def __init__(self):
        """Initialize RAG evaluator."""
        super().__init__()
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
    
    def evaluate_retrieval(self,
                          retrieved_ids: List[str],
                          relevant_ids: Set[str],
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate retrieval quality.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            k_values: List of K values for metrics
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        # Recall@K
        for k in k_values:
            results[f'recall@{k}'] = self.retrieval_metrics.recall_at_k(
                retrieved_ids, relevant_ids, k
            )
        
        # Precision@K
        for k in k_values:
            results[f'precision@{k}'] = self.retrieval_metrics.precision_at_k(
                retrieved_ids, relevant_ids, k
            )
        
        # MRR
        results['mrr'] = self.retrieval_metrics.mean_reciprocal_rank(
            retrieved_ids, relevant_ids
        )
        
        # NDCG@K
        for k in k_values:
            results[f'ndcg@{k}'] = self.retrieval_metrics.ndcg_at_k(
                retrieved_ids, relevant_ids, k
            )
        
        return results
    
    def evaluate_generation(self,
                           prediction: str,
                           ground_truth: str,
                           include_rouge: bool = True,
                           include_bleu: bool = True) -> Dict[str, float]:
        """
        Evaluate generation quality.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            include_rouge: Whether to calculate ROUGE scores
            include_bleu: Whether to calculate BLEU score
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        # Exact Match
        results['exact_match'] = self.generation_metrics.exact_match(
            prediction, ground_truth
        )
        
        # F1 Score
        results['f1'] = self.generation_metrics.f1_score(
            prediction, ground_truth
        )
        
        # BLEU
        if include_bleu:
            results['bleu'] = self.generation_metrics.bleu_score(
                prediction, ground_truth
            )
        
        # ROUGE
        if include_rouge:
            rouge_scores = self.generation_metrics.rouge_scores(
                prediction, ground_truth
            )
            results.update(rouge_scores)
        
        return results
    
    def evaluate_end_to_end(self,
                           prediction: str,
                           ground_truth: str,
                           retrieved_ids: List[str],
                           relevant_ids: Set[str],
                           retrieval_time: float,
                           generation_time: float) -> Dict[str, Any]:
        """
        Evaluate complete RAG pipeline.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            retrieved_ids: Retrieved document IDs
            relevant_ids: Relevant document IDs
            retrieval_time: Retrieval time in seconds
            generation_time: Generation time in seconds
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Retrieval metrics
        retrieval_results = self.evaluate_retrieval(retrieved_ids, relevant_ids)
        results['retrieval'] = retrieval_results
        
        # Generation metrics
        generation_results = self.evaluate_generation(prediction, ground_truth)
        results['generation'] = generation_results
        
        # Timing
        results['timing'] = {
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + generation_time
        }
        
        # Context relevance (did we retrieve any relevant docs?)
        results['context_relevance'] = 1.0 if retrieval_results['recall@5'] > 0 else 0.0
        
        return results
