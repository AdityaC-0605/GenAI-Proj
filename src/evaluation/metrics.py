"""Metrics calculation for question answering evaluation."""

import re
import string
import logging
from typing import List, Dict, Any
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate QA evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate exact match score after normalization.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        pred_normalized = self._normalize_text(prediction)
        gt_normalized = self._normalize_text(ground_truth)
        
        return 1.0 if pred_normalized == gt_normalized else 0.0
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score between 0.0 and 1.0
        """
        pred_tokens = self._tokenize(self._normalize_text(prediction))
        gt_tokens = self._tokenize(self._normalize_text(ground_truth))
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        # Calculate token overlap
        pred_counter = Counter(pred_tokens)
        gt_counter = Counter(gt_tokens)
        
        common = pred_counter & gt_counter
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def bleu_score(self, prediction: str, ground_truth: str, max_n: int = 4) -> float:
        """
        Calculate BLEU score for generative models.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            max_n: Maximum n-gram size
            
        Returns:
            BLEU score between 0.0 and 1.0
        """
        pred_tokens = self._tokenize(self._normalize_text(prediction))
        gt_tokens = self._tokenize(self._normalize_text(ground_truth))
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # Calculate brevity penalty
        bp = self._brevity_penalty(len(pred_tokens), len(gt_tokens))
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            gt_ngrams = self._get_ngrams(gt_tokens, n)
            
            if len(pred_ngrams) == 0:
                break
            
            matches = sum((pred_ngrams & gt_ngrams).values())
            precision = matches / len(pred_ngrams)
            precisions.append(precision)
        
        if len(precisions) == 0:
            return 0.0
        
        # Geometric mean of precisions
        geo_mean = np.exp(np.mean([np.log(p) if p > 0 else -np.inf for p in precisions]))
        
        if np.isinf(geo_mean) or np.isnan(geo_mean):
            return 0.0
        
        bleu = bp * geo_mean
        
        return float(bleu)
    
    def rouge_scores(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        pred_tokens = self._tokenize(self._normalize_text(prediction))
        gt_tokens = self._tokenize(self._normalize_text(ground_truth))
        
        scores = {
            'rouge_1': self._rouge_n(pred_tokens, gt_tokens, 1),
            'rouge_2': self._rouge_n(pred_tokens, gt_tokens, 2),
            'rouge_l': self._rouge_l(pred_tokens, gt_tokens)
        }
        
        return scores
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove articles (English)
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """
        Get n-grams from tokens.
        
        Args:
            tokens: List of tokens
            n: N-gram size
            
        Returns:
            Counter of n-grams
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        
        return Counter(ngrams)
    
    def _brevity_penalty(self, pred_len: int, ref_len: int) -> float:
        """
        Calculate BLEU brevity penalty.
        
        Args:
            pred_len: Length of prediction
            ref_len: Length of reference
            
        Returns:
            Brevity penalty
        """
        if pred_len >= ref_len:
            return 1.0
        
        return np.exp(1 - ref_len / pred_len)
    
    def _rouge_n(self, pred_tokens: List[str], gt_tokens: List[str], n: int) -> float:
        """
        Calculate ROUGE-N score.
        
        Args:
            pred_tokens: Predicted tokens
            gt_tokens: Ground truth tokens
            n: N-gram size
            
        Returns:
            ROUGE-N F1 score
        """
        if len(pred_tokens) < n or len(gt_tokens) < n:
            return 0.0
        
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        gt_ngrams = self._get_ngrams(gt_tokens, n)
        
        if len(gt_ngrams) == 0:
            return 0.0
        
        matches = sum((pred_ngrams & gt_ngrams).values())
        
        if matches == 0:
            return 0.0
        
        precision = matches / sum(pred_ngrams.values()) if sum(pred_ngrams.values()) > 0 else 0.0
        recall = matches / sum(gt_ngrams.values())
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def _rouge_l(self, pred_tokens: List[str], gt_tokens: List[str]) -> float:
        """
        Calculate ROUGE-L score (longest common subsequence).
        
        Args:
            pred_tokens: Predicted tokens
            gt_tokens: Ground truth tokens
            
        Returns:
            ROUGE-L F1 score
        """
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        lcs_length = self._lcs_length(pred_tokens, gt_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(gt_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Calculate longest common subsequence length.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Length of LCS
        """
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def calculate_all_metrics(
        self,
        prediction: str,
        ground_truth: str,
        include_generative: bool = False
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            include_generative: Whether to include generative metrics (BLEU, ROUGE)
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'exact_match': self.exact_match(prediction, ground_truth),
            'f1_score': self.f1_score(prediction, ground_truth)
        }
        
        if include_generative:
            metrics['bleu_score'] = self.bleu_score(prediction, ground_truth)
            rouge = self.rouge_scores(prediction, ground_truth)
            metrics.update(rouge)
        
        return metrics
