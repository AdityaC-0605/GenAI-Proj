"""Statistical analysis for model comparison."""

import logging
import numpy as np
from typing import List, Dict, Tuple, Any
from scipy import stats

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Perform statistical analysis for model comparison."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            significance_level: Significance level for hypothesis tests
        """
        self.significance_level = significance_level
    
    def paired_t_test(
        self,
        model_a_scores: List[float],
        model_b_scores: List[float],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict[str, Any]:
        """
        Perform paired t-test comparing two models.
        
        Args:
            model_a_scores: Scores from model A
            model_b_scores: Scores from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            
        Returns:
            Dictionary with test results
        """
        if len(model_a_scores) != len(model_b_scores):
            raise ValueError("Score lists must have the same length")
        
        if len(model_a_scores) < 2:
            logger.warning("Not enough samples for t-test")
            return {
                'statistic': None,
                'p_value': None,
                'significant': False,
                'message': 'Insufficient samples'
            }
        
        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
        
        # Determine significance
        significant = p_value < self.significance_level
        
        # Determine which model is better
        mean_a = np.mean(model_a_scores)
        mean_b = np.mean(model_b_scores)
        
        if significant:
            if mean_a > mean_b:
                better_model = model_a_name
            else:
                better_model = model_b_name
        else:
            better_model = "No significant difference"
        
        result = {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'significance_level': self.significance_level,
            'mean_a': float(mean_a),
            'mean_b': float(mean_b),
            'better_model': better_model,
            'message': self._format_test_message(
                model_a_name, model_b_name, mean_a, mean_b, p_value, significant
            )
        }
        
        return result
    
    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000
    ) -> Dict[str, float]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            scores: List of scores
            confidence_level: Confidence level (default: 0.95)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with mean and confidence interval
        """
        if len(scores) < 2:
            logger.warning("Not enough samples for bootstrap")
            return {
                'mean': np.mean(scores) if scores else 0.0,
                'lower': 0.0,
                'upper': 0.0,
                'confidence_level': confidence_level
            }
        
        scores_array = np.array(scores)
        bootstrap_means = []
        
        # Generate bootstrap samples
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores_array, size=len(scores_array), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_means, lower_percentile)
        upper = np.percentile(bootstrap_means, upper_percentile)
        
        return {
            'mean': float(np.mean(scores_array)),
            'lower': float(lower),
            'upper': float(upper),
            'confidence_level': confidence_level
        }
    
    def cohens_d(
        self,
        model_a_scores: List[float],
        model_b_scores: List[float]
    ) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            model_a_scores: Scores from model A
            model_b_scores: Scores from model B
            
        Returns:
            Cohen's d effect size
        """
        if len(model_a_scores) < 2 or len(model_b_scores) < 2:
            return 0.0
        
        mean_a = np.mean(model_a_scores)
        mean_b = np.mean(model_b_scores)
        
        std_a = np.std(model_a_scores, ddof=1)
        std_b = np.std(model_b_scores, ddof=1)
        
        n_a = len(model_a_scores)
        n_b = len(model_b_scores)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            return 0.0
        
        d = (mean_a - mean_b) / pooled_std
        
        return float(d)
    
    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: float = None
    ) -> Dict[str, Any]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of p-values
            alpha: Significance level (uses instance level if None)
            
        Returns:
            Dictionary with corrected results
        """
        if alpha is None:
            alpha = self.significance_level
        
        n_comparisons = len(p_values)
        corrected_alpha = alpha / n_comparisons
        
        significant = [p < corrected_alpha for p in p_values]
        
        return {
            'original_alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'n_comparisons': n_comparisons,
            'p_values': p_values,
            'significant': significant,
            'n_significant': sum(significant)
        }
    
    def compare_multiple_models(
        self,
        model_scores: Dict[str, List[float]],
        apply_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple models with pairwise tests.
        
        Args:
            model_scores: Dictionary mapping model names to score lists
            apply_correction: Whether to apply Bonferroni correction
            
        Returns:
            Dictionary with all pairwise comparisons
        """
        model_names = list(model_scores.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            logger.warning("Need at least 2 models for comparison")
            return {}
        
        comparisons = []
        p_values = []
        
        # Perform all pairwise comparisons
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_a = model_names[i]
                model_b = model_names[j]
                
                result = self.paired_t_test(
                    model_scores[model_a],
                    model_scores[model_b],
                    model_a,
                    model_b
                )
                
                comparisons.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'result': result
                })
                
                if result['p_value'] is not None:
                    p_values.append(result['p_value'])
        
        # Apply Bonferroni correction if requested
        correction_result = None
        if apply_correction and p_values:
            correction_result = self.bonferroni_correction(p_values)
            
            # Update significance based on corrected alpha
            for i, comparison in enumerate(comparisons):
                if i < len(correction_result['significant']):
                    comparison['result']['significant_corrected'] = correction_result['significant'][i]
        
        return {
            'comparisons': comparisons,
            'correction': correction_result,
            'n_models': n_models,
            'n_comparisons': len(comparisons)
        }
    
    def _format_test_message(
        self,
        model_a_name: str,
        model_b_name: str,
        mean_a: float,
        mean_b: float,
        p_value: float,
        significant: bool
    ) -> str:
        """
        Format test result message.
        
        Args:
            model_a_name: Name of model A
            model_b_name: Name of model B
            mean_a: Mean score of model A
            mean_b: Mean score of model B
            p_value: P-value from test
            significant: Whether result is significant
            
        Returns:
            Formatted message
        """
        if significant:
            if mean_a > mean_b:
                return (f"{model_a_name} (mean={mean_a:.4f}) significantly outperforms "
                       f"{model_b_name} (mean={mean_b:.4f}) with p={p_value:.4f}")
            else:
                return (f"{model_b_name} (mean={mean_b:.4f}) significantly outperforms "
                       f"{model_a_name} (mean={mean_a:.4f}) with p={p_value:.4f}")
        else:
            return (f"No significant difference between {model_a_name} (mean={mean_a:.4f}) "
                   f"and {model_b_name} (mean={mean_b:.4f}) with p={p_value:.4f}")
    
    def interpret_effect_size(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            cohens_d: Cohen's d value
            
        Returns:
            Interpretation string
        """
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
