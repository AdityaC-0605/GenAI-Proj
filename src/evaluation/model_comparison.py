"""Model comparison framework for Cross-Lingual QA."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from src.models.base_model import QAModelWrapper
from src.data_models import EvaluationResult
from src.evaluation.statistical_analysis import StatisticalAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Comparison results between two models."""
    model_a_name: str
    model_b_name: str
    performance_comparison: Dict[str, float]  # metric -> difference
    statistical_significance: Dict[str, bool]
    efficiency_comparison: Dict[str, float]
    best_for_scenarios: Dict[str, str]  # scenario -> better model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ComparisonAnalyzer:
    """Analyzer for comparing QA models."""
    
    def __init__(
        self,
        statistical_analyzer: Optional[StatisticalAnalyzer] = None,
        output_dir: str = "experiments/comparisons"
    ):
        """
        Initialize comparison analyzer.
        
        Args:
            statistical_analyzer: Statistical analyzer instance
            output_dir: Directory to save comparison results
        """
        self.statistical_analyzer = statistical_analyzer or StatisticalAnalyzer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_models(
        self,
        model_a: QAModelWrapper,
        model_b: QAModelWrapper,
        model_a_results: Dict[Tuple[str, str], EvaluationResult],
        model_b_results: Dict[Tuple[str, str], EvaluationResult],
        model_a_name: str,
        model_b_name: str,
        measure_efficiency: bool = True,
        save_results: bool = True
    ) -> ModelComparison:
        """
        Compare two models comprehensively.
        
        Args:
            model_a: First model
            model_b: Second model
            model_a_results: Evaluation results for model A
            model_b_results: Evaluation results for model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            measure_efficiency: Whether to measure inference efficiency
            save_results: Whether to save comparison results
            
        Returns:
            ModelComparison object
        """
        logger.info(f"Comparing {model_a_name} vs {model_b_name}")
        
        # Performance comparison
        performance_comparison = self._compare_performance(
            model_a_results,
            model_b_results
        )
        
        # Statistical significance
        statistical_significance = self._test_significance(
            model_a_results,
            model_b_results,
            model_a_name,
            model_b_name
        )
        
        # Efficiency comparison
        efficiency_comparison = {}
        if measure_efficiency:
            efficiency_comparison = self._compare_efficiency(
                model_a,
                model_b,
                model_a_name,
                model_b_name
            )
        
        # Best for scenarios
        best_for_scenarios = self._identify_best_scenarios(
            model_a_results,
            model_b_results,
            model_a_name,
            model_b_name
        )
        
        comparison = ModelComparison(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            performance_comparison=performance_comparison,
            statistical_significance=statistical_significance,
            efficiency_comparison=efficiency_comparison,
            best_for_scenarios=best_for_scenarios
        )
        
        if save_results:
            self._save_comparison(comparison)
        
        return comparison
    
    def _compare_performance(
        self,
        results_a: Dict[Tuple[str, str], EvaluationResult],
        results_b: Dict[Tuple[str, str], EvaluationResult]
    ) -> Dict[str, float]:
        """
        Compare performance metrics.
        
        Args:
            results_a: Results from model A
            results_b: Results from model B
            
        Returns:
            Dictionary of performance differences
        """
        # Calculate aggregate metrics
        agg_a = self._calculate_aggregate(results_a)
        agg_b = self._calculate_aggregate(results_b)
        
        comparison = {
            'f1_difference': agg_a['f1_score'] - agg_b['f1_score'],
            'em_difference': agg_a['exact_match'] - agg_b['exact_match'],
            'f1_relative': (agg_a['f1_score'] - agg_b['f1_score']) / agg_b['f1_score'] if agg_b['f1_score'] > 0 else 0.0,
            'em_relative': (agg_a['exact_match'] - agg_b['exact_match']) / agg_b['exact_match'] if agg_b['exact_match'] > 0 else 0.0
        }
        
        return comparison
    
    def _test_significance(
        self,
        results_a: Dict[Tuple[str, str], EvaluationResult],
        results_b: Dict[Tuple[str, str], EvaluationResult],
        model_a_name: str,
        model_b_name: str
    ) -> Dict[str, bool]:
        """
        Test statistical significance of differences.
        
        Args:
            results_a: Results from model A
            results_b: Results from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            
        Returns:
            Dictionary indicating significance for each metric
        """
        # Get common language pairs
        common_pairs = set(results_a.keys()) & set(results_b.keys())
        
        if not common_pairs:
            logger.warning("No common language pairs for significance testing")
            return {'f1_score': False, 'exact_match': False}
        
        # Extract scores for each language pair
        f1_scores_a = [results_a[pair].f1_score for pair in common_pairs]
        f1_scores_b = [results_b[pair].f1_score for pair in common_pairs]
        
        em_scores_a = [results_a[pair].exact_match for pair in common_pairs]
        em_scores_b = [results_b[pair].exact_match for pair in common_pairs]
        
        # Perform t-tests
        f1_test = self.statistical_analyzer.paired_t_test(
            f1_scores_a,
            f1_scores_b,
            model_a_name,
            model_b_name
        )
        
        em_test = self.statistical_analyzer.paired_t_test(
            em_scores_a,
            em_scores_b,
            model_a_name,
            model_b_name
        )
        
        return {
            'f1_score': f1_test['significant'],
            'exact_match': em_test['significant'],
            'f1_p_value': f1_test['p_value'],
            'em_p_value': em_test['p_value']
        }
    
    def _compare_efficiency(
        self,
        model_a: QAModelWrapper,
        model_b: QAModelWrapper,
        model_a_name: str,
        model_b_name: str,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Compare inference efficiency.
        
        Args:
            model_a: First model
            model_b: Second model
            model_a_name: Name of model A
            model_b_name: Name of model B
            num_samples: Number of samples for timing
            
        Returns:
            Dictionary of efficiency metrics
        """
        logger.info("Measuring inference efficiency...")
        
        # Sample question and context
        sample_question = "What is the capital of France?"
        sample_context = "Paris is the capital and most populous city of France."
        
        # Measure model A
        times_a = []
        for _ in range(num_samples):
            start = time.time()
            model_a.predict(sample_question, sample_context, "en", "en")
            times_a.append(time.time() - start)
        
        # Measure model B
        times_b = []
        for _ in range(num_samples):
            start = time.time()
            model_b.predict(sample_question, sample_context, "en", "en")
            times_b.append(time.time() - start)
        
        avg_latency_a = np.mean(times_a) * 1000  # Convert to ms
        avg_latency_b = np.mean(times_b) * 1000
        
        throughput_a = 1000 / avg_latency_a  # queries per second
        throughput_b = 1000 / avg_latency_b
        
        return {
            f'{model_a_name}_latency_ms': float(avg_latency_a),
            f'{model_b_name}_latency_ms': float(avg_latency_b),
            'latency_difference_ms': float(avg_latency_a - avg_latency_b),
            f'{model_a_name}_throughput_qps': float(throughput_a),
            f'{model_b_name}_throughput_qps': float(throughput_b),
            'throughput_ratio': float(throughput_a / throughput_b) if throughput_b > 0 else 0.0
        }
    
    def _identify_best_scenarios(
        self,
        results_a: Dict[Tuple[str, str], EvaluationResult],
        results_b: Dict[Tuple[str, str], EvaluationResult],
        model_a_name: str,
        model_b_name: str
    ) -> Dict[str, str]:
        """
        Identify which model is best for different scenarios.
        
        Args:
            results_a: Results from model A
            results_b: Results from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            
        Returns:
            Dictionary mapping scenarios to better model
        """
        scenarios = {}
        
        # Overall best
        agg_a = self._calculate_aggregate(results_a)
        agg_b = self._calculate_aggregate(results_b)
        
        if agg_a['f1_score'] > agg_b['f1_score']:
            scenarios['overall'] = model_a_name
        else:
            scenarios['overall'] = model_b_name
        
        # Best by language pair category
        categories = self._categorize_language_pairs(results_a, results_b)
        
        for category, pairs in categories.items():
            if not pairs:
                continue
            
            avg_f1_a = np.mean([results_a[pair].f1_score for pair in pairs if pair in results_a])
            avg_f1_b = np.mean([results_b[pair].f1_score for pair in pairs if pair in results_b])
            
            scenarios[category] = model_a_name if avg_f1_a > avg_f1_b else model_b_name
        
        return scenarios
    
    def _categorize_language_pairs(
        self,
        results_a: Dict[Tuple[str, str], EvaluationResult],
        results_b: Dict[Tuple[str, str], EvaluationResult]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Categorize language pairs by type.
        
        Args:
            results_a: Results from model A
            results_b: Results from model B
            
        Returns:
            Dictionary mapping categories to language pairs
        """
        common_pairs = set(results_a.keys()) & set(results_b.keys())
        
        high_resource = {'en', 'es', 'fr', 'de', 'zh'}
        low_resource = {'ar', 'hi', 'ja', 'ko'}
        
        similar_families = {
            ('en', 'en'), ('en', 'de'), ('es', 'fr'), ('fr', 'es'),
            ('de', 'en'), ('de', 'de')
        }
        
        categories = {
            'high_to_high': [],
            'high_to_low': [],
            'low_to_low': [],
            'similar_families': [],
            'distant_families': []
        }
        
        for q_lang, c_lang in common_pairs:
            # High/low resource categorization
            if q_lang in high_resource and c_lang in high_resource:
                categories['high_to_high'].append((q_lang, c_lang))
            elif q_lang in high_resource and c_lang in low_resource:
                categories['high_to_low'].append((q_lang, c_lang))
            elif q_lang in low_resource and c_lang in low_resource:
                categories['low_to_low'].append((q_lang, c_lang))
            
            # Language family categorization
            if (q_lang, c_lang) in similar_families or (c_lang, q_lang) in similar_families:
                categories['similar_families'].append((q_lang, c_lang))
            else:
                categories['distant_families'].append((q_lang, c_lang))
        
        return categories
    
    def _calculate_aggregate(
        self,
        results: Dict[Tuple[str, str], EvaluationResult]
    ) -> Dict[str, float]:
        """
        Calculate aggregate metrics.
        
        Args:
            results: Evaluation results
            
        Returns:
            Dictionary of aggregate metrics
        """
        if not results:
            return {'f1_score': 0.0, 'exact_match': 0.0}
        
        total_examples = sum(r.num_examples for r in results.values())
        
        weighted_f1 = sum(
            r.f1_score * r.num_examples for r in results.values()
        ) / total_examples
        
        weighted_em = sum(
            r.exact_match * r.num_examples for r in results.values()
        ) / total_examples
        
        return {
            'f1_score': weighted_f1,
            'exact_match': weighted_em
        }
    
    def calculate_transfer_efficiency(
        self,
        monolingual_results: Dict[Tuple[str, str], EvaluationResult],
        crosslingual_results: Dict[Tuple[str, str], EvaluationResult]
    ) -> Dict[str, float]:
        """
        Calculate transfer efficiency ratio.
        
        Args:
            monolingual_results: Results on monolingual pairs (e.g., en-en)
            crosslingual_results: Results on cross-lingual pairs
            
        Returns:
            Dictionary of transfer efficiency metrics
        """
        # Get monolingual performance (baseline)
        monolingual_pairs = [(q, c) for q, c in monolingual_results.keys() if q == c]
        
        if not monolingual_pairs:
            logger.warning("No monolingual pairs found for transfer efficiency calculation")
            return {}
        
        mono_f1 = np.mean([monolingual_results[pair].f1_score for pair in monolingual_pairs])
        mono_em = np.mean([monolingual_results[pair].exact_match for pair in monolingual_pairs])
        
        # Get cross-lingual performance
        cross_pairs = [(q, c) for q, c in crosslingual_results.keys() if q != c]
        
        if not cross_pairs:
            return {}
        
        cross_f1 = np.mean([crosslingual_results[pair].f1_score for pair in cross_pairs])
        cross_em = np.mean([crosslingual_results[pair].exact_match for pair in cross_pairs])
        
        return {
            'f1_transfer_ratio': float(cross_f1 / mono_f1) if mono_f1 > 0 else 0.0,
            'em_transfer_ratio': float(cross_em / mono_em) if mono_em > 0 else 0.0,
            'f1_gap': float(mono_f1 - cross_f1),
            'em_gap': float(mono_em - cross_em)
        }
    
    def _save_comparison(self, comparison: ModelComparison):
        """
        Save comparison results.
        
        Args:
            comparison: ModelComparison object
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{comparison.model_a_name}_vs_{comparison.model_b_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(comparison.to_dict(), f, indent=2)
        
        logger.info(f"Comparison results saved to {filepath}")
