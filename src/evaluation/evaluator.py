"""Evaluation runner for Cross-Lingual QA models."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime

from src.models.base_model import QAModelWrapper
from src.data_models import QAExample, QAPrediction, EvaluationResult
from src.evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation runner for QA models."""
    
    def __init__(
        self,
        metrics_calculator: Optional[MetricsCalculator] = None,
        output_dir: str = "experiments/evaluations"
    ):
        """
        Initialize evaluator.
        
        Args:
            metrics_calculator: Metrics calculator instance
            output_dir: Directory to save evaluation results
        """
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        model: QAModelWrapper,
        examples: List[QAExample],
        model_name: str,
        dataset_name: str,
        include_generative_metrics: bool = False,
        save_results: bool = True
    ) -> Dict[Tuple[str, str], EvaluationResult]:
        """
        Evaluate model on examples.
        
        Args:
            model: QA model wrapper
            examples: List of QA examples
            model_name: Name of the model
            dataset_name: Name of the dataset
            include_generative_metrics: Whether to include BLEU/ROUGE
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary mapping language pairs to evaluation results
        """
        logger.info(f"Starting evaluation of {model_name} on {dataset_name}")
        logger.info(f"Total examples: {len(examples)}")
        
        # Group examples by language pair
        language_pair_examples = self._group_by_language_pair(examples)
        
        logger.info(f"Evaluating {len(language_pair_examples)} language pairs")
        
        # Evaluate each language pair
        results = {}
        for lang_pair, pair_examples in language_pair_examples.items():
            logger.info(f"Evaluating language pair {lang_pair}: {len(pair_examples)} examples")
            
            result = self._evaluate_language_pair(
                model,
                pair_examples,
                model_name,
                dataset_name,
                lang_pair,
                include_generative_metrics
            )
            
            results[lang_pair] = result
            
            logger.info(
                f"Language pair {lang_pair}: "
                f"EM={result.exact_match:.4f}, F1={result.f1_score:.4f}"
            )
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        logger.info(
            f"Aggregate metrics: "
            f"EM={aggregate_metrics['exact_match']:.4f}, "
            f"F1={aggregate_metrics['f1_score']:.4f}"
        )
        
        # Save results
        if save_results:
            self._save_results(results, aggregate_metrics, model_name, dataset_name)
        
        return results
    
    def _evaluate_language_pair(
        self,
        model: QAModelWrapper,
        examples: List[QAExample],
        model_name: str,
        dataset_name: str,
        language_pair: Tuple[str, str],
        include_generative_metrics: bool
    ) -> EvaluationResult:
        """
        Evaluate model on a single language pair.
        
        Args:
            model: QA model wrapper
            examples: List of examples for this language pair
            model_name: Name of the model
            dataset_name: Name of the dataset
            language_pair: (question_lang, context_lang) tuple
            include_generative_metrics: Whether to include BLEU/ROUGE
            
        Returns:
            EvaluationResult for this language pair
        """
        predictions = []
        exact_matches = []
        f1_scores = []
        
        for example in examples:
            # Get prediction
            prediction = model.predict(
                question=example.question,
                context=example.context,
                question_lang=example.question_language,
                context_lang=example.context_language
            )
            
            predictions.append(prediction)
            
            # Calculate metrics for each ground truth answer
            example_em_scores = []
            example_f1_scores = []
            
            for answer in example.answers:
                em = self.metrics_calculator.exact_match(
                    prediction.answer_text,
                    answer.text
                )
                f1 = self.metrics_calculator.f1_score(
                    prediction.answer_text,
                    answer.text
                )
                
                example_em_scores.append(em)
                example_f1_scores.append(f1)
            
            # Take maximum score across all ground truth answers
            exact_matches.append(max(example_em_scores) if example_em_scores else 0.0)
            f1_scores.append(max(example_f1_scores) if example_f1_scores else 0.0)
        
        # Calculate average metrics
        avg_exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        return EvaluationResult(
            model_name=model_name,
            dataset_name=dataset_name,
            language_pair=language_pair,
            exact_match=avg_exact_match,
            f1_score=avg_f1_score,
            num_examples=len(examples),
            predictions=predictions
        )
    
    def _group_by_language_pair(
        self,
        examples: List[QAExample]
    ) -> Dict[Tuple[str, str], List[QAExample]]:
        """
        Group examples by language pair.
        
        Args:
            examples: List of QA examples
            
        Returns:
            Dictionary mapping language pairs to examples
        """
        grouped = defaultdict(list)
        
        for example in examples:
            lang_pair = (example.question_language, example.context_language)
            grouped[lang_pair].append(example)
        
        return dict(grouped)
    
    def _calculate_aggregate_metrics(
        self,
        results: Dict[Tuple[str, str], EvaluationResult]
    ) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all language pairs.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            Dictionary of aggregate metrics
        """
        if not results:
            return {'exact_match': 0.0, 'f1_score': 0.0}
        
        total_examples = sum(r.num_examples for r in results.values())
        
        # Weighted average by number of examples
        weighted_em = sum(
            r.exact_match * r.num_examples for r in results.values()
        ) / total_examples
        
        weighted_f1 = sum(
            r.f1_score * r.num_examples for r in results.values()
        ) / total_examples
        
        return {
            'exact_match': weighted_em,
            'f1_score': weighted_f1,
            'num_language_pairs': len(results),
            'total_examples': total_examples
        }
    
    def _save_results(
        self,
        results: Dict[Tuple[str, str], EvaluationResult],
        aggregate_metrics: Dict[str, float],
        model_name: str,
        dataset_name: str
    ):
        """
        Save evaluation results to disk.
        
        Args:
            results: Dictionary of evaluation results
            aggregate_metrics: Aggregate metrics
            model_name: Name of the model
            dataset_name: Name of the dataset
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert results to serializable format
        results_dict = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'timestamp': timestamp,
            'aggregate_metrics': aggregate_metrics,
            'language_pair_results': {}
        }
        
        for lang_pair, result in results.items():
            lang_pair_key = f"{lang_pair[0]}-{lang_pair[1]}"
            results_dict['language_pair_results'][lang_pair_key] = {
                'exact_match': result.exact_match,
                'f1_score': result.f1_score,
                'num_examples': result.num_examples
            }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load evaluation results from disk.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Dictionary of evaluation results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        return results
    
    def compare_results(
        self,
        results_a: Dict[Tuple[str, str], EvaluationResult],
        results_b: Dict[Tuple[str, str], EvaluationResult],
        model_a_name: str,
        model_b_name: str
    ) -> Dict[str, Any]:
        """
        Compare results from two models.
        
        Args:
            results_a: Results from model A
            results_b: Results from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'language_pair_comparisons': {}
        }
        
        # Compare each language pair
        common_pairs = set(results_a.keys()) & set(results_b.keys())
        
        for lang_pair in common_pairs:
            result_a = results_a[lang_pair]
            result_b = results_b[lang_pair]
            
            lang_pair_key = f"{lang_pair[0]}-{lang_pair[1]}"
            comparison['language_pair_comparisons'][lang_pair_key] = {
                'exact_match': {
                    model_a_name: result_a.exact_match,
                    model_b_name: result_b.exact_match,
                    'difference': result_a.exact_match - result_b.exact_match
                },
                'f1_score': {
                    model_a_name: result_a.f1_score,
                    model_b_name: result_b.f1_score,
                    'difference': result_a.f1_score - result_b.f1_score
                }
            }
        
        # Calculate aggregate comparison
        agg_a = self._calculate_aggregate_metrics(results_a)
        agg_b = self._calculate_aggregate_metrics(results_b)
        
        comparison['aggregate_comparison'] = {
            'exact_match': {
                model_a_name: agg_a['exact_match'],
                model_b_name: agg_b['exact_match'],
                'difference': agg_a['exact_match'] - agg_b['exact_match']
            },
            'f1_score': {
                model_a_name: agg_a['f1_score'],
                model_b_name: agg_b['f1_score'],
                'difference': agg_a['f1_score'] - agg_b['f1_score']
            }
        }
        
        return comparison
