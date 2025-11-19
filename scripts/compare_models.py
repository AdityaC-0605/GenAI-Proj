#!/usr/bin/env python3
"""Model comparison script for Cross-Lingual QA."""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.model_comparison import ModelComparator
from src.evaluation.statistical_analysis import StatisticalAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two Cross-Lingual QA models"
    )
    
    parser.add_argument(
        '--results-a',
        type=str,
        required=True,
        help='Path to results file for model A'
    )
    
    parser.add_argument(
        '--results-b',
        type=str,
        required=True,
        help='Path to results file for model B'
    )
    
    parser.add_argument(
        '--model-a-name',
        type=str,
        default='Model A',
        help='Name for model A'
    )
    
    parser.add_argument(
        '--model-b-name',
        type=str,
        default='Model B',
        help='Name for model B'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/comparisons',
        help='Directory to save comparison results'
    )
    
    parser.add_argument(
        '--generate-visualizations',
        action='store_true',
        help='Generate comparison visualizations'
    )
    
    return parser.parse_args()


def load_results(results_path: str):
    """
    Load evaluation results from file.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        Dictionary of results
    """
    logger.info(f"Loading results from {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def main():
    """Main comparison function."""
    args = parse_args()
    
    logger.info("Starting model comparison")
    
    # Load results
    results_a = load_results(args.results_a)
    results_b = load_results(args.results_b)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparator
    comparator = ModelComparator()
    
    # Perform comparison
    logger.info("Comparing models...")
    
    comparison = comparator.compare(
        results_a=results_a,
        results_b=results_b,
        model_a_name=args.model_a_name,
        model_b_name=args.model_b_name
    )
    
    # Print comparison summary
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    
    logger.info(f"\nModel A: {args.model_a_name}")
    logger.info(f"Model B: {args.model_b_name}")
    
    # Aggregate metrics
    agg_a = results_a.get('aggregate_metrics', {})
    agg_b = results_b.get('aggregate_metrics', {})
    
    logger.info("\nAggregate Performance:")
    logger.info(f"  Exact Match:")
    logger.info(f"    {args.model_a_name}: {agg_a.get('exact_match', 0):.4f}")
    logger.info(f"    {args.model_b_name}: {agg_b.get('exact_match', 0):.4f}")
    logger.info(f"    Difference: {agg_a.get('exact_match', 0) - agg_b.get('exact_match', 0):.4f}")
    
    logger.info(f"  F1 Score:")
    logger.info(f"    {args.model_a_name}: {agg_a.get('f1_score', 0):.4f}")
    logger.info(f"    {args.model_b_name}: {agg_b.get('f1_score', 0):.4f}")
    logger.info(f"    Difference: {agg_a.get('f1_score', 0) - agg_b.get('f1_score', 0):.4f}")
    
    # Language pair comparison
    logger.info("\nLanguage Pair Comparison:")
    
    lang_pairs_a = results_a.get('language_pair_results', {})
    lang_pairs_b = results_b.get('language_pair_results', {})
    
    common_pairs = set(lang_pairs_a.keys()) & set(lang_pairs_b.keys())
    
    for pair in sorted(common_pairs):
        f1_a = lang_pairs_a[pair].get('f1_score', 0)
        f1_b = lang_pairs_b[pair].get('f1_score', 0)
        diff = f1_a - f1_b
        
        winner = args.model_a_name if diff > 0 else args.model_b_name
        
        logger.info(
            f"  {pair}: "
            f"A={f1_a:.4f}, B={f1_b:.4f}, "
            f"Diff={diff:+.4f} (Winner: {winner})"
        )
    
    # Statistical significance
    logger.info("\nStatistical Analysis:")
    
    analyzer = StatisticalAnalyzer()
    
    # Extract F1 scores for paired t-test
    f1_scores_a = [lang_pairs_a[pair]['f1_score'] for pair in common_pairs]
    f1_scores_b = [lang_pairs_b[pair]['f1_score'] for pair in common_pairs]
    
    t_stat, p_value = analyzer.paired_t_test(f1_scores_a, f1_scores_b)
    
    logger.info(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        logger.info("  Result: Statistically significant difference (p < 0.05)")
    else:
        logger.info("  Result: No statistically significant difference (p >= 0.05)")
    
    # Effect size
    effect_size = analyzer.cohens_d(f1_scores_a, f1_scores_b)
    logger.info(f"  Cohen's d: {effect_size:.4f}")
    
    # Save comparison results
    comparison_output = {
        'model_a': args.model_a_name,
        'model_b': args.model_b_name,
        'aggregate_comparison': {
            'exact_match': {
                'model_a': agg_a.get('exact_match', 0),
                'model_b': agg_b.get('exact_match', 0),
                'difference': agg_a.get('exact_match', 0) - agg_b.get('exact_match', 0)
            },
            'f1_score': {
                'model_a': agg_a.get('f1_score', 0),
                'model_b': agg_b.get('f1_score', 0),
                'difference': agg_a.get('f1_score', 0) - agg_b.get('f1_score', 0)
            }
        },
        'statistical_analysis': {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': effect_size,
            'significant': p_value < 0.05
        },
        'language_pair_comparison': {}
    }
    
    for pair in sorted(common_pairs):
        comparison_output['language_pair_comparison'][pair] = {
            'model_a_f1': lang_pairs_a[pair]['f1_score'],
            'model_b_f1': lang_pairs_b[pair]['f1_score'],
            'difference': lang_pairs_a[pair]['f1_score'] - lang_pairs_b[pair]['f1_score']
        }
    
    # Save to file
    output_path = output_dir / "comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(comparison_output, f, indent=2)
    
    logger.info(f"\nComparison results saved to {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
