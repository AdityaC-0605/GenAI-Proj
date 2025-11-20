#!/usr/bin/env python
"""
Script to compare RAG vs fine-tuning approaches.

Loads evaluation results from both approaches and generates comparison reports.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_results(file_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_metrics(rag_results: Dict, finetuned_results: Dict) -> Dict[str, Any]:
    """Compare metrics between RAG and fine-tuned approaches."""
    comparison = {
        'rag': {},
        'finetuned': {},
        'differences': {},
        'winner': {}
    }
    
    # Extract metrics
    if 'average_metrics' in rag_results:
        rag_metrics = rag_results['average_metrics']
        if 'generation' in rag_metrics:
            comparison['rag'] = rag_metrics['generation']
    
    if 'average_metrics' in finetuned_results:
        ft_metrics = finetuned_results['average_metrics']
        if 'generation' in ft_metrics:
            comparison['finetuned'] = ft_metrics['generation']
    
    # Calculate differences
    for metric in comparison['rag'].keys():
        if metric in comparison['finetuned']:
            rag_val = comparison['rag'][metric]
            ft_val = comparison['finetuned'][metric]
            diff = rag_val - ft_val
            comparison['differences'][metric] = diff
            comparison['winner'][metric] = 'RAG' if diff > 0 else 'Fine-tuned'
    
    return comparison


def generate_comparison_table(comparison: Dict) -> str:
    """Generate comparison table as string."""
    table = []
    table.append("=" * 80)
    table.append("RAG vs Fine-Tuning Comparison")
    table.append("=" * 80)
    table.append("")
    table.append(f"{'Metric':<20} {'RAG':<15} {'Fine-tuned':<15} {'Difference':<15} {'Winner':<15}")
    table.append("-" * 80)
    
    for metric in comparison['rag'].keys():
        rag_val = comparison['rag'][metric]
        ft_val = comparison['finetuned'].get(metric, 0.0)
        diff = comparison['differences'].get(metric, 0.0)
        winner = comparison['winner'].get(metric, 'N/A')
        
        table.append(
            f"{metric:<20} {rag_val:<15.4f} {ft_val:<15.4f} "
            f"{diff:+<15.4f} {winner:<15}"
        )
    
    table.append("=" * 80)
    return "\n".join(table)


def generate_timing_comparison(rag_results: Dict, finetuned_results: Dict) -> str:
    """Generate timing comparison."""
    table = []
    table.append("")
    table.append("Timing Comparison")
    table.append("-" * 80)
    
    rag_timing = rag_results.get('average_metrics', {}).get('timing', {})
    ft_timing = finetuned_results.get('average_metrics', {}).get('timing', {})
    
    table.append(f"{'Metric':<30} {'RAG':<20} {'Fine-tuned':<20}")
    table.append("-" * 80)
    
    for metric in ['retrieval_time', 'generation_time', 'total_time']:
        rag_val = rag_timing.get(metric, 0.0)
        ft_val = ft_timing.get(metric, 0.0)
        table.append(f"{metric:<30} {rag_val:<20.4f}s {ft_val:<20.4f}s")
    
    return "\n".join(table)


def generate_cost_analysis(rag_results: Dict, finetuned_results: Dict) -> str:
    """Generate cost analysis."""
    table = []
    table.append("")
    table.append("Cost Analysis")
    table.append("-" * 80)
    
    # Estimated costs (these would come from actual usage data)
    rag_setup_time = "5 minutes"
    ft_setup_time = "4-8 hours"
    
    rag_update_cost = "Add to database (~1s per doc)"
    ft_update_cost = "Retrain entire model (hours)"
    
    rag_inference_cost = "Storage + API/Compute"
    ft_inference_cost = "GPU inference"
    
    table.append(f"{'Aspect':<30} {'RAG':<25} {'Fine-tuned':<25}")
    table.append("-" * 80)
    table.append(f"{'Setup Time':<30} {rag_setup_time:<25} {ft_setup_time:<25}")
    table.append(f"{'Update Cost':<30} {rag_update_cost:<25} {ft_update_cost:<25}")
    table.append(f"{'Inference Cost':<30} {rag_inference_cost:<25} {ft_inference_cost:<25}")
    
    return "\n".join(table)


def generate_scenario_analysis() -> str:
    """Generate scenario-based analysis."""
    table = []
    table.append("")
    table.append("Scenario Analysis: When to Use Each Approach")
    table.append("=" * 80)
    
    scenarios = [
        {
            'scenario': 'Rapid Prototyping',
            'rag': '✓ Excellent - Deploy in minutes',
            'finetuned': '✗ Poor - Requires training time'
        },
        {
            'scenario': 'Frequent Updates',
            'rag': '✓ Excellent - Add to database',
            'finetuned': '✗ Poor - Requires retraining'
        },
        {
            'scenario': 'Explainability Required',
            'rag': '✓ Excellent - Shows sources',
            'finetuned': '✗ Poor - Black box'
        },
        {
            'scenario': 'Low Latency Critical',
            'rag': '~ Moderate - 2-5s',
            'finetuned': '✓ Excellent - ~100ms'
        },
        {
            'scenario': 'Maximum Accuracy',
            'rag': '~ Good - 85%+',
            'finetuned': '✓ Excellent - 90%+'
        },
        {
            'scenario': 'Cross-Lingual Support',
            'rag': '✓ Excellent - Native support',
            'finetuned': '~ Moderate - Needs multilingual training'
        },
        {
            'scenario': 'Limited Resources',
            'rag': '✓ Good - No training needed',
            'finetuned': '✗ Poor - Requires GPU training'
        }
    ]
    
    for scenario in scenarios:
        table.append(f"\n{scenario['scenario']}:")
        table.append(f"  RAG:        {scenario['rag']}")
        table.append(f"  Fine-tuned: {scenario['finetuned']}")
    
    return "\n".join(table)


def plot_comparison(comparison: Dict, output_file: str):
    """Generate comparison visualization."""
    try:
        metrics = list(comparison['rag'].keys())
        rag_values = [comparison['rag'][m] for m in metrics]
        ft_values = [comparison['finetuned'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, rag_values, width, label='RAG', alpha=0.8)
        ax.bar(x + width/2, ft_values, width, label='Fine-tuned', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('RAG vs Fine-Tuning: Metric Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    except Exception as e:
        print(f"Could not generate visualization: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare RAG vs fine-tuning approaches"
    )
    parser.add_argument(
        "--rag-results",
        type=str,
        required=True,
        help="Path to RAG evaluation results JSON"
    )
    parser.add_argument(
        "--finetuned-results",
        type=str,
        required=True,
        help="Path to fine-tuned model evaluation results JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.txt",
        help="Output file for comparison report"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="comparison_plot.png",
        help="Output file for comparison plot"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RAG vs Fine-Tuning Comparison")
    print("=" * 80)
    print()
    
    # Load results
    print(f"Loading RAG results from {args.rag_results}")
    rag_results = load_results(args.rag_results)
    
    print(f"Loading fine-tuned results from {args.finetuned_results}")
    finetuned_results = load_results(args.finetuned_results)
    
    # Compare metrics
    print("\nComparing metrics...")
    comparison = compare_metrics(rag_results, finetuned_results)
    
    # Generate report
    report = []
    report.append(generate_comparison_table(comparison))
    report.append(generate_timing_comparison(rag_results, finetuned_results))
    report.append(generate_cost_analysis(rag_results, finetuned_results))
    report.append(generate_scenario_analysis())
    
    report_text = "\n".join(report)
    
    # Print to console
    print("\n" + report_text)
    
    # Save to file
    with open(args.output, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to {args.output}")
    
    # Generate plot
    plot_comparison(comparison, args.plot)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    # Count wins
    rag_wins = sum(1 for w in comparison['winner'].values() if w == 'RAG')
    ft_wins = sum(1 for w in comparison['winner'].values() if w == 'Fine-tuned')
    
    print(f"RAG wins: {rag_wins} metrics")
    print(f"Fine-tuned wins: {ft_wins} metrics")
    print()
    print("Recommendation:")
    print("  - Use RAG for: Rapid deployment, frequent updates, explainability")
    print("  - Use Fine-tuning for: Maximum accuracy, low latency requirements")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
