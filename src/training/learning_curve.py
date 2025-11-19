"""Learning curve generation for few-shot experiments."""

import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class LearningCurveGenerator:
    """Generate learning curves showing performance vs number of shots."""
    
    def __init__(self, output_dir: str = "experiments/learning_curves"):
        """
        Initialize learning curve generator.
        
        Args:
            output_dir: Directory to save learning curve plots and data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def generate_learning_curve(
        self,
        shot_results: Dict[int, Dict[str, float]],
        metric_name: str = "f1_score",
        model_name: str = "model",
        save_plot: bool = True,
        save_data: bool = True
    ) -> Dict[str, Any]:
        """
        Generate learning curve from few-shot experiment results.
        
        Args:
            shot_results: Dictionary mapping num_shots to metrics
                         e.g., {1: {'f1_score': 0.45, 'exact_match': 0.32}, ...}
            metric_name: Name of metric to plot
            model_name: Name of the model
            save_plot: Whether to save the plot
            save_data: Whether to save the data as JSON
            
        Returns:
            Dictionary containing curve data and statistics
        """
        if not shot_results:
            logger.warning("No results provided for learning curve generation")
            return {}
        
        # Extract data
        shot_counts = sorted(shot_results.keys())
        metric_values = [shot_results[k].get(metric_name, 0.0) for k in shot_counts]
        
        # Calculate statistics
        curve_data = {
            'model_name': model_name,
            'metric_name': metric_name,
            'shot_counts': shot_counts,
            'metric_values': metric_values,
            'improvement': self._calculate_improvement(metric_values),
            'efficiency': self._calculate_efficiency(shot_counts, metric_values)
        }
        
        # Create plot
        if save_plot:
            self._plot_learning_curve(
                shot_counts,
                metric_values,
                metric_name,
                model_name
            )
        
        # Save data
        if save_data:
            self._save_curve_data(curve_data, model_name, metric_name)
        
        logger.info(f"Learning curve generated for {model_name} on {metric_name}")
        return curve_data
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[int, Dict[str, float]]],
        metric_name: str = "f1_score",
        save_plot: bool = True
    ) -> Dict[str, Any]:
        """
        Compare learning curves across multiple models.
        
        Args:
            model_results: Dictionary mapping model names to shot results
                          e.g., {'mbert': {1: {...}, 5: {...}}, 'mt5': {...}}
            metric_name: Name of metric to plot
            save_plot: Whether to save the comparison plot
            
        Returns:
            Dictionary containing comparison data
        """
        if not model_results:
            logger.warning("No model results provided for comparison")
            return {}
        
        comparison_data = {
            'metric_name': metric_name,
            'models': {}
        }
        
        plt.figure(figsize=(12, 7))
        
        for model_name, shot_results in model_results.items():
            shot_counts = sorted(shot_results.keys())
            metric_values = [shot_results[k].get(metric_name, 0.0) for k in shot_counts]
            
            # Plot
            plt.plot(
                shot_counts,
                metric_values,
                marker='o',
                linewidth=2,
                markersize=8,
                label=model_name
            )
            
            # Store data
            comparison_data['models'][model_name] = {
                'shot_counts': shot_counts,
                'metric_values': metric_values,
                'improvement': self._calculate_improvement(metric_values),
                'efficiency': self._calculate_efficiency(shot_counts, metric_values)
            }
        
        # Format plot
        plt.xlabel('Number of Shots', fontsize=12)
        plt.ylabel(self._format_metric_name(metric_name), fontsize=12)
        plt.title(f'Learning Curves: {self._format_metric_name(metric_name)} vs Number of Shots', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis to log scale if appropriate
        if max(shot_counts) / min(shot_counts) > 10:
            plt.xscale('log')
            plt.xticks(shot_counts, shot_counts)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / f"learning_curve_comparison_{metric_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {plot_path}")
        
        plt.close()
        
        return comparison_data
    
    def _plot_learning_curve(
        self,
        shot_counts: List[int],
        metric_values: List[float],
        metric_name: str,
        model_name: str
    ):
        """
        Create and save learning curve plot.
        
        Args:
            shot_counts: List of shot counts
            metric_values: List of metric values
            metric_name: Name of the metric
            model_name: Name of the model
        """
        plt.figure(figsize=(10, 6))
        
        # Plot line with markers
        plt.plot(
            shot_counts,
            metric_values,
            marker='o',
            linewidth=2,
            markersize=8,
            color='#2E86AB',
            label=model_name
        )
        
        # Add value labels on points
        for x, y in zip(shot_counts, metric_values):
            plt.annotate(
                f'{y:.3f}',
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9
            )
        
        # Format plot
        plt.xlabel('Number of Shots', fontsize=12)
        plt.ylabel(self._format_metric_name(metric_name), fontsize=12)
        plt.title(f'{model_name}: {self._format_metric_name(metric_name)} vs Number of Shots', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis to log scale if appropriate
        if max(shot_counts) / min(shot_counts) > 10:
            plt.xscale('log')
            plt.xticks(shot_counts, shot_counts)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"learning_curve_{model_name}_{metric_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curve plot saved to {plot_path}")
        
        plt.close()
    
    def _calculate_improvement(self, metric_values: List[float]) -> Dict[str, float]:
        """
        Calculate improvement statistics.
        
        Args:
            metric_values: List of metric values
            
        Returns:
            Dictionary with improvement statistics
        """
        if len(metric_values) < 2:
            return {'total': 0.0, 'average_per_step': 0.0}
        
        total_improvement = metric_values[-1] - metric_values[0]
        avg_improvement = total_improvement / (len(metric_values) - 1)
        
        return {
            'total': float(total_improvement),
            'average_per_step': float(avg_improvement),
            'relative': float(total_improvement / metric_values[0]) if metric_values[0] > 0 else 0.0
        }
    
    def _calculate_efficiency(
        self,
        shot_counts: List[int],
        metric_values: List[float]
    ) -> Dict[str, float]:
        """
        Calculate learning efficiency (improvement per shot).
        
        Args:
            shot_counts: List of shot counts
            metric_values: List of metric values
            
        Returns:
            Dictionary with efficiency metrics
        """
        if len(shot_counts) < 2:
            return {'overall': 0.0}
        
        # Calculate efficiency between consecutive points
        efficiencies = []
        for i in range(1, len(shot_counts)):
            shot_diff = shot_counts[i] - shot_counts[i-1]
            metric_diff = metric_values[i] - metric_values[i-1]
            if shot_diff > 0:
                efficiencies.append(metric_diff / shot_diff)
        
        overall_efficiency = (metric_values[-1] - metric_values[0]) / (shot_counts[-1] - shot_counts[0])
        
        return {
            'overall': float(overall_efficiency),
            'per_interval': [float(e) for e in efficiencies],
            'diminishing_returns': efficiencies[0] > efficiencies[-1] if len(efficiencies) > 1 else False
        }
    
    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric name for display."""
        name_map = {
            'f1_score': 'F1 Score',
            'exact_match': 'Exact Match',
            'bleu_score': 'BLEU Score',
            'rouge_l': 'ROUGE-L'
        }
        return name_map.get(metric_name, metric_name.replace('_', ' ').title())
    
    def _save_curve_data(
        self,
        curve_data: Dict[str, Any],
        model_name: str,
        metric_name: str
    ):
        """
        Save learning curve data as JSON.
        
        Args:
            curve_data: Learning curve data
            model_name: Name of the model
            metric_name: Name of the metric
        """
        data_path = self.output_dir / f"learning_curve_{model_name}_{metric_name}.json"
        
        with open(data_path, 'w') as f:
            json.dump(curve_data, f, indent=2)
        
        logger.info(f"Learning curve data saved to {data_path}")
