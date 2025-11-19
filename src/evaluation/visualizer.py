"""Visualization module for Cross-Lingual QA evaluation results."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceVisualizer:
    """Create performance visualizations."""
    
    def __init__(self, output_dir: str = "experiments/visualizations"):
        """
        Initialize performance visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def create_performance_heatmap(
        self,
        results: Dict[Tuple[str, str], float],
        metric_name: str = "F1 Score",
        title: str = "Cross-Lingual QA Performance",
        output_filename: str = "performance_heatmap.png"
    ):
        """
        Generate performance heatmap for language pairs.
        
        Args:
            results: Dictionary mapping (q_lang, c_lang) to metric value
            metric_name: Name of the metric
            title: Plot title
            output_filename: Output filename
        """
        logger.info(f"Creating performance heatmap: {title}")
        
        # Extract unique languages
        q_langs = sorted(set(pair[0] for pair in results.keys()))
        c_langs = sorted(set(pair[1] for pair in results.keys()))
        
        # Create matrix
        matrix = np.zeros((len(q_langs), len(c_langs)))
        
        for i, q_lang in enumerate(q_langs):
            for j, c_lang in enumerate(c_langs):
                if (q_lang, c_lang) in results:
                    matrix[i, j] = results[(q_lang, c_lang)]
                else:
                    matrix[i, j] = np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=c_langs,
            yticklabels=q_langs,
            cbar_kws={'label': metric_name},
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_xlabel('Context Language', fontsize=12)
        ax.set_ylabel('Question Language', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved to {output_path}")
    
    def create_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Optional[List[float]] = None,
        val_metrics: Optional[List[float]] = None,
        metric_name: str = "F1 Score",
        output_filename: str = "training_curves.png"
    ):
        """
        Generate training and validation curves.
        
        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            train_metrics: Training metrics per epoch (optional)
            val_metrics: Validation metrics per epoch (optional)
            metric_name: Name of the metric
            output_filename: Output filename
        """
        logger.info("Creating training curves")
        
        epochs = list(range(1, len(train_losses) + 1))
        
        # Determine number of subplots
        n_plots = 2 if train_metrics is not None else 1
        
        fig, axes = plt.subplots(1, n_plots, figsize=(12 * n_plots, 8))
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot losses
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot metrics if provided
        if train_metrics is not None and val_metrics is not None:
            axes[1].plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}', linewidth=2)
            axes[1].plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel(metric_name, fontsize=12)
            axes[1].set_title(f'Training and Validation {metric_name}', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {output_path}")
    
    def create_few_shot_learning_curves(
        self,
        shot_counts: List[int],
        scores: Dict[int, float],
        metric_name: str = "F1 Score",
        title: str = "Few-Shot Learning Curve",
        output_filename: str = "few_shot_learning_curve.png"
    ):
        """
        Generate few-shot learning curves.
        
        Args:
            shot_counts: List of shot counts
            scores: Dictionary mapping shot count to metric value
            metric_name: Name of the metric
            title: Plot title
            output_filename: Output filename
        """
        logger.info("Creating few-shot learning curves")
        
        # Sort by shot count
        sorted_shots = sorted(shot_counts)
        sorted_scores = [scores[shot] for shot in sorted_shots]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sorted_shots, sorted_scores, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Shots', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for shot, score in zip(sorted_shots, sorted_scores):
            ax.annotate(
                f'{score:.3f}',
                (shot, score),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9
            )
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Few-shot learning curve saved to {output_path}")


class ErrorVisualizer:
    """Create error analysis visualizations."""
    
    def __init__(self, output_dir: str = "experiments/visualizations"):
        """
        Initialize error visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
    
    def create_error_distribution_chart(
        self,
        error_counts: Dict[str, int],
        title: str = "Error Distribution by Category",
        output_filename: str = "error_distribution.png"
    ):
        """
        Generate error distribution bar chart.
        
        Args:
            error_counts: Dictionary mapping error categories to counts
            title: Plot title
            output_filename: Output filename
        """
        logger.info("Creating error distribution chart")
        
        categories = list(error_counts.keys())
        counts = list(error_counts.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(categories, counts, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        ax.set_xlabel('Error Category', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error distribution chart saved to {output_path}")
    
    def create_linguistic_distance_scatter(
        self,
        distance_data: Dict[str, List[Tuple[Tuple[str, str], float]]],
        metric_name: str = "F1 Score",
        title: str = "Performance vs Linguistic Distance",
        output_filename: str = "linguistic_distance_scatter.png"
    ):
        """
        Generate scatter plot of performance vs linguistic distance.
        
        Args:
            distance_data: Dictionary mapping distance categories to (lang_pair, score) tuples
            metric_name: Name of the metric
            title: Plot title
            output_filename: Output filename
        """
        logger.info("Creating linguistic distance scatter plot")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {'same_language': 'green', 'similar_family': 'orange', 'different_family': 'red'}
        labels = {
            'same_language': 'Same Language',
            'similar_family': 'Similar Family',
            'different_family': 'Different Family'
        }
        
        for category, data in distance_data.items():
            if data:
                scores = [score for _, score in data]
                x_positions = [list(distance_data.keys()).index(category)] * len(scores)
                
                ax.scatter(
                    x_positions,
                    scores,
                    c=colors[category],
                    label=labels[category],
                    alpha=0.6,
                    s=100
                )
        
        ax.set_xticks(range(len(distance_data)))
        ax.set_xticklabels([labels[cat] for cat in distance_data.keys()])
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Linguistic distance scatter plot saved to {output_path}")
    
    def create_confusion_matrix(
        self,
        confusion_data: Dict[str, Dict[str, int]],
        title: str = "Confusion Matrix by Question Type",
        output_filename: str = "confusion_matrix.png"
    ):
        """
        Generate confusion matrix visualization.
        
        Args:
            confusion_data: Dictionary mapping question types to correct/incorrect counts
            title: Plot title
            output_filename: Output filename
        """
        logger.info("Creating confusion matrix")
        
        question_types = list(confusion_data.keys())
        correct_counts = [confusion_data[qt]['correct'] for qt in question_types]
        incorrect_counts = [confusion_data[qt]['incorrect'] for qt in question_types]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(question_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, correct_counts, width, label='Correct', color='green', alpha=0.8)
        bars2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='red', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        
        ax.set_xlabel('Question Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(question_types)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")


class HTMLReportGenerator:
    """Generate HTML reports with visualizations."""
    
    def __init__(self, output_dir: str = "experiments/reports"):
        """
        Initialize HTML report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        model_name: str,
        dataset_name: str,
        aggregate_metrics: Dict[str, float],
        language_pair_results: Dict[str, Dict[str, float]],
        error_analysis: Dict[str, Any],
        visualization_paths: Dict[str, str],
        output_filename: str = "evaluation_report.html"
    ):
        """
        Generate interactive HTML report.
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            aggregate_metrics: Aggregate metrics
            language_pair_results: Results by language pair
            error_analysis: Error analysis results
            visualization_paths: Paths to visualization images
            output_filename: Output filename
        """
        logger.info(f"Generating HTML report for {model_name}")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report: {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .visualization {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
        }}
        .error-category {{
            display: inline-block;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 4px;
            background-color: #e74c3c;
            color: white;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Cross-Lingual QA Evaluation Report</h1>
    
    <div class="metric-card">
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Dataset:</strong> {dataset_name}</p>
        <p><strong>Generated:</strong> {self._get_timestamp()}</p>
    </div>
    
    <h2>Aggregate Metrics</h2>
    <div style="display: flex; gap: 20px;">
        <div class="metric-card" style="flex: 1;">
            <div class="metric-label">Exact Match</div>
            <div class="metric-value">{aggregate_metrics.get('exact_match', 0):.3f}</div>
        </div>
        <div class="metric-card" style="flex: 1;">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">{aggregate_metrics.get('f1_score', 0):.3f}</div>
        </div>
        <div class="metric-card" style="flex: 1;">
            <div class="metric-label">Language Pairs</div>
            <div class="metric-value">{aggregate_metrics.get('num_language_pairs', 0)}</div>
        </div>
    </div>
    
    <h2>Performance by Language Pair</h2>
    <table>
        <thead>
            <tr>
                <th>Language Pair</th>
                <th>Exact Match</th>
                <th>F1 Score</th>
                <th>Examples</th>
            </tr>
        </thead>
        <tbody>
"""
        
        # Add language pair results
        for lang_pair, metrics in sorted(language_pair_results.items()):
            html_content += f"""
            <tr>
                <td>{lang_pair}</td>
                <td>{metrics.get('exact_match', 0):.3f}</td>
                <td>{metrics.get('f1_score', 0):.3f}</td>
                <td>{metrics.get('num_examples', 0)}</td>
            </tr>
"""
        
        html_content += """
        </tbody>
    </table>
    
    <h2>Performance Visualizations</h2>
"""
        
        # Add visualizations
        for viz_name, viz_path in visualization_paths.items():
            html_content += f"""
    <div class="visualization">
        <h3>{viz_name}</h3>
        <img src="{viz_path}" alt="{viz_name}">
    </div>
"""
        
        # Add error analysis
        if error_analysis:
            html_content += """
    <h2>Error Analysis</h2>
    <div class="metric-card">
        <h3>Error Distribution</h3>
"""
            
            error_dist = error_analysis.get('error_distribution', {})
            for error_type, count in error_dist.items():
                html_content += f'<span class="error-category">{error_type}: {count}</span>'
            
            html_content += """
    </div>
"""
        
        html_content += """
    <h2>Summary</h2>
    <div class="metric-card">
        <p>This report provides a comprehensive evaluation of the cross-lingual question answering model.</p>
        <p>Key findings:</p>
        <ul>
            <li>The model was evaluated on {num_pairs} language pairs</li>
            <li>Overall F1 score: {f1:.3f}</li>
            <li>Overall Exact Match: {em:.3f}</li>
        </ul>
    </div>
    
</body>
</html>
""".format(
            num_pairs=aggregate_metrics.get('num_language_pairs', 0),
            f1=aggregate_metrics.get('f1_score', 0),
            em=aggregate_metrics.get('exact_match', 0)
        )
        
        # Save report
        output_path = self.output_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
