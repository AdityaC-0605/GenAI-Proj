#!/usr/bin/env python3
"""Fast evaluation script with batching and progress tracking."""

import argparse
import logging
import sys
from pathlib import Path
import json
from tqdm import tqdm

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.squad_loader import SQuADLoader
from src.models.mbert_wrapper import MBERTModelWrapper
from src.models.mt5_wrapper import MT5ModelWrapper
from src.evaluation.metrics import MetricsCalculator
from src.data_models import QAExample

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_with_batching(
    model,
    examples: list,
    batch_size: int = 16,
    max_examples: int = None
):
    """Evaluate model with batching and progress tracking."""
    metrics_calc = MetricsCalculator()
    
    if max_examples:
        examples = examples[:max_examples]
        logger.info(f"Evaluating on {len(examples)} examples (limited from {len(examples)})")
    
    predictions = []
    exact_matches = []
    f1_scores = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
        batch_examples = examples[i:i+batch_size]
        
        for example in batch_examples:
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
                em = metrics_calc.exact_match(
                    prediction.answer_text,
                    answer.text
                )
                f1 = metrics_calc.f1_score(
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
    
    return {
        'exact_match': avg_exact_match,
        'f1_score': avg_f1_score,
        'num_examples': len(examples),
        'predictions': predictions
    }


def main():
    parser = argparse.ArgumentParser(description="Fast evaluation with progress tracking")
    parser.add_argument('--model', type=str, choices=['mbert', 'mt5'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--dataset-type', type=str, default='squad')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-examples', type=int, default=None, 
                       help='Limit number of examples (for quick testing)')
    parser.add_argument('--output-dir', type=str, default='experiments/evaluations')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading {args.dataset_type} data from {args.data_path}")
    loader = SQuADLoader()
    examples = loader.load(args.data_path)
    logger.info(f"Loaded {len(examples)} examples")
    
    # Load model
    logger.info(f"Loading {args.model} model from {args.checkpoint}")
    if args.model == 'mbert':
        model = MBERTModelWrapper()
    else:
        model = MT5ModelWrapper()
    
    model.load(args.checkpoint)
    logger.info("Model loaded successfully")
    
    # Evaluate
    logger.info("Starting evaluation...")
    results = evaluate_with_batching(
        model,
        examples,
        batch_size=args.batch_size,
        max_examples=args.max_examples
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Exact Match: {results['exact_match']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Number of Examples: {results['num_examples']}")
    print("="*50)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.model}_{args.dataset_type}_fast_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {filepath}")


if __name__ == "__main__":
    main()

