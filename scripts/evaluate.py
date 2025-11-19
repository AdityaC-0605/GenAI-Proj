#!/usr/bin/env python3
"""Evaluation script for Cross-Lingual QA models."""

import argparse
import logging
import sys
from pathlib import Path
import json

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress urllib3 warnings (harmless on macOS)
try:
    from src.utils.warning_suppressor import suppress_urllib3_warnings
    suppress_urllib3_warnings()
except ImportError:
    import warnings
    warnings.filterwarnings('ignore', message='.*urllib3.*OpenSSL.*', category=UserWarning)

from src.data.xquad_loader import XQuADLoader
from src.data.mlqa_loader import MLQALoader
from src.data.tydiqa_loader import TyDiQALoader
from src.data.squad_loader import SQuADLoader
from src.models.mbert_wrapper import MBERTModelWrapper
from src.models.mt5_wrapper import MT5ModelWrapper
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Cross-Lingual QA model"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['mbert', 'mt5'],
        required=True,
        help='Model type to evaluate'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to evaluation dataset'
    )
    
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['xquad', 'mlqa', 'tydiqa', 'squad'],
        required=True,
        help='Type of dataset'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/evaluations',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--language-pairs',
        type=str,
        nargs='+',
        default=None,
        help='Specific language pairs to evaluate (e.g., en-es en-zh)'
    )
    
    parser.add_argument(
        '--include-generative-metrics',
        action='store_true',
        help='Include BLEU/ROUGE metrics (for mT5)'
    )
    
    return parser.parse_args()


def load_data(data_path: str, dataset_type: str):
    """
    Load evaluation dataset.
    
    Args:
        data_path: Path to dataset
        dataset_type: 'xquad', 'mlqa', 'tydiqa', or 'squad'
        
    Returns:
        List of QA examples
    """
    logger.info(f"Loading {dataset_type} data from {data_path}")
    
    if dataset_type == 'xquad':
        loader = XQuADLoader()
    elif dataset_type == 'mlqa':
        loader = MLQALoader()
    elif dataset_type == 'tydiqa':
        loader = TyDiQALoader()
    elif dataset_type == 'squad':
        loader = SQuADLoader()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    examples = loader.load(data_path)
    
    logger.info(f"Loaded {len(examples)} examples")
    
    return examples


def filter_language_pairs(examples, language_pairs):
    """
    Filter examples by language pairs.
    
    Args:
        examples: List of QA examples
        language_pairs: List of language pair strings (e.g., ['en-es', 'en-zh'])
        
    Returns:
        Filtered list of examples
    """
    if language_pairs is None:
        return examples
    
    # Parse language pairs
    pairs = set()
    for pair_str in language_pairs:
        q_lang, c_lang = pair_str.split('-')
        pairs.add((q_lang, c_lang))
    
    # Filter examples
    filtered = [
        ex for ex in examples
        if (ex.question_language, ex.context_language) in pairs
    ]
    
    logger.info(f"Filtered to {len(filtered)} examples for specified language pairs")
    
    return filtered


def load_model(model_type: str, checkpoint_path: str):
    """
    Load model from checkpoint.
    
    Args:
        model_type: 'mbert' or 'mt5'
        checkpoint_path: Path to checkpoint
        
    Returns:
        Model wrapper instance
    """
    logger.info(f"Loading {model_type} model from {checkpoint_path}")
    
    if model_type == 'mbert':
        model = MBERTModelWrapper()
    elif model_type == 'mt5':
        model = MT5ModelWrapper()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Model loaded successfully")
    
    return model


def main():
    """Main evaluation function."""
    args = parse_args()
    
    logger.info(f"Starting evaluation of {args.model} model")
    
    # Load data
    examples = load_data(args.data_path, args.dataset_type)
    
    # Filter by language pairs if specified
    if args.language_pairs:
        examples = filter_language_pairs(examples, args.language_pairs)
    
    # Load model
    model = load_model(args.model, args.checkpoint)
    
    # Create evaluator
    evaluator = Evaluator(
        metrics_calculator=MetricsCalculator(),
        output_dir=args.output_dir
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    
    results = evaluator.evaluate(
        model=model,
        examples=examples,
        model_name=args.model,
        dataset_name=args.dataset_type,
        include_generative_metrics=args.include_generative_metrics,
        save_results=True
    )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    for lang_pair, result in sorted(results.items()):
        logger.info(
            f"{lang_pair[0]}-{lang_pair[1]}: "
            f"EM={result.exact_match:.4f}, F1={result.f1_score:.4f} "
            f"({result.num_examples} examples)"
        )
    
    # Calculate aggregate
    total_examples = sum(r.num_examples for r in results.values())
    weighted_em = sum(
        r.exact_match * r.num_examples for r in results.values()
    ) / total_examples
    weighted_f1 = sum(
        r.f1_score * r.num_examples for r in results.values()
    ) / total_examples
    
    logger.info("="*50)
    logger.info(f"AGGREGATE: EM={weighted_em:.4f}, F1={weighted_f1:.4f}")
    logger.info("="*50)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
