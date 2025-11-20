#!/usr/bin/env python
"""
Compare trained mBERT vs RAG with mT5 for Cross-Lingual QA.

This script compares:
1. Your trained mBERT model (extractive QA)
2. RAG system with pre-trained mT5 (no training needed)

For cross-lingual question answering across multiple language pairs.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mbert import MBERTForQA
from src.data.data_loader import load_squad_data, load_xquad_data
from src.evaluation.metrics import calculate_metrics

# RAG imports
from src.rag.config import load_rag_config
from src.rag.logging_config import setup_logging
from src.rag.vector_db import VectorDatabaseFactory
from src.rag.embeddings import EmbeddingManager
from src.rag.retrieval import SemanticRetriever
from src.rag.generation import MT5Generator
from src.rag.pipeline import RAGPipeline
from src.rag.evaluation import RAGEvaluator


def load_mbert_model(checkpoint_path: str, device: str = 'auto'):
    """Load trained mBERT model."""
    print(f"Loading mBERT model from {checkpoint_path}")
    
    model = MBERTForQA()
    
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ mBERT model loaded on {device}")
    return model, device


def setup_rag_system(config_name: str = 'mt5'):
    """Set up RAG system with mT5."""
    print("Setting up RAG system with mT5...")
    
    # Load configuration
    config = load_rag_config(config_name)
    logger = setup_logging(config)
    
    # Initialize vector database
    vector_db = VectorDatabaseFactory.create(config.vector_db)
    try:
        vector_db.load()
        print(f"✓ Loaded vector database with {vector_db.count()} documents")
    except:
        print("⚠️  Vector database not found. Please index data first:")
        print("  python scripts/rag/create_vector_index.py --dataset data/squad/train-v2.0.json --dataset-type squad")
        sys.exit(1)
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager(
        model_name=config.embedding.model_name,
        device='auto'
    )
    print(f"✓ Loaded embedding model: {config.embedding.model_name}")
    
    # Initialize retriever
    retriever = SemanticRetriever(
        vector_db=vector_db,
        embedding_manager=embedding_manager,
        top_k=5
    )
    
    # Initialize mT5 generator (no training needed!)
    generator = MT5Generator(
        model_name='google/mt5-base',
        device='auto'
    )
    print("✓ Loaded pre-trained mT5 (no training required)")
    
    # Create RAG pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator
    )
    
    print("✓ RAG system ready")
    return pipeline


def evaluate_mbert(model, device, test_data, language_pair):
    """Evaluate mBERT model."""
    print(f"\nEvaluating mBERT on {language_pair}...")
    
    results = []
    total_time = 0
    
    for item in tqdm(test_data, desc="mBERT"):
        start_time = time.time()
        
        # Get prediction from mBERT
        with torch.no_grad():
            prediction = model.predict(
                question=item['question'],
                context=item['context']
            )
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Calculate metrics
        metrics = calculate_metrics(
            prediction=prediction,
            ground_truth=item['answer']
        )
        
        results.append({
            'question': item['question'],
            'prediction': prediction,
            'ground_truth': item['answer'],
            'metrics': metrics,
            'time': elapsed
        })
    
    # Aggregate metrics
    avg_metrics = {
        'exact_match': sum(r['metrics']['exact_match'] for r in results) / len(results),
        'f1': sum(r['metrics']['f1'] for r in results) / len(results),
        'avg_time': total_time / len(results),
        'total_time': total_time
    }
    
    return results, avg_metrics


def evaluate_rag_mt5(pipeline, test_data, language_pair):
    """Evaluate RAG with mT5."""
    print(f"\nEvaluating RAG+mT5 on {language_pair}...")
    
    evaluator = RAGEvaluator()
    results = []
    total_time = 0
    
    for item in tqdm(test_data, desc="RAG+mT5"):
        start_time = time.time()
        
        # Get prediction from RAG
        response = pipeline.answer_question(
            question=item['question'],
            top_k=5
        )
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Calculate metrics
        metrics = evaluator.evaluate_generation(
            prediction=response.answer,
            ground_truth=item['answer']
        )
        
        results.append({
            'question': item['question'],
            'prediction': response.answer,
            'ground_truth': item['answer'],
            'confidence': response.confidence,
            'contexts': [{'text': c.text, 'score': c.score} for c in response.contexts[:3]],
            'metrics': metrics,
            'time': elapsed
        })
    
    # Aggregate metrics
    avg_metrics = {
        'exact_match': sum(r['metrics']['exact_match'] for r in results) / len(results),
        'f1': sum(r['metrics']['f1'] for r in results) / len(results),
        'avg_time': total_time / len(results),
        'total_time': total_time
    }
    
    return results, avg_metrics


def compare_results(mbert_metrics, rag_metrics, language_pair):
    """Compare and display results."""
    print("\n" + "=" * 80)
    print(f"COMPARISON RESULTS: {language_pair}")
    print("=" * 80)
    
    print(f"\n{'Metric':<20} {'mBERT (Trained)':<20} {'RAG+mT5 (No Training)':<25} {'Winner':<15}")
    print("-" * 80)
    
    # Exact Match
    em_diff = rag_metrics['exact_match'] - mbert_metrics['exact_match']
    em_winner = 'RAG+mT5' if em_diff > 0 else 'mBERT' if em_diff < 0 else 'Tie'
    print(f"{'Exact Match':<20} {mbert_metrics['exact_match']:<20.4f} {rag_metrics['exact_match']:<25.4f} {em_winner:<15}")
    
    # F1 Score
    f1_diff = rag_metrics['f1'] - mbert_metrics['f1']
    f1_winner = 'RAG+mT5' if f1_diff > 0 else 'mBERT' if f1_diff < 0 else 'Tie'
    print(f"{'F1 Score':<20} {mbert_metrics['f1']:<20.4f} {rag_metrics['f1']:<25.4f} {f1_winner:<15}")
    
    # Latency
    latency_winner = 'mBERT' if mbert_metrics['avg_time'] < rag_metrics['avg_time'] else 'RAG+mT5'
    print(f"{'Avg Latency (s)':<20} {mbert_metrics['avg_time']:<20.4f} {rag_metrics['avg_time']:<25.4f} {latency_winner:<15}")
    
    print("\n" + "=" * 80)
    
    # Summary
    print("\nSummary:")
    print(f"  • mBERT: Trained model, extractive QA")
    print(f"  • RAG+mT5: No training needed, generative QA with retrieval")
    print(f"  • Language Pair: {language_pair}")
    
    return {
        'language_pair': language_pair,
        'mbert': mbert_metrics,
        'rag_mt5': rag_metrics,
        'winner': {
            'exact_match': em_winner,
            'f1': f1_winner,
            'latency': latency_winner
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare trained mBERT vs RAG+mT5 for cross-lingual QA"
    )
    parser.add_argument(
        "--mbert-checkpoint",
        type=str,
        default="models/mbert_retrained/best_model.pt",
        help="Path to trained mBERT checkpoint"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data (SQuAD or XQuAD format)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=['squad', 'xquad', 'mlqa'],
        default='squad',
        help="Dataset type"
    )
    parser.add_argument(
        "--language-pair",
        type=str,
        default="en-en",
        help="Language pair (e.g., en-en, en-es, es-en)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("mBERT vs RAG+mT5 Comparison for Cross-Lingual QA")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • mBERT checkpoint: {args.mbert_checkpoint}")
    print(f"  • Test data: {args.test_data}")
    print(f"  • Language pair: {args.language_pair}")
    print(f"  • Dataset type: {args.dataset_type}")
    
    # Load test data
    print(f"\nLoading test data...")
    if args.dataset_type == 'squad':
        test_data = load_squad_data(args.test_data)
    elif args.dataset_type == 'xquad':
        test_data = load_xquad_data(args.test_data)
    else:
        print(f"Dataset type {args.dataset_type} not yet supported")
        return 1
    
    if args.limit:
        test_data = test_data[:args.limit]
    
    print(f"✓ Loaded {len(test_data)} examples")
    
    # Load mBERT model
    mbert_model, device = load_mbert_model(args.mbert_checkpoint)
    
    # Setup RAG system
    rag_pipeline = setup_rag_system('mt5')
    
    # Evaluate mBERT
    mbert_results, mbert_metrics = evaluate_mbert(
        mbert_model, device, test_data, args.language_pair
    )
    
    # Evaluate RAG+mT5
    rag_results, rag_metrics = evaluate_rag_mt5(
        rag_pipeline, test_data, args.language_pair
    )
    
    # Compare results
    comparison = compare_results(mbert_metrics, rag_metrics, args.language_pair)
    
    # Save results
    output_data = {
        'configuration': {
            'mbert_checkpoint': args.mbert_checkpoint,
            'test_data': args.test_data,
            'language_pair': args.language_pair,
            'num_examples': len(test_data)
        },
        'comparison': comparison,
        'detailed_results': {
            'mbert': mbert_results,
            'rag_mt5': rag_results
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
