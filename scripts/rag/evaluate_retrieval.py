#!/usr/bin/env python
"""
Script to evaluate retrieval quality on QA datasets.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Set
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.config import load_rag_config
from src.rag.logging_config import setup_logging
from src.rag.vector_db import VectorDatabaseFactory
from src.rag.embeddings import EmbeddingManager
from src.rag.retrieval import SemanticRetriever
from src.rag.evaluation import RAGEvaluator
from src.rag.indexing import DocumentProcessor


def load_ground_truth(dataset_path: str, dataset_type: str) -> List[Dict]:
    """Load ground truth from dataset."""
    processor = DocumentProcessor()
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ground_truth = []
    
    for article in data.get('data', []):
        title = article.get('title', 'Unknown')
        
        for paragraph in article.get('paragraphs', []):
            context = paragraph['context']
            context_id = processor._generate_id(title, context)
            
            for qa in paragraph.get('qas', []):
                question = qa['question']
                
                ground_truth.append({
                    'question': question,
                    'relevant_context_ids': {context_id}
                })
    
    return ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        choices=["squad", "xquad", "mlqa", "tydiqa"],
        help="Dataset type"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="retrieval_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_rag_config(args.config)
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("Retrieval Evaluation")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing components...")
    vector_db = VectorDatabaseFactory.create(config.vector_db)
    vector_db.load()
    
    embedding_manager = EmbeddingManager(
        model_name=config.embedding.model_name,
        device=config.embedding.get('device', 'auto')
    )
    
    retriever = SemanticRetriever(
        vector_db=vector_db,
        embedding_manager=embedding_manager,
        top_k=args.top_k
    )
    
    evaluator = RAGEvaluator()
    
    # Load ground truth
    logger.info(f"Loading ground truth from {args.dataset}")
    ground_truth = load_ground_truth(args.dataset, args.dataset_type)
    logger.info(f"Loaded {len(ground_truth)} questions")
    
    # Evaluate
    logger.info("Evaluating retrieval...")
    all_results = []
    
    for item in tqdm(ground_truth, desc="Evaluating"):
        question = item['question']
        relevant_ids = item['relevant_context_ids']
        
        # Retrieve
        results = retriever.retrieve(question, top_k=args.top_k)
        retrieved_ids = [r.id for r in results]
        
        # Evaluate
        metrics = evaluator.evaluate_retrieval(
            retrieved_ids,
            relevant_ids,
            k_values=[1, 3, 5, 10]
        )
        
        all_results.append({
            'question': question,
            'metrics': metrics
        })
    
    # Aggregate results
    logger.info("Aggregating results...")
    avg_metrics = {}
    metric_names = all_results[0]['metrics'].keys()
    
    for metric_name in metric_names:
        values = [r['metrics'][metric_name] for r in all_results]
        avg_metrics[metric_name] = sum(values) / len(values)
    
    # Print results
    logger.info("=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    
    for metric_name, value in sorted(avg_metrics.items()):
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Save results
    output_data = {
        'config': args.config,
        'dataset': args.dataset,
        'top_k': args.top_k,
        'num_questions': len(ground_truth),
        'average_metrics': avg_metrics,
        'per_question_results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
