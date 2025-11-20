#!/usr/bin/env python
"""
Script to evaluate end-to-end RAG system on QA datasets.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.config import load_rag_config
from src.rag.logging_config import setup_logging
from src.rag.vector_db import VectorDatabaseFactory
from src.rag.embeddings import EmbeddingManager
from src.rag.retrieval import SemanticRetriever
from src.rag.generation import GeneratorFactory
from src.rag.pipeline import RAGPipeline
from src.rag.evaluation import RAGEvaluator
from src.rag.indexing import DocumentProcessor


def load_evaluation_data(dataset_path: str) -> List[Dict]:
    """Load evaluation data with questions and answers."""
    processor = DocumentProcessor()
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    eval_data = []
    
    for article in data.get('data', []):
        title = article.get('title', 'Unknown')
        
        for paragraph in article.get('paragraphs', []):
            context = paragraph['context']
            context_id = processor._generate_id(title, context)
            
            for qa in paragraph.get('qas', []):
                question = qa['question']
                answers = qa.get('answers', [])
                
                if answers:
                    ground_truth = answers[0]['text']
                    
                    eval_data.append({
                        'question': question,
                        'ground_truth': ground_truth,
                        'relevant_context_ids': {context_id}
                    })
    
    return eval_data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate end-to-end RAG system"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rag_evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_rag_config(args.config)
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("End-to-End RAG Evaluation")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing RAG pipeline...")
    vector_db = VectorDatabaseFactory.create(config.vector_db)
    vector_db.load()
    
    embedding_manager = EmbeddingManager(
        model_name=config.embedding.model_name,
        device=config.embedding.get('device', 'auto')
    )
    
    retriever = SemanticRetriever(
        vector_db=vector_db,
        embedding_manager=embedding_manager,
        top_k=config.retrieval.top_k
    )
    
    generator = GeneratorFactory.create(config.generator)
    
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator
    )
    
    evaluator = RAGEvaluator()
    
    # Load evaluation data
    logger.info(f"Loading evaluation data from {args.dataset}")
    eval_data = load_evaluation_data(args.dataset)
    
    if args.limit:
        eval_data = eval_data[:args.limit]
    
    logger.info(f"Evaluating {len(eval_data)} questions")
    
    # Evaluate
    logger.info("Running evaluation...")
    all_results = []
    
    start_time = time.time()
    
    for item in tqdm(eval_data, desc="Evaluating"):
        question = item['question']
        ground_truth = item['ground_truth']
        relevant_ids = item['relevant_context_ids']
        
        # Run RAG pipeline
        response = pipeline.answer_question(question)
        
        # Get retrieved IDs
        retrieved_ids = [ctx.id for ctx in response.contexts]
        
        # Evaluate
        metrics = evaluator.evaluate_end_to_end(
            prediction=response.answer,
            ground_truth=ground_truth,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            retrieval_time=response.retrieval_time,
            generation_time=response.generation_time
        )
        
        all_results.append({
            'question': question,
            'prediction': response.answer,
            'ground_truth': ground_truth,
            'confidence': response.confidence,
            'metrics': metrics
        })
    
    total_time = time.time() - start_time
    
    # Aggregate results
    logger.info("Aggregating results...")
    
    # Average retrieval metrics
    avg_retrieval = {}
    for metric_name in all_results[0]['metrics']['retrieval'].keys():
        values = [r['metrics']['retrieval'][metric_name] for r in all_results]
        avg_retrieval[metric_name] = sum(values) / len(values)
    
    # Average generation metrics
    avg_generation = {}
    for metric_name in all_results[0]['metrics']['generation'].keys():
        values = [r['metrics']['generation'][metric_name] for r in all_results]
        avg_generation[metric_name] = sum(values) / len(values)
    
    # Average timing
    avg_timing = {}
    for metric_name in all_results[0]['metrics']['timing'].keys():
        values = [r['metrics']['timing'][metric_name] for r in all_results]
        avg_timing[metric_name] = sum(values) / len(values)
    
    # Context relevance
    context_relevance_values = [r['metrics']['context_relevance'] for r in all_results]
    avg_context_relevance = sum(context_relevance_values) / len(context_relevance_values)
    
    # Print results
    logger.info("=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    
    logger.info("\nRetrieval Metrics:")
    for metric_name, value in sorted(avg_retrieval.items()):
        logger.info(f"  {metric_name}: {value:.4f}")
    
    logger.info("\nGeneration Metrics:")
    for metric_name, value in sorted(avg_generation.items()):
        logger.info(f"  {metric_name}: {value:.4f}")
    
    logger.info("\nTiming:")
    for metric_name, value in sorted(avg_timing.items()):
        logger.info(f"  {metric_name}: {value:.4f}s")
    
    logger.info(f"\nContext Relevance: {avg_context_relevance:.4f}")
    logger.info(f"Total Evaluation Time: {total_time:.2f}s")
    logger.info(f"Questions per Second: {len(eval_data) / total_time:.2f}")
    
    # Save results
    output_data = {
        'config': args.config,
        'dataset': args.dataset,
        'num_questions': len(eval_data),
        'total_time': total_time,
        'average_metrics': {
            'retrieval': avg_retrieval,
            'generation': avg_generation,
            'timing': avg_timing,
            'context_relevance': avg_context_relevance
        },
        'per_question_results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
