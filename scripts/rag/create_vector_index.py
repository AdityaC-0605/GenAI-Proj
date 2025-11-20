#!/usr/bin/env python
"""
Script to create vector database index from QA datasets.

Supports SQuAD, XQuAD, MLQA, and TyDi QA datasets.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.config import load_rag_config
from src.rag.logging_config import setup_logging
from src.rag.vector_db import VectorDatabaseFactory
from src.rag.embeddings import EmbeddingManager
from src.rag.indexing import DocumentProcessor, VectorIndexBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Create vector database index from QA datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        choices=["squad", "xquad", "mlqa", "tydiqa"],
        help="Dataset type"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (for xquad, mlqa, tydiqa)"
    )
    parser.add_argument(
        "--question-language",
        type=str,
        default=None,
        help="Question language (for mlqa)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (train, dev, test)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for indexing"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Add to existing index instead of creating new"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_rag_config(args.config)
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("Vector Index Creation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Dataset type: {args.dataset_type}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Incremental: {args.incremental}")
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Vector database
    vector_db = VectorDatabaseFactory.create(config.vector_db)
    if args.incremental:
        logger.info("Loading existing index...")
        vector_db.load()
        logger.info(f"Existing index has {vector_db.count()} documents")
    
    # Embedding manager
    embedding_manager = EmbeddingManager(
        model_name=config.embedding.model_name,
        device=config.embedding.get('device', 'auto'),
        cache_size=config.embedding.get('cache_size', 10000)
    )
    
    # Document processor
    processor = DocumentProcessor()
    
    # Process dataset
    logger.info("Processing dataset...")
    
    if args.dataset_type == "squad":
        documents = processor.process_squad_dataset(
            args.dataset,
            split=args.split
        )
    elif args.dataset_type == "xquad":
        documents = processor.process_xquad_dataset(
            args.dataset,
            language=args.language,
            split=args.split
        )
    elif args.dataset_type == "mlqa":
        question_lang = args.question_language or args.language
        documents = processor.process_mlqa_dataset(
            args.dataset,
            context_lang=args.language,
            question_lang=question_lang,
            split=args.split
        )
    elif args.dataset_type == "tydiqa":
        documents = processor.process_tydiqa_dataset(
            args.dataset,
            language=args.language,
            split=args.split
        )
    else:
        logger.error(f"Unsupported dataset type: {args.dataset_type}")
        return 1
    
    # Build index
    logger.info("Building vector index...")
    builder = VectorIndexBuilder(vector_db, embedding_manager)
    
    if args.incremental:
        builder.add_documents_incremental(documents, batch_size=args.batch_size)
    else:
        builder.build_index(documents, batch_size=args.batch_size)
    
    # Print statistics
    logger.info("=" * 60)
    logger.info("Indexing Complete!")
    logger.info("=" * 60)
    logger.info(f"Total documents in index: {vector_db.count()}")
    
    cache_stats = embedding_manager.get_cache_stats()
    logger.info(f"Embedding cache hit rate: {cache_stats.get('hit_rate', 0.0):.2%}")
    if 'size' in cache_stats and 'max_size' in cache_stats:
        logger.info(f"Embedding cache size: {cache_stats['size']}/{cache_stats['max_size']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
