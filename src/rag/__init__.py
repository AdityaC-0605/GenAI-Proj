"""
RAG (Retrieval-Augmented Generation) module for Cross-Lingual Question Answering.

This module provides a complete RAG implementation including:
- Vector database abstraction layer
- Embedding model management
- Semantic and hybrid retrieval
- Answer generation with multiple backends
- End-to-end RAG pipeline
"""

__version__ = "0.1.0"

# Import main components
from .vector_db import (
    VectorDatabase,
    ChromaDBBackend,
    FAISSBackend,
    VectorDatabaseFactory,
    Document,
    SearchResult
)
from .embeddings import EmbeddingManager
from .config import RAGConfig, load_rag_config
from .logging_config import setup_logging, get_logger, LoggerMixin
from .retrieval import (
    SemanticRetriever,
    HybridRetriever,
    ContextReranker,
    RetrievalResult
)
from .generation import (
    AnswerGenerator,
    MT5Generator,
    OpenAIGenerator,
    OllamaGenerator,
    GeneratorFactory,
    GenerationResult
)
from .pipeline import RAGPipeline, BatchRAGPipeline, RAGResponse
from .indexing import DocumentProcessor, VectorIndexBuilder, IndexDocument

__all__ = [
    "__version__",
    # Vector Database
    "VectorDatabase",
    "ChromaDBBackend",
    "FAISSBackend",
    "VectorDatabaseFactory",
    "Document",
    "SearchResult",
    # Embeddings
    "EmbeddingManager",
    # Configuration
    "RAGConfig",
    "load_rag_config",
    # Logging
    "setup_logging",
    "get_logger",
    "LoggerMixin",
    # Retrieval
    "SemanticRetriever",
    "HybridRetriever",
    "ContextReranker",
    "RetrievalResult",
    # Generation
    "AnswerGenerator",
    "MT5Generator",
    "OpenAIGenerator",
    "OllamaGenerator",
    "GeneratorFactory",
    "GenerationResult",
    # Pipeline
    "RAGPipeline",
    "BatchRAGPipeline",
    "RAGResponse",
    # Indexing
    "DocumentProcessor",
    "VectorIndexBuilder",
    "IndexDocument",
]
