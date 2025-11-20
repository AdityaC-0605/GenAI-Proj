"""
Retrieval modules for RAG system.

Implements semantic retrieval, hybrid search, and re-ranking capabilities.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from .vector_db import VectorDatabase, SearchResult
from .embeddings import EmbeddingManager
from .logging_config import LoggerMixin


@dataclass
class RetrievalResult:
    """Single retrieval result with ranking information."""
    text: str
    score: float
    metadata: Dict[str, Any]
    rank: int
    id: str


class SemanticRetriever(LoggerMixin):
    """Semantic search using vector embeddings."""
    
    def __init__(self,
                 vector_db: VectorDatabase,
                 embedding_manager: EmbeddingManager,
                 top_k: int = 5,
                 similarity_threshold: float = 0.0):
        """
        Initialize semantic retriever.
        
        Args:
            vector_db: Vector database instance
            embedding_manager: Embedding manager instance
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
        """
        super().__init__()
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        self.logger.info(
            f"SemanticRetriever initialized (top_k={top_k}, "
            f"threshold={similarity_threshold})"
        )
    
    def retrieve(self,
                question: str,
                top_k: Optional[int] = None,
                language: Optional[str] = None,
                filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant contexts for a question.
        
        Args:
            question: Question text
            top_k: Number of results (overrides default)
            language: Filter by language
            filters: Additional metadata filters
            
        Returns:
            List of RetrievalResult objects
        """
        k = top_k if top_k is not None else self.top_k
        
        self.logger.debug(f"Retrieving contexts for question: {question[:50]}...")
        
        # Embed question
        question_embedding = self.embedding_manager.embed_texts(question)
        
        # Apply language filter
        if language:
            if filters is None:
                filters = {}
            filters['language'] = language
        
        # Search vector database
        search_results = self.vector_db.search(
            query_embedding=question_embedding,
            top_k=k,
            filters=filters
        )
        
        # Convert to RetrievalResult and filter by threshold
        retrieval_results = []
        for i, result in enumerate(search_results):
            if result.score >= self.similarity_threshold:
                retrieval_results.append(RetrievalResult(
                    text=result.document,
                    score=result.score,
                    metadata=result.metadata,
                    rank=i + 1,
                    id=result.id
                ))
        
        self.logger.debug(f"Retrieved {len(retrieval_results)} contexts")
        return retrieval_results
    
    def retrieve_batch(self,
                      questions: List[str],
                      top_k: Optional[int] = None,
                      **kwargs) -> List[List[RetrievalResult]]:
        """
        Retrieve contexts for multiple questions.
        
        Args:
            questions: List of questions
            top_k: Number of results per question
            **kwargs: Additional arguments for retrieve()
            
        Returns:
            List of retrieval result lists
        """
        results = []
        for question in questions:
            result = self.retrieve(question, top_k=top_k, **kwargs)
            results.append(result)
        
        return results



class HybridRetriever(LoggerMixin):
    """Hybrid retrieval combining semantic and keyword search."""
    
    def __init__(self,
                 semantic_retriever: SemanticRetriever,
                 alpha: float = 0.7,
                 build_bm25: bool = True):
        """
        Initialize hybrid retriever.
        
        Args:
            semantic_retriever: SemanticRetriever instance
            alpha: Weight for semantic search (0-1), 1-alpha for keyword search
            build_bm25: Whether to build BM25 index on initialization
        """
        super().__init__()
        
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for hybrid search. "
                "Install with: pip install rank-bm25"
            )
        
        self.semantic_retriever = semantic_retriever
        self.alpha = alpha
        self.BM25Okapi = BM25Okapi
        
        self.bm25_index = None
        self.documents = []
        self.document_ids = []
        self.document_metadata = []
        
        self.logger.info(f"HybridRetriever initialized (alpha={alpha})")
        
        if build_bm25:
            self._build_bm25_from_vector_db()
    
    def _build_bm25_from_vector_db(self):
        """Build BM25 index from documents in vector database."""
        self.logger.info("Building BM25 index from vector database...")
        
        # This is a simplified approach - in production, you'd want to
        # store documents separately or retrieve them from the vector DB
        # For now, we'll build it when documents are explicitly provided
        self.logger.warning(
            "BM25 index not built automatically. "
            "Call build_bm25_index() with documents."
        )
    
    def build_bm25_index(self,
                        documents: List[str],
                        document_ids: List[str],
                        metadata: List[Dict[str, Any]]):
        """
        Build BM25 index for keyword search.
        
        Args:
            documents: List of document texts
            document_ids: List of document IDs
            metadata: List of metadata dictionaries
        """
        self.logger.info(f"Building BM25 index with {len(documents)} documents...")
        
        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Build BM25 index
        self.bm25_index = self.BM25Okapi(tokenized_docs)
        self.documents = documents
        self.document_ids = document_ids
        self.document_metadata = metadata
        
        self.logger.info("BM25 index built successfully")
    
    def retrieve(self,
                question: str,
                top_k: int = 5,
                **kwargs) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining semantic and keyword search.
        
        Args:
            question: Question text
            top_k: Number of results to return
            **kwargs: Additional arguments for semantic retrieval
            
        Returns:
            List of RetrievalResult objects
        """
        if self.bm25_index is None:
            self.logger.warning("BM25 index not built, falling back to semantic search only")
            return self.semantic_retriever.retrieve(question, top_k=top_k, **kwargs)
        
        self.logger.debug(f"Hybrid retrieval for: {question[:50]}...")
        
        # Semantic search (get more results for merging)
        semantic_results = self.semantic_retriever.retrieve(
            question,
            top_k=top_k * 2,
            **kwargs
        )
        
        # Keyword search (BM25)
        tokenized_query = question.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        normalized_bm25 = bm25_scores / max_bm25
        
        # Combine scores
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result.id
            combined_results[doc_id] = {
                'text': result.text,
                'metadata': result.metadata,
                'semantic_score': result.score,
                'bm25_score': 0.0,
                'id': doc_id
            }
        
        # Add BM25 scores
        for i, (doc_id, doc_text) in enumerate(zip(self.document_ids, self.documents)):
            if doc_id in combined_results:
                combined_results[doc_id]['bm25_score'] = float(normalized_bm25[i])
            else:
                # Document found by BM25 but not semantic search
                combined_results[doc_id] = {
                    'text': doc_text,
                    'metadata': self.document_metadata[i],
                    'semantic_score': 0.0,
                    'bm25_score': float(normalized_bm25[i]),
                    'id': doc_id
                }
        
        # Calculate combined scores
        for doc_id in combined_results:
            semantic_score = combined_results[doc_id]['semantic_score']
            bm25_score = combined_results[doc_id]['bm25_score']
            combined_results[doc_id]['combined_score'] = (
                self.alpha * semantic_score + (1 - self.alpha) * bm25_score
            )
        
        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]
        
        # Convert to RetrievalResult
        retrieval_results = []
        for i, result in enumerate(sorted_results):
            retrieval_results.append(RetrievalResult(
                text=result['text'],
                score=result['combined_score'],
                metadata=result['metadata'],
                rank=i + 1,
                id=result['id']
            ))
        
        self.logger.debug(f"Hybrid retrieval returned {len(retrieval_results)} results")
        return retrieval_results



class ContextReranker(LoggerMixin):
    """Re-rank contexts using cross-encoder model."""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize context re-ranker.
        
        Args:
            model_name: Cross-encoder model name
        """
        super().__init__()
        
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for re-ranking. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.logger.info("Cross-encoder model loaded successfully")
    
    def rerank(self,
              question: str,
              results: List[RetrievalResult],
              top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Re-rank retrieval results using cross-encoder.
        
        Args:
            question: Question text
            results: List of RetrievalResult objects
            top_k: Number of results to return (None = all)
            
        Returns:
            Re-ranked list of RetrievalResult objects
        """
        if not results:
            return results
        
        self.logger.debug(f"Re-ranking {len(results)} results")
        
        # Create question-context pairs
        pairs = [(question, r.text) for r in results]
        
        # Score pairs with cross-encoder
        scores = self.model.predict(pairs)
        
        # Update scores in results
        reranked_results = []
        for result, score in zip(results, scores):
            reranked_results.append(RetrievalResult(
                text=result.text,
                score=float(score),
                metadata=result.metadata,
                rank=0,  # Will be updated after sorting
                id=result.id
            ))
        
        # Sort by new scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result.rank = i + 1
        
        # Return top-k if specified
        if top_k is not None:
            reranked_results = reranked_results[:top_k]
        
        self.logger.debug(f"Re-ranking complete, returning {len(reranked_results)} results")
        return reranked_results
