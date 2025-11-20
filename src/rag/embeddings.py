"""
Embedding model manager for RAG system.

Handles loading and inference of multilingual embedding models using
sentence-transformers library.
"""

from typing import List, Union, Optional
import numpy as np
import torch
from functools import lru_cache

from .logging_config import LoggerMixin


class EmbeddingManager(LoggerMixin):
    """Manages embedding model loading and inference."""
    
    # Supported embedding models with their dimensions
    SUPPORTED_MODELS = {
        'paraphrase-multilingual-MiniLM-L12-v2': 384,
        'paraphrase-multilingual-mpnet-base-v2': 768,
        'multilingual-e5-large': 1024,
        'bge-m3': 1024,
    }
    
    def __init__(self, 
                 model_name: str = 'paraphrase-multilingual-mpnet-base-v2',
                 device: str = 'auto',
                 normalize_embeddings: bool = True,
                 cache_size: int = 10000):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            normalize_embeddings: Whether to normalize embeddings for cosine similarity
            cache_size: Size of embedding cache (0 to disable)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.cache_size = cache_size
        
        # Determine device
        self.device = self._get_device(device)
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.logger.info(
            f"Embedding model loaded: {model_name} "
            f"(dimension: {self.dimension}, device: {self.device})"
        )
        
        # Initialize cache
        self._embedding_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _get_device(self, device: str) -> str:
        """
        Determine optimal device for inference.
        
        Args:
            device: Device preference ('auto', 'cpu', 'cuda', 'mps')
            
        Returns:
            Device string
        """
        if device == 'auto':
            # Check for MPS (Apple Silicon)
            if torch.backends.mps.is_available():
                return 'mps'
            # Check for CUDA
            elif torch.cuda.is_available():
                return 'cuda'
            # Fallback to CPU
            return 'cpu'
        
        return device
    
    def embed_texts(self,
                   texts: Union[str, List[str]],
                   batch_size: int = 32,
                   show_progress: bool = False,
                   use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            use_cache: Whether to use embedding cache
            
        Returns:
            Embeddings as numpy array (N x D)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Check cache
        if use_cache and self.cache_size > 0:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self._embedding_cache:
                    cached_embeddings.append((i, self._embedding_cache[text]))
                    self._cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self._cache_misses += 1
            
            # If all cached, return immediately
            if not uncached_texts:
                embeddings = np.array([emb for _, emb in sorted(cached_embeddings)])
                return embeddings[0] if single_text else embeddings
            
            # Encode uncached texts
            new_embeddings = self._encode_texts(
                uncached_texts,
                batch_size=batch_size,
                show_progress=show_progress
            )
            
            # Update cache
            for text, embedding in zip(uncached_texts, new_embeddings):
                if len(self._embedding_cache) < self.cache_size:
                    self._embedding_cache[text] = embedding
            
            # Combine cached and new embeddings
            all_embeddings = cached_embeddings + list(zip(uncached_indices, new_embeddings))
            all_embeddings.sort(key=lambda x: x[0])
            embeddings = np.array([emb for _, emb in all_embeddings])
        else:
            # No caching
            embeddings = self._encode_texts(
                texts,
                batch_size=batch_size,
                show_progress=show_progress
            )
        
        return embeddings[0] if single_text else embeddings
    
    def _encode_texts(self,
                     texts: List[str],
                     batch_size: int = 32,
                     show_progress: bool = False) -> np.ndarray:
        """
        Encode texts using the model.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Embeddings as numpy array
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self._embedding_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.info("Embedding cache cleared")
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """Get list of supported embedding models."""
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_model_dimension(cls, model_name: str) -> Optional[int]:
        """Get dimension for a supported model."""
        return cls.SUPPORTED_MODELS.get(model_name)
