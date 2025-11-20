"""
Vector database abstraction layer for RAG system.

Provides a unified interface for different vector database backends including
ChromaDB, FAISS, Pinecone, and Qdrant.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pathlib import Path

from .logging_config import LoggerMixin


@dataclass
class Document:
    """Document for indexing in vector database."""
    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Result from vector database search."""
    document: str
    score: float
    metadata: Dict[str, Any]
    distance: float
    id: str


class VectorDatabase(ABC, LoggerMixin):
    """Abstract base class for vector databases."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vector database.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self.collection_name = config.get('collection_name', 'default')
        self.distance_metric = config.get('distance_metric', 'cosine')
        
    @abstractmethod
    def add_documents(self, 
                     documents: List[str],
                     embeddings: np.ndarray,
                     metadata: List[Dict[str, Any]],
                     ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents with embeddings and metadata to the database.
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings (N x D array)
            metadata: List of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def search(self,
              query_embedding: np.ndarray,
              top_k: int = 5,
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents by ID.
        
        Args:
            document_ids: List of document IDs to delete
        """
        pass
    
    @abstractmethod
    def save(self, path: Optional[str] = None) -> None:
        """
        Persist index to disk.
        
        Args:
            path: Optional path to save index
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load index from disk.
        
        Args:
            path: Path to load index from
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        Get number of documents in the database.
        
        Returns:
            Number of documents
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config



class ChromaDBBackend(VectorDatabase):
    """ChromaDB implementation of vector database."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ChromaDB backend.
        
        Args:
            config: ChromaDB configuration
        """
        super().__init__(config)
        
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        
        self.persist_directory = config.get('persist_directory', 'data/vector_index/chromadb')
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        distance_map = {
            'cosine': 'cosine',
            'l2': 'l2',
            'ip': 'ip'
        }
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": distance_map.get(self.distance_metric, 'cosine')}
        )
        
        self.logger.info(f"ChromaDB initialized with collection '{self.collection_name}'")
    
    def add_documents(self,
                     documents: List[str],
                     embeddings: np.ndarray,
                     metadata: List[Dict[str, Any]],
                     ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to ChromaDB."""
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Convert numpy array to list for ChromaDB
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        # ChromaDB requires all metadata values to be strings, ints, floats, or bools
        # Convert lists to strings
        clean_metadata = []
        for meta in metadata:
            clean_meta = {}
            for key, value in meta.items():
                if isinstance(value, (list, tuple)):
                    clean_meta[key] = str(value)
                elif isinstance(value, (str, int, float, bool)):
                    clean_meta[key] = value
                else:
                    clean_meta[key] = str(value)
            clean_metadata.append(clean_meta)
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings_list,
            metadatas=clean_metadata,
            ids=ids
        )
        
        self.logger.debug(f"Added {len(documents)} documents to ChromaDB")
        return ids
    
    def search(self,
              query_embedding: np.ndarray,
              top_k: int = 5,
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search ChromaDB for similar documents."""
        # Convert numpy array to list
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Handle 2D array (batch of 1)
        if isinstance(query_list[0], list):
            query_list = query_list[0]
        
        # Build where clause for filtering
        where_clause = None
        if filters:
            where_clause = filters
        
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            where=where_clause
        )
        
        # Parse results
        search_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                search_results.append(SearchResult(
                    document=results['documents'][0][i],
                    score=1.0 - results['distances'][0][i] if self.distance_metric == 'cosine' else results['distances'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    distance=results['distances'][0][i],
                    id=results['ids'][0][i]
                ))
        
        self.logger.debug(f"Found {len(search_results)} results")
        return search_results
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from ChromaDB."""
        self.collection.delete(ids=document_ids)
        self.logger.debug(f"Deleted {len(document_ids)} documents")
    
    def save(self, path: Optional[str] = None) -> None:
        """ChromaDB auto-persists, but we can trigger explicit persist."""
        # ChromaDB with persist_directory auto-saves
        self.logger.info("ChromaDB auto-persists to disk")
    
    def load(self, path: str = None) -> None:
        """Load is handled automatically by ChromaDB on initialization."""
        # ChromaDB PersistentClient loads automatically
        # Just verify the collection has data
        count = self.collection.count()
        if count == 0:
            raise ValueError("ChromaDB collection is empty")
        self.logger.info(f"ChromaDB loaded with {count} documents")
    
    def count(self) -> int:
        """Get number of documents in ChromaDB."""
        return self.collection.count()



class FAISSBackend(VectorDatabase):
    """FAISS implementation of vector database."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FAISS backend.
        
        Args:
            config: FAISS configuration
        """
        super().__init__(config)
        
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        
        self.faiss = faiss
        self.persist_directory = config.get('persist_directory', 'data/vector_index/faiss')
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.index_type = config.get('index_type', 'Flat')
        self.dimension = config.get('dimension', 768)
        
        # Initialize index
        self.index = None
        self.documents = []
        self.metadata = []
        self.ids = []
        
        self._create_index()
        
        self.logger.info(f"FAISS initialized with {self.index_type} index")
    
    def _create_index(self):
        """Create FAISS index based on configuration."""
        if self.index_type == 'Flat':
            # Exact search
            if self.distance_metric == 'cosine':
                self.index = self.faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
            else:
                self.index = self.faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == 'HNSW':
            # HNSW index for fast approximate search
            M = self.config.get('M', 32)
            if self.distance_metric == 'cosine':
                self.index = self.faiss.IndexHNSWFlat(self.dimension, M, self.faiss.METRIC_INNER_PRODUCT)
            else:
                self.index = self.faiss.IndexHNSWFlat(self.dimension, M, self.faiss.METRIC_L2)
            
            ef_construction = self.config.get('efConstruction', 40)
            ef_search = self.config.get('efSearch', 16)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        
        elif self.index_type == 'IVF':
            # IVF index for large-scale search
            nlist = self.config.get('nlist', 100)
            quantizer = self.faiss.IndexFlatL2(self.dimension)
            self.index = self.faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.nprobe = self.config.get('nprobe', 10)
    
    def add_documents(self,
                     documents: List[str],
                     embeddings: np.ndarray,
                     metadata: List[Dict[str, Any]],
                     ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to FAISS index."""
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Ensure embeddings are float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.distance_metric == 'cosine':
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        # Train index if needed (IVF)
        if self.index_type == 'IVF' and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and metadata separately
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        self.ids.extend(ids)
        
        self.logger.debug(f"Added {len(documents)} documents to FAISS")
        return ids
    
    def search(self,
              query_embedding: np.ndarray,
              top_k: int = 5,
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search FAISS index for similar documents."""
        # Ensure query is float32 and 2D
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        if self.distance_metric == 'cosine':
            norm = np.linalg.norm(query_embedding)
            query_embedding = query_embedding / norm
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Parse results
        search_results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            
            # Apply metadata filters
            if filters:
                meta = self.metadata[idx]
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue
            
            # Convert distance to score
            distance = float(distances[0][i])
            if self.distance_metric == 'cosine':
                score = distance  # Inner product is already similarity
            else:
                score = 1.0 / (1.0 + distance)  # Convert L2 distance to similarity
            
            search_results.append(SearchResult(
                document=self.documents[idx],
                score=score,
                metadata=self.metadata[idx],
                distance=distance,
                id=self.ids[idx]
            ))
        
        self.logger.debug(f"Found {len(search_results)} results")
        return search_results
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from FAISS (requires rebuilding index)."""
        # FAISS doesn't support deletion, so we need to rebuild
        indices_to_keep = [i for i, doc_id in enumerate(self.ids) if doc_id not in document_ids]
        
        if not indices_to_keep:
            # All documents deleted
            self._create_index()
            self.documents = []
            self.metadata = []
            self.ids = []
            self.logger.debug("All documents deleted, index reset")
            return
        
        # Rebuild index with remaining documents
        kept_docs = [self.documents[i] for i in indices_to_keep]
        kept_meta = [self.metadata[i] for i in indices_to_keep]
        kept_ids = [self.ids[i] for i in indices_to_keep]
        
        # Get embeddings (we need to re-embed or store them)
        # For now, we'll just update the metadata
        self.documents = kept_docs
        self.metadata = kept_meta
        self.ids = kept_ids
        
        self.logger.warning("FAISS deletion requires index rebuild - not fully implemented")
    
    def save(self, path: Optional[str] = None) -> None:
        """Save FAISS index to disk."""
        import pickle
        
        if path is None:
            path = self.persist_directory
        
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = Path(path) / f"{self.collection_name}.index"
        self.faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = Path(path) / f"{self.collection_name}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'ids': self.ids,
                'config': self.config
            }, f)
        
        self.logger.info(f"FAISS index saved to {path}")
    
    def load(self, path: str) -> None:
        """Load FAISS index from disk."""
        import pickle
        
        # Load FAISS index
        index_path = Path(path) / f"{self.collection_name}.index"
        self.index = self.faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = Path(path) / f"{self.collection_name}_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.ids = data['ids']
        
        self.logger.info(f"FAISS index loaded from {path}")
    
    def count(self) -> int:
        """Get number of documents in FAISS index."""
        return self.index.ntotal



class VectorDatabaseFactory:
    """Factory for creating vector database instances."""
    
    _backends = {
        'chromadb': ChromaDBBackend,
        'faiss': FAISSBackend,
    }
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> VectorDatabase:
        """
        Create vector database instance based on configuration.
        
        Args:
            config: Database configuration with 'type' field
            
        Returns:
            VectorDatabase instance
            
        Raises:
            ValueError: If database type is not supported
        """
        db_type = config.get('type', 'chromadb').lower()
        
        if db_type not in cls._backends:
            raise ValueError(
                f"Unsupported vector database type: {db_type}. "
                f"Supported types: {list(cls._backends.keys())}"
            )
        
        backend_class = cls._backends[db_type]
        return backend_class(config)
    
    @classmethod
    def register_backend(cls, name: str, backend_class: type):
        """
        Register a custom vector database backend.
        
        Args:
            name: Backend name
            backend_class: Backend class (must inherit from VectorDatabase)
        """
        if not issubclass(backend_class, VectorDatabase):
            raise TypeError("Backend class must inherit from VectorDatabase")
        
        cls._backends[name] = backend_class
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """Get list of available backend types."""
        return list(cls._backends.keys())
