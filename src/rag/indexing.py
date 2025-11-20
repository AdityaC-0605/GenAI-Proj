"""
Dataset indexing pipeline for RAG system.

Processes QA datasets (SQuAD, XQuAD, MLQA, TyDi QA) into indexable documents
and builds vector database indices.
"""

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from .vector_db import VectorDatabase, Document
from .embeddings import EmbeddingManager
from .logging_config import LoggerMixin


@dataclass
class IndexDocument:
    """Document for vector database indexing."""
    id: str
    text: str
    metadata: Dict[str, Any]


class DocumentProcessor(LoggerMixin):
    """Process QA datasets into indexable documents."""
    
    def __init__(self):
        """Initialize document processor."""
        super().__init__()
    
    def _generate_id(self, title: str, text: str) -> str:
        """
        Generate unique document ID from title and text.
        
        Args:
            title: Document title
            text: Document text
            
        Returns:
            Unique document ID
        """
        content = f"{title}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def process_squad_dataset(self, 
                             dataset_path: str,
                             split: str = 'train') -> List[IndexDocument]:
        """
        Process SQuAD format dataset.
        
        Args:
            dataset_path: Path to SQuAD JSON file
            split: Dataset split (train, dev, test)
            
        Returns:
            List of IndexDocument objects
        """
        self.logger.info(f"Processing SQuAD dataset from {dataset_path}")
        
        data = self._load_json(dataset_path)
        documents = []
        
        for article in data.get('data', []):
            title = article.get('title', 'Unknown')
            
            for paragraph in article.get('paragraphs', []):
                context = paragraph['context']
                qas = paragraph.get('qas', [])
                
                # Extract questions and answers
                questions = [qa['question'] for qa in qas]
                answers = []
                for qa in qas:
                    if qa.get('answers'):
                        answers.append(qa['answers'][0]['text'])
                    elif qa.get('plausible_answers'):  # SQuAD 2.0
                        answers.append(qa['plausible_answers'][0]['text'])
                    else:
                        answers.append('')
                
                doc = IndexDocument(
                    id=self._generate_id(title, context),
                    text=context,
                    metadata={
                        'title': title,
                        'language': 'en',
                        'dataset': 'squad',
                        'split': split,
                        'questions': str(questions),  # Convert to string for storage
                        'answers': str(answers),
                        'num_questions': len(questions)
                    }
                )
                documents.append(doc)
        
        self.logger.info(f"Processed {len(documents)} documents from SQuAD")
        return documents
    
    def process_xquad_dataset(self,
                             dataset_path: str,
                             language: str,
                             split: str = 'test') -> List[IndexDocument]:
        """
        Process XQuAD dataset for specific language.
        
        Args:
            dataset_path: Path to XQuAD JSON file
            language: Language code (e.g., 'es', 'de', 'zh')
            split: Dataset split
            
        Returns:
            List of IndexDocument objects
        """
        self.logger.info(f"Processing XQuAD dataset ({language}) from {dataset_path}")
        
        data = self._load_json(dataset_path)
        documents = []
        
        for article in data.get('data', []):
            title = article.get('title', 'Unknown')
            
            for paragraph in article.get('paragraphs', []):
                context = paragraph['context']
                qas = paragraph.get('qas', [])
                
                questions = [qa['question'] for qa in qas]
                answers = [qa['answers'][0]['text'] if qa.get('answers') else '' 
                          for qa in qas]
                
                doc = IndexDocument(
                    id=self._generate_id(title, context),
                    text=context,
                    metadata={
                        'title': title,
                        'language': language,
                        'dataset': 'xquad',
                        'split': split,
                        'questions': str(questions),
                        'answers': str(answers),
                        'num_questions': len(questions)
                    }
                )
                documents.append(doc)
        
        self.logger.info(f"Processed {len(documents)} documents from XQuAD ({language})")
        return documents
    
    def process_mlqa_dataset(self,
                            dataset_path: str,
                            context_lang: str,
                            question_lang: str,
                            split: str = 'test') -> List[IndexDocument]:
        """
        Process MLQA dataset with language pair.
        
        Args:
            dataset_path: Path to MLQA JSON file
            context_lang: Context language code
            question_lang: Question language code
            split: Dataset split
            
        Returns:
            List of IndexDocument objects
        """
        self.logger.info(
            f"Processing MLQA dataset (context: {context_lang}, "
            f"question: {question_lang}) from {dataset_path}"
        )
        
        data = self._load_json(dataset_path)
        documents = []
        
        for article in data.get('data', []):
            title = article.get('title', 'Unknown')
            
            for paragraph in article.get('paragraphs', []):
                context = paragraph['context']
                qas = paragraph.get('qas', [])
                
                questions = [qa['question'] for qa in qas]
                answers = [qa['answers'][0]['text'] if qa.get('answers') else '' 
                          for qa in qas]
                
                doc = IndexDocument(
                    id=self._generate_id(title, context),
                    text=context,
                    metadata={
                        'title': title,
                        'language': context_lang,
                        'question_language': question_lang,
                        'dataset': 'mlqa',
                        'split': split,
                        'questions': str(questions),
                        'answers': str(answers),
                        'num_questions': len(questions)
                    }
                )
                documents.append(doc)
        
        self.logger.info(
            f"Processed {len(documents)} documents from MLQA "
            f"({context_lang}-{question_lang})"
        )
        return documents
    
    def process_tydiqa_dataset(self,
                              dataset_path: str,
                              language: str,
                              split: str = 'train') -> List[IndexDocument]:
        """
        Process TyDi QA dataset.
        
        Args:
            dataset_path: Path to TyDi QA JSON file
            language: Language code
            split: Dataset split
            
        Returns:
            List of IndexDocument objects
        """
        self.logger.info(f"Processing TyDi QA dataset ({language}) from {dataset_path}")
        
        data = self._load_json(dataset_path)
        documents = []
        
        for article in data.get('data', []):
            title = article.get('title', 'Unknown')
            
            for paragraph in article.get('paragraphs', []):
                context = paragraph['context']
                qas = paragraph.get('qas', [])
                
                questions = [qa['question'] for qa in qas]
                answers = [qa['answers'][0]['text'] if qa.get('answers') else '' 
                          for qa in qas]
                
                doc = IndexDocument(
                    id=self._generate_id(title, context),
                    text=context,
                    metadata={
                        'title': title,
                        'language': language,
                        'dataset': 'tydiqa',
                        'split': split,
                        'questions': str(questions),
                        'answers': str(answers),
                        'num_questions': len(questions)
                    }
                )
                documents.append(doc)
        
        self.logger.info(f"Processed {len(documents)} documents from TyDi QA ({language})")
        return documents



class VectorIndexBuilder(LoggerMixin):
    """Build vector database index from documents."""
    
    def __init__(self,
                 vector_db: VectorDatabase,
                 embedding_manager: EmbeddingManager):
        """
        Initialize vector index builder.
        
        Args:
            vector_db: Vector database instance
            embedding_manager: Embedding manager instance
        """
        super().__init__()
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
    
    def build_index(self,
                   documents: List[IndexDocument],
                   batch_size: int = 100,
                   show_progress: bool = True) -> None:
        """
        Build index from documents.
        
        Args:
            documents: List of IndexDocument objects
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar
        """
        self.logger.info(f"Building index from {len(documents)} documents...")
        
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        progress_bar = tqdm(
            total=len(documents),
            desc="Indexing documents",
            disable=not show_progress
        )
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            try:
                # Extract texts, metadata, and IDs
                texts = [doc.text for doc in batch]
                metadata = [doc.metadata for doc in batch]
                ids = [doc.id for doc in batch]
                
                # Generate embeddings
                self.logger.debug(f"Generating embeddings for batch {i//batch_size + 1}/{total_batches}")
                embeddings = self.embedding_manager.embed_texts(
                    texts,
                    batch_size=batch_size,
                    show_progress=False
                )
                
                # Add to vector database
                self.logger.debug(f"Adding batch to vector database")
                self.vector_db.add_documents(
                    documents=texts,
                    embeddings=embeddings,
                    metadata=metadata,
                    ids=ids
                )
                
                progress_bar.update(len(batch))
                
            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        progress_bar.close()
        
        # Save index
        self.logger.info("Saving vector database index...")
        self.vector_db.save()
        
        # Log statistics
        doc_count = self.vector_db.count()
        cache_stats = self.embedding_manager.get_cache_stats()
        
        self.logger.info(
            f"Index built successfully: {doc_count} documents indexed, "
            f"cache hit rate: {cache_stats['hit_rate']:.2%}"
        )
    
    def add_documents_incremental(self,
                                 documents: List[IndexDocument],
                                 batch_size: int = 100) -> None:
        """
        Add documents to existing index incrementally.
        
        Args:
            documents: List of IndexDocument objects to add
            batch_size: Batch size for processing
        """
        self.logger.info(f"Adding {len(documents)} documents to existing index...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            texts = [doc.text for doc in batch]
            metadata = [doc.metadata for doc in batch]
            ids = [doc.id for doc in batch]
            
            # Generate embeddings
            embeddings = self.embedding_manager.embed_texts(texts, batch_size=batch_size)
            
            # Add to vector database
            self.vector_db.add_documents(
                documents=texts,
                embeddings=embeddings,
                metadata=metadata,
                ids=ids
            )
            
            self.logger.debug(f"Added {i+len(batch)}/{len(documents)} documents")
        
        # Save index
        self.vector_db.save()
        self.logger.info(f"Incremental indexing complete: {len(documents)} documents added")
