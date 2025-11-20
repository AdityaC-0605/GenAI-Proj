"""
Advanced retrieval features for RAG system.

Implements query expansion, multi-hop retrieval, and context chunking.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re

from .retrieval import SemanticRetriever, RetrievalResult
from .generation import AnswerGenerator
from .logging_config import LoggerMixin


class QueryExpander(LoggerMixin):
    """Expand queries to improve retrieval recall."""
    
    def __init__(self, generator: Optional[AnswerGenerator] = None):
        """
        Initialize query expander.
        
        Args:
            generator: Optional generator for LLM-based expansion
        """
        super().__init__()
        self.generator = generator
    
    def expand_query(self, question: str, num_expansions: int = 3) -> List[str]:
        """
        Expand query into similar questions.
        
        Args:
            question: Original question
            num_expansions: Number of expanded queries to generate
            
        Returns:
            List of expanded queries (including original)
        """
        expanded = [question]  # Always include original
        
        if self.generator:
            # LLM-based expansion
            try:
                prompt = f"""Generate {num_expansions} alternative ways to ask this question:

Question: {question}

Alternative questions (one per line):"""
                
                result = self.generator.generate_answer(
                    question=prompt,
                    contexts=[],
                    max_length=200
                )
                
                # Parse alternatives
                alternatives = [
                    line.strip()
                    for line in result.answer.split('\n')
                    if line.strip() and not line.strip().startswith(('Question:', 'Alternative'))
                ]
                
                expanded.extend(alternatives[:num_expansions])
                self.logger.debug(f"Generated {len(alternatives)} query expansions")
                
            except Exception as e:
                self.logger.warning(f"Query expansion failed: {e}")
        else:
            # Rule-based expansion (simple)
            expanded.extend(self._rule_based_expansion(question, num_expansions))
        
        return expanded[:num_expansions + 1]
    
    def _rule_based_expansion(self, question: str, num: int) -> List[str]:
        """Simple rule-based query expansion."""
        expansions = []
        
        # Add "what is" variant
        if not question.lower().startswith('what'):
            expansions.append(f"What is {question.lower().replace('?', '')}?")
        
        # Add "explain" variant
        if not question.lower().startswith('explain'):
            expansions.append(f"Explain {question.lower().replace('?', '')}")
        
        # Add "describe" variant
        if not question.lower().startswith('describe'):
            expansions.append(f"Describe {question.lower().replace('?', '')}")
        
        return expansions[:num]


class MultiHopRetriever(LoggerMixin):
    """Multi-hop retrieval for complex questions."""
    
    def __init__(self,
                 retriever: SemanticRetriever,
                 generator: AnswerGenerator,
                 max_hops: int = 3):
        """
        Initialize multi-hop retriever.
        
        Args:
            retriever: Base retriever
            generator: Generator for intermediate answers
            max_hops: Maximum number of retrieval hops
        """
        super().__init__()
        self.retriever = retriever
        self.generator = generator
        self.max_hops = max_hops
    
    def retrieve_multi_hop(self,
                          question: str,
                          top_k: int = 5) -> List[RetrievalResult]:
        """
        Perform multi-hop retrieval.
        
        Args:
            question: Original question
            top_k: Number of results per hop
            
        Returns:
            Combined retrieval results from all hops
        """
        self.logger.info(f"Starting multi-hop retrieval for: {question[:50]}...")
        
        all_results = {}  # Use dict to deduplicate by ID
        current_query = question
        
        for hop in range(self.max_hops):
            self.logger.debug(f"Hop {hop + 1}: {current_query[:50]}...")
            
            # Retrieve for current query
            results = self.retriever.retrieve(current_query, top_k=top_k)
            
            if not results:
                self.logger.debug(f"No results at hop {hop + 1}, stopping")
                break
            
            # Add to combined results
            for result in results:
                if result.id not in all_results:
                    all_results[result.id] = result
            
            # Generate intermediate answer for next hop
            if hop < self.max_hops - 1:
                try:
                    contexts = [r.text for r in results[:3]]
                    intermediate = self.generator.generate_answer(
                        question=current_query,
                        contexts=contexts,
                        max_length=50
                    )
                    
                    # Use answer to formulate next query
                    current_query = f"{question} {intermediate.answer}"
                    self.logger.debug(f"Next query: {current_query[:50]}...")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate intermediate answer: {e}")
                    break
        
        # Convert back to list and re-rank
        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        self.logger.info(f"Multi-hop retrieval complete: {len(combined_results)} unique results")
        return combined_results[:top_k * 2]  # Return more results


class ContextChunker(LoggerMixin):
    """Chunk long documents for better retrieval."""
    
    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        Initialize context chunker.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        # Simple sentence-based chunking
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'id': f"{doc_id}_chunk_{len(chunks)}",
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_index': len(chunks),
                        'parent_doc_id': doc_id,
                        'is_chunk': True
                    }
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.chunk_overlap // 20:]  # Rough estimate
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'id': f"{doc_id}_chunk_{len(chunks)}",
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'parent_doc_id': doc_id,
                    'is_chunk': True
                }
            })
        
        self.logger.debug(f"Chunked document {doc_id} into {len(chunks)} chunks")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def merge_chunks(self, chunks: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Merge chunks from same document.
        
        Args:
            chunks: List of chunk results
            
        Returns:
            Merged results
        """
        # Group by parent document
        doc_groups: Dict[str, List[RetrievalResult]] = {}
        
        for chunk in chunks:
            parent_id = chunk.metadata.get('parent_doc_id', chunk.id)
            if parent_id not in doc_groups:
                doc_groups[parent_id] = []
            doc_groups[parent_id].append(chunk)
        
        # Merge chunks from same document
        merged = []
        for parent_id, group in doc_groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Combine chunks
                combined_text = ' ... '.join(c.text for c in sorted(group, key=lambda x: x.metadata.get('chunk_index', 0)))
                avg_score = sum(c.score for c in group) / len(group)
                
                merged_result = RetrievalResult(
                    text=combined_text,
                    score=avg_score,
                    metadata={
                        **group[0].metadata,
                        'num_chunks_merged': len(group),
                        'is_merged': True
                    },
                    rank=0,
                    id=parent_id
                )
                merged.append(merged_result)
        
        # Re-rank
        merged.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(merged):
            result.rank = i + 1
        
        self.logger.debug(f"Merged {len(chunks)} chunks into {len(merged)} documents")
        return merged


class AdvancedRetriever(LoggerMixin):
    """Combined advanced retrieval features."""
    
    def __init__(self,
                 retriever: SemanticRetriever,
                 generator: Optional[AnswerGenerator] = None,
                 enable_expansion: bool = True,
                 enable_multi_hop: bool = False,
                 enable_chunking: bool = False):
        """
        Initialize advanced retriever.
        
        Args:
            retriever: Base retriever
            generator: Generator for expansion and multi-hop
            enable_expansion: Enable query expansion
            enable_multi_hop: Enable multi-hop retrieval
            enable_chunking: Enable context chunking
        """
        super().__init__()
        self.retriever = retriever
        self.enable_expansion = enable_expansion
        self.enable_multi_hop = enable_multi_hop
        self.enable_chunking = enable_chunking
        
        # Initialize components
        self.expander = QueryExpander(generator) if enable_expansion else None
        self.multi_hop = MultiHopRetriever(retriever, generator) if enable_multi_hop and generator else None
        self.chunker = ContextChunker() if enable_chunking else None
    
    def retrieve(self, question: str, top_k: int = 5, **kwargs) -> List[RetrievalResult]:
        """
        Advanced retrieval with all enabled features.
        
        Args:
            question: Question text
            top_k: Number of results
            **kwargs: Additional arguments
            
        Returns:
            Retrieved results
        """
        all_results = {}
        
        # Query expansion
        if self.enable_expansion and self.expander:
            queries = self.expander.expand_query(question)
            self.logger.debug(f"Expanded to {len(queries)} queries")
            
            for query in queries:
                results = self.retriever.retrieve(query, top_k=top_k, **kwargs)
                for result in results:
                    if result.id not in all_results:
                        all_results[result.id] = result
        else:
            # Single query
            results = self.retriever.retrieve(question, top_k=top_k, **kwargs)
            for result in results:
                all_results[result.id] = result
        
        # Multi-hop retrieval
        if self.enable_multi_hop and self.multi_hop:
            hop_results = self.multi_hop.retrieve_multi_hop(question, top_k=top_k)
            for result in hop_results:
                if result.id not in all_results:
                    all_results[result.id] = result
        
        # Combine and re-rank
        combined = list(all_results.values())
        combined.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined):
            result.rank = i + 1
        
        return combined[:top_k]
