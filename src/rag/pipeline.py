"""
End-to-end RAG pipeline for question answering.

Orchestrates retrieval and generation components to provide complete
question answering functionality.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .retrieval import SemanticRetriever, HybridRetriever, ContextReranker, RetrievalResult
from .generation import AnswerGenerator, GenerationResult
from .logging_config import LoggerMixin


@dataclass
class RAGResponse:
    """Complete RAG response."""
    answer: str
    confidence: float
    contexts: List[RetrievalResult]
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: Dict[str, Any]


class RAGPipeline(LoggerMixin):
    """End-to-end RAG pipeline orchestrating retrieval and generation."""
    
    def __init__(self,
                 retriever: SemanticRetriever,
                 generator: AnswerGenerator,
                 reranker: Optional[ContextReranker] = None,
                 use_reranking: bool = False):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Retriever instance (SemanticRetriever or HybridRetriever)
            generator: Generator instance
            reranker: Optional re-ranker instance
            use_reranking: Whether to use re-ranking by default
        """
        super().__init__()
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.use_reranking = use_reranking
        
        self.logger.info("RAG Pipeline initialized")
        self.logger.info(f"Retriever: {type(retriever).__name__}")
        self.logger.info(f"Generator: {type(generator).__name__}")
        if reranker:
            self.logger.info(f"Re-ranker: {type(reranker).__name__}")
    
    def answer_question(self,
                       question: str,
                       top_k: int = 5,
                       language: Optional[str] = None,
                       use_reranking: Optional[bool] = None,
                       **kwargs) -> RAGResponse:
        """
        Answer a question using RAG.
        
        Args:
            question: Question text
            top_k: Number of contexts to retrieve
            language: Filter by language
            use_reranking: Whether to use re-ranking (overrides default)
            **kwargs: Additional arguments for retrieval and generation
            
        Returns:
            RAGResponse object
        """
        start_time = time.time()
        
        self.logger.debug(f"Processing question: {question[:50]}...")
        
        try:
            # Step 1: Retrieve contexts
            retrieval_start = time.time()
            
            # Retrieve more contexts if re-ranking
            should_rerank = use_reranking if use_reranking is not None else self.use_reranking
            retrieve_k = top_k * 2 if should_rerank and self.reranker else top_k
            
            retrieval_results = self.retriever.retrieve(
                question=question,
                top_k=retrieve_k,
                language=language
            )
            
            retrieval_time = time.time() - retrieval_start
            self.logger.debug(f"Retrieved {len(retrieval_results)} contexts in {retrieval_time:.2f}s")
            
            # Step 2: Optional re-ranking
            if should_rerank and self.reranker and retrieval_results:
                rerank_start = time.time()
                retrieval_results = self.reranker.rerank(
                    question=question,
                    results=retrieval_results,
                    top_k=top_k
                )
                rerank_time = time.time() - rerank_start
                self.logger.debug(f"Re-ranked contexts in {rerank_time:.2f}s")
                retrieval_time += rerank_time
            
            # Handle no results
            if not retrieval_results:
                self.logger.warning("No contexts retrieved")
                return RAGResponse(
                    answer="I couldn't find relevant information to answer this question.",
                    confidence=0.0,
                    contexts=[],
                    retrieval_time=retrieval_time,
                    generation_time=0.0,
                    total_time=time.time() - start_time,
                    metadata={
                        'error': 'no_contexts_found',
                        'question': question
                    }
                )
            
            # Step 3: Generate answer
            generation_start = time.time()
            contexts = [r.text for r in retrieval_results]
            
            generation_result = self.generator.generate_answer(
                question=question,
                contexts=contexts,
                **kwargs
            )
            
            generation_time = time.time() - generation_start
            self.logger.debug(f"Generated answer in {generation_time:.2f}s")
            
            # Step 4: Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                retrieval_results,
                generation_result
            )
            
            total_time = time.time() - start_time
            
            response = RAGResponse(
                answer=generation_result.answer,
                confidence=overall_confidence,
                contexts=retrieval_results,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                metadata={
                    'question': question,
                    'num_contexts_retrieved': len(retrieval_results),
                    'num_contexts_used': len(contexts),
                    'retrieval_scores': [r.score for r in retrieval_results],
                    'generation_confidence': generation_result.confidence,
                    'generator_metadata': generation_result.metadata,
                    'language': language
                }
            )
            
            self.logger.info(
                f"Question answered in {total_time:.2f}s "
                f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)"
            )
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error processing question: {e}", exc_info=True)
            return RAGResponse(
                answer=f"An error occurred while processing your question: {str(e)}",
                confidence=0.0,
                contexts=[],
                retrieval_time=0.0,
                generation_time=0.0,
                total_time=time.time() - start_time,
                metadata={
                    'error': str(e),
                    'question': question
                }
            )
    
    def _calculate_overall_confidence(self,
                                     retrieval_results: List[RetrievalResult],
                                     generation_result: GenerationResult) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            retrieval_results: List of retrieval results
            generation_result: Generation result
            
        Returns:
            Overall confidence score (0-1)
        """
        if not retrieval_results:
            return 0.0
        
        # Average retrieval score from top 3 results
        top_results = retrieval_results[:3]
        avg_retrieval_score = sum(r.score for r in top_results) / len(top_results)
        
        # Generation confidence
        generation_confidence = generation_result.confidence
        
        # Weighted average: 60% generation, 40% retrieval
        overall_confidence = 0.6 * generation_confidence + 0.4 * avg_retrieval_score
        
        return overall_confidence


class BatchRAGPipeline(RAGPipeline):
    """RAG pipeline with batch processing support."""
    
    def __init__(self,
                 retriever: SemanticRetriever,
                 generator: AnswerGenerator,
                 reranker: Optional[ContextReranker] = None,
                 use_reranking: bool = False,
                 max_workers: int = 4):
        """
        Initialize batch RAG pipeline.
        
        Args:
            retriever: Retriever instance
            generator: Generator instance
            reranker: Optional re-ranker instance
            use_reranking: Whether to use re-ranking
            max_workers: Maximum number of parallel workers
        """
        super().__init__(retriever, generator, reranker, use_reranking)
        self.max_workers = max_workers
        self.logger.info(f"Batch processing enabled with {max_workers} workers")
    
    def answer_questions_batch(self,
                              questions: List[str],
                              parallel: bool = True,
                              **kwargs) -> List[RAGResponse]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            parallel: Whether to process in parallel
            **kwargs: Additional arguments for answer_question()
            
        Returns:
            List of RAGResponse objects
        """
        self.logger.info(f"Processing batch of {len(questions)} questions")
        
        if not parallel or len(questions) == 1:
            # Sequential processing
            responses = []
            for question in questions:
                response = self.answer_question(question, **kwargs)
                responses.append(response)
            return responses
        
        # Parallel processing
        responses = [None] * len(questions)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.answer_question, question, **kwargs): i
                for i, question in enumerate(questions)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    responses[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing question {index}: {e}")
                    responses[index] = RAGResponse(
                        answer=f"Error: {str(e)}",
                        confidence=0.0,
                        contexts=[],
                        retrieval_time=0.0,
                        generation_time=0.0,
                        total_time=0.0,
                        metadata={'error': str(e)}
                    )
        
        self.logger.info(f"Batch processing complete: {len(responses)} responses")
        return responses
