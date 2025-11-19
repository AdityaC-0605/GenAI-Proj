"""Request handler for QA inference."""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio

from src.data_models import QAPrediction

logger = logging.getLogger(__name__)


@dataclass
class QARequest:
    """Question answering request."""
    question: str
    context: str
    question_language: Optional[str] = None
    context_language: Optional[str] = None
    model_id: Optional[str] = None


@dataclass
class QAResponse:
    """Question answering response."""
    answer: str
    confidence: float
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    processing_time_ms: float = 0.0
    model_used: str = ""
    question_language: str = ""
    context_language: str = ""


class RequestHandler:
    """Handles and validates inference requests."""
    
    # Supported languages
    SUPPORTED_QUESTION_LANGUAGES = {'en', 'es', 'fr', 'de', 'zh', 'ar'}
    SUPPORTED_CONTEXT_LANGUAGES = {'en', 'es', 'fr', 'de', 'zh', 'ar', 'hi', 'ja', 'ko'}
    
    # Limits
    MAX_QUESTION_LENGTH = 512
    MAX_CONTEXT_LENGTH = 4096
    MAX_BATCH_SIZE = 32
    BATCH_TIMEOUT_MS = 100
    
    def __init__(
        self,
        model_manager,
        enable_batching: bool = True,
        max_batch_size: int = 32,
        batch_timeout_ms: float = 100
    ):
        """
        Initialize request handler.
        
        Args:
            model_manager: ModelManager instance for loading models
            enable_batching: Whether to enable dynamic batching
            max_batch_size: Maximum batch size
            batch_timeout_ms: Timeout for batch accumulation (ms)
        """
        self.model_manager = model_manager
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        # Batch queue
        self.batch_queue: List[QARequest] = []
        self.batch_lock = asyncio.Lock()
    
    def validate_request(
        self,
        question: str,
        context: str,
        question_language: Optional[str] = None,
        context_language: Optional[str] = None
    ) -> Optional[str]:
        """
        Validate QA request parameters.
        
        Args:
            question: Question text
            context: Context text
            question_language: Question language code (optional)
            context_language: Context language code (optional)
            
        Returns:
            Error message if invalid, None if valid
        """
        # Check required fields
        if not question or not question.strip():
            return "Question is required and cannot be empty"
        
        if not context or not context.strip():
            return "Context is required and cannot be empty"
        
        # Check length limits
        if len(question) > self.MAX_QUESTION_LENGTH:
            return f"Question exceeds maximum length of {self.MAX_QUESTION_LENGTH} characters"
        
        if len(context) > self.MAX_CONTEXT_LENGTH:
            return f"Context exceeds maximum length of {self.MAX_CONTEXT_LENGTH} characters"
        
        # Validate language codes if provided
        if question_language:
            if question_language not in self.SUPPORTED_QUESTION_LANGUAGES:
                return f"Unsupported question language: {question_language}"
        
        if context_language:
            if context_language not in self.SUPPORTED_CONTEXT_LANGUAGES:
                return f"Unsupported context language: {context_language}"
        
        return None
    
    def validate_batch(self, requests: List[QARequest]) -> tuple[bool, Optional[str]]:
        """
        Validate batch of requests.
        
        Args:
            requests: List of QA requests
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not requests:
            return False, "Batch cannot be empty"
        
        if len(requests) > self.MAX_BATCH_SIZE:
            return False, f"Batch size {len(requests)} exceeds maximum of {self.MAX_BATCH_SIZE}"
        
        # Validate each request
        for i, request in enumerate(requests):
            is_valid, error = self.validate_request(request)
            if not is_valid:
                return False, f"Request {i}: {error}"
        
        return True, None
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text (placeholder implementation).
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        # Simple heuristic-based detection (placeholder)
        # In production, use langdetect or fasttext
        
        # Check for common patterns
        if any(char in text for char in '的中国'):
            return 'zh'
        elif any(char in text for char in 'العربية'):
            return 'ar'
        elif any(char in text for char in 'हिन्दी'):
            return 'hi'
        elif any(char in text for char in '日本語'):
            return 'ja'
        elif any(char in text for char in '한국어'):
            return 'ko'
        
        # Default to English for Latin script
        return 'en'
    
    def prepare_request(self, request: QARequest) -> QARequest:
        """
        Prepare request by filling in missing fields.
        
        Args:
            request: QA request
            
        Returns:
            Prepared request
        """
        # Auto-detect languages if not provided
        if not request.question_language:
            request.question_language = self.detect_language(request.question)
        
        if not request.context_language:
            request.context_language = self.detect_language(request.context)
        
        return request
    
    async def add_to_batch(self, request: QARequest) -> int:
        """
        Add request to batch queue.
        
        Args:
            request: QA request
            
        Returns:
            Current batch size
        """
        async with self.batch_lock:
            self.batch_queue.append(request)
            return len(self.batch_queue)
    
    async def get_batch(self) -> List[QARequest]:
        """
        Get accumulated batch.
        
        Returns:
            List of requests in batch
        """
        async with self.batch_lock:
            batch = self.batch_queue.copy()
            self.batch_queue.clear()
            return batch
    
    def should_flush_batch(self, batch_size: int, time_since_first: float) -> bool:
        """
        Determine if batch should be flushed.
        
        Args:
            batch_size: Current batch size
            time_since_first: Time since first request in batch (ms)
            
        Returns:
            Whether to flush batch
        """
        return (batch_size >= self.max_batch_size or 
                time_since_first >= self.batch_timeout_ms)
    
    def format_response(
        self,
        answer: str,
        confidence: float,
        request: QARequest,
        model_id: str,
        processing_time_ms: float,
        start_position: Optional[int] = None,
        end_position: Optional[int] = None
    ) -> QAResponse:
        """
        Format prediction into response.
        
        Args:
            answer: Predicted answer
            confidence: Confidence score
            request: Original request
            model_id: Model identifier
            processing_time_ms: Processing time in milliseconds
            start_position: Start position in context (for extractive)
            end_position: End position in context (for extractive)
            
        Returns:
            QAResponse object
        """
        return QAResponse(
            answer=answer,
            confidence=confidence,
            start_position=start_position,
            end_position=end_position,
            processing_time_ms=processing_time_ms,
            model_used=model_id,
            question_language=request.question_language or "",
            context_language=request.context_language or ""
        )
    
    def process_request(
        self,
        question: str,
        context: str,
        question_lang: str,
        context_lang: str,
        model_name: str = "mbert"
    ) -> QAPrediction:
        """
        Process a QA request and return prediction.
        
        Args:
            question: Question text
            context: Context text
            question_lang: Question language code
            context_lang: Context language code
            model_name: Model to use ('mbert' or 'mt5')
            
        Returns:
            QAPrediction with answer and metadata
        """
        # Load model if not already loaded
        model = self.model_manager.get_model(f"{model_name}_pretrained")
        if model is None:
            model = self.model_manager.load_model(
                model_type=model_name,
                model_id=f"{model_name}_pretrained"
            )
        
        # Generate prediction
        prediction = model.predict(
            question=question,
            context=context,
            question_lang=question_lang,
            context_lang=context_lang
        )
        
        return prediction
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """
        Get supported language pairs.
        
        Returns:
            Dictionary with supported languages
        """
        return {
            'question_languages': sorted(list(self.SUPPORTED_QUESTION_LANGUAGES)),
            'context_languages': sorted(list(self.SUPPORTED_CONTEXT_LANGUAGES)),
            'total_pairs': len(self.SUPPORTED_QUESTION_LANGUAGES) * len(self.SUPPORTED_CONTEXT_LANGUAGES)
        }
