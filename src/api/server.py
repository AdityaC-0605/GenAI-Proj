"""FastAPI server for Cross-Lingual QA system."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Suppress urllib3 warnings (harmless on macOS)
try:
    from src.utils.warning_suppressor import suppress_urllib3_warnings
    suppress_urllib3_warnings()
except ImportError:
    pass

from src.inference.model_manager import ModelManager
from src.inference.request_handler import RequestHandler

logger = logging.getLogger(__name__)


# Request/Response Models
class QARequest(BaseModel):
    """Request model for question answering."""
    question: str = Field(..., description="Question text")
    context: str = Field(..., description="Context/document text")
    question_language: str = Field(..., description="Question language code (e.g., 'en', 'es')")
    context_language: str = Field(..., description="Context language code")
    model_name: Optional[str] = Field(None, description="Model to use (default: mbert)")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What is the capital of France?",
                "context": "Paris is the capital and most populous city of France.",
                "question_language": "en",
                "context_language": "en",
                "model_name": "mbert"
            }
        }


class QAResponse(BaseModel):
    """Response model for question answering."""
    answer: str = Field(..., description="Predicted answer")
    confidence: float = Field(..., description="Confidence score (0-1)")
    start_position: Optional[int] = Field(None, description="Start position in context (extractive models)")
    end_position: Optional[int] = Field(None, description="End position in context (extractive models)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_used: str = Field(..., description="Model that generated the answer")
    question_language: str = Field(..., description="Question language")
    context_language: str = Field(..., description="Context language")


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    type: str
    description: str
    supported_languages: List[str]


class LanguageSupport(BaseModel):
    """Supported languages information."""
    question_languages: List[str]
    context_languages: List[str]
    total_pairs: int


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: int


# Initialize FastAPI app
app = FastAPI(
    title="Cross-Lingual QA API",
    description="API for cross-lingual question answering using mBERT and mT5 models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_manager = ModelManager()
request_handler = RequestHandler(model_manager)

# Supported languages
QUESTION_LANGUAGES = ['en', 'es', 'fr', 'de', 'zh', 'ar']
CONTEXT_LANGUAGES = ['en', 'es', 'fr', 'de', 'zh', 'ar', 'hi', 'ja', 'ko']


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Cross-Lingual QA API server")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Cross-Lingual QA API server")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Cross-Lingual QA API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthStatus, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system health status.
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(model_manager.loaded_models)
    )


@app.post("/predict", response_model=QAResponse, tags=["Question Answering"])
async def predict(request: QARequest):
    """
    Answer a question based on provided context.
    
    Args:
        request: QA request with question, context, and languages
        
    Returns:
        QA response with answer and metadata
    """
    start_time = time.time()
    
    try:
        # Validate request
        validation_error = request_handler.validate_request(
            request.question,
            request.context,
            request.question_language,
            request.context_language
        )
        
        if validation_error:
            raise HTTPException(status_code=400, detail=validation_error)
        
        # Process request
        prediction = request_handler.process_request(
            question=request.question,
            context=request.context,
            question_lang=request.question_language,
            context_lang=request.context_language,
            model_name=request.model_name or "mbert"
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return QAResponse(
            answer=prediction.answer_text,
            confidence=prediction.confidence,
            start_position=prediction.start_position,
            end_position=prediction.end_position,
            processing_time_ms=processing_time,
            model_used=request.model_name or "mbert",
            question_language=request.question_language,
            context_language=request.context_language
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[QAResponse], tags=["Question Answering"])
async def predict_batch(requests: List[QARequest]):
    """
    Answer multiple questions in batch.
    
    Args:
        requests: List of QA requests
        
    Returns:
        List of QA responses
    """
    if len(requests) > 32:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 32 requests"
        )
    
    responses = []
    
    for req in requests:
        try:
            response = await predict(req)
            responses.append(response)
        except HTTPException as e:
            # Add error response
            responses.append(QAResponse(
                answer="",
                confidence=0.0,
                processing_time_ms=0.0,
                model_used=req.model_name or "mbert",
                question_language=req.question_language,
                context_language=req.context_language
            ))
    
    return responses


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """
    List available models.
    
    Returns:
        List of available models with their information
    """
    models = [
        ModelInfo(
            name="mbert",
            type="extractive",
            description="Multilingual BERT for extractive question answering",
            supported_languages=CONTEXT_LANGUAGES
        ),
        ModelInfo(
            name="mt5",
            type="generative",
            description="Multilingual T5 for generative question answering",
            supported_languages=CONTEXT_LANGUAGES
        )
    ]
    
    return models


@app.get("/languages", response_model=LanguageSupport, tags=["Languages"])
async def list_languages():
    """
    List supported language pairs.
    
    Returns:
        Information about supported languages
    """
    return LanguageSupport(
        question_languages=QUESTION_LANGUAGES,
        context_languages=CONTEXT_LANGUAGES,
        total_pairs=len(QUESTION_LANGUAGES) * len(CONTEXT_LANGUAGES)
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
