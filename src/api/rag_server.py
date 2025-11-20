"""
FastAPI server for RAG-based Cross-Lingual Question Answering.

Provides REST API endpoints for question answering with backward compatibility
to the existing fine-tuning system.
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import time
import logging
import os

from src.rag.config import load_rag_config
from src.rag.logging_config import setup_logging
from src.rag.vector_db import VectorDatabaseFactory
from src.rag.embeddings import EmbeddingManager
from src.rag.retrieval import SemanticRetriever, HybridRetriever, ContextReranker
from src.rag.generation import GeneratorFactory
from src.rag.pipeline import RAGPipeline, BatchRAGPipeline
from src.api.middleware import APIKeyAuth, RateLimitMiddleware, UsageTracker


# Initialize FastAPI app
app = FastAPI(
    title="RAG Cross-Lingual QA API",
    description="Retrieval-Augmented Generation system for cross-lingual question answering",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
rate_limit = int(os.getenv("RATE_LIMIT", "100"))  # requests per minute
app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit)

# Global variables for RAG components
rag_pipeline: Optional[RAGPipeline] = None
batch_rag_pipeline: Optional[BatchRAGPipeline] = None
config = None
logger = None
api_key_auth: Optional[APIKeyAuth] = None
usage_tracker: UsageTracker = UsageTracker()


# Request/Response Models
class PredictRequest(BaseModel):
    """Request model for single question prediction."""
    question: str = Field(..., description="Question text")
    question_language: Optional[str] = Field(None, description="Question language code")
    context_language: Optional[str] = Field(None, description="Context language code")
    context: Optional[str] = Field(None, description="Optional context (for compatibility)")
    top_k: int = Field(5, description="Number of contexts to retrieve", ge=1, le=20)
    use_reranking: bool = Field(False, description="Whether to use context re-ranking")
    max_length: int = Field(100, description="Maximum answer length", ge=10, le=500)
    temperature: float = Field(0.7, description="Generation temperature", ge=0.0, le=2.0)


class ContextInfo(BaseModel):
    """Context information in response."""
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any]


class PredictResponse(BaseModel):
    """Response model for predictions."""
    answer: str
    confidence: float
    contexts: Optional[List[ContextInfo]] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    total_time: float
    metadata: Dict[str, Any]


class BatchPredictRequest(BaseModel):
    """Request model for batch predictions."""
    questions: List[PredictRequest]
    parallel: bool = Field(True, description="Whether to process in parallel")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vector_db: str
    generator: str
    embedding_model: str
    uptime: float


class ModelsResponse(BaseModel):
    """Available models response."""
    embedding_models: List[str]
    generator_types: List[str]
    current_embedding_model: str
    current_generator: str


class LanguagesResponse(BaseModel):
    """Supported languages response."""
    languages: List[Dict[str, str]]
    total_count: int


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    global rag_pipeline, batch_rag_pipeline, config, logger, api_key_auth
    
    try:
        # Load configuration
        config_name = "default"  # Can be overridden via environment variable
        config = load_rag_config(config_name)
        
        # Setup logging
        logger = setup_logging(config)
        logger.info("Starting RAG API server...")
        
        # Initialize vector database
        logger.info("Initializing vector database...")
        vector_db = VectorDatabaseFactory.create(config.vector_db)
        
        # Load existing index
        logger.info("Loading vector database index...")
        vector_db.load()
        logger.info(f"Loaded {vector_db.count()} documents")
        
        # Initialize embedding manager
        logger.info("Initializing embedding manager...")
        embedding_manager = EmbeddingManager(
            model_name=config.embedding.model_name,
            device=config.embedding.get('device', 'auto'),
            cache_size=config.embedding.get('cache_size', 10000)
        )
        
        # Initialize retriever
        logger.info("Initializing retriever...")
        if config.retrieval.get('use_hybrid_search', False):
            semantic_retriever = SemanticRetriever(
                vector_db=vector_db,
                embedding_manager=embedding_manager,
                top_k=config.retrieval.top_k
            )
            retriever = HybridRetriever(
                semantic_retriever=semantic_retriever,
                alpha=config.retrieval.get('hybrid_alpha', 0.7),
                build_bm25=False  # Build on demand
            )
        else:
            retriever = SemanticRetriever(
                vector_db=vector_db,
                embedding_manager=embedding_manager,
                top_k=config.retrieval.top_k
            )
        
        # Initialize re-ranker (optional)
        reranker = None
        if config.retrieval.get('use_reranking', False):
            logger.info("Initializing re-ranker...")
            reranker = ContextReranker(
                model_name=config.retrieval.get('reranker_model', 
                                               'cross-encoder/ms-marco-MiniLM-L-6-v2')
            )
        
        # Initialize generator
        logger.info("Initializing generator...")
        generator = GeneratorFactory.create(config.generator)
        
        # Initialize RAG pipelines
        logger.info("Initializing RAG pipelines...")
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            use_reranking=config.retrieval.get('use_reranking', False)
        )
        
        batch_rag_pipeline = BatchRAGPipeline(
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            use_reranking=config.retrieval.get('use_reranking', False),
            max_workers=config.performance.get('connection_pool_size', 4)
        )
        
        # Initialize API key authentication (optional)
        api_keys = {}
        if config.api.get('enable_auth', False):
            # Load API keys from environment or config
            api_key_str = os.getenv("API_KEYS", "")
            if api_key_str:
                # Format: "key1:user1,key2:user2"
                for pair in api_key_str.split(','):
                    if ':' in pair:
                        key, user = pair.split(':', 1)
                        api_keys[key.strip()] = user.strip()
        
        api_key_auth = APIKeyAuth(api_keys)
        if api_key_auth.enabled:
            logger.info(f"API key authentication enabled ({len(api_keys)} keys)")
        else:
            logger.info("API key authentication disabled")
        
        logger.info("RAG API server started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start RAG API server: {e}", exc_info=True)
        raise


# API Endpoints
@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    user_id: Optional[str] = Depends(lambda: api_key_auth)
):
    """
    Answer a single question using RAG.
    
    This endpoint maintains backward compatibility with the existing fine-tuning system.
    Requires API key if authentication is enabled.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        start_time = time.time()
        
        # Determine language filter
        language = request.context_language or request.question_language
        
        # Answer question
        response = rag_pipeline.answer_question(
            question=request.question,
            top_k=request.top_k,
            language=language,
            use_reranking=request.use_reranking,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Convert contexts to response format
        contexts = [
            ContextInfo(
                text=ctx.text,
                score=ctx.score,
                rank=ctx.rank,
                metadata=ctx.metadata
            )
            for ctx in response.contexts
        ]
        
        return PredictResponse(
            answer=response.answer,
            confidence=response.confidence,
            contexts=contexts,
            retrieval_time=response.retrieval_time,
            generation_time=response.generation_time,
            total_time=time.time() - start_time,
            metadata=response.metadata
        )
    
    except Exception as e:
        logger.error(f"Error processing prediction request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictResponse])
async def predict_batch(request: BatchPredictRequest):
    """
    Answer multiple questions in batch.
    
    Supports parallel processing for improved throughput.
    """
    if batch_rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Batch RAG pipeline not initialized")
    
    try:
        # Extract questions and parameters
        questions = [req.question for req in request.questions]
        
        # Use first request's parameters as defaults (can be enhanced)
        first_req = request.questions[0] if request.questions else PredictRequest(question="")
        
        # Answer questions in batch
        responses = batch_rag_pipeline.answer_questions_batch(
            questions=questions,
            parallel=request.parallel,
            top_k=first_req.top_k,
            language=first_req.context_language or first_req.question_language,
            use_reranking=first_req.use_reranking,
            max_length=first_req.max_length,
            temperature=first_req.temperature
        )
        
        # Convert to response format
        result = []
        for response in responses:
            contexts = [
                ContextInfo(
                    text=ctx.text,
                    score=ctx.score,
                    rank=ctx.rank,
                    metadata=ctx.metadata
                )
                for ctx in response.contexts
            ]
            
            result.append(PredictResponse(
                answer=response.answer,
                confidence=response.confidence,
                contexts=contexts,
                retrieval_time=response.retrieval_time,
                generation_time=response.generation_time,
                total_time=response.total_time,
                metadata=response.metadata
            ))
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing batch prediction request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Verifies that all RAG components are initialized and operational.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Check vector database
        vector_db_status = "connected"
        try:
            doc_count = rag_pipeline.retriever.vector_db.count()
            vector_db_status = f"connected ({doc_count} documents)"
        except Exception as e:
            vector_db_status = f"error: {str(e)}"
        
        # Check generator
        generator_status = "loaded"
        generator_type = type(rag_pipeline.generator).__name__
        
        # Check embedding model
        embedding_model = rag_pipeline.retriever.embedding_manager.model_name
        
        return HealthResponse(
            status="healthy",
            vector_db=vector_db_status,
            generator=f"{generator_type} - {generator_status}",
            embedding_model=embedding_model,
            uptime=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0.0
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    List available models and current configuration.
    """
    if config is None:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    
    return ModelsResponse(
        embedding_models=[
            "paraphrase-multilingual-MiniLM-L12-v2",
            "paraphrase-multilingual-mpnet-base-v2",
            "multilingual-e5-large",
            "bge-m3"
        ],
        generator_types=["mt5", "openai", "ollama"],
        current_embedding_model=config.embedding.model_name,
        current_generator=config.generator.type
    )


@app.get("/languages", response_model=LanguagesResponse)
async def list_languages():
    """
    List supported languages.
    """
    languages = [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ar", "name": "Arabic"},
        {"code": "hi", "name": "Hindi"},
        {"code": "ja", "name": "Japanese"},
        {"code": "ko", "name": "Korean"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "ru", "name": "Russian"},
        {"code": "it", "name": "Italian"},
    ]
    
    return LanguagesResponse(
        languages=languages,
        total_count=len(languages)
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG Cross-Lingual QA API",
        "version": "0.1.0",
        "description": "Retrieval-Augmented Generation system for cross-lingual question answering",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "health": "/health",
            "models": "/models",
            "languages": "/languages",
            "docs": "/docs"
        }
    }


# Store start time
@app.on_event("startup")
async def store_start_time():
    """Store server start time for uptime calculation."""
    app.state.start_time = time.time()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
