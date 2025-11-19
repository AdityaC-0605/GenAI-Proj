"""Middleware for API authentication and rate limiting."""

import logging
import time
from typing import Optional, Dict
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# API Key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Simple in-memory API key store (use database in production)
VALID_API_KEYS = {
    "demo-key-12345": {"name": "Demo User", "rate_limit": 100},
    "test-key-67890": {"name": "Test User", "rate_limit": 50}
}


class APIKeyAuth:
    """API key authentication handler."""
    
    def __init__(self, required: bool = False):
        """
        Initialize API key auth.
        
        Args:
            required: Whether API key is required
        """
        self.required = required
    
    async def __call__(self, api_key: Optional[str] = None) -> Optional[Dict]:
        """
        Validate API key.
        
        Args:
            api_key: API key from header
            
        Returns:
            User info if valid, None if not required
            
        Raises:
            HTTPException: If API key is invalid or missing when required
        """
        if not self.required and api_key is None:
            return None
        
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is required"
            )
        
        user_info = VALID_API_KEYS.get(api_key)
        
        if user_info is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return user_info


class TokenBucket:
    """Token bucket for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        
        self.last_refill = now
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get time to wait until tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Wait time in seconds
        """
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm."""
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (default: requests_per_minute)
        """
        super().__init__(app)
        
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        
        # Token buckets per client (IP or API key)
        self.buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=self.burst_size,
                refill_rate=self.requests_per_minute / 60.0
            )
        )
        
        # Cleanup old buckets periodically
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)
        
        # Get client identifier (API key or IP)
        client_id = self._get_client_id(request)
        
        # Get or create token bucket
        bucket = self.buckets[client_id]
        
        # Try to consume token
        if not bucket.consume():
            wait_time = bucket.get_wait_time()
            
            logger.warning(f"Rate limit exceeded for {client_id}")
            
            return HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {wait_time:.1f} seconds",
                headers={"Retry-After": str(int(wait_time) + 1)}
            )
        
        # Cleanup old buckets periodically
        self._cleanup_buckets()
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Client identifier (API key or IP)
        """
        # Try to get API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _cleanup_buckets(self):
        """Remove old token buckets to free memory."""
        now = time.time()
        
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        # Remove buckets that haven't been used recently
        inactive_threshold = now - 600  # 10 minutes
        
        inactive_clients = [
            client_id
            for client_id, bucket in self.buckets.items()
            if bucket.last_refill < inactive_threshold
        ]
        
        for client_id in inactive_clients:
            del self.buckets[client_id]
        
        if inactive_clients:
            logger.info(f"Cleaned up {len(inactive_clients)} inactive rate limit buckets")
        
        self.last_cleanup = now


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests."""
    
    async def dispatch(self, request: Request, call_next):
        """
        Log request and response.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        duration = (time.time() - start_time) * 1000
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code} duration={duration:.2f}ms"
        )
        
        return response
