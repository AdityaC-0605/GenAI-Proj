"""
Middleware for API authentication and rate limiting.
"""

import time
from typing import Optional, Dict
from collections import defaultdict
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


# API Key Authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyAuth:
    """Simple API key authentication."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize API key authentication.
        
        Args:
            api_keys: Dictionary of API keys to user IDs
        """
        self.api_keys = api_keys or {}
        self.enabled = len(self.api_keys) > 0
    
    async def __call__(self, api_key: Optional[str] = None) -> Optional[str]:
        """
        Validate API key.
        
        Args:
            api_key: API key from header
            
        Returns:
            User ID if valid, None if authentication disabled
            
        Raises:
            HTTPException: If API key is invalid
        """
        if not self.enabled:
            return None
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        user_id = self.api_keys.get(api_key)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return user_id


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm."""
    
    def __init__(self, app, requests_per_minute: int = 100):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per client
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_minute / 60.0
        
        # Token buckets: {client_id: {'tokens': float, 'last_update': float}}
        self.buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'tokens': float(requests_per_minute), 'last_update': time.time()}
        )
        
        logger.info(f"Rate limiting enabled: {requests_per_minute} requests/minute")
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier from request.
        
        Args:
            request: FastAPI request
            
        Returns:
            Client identifier (IP address or API key)
        """
        # Use API key if available, otherwise use IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        
        # Get client IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
    
    def _refill_tokens(self, bucket: Dict[str, float]) -> None:
        """
        Refill tokens based on elapsed time.
        
        Args:
            bucket: Token bucket dictionary
        """
        now = time.time()
        elapsed = now - bucket['last_update']
        
        # Add tokens based on elapsed time
        bucket['tokens'] = min(
            self.requests_per_minute,
            bucket['tokens'] + elapsed * self.requests_per_second
        )
        bucket['last_update'] = now
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/endpoint
            
        Returns:
            Response or rate limit error
        """
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)
        
        # Get client ID
        client_id = self._get_client_id(request)
        
        # Get or create bucket
        bucket = self.buckets[client_id]
        
        # Refill tokens
        self._refill_tokens(bucket)
        
        # Check if request can proceed
        if bucket['tokens'] >= 1.0:
            bucket['tokens'] -= 1.0
            
            # Add rate limit headers
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(int(bucket['tokens']))
            response.headers["X-RateLimit-Reset"] = str(int(bucket['last_update'] + 60))
            
            return response
        else:
            # Rate limit exceeded
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            
            retry_after = int((1.0 - bucket['tokens']) / self.requests_per_second)
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(bucket['last_update'] + 60))
                }
            )


class UsageTracker:
    """Track API usage per client."""
    
    def __init__(self):
        """Initialize usage tracker."""
        # Usage stats: {client_id: {'requests': int, 'errors': int, 'total_time': float}}
        self.stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'requests': 0, 'errors': 0, 'total_time': 0.0}
        )
    
    def record_request(self, client_id: str, duration: float, error: bool = False):
        """
        Record request statistics.
        
        Args:
            client_id: Client identifier
            duration: Request duration in seconds
            error: Whether request resulted in error
        """
        stats = self.stats[client_id]
        stats['requests'] += 1
        stats['total_time'] += duration
        if error:
            stats['errors'] += 1
    
    def get_stats(self, client_id: Optional[str] = None) -> Dict:
        """
        Get usage statistics.
        
        Args:
            client_id: Optional client ID (None for all clients)
            
        Returns:
            Usage statistics
        """
        if client_id:
            return dict(self.stats.get(client_id, {}))
        
        # Aggregate stats
        total_requests = sum(s['requests'] for s in self.stats.values())
        total_errors = sum(s['errors'] for s in self.stats.values())
        total_time = sum(s['total_time'] for s in self.stats.values())
        
        return {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': total_errors / total_requests if total_requests > 0 else 0.0,
            'avg_duration': total_time / total_requests if total_requests > 0 else 0.0,
            'clients': len(self.stats)
        }
