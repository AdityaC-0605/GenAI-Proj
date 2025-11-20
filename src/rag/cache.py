"""
Caching utilities for RAG system.

Implements query caching and result caching using Redis or in-memory cache.
"""

import time
import hashlib
import json
from typing import Any, Optional, Dict
from collections import OrderedDict

from .logging_config import LoggerMixin


class LRUCache:
    """Simple LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self.cache:
            # Update existing item
            self.cache.move_to_end(key)
        else:
            # Add new item
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class QueryCache(LoggerMixin):
    """Cache for query results with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600, use_redis: bool = False):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum number of cached queries
            ttl: Time to live in seconds
            use_redis: Whether to use Redis (requires redis-py)
        """
        super().__init__()
        self.max_size = max_size
        self.ttl = ttl
        self.use_redis = use_redis
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                self.redis_client.ping()
                self.logger.info("Connected to Redis for query caching")
            except Exception as e:
                self.logger.warning(f"Redis not available, using in-memory cache: {e}")
                self.use_redis = False
                self.cache = LRUCache(max_size)
        else:
            self.cache = LRUCache(max_size)
    
    def _make_key(self, question: str, **kwargs) -> str:
        """Generate cache key from question and parameters."""
        # Create deterministic key from question and parameters
        key_data = {
            'question': question,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, question: str, **kwargs) -> Optional[Any]:
        """
        Get cached result for question.
        
        Args:
            question: Question text
            **kwargs: Additional parameters (top_k, language, etc.)
            
        Returns:
            Cached result or None
        """
        key = self._make_key(question, **kwargs)
        
        if self.use_redis:
            try:
                cached = self.redis_client.get(f"query:{key}")
                if cached:
                    self.logger.debug(f"Cache hit for question: {question[:50]}...")
                    return json.loads(cached)
            except Exception as e:
                self.logger.error(f"Redis get error: {e}")
                return None
        else:
            result = self.cache.get(key)
            if result:
                # Check TTL
                if time.time() - result['timestamp'] < self.ttl:
                    self.logger.debug(f"Cache hit for question: {question[:50]}...")
                    return result['data']
                else:
                    # Expired
                    self.cache.cache.pop(key, None)
            
            return None
    
    def put(self, question: str, result: Any, **kwargs) -> None:
        """
        Cache result for question.
        
        Args:
            question: Question text
            result: Result to cache
            **kwargs: Additional parameters
        """
        key = self._make_key(question, **kwargs)
        
        if self.use_redis:
            try:
                self.redis_client.setex(
                    f"query:{key}",
                    self.ttl,
                    json.dumps(result)
                )
                self.logger.debug(f"Cached result for question: {question[:50]}...")
            except Exception as e:
                self.logger.error(f"Redis set error: {e}")
        else:
            cached_item = {
                'data': result,
                'timestamp': time.time()
            }
            self.cache.put(key, cached_item)
            self.logger.debug(f"Cached result for question: {question[:50]}...")
    
    def clear(self) -> None:
        """Clear cache."""
        if self.use_redis:
            try:
                # Clear all query keys
                for key in self.redis_client.scan_iter("query:*"):
                    self.redis_client.delete(key)
                self.logger.info("Cleared Redis query cache")
            except Exception as e:
                self.logger.error(f"Redis clear error: {e}")
        else:
            self.cache.clear()
            self.logger.info("Cleared in-memory query cache")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.use_redis:
            try:
                # Count query keys
                count = sum(1 for _ in self.redis_client.scan_iter("query:*"))
                return {
                    'backend': 'redis',
                    'size': count,
                    'ttl': self.ttl
                }
            except Exception as e:
                self.logger.error(f"Redis stats error: {e}")
                return {'backend': 'redis', 'error': str(e)}
        else:
            stats = self.cache.stats()
            stats['backend'] = 'memory'
            stats['ttl'] = self.ttl
            return stats
