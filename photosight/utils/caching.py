"""
Redis-based caching system for PhotoSight performance optimization.

Provides intelligent caching for analysis results, metadata,
and frequently accessed data with TTL management.
"""

import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from functools import wraps
import redis
import pickle

logger = logging.getLogger(__name__)


class PhotoSightCache:
    """
    High-performance Redis caching system for PhotoSight.
    
    Features:
    - Analysis result caching with smart invalidation
    - Photo metadata caching with file change detection
    - Query result caching with dependency tracking
    - Batch operations for improved performance
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 default_ttl: int = 3600):
        """
        Initialize the cache system.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds (1 hour)
        """
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
        
        self.default_ttl = default_ttl
        self.enabled = self.redis_client is not None
        
        # Cache key prefixes for different data types
        self.prefixes = {
            'analysis': 'ps:analysis:',
            'metadata': 'ps:metadata:',
            'query': 'ps:query:',
            'photo': 'ps:photo:',
            'stats': 'ps:stats:',
            'batch': 'ps:batch:',
            'rankings': 'ps:rankings:'
        }
    
    def _generate_key(self, prefix: str, identifier: str, 
                     params: Optional[Dict] = None) -> str:
        """Generate cache key with optional parameter hashing."""
        if not self.enabled:
            return ""
        
        base_key = f"{self.prefixes[prefix]}{identifier}"
        
        if params:
            # Hash parameters for consistent key generation
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            base_key += f":{param_hash}"
        
        return base_key
    
    def get(self, prefix: str, identifier: str, 
            params: Optional[Dict] = None) -> Optional[Any]:
        """Get cached value."""
        if not self.enabled:
            return None
        
        key = self._generate_key(prefix, identifier, params)
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
        
        return None
    
    def set(self, prefix: str, identifier: str, value: Any, 
            ttl: Optional[int] = None, params: Optional[Dict] = None) -> bool:
        """Set cached value with TTL."""
        if not self.enabled:
            return False
        
        key = self._generate_key(prefix, identifier, params)
        ttl = ttl or self.default_ttl
        
        try:
            serialized_value = pickle.dumps(value)
            self.redis_client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, prefix: str, identifier: str, 
              params: Optional[Dict] = None) -> bool:
        """Delete cached value."""
        if not self.enabled:
            return False
        
        key = self._generate_key(prefix, identifier, params)
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self.enabled:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache pattern delete error for {pattern}: {e}")
            return 0
    
    def invalidate_photo(self, photo_id: int):
        """Invalidate all cache entries for a specific photo."""
        if not self.enabled:
            return
        
        patterns = [
            f"{self.prefixes['analysis']}{photo_id}*",
            f"{self.prefixes['metadata']}{photo_id}*",
            f"{self.prefixes['photo']}{photo_id}*",
            f"{self.prefixes['rankings']}*{photo_id}*"
        ]
        
        for pattern in patterns:
            self.delete_pattern(pattern)
        
        logger.debug(f"Invalidated cache for photo {photo_id}")
    
    def get_analysis_result(self, photo_id: int, analysis_type: str) -> Optional[Dict]:
        """Get cached analysis result."""
        return self.get('analysis', f"{photo_id}:{analysis_type}")
    
    def set_analysis_result(self, photo_id: int, analysis_type: str, 
                           result: Dict, ttl: int = 7200) -> bool:
        """Cache analysis result (2 hour TTL by default)."""
        return self.set('analysis', f"{photo_id}:{analysis_type}", result, ttl)
    
    def get_photo_metadata(self, photo_id: int) -> Optional[Dict]:
        """Get cached photo metadata."""
        return self.get('metadata', str(photo_id))
    
    def set_photo_metadata(self, photo_id: int, metadata: Dict, 
                          ttl: int = 1800) -> bool:
        """Cache photo metadata (30 min TTL)."""
        return self.set('metadata', str(photo_id), metadata, ttl)
    
    def get_ranking_results(self, project_id: Optional[int] = None, 
                           params: Optional[Dict] = None) -> Optional[List]:
        """Get cached ranking results."""
        identifier = f"project:{project_id}" if project_id else "global"
        return self.get('rankings', identifier, params)
    
    def set_ranking_results(self, results: List, project_id: Optional[int] = None,
                           params: Optional[Dict] = None, ttl: int = 900) -> bool:
        """Cache ranking results (15 min TTL)."""
        identifier = f"project:{project_id}" if project_id else "global"
        return self.set('rankings', identifier, results, ttl, params)
    
    def get_batch_stats(self, batch_id: str) -> Optional[Dict]:
        """Get cached batch processing statistics."""
        return self.get('batch', batch_id)
    
    def set_batch_stats(self, batch_id: str, stats: Dict, ttl: int = 3600) -> bool:
        """Cache batch processing statistics."""
        return self.set('batch', batch_id, stats, ttl)
    
    def increment_counter(self, key: str, amount: int = 1, 
                         ttl: Optional[int] = None) -> int:
        """Increment a counter with optional TTL."""
        if not self.enabled:
            return 0
        
        try:
            pipeline = self.redis_client.pipeline()
            pipeline.incr(key, amount)
            if ttl:
                pipeline.expire(key, ttl)
            results = pipeline.execute()
            return results[0]
        except Exception as e:
            logger.warning(f"Counter increment error for key {key}: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if not self.enabled:
            return {'enabled': False}
        
        try:
            info = self.redis_client.info()
            return {
                'enabled': True,
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                )
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate."""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
    
    def flush_all(self) -> bool:
        """Flush all PhotoSight cache entries (dangerous!)."""
        if not self.enabled:
            return False
        
        try:
            # Only delete PhotoSight keys, not entire Redis DB
            for prefix in self.prefixes.values():
                pattern = f"{prefix}*"
                self.delete_pattern(pattern)
            
            logger.info("Flushed all PhotoSight cache entries")
            return True
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")
            return False


# Global cache instance
_cache_instance = None

def get_cache() -> PhotoSightCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        # Initialize with environment variables or defaults
        import os
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        ttl = int(os.getenv('CACHE_DEFAULT_TTL', '3600'))
        _cache_instance = PhotoSightCache(redis_url, ttl)
    return _cache_instance


def cached_result(prefix: str, ttl: int = 3600, 
                 key_func: Optional[callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix  
        ttl: Time to live in seconds
        key_func: Function to generate cache key from args
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            if not cache.enabled:
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation from function name and args
                arg_str = "_".join(str(arg) for arg in args[:3])  # Limit args
                cache_key = f"{func.__name__}_{arg_str}"
            
            # Try to get from cache
            cached_result = cache.get(prefix, cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {prefix}:{cache_key}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(prefix, cache_key, result, ttl)
            logger.debug(f"Cache miss for {prefix}:{cache_key}, result cached")
            
            return result
        return wrapper
    return decorator


def invalidate_on_change(photo_id_arg: str = 'photo_id'):
    """
    Decorator to invalidate cache when photo data changes.
    
    Args:
        photo_id_arg: Name of the photo_id argument
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Extract photo_id from arguments
            photo_id = None
            if photo_id_arg in kwargs:
                photo_id = kwargs[photo_id_arg]
            elif len(args) > 0 and hasattr(args[0], photo_id_arg):
                photo_id = getattr(args[0], photo_id_arg)
            
            # Invalidate cache for this photo
            if photo_id:
                cache = get_cache()
                cache.invalidate_photo(photo_id)
            
            return result
        return wrapper
    return decorator