"""
Proxy Manager for multi-resolution image handling.

Manages the generation, caching, and retrieval of proxy images at different
resolution levels for responsive preview rendering.
"""

import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
import time
import threading
from collections import OrderedDict

from .models import (
    ProxyLevel, ProxyInfo, PreviewConfig, MemoryStats, 
    MemoryPressure, PreviewState
)

logger = logging.getLogger(__name__)


class ProxyCache:
    """
    LRU cache for proxy images with memory pressure handling.
    """
    
    def __init__(self, max_memory_mb: int = 512, max_items: int = 100):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_items = max_items
        self.current_memory = 0
        self.cache: OrderedDict[str, ProxyInfo] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[ProxyInfo]:
        """Get proxy from cache, updating LRU order."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                proxy = self.cache.pop(key)
                self.cache[key] = proxy
                proxy.last_accessed = time.time()
                return proxy
            return None
    
    def put(self, key: str, proxy: ProxyInfo) -> bool:
        """Add proxy to cache, handling eviction if needed."""
        with self._lock:
            # Calculate memory usage
            if proxy.memory_data is not None:
                memory_size = proxy.memory_data.nbytes
            else:
                memory_size = proxy.file_size
            
            # Check if we need to evict items
            while (len(self.cache) >= self.max_items or 
                   self.current_memory + memory_size > self.max_memory_bytes):
                if not self.cache:
                    break
                
                # Remove least recently used item
                old_key, old_proxy = self.cache.popitem(last=False)
                if old_proxy.memory_data is not None:
                    self.current_memory -= old_proxy.memory_data.nbytes
                else:
                    self.current_memory -= old_proxy.file_size
                
                logger.debug(f"Evicted proxy {old_key} from cache")
            
            # Add new item
            self.cache[key] = proxy
            self.current_memory += memory_size
            proxy.last_accessed = time.time()
            
            logger.debug(f"Added proxy {key} to cache (memory: {self.current_memory / 1024 / 1024:.1f}MB)")
            return True
    
    def remove(self, key: str) -> bool:
        """Remove proxy from cache."""
        with self._lock:
            if key in self.cache:
                proxy = self.cache.pop(key)
                if proxy.memory_data is not None:
                    self.current_memory -= proxy.memory_data.nbytes
                else:
                    self.current_memory -= proxy.file_size
                return True
            return False
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self.cache.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        with self._lock:
            return {
                'items': len(self.cache),
                'memory_mb': self.current_memory / 1024 / 1024,
                'memory_usage_percent': (self.current_memory / self.max_memory_bytes) * 100,
                'hit_ratio': getattr(self, '_hit_ratio', 0.0)
            }


class ProxyManager:
    """
    Manages proxy image generation, caching, and retrieval.
    
    Handles multiple resolution levels with intelligent caching and
    automatic generation based on usage patterns.
    """
    
    # Resolution limits for each proxy level (max dimension)
    PROXY_DIMENSIONS = {
        ProxyLevel.THUMBNAIL: 256,
        ProxyLevel.SMALL: 512,
        ProxyLevel.MEDIUM: 1024,
        ProxyLevel.LARGE: 2048,
        ProxyLevel.FULL: None  # Original size
    }
    
    def __init__(self, config: PreviewConfig):
        self.config = config
        self.cache = ProxyCache(
            max_memory_mb=config.max_memory_cache_mb,
            max_items=200
        )
        self.state = PreviewState()
        self._disk_cache_dir = config.cache_directory or Path.home() / '.photosight' / 'proxy_cache'
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ProxyManager initialized with cache dir: {self._disk_cache_dir}")
    
    def get_proxy(self, image_path: Path, recipe_hash: str, 
                  level: ProxyLevel, generate_if_missing: bool = True) -> Optional[ProxyInfo]:
        """
        Get proxy image at specified level.
        
        Args:
            image_path: Path to source image
            recipe_hash: Hash of processing recipe
            level: Desired proxy level
            generate_if_missing: Whether to generate proxy if not cached
            
        Returns:
            ProxyInfo if available, None if not found and not generated
        """
        cache_key = self._get_cache_key(image_path, recipe_hash, level)
        
        # Try memory cache first
        proxy = self.cache.get(cache_key)
        if proxy:
            logger.debug(f"Proxy cache hit: {cache_key}")
            return proxy
        
        # Try disk cache
        proxy = self._load_from_disk(cache_key, level)
        if proxy:
            logger.debug(f"Proxy disk cache hit: {cache_key}")
            # Add to memory cache
            self.cache.put(cache_key, proxy)
            return proxy
        
        # Generate if requested and not found
        if generate_if_missing:
            logger.debug(f"Generating proxy: {cache_key}")
            return self._generate_proxy(image_path, recipe_hash, level)
        
        return None
    
    def generate_proxy_levels(self, image_path: Path, recipe_hash: str, 
                            levels: Optional[List[ProxyLevel]] = None) -> Dict[ProxyLevel, ProxyInfo]:
        """
        Generate multiple proxy levels efficiently.
        
        Args:
            image_path: Path to source image
            recipe_hash: Hash of processing recipe
            levels: Proxy levels to generate (default: config.auto_generate_levels)
            
        Returns:
            Dictionary mapping levels to generated proxies
        """
        if levels is None:
            levels = self.config.auto_generate_levels
        
        results = {}
        
        # Generate in order from largest to smallest for efficiency
        sorted_levels = sorted(levels, key=lambda x: self.PROXY_DIMENSIONS[x] or float('inf'), reverse=True)
        
        base_image = None
        
        for level in sorted_levels:
            try:
                proxy = self._generate_proxy(image_path, recipe_hash, level, base_image)
                if proxy:
                    results[level] = proxy
                    # Use this proxy as base for smaller ones if it has memory data
                    if proxy.memory_data is not None and base_image is None:
                        base_image = proxy.memory_data
                        
            except Exception as e:
                logger.error(f"Failed to generate proxy {level} for {image_path}: {e}")
        
        return results
    
    def invalidate_proxies(self, image_path: Path, recipe_hash: Optional[str] = None):
        """
        Invalidate cached proxies for an image.
        
        Args:
            image_path: Path to source image
            recipe_hash: Specific recipe hash to invalidate (None for all)
        """
        if recipe_hash:
            # Invalidate specific recipe
            for level in ProxyLevel:
                cache_key = self._get_cache_key(image_path, recipe_hash, level)
                self.cache.remove(cache_key)
                self._remove_from_disk(cache_key)
        else:
            # Invalidate all proxies for this image
            # This is more complex as we need to find all cache keys
            # For now, we'll clear the entire cache (could be optimized)
            logger.warning(f"Clearing entire proxy cache due to invalidation of {image_path}")
            self.cache.clear()
    
    def get_optimal_proxy(self, image_path: Path, recipe_hash: str,
                         target_dimensions: Tuple[int, int]) -> Optional[ProxyInfo]:
        """
        Get the optimal proxy for given target dimensions.
        
        Args:
            image_path: Path to source image  
            recipe_hash: Hash of processing recipe
            target_dimensions: Desired (width, height)
            
        Returns:
            Best matching proxy or None
        """
        target_max_dim = max(target_dimensions)
        
        # Find the best proxy level for these dimensions
        best_level = ProxyLevel.FULL
        for level in ProxyLevel:
            level_max_dim = self.PROXY_DIMENSIONS[level]
            if level_max_dim and level_max_dim >= target_max_dim:
                best_level = level
                break
        
        return self.get_proxy(image_path, recipe_hash, best_level)
    
    def pregenerate_proxies(self, image_paths: List[Path], recipe_hash: str, 
                          priority_levels: Optional[List[ProxyLevel]] = None):
        """
        Pregenerate proxies for multiple images (background task).
        
        Args:
            image_paths: List of image paths
            recipe_hash: Processing recipe hash
            priority_levels: Levels to generate first
        """
        if priority_levels is None:
            priority_levels = [ProxyLevel.THUMBNAIL, ProxyLevel.SMALL]
        
        logger.info(f"Pregenerating proxies for {len(image_paths)} images")
        
        for image_path in image_paths:
            try:
                # Generate priority levels first
                for level in priority_levels:
                    self.get_proxy(image_path, recipe_hash, level, generate_if_missing=True)
                
                # Then generate remaining levels if memory allows
                remaining_levels = [l for l in self.config.auto_generate_levels 
                                  if l not in priority_levels]
                
                memory_stats = self._get_memory_stats()
                if memory_stats.pressure_level in [MemoryPressure.LOW, MemoryPressure.MEDIUM]:
                    for level in remaining_levels:
                        self.get_proxy(image_path, recipe_hash, level, generate_if_missing=True)
                
            except Exception as e:
                logger.error(f"Failed to pregenerate proxies for {image_path}: {e}")
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """
        Clean up old cached proxies.
        
        Args:
            max_age_hours: Maximum age of cached items to keep
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # Clean memory cache
        with self.cache._lock:
            to_remove = []
            for key, proxy in self.cache.cache.items():
                if proxy.last_accessed < cutoff_time:
                    to_remove.append(key)
            
            for key in to_remove:
                self.cache.remove(key)
        
        # Clean disk cache
        for cache_file in self._disk_cache_dir.glob('*.proxy'):
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
            except Exception as e:
                logger.error(f"Failed to remove old cache file {cache_file}: {e}")
        
        logger.info(f"Cache cleanup completed, removed {len(to_remove)} items")
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get detailed cache statistics."""
        memory_stats = self.cache.get_stats()
        
        # Count disk cache files
        disk_files = list(self._disk_cache_dir.glob('*.proxy'))
        disk_size_mb = sum(f.stat().st_size for f in disk_files) / 1024 / 1024
        
        return {
            **memory_stats,
            'disk_files': len(disk_files),
            'disk_size_mb': disk_size_mb,
            'cache_directory': str(self._disk_cache_dir)
        }
    
    def _generate_proxy(self, image_path: Path, recipe_hash: str, 
                       level: ProxyLevel, base_image: Optional[np.ndarray] = None) -> Optional[ProxyInfo]:
        """Generate a single proxy image."""
        try:
            # Load base image if not provided
            if base_image is None:
                if not image_path.exists():
                    logger.error(f"Source image not found: {image_path}")
                    return None
                
                base_image = cv2.imread(str(image_path))
                if base_image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    return None
                
                # Convert BGR to RGB
                base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
                base_image = base_image.astype(np.float32) / 255.0
            
            # Calculate target dimensions
            target_dimensions = self._calculate_proxy_dimensions(base_image.shape[:2], level)
            
            # Resize image
            if level == ProxyLevel.FULL:
                proxy_image = base_image.copy()
            else:
                proxy_image = cv2.resize(
                    base_image, 
                    target_dimensions, 
                    interpolation=cv2.INTER_AREA if target_dimensions[0] < base_image.shape[1] else cv2.INTER_CUBIC
                )
            
            # Create proxy info
            proxy = ProxyInfo(
                level=level,
                dimensions=target_dimensions,
                memory_data=proxy_image,
                recipe_hash=recipe_hash,
                file_size=proxy_image.nbytes
            )
            
            # Cache the proxy
            cache_key = self._get_cache_key(image_path, recipe_hash, level)
            self.cache.put(cache_key, proxy)
            
            # Optionally save to disk for larger proxies
            if level in [ProxyLevel.LARGE, ProxyLevel.FULL]:
                self._save_to_disk(cache_key, proxy)
            
            logger.debug(f"Generated proxy {level} for {image_path}: {target_dimensions}")
            return proxy
            
        except Exception as e:
            logger.error(f"Failed to generate proxy {level} for {image_path}: {e}")
            return None
    
    def _calculate_proxy_dimensions(self, original_shape: Tuple[int, int], 
                                  level: ProxyLevel) -> Tuple[int, int]:
        """Calculate proxy dimensions maintaining aspect ratio."""
        height, width = original_shape
        max_dim = self.PROXY_DIMENSIONS[level]
        
        if max_dim is None:
            return (width, height)
        
        # Calculate scaling factor
        scale = min(max_dim / width, max_dim / height)
        
        if scale >= 1.0:
            return (width, height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return (new_width, new_height)
    
    def _get_cache_key(self, image_path: Path, recipe_hash: str, level: ProxyLevel) -> str:
        """Generate cache key for proxy."""
        key_data = f"{image_path}:{recipe_hash}:{level.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _save_to_disk(self, cache_key: str, proxy: ProxyInfo):
        """Save proxy to disk cache."""
        try:
            cache_file = self._disk_cache_dir / f"{cache_key}.proxy"
            
            # Save as pickle for now (could optimize with other formats)
            with open(cache_file, 'wb') as f:
                pickle.dump(proxy, f)
            
            proxy.file_path = cache_file
            
        except Exception as e:
            logger.error(f"Failed to save proxy to disk: {e}")
    
    def _load_from_disk(self, cache_key: str, level: ProxyLevel) -> Optional[ProxyInfo]:
        """Load proxy from disk cache."""
        try:
            cache_file = self._disk_cache_dir / f"{cache_key}.proxy"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                proxy = pickle.load(f)
            
            proxy.file_path = cache_file
            proxy.last_accessed = time.time()
            
            return proxy
            
        except Exception as e:
            logger.error(f"Failed to load proxy from disk: {e}")
            return None
    
    def _remove_from_disk(self, cache_key: str):
        """Remove proxy from disk cache."""
        try:
            cache_file = self._disk_cache_dir / f"{cache_key}.proxy"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.error(f"Failed to remove proxy from disk: {e}")
    
    def _get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics (simplified)."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            # Determine pressure level
            usage_percent = memory.percent / 100
            pressure = MemoryPressure.LOW
            
            for level, threshold in self.config.memory_pressure_thresholds.items():
                if usage_percent >= threshold:
                    pressure = level
            
            return MemoryStats(
                total_memory_mb=memory.total / 1024 / 1024,
                used_memory_mb=memory.used / 1024 / 1024,
                available_memory_mb=memory.available / 1024 / 1024,
                pressure_level=pressure,
                cache_memory_mb=self.cache.current_memory / 1024 / 1024,
                processing_memory_mb=0  # Would need to track this separately
            )
            
        except ImportError:
            # Fallback if psutil not available
            return MemoryStats(
                total_memory_mb=8192,  # Assume 8GB
                used_memory_mb=4096,   # Assume 50% used
                available_memory_mb=4096,
                pressure_level=MemoryPressure.MEDIUM,
                cache_memory_mb=self.cache.current_memory / 1024 / 1024,
                processing_memory_mb=0
            )