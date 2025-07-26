"""
Memory Manager for efficient large image processing.

Handles tile-based processing, memory pressure monitoring, and
intelligent caching to maintain responsive performance even with
large images and limited memory.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import gc
from collections import OrderedDict
from dataclasses import dataclass, field

from .models import (
    TileInfo, MemoryStats, MemoryPressure, PreviewConfig
)

logger = logging.getLogger(__name__)


class TileManager:
    """
    Manages image tiles for memory-efficient processing.
    
    Automatically splits large images into manageable tiles with
    appropriate overlap for seamless blending.
    """
    
    def __init__(self, config: PreviewConfig):
        self.config = config
        self.tile_cache: Dict[str, TileInfo] = {}
        self.cache_memory_usage = 0
        self._lock = threading.RLock()
        
        logger.info(f"TileManager initialized (tile_size: {config.tile_size}, overlap: {config.overlap_pixels})")
    
    def create_tiles(self, image: np.ndarray, tile_id_prefix: str = "") -> List[TileInfo]:
        """
        Create tiles from a large image.
        
        Args:
            image: Input image array
            tile_id_prefix: Prefix for tile IDs
            
        Returns:
            List of TileInfo objects
        """
        height, width = image.shape[:2]
        tile_size = self.config.tile_size
        overlap = self.config.overlap_pixels
        
        # If image is small enough, return single tile
        if max(height, width) <= tile_size:
            tile_id = f"{tile_id_prefix}single"
            return [TileInfo(
                tile_id=tile_id,
                x=0, y=0,
                width=width, height=height,
                overlap_x=0, overlap_y=0,
                data=image,
                processed=False,
                cached=False
            )]
        
        tiles = []
        tile_index = 0
        
        y = 0
        while y < height:
            x = 0
            while x < width:
                # Calculate actual tile dimensions
                tile_width = min(tile_size, width - x)
                tile_height = min(tile_size, height - y)
                
                # Calculate overlap regions
                overlap_left = overlap if x > 0 else 0
                overlap_top = overlap if y > 0 else 0
                overlap_right = overlap if x + tile_width < width else 0
                overlap_bottom = overlap if y + tile_height < height else 0
                
                # Adjust extraction region to include overlap
                extract_x = max(0, x - overlap_left)
                extract_y = max(0, y - overlap_top)
                extract_width = min(width - extract_x, tile_width + overlap_left + overlap_right)
                extract_height = min(height - extract_y, tile_height + overlap_top + overlap_bottom)
                
                # Extract tile data
                tile_data = image[extract_y:extract_y + extract_height, 
                                extract_x:extract_x + extract_width]
                
                tile_id = f"{tile_id_prefix}tile_{tile_index:04d}"
                tile = TileInfo(
                    tile_id=tile_id,
                    x=x, y=y,
                    width=tile_width, height=tile_height,
                    overlap_x=overlap_left, overlap_y=overlap_top,
                    data=tile_data,
                    processed=False,
                    cached=False
                )
                
                tiles.append(tile)
                tile_index += 1
                
                x += tile_size - overlap
                if x >= width:
                    break
            
            y += tile_size - overlap
            if y >= height:
                break
        
        logger.debug(f"Created {len(tiles)} tiles for {width}x{height} image")
        return tiles
    
    def blend_tiles(self, tiles: List[TileInfo], output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Blend processed tiles back into a single image.
        
        Args:
            tiles: List of processed tiles
            output_shape: (height, width) of output image
            
        Returns:
            Blended output image
        """
        height, width = output_shape
        channels = tiles[0].data.shape[2] if len(tiles[0].data.shape) > 2 else 1
        
        if channels > 1:
            result = np.zeros((height, width, channels), dtype=np.float32)
        else:
            result = np.zeros((height, width), dtype=np.float32)
        
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        for tile in tiles:
            if not tile.processed or tile.data is None:
                logger.warning(f"Skipping unprocessed tile {tile.tile_id}")
                continue
            
            # Create blending weights with smooth transitions at overlaps
            tile_weights = self._create_tile_weights(tile)
            
            # Extract the core region (without overlap)
            core_height, core_width = tile.height, tile.width
            if len(tile.data.shape) > 2:
                core_data = tile.data[tile.overlap_y:tile.overlap_y + core_height,
                                   tile.overlap_x:tile.overlap_x + core_width]
            else:
                core_data = tile.data[tile.overlap_y:tile.overlap_y + core_height,
                                   tile.overlap_x:tile.overlap_x + core_width]
            
            # Blend into result
            y_end = tile.y + core_height
            x_end = tile.x + core_width
            
            if channels > 1:
                result[tile.y:y_end, tile.x:x_end] += core_data * tile_weights[:, :, np.newaxis]
            else:
                result[tile.y:y_end, tile.x:x_end] += core_data * tile_weights
            
            weight_map[tile.y:y_end, tile.x:x_end] += tile_weights
        
        # Normalize by weights
        valid_mask = weight_map > 0
        if channels > 1:
            result[valid_mask] /= weight_map[valid_mask, np.newaxis]
        else:
            result[valid_mask] /= weight_map[valid_mask]
        
        return result
    
    def cache_tile(self, tile: TileInfo) -> bool:
        """
        Cache a processed tile.
        
        Args:
            tile: Tile to cache
            
        Returns:
            True if cached successfully
        """
        with self._lock:
            if tile.data is None:
                return False
            
            # Check memory pressure
            tile_memory = tile.data.nbytes
            max_cache_memory = self.config.max_memory_cache_mb * 1024 * 1024
            
            # Evict old tiles if needed
            while (self.cache_memory_usage + tile_memory > max_cache_memory and 
                   len(self.tile_cache) > 0):
                self._evict_oldest_tile()
            
            # Add to cache
            self.tile_cache[tile.tile_id] = tile
            self.cache_memory_usage += tile_memory
            tile.cached = True
            tile.last_accessed = time.time()
            
            logger.debug(f"Cached tile {tile.tile_id} (cache size: {self.cache_memory_usage / 1024 / 1024:.1f}MB)")
            return True
    
    def get_cached_tile(self, tile_id: str) -> Optional[TileInfo]:
        """Get a cached tile."""
        with self._lock:
            if tile_id in self.tile_cache:
                tile = self.tile_cache[tile_id]
                tile.last_accessed = time.time()
                return tile
            return None
    
    def clear_cache(self):
        """Clear all cached tiles."""
        with self._lock:
            self.tile_cache.clear()
            self.cache_memory_usage = 0
            gc.collect()  # Force garbage collection
        
        logger.debug("Tile cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get tile cache statistics."""
        with self._lock:
            return {
                'cached_tiles': len(self.tile_cache),
                'cache_memory_mb': self.cache_memory_usage / 1024 / 1024,
                'max_cache_mb': self.config.max_memory_cache_mb
            }
    
    def _create_tile_weights(self, tile: TileInfo) -> np.ndarray:
        """Create smooth blending weights for a tile."""
        weights = np.ones((tile.height, tile.width), dtype=np.float32)
        
        # Create smooth transitions at overlap regions
        if tile.overlap_x > 0:
            # Left edge fade
            fade_x = np.linspace(0, 1, tile.overlap_x)
            weights[:, :tile.overlap_x] *= fade_x[np.newaxis, :]
            
            # Right edge fade
            if tile.width > tile.overlap_x:
                weights[:, -tile.overlap_x:] *= fade_x[np.newaxis, ::-1]
        
        if tile.overlap_y > 0:
            # Top edge fade
            fade_y = np.linspace(0, 1, tile.overlap_y)
            weights[:tile.overlap_y, :] *= fade_y[:, np.newaxis]
            
            # Bottom edge fade
            if tile.height > tile.overlap_y:
                weights[-tile.overlap_y:, :] *= fade_y[::-1, np.newaxis]
        
        return weights
    
    def _evict_oldest_tile(self):
        """Evict the least recently accessed tile."""
        if not self.tile_cache:
            return
        
        # Find oldest tile
        oldest_tile_id = min(self.tile_cache.keys(), 
                           key=lambda tid: self.tile_cache[tid].last_accessed)
        
        # Remove from cache
        tile = self.tile_cache.pop(oldest_tile_id)
        self.cache_memory_usage -= tile.data.nbytes if tile.data is not None else 0
        tile.cached = False


class MemoryManager:
    """
    Monitors system memory and manages processing strategies accordingly.
    
    Automatically adapts processing behavior based on available memory,
    switching between full-image and tile-based processing as needed.
    """
    
    def __init__(self, config: PreviewConfig):
        self.config = config
        self.tile_manager = TileManager(config)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stats_history: List[MemoryStats] = []
        self._lock = threading.RLock()
        
        # Current memory state
        self.current_stats: Optional[MemoryStats] = None
        self.pressure_callbacks: Dict[MemoryPressure, List[callable]] = {
            pressure: [] for pressure in MemoryPressure
        }
        
        logger.info("MemoryManager initialized")
    
    def start_monitoring(self, interval: float = 5.0):
        """Start background memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            name="MemoryMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info(f"Memory monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Memory monitoring stopped")
    
    def register_pressure_callback(self, pressure: MemoryPressure, callback: callable):
        """Register callback for memory pressure level."""
        with self._lock:
            self.pressure_callbacks[pressure].append(callback)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        return self._collect_memory_stats()
    
    def should_use_tiles(self, image_shape: Tuple[int, int]) -> bool:
        """
        Determine if tile-based processing should be used.
        
        Args:
            image_shape: (height, width) of image
            
        Returns:
            True if tiles should be used
        """
        height, width = image_shape
        
        # Always use tiles for very large images
        if max(height, width) > 4096:
            return True
        
        # Check current memory pressure
        stats = self.get_memory_stats()
        
        # Estimate memory needed for full processing
        channels = 3  # Assume RGB
        bytes_per_pixel = 4  # float32
        estimated_memory_mb = (height * width * channels * bytes_per_pixel) / 1024 / 1024
        
        # Use tiles if image would use too much memory or pressure is high
        if (estimated_memory_mb > 500 or 
            stats.pressure_level in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]):
            return True
        
        return False
    
    def process_with_memory_management(self, image: np.ndarray, 
                                     processor_func: callable,
                                     *args, **kwargs) -> np.ndarray:
        """
        Process image with automatic memory management.
        
        Args:
            image: Input image
            processor_func: Processing function to apply
            *args, **kwargs: Arguments for processor function
            
        Returns:
            Processed image
        """
        # Check if we should use tile-based processing
        if self.should_use_tiles(image.shape[:2]):
            return self._process_with_tiles(image, processor_func, *args, **kwargs)
        else:
            return self._process_full_image(image, processor_func, *args, **kwargs)
    
    def optimize_for_memory_pressure(self, pressure: MemoryPressure) -> Dict[str, Any]:
        """
        Get optimized settings for current memory pressure.
        
        Args:
            pressure: Current memory pressure level
            
        Returns:
            Dictionary of optimization settings
        """
        optimizations = {
            'use_tiles': False,
            'tile_size': self.config.tile_size,
            'cache_enabled': True,
            'quality_reduction': 0.0,
            'thread_reduction': 0
        }
        
        if pressure == MemoryPressure.MEDIUM:
            optimizations.update({
                'use_tiles': True,
                'tile_size': self.config.tile_size // 2,
                'quality_reduction': 0.1
            })
        elif pressure == MemoryPressure.HIGH:
            optimizations.update({
                'use_tiles': True,
                'tile_size': self.config.tile_size // 4,
                'cache_enabled': False,
                'quality_reduction': 0.2,
                'thread_reduction': 1
            })
        elif pressure == MemoryPressure.CRITICAL:
            optimizations.update({
                'use_tiles': True,
                'tile_size': 256,  # Very small tiles
                'cache_enabled': False,
                'quality_reduction': 0.5,
                'thread_reduction': 2
            })
        
        return optimizations
    
    def cleanup_memory(self):
        """Perform aggressive memory cleanup."""
        logger.info("Performing memory cleanup...")
        
        # Clear tile cache
        self.tile_manager.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Trigger memory cleanup callbacks
        with self._lock:
            for callback in self.pressure_callbacks[MemoryPressure.CRITICAL]:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Memory cleanup callback failed: {e}")
    
    def _monitor_loop(self, interval: float):
        """Background memory monitoring loop."""
        while self._monitoring:
            try:
                stats = self._collect_memory_stats()
                
                with self._lock:
                    self.current_stats = stats
                    
                    # Keep history (last 100 samples)
                    self._stats_history.append(stats)
                    if len(self._stats_history) > 100:
                        self._stats_history.pop(0)
                    
                    # Trigger callbacks for current pressure level
                    for callback in self.pressure_callbacks[stats.pressure_level]:
                        try:
                            callback()
                        except Exception as e:
                            logger.error(f"Memory pressure callback failed: {e}")
                
                # Log warnings for high memory pressure
                if stats.pressure_level == MemoryPressure.HIGH:
                    logger.warning(f"High memory pressure: {stats.usage_percentage:.1f}%")
                elif stats.pressure_level == MemoryPressure.CRITICAL:
                    logger.error(f"Critical memory pressure: {stats.usage_percentage:.1f}%")
                    self.cleanup_memory()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_memory_stats(self) -> MemoryStats:
        """Collect current memory statistics."""
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
                cache_memory_mb=self.tile_manager.cache_memory_usage / 1024 / 1024,
                processing_memory_mb=0  # Would need separate tracking
            )
            
        except ImportError:
            # Fallback if psutil not available
            return MemoryStats(
                total_memory_mb=8192,
                used_memory_mb=4096,
                available_memory_mb=4096,
                pressure_level=MemoryPressure.MEDIUM,
                cache_memory_mb=self.tile_manager.cache_memory_usage / 1024 / 1024,
                processing_memory_mb=0
            )
    
    def _process_with_tiles(self, image: np.ndarray, processor_func: callable,
                           *args, **kwargs) -> np.ndarray:
        """Process image using tile-based approach."""
        logger.debug(f"Processing {image.shape} with tiles")
        
        # Create tiles
        tiles = self.tile_manager.create_tiles(image, "proc_")
        
        # Process each tile
        for tile in tiles:
            try:
                # Apply processing function to tile
                processed_data = processor_func(tile.data, *args, **kwargs)
                tile.data = processed_data
                tile.processed = True
                
                # Cache tile if beneficial
                if len(tiles) > 4:  # Only cache for larger tile sets
                    self.tile_manager.cache_tile(tile)
                
            except Exception as e:
                logger.error(f"Failed to process tile {tile.tile_id}: {e}")
                tile.processed = False
        
        # Blend tiles back together
        result = self.tile_manager.blend_tiles(tiles, image.shape[:2])
        
        return result
    
    def _process_full_image(self, image: np.ndarray, processor_func: callable,
                           *args, **kwargs) -> np.ndarray:
        """Process image as a single unit."""
        logger.debug(f"Processing {image.shape} as full image")
        return processor_func(image, *args, **kwargs)