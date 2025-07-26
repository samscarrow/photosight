"""
Data models for the PhotoSight preview system.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any, Union
from enum import Enum
import numpy as np
from pathlib import Path
import time
import threading


class ProxyLevel(Enum):
    """Available proxy resolution levels."""
    THUMBNAIL = "thumbnail"    # 256px max dimension
    SMALL = "small"           # 512px max dimension  
    MEDIUM = "medium"         # 1024px max dimension
    LARGE = "large"           # 2048px max dimension
    FULL = "full"             # Original resolution


class ProcessingMode(Enum):
    """Processing quality modes for different use cases."""
    PREVIEW = "preview"       # Fast, lower quality for real-time feedback
    BALANCED = "balanced"     # Good quality/speed balance
    EXPORT = "export"         # Full quality for final output


class TaskPriority(Enum):
    """Task priority levels for threading system."""
    URGENT = 1               # UI blocking operations
    HIGH = 2                 # User-initiated actions
    NORMAL = 3               # Background processing
    LOW = 4                  # Precomputation/caching


class MemoryPressure(Enum):
    """System memory pressure levels."""
    LOW = "low"              # < 60% memory used
    MEDIUM = "medium"        # 60-80% memory used
    HIGH = "high"            # 80-90% memory used
    CRITICAL = "critical"    # > 90% memory used


@dataclass
class ProxyInfo:
    """Information about a proxy image."""
    level: ProxyLevel
    dimensions: Tuple[int, int]      # (width, height)
    file_path: Optional[Path] = None # Cached file location
    memory_data: Optional[np.ndarray] = None  # In-memory data
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    file_size: int = 0               # Size in bytes
    recipe_hash: Optional[str] = None # Hash of processing recipe


@dataclass
class PreviewConfig:
    """Configuration for preview processing."""
    # Default proxy levels to generate
    auto_generate_levels: List[ProxyLevel] = field(default_factory=lambda: [
        ProxyLevel.THUMBNAIL,
        ProxyLevel.SMALL,
        ProxyLevel.MEDIUM
    ])
    
    # Quality settings per level
    preview_quality: Dict[ProxyLevel, ProcessingMode] = field(default_factory=lambda: {
        ProxyLevel.THUMBNAIL: ProcessingMode.PREVIEW,
        ProxyLevel.SMALL: ProcessingMode.PREVIEW,
        ProxyLevel.MEDIUM: ProcessingMode.BALANCED,
        ProxyLevel.LARGE: ProcessingMode.BALANCED,
        ProxyLevel.FULL: ProcessingMode.EXPORT
    })
    
    # Cache settings
    max_memory_cache_mb: int = 512   # Maximum memory cache size
    max_disk_cache_mb: int = 2048    # Maximum disk cache size
    cache_directory: Optional[Path] = None  # Disk cache location
    
    # Performance settings
    max_worker_threads: int = 4      # Processing thread pool size
    tile_size: int = 512            # Tile size for large images
    overlap_pixels: int = 32        # Tile overlap for seamless blending
    
    # Memory pressure thresholds
    memory_pressure_thresholds: Dict[MemoryPressure, float] = field(default_factory=lambda: {
        MemoryPressure.LOW: 0.6,
        MemoryPressure.MEDIUM: 0.8,
        MemoryPressure.HIGH: 0.9,
        MemoryPressure.CRITICAL: 0.95
    })
    
    # Progressive enhancement settings
    progressive_update_interval: float = 0.1  # Seconds between updates
    max_progressive_steps: int = 3           # Maximum enhancement steps


@dataclass
class ProcessingTask:
    """Represents a processing task in the queue."""
    task_id: str
    source_path: Path
    recipe: Any  # ProcessingRecipe (avoiding circular import)
    target_level: ProxyLevel
    priority: TaskPriority
    callback: Optional[callable] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[Exception] = None
    result: Optional[np.ndarray] = None
    progress: float = 0.0
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.completed_at is not None
    
    @property
    def is_failed(self) -> bool:
        """Check if task failed."""
        return self.error is not None
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class TileInfo:
    """Information about an image tile."""
    tile_id: str
    x: int                          # Top-left x coordinate
    y: int                          # Top-left y coordinate
    width: int                      # Tile width
    height: int                     # Tile height
    overlap_x: int = 0             # Overlap on x-axis
    overlap_y: int = 0             # Overlap on y-axis
    data: Optional[np.ndarray] = None  # Tile image data
    processed: bool = False         # Whether tile has been processed
    cached: bool = False           # Whether tile is cached
    last_accessed: float = field(default_factory=time.time)


@dataclass
class MemoryStats:
    """System memory usage statistics."""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    pressure_level: MemoryPressure
    cache_memory_mb: float
    processing_memory_mb: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def usage_percentage(self) -> float:
        """Get memory usage as percentage."""
        return (self.used_memory_mb / self.total_memory_mb) * 100


@dataclass
class PreviewState:
    """Current state of preview system."""
    current_image: Optional[Path] = None
    current_recipe_hash: Optional[str] = None
    available_proxies: Dict[ProxyLevel, ProxyInfo] = field(default_factory=dict)
    active_tasks: List[ProcessingTask] = field(default_factory=list)
    memory_stats: Optional[MemoryStats] = None
    processing_mode: ProcessingMode = ProcessingMode.PREVIEW
    
    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    def get_best_proxy(self, target_level: ProxyLevel) -> Optional[ProxyInfo]:
        """Get the best available proxy for the target level."""
        with self._lock:
            # First try exact match
            if target_level in self.available_proxies:
                proxy = self.available_proxies[target_level]
                proxy.last_accessed = time.time()
                return proxy
            
            # Find closest higher resolution proxy
            level_order = [
                ProxyLevel.THUMBNAIL,
                ProxyLevel.SMALL, 
                ProxyLevel.MEDIUM,
                ProxyLevel.LARGE,
                ProxyLevel.FULL
            ]
            
            target_idx = level_order.index(target_level)
            
            # Look for higher resolution proxies
            for i in range(target_idx + 1, len(level_order)):
                level = level_order[i]
                if level in self.available_proxies:
                    proxy = self.available_proxies[level]
                    proxy.last_accessed = time.time()
                    return proxy
            
            # Fall back to lower resolution if needed
            for i in range(target_idx - 1, -1, -1):
                level = level_order[i]
                if level in self.available_proxies:
                    proxy = self.available_proxies[level]
                    proxy.last_accessed = time.time()
                    return proxy
            
            return None
    
    def add_proxy(self, level: ProxyLevel, proxy: ProxyInfo):
        """Add a proxy to the available proxies."""
        with self._lock:
            self.available_proxies[level] = proxy
    
    def remove_proxy(self, level: ProxyLevel):
        """Remove a proxy from available proxies."""
        with self._lock:
            if level in self.available_proxies:
                del self.available_proxies[level]


@dataclass
class ProgressiveState:
    """State for progressive enhancement rendering."""
    current_step: int = 0
    max_steps: int = 3
    last_update: float = field(default_factory=time.time)
    enhancement_queue: List[ProcessingTask] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        """Check if progressive enhancement is complete."""
        return self.current_step >= self.max_steps
    
    def should_update(self, interval: float) -> bool:
        """Check if enough time has passed for next update."""
        return time.time() - self.last_update >= interval