"""
PhotoSight Interactive Preview System - Main Integration Layer

Provides a unified interface for responsive photo editing with:
- Multi-resolution proxy generation
- Progressive enhancement
- Memory-efficient processing
- Background threading
- Smart caching
"""

import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import threading

from .models import (
    ProxyLevel, ProcessingMode, TaskPriority, PreviewConfig,
    ProcessingTask, PreviewState, ProgressiveState
)
from .proxy_manager import ProxyManager
from .preview_processor import PreviewProcessor
from .threading_manager import ThreadingManager, ProgressiveProcessor
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class PhotoSightPreviewSystem:
    """
    Main interface for the PhotoSight interactive preview system.
    
    Provides high-level methods for:
    - Loading and processing images with immediate preview
    - Progressive quality enhancement
    - Responsive parameter adjustment
    - Memory-efficient large image handling
    """
    
    def __init__(self, config: Optional[PreviewConfig] = None):
        # Use default config if none provided
        self.config = config or PreviewConfig()
        
        # Initialize subsystems
        self.proxy_manager = ProxyManager(self.config)
        self.preview_processor = PreviewProcessor(self.config)
        self.threading_manager = ThreadingManager(self.config)
        self.memory_manager = MemoryManager(self.config)
        self.progressive_processor = ProgressiveProcessor(self.threading_manager, self.config)
        
        # System state
        self.state = PreviewState()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._shutdown = False
        
        # Start background services
        self.memory_manager.start_monitoring()
        
        logger.info("PhotoSight Preview System initialized")
    
    def load_image_for_editing(self, image_path: Path, 
                              recipe: Any,  # ProcessingRecipe
                              preview_callback: Optional[Callable[[np.ndarray], None]] = None,
                              level: ProxyLevel = ProxyLevel.MEDIUM) -> str:
        """
        Load an image for interactive editing.
        
        Args:
            image_path: Path to the image file
            recipe: Processing recipe to apply
            preview_callback: Callback for preview updates
            level: Initial preview level
            
        Returns:
            Session ID for this editing session
        """
        session_id = f"edit_{int(time.time() * 1000)}"
        
        # Calculate recipe hash for caching
        recipe_hash = self._calculate_recipe_hash(recipe)
        
        # Store session info
        self.active_sessions[session_id] = {
            'image_path': image_path,
            'recipe': recipe,
            'recipe_hash': recipe_hash,
            'callback': preview_callback,
            'current_level': level,
            'start_time': time.time()
        }
        
        # Update system state
        with self.state._lock:
            self.state.current_image = image_path
            self.state.current_recipe_hash = recipe_hash
        
        # Start progressive processing if callback provided
        if preview_callback:
            progressive_session = self.progressive_processor.start_progressive_processing(
                str(image_path), recipe, preview_callback
            )
            self.active_sessions[session_id]['progressive_session'] = progressive_session
        else:
            # Just generate the requested level
            self._submit_preview_task(session_id, image_path, recipe, level)
        
        logger.info(f"Started editing session {session_id} for {image_path}")
        return session_id
    
    def update_recipe(self, session_id: str, new_recipe: Any,
                     immediate_preview: bool = True) -> bool:
        """
        Update processing recipe for an active session.
        
        Args:
            session_id: Active editing session ID
            new_recipe: Updated processing recipe
            immediate_preview: Whether to immediately update preview
            
        Returns:
            True if update was successful
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return False
        
        session = self.active_sessions[session_id]
        old_recipe_hash = session['recipe_hash']
        new_recipe_hash = self._calculate_recipe_hash(new_recipe)
        
        # Update session
        session['recipe'] = new_recipe
        session['recipe_hash'] = new_recipe_hash
        
        # If recipe changed significantly, invalidate old proxies
        if old_recipe_hash != new_recipe_hash:
            self.proxy_manager.invalidate_proxies(session['image_path'], old_recipe_hash)
        
        # Update preview if requested
        if immediate_preview and session.get('callback'):
            # Cancel old progressive processing
            if 'progressive_session' in session:
                self.progressive_processor.cancel_progressive_processing(
                    session['progressive_session']
                )
            
            # Start new progressive processing
            progressive_session = self.progressive_processor.start_progressive_processing(
                str(session['image_path']), new_recipe, session['callback']
            )
            session['progressive_session'] = progressive_session
        
        logger.debug(f"Updated recipe for session {session_id}")
        return True
    
    def get_preview(self, session_id: str, level: ProxyLevel,
                   callback: Optional[Callable[[np.ndarray], None]] = None) -> Optional[np.ndarray]:
        """
        Get preview at specific quality level.
        
        Args:
            session_id: Active editing session ID
            level: Desired preview level
            callback: Optional callback for async delivery
            
        Returns:
            Preview image if immediately available, None if processing
        """
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        image_path = session['image_path']
        recipe = session['recipe']
        recipe_hash = session['recipe_hash']
        
        # Try to get cached proxy first
        proxy = self.proxy_manager.get_proxy(
            image_path, recipe_hash, level, generate_if_missing=False
        )
        
        if proxy and proxy.memory_data is not None:
            logger.debug(f"Returning cached preview for session {session_id}, level {level.value}")
            if callback:
                callback(proxy.memory_data)
            return proxy.memory_data
        
        # Submit background task if not available
        if callback:
            self._submit_preview_task(session_id, image_path, recipe, level, callback)
        
        return None
    
    def pregenerate_proxies(self, image_paths: List[Path], recipe: Any,
                          priority_levels: Optional[List[ProxyLevel]] = None):
        """
        Pregenerate proxies for a batch of images.
        
        Args:
            image_paths: List of image paths
            recipe: Processing recipe
            priority_levels: Levels to generate first
        """
        recipe_hash = self._calculate_recipe_hash(recipe)
        
        # Submit as low-priority background tasks
        for image_path in image_paths:
            for level in (priority_levels or [ProxyLevel.THUMBNAIL, ProxyLevel.SMALL]):
                task = ProcessingTask(
                    task_id=f"pregen_{image_path.stem}_{level.value}_{int(time.time())}",
                    source_path=image_path,
                    recipe=recipe,
                    target_level=level,
                    priority=TaskPriority.LOW
                )
                
                self.threading_manager.submit_task(task)
        
        logger.info(f"Submitted pregeneration tasks for {len(image_paths)} images")
    
    def get_optimal_preview_level(self, target_size: Tuple[int, int]) -> ProxyLevel:
        """
        Get optimal preview level for target display size.
        
        Args:
            target_size: (width, height) of target display
            
        Returns:
            Recommended proxy level
        """
        max_dim = max(target_size)
        
        if max_dim <= 256:
            return ProxyLevel.THUMBNAIL
        elif max_dim <= 512:
            return ProxyLevel.SMALL
        elif max_dim <= 1024:
            return ProxyLevel.MEDIUM
        elif max_dim <= 2048:
            return ProxyLevel.LARGE
        else:
            return ProxyLevel.FULL
    
    def get_processing_estimate(self, image_path: Path, recipe: Any,
                               level: ProxyLevel) -> float:
        """
        Estimate processing time for given parameters.
        
        Args:
            image_path: Path to image file
            recipe: Processing recipe
            level: Target proxy level
            
        Returns:
            Estimated processing time in seconds
        """
        # Get image dimensions (simplified - would normally read from EXIF)
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is not None:
                height, width = img.shape[:2]
                
                # Adjust for proxy level
                max_dim = self.proxy_manager.PROXY_DIMENSIONS[level]
                if max_dim:
                    scale = min(max_dim / width, max_dim / height, 1.0)
                    height = int(height * scale)
                    width = int(width * scale)
                
                # Get mode for this level
                mode = self.config.preview_quality[level]
                
                return self.preview_processor.get_processing_estimate(
                    (height, width), recipe, mode
                )
        except Exception as e:
            logger.error(f"Failed to estimate processing time: {e}")
        
        # Fallback estimate based on level
        estimates = {
            ProxyLevel.THUMBNAIL: 0.1,
            ProxyLevel.SMALL: 0.2,
            ProxyLevel.MEDIUM: 0.5,
            ProxyLevel.LARGE: 1.0,
            ProxyLevel.FULL: 2.0
        }
        return estimates.get(level, 1.0)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'proxy_cache': self.proxy_manager.get_cache_stats(),
            'processing_queue': self.threading_manager.get_queue_stats(),
            'memory': self.memory_manager.get_memory_stats().__dict__,
            'tile_cache': self.memory_manager.tile_manager.get_cache_stats(),
            'active_sessions': len(self.active_sessions),
            'uptime': time.time() - getattr(self, '_start_time', time.time())
        }
    
    def cleanup_session(self, session_id: str):
        """Clean up an editing session."""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Cancel progressive processing if active
        if 'progressive_session' in session:
            self.progressive_processor.cancel_progressive_processing(
                session['progressive_session']
            )
        
        # Clean up session data
        del self.active_sessions[session_id]
        
        logger.debug(f"Cleaned up session {session_id}")
    
    def cleanup_old_sessions(self, max_age_hours: int = 1):
        """Clean up old inactive sessions."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        to_cleanup = []
        for session_id, session in self.active_sessions.items():
            if session['start_time'] < cutoff_time:
                to_cleanup.append(session_id)
        
        for session_id in to_cleanup:
            self.cleanup_session(session_id)
        
        logger.info(f"Cleaned up {len(to_cleanup)} old sessions")
    
    def shutdown(self):
        """Shutdown the preview system."""
        if self._shutdown:
            return
        
        logger.info("Shutting down PhotoSight Preview System...")
        self._shutdown = True
        
        # Cancel all active sessions
        for session_id in list(self.active_sessions.keys()):
            self.cleanup_session(session_id)
        
        # Shutdown subsystems
        self.threading_manager.shutdown()
        self.memory_manager.stop_monitoring()
        
        # Cleanup caches
        self.proxy_manager.cache.clear()
        self.memory_manager.tile_manager.clear_cache()
        
        logger.info("PhotoSight Preview System shutdown complete")
    
    def _submit_preview_task(self, session_id: str, image_path: Path,
                           recipe: Any, level: ProxyLevel,
                           callback: Optional[Callable] = None):
        """Submit a preview processing task."""
        task = ProcessingTask(
            task_id=f"{session_id}_{level.value}_{int(time.time() * 1000)}",
            source_path=image_path,
            recipe=recipe,
            target_level=level,
            priority=TaskPriority.HIGH,
            callback=callback
        )
        
        self.threading_manager.submit_task(task)
    
    def _calculate_recipe_hash(self, recipe: Any) -> str:
        """Calculate hash of processing recipe for caching."""
        try:
            # Convert recipe to string representation
            if hasattr(recipe, 'to_json'):
                recipe_str = recipe.to_json()
            else:
                recipe_str = str(recipe)
            
            return hashlib.md5(recipe_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Failed to calculate recipe hash: {e}")
            return f"fallback_{int(time.time())}"
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()