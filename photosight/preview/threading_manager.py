"""
Threading Manager for responsive photo processing.

Handles background processing, task queuing, and thread pool management
to ensure UI responsiveness while processing images.
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from queue import PriorityQueue, Empty
from typing import Dict, List, Optional, Callable, Any
import uuid
import numpy as np
from dataclasses import dataclass, field

from .models import (
    ProcessingTask, TaskPriority, ProcessingMode, ProxyLevel, 
    PreviewConfig, ProgressiveState
)

logger = logging.getLogger(__name__)


@dataclass
class PriorityTask:
    """Wrapper for tasks in priority queue."""
    priority: int
    timestamp: float
    task: ProcessingTask
    
    def __lt__(self, other):
        """Compare by priority first, then timestamp."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class ThreadingManager:
    """
    Manages background processing threads with priority queuing.
    
    Provides responsive processing by managing different types of tasks:
    - URGENT: UI blocking operations (highest priority)
    - HIGH: User-initiated actions (previews, exports)
    - NORMAL: Background processing (proxy generation)
    - LOW: Precomputation and caching
    """
    
    def __init__(self, config: PreviewConfig):
        self.config = config
        self.max_workers = config.max_worker_threads
        
        # Task management
        self.task_queue = PriorityQueue()
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingTask] = {}
        self.task_futures: Dict[str, Future] = {}
        
        # Thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="PhotoSight-Worker"
        )
        
        # Shutdown management
        self._shutdown = False
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"ThreadingManager initialized with {self.max_workers} workers")
    
    def submit_task(self, task: ProcessingTask, 
                   callback: Optional[Callable] = None) -> str:
        """
        Submit a processing task for background execution.
        
        Args:
            task: Processing task to execute
            callback: Optional callback when task completes
            
        Returns:
            Task ID for tracking
        """
        if self._shutdown:
            raise RuntimeError("ThreadingManager is shutting down")
        
        with self._lock:
            # Set callback if provided
            if callback:
                task.callback = callback
            
            # Add to queue with priority
            priority_value = task.priority.value
            priority_task = PriorityTask(
                priority=priority_value,
                timestamp=time.time(),
                task=task
            )
            
            self.task_queue.put(priority_task)
            self.active_tasks[task.task_id] = task
            self.stats['tasks_submitted'] += 1
            
            logger.debug(f"Submitted task {task.task_id} with priority {task.priority.name}")
            
            # Start processing if not already running
            self._ensure_workers_running()
            
            return task.task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled, False if not found or already completed
        """
        with self._lock:
            # Check if task is in queue (not yet started)
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.started_at is None:
                    # Task is still in queue, mark as cancelled
                    task.error = Exception("Task cancelled by user")
                    task.completed_at = time.time()
                    self._move_to_completed(task_id)
                    logger.debug(f"Cancelled queued task {task_id}")
                    return True
            
            # Check if task has a future (running)
            if task_id in self.task_futures:
                future = self.task_futures[task_id]
                if future.cancel():
                    # Successfully cancelled
                    if task_id in self.active_tasks:
                        task = self.active_tasks[task_id]
                        task.error = Exception("Task cancelled by user")
                        task.completed_at = time.time()
                        self._move_to_completed(task_id)
                    logger.debug(f"Cancelled running task {task_id}")
                    return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Get current status of a task."""
        with self._lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue and processing statistics."""
        with self._lock:
            # Count tasks by priority in queue
            priority_counts = {p.name: 0 for p in TaskPriority}
            
            # This is approximate since we can't easily iterate PriorityQueue
            # In practice, you might want to use a different queue implementation
            queue_size = self.task_queue.qsize()
            
            active_count = len(self.active_tasks)
            completed_count = len(self.completed_tasks)
            
            return {
                'queue_size': queue_size,
                'active_tasks': active_count,
                'completed_tasks': completed_count,
                'priority_counts': priority_counts,
                'worker_threads': self.max_workers,
                **self.stats
            }
    
    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> bool:
        """
        Wait for specific tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                incomplete_tasks = []
                for task_id in task_ids:
                    task = self.get_task_status(task_id)
                    if task and not task.is_completed:
                        incomplete_tasks.append(task_id)
                
                if not incomplete_tasks:
                    return True
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for tasks: {incomplete_tasks}")
                return False
            
            time.sleep(0.1)
    
    def clear_completed_tasks(self, max_age_seconds: int = 3600):
        """Clear old completed tasks to free memory."""
        cutoff_time = time.time() - max_age_seconds
        
        with self._lock:
            to_remove = []
            for task_id, task in self.completed_tasks.items():
                if task.completed_at and task.completed_at < cutoff_time:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.completed_tasks[task_id]
                if task_id in self.task_futures:
                    del self.task_futures[task_id]
            
            logger.debug(f"Cleared {len(to_remove)} old completed tasks")
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = 30.0):
        """
        Shutdown the threading manager.
        
        Args:
            wait: Whether to wait for current tasks to complete
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Shutting down ThreadingManager...")
        
        self._shutdown = True
        
        # Cancel all pending tasks
        with self._lock:
            for task_id in list(self.active_tasks.keys()):
                self.cancel_task(task_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        
        logger.info("ThreadingManager shutdown complete")
    
    def _ensure_workers_running(self):
        """Ensure worker threads are processing the queue."""
        # Submit worker tasks to process the queue
        for _ in range(min(self.max_workers, self.task_queue.qsize())):
            self.executor.submit(self._worker_loop)
    
    def _worker_loop(self):
        """Main worker loop that processes tasks from the queue."""
        while not self._shutdown:
            try:
                # Get next task from queue (with timeout to allow shutdown)
                try:
                    priority_task = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                task = priority_task.task
                
                # Check if task is still active (not cancelled)
                with self._lock:
                    if task.task_id not in self.active_tasks:
                        continue
                
                # Execute the task
                self._execute_task(task)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    def _execute_task(self, task: ProcessingTask):
        """Execute a single processing task."""
        task.started_at = time.time()
        
        try:
            logger.debug(f"Starting task {task.task_id} ({task.priority.name})")
            
            # Import here to avoid circular dependencies
            from .preview_processor import PreviewProcessor
            from .proxy_manager import ProxyManager
            from ..processing.raw_processor import ProcessingRecipe
            
            # Load the image
            import cv2
            image = cv2.imread(str(task.source_path))
            if image is None:
                raise ValueError(f"Could not load image: {task.source_path}")
            
            # Convert to RGB float32
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            
            # Determine processing mode based on target level
            if task.target_level in [ProxyLevel.THUMBNAIL, ProxyLevel.SMALL]:
                mode = ProcessingMode.PREVIEW
            elif task.target_level == ProxyLevel.MEDIUM:
                mode = ProcessingMode.BALANCED
            else:
                mode = ProcessingMode.EXPORT
            
            # Process the image
            processor = PreviewProcessor(self.config)
            result = processor.process_for_preview(image, task.recipe, mode, task.target_level)
            
            # Store result
            task.result = result
            task.progress = 1.0
            task.completed_at = time.time()
            
            # Update statistics
            duration = task.duration
            if duration:
                self.stats['total_processing_time'] += duration
                self.stats['tasks_completed'] += 1
            
            # Call callback if provided
            if task.callback:
                try:
                    task.callback(task)
                except Exception as e:
                    logger.error(f"Task callback failed: {e}")
            
            logger.debug(f"Completed task {task.task_id} in {duration:.3f}s")
            
        except Exception as e:
            task.error = e
            task.completed_at = time.time()
            self.stats['tasks_failed'] += 1
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Move task to completed
            self._move_to_completed(task.task_id)
    
    def _move_to_completed(self, task_id: str):
        """Move task from active to completed."""
        with self._lock:
            if task_id in self.active_tasks:
                task = self.active_tasks.pop(task_id)
                self.completed_tasks[task_id] = task


class ProgressiveProcessor:
    """
    Manages progressive enhancement of images.
    
    Provides fast initial preview that is progressively enhanced
    with better quality as processing time allows.
    """
    
    def __init__(self, threading_manager: ThreadingManager, config: PreviewConfig):
        self.threading_manager = threading_manager
        self.config = config
        self.progressive_states: Dict[str, ProgressiveState] = {}
        self._lock = threading.RLock()
        
        logger.info("ProgressiveProcessor initialized")
    
    def start_progressive_processing(self, image_path: str, recipe: Any,
                                   update_callback: Callable[[np.ndarray], None]) -> str:
        """
        Start progressive processing of an image.
        
        Args:
            image_path: Path to image file
            recipe: Processing recipe
            update_callback: Called with each progressive update
            
        Returns:
            Processing session ID
        """
        session_id = str(uuid.uuid4())
        
        with self._lock:
            state = ProgressiveState(
                max_steps=self.config.max_progressive_steps
            )
            self.progressive_states[session_id] = state
        
        # Submit progressive enhancement tasks
        self._submit_progressive_tasks(session_id, image_path, recipe, update_callback)
        
        return session_id
    
    def cancel_progressive_processing(self, session_id: str):
        """Cancel progressive processing session."""
        with self._lock:
            if session_id in self.progressive_states:
                state = self.progressive_states[session_id]
                
                # Cancel all queued enhancement tasks
                for task in state.enhancement_queue:
                    self.threading_manager.cancel_task(task.task_id)
                
                del self.progressive_states[session_id]
                logger.debug(f"Cancelled progressive processing session {session_id}")
    
    def _submit_progressive_tasks(self, session_id: str, image_path: str, 
                                 recipe: Any, callback: Callable):
        """Submit tasks for progressive enhancement."""
        from pathlib import Path
        
        path = Path(image_path)
        
        # Step 1: Fast preview (highest priority)
        task1 = ProcessingTask(
            task_id=f"{session_id}_step1",
            source_path=path,
            recipe=recipe,
            target_level=ProxyLevel.SMALL,
            priority=TaskPriority.HIGH,
            callback=lambda t: self._progressive_callback(session_id, 1, t, callback)
        )
        
        # Step 2: Medium quality
        task2 = ProcessingTask(
            task_id=f"{session_id}_step2",
            source_path=path,
            recipe=recipe,
            target_level=ProxyLevel.MEDIUM,
            priority=TaskPriority.NORMAL,
            callback=lambda t: self._progressive_callback(session_id, 2, t, callback)
        )
        
        # Step 3: High quality
        task3 = ProcessingTask(
            task_id=f"{session_id}_step3",
            source_path=path,
            recipe=recipe,
            target_level=ProxyLevel.LARGE,
            priority=TaskPriority.LOW,
            callback=lambda t: self._progressive_callback(session_id, 3, t, callback)
        )
        
        # Submit tasks
        self.threading_manager.submit_task(task1)
        self.threading_manager.submit_task(task2)
        self.threading_manager.submit_task(task3)
        
        # Store in state
        with self._lock:
            if session_id in self.progressive_states:
                state = self.progressive_states[session_id]
                state.enhancement_queue = [task1, task2, task3]
    
    def _progressive_callback(self, session_id: str, step: int, 
                            task: ProcessingTask, user_callback: Callable):
        """Handle completion of progressive enhancement step."""
        with self._lock:
            if session_id not in self.progressive_states:
                return
            
            state = self.progressive_states[session_id]
            
            # Check if we should update
            if not state.should_update(self.config.progressive_update_interval):
                return
            
            # Update state
            state.current_step = max(state.current_step, step)
            state.last_update = time.time()
        
        # Call user callback with result
        if task.result is not None and not task.is_failed:
            try:
                user_callback(task.result)
                logger.debug(f"Progressive update {step} for session {session_id}")
            except Exception as e:
                logger.error(f"Progressive callback failed: {e}")
        
        # Clean up if complete
        if state.is_complete:
            with self._lock:
                if session_id in self.progressive_states:
                    del self.progressive_states[session_id]