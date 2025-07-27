"""
Async processing utilities for PhotoSight performance optimization.

Converts blocking operations to async/await patterns and provides
concurrent processing capabilities for CPU-intensive tasks.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, partial
from typing import Any, Callable, List, Dict, Optional, Union, Awaitable
from pathlib import Path
import threading
from queue import Queue
import multiprocessing as mp

logger = logging.getLogger(__name__)


class AsyncProcessor:
    """
    High-performance async processing manager for PhotoSight.
    
    Features:
    - CPU-bound task processing in separate processes
    - I/O-bound task processing in thread pool
    - Batch processing with progress tracking
    - Resource management and cleanup
    - Graceful shutdown handling
    """
    
    def __init__(self, max_workers: Optional[int] = None, 
                 max_processes: Optional[int] = None):
        """
        Initialize the async processor.
        
        Args:
            max_workers: Maximum thread pool workers (default: CPU count * 4)
            max_processes: Maximum process pool workers (default: CPU count)
        """
        self.max_workers = max_workers or (mp.cpu_count() * 4)
        self.max_processes = max_processes or mp.cpu_count()
        
        # Executors for different types of work
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
        
        # Active tasks tracking
        self.active_tasks = set()
        self.task_counter = 0
        self._shutdown = False
        
        logger.info(f"AsyncProcessor initialized: {self.max_workers} threads, {self.max_processes} processes")
    
    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a blocking function in a thread pool.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if self._shutdown:
            raise RuntimeError("AsyncProcessor is shutting down")
        
        loop = asyncio.get_event_loop()
        task_id = self._get_task_id()
        
        try:
            self.active_tasks.add(task_id)
            logger.debug(f"Starting thread task {task_id}: {func.__name__}")
            
            # Execute in thread pool
            result = await loop.run_in_executor(
                self.thread_pool, 
                partial(func, *args, **kwargs)
            )
            
            logger.debug(f"Completed thread task {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"Thread task {task_id} failed: {e}")
            raise
        finally:
            self.active_tasks.discard(task_id)
    
    async def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a CPU-intensive function in a separate process.
        
        Args:
            func: Function to execute (must be picklable)
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if self._shutdown:
            raise RuntimeError("AsyncProcessor is shutting down")
        
        loop = asyncio.get_event_loop()
        task_id = self._get_task_id()
        
        try:
            self.active_tasks.add(task_id)
            logger.debug(f"Starting process task {task_id}: {func.__name__}")
            
            # Execute in process pool
            result = await loop.run_in_executor(
                self.process_pool,
                partial(func, *args, **kwargs)
            )
            
            logger.debug(f"Completed process task {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"Process task {task_id} failed: {e}")
            raise
        finally:
            self.active_tasks.discard(task_id)
    
    async def batch_process(self, items: List[Any], func: Callable,
                           batch_size: int = 10, use_processes: bool = False,
                           progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        Process a large number of items in batches.
        
        Args:
            items: Items to process
            func: Function to apply to each item
            batch_size: Number of items per batch
            use_processes: Use process pool instead of thread pool
            progress_callback: Optional progress callback function
            
        Returns:
            List of results in same order as input
        """
        if not items:
            return []
        
        results = [None] * len(items)
        total_items = len(items)
        completed_items = 0
        
        # Process in batches
        for i in range(0, total_items, batch_size):
            batch = items[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, total_items)))
            
            # Create async tasks for this batch
            if use_processes:
                tasks = [
                    self.run_in_process(func, item)
                    for item in batch
                ]
            else:
                tasks = [
                    self.run_in_thread(func, item)
                    for item in batch
                ]
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for idx, result in zip(batch_indices, batch_results):
                results[idx] = result
                completed_items += 1
                
                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback(completed_items, total_items)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
            
            logger.debug(f"Completed batch {i//batch_size + 1}, "
                        f"total progress: {completed_items}/{total_items}")
        
        return results
    
    async def concurrent_map(self, func: Callable, items: List[Any],
                            max_concurrent: int = 10, 
                            use_processes: bool = False) -> List[Any]:
        """
        Apply function to items with controlled concurrency.
        
        Args:
            func: Function to apply
            items: Items to process
            max_concurrent: Maximum concurrent operations
            use_processes: Use process pool instead of thread pool
            
        Returns:
            List of results in same order as input
        """
        if not items:
            return []
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_item(item):
            async with semaphore:
                if use_processes:
                    return await self.run_in_process(func, item)
                else:
                    return await self.run_in_thread(func, item)
        
        # Create tasks for all items
        tasks = [process_item(item) for item in items]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def timed_execution(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """
        Execute function and measure execution time.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, execution_time_seconds)
        """
        start_time = time.perf_counter()
        result = await self.run_in_thread(func, *args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        return result, execution_time
    
    def _get_task_id(self) -> str:
        """Generate unique task ID."""
        self.task_counter += 1
        return f"task_{self.task_counter}"
    
    async def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """
        Shutdown the async processor.
        
        Args:
            wait: Wait for active tasks to complete
            timeout: Maximum time to wait for shutdown
        """
        self._shutdown = True
        
        if wait and self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            
            # Wait for active tasks with timeout
            start_time = time.time()
            while self.active_tasks and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            if self.active_tasks:
                logger.warning(f"Timeout waiting for {len(self.active_tasks)} tasks")
        
        # Shutdown executors
        self.thread_pool.shutdown(wait=wait)
        self.process_pool.shutdown(wait=wait)
        
        logger.info("AsyncProcessor shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processor statistics."""
        return {
            'max_workers': self.max_workers,
            'max_processes': self.max_processes,
            'active_tasks': len(self.active_tasks),
            'total_tasks_created': self.task_counter,
            'shutdown': self._shutdown
        }


# Global async processor instance
_processor_instance: Optional[AsyncProcessor] = None
_processor_lock = threading.Lock()

def get_async_processor() -> AsyncProcessor:
    """Get the global async processor instance."""
    global _processor_instance
    
    if _processor_instance is None:
        with _processor_lock:
            if _processor_instance is None:
                _processor_instance = AsyncProcessor()
    
    return _processor_instance


# Decorator utilities

def async_io_bound(func: Callable) -> Callable:
    """
    Decorator to make a blocking I/O function async-friendly.
    
    Usage:
        @async_io_bound
        def slow_file_operation(path):
            return Path(path).read_text()
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        processor = get_async_processor()
        return await processor.run_in_thread(func, *args, **kwargs)
    
    return wrapper


def async_cpu_bound(func: Callable) -> Callable:
    """
    Decorator to make a CPU-intensive function async-friendly.
    
    Usage:
        @async_cpu_bound
        def compute_heavy_analysis(image_data):
            return expensive_computation(image_data)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        processor = get_async_processor()
        return await processor.run_in_process(func, *args, **kwargs)
    
    return wrapper


def async_timed(func: Callable) -> Callable:
    """
    Decorator to add timing to async functions.
    
    Usage:
        @async_timed
        async def some_operation():
            return await expensive_async_operation()
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


class AsyncTaskQueue:
    """
    Async task queue for managing background processing.
    
    Useful for handling long-running operations without blocking
    the main application.
    """
    
    def __init__(self, max_concurrent: int = 5):
        """
        Initialize the task queue.
        
        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = asyncio.Queue()
        self.running_tasks = set()
        self.completed_tasks = []
        self.failed_tasks = []
        self._worker_task = None
        self._shutdown = False
    
    async def start(self):
        """Start the task queue worker."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())
    
    async def add_task(self, func: Callable, *args, task_id: Optional[str] = None, **kwargs):
        """
        Add a task to the queue.
        
        Args:
            func: Function to execute
            *args: Function arguments
            task_id: Optional task identifier
            **kwargs: Function keyword arguments
        """
        task_id = task_id or f"task_{time.time()}"
        await self.queue.put({
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'created_at': time.time()
        })
    
    async def _worker(self):
        """Background worker to process tasks."""
        processor = get_async_processor()
        
        while not self._shutdown:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Execute task with semaphore control
                task_coroutine = self._execute_task(processor, task)
                asyncio.create_task(task_coroutine)
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(f"Task queue worker error: {e}")
    
    async def _execute_task(self, processor: AsyncProcessor, task: Dict):
        """Execute a single task."""
        async with self.semaphore:
            task_id = task['id']
            self.running_tasks.add(task_id)
            
            try:
                logger.debug(f"Executing task {task_id}")
                start_time = time.time()
                
                # Execute task
                result = await processor.run_in_thread(
                    task['func'], 
                    *task['args'], 
                    **task['kwargs']
                )
                
                execution_time = time.time() - start_time
                
                # Record completion
                self.completed_tasks.append({
                    'id': task_id,
                    'result': result,
                    'execution_time': execution_time,
                    'completed_at': time.time()
                })
                
                logger.debug(f"Task {task_id} completed in {execution_time:.3f}s")
                
            except Exception as e:
                # Record failure
                execution_time = time.time() - start_time
                self.failed_tasks.append({
                    'id': task_id,
                    'error': str(e),
                    'execution_time': execution_time,
                    'failed_at': time.time()
                })
                
                logger.error(f"Task {task_id} failed after {execution_time:.3f}s: {e}")
                
            finally:
                self.running_tasks.discard(task_id)
    
    async def shutdown(self):
        """Shutdown the task queue."""
        self._shutdown = True
        
        if self._worker_task:
            await self._worker_task
        
        # Wait for running tasks to complete
        while self.running_tasks:
            await asyncio.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task queue statistics."""
        return {
            'queue_size': self.queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'max_concurrent': self.max_concurrent
        }


# Shutdown handler for cleanup
async def shutdown_async_processing():
    """Cleanup function for graceful shutdown."""
    global _processor_instance
    
    if _processor_instance:
        await _processor_instance.shutdown()
        _processor_instance = None
        logger.info("Async processing shutdown complete")