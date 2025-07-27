"""
Concurrent processing pipeline for PhotoSight performance optimization.

Provides high-performance concurrent processing for RAW photo analysis,
YOLO detection, and aesthetic analysis with smart resource management.
"""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from multiprocessing import Queue, Process, Manager
import multiprocessing as mp
import queue
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing pipeline stages."""
    RAW_DECODE = "raw_decode"
    TECHNICAL_ANALYSIS = "technical_analysis" 
    AESTHETIC_ANALYSIS = "aesthetic_analysis"
    YOLO_DETECTION = "yolo_detection"
    METADATA_EXTRACTION = "metadata_extraction"
    QUALITY_RANKING = "quality_ranking"
    DATABASE_WRITE = "database_write"


@dataclass
class ProcessingTask:
    """Represents a single processing task."""
    photo_id: int
    file_path: Path
    stage: ProcessingStage
    priority: int = 5  # 1-10, higher is more important
    metadata: Dict = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class ProcessingResult:
    """Represents processing result."""
    task: ProcessingTask
    result: Any
    execution_time: float
    error: Optional[str] = None
    success: bool = True


class ConcurrentPipeline:
    """
    High-performance concurrent processing pipeline for PhotoSight.
    
    Features:
    - Multi-stage pipeline with dependency management
    - Priority-based task scheduling
    - Resource-aware scaling
    - Progress tracking and monitoring
    - Graceful degradation under load
    """
    
    def __init__(self, max_workers: Optional[int] = None,
                 memory_limit_gb: float = 8.0,
                 enable_monitoring: bool = True):
        """
        Initialize the concurrent pipeline.
        
        Args:
            max_workers: Maximum concurrent workers (default: CPU count)
            memory_limit_gb: Memory limit for auto-scaling
            enable_monitoring: Enable performance monitoring
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.memory_limit_gb = memory_limit_gb
        self.enable_monitoring = enable_monitoring
        
        # Processing queues for each stage
        self.stage_queues = {
            stage: Queue(maxsize=100) for stage in ProcessingStage
        }
        
        # Result queue
        self.result_queue = Queue(maxsize=1000)
        
        # Worker processes
        self.workers = {}
        self.worker_stats = Manager().dict()
        
        # Pipeline state
        self.running = False
        self.shutdown_event = mp.Event()
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'total_errors': 0,
            'average_processing_time': 0.0,
            'stage_throughput': {stage: 0 for stage in ProcessingStage},
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0.0
        }
        
        # Stage dependencies (which stages must complete before others)
        self.stage_dependencies = {
            ProcessingStage.TECHNICAL_ANALYSIS: [ProcessingStage.RAW_DECODE],
            ProcessingStage.AESTHETIC_ANALYSIS: [ProcessingStage.RAW_DECODE],
            ProcessingStage.YOLO_DETECTION: [ProcessingStage.RAW_DECODE],
            ProcessingStage.QUALITY_RANKING: [
                ProcessingStage.TECHNICAL_ANALYSIS,
                ProcessingStage.AESTHETIC_ANALYSIS,
                ProcessingStage.YOLO_DETECTION
            ],
            ProcessingStage.DATABASE_WRITE: [ProcessingStage.QUALITY_RANKING]
        }
        
        logger.info(f"ConcurrentPipeline initialized with {self.max_workers} workers")
    
    def start(self):
        """Start the processing pipeline."""
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start worker processes for each stage
        for stage in ProcessingStage:
            worker_count = self._get_optimal_worker_count(stage)
            
            for i in range(worker_count):
                worker_id = f"{stage.value}_worker_{i}"
                worker = Process(
                    target=self._stage_worker,
                    args=(stage, worker_id),
                    daemon=True
                )
                worker.start()
                self.workers[worker_id] = worker
                
                logger.debug(f"Started worker {worker_id}")
        
        # Start result collector
        result_collector = Process(
            target=self._result_collector,
            daemon=True
        )
        result_collector.start()
        self.workers['result_collector'] = result_collector
        
        # Start monitoring if enabled
        if self.enable_monitoring:
            monitor_thread = threading.Thread(
                target=self._monitor_performance,
                daemon=True
            )
            monitor_thread.start()
        
        logger.info(f"Pipeline started with {len(self.workers)} workers")
    
    def stop(self, timeout: float = 30.0):
        """Stop the processing pipeline."""
        if not self.running:
            return
        
        logger.info("Stopping processing pipeline...")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for workers to finish
        start_time = time.time()
        for worker_id, worker in self.workers.items():
            remaining_time = max(0, timeout - (time.time() - start_time))
            worker.join(timeout=remaining_time)
            
            if worker.is_alive():
                logger.warning(f"Force terminating worker {worker_id}")
                worker.terminate()
                worker.join()
        
        self.workers.clear()
        logger.info("Pipeline stopped")
    
    def submit_photo(self, photo_id: int, file_path: Path, 
                    priority: int = 5, metadata: Optional[Dict] = None) -> bool:
        """
        Submit a photo for processing.
        
        Args:
            photo_id: Photo database ID
            file_path: Path to RAW photo file
            priority: Processing priority (1-10)
            metadata: Optional metadata dict
            
        Returns:
            True if successfully queued
        """
        if not self.running:
            logger.error("Pipeline not running")
            return False
        
        # Create initial task for RAW decode stage
        task = ProcessingTask(
            photo_id=photo_id,
            file_path=file_path,
            stage=ProcessingStage.RAW_DECODE,
            priority=priority,
            metadata=metadata or {}
        )
        
        try:
            # Add to RAW decode queue
            self.stage_queues[ProcessingStage.RAW_DECODE].put(task, timeout=1.0)
            logger.debug(f"Queued photo {photo_id} for processing")
            return True
            
        except queue.Full:
            logger.warning(f"Queue full, dropping photo {photo_id}")
            return False
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get current queue statistics."""
        return {
            stage.value: self.stage_queues[stage].qsize()
            for stage in ProcessingStage
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = dict(self.stats)
        stats['queue_sizes'] = self.get_queue_stats()
        stats['worker_count'] = len(self.workers)
        stats['running'] = self.running
        return stats
    
    def _get_optimal_worker_count(self, stage: ProcessingStage) -> int:
        """Determine optimal worker count for a stage."""
        # CPU-intensive stages get fewer workers
        cpu_intensive_stages = {
            ProcessingStage.RAW_DECODE,
            ProcessingStage.TECHNICAL_ANALYSIS,
            ProcessingStage.AESTHETIC_ANALYSIS,
            ProcessingStage.YOLO_DETECTION
        }
        
        if stage in cpu_intensive_stages:
            return max(1, self.max_workers // 2)
        else:
            # I/O bound stages can have more workers
            return min(self.max_workers, 4)
    
    def _stage_worker(self, stage: ProcessingStage, worker_id: str):
        """Worker process for a specific processing stage."""
        logger.debug(f"Worker {worker_id} started for stage {stage.value}")
        
        processed_count = 0
        error_count = 0
        
        while not self.shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                task = self.stage_queues[stage].get(timeout=1.0)
                
                start_time = time.perf_counter()
                
                # Process the task
                try:
                    result = self._process_task(task)
                    execution_time = time.perf_counter() - start_time
                    
                    processing_result = ProcessingResult(
                        task=task,
                        result=result,
                        execution_time=execution_time,
                        success=True
                    )
                    
                    processed_count += 1
                    
                except Exception as e:
                    execution_time = time.perf_counter() - start_time
                    
                    processing_result = ProcessingResult(
                        task=task,
                        result=None,
                        execution_time=execution_time,
                        error=str(e),
                        success=False
                    )
                    
                    error_count += 1
                    logger.error(f"Task processing error in {worker_id}: {e}")
                
                # Send result to collector
                try:
                    self.result_queue.put(processing_result, timeout=1.0)
                except queue.Full:
                    logger.warning(f"Result queue full, dropping result from {worker_id}")
                
                # Update worker stats
                self.worker_stats[worker_id] = {
                    'processed': processed_count,
                    'errors': error_count,
                    'last_activity': time.time()
                }
                
            except queue.Empty:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.debug(f"Worker {worker_id} shutting down")
    
    def _process_task(self, task: ProcessingTask) -> Any:
        """
        Process a single task based on its stage.
        
        Args:
            task: ProcessingTask to execute
            
        Returns:
            Processing result
        """
        stage_processors = {
            ProcessingStage.RAW_DECODE: self._process_raw_decode,
            ProcessingStage.TECHNICAL_ANALYSIS: self._process_technical_analysis,
            ProcessingStage.AESTHETIC_ANALYSIS: self._process_aesthetic_analysis,
            ProcessingStage.YOLO_DETECTION: self._process_yolo_detection,
            ProcessingStage.METADATA_EXTRACTION: self._process_metadata_extraction,
            ProcessingStage.QUALITY_RANKING: self._process_quality_ranking,
            ProcessingStage.DATABASE_WRITE: self._process_database_write
        }
        
        processor = stage_processors.get(task.stage)
        if not processor:
            raise ValueError(f"No processor for stage {task.stage}")
        
        return processor(task)
    
    def _process_raw_decode(self, task: ProcessingTask) -> Dict:
        """Process RAW decode stage."""
        # Simulate RAW processing
        time.sleep(0.1)  # Replace with actual RAW processing
        
        # Queue next stages
        next_stages = [
            ProcessingStage.TECHNICAL_ANALYSIS,
            ProcessingStage.AESTHETIC_ANALYSIS,
            ProcessingStage.YOLO_DETECTION,
            ProcessingStage.METADATA_EXTRACTION
        ]
        
        for next_stage in next_stages:
            next_task = ProcessingTask(
                photo_id=task.photo_id,
                file_path=task.file_path,
                stage=next_stage,
                priority=task.priority,
                metadata=task.metadata
            )
            
            try:
                self.stage_queues[next_stage].put(next_task, timeout=0.1)
            except queue.Full:
                logger.warning(f"Queue full for stage {next_stage}")
        
        return {'decoded': True, 'image_data': 'mock_data'}
    
    def _process_technical_analysis(self, task: ProcessingTask) -> Dict:
        """Process technical analysis stage."""
        time.sleep(0.2)  # Replace with actual technical analysis
        return {
            'sharpness_score': 0.85,
            'exposure_quality': 0.92,
            'noise_level': 0.15
        }
    
    def _process_aesthetic_analysis(self, task: ProcessingTask) -> Dict:
        """Process aesthetic analysis stage."""
        time.sleep(0.3)  # Replace with actual aesthetic analysis
        return {
            'composition_score': 0.78,
            'color_harmony': 0.88,
            'visual_impact': 0.82
        }
    
    def _process_yolo_detection(self, task: ProcessingTask) -> Dict:
        """Process YOLO detection stage."""
        time.sleep(0.4)  # Replace with actual YOLO detection
        return {
            'detections': [
                {'class': 'person', 'confidence': 0.95, 'bbox': [100, 100, 200, 300]}
            ],
            'detection_count': 1
        }
    
    def _process_metadata_extraction(self, task: ProcessingTask) -> Dict:
        """Process metadata extraction stage."""
        time.sleep(0.05)  # Replace with actual metadata extraction
        return {
            'camera': 'Sony A7R V',
            'lens': '24-70mm f/2.8',
            'iso': 800,
            'aperture': 2.8
        }
    
    def _process_quality_ranking(self, task: ProcessingTask) -> Dict:
        """Process quality ranking stage."""
        time.sleep(0.1)  # Replace with actual ranking calculation
        
        # Queue database write
        db_task = ProcessingTask(
            photo_id=task.photo_id,
            file_path=task.file_path,
            stage=ProcessingStage.DATABASE_WRITE,
            priority=task.priority,
            metadata=task.metadata
        )
        
        try:
            self.stage_queues[ProcessingStage.DATABASE_WRITE].put(db_task, timeout=0.1)
        except queue.Full:
            logger.warning("Database write queue full")
        
        return {
            'overall_quality_score': 0.85,
            'ranking': 7.8
        }
    
    def _process_database_write(self, task: ProcessingTask) -> Dict:
        """Process database write stage."""
        time.sleep(0.05)  # Replace with actual database write
        return {'database_updated': True}
    
    def _result_collector(self):
        """Collect and aggregate processing results."""
        logger.debug("Result collector started")
        
        total_processed = 0
        total_errors = 0
        total_time = 0.0
        
        while not self.shutdown_event.is_set():
            try:
                result = self.result_queue.get(timeout=1.0)
                
                # Update statistics
                total_processed += 1
                total_time += result.execution_time
                
                if not result.success:
                    total_errors += 1
                
                # Update global stats
                self.stats.update({
                    'total_processed': total_processed,
                    'total_errors': total_errors,
                    'average_processing_time': total_time / max(total_processed, 1)
                })
                
                # Log completion
                if result.success:
                    logger.debug(f"Photo {result.task.photo_id} "
                               f"stage {result.task.stage.value} "
                               f"completed in {result.execution_time:.3f}s")
                else:
                    logger.error(f"Photo {result.task.photo_id} "
                               f"stage {result.task.stage.value} "
                               f"failed: {result.error}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Result collector error: {e}")
        
        logger.debug("Result collector shutting down")
    
    def _monitor_performance(self):
        """Monitor system performance and auto-scale if needed."""
        import psutil
        
        logger.debug("Performance monitor started")
        
        while self.running:
            try:
                # Get system metrics
                memory_usage = psutil.virtual_memory()
                cpu_usage = psutil.cpu_percent(interval=1.0)
                
                self.stats.update({
                    'memory_usage_mb': memory_usage.used / 1024 / 1024,
                    'cpu_usage_percent': cpu_usage
                })
                
                # Check for memory pressure
                memory_usage_gb = memory_usage.used / 1024 / 1024 / 1024
                if memory_usage_gb > self.memory_limit_gb:
                    logger.warning(f"High memory usage: {memory_usage_gb:.1f}GB")
                
                # Log performance stats periodically
                if total_processed := self.stats['total_processed']:
                    if total_processed % 100 == 0:
                        logger.info(f"Performance: {total_processed} processed, "
                                  f"{self.stats['total_errors']} errors, "
                                  f"{self.stats['average_processing_time']:.3f}s avg time")
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                time.sleep(10)
        
        logger.debug("Performance monitor shutting down")


def create_processing_pipeline(max_workers: Optional[int] = None,
                             memory_limit_gb: float = 8.0) -> ConcurrentPipeline:
    """
    Create and configure a processing pipeline.
    
    Args:
        max_workers: Maximum concurrent workers
        memory_limit_gb: Memory limit for auto-scaling
        
    Returns:
        Configured ConcurrentPipeline instance
    """
    pipeline = ConcurrentPipeline(
        max_workers=max_workers,
        memory_limit_gb=memory_limit_gb,
        enable_monitoring=True
    )
    
    return pipeline