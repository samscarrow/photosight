"""
Batch Processing Manager for PhotoSight API

Handles batch processing of multiple images with progress tracking.
"""

import logging
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import uuid
import requests
from concurrent.futures import ThreadPoolExecutor, Future

from ..processing.raw_processor import RawProcessor, ProcessingRecipe
from .models import BatchProcessingRequest, BatchJobStatus

logger = logging.getLogger(__name__)


class BatchJob:
    """Represents a batch processing job."""
    
    def __init__(self, job_id: str, request: BatchProcessingRequest):
        self.job_id = job_id
        self.request = request
        self.status = BatchJobStatus(
            job_id=job_id,
            state='queued',
            total_images=len(request.images)
        )
        self.futures: List[Future] = []
        self._lock = threading.RLock()
    
    def update_progress(self, processed: int = 0, failed: int = 0):
        """Update job progress."""
        with self._lock:
            self.status.processed_images += processed
            self.status.failed_images += failed
            
            # Update state if complete
            if (self.status.processed_images + self.status.failed_images) >= self.status.total_images:
                self.status.state = 'completed'
                self.status.completed_at = datetime.utcnow()
    
    def add_result(self, image_path: str, output_path: str, processing_time: float):
        """Add successful result."""
        with self._lock:
            self.status.results.append({
                'image_path': image_path,
                'output_path': output_path,
                'processing_time': processing_time,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def add_error(self, image_path: str, error: str):
        """Add error result."""
        with self._lock:
            self.status.errors.append({
                'image_path': image_path,
                'error': error,
                'timestamp': datetime.utcnow().isoformat()
            })


class BatchProcessor:
    """
    Manages batch processing of images.
    
    Features:
    - Concurrent processing with configurable workers
    - Progress tracking and reporting
    - Webhook notifications
    - Priority-based queuing
    """
    
    def __init__(self, max_workers: int = 4, max_queue_size: int = 100):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Job management
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self._lock = threading.RLock()
        
        # Processing pool
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="BatchWorker"
        )
        
        # Start worker thread
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="BatchProcessorWorker",
            daemon=True
        )
        self.worker_thread.start()
        
        # Statistics
        self.stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'total_images_processed': 0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"BatchProcessor initialized with {max_workers} workers")
    
    def submit_job(self, request: BatchProcessingRequest) -> str:
        """
        Submit a new batch processing job.
        
        Args:
            request: Batch processing request
            
        Returns:
            Job ID for tracking
        """
        job_id = f"batch_{uuid.uuid4().hex[:12]}"
        
        with self._lock:
            # Create job
            job = BatchJob(job_id, request)
            self.jobs[job_id] = job
            
            # Update stats
            self.stats['total_jobs'] += 1
            
            # Calculate priority (0 = highest)
            priority_map = {'urgent': 0, 'high': 1, 'normal': 2, 'low': 3}
            priority = priority_map.get(request.priority, 2)
            
            # Add to queue
            self.job_queue.put((priority, time.time(), job))
            
            logger.info(f"Submitted batch job {job_id} with {len(request.images)} images")
            
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[BatchJobStatus]:
        """Get current job status."""
        with self._lock:
            job = self.jobs.get(job_id)
            return job.status if job else None
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a batch job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancelled, False if not found or already complete
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            
            if job.status.state in ['completed', 'failed']:
                return False
            
            # Cancel futures
            for future in job.futures:
                future.cancel()
            
            # Update status
            job.status.state = 'cancelled'
            job.status.completed_at = datetime.utcnow()
            
            logger.info(f"Cancelled batch job {job_id}")
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        with self._lock:
            active_jobs = sum(
                1 for job in self.jobs.values()
                if job.status.state in ['queued', 'processing']
            )
            
            return {
                **self.stats,
                'active_jobs': active_jobs,
                'queue_size': self.job_queue.qsize(),
                'max_workers': self.max_workers
            }
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs."""
        cutoff_time = datetime.utcnow()
        
        with self._lock:
            to_remove = []
            for job_id, job in self.jobs.items():
                if job.status.completed_at:
                    age = cutoff_time - job.status.completed_at
                    if age.total_seconds() > max_age_hours * 3600:
                        to_remove.append(job_id)
            
            for job_id in to_remove:
                del self.jobs[job_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old batch jobs")
    
    def shutdown(self, wait: bool = True):
        """Shutdown the batch processor."""
        logger.info("Shutting down BatchProcessor...")
        
        self.running = False
        
        # Cancel all pending jobs
        with self._lock:
            for job in self.jobs.values():
                if job.status.state in ['queued', 'processing']:
                    self.cancel_job(job.job_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        
        # Wait for worker thread
        if wait and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        logger.info("BatchProcessor shutdown complete")
    
    def _worker_loop(self):
        """Main worker loop for processing jobs."""
        while self.running:
            try:
                # Get next job (with timeout to allow shutdown)
                try:
                    priority, timestamp, job = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the job
                self._process_job(job)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    def _process_job(self, job: BatchJob):
        """Process a batch job."""
        logger.info(f"Starting batch job {job.job_id}")
        
        # Update state
        job.status.state = 'processing'
        
        # Create processor
        processor = RawProcessor()
        
        # Process each image
        for i, image_path in enumerate(job.request.images):
            # Submit to thread pool
            future = self.executor.submit(
                self._process_single_image,
                processor,
                image_path,
                job.request.recipe,
                job.request.output_format,
                job.request.output_quality,
                job
            )
            
            job.futures.append(future)
        
        # Wait for all images to complete
        for future in job.futures:
            try:
                future.result()  # This blocks until complete
            except Exception as e:
                logger.error(f"Image processing error: {e}")
        
        # Send webhook notification if configured
        if job.request.webhook_url and job.status.state == 'completed':
            self._send_webhook_notification(job)
        
        # Update stats
        with self._lock:
            self.stats['completed_jobs'] += 1
            self.stats['total_images_processed'] += job.status.processed_images
    
    def _process_single_image(self, processor: RawProcessor, image_path: str,
                            recipe_dict: Dict[str, Any], output_format: str,
                            output_quality: int, job: BatchJob):
        """Process a single image in the batch."""
        start_time = time.time()
        
        try:
            # Parse recipe
            recipe = ProcessingRecipe.from_dict(recipe_dict)
            
            # Process image
            image_path_obj = Path(image_path)
            result = processor.process_image(image_path_obj, recipe)
            
            if result is None:
                raise ValueError("Processing returned None")
            
            # Generate output path
            output_dir = Path('./output/batch') / job.job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{image_path_obj.stem}_processed.{output_format}"
            output_path = output_dir / output_filename
            
            # Save result
            import numpy as np
            from PIL import Image
            
            # Convert to uint8
            img_array = (np.clip(result, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='RGB')
            
            # Save with specified format and quality
            save_kwargs = {'quality': output_quality} if output_format in ['jpeg', 'jpg'] else {}
            img.save(output_path, output_format.upper(), **save_kwargs)
            
            # Record result
            processing_time = time.time() - start_time
            job.add_result(image_path, str(output_path), processing_time)
            job.update_progress(processed=1)
            
            logger.debug(f"Processed {image_path} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            job.add_error(image_path, str(e))
            job.update_progress(failed=1)
    
    def _send_webhook_notification(self, job: BatchJob):
        """Send webhook notification for completed job."""
        try:
            payload = {
                'job_id': job.job_id,
                'status': job.status.to_dict(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            response = requests.post(
                job.request.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Webhook notification sent for job {job.job_id}")
            else:
                logger.warning(f"Webhook notification failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")