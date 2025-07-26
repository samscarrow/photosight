"""
Celery tasks for PhotoSight API background processing
"""

import os
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import uuid

from celery import Task, group, chain
from celery.exceptions import SoftTimeLimitExceeded
import redis
import numpy as np

from .celery_app import celery_app
from ..processing import PhotoProcessor, ProcessingRecipe
from ..processing.preview import PreviewGenerator
from ..db import get_session
from ..db.models import Photo, ProcessingJob, JobStatus
from .websocket_events import emit_progress_update, emit_preview_update

logger = logging.getLogger(__name__)

# Redis client for caching and coordination
redis_client = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))


class PhotoProcessingTask(Task):
    """Base task class with common functionality"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {exc}")
        
        # Update job status in database
        job_id = kwargs.get('job_id')
        if job_id:
            with get_session() as session:
                job = session.query(ProcessingJob).filter_by(id=job_id).first()
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = str(exc)
                    job.completed_at = datetime.utcnow()
                    session.commit()
        
        # Emit failure notification
        emit_progress_update(job_id, {
            'status': 'failed',
            'error': str(exc)
        })
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Task {task_id} retrying: {exc}")
        
        job_id = kwargs.get('job_id')
        if job_id:
            emit_progress_update(job_id, {
                'status': 'retrying',
                'retry_count': self.request.retries
            })
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.info(f"Task {task_id} completed successfully")


@celery_app.task(base=PhotoProcessingTask, bind=True, name='photosight.api.tasks.process_photo')
def process_photo(self, photo_id: str, recipe: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a single photo with the given recipe
    
    Args:
        photo_id: ID of the photo to process
        recipe: Processing recipe dictionary
        job_id: Optional job ID for tracking
        
    Returns:
        Dict with processing results
    """
    try:
        logger.info(f"Processing photo {photo_id} with job {job_id}")
        
        # Update job status
        if job_id:
            _update_job_status(job_id, JobStatus.PROCESSING, {'current_photo': photo_id})
        
        # Get photo from database
        with get_session() as session:
            photo = session.query(Photo).filter_by(id=photo_id).first()
            if not photo:
                raise ValueError(f"Photo {photo_id} not found")
            
            file_path = photo.file_path
        
        # Initialize processor
        processor = PhotoProcessor()
        recipe_obj = ProcessingRecipe.from_dict(recipe)
        
        # Process the photo
        result = processor.process(file_path, recipe_obj)
        
        # Save processed image
        output_path = _save_processed_image(photo_id, result['image'], recipe.get('export_settings', {}))
        
        # Generate preview
        preview_url = _generate_preview(photo_id, result['image'])
        
        # Update database
        with get_session() as session:
            photo = session.query(Photo).filter_by(id=photo_id).first()
            photo.status = 'completed'
            photo.output_path = output_path
            photo.preview_url = preview_url
            photo.processed_at = datetime.utcnow()
            photo.processing_metadata = {
                'recipe': recipe,
                'processing_time_ms': result.get('processing_time_ms', 0),
                'quality_metrics': result.get('quality_metrics', {})
            }
            session.commit()
        
        # Cache the result
        _cache_processing_result(photo_id, {
            'output_path': output_path,
            'preview_url': preview_url,
            'metadata': result.get('metadata', {})
        })
        
        # Emit completion event
        emit_preview_update(photo_id, preview_url)
        
        return {
            'photo_id': photo_id,
            'output_path': output_path,
            'preview_url': preview_url,
            'processing_time_ms': result.get('processing_time_ms', 0)
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Task timeout for photo {photo_id}")
        raise
    except Exception as e:
        logger.error(f"Error processing photo {photo_id}: {e}")
        raise


@celery_app.task(base=PhotoProcessingTask, bind=True, name='photosight.api.tasks.batch_process')
def batch_process(self, job_id: str, photo_ids: List[str], recipe: Dict[str, Any], 
                 priority: str = 'normal') -> Dict[str, Any]:
    """
    Process multiple photos in batch
    
    Args:
        job_id: Batch job ID
        photo_ids: List of photo IDs to process
        recipe: Processing recipe
        priority: Job priority
        
    Returns:
        Dict with batch results
    """
    try:
        logger.info(f"Starting batch job {job_id} with {len(photo_ids)} photos")
        
        # Update job status
        _update_job_status(job_id, JobStatus.PROCESSING, {
            'total_photos': len(photo_ids),
            'processed': 0
        })
        
        # Create individual tasks
        tasks = []
        for i, photo_id in enumerate(photo_ids):
            task = process_photo.s(photo_id, recipe, job_id)
            tasks.append(task)
            
            # Emit progress
            progress = (i / len(photo_ids)) * 100
            emit_progress_update(job_id, {
                'progress': progress,
                'current': i,
                'total': len(photo_ids)
            })
        
        # Execute tasks based on priority
        if priority == 'high':
            # Process in parallel with higher concurrency
            job = group(tasks).apply_async(queue='high_priority')
        else:
            # Process with normal concurrency
            job = group(tasks).apply_async(queue='normal')
        
        # Wait for completion
        results = job.get(timeout=3600)  # 1 hour timeout
        
        # Aggregate results
        successful = sum(1 for r in results if r and 'error' not in r)
        failed = len(results) - successful
        
        # Update final job status
        _update_job_status(job_id, JobStatus.COMPLETED, {
            'total_photos': len(photo_ids),
            'successful': successful,
            'failed': failed,
            'completed_at': datetime.utcnow().isoformat()
        })
        
        return {
            'job_id': job_id,
            'total': len(photo_ids),
            'successful': successful,
            'failed': failed,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}")
        _update_job_status(job_id, JobStatus.FAILED, {'error': str(e)})
        raise


@celery_app.task(name='photosight.api.tasks.generate_preview')
def generate_preview(photo_id: str, image_data: Optional[bytes] = None) -> str:
    """Generate or update preview for a photo"""
    try:
        if not image_data:
            # Load from file
            with get_session() as session:
                photo = session.query(Photo).filter_by(id=photo_id).first()
                if not photo:
                    raise ValueError(f"Photo {photo_id} not found")
                
                # Load image
                image_data = _load_image(photo.file_path)
        
        # Generate preview
        preview_generator = PreviewGenerator()
        preview_path = preview_generator.generate(image_data, photo_id)
        
        # Update database
        with get_session() as session:
            photo = session.query(Photo).filter_by(id=photo_id).first()
            photo.preview_url = f"/previews/{photo_id}.jpg"
            session.commit()
        
        # Cache preview URL
        redis_client.setex(f"preview:{photo_id}", 3600, preview_path)
        
        return preview_path
        
    except Exception as e:
        logger.error(f"Failed to generate preview for {photo_id}: {e}")
        raise


@celery_app.task(name='photosight.api.tasks.export_photo')
def export_photo(photo_id: str, export_settings: Dict[str, Any]) -> str:
    """Export processed photo with specific settings"""
    try:
        # Get processed image from cache or database
        cached = redis_client.get(f"processed:{photo_id}")
        if cached:
            image_data = np.frombuffer(cached, dtype=np.uint8)
        else:
            with get_session() as session:
                photo = session.query(Photo).filter_by(id=photo_id).first()
                if not photo or not photo.output_path:
                    raise ValueError(f"No processed image for photo {photo_id}")
                
                image_data = _load_image(photo.output_path)
        
        # Apply export settings
        output_path = _export_with_settings(photo_id, image_data, export_settings)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to export photo {photo_id}: {e}")
        raise


@celery_app.task(name='photosight.api.tasks.cleanup_old_files')
def cleanup_old_files(file_type: str = 'previews', hours_old: int = 24) -> Dict[str, int]:
    """Clean up old temporary files"""
    try:
        cleanup_path = Path(f"/tmp/photosight/{file_type}")
        if not cleanup_path.exists():
            return {'cleaned': 0}
        
        cutoff_time = datetime.now() - timedelta(hours=hours_old)
        cleaned = 0
        
        for file_path in cleanup_path.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    cleaned += 1
        
        logger.info(f"Cleaned {cleaned} {file_type} files older than {hours_old} hours")
        return {'cleaned': cleaned}
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {'cleaned': 0, 'error': str(e)}


@celery_app.task(name='photosight.api.tasks.update_processing_stats')
def update_processing_stats() -> Dict[str, Any]:
    """Update processing statistics in cache"""
    try:
        with get_session() as session:
            # Get stats from last 24 hours
            since = datetime.utcnow() - timedelta(hours=24)
            
            total_processed = session.query(Photo).filter(
                Photo.processed_at >= since
            ).count()
            
            avg_time = session.query(
                func.avg(Photo.processing_metadata['processing_time_ms'])
            ).filter(
                Photo.processed_at >= since
            ).scalar() or 0
            
            # Get queue sizes
            queue_sizes = {
                'high_priority': _get_queue_size('high_priority'),
                'normal': _get_queue_size('normal'),
                'low_priority': _get_queue_size('low_priority')
            }
            
            stats = {
                'processed_24h': total_processed,
                'avg_processing_time_ms': float(avg_time),
                'queue_sizes': queue_sizes,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache stats
            redis_client.setex('processing_stats', 300, json.dumps(stats))
            
            return stats
            
    except Exception as e:
        logger.error(f"Failed to update stats: {e}")
        return {}


# Helper functions

def _update_job_status(job_id: str, status: JobStatus, metadata: Dict[str, Any]) -> None:
    """Update job status in database and cache"""
    with get_session() as session:
        job = session.query(ProcessingJob).filter_by(id=job_id).first()
        if job:
            job.status = status
            job.metadata.update(metadata)
            if status == JobStatus.PROCESSING and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                job.completed_at = datetime.utcnow()
            session.commit()
    
    # Update cache
    redis_client.setex(f"job:{job_id}", 3600, json.dumps({
        'status': status.value,
        'metadata': metadata
    }))


def _save_processed_image(photo_id: str, image_data: np.ndarray, 
                         export_settings: Dict[str, Any]) -> str:
    """Save processed image to disk"""
    output_dir = Path(os.environ.get('OUTPUT_DIR', '/app/output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    format = export_settings.get('format', 'jpeg')
    quality = export_settings.get('quality', 90)
    
    filename = f"{photo_id}.{format}"
    output_path = output_dir / filename
    
    # Save image based on format
    if format == 'jpeg':
        cv2.imwrite(str(output_path), image_data, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif format == 'png':
        cv2.imwrite(str(output_path), image_data, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(str(output_path), image_data)
    
    return str(output_path)


def _generate_preview(photo_id: str, image_data: np.ndarray) -> str:
    """Generate preview image"""
    preview_dir = Path('/tmp/photosight/previews')
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    # Resize for preview
    height, width = image_data.shape[:2]
    max_size = 800
    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        preview = cv2.resize(image_data, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        preview = image_data
    
    # Save preview
    preview_path = preview_dir / f"{photo_id}.jpg"
    cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return f"/api/v1/previews/{photo_id}.jpg"


def _cache_processing_result(photo_id: str, result: Dict[str, Any]) -> None:
    """Cache processing result in Redis"""
    redis_client.setex(
        f"result:{photo_id}", 
        3600,  # 1 hour
        json.dumps(result)
    )


def _get_queue_size(queue_name: str) -> int:
    """Get number of pending tasks in queue"""
    from kombu import Connection, Queue
    
    with Connection(os.environ.get('REDIS_URL', 'redis://localhost:6379/0')) as conn:
        queue = Queue(queue_name, connection=conn)
        return queue.queue_len()


def _load_image(file_path: str) -> np.ndarray:
    """Load image from file"""
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


def _export_with_settings(photo_id: str, image_data: np.ndarray, 
                         settings: Dict[str, Any]) -> str:
    """Export image with specific settings"""
    # Implementation would handle resize, format conversion, etc.
    return _save_processed_image(photo_id, image_data, settings)