"""
Improved Celery tasks with chord pattern for robust batch processing.
Provides better error handling, progress tracking, and completion guarantees.
"""

import os
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import uuid
import traceback

from celery import Task, group, chord, chain, signature
from celery.exceptions import SoftTimeLimitExceeded, Retry
import redis

from .celery_app import celery_app
from ..processing import PhotoProcessor, ProcessingRecipe
from ..processing.preview import PreviewGenerator
from ..db import get_session
from ..db.models import Photo, ProcessingJob, JobStatus
from .websocket_events_improved import emitter
from ..storage.abstract import create_storage, StorageManager

logger = logging.getLogger(__name__)

# Redis client for caching and coordination
redis_client = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))

# Storage configuration
storage_config = {
    'type': os.environ.get('STORAGE_TYPE', 'local'),
    'base_path': os.environ.get('STORAGE_PATH', '/var/photosight/storage'),
    # S3 config if using S3
    'bucket': os.environ.get('S3_BUCKET'),
    'region': os.environ.get('S3_REGION', 'us-east-1'),
    'access_key': os.environ.get('AWS_ACCESS_KEY_ID'),
    'secret_key': os.environ.get('AWS_SECRET_ACCESS_KEY')
}

# Initialize storage manager with fallback
primary_storage = create_storage(storage_config)
fallback_storage = create_storage({'type': 'local', 'base_path': '/tmp/photosight/fallback'})
storage_manager = StorageManager(primary_storage, fallback_storage)


class PhotoProcessingTask(Task):
    """Enhanced base task class with improved error handling and notifications"""
    
    autoretry_for = (ConnectionError, TimeoutError)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True
    retry_backoff_max = 600  # Max 10 minutes between retries
    retry_jitter = True
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with detailed error reporting"""
        logger.error(f"Task {task_id} failed: {exc}\n{einfo}")
        
        # Extract identifiers
        job_id = kwargs.get('job_id')
        photo_id = kwargs.get('photo_id') or (args[0] if args else None)
        
        # Update job status in database
        if job_id:
            with get_session() as session:
                job = session.query(ProcessingJob).filter_by(id=job_id).first()
                if job:
                    job.failed_count += 1
                    job.error_message = f"{exc.__class__.__name__}: {str(exc)}"
                    job.error_details = {
                        'task_id': task_id,
                        'photo_id': photo_id,
                        'traceback': str(einfo),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    session.commit()
        
        # Emit detailed error notification
        emitter.error(
            entity_type='photo' if photo_id else 'job',
            entity_id=str(photo_id or job_id),
            error_message=str(exc),
            error_code=exc.__class__.__name__,
            details={
                'task_id': task_id,
                'job_id': job_id,
                'photo_id': photo_id,
                'retry_count': self.request.retries
            }
        )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry with progress notification"""
        logger.warning(f"Task {task_id} retrying (attempt {self.request.retries + 1}): {exc}")
        
        job_id = kwargs.get('job_id')
        if job_id:
            emitter.progress_update(
                job_id=job_id,
                progress=0,
                stage='retrying',
                message=f"Retrying after error: {exc}",
                metadata={'retry_count': self.request.retries + 1}
            )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success with metrics"""
        logger.info(f"Task {task_id} completed successfully")
        # Could add metrics collection here


@celery_app.task(base=PhotoProcessingTask, bind=True, name='photosight.process_photo')
def process_photo(self, photo_id: str, recipe: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a single photo with the given recipe.
    
    Args:
        photo_id: ID of the photo to process
        recipe: Processing recipe dictionary
        job_id: Optional job ID for tracking
        
    Returns:
        Dict with processing results
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Processing photo {photo_id} for job {job_id}")
        
        # Get photo from database
        with get_session() as session:
            photo = session.query(Photo).filter_by(id=photo_id).first()
            if not photo:
                raise ValueError(f"Photo {photo_id} not found")
            
            # Update photo status
            photo.processing_status = 'processing'
            photo.processing_started_at = start_time
            session.commit()
            
            file_path = photo.file_path
            photo_data = {
                'id': photo.id,
                'filename': photo.filename,
                'width': photo.width,
                'height': photo.height
            }
        
        # Load image data using storage manager
        image_data = storage_manager.load(file_path)
        
        # Initialize processor
        processor = PhotoProcessor()
        recipe_obj = ProcessingRecipe(**recipe)
        
        # Process the image
        processed_data, metadata = processor.process(
            image_data=image_data,
            recipe=recipe_obj,
            photo_metadata=photo_data
        )
        
        # Generate output filename
        output_filename = f"{photo_id}_processed_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path = f"processed/{job_id}/{output_filename}" if job_id else f"processed/{output_filename}"
        
        # Save processed image
        final_path = storage_manager.save(
            data=processed_data,
            destination_path=output_path,
            metadata={
                'photo_id': photo_id,
                'job_id': job_id,
                'recipe': str(recipe),
                'processing_time': str(datetime.utcnow() - start_time)
            }
        )
        
        # Update database
        with get_session() as session:
            photo = session.query(Photo).filter_by(id=photo_id).first()
            photo.processed_path = final_path
            photo.processing_status = 'completed'
            photo.processing_completed_at = datetime.utcnow()
            photo.processing_recipe = recipe
            photo.processing_metadata = metadata
            session.commit()
        
        # Get URL for the processed image
        processed_url = storage_manager.get_url(final_path, expires_in=3600)
        
        # Emit completion event
        result = {
            'photo_id': photo_id,
            'processed_url': processed_url,
            'processing_time': (datetime.utcnow() - start_time).total_seconds(),
            'metadata': metadata
        }
        
        emitter.processing_complete(photo_id, result, job_id)
        
        return result
        
    except Exception as e:
        # Update photo status on error
        with get_session() as session:
            photo = session.query(Photo).filter_by(id=photo_id).first()
            if photo:
                photo.processing_status = 'failed'
                photo.processing_error = str(e)
                session.commit()
        
        raise


@celery_app.task(name='photosight.finalize_batch_job')
def finalize_batch_job(results: List[Dict[str, Any]], job_id: str) -> Dict[str, Any]:
    """
    Finalize a batch processing job.
    This task runs after all photos in a batch have been processed.
    
    Args:
        results: List of results from individual photo processing tasks
        job_id: Batch job ID
        
    Returns:
        Final job summary
    """
    logger.info(f"Finalizing batch job {job_id}")
    
    try:
        # Calculate summary statistics
        successful_results = [r for r in results if r and not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        total_processing_time = sum(r.get('processing_time', 0) for r in successful_results)
        
        # Update job in database
        with get_session() as session:
            job = session.query(ProcessingJob).filter_by(id=job_id).first()
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            job.status = JobStatus.COMPLETED if not failed_results else JobStatus.PARTIAL
            job.completed_at = datetime.utcnow()
            job.processed_count = len(successful_results)
            job.failed_count = len(failed_results)
            job.total_processing_time = total_processing_time
            
            # Store detailed results
            job.results = {
                'successful': [r['photo_id'] for r in successful_results],
                'failed': [str(e) for e in failed_results],
                'average_processing_time': total_processing_time / len(successful_results) if successful_results else 0
            }
            
            session.commit()
            
            summary = {
                'job_id': job_id,
                'status': job.status,
                'total_photos': job.total_count,
                'processed_photos': job.processed_count,
                'failed_photos': job.failed_count,
                'processing_time': total_processing_time,
                'completed_at': job.completed_at.isoformat()
            }
        
        # Send notifications
        emitter.batch_complete(job_id, summary)
        
        # Send webhook if configured
        _send_webhook_notification(job_id, summary)
        
        # Cleanup temporary data
        _cleanup_job_data(job_id)
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to finalize job {job_id}: {e}")
        
        # Update job status to failed
        with get_session() as session:
            job = session.query(ProcessingJob).filter_by(id=job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.error_message = f"Finalization failed: {str(e)}"
                job.completed_at = datetime.utcnow()
                session.commit()
        
        raise


@celery_app.task(name='photosight.batch_process')
def batch_process(photo_ids: List[str], recipe: Dict[str, Any], 
                  job_name: Optional[str] = None, 
                  webhook_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Process multiple photos as a batch using Celery chord.
    
    Args:
        photo_ids: List of photo IDs to process
        recipe: Processing recipe
        job_name: Optional name for the job
        webhook_url: Optional webhook for completion notification
        
    Returns:
        Job information including job_id and task_id
    """
    # Create job record
    job_id = str(uuid.uuid4())
    
    with get_session() as session:
        job = ProcessingJob(
            id=job_id,
            name=job_name or f"Batch job {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            status=JobStatus.PENDING,
            total_count=len(photo_ids),
            processed_count=0,
            failed_count=0,
            created_at=datetime.utcnow(),
            recipe=recipe,
            webhook_url=webhook_url
        )
        session.add(job)
        session.commit()
    
    # Store job metadata in Redis for quick access
    job_meta = {
        'job_id': job_id,
        'total_photos': len(photo_ids),
        'recipe': recipe,
        'created_at': datetime.utcnow().isoformat()
    }
    redis_client.setex(f"job:{job_id}", 86400, json.dumps(job_meta))  # 24 hour TTL
    
    # Create a group of processing tasks
    header = group(
        process_photo.s(photo_id, recipe, job_id=job_id)
        for photo_id in photo_ids
    )
    
    # Create a chord that executes finalize_batch_job after all photos are processed
    # The chord ensures the finalizer runs even if some photos fail
    result = chord(header)(finalize_batch_job.s(job_id=job_id))
    
    # Update job with task ID
    with get_session() as session:
        job = session.query(ProcessingJob).filter_by(id=job_id).first()
        job.celery_task_id = result.id
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        session.commit()
    
    # Emit initial progress
    emitter.progress_update(
        job_id=job_id,
        progress=0,
        stage='started',
        message=f'Processing {len(photo_ids)} photos'
    )
    
    return {
        'job_id': job_id,
        'task_id': result.id,
        'total_photos': len(photo_ids),
        'status': 'processing'
    }


@celery_app.task(name='photosight.generate_preview')
def generate_preview(photo_id: str, recipe: Dict[str, Any], 
                     session_id: Optional[str] = None,
                     size: str = 'medium') -> Dict[str, Any]:
    """
    Generate a preview for a photo with the given recipe.
    
    Args:
        photo_id: Photo ID
        recipe: Processing recipe
        session_id: Optional session ID for updates
        size: Preview size (small, medium, large)
        
    Returns:
        Preview information
    """
    try:
        # Get photo data
        with get_session() as session:
            photo = session.query(Photo).filter_by(id=photo_id).first()
            if not photo:
                raise ValueError(f"Photo {photo_id} not found")
            
            file_path = photo.file_path
        
        # Load image data
        image_data = storage_manager.load(file_path)
        
        # Generate preview
        preview_gen = PreviewGenerator()
        preview_data = preview_gen.generate(
            image_data=image_data,
            recipe=ProcessingRecipe(**recipe),
            size=size
        )
        
        # Save preview
        preview_filename = f"{photo_id}_preview_{size}_{uuid.uuid4().hex[:8]}.jpg"
        preview_path = f"previews/{session_id}/{preview_filename}" if session_id else f"previews/{preview_filename}"
        
        final_path = storage_manager.save(
            data=preview_data,
            destination_path=preview_path,
            metadata={
                'photo_id': photo_id,
                'session_id': session_id,
                'recipe': str(recipe),
                'size': size
            }
        )
        
        # Get preview URL
        preview_url = storage_manager.get_url(final_path, expires_in=3600)
        
        # Emit preview update
        emitter.preview_update(
            photo_id=photo_id,
            preview_url=preview_url,
            session_id=session_id,
            recipe=recipe
        )
        
        return {
            'photo_id': photo_id,
            'preview_url': preview_url,
            'size': size
        }
        
    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        raise


# Helper functions

def _send_webhook_notification(job_id: str, summary: Dict[str, Any]) -> None:
    """Send webhook notification for job completion."""
    try:
        with get_session() as session:
            job = session.query(ProcessingJob).filter_by(id=job_id).first()
            if job and job.webhook_url:
                import requests
                response = requests.post(
                    job.webhook_url,
                    json={
                        'event': 'batch_complete',
                        'job_id': job_id,
                        'summary': summary
                    },
                    timeout=10
                )
                logger.info(f"Webhook sent to {job.webhook_url}, status: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to send webhook: {e}")


def _cleanup_job_data(job_id: str) -> None:
    """Cleanup temporary data for a completed job."""
    try:
        # Remove Redis metadata
        redis_client.delete(f"job:{job_id}")
        
        # Could also clean up temporary files if needed
        logger.info(f"Cleaned up data for job {job_id}")
    except Exception as e:
        logger.error(f"Cleanup failed for job {job_id}: {e}")


# Additional utility tasks

@celery_app.task(name='photosight.cleanup_old_jobs')
def cleanup_old_jobs(days: int = 30) -> Dict[str, int]:
    """
    Clean up old completed jobs and their data.
    
    Args:
        days: Number of days to keep completed jobs
        
    Returns:
        Cleanup statistics
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with get_session() as session:
        old_jobs = session.query(ProcessingJob).filter(
            ProcessingJob.completed_at < cutoff_date,
            ProcessingJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED])
        ).all()
        
        cleaned_count = 0
        for job in old_jobs:
            try:
                # Clean up stored files
                if job.results:
                    for photo_id in job.results.get('successful', []):
                        # Could delete processed files if needed
                        pass
                
                # Delete job record
                session.delete(job)
                cleaned_count += 1
                
            except Exception as e:
                logger.error(f"Failed to clean up job {job.id}: {e}")
        
        session.commit()
    
    return {'cleaned_jobs': cleaned_count}


# Schedule periodic cleanup
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'cleanup-old-jobs': {
        'task': 'photosight.cleanup_old_jobs',
        'schedule': crontab(hour=2, minute=0),  # Run at 2 AM daily
        'args': (30,)  # Keep jobs for 30 days
    }
}