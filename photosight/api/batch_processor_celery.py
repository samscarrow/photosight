"""
Enhanced batch processor using Celery for persistent job queue
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import redis
from celery import group, chain, chord
from celery.result import AsyncResult

from .celery_app import celery_app
from .tasks import process_photo, batch_process
from ..db import get_session
from ..db.models import ProcessingJob, JobStatus, Photo

logger = logging.getLogger(__name__)

# Redis client for job metadata
redis_client = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))


@dataclass
class BatchJobInfo:
    """Information about a batch processing job"""
    job_id: str
    status: JobStatus
    total_items: int
    processed_items: int
    failed_items: int
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: str = 'normal'
    results: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'status': self.status.value,
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'priority': self.priority,
            'results': self.results or []
        }


class CeleryBatchProcessor:
    """Batch processor using Celery for distributed task processing"""
    
    def __init__(self):
        self.redis_client = redis_client
    
    def create_job(self, photo_ids: List[str], recipe: Dict[str, Any],
                   output_format: str = 'jpeg', priority: str = 'normal') -> str:
        """
        Create a new batch processing job
        
        Args:
            photo_ids: List of photo IDs to process
            recipe: Processing recipe
            output_format: Output format (jpeg, png, tiff)
            priority: Job priority (low, normal, high)
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        # Create job record in database
        with get_session() as session:
            job = ProcessingJob(
                id=job_id,
                status=JobStatus.PENDING,
                total_items=len(photo_ids),
                processed_items=0,
                failed_items=0,
                priority=priority,
                created_at=datetime.utcnow(),
                metadata={
                    'photo_ids': photo_ids,
                    'recipe': recipe,
                    'output_format': output_format
                }
            )
            session.add(job)
            session.commit()
        
        # Update recipe with output format
        recipe['export_settings'] = recipe.get('export_settings', {})
        recipe['export_settings']['format'] = output_format
        
        # Submit to Celery
        task = batch_process.apply_async(
            args=[job_id, photo_ids, recipe, priority],
            queue=self._get_queue_for_priority(priority),
            task_id=f"batch-{job_id}"
        )
        
        # Cache job metadata
        self._cache_job_info(job_id, {
            'celery_task_id': task.id,
            'status': 'pending',
            'total': len(photo_ids),
            'priority': priority
        })
        
        logger.info(f"Created batch job {job_id} with {len(photo_ids)} photos")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[BatchJobInfo]:
        """Get current status of a batch job"""
        # Check cache first
        cached = self.redis_client.get(f"job:{job_id}")
        if cached:
            import json
            job_data = json.loads(cached)
            return self._job_data_to_info(job_data)
        
        # Get from database
        with get_session() as session:
            job = session.query(ProcessingJob).filter_by(id=job_id).first()
            if not job:
                return None
            
            # Get Celery task status
            celery_task_id = f"batch-{job_id}"
            task_result = AsyncResult(celery_task_id, app=celery_app)
            
            # Update status based on Celery task
            if task_result.state == 'PENDING':
                job.status = JobStatus.PENDING
            elif task_result.state == 'STARTED':
                job.status = JobStatus.PROCESSING
            elif task_result.state == 'SUCCESS':
                job.status = JobStatus.COMPLETED
            elif task_result.state == 'FAILURE':
                job.status = JobStatus.FAILED
            
            # Get progress from task meta
            if task_result.info and isinstance(task_result.info, dict):
                job.processed_items = task_result.info.get('processed', 0)
                job.failed_items = task_result.info.get('failed', 0)
            
            session.commit()
            
            return BatchJobInfo(
                job_id=job.id,
                status=job.status,
                total_items=job.total_items,
                processed_items=job.processed_items,
                failed_items=job.failed_items,
                progress=(job.processed_items / job.total_items * 100) if job.total_items > 0 else 0,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                priority=job.priority,
                results=job.metadata.get('results', [])
            )
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running batch job"""
        try:
            # Revoke Celery task
            celery_task_id = f"batch-{job_id}"
            celery_app.control.revoke(celery_task_id, terminate=True)
            
            # Update database
            with get_session() as session:
                job = session.query(ProcessingJob).filter_by(id=job_id).first()
                if job:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.utcnow()
                    session.commit()
            
            # Clear cache
            self.redis_client.delete(f"job:{job_id}")
            
            logger.info(f"Cancelled batch job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def retry_failed_items(self, job_id: str) -> Optional[str]:
        """Retry failed items from a batch job"""
        with get_session() as session:
            job = session.query(ProcessingJob).filter_by(id=job_id).first()
            if not job:
                return None
            
            # Get failed photo IDs
            failed_ids = [
                result['photo_id'] 
                for result in job.metadata.get('results', [])
                if result.get('status') == 'failed'
            ]
            
            if not failed_ids:
                return None
            
            # Create new job for failed items
            recipe = job.metadata.get('recipe', {})
            output_format = job.metadata.get('output_format', 'jpeg')
            
            return self.create_job(failed_ids, recipe, output_format, 'high')
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about processing queues"""
        stats = {}
        
        for queue_name in ['high_priority', 'normal', 'low_priority']:
            # Get queue size from Celery
            inspect = celery_app.control.inspect()
            reserved = inspect.reserved()
            active = inspect.active()
            
            queue_size = 0
            active_tasks = 0
            
            if reserved:
                for worker, tasks in reserved.items():
                    queue_size += len([t for t in tasks if t['queue'] == queue_name])
            
            if active:
                for worker, tasks in active.items():
                    active_tasks += len([t for t in tasks if t['queue'] == queue_name])
            
            stats[queue_name] = {
                'pending': queue_size,
                'active': active_tasks,
                'total': queue_size + active_tasks
            }
        
        # Get worker stats
        worker_stats = inspect.stats()
        if worker_stats:
            stats['workers'] = {
                'count': len(worker_stats),
                'details': worker_stats
            }
        
        return stats
    
    def process_single_photo(self, photo_id: str, recipe: Dict[str, Any],
                           priority: str = 'high') -> str:
        """Process a single photo immediately"""
        task = process_photo.apply_async(
            args=[photo_id, recipe, None],
            queue=self._get_queue_for_priority(priority),
            task_id=f"single-{photo_id}-{uuid.uuid4()}"
        )
        
        return task.id
    
    def _get_queue_for_priority(self, priority: str) -> str:
        """Get queue name for priority level"""
        return {
            'low': 'low_priority',
            'normal': 'normal',
            'high': 'high_priority'
        }.get(priority, 'normal')
    
    def _cache_job_info(self, job_id: str, info: Dict[str, Any]) -> None:
        """Cache job information in Redis"""
        import json
        self.redis_client.setex(
            f"job:{job_id}",
            3600,  # 1 hour TTL
            json.dumps(info)
        )
    
    def _job_data_to_info(self, data: Dict[str, Any]) -> BatchJobInfo:
        """Convert job data to BatchJobInfo"""
        return BatchJobInfo(
            job_id=data.get('job_id'),
            status=JobStatus(data.get('status', 'pending')),
            total_items=data.get('total', 0),
            processed_items=data.get('processed', 0),
            failed_items=data.get('failed', 0),
            progress=data.get('progress', 0),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat())),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            priority=data.get('priority', 'normal'),
            results=data.get('results', [])
        )