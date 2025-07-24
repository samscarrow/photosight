"""
Queue-based processing integration for PhotoSight
Provides high-level interface to the Redis queue infrastructure
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib

import aiohttp
import redis
from redis import asyncio as aioredis

logger = logging.getLogger(__name__)


class Priority(str, Enum):
    """Processing priority levels"""
    CRITICAL = "critical"  # Wedding/event in progress
    HIGH = "high"         # Client work
    NORMAL = "normal"     # Regular processing
    LOW = "low"           # Test shots, experiments


class ProcessingType(str, Enum):
    """Types of processing workflows"""
    FULL = "full"                # Complete pipeline
    QUICK = "quick"              # Quick preview + basic analysis
    ANALYSIS_ONLY = "analysis"   # Just analysis, no processing
    EXPORT_ONLY = "export"       # Just generate exports
    CUSTOM = "custom"            # Custom task list


@dataclass
class QueuedPhoto:
    """Represents a photo queued for processing"""
    photo_id: int
    file_path: str
    import_id: str
    priority: Priority
    status: str
    queued_at: datetime
    metadata: Dict[str, Any]


class PhotoSightQueueClient:
    """Client for interacting with PhotoSight queue infrastructure"""
    
    def __init__(
        self,
        api_url: str = None,
        redis_url: str = None,
        timeout: int = 30
    ):
        # Use cluster services by default
        self.api_url = api_url or "http://photosight-queue-api.photosight-ai.svc.cluster.local:8000"
        self.redis_url = redis_url or "redis://redis.photosight-ai.svc.cluster.local:6379"
        self.timeout = timeout
        
        # Connections
        self._session: Optional[aiohttp.ClientSession] = None
        self._redis: Optional[aioredis.Redis] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def connect(self):
        """Initialize connections"""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        
        if not self._redis:
            self._redis = await aioredis.from_url(self.redis_url)
            
    async def close(self):
        """Close connections"""
        if self._session:
            await self._session.close()
            self._session = None
            
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    async def import_photo(
        self,
        file_path: Union[str, Path],
        project_id: Optional[int] = None,
        priority: Priority = Priority.NORMAL,
        processing_type: ProcessingType = ProcessingType.FULL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Queue a photo for import and processing
        
        Args:
            file_path: Path to the photo file
            project_id: Optional project to associate with
            priority: Processing priority
            processing_type: Type of processing workflow
            metadata: Additional metadata
            
        Returns:
            Import job information
        """
        if not self._session:
            await self.connect()
            
        payload = {
            "file_path": str(file_path),
            "project_id": project_id,
            "priority": priority.value,
            "processing_type": processing_type.value,
            "metadata": metadata or {}
        }
        
        async with self._session.post(
            f"{self.api_url}/photos/import",
            json=payload
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def import_batch(
        self,
        file_paths: List[Union[str, Path]],
        project_id: Optional[int] = None,
        priority: Priority = Priority.NORMAL,
        processing_type: ProcessingType = ProcessingType.FULL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queue multiple photos for import
        
        Returns:
            List of import job information
        """
        tasks = []
        for file_path in file_paths:
            task = self.import_photo(
                file_path=file_path,
                project_id=project_id,
                priority=priority,
                processing_type=processing_type,
                metadata=metadata
            )
            tasks.append(task)
            
        return await asyncio.gather(*tasks)
    
    async def reprocess_photo(
        self,
        photo_id: int,
        tasks: List[str],
        priority: Priority = Priority.NORMAL
    ) -> Dict[str, Any]:
        """
        Reprocess specific tasks for a photo
        
        Args:
            photo_id: Photo ID to reprocess
            tasks: List of task types (e.g., ['blur', 'face', 'object'])
            priority: Processing priority
            
        Returns:
            Job information
        """
        if not self._session:
            await self.connect()
            
        async with self._session.post(
            f"{self.api_url}/photos/{photo_id}/reprocess",
            json={
                "tasks": tasks,
                "priority": priority.value
            }
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific job"""
        if not self._session:
            await self.connect()
            
        async with self._session.get(
            f"{self.api_url}/jobs/{job_id}"
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get result of a completed job"""
        if not self._session:
            await self.connect()
            
        async with self._session.get(
            f"{self.api_url}/jobs/{job_id}/result"
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def wait_for_job(
        self,
        job_id: str,
        timeout: int = 300,
        poll_interval: float = 1.0
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete
        
        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check status
            
        Returns:
            Job result
            
        Raises:
            TimeoutError: If job doesn't complete in time
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            status = await self.get_job_status(job_id)
            
            if status.get("status") == "completed":
                return await self.get_job_result(job_id)
            elif status.get("status") == "failed":
                raise Exception(f"Job failed: {status.get('error')}")
            
            await asyncio.sleep(poll_interval)
            
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get status of all queues"""
        if not self._session:
            await self.connect()
            
        async with self._session.get(
            f"{self.api_url}/queue/status"
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def create_batch_job(
        self,
        photo_ids: List[int],
        recipe_id: Optional[int] = None,
        tasks: Optional[List[str]] = None,
        priority: Priority = Priority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a batch processing job
        
        Args:
            photo_ids: List of photo IDs to process
            recipe_id: Optional recipe to apply
            tasks: Optional list of specific tasks
            priority: Processing priority
            metadata: Additional metadata
            
        Returns:
            Batch job information
        """
        if not self._session:
            await self.connect()
            
        payload = {
            "photo_ids": photo_ids,
            "priority": priority.value,
            "metadata": metadata or {}
        }
        
        if recipe_id:
            payload["recipe_id"] = recipe_id
        if tasks:
            payload["tasks"] = tasks
            
        async with self._session.post(
            f"{self.api_url}/batch/process",
            json=payload
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status of a batch job"""
        if not self._session:
            await self.connect()
            
        async with self._session.get(
            f"{self.api_url}/batch/{batch_id}/status"
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def subscribe_to_events(
        self,
        event_types: List[str],
        callback: callable
    ):
        """
        Subscribe to real-time processing events
        
        Args:
            event_types: List of event types to subscribe to
            callback: Async function to call with event data
        """
        if not self._redis:
            await self.connect()
            
        pubsub = self._redis.pubsub()
        
        # Subscribe to channels
        channels = [f"events:{event_type}" for event_type in event_types]
        await pubsub.subscribe(*channels)
        
        # Listen for messages
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    event_data = json.loads(message["data"])
                    await callback(event_data)
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics summary"""
        if not self._session:
            await self.connect()
            
        async with self._session.get(
            f"{self.api_url}/metrics/summary"
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


# Convenience functions for common operations
async def queue_photo_for_processing(
    file_path: Union[str, Path],
    priority: Priority = Priority.NORMAL,
    wait: bool = False
) -> Union[Dict[str, Any], QueuedPhoto]:
    """
    Simple function to queue a photo for processing
    
    Args:
        file_path: Path to photo
        priority: Processing priority
        wait: Whether to wait for completion
        
    Returns:
        Import info or completed result if wait=True
    """
    async with PhotoSightQueueClient() as client:
        result = await client.import_photo(
            file_path=file_path,
            priority=priority
        )
        
        if wait:
            import_id = result["import_id"]
            # Wait for import job
            import_result = await client.wait_for_job(f"{import_id}-import")
            
            # Get photo ID from result
            photo_id = import_result.get("photo_id")
            
            return QueuedPhoto(
                photo_id=photo_id,
                file_path=str(file_path),
                import_id=import_id,
                priority=priority,
                status="completed",
                queued_at=datetime.utcnow(),
                metadata=import_result
            )
        
        return result


async def get_yolo_detections(photo_id: int) -> Optional[Dict[str, Any]]:
    """
    Get YOLO object detection results for a photo
    
    Args:
        photo_id: Photo ID
        
    Returns:
        Detection results or None if not processed
    """
    async with PhotoSightQueueClient() as client:
        # Check Redis cache first
        if client._redis:
            # Get photo checksum from DB
            # For now, use photo_id as cache key
            cache_key = f"cache:yolo:photo_{photo_id}"
            cached = await client._redis.get(cache_key)
            
            if cached:
                return json.loads(cached)
        
    return None


# Integration with existing PhotoSight code
class QueuedProcessor:
    """Adapter to use queue-based processing with existing PhotoSight code"""
    
    def __init__(self, client: Optional[PhotoSightQueueClient] = None):
        self.client = client or PhotoSightQueueClient()
        
    async def process_raw_file(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        recipe_id: Optional[int] = None,
        priority: Priority = Priority.NORMAL
    ) -> Dict[str, Any]:
        """
        Process a RAW file using the queue system
        
        Compatible with existing PhotoSight API
        """
        # Queue for import and processing
        result = await self.client.import_photo(
            file_path=file_path,
            priority=priority,
            processing_type=ProcessingType.FULL,
            metadata={
                "output_dir": str(output_dir) if output_dir else None,
                "recipe_id": recipe_id
            }
        )
        
        # Wait for completion if output requested
        if output_dir:
            import_id = result["import_id"]
            # Wait for export job
            export_result = await self.client.wait_for_job(
                f"{import_id}-export",
                timeout=600  # 10 minutes for full processing
            )
            
            return export_result
            
        return result