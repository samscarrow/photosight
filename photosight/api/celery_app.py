"""
Celery configuration for PhotoSight API background tasks
"""

import os
from celery import Celery
from kombu import Exchange, Queue
from datetime import timedelta

# Initialize Celery
celery_app = Celery('photosight')

# Configuration
celery_app.conf.update(
    # Broker settings (Redis)
    broker_url=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    result_backend=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'photosight.api.tasks.process_photo': {'queue': 'high_priority'},
        'photosight.api.tasks.batch_process': {'queue': 'normal'},
        'photosight.api.tasks.generate_preview': {'queue': 'high_priority'},
        'photosight.api.tasks.export_photo': {'queue': 'normal'},
        'photosight.api.tasks.cleanup_old_files': {'queue': 'low_priority'},
    },
    
    # Queue configuration
    task_queues=(
        Queue('high_priority', Exchange('high_priority'), routing_key='high_priority'),
        Queue('normal', Exchange('normal'), routing_key='normal'),
        Queue('low_priority', Exchange('low_priority'), routing_key='low_priority'),
    ),
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
    
    # Time limits
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,  # 10 minutes hard limit
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-previews': {
            'task': 'photosight.api.tasks.cleanup_old_files',
            'schedule': timedelta(hours=1),
            'args': ('previews', 24),  # Clean previews older than 24 hours
        },
        'cleanup-old-exports': {
            'task': 'photosight.api.tasks.cleanup_old_files',
            'schedule': timedelta(hours=6),
            'args': ('exports', 168),  # Clean exports older than 7 days
        },
        'update-processing-stats': {
            'task': 'photosight.api.tasks.update_processing_stats',
            'schedule': timedelta(minutes=5),
        },
    },
    
    # Error handling
    task_default_retry_delay=60,  # 60 seconds
    task_max_retries=3,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Import task modules to register them
celery_app.autodiscover_tasks(['photosight.api'], force=True)