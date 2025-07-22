"""
Logging utilities for PhotoSight
Provides structured logging and progress tracking
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Provides structured logging with metadata"""
    
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            metadata: Default metadata to include in all logs
        """
        self.logger = logging.getLogger(name)
        self.metadata = metadata or {}
        
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with metadata"""
        data = {**self.metadata, **kwargs}
        if data:
            return f"{message} | {json.dumps(data, default=str)}"
        return message
        
    def debug(self, message: str, **kwargs):
        """Log debug message with metadata"""
        self.logger.debug(self._format_message(message, **kwargs))
        
    def info(self, message: str, **kwargs):
        """Log info message with metadata"""
        self.logger.info(self._format_message(message, **kwargs))
        
    def warning(self, message: str, **kwargs):
        """Log warning message with metadata"""
        self.logger.warning(self._format_message(message, **kwargs))
        
    def error(self, message: str, **kwargs):
        """Log error message with metadata"""
        self.logger.error(self._format_message(message, **kwargs))
        
    def critical(self, message: str, **kwargs):
        """Log critical message with metadata"""
        self.logger.critical(self._format_message(message, **kwargs))


class ProcessingStats:
    """Tracks processing statistics"""
    
    def __init__(self):
        """Initialize processing statistics"""
        self.start_time = datetime.now()
        self.total_files = 0
        self.processed_files = 0
        self.accepted_files = 0
        self.rejected_files = 0
        self.rejection_reasons = {}
        self.errors = []
        self.processing_times = []
        
    def set_total(self, total: int):
        """Set total number of files to process"""
        self.total_files = total
        
    def add_result(self, accepted: bool, rejection_reason: Optional[str] = None,
                   processing_time: Optional[float] = None):
        """
        Add a processing result
        
        Args:
            accepted: Whether file was accepted
            rejection_reason: Reason for rejection if applicable
            processing_time: Time taken to process file
        """
        self.processed_files += 1
        
        if accepted:
            self.accepted_files += 1
        else:
            self.rejected_files += 1
            if rejection_reason:
                self.rejection_reasons[rejection_reason] = \
                    self.rejection_reasons.get(rejection_reason, 0) + 1
                    
        if processing_time:
            self.processing_times.append(processing_time)
            
    def add_error(self, file_path: str, error: str):
        """Add an error"""
        self.errors.append({
            'file': file_path,
            'error': error,
            'time': datetime.now()
        })
        
    def get_progress_percentage(self) -> float:
        """Get progress percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
        
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
        
    def get_average_processing_time(self) -> float:
        """Get average processing time per file"""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
        
    def get_estimated_remaining_time(self) -> float:
        """Get estimated remaining time in seconds"""
        if self.processed_files == 0:
            return 0.0
            
        avg_time = self.get_average_processing_time()
        remaining_files = self.total_files - self.processed_files
        return avg_time * remaining_files
        
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary"""
        elapsed = self.get_elapsed_time()
        
        return {
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'accepted_files': self.accepted_files,
            'rejected_files': self.rejected_files,
            'acceptance_rate': (self.accepted_files / self.processed_files * 100) 
                             if self.processed_files > 0 else 0,
            'rejection_reasons': self.rejection_reasons,
            'errors': len(self.errors),
            'elapsed_time': elapsed,
            'average_time_per_file': self.get_average_processing_time(),
            'files_per_second': self.processed_files / elapsed if elapsed > 0 else 0
        }
        
    def print_summary(self):
        """Print processing summary to console"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total files:      {summary['total_files']}")
        print(f"Processed:        {summary['processed_files']}")
        print(f"Accepted:         {summary['accepted_files']} ({summary['acceptance_rate']:.1f}%)")
        print(f"Rejected:         {summary['rejected_files']}")
        
        if summary['rejection_reasons']:
            print("\nRejection reasons:")
            for reason, count in sorted(summary['rejection_reasons'].items()):
                print(f"  - {reason}: {count}")
                
        print(f"\nErrors:           {summary['errors']}")
        print(f"Elapsed time:     {summary['elapsed_time']:.1f}s")
        print(f"Avg time/file:    {summary['average_time_per_file']:.2f}s")
        print(f"Processing rate:  {summary['files_per_second']:.1f} files/s")
        print("="*60)
        
        if self.errors:
            print("\nERRORS:")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  - {error['file']}: {error['error']}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")


def setup_console_logging(level: str = "INFO", color: bool = True):
    """
    Setup console logging with optional color support
    
    Args:
        level: Logging level
        color: Whether to use colored output
    """
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter
    if color and sys.stdout.isatty():
        # Use colored formatter if supported
        try:
            import colorlog
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s - %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        except ImportError:
            # Fallback to standard formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    root_logger.addHandler(console_handler)