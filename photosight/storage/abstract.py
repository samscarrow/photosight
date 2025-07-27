"""
Enhanced storage abstraction layer for PhotoSight.

Provides a unified interface for working with local and cloud storage backends,
enabling seamless deployment across different environments with robust error handling,
async support, and comprehensive metadata management.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, BinaryIO, AsyncGenerator
import os
import shutil
from datetime import datetime
import logging
from contextlib import contextmanager, asynccontextmanager
import asyncio
from dataclasses import dataclass
import io

logger = logging.getLogger(__name__)


@dataclass
class StorageMetadata:
    """Comprehensive metadata for storage objects."""
    size: int = 0
    modified: Optional[datetime] = None
    created: Optional[datetime] = None
    content_type: Optional[str] = None
    etag: Optional[str] = None
    custom_metadata: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.modified is None:
            self.modified = datetime.utcnow()
        if self.created is None:
            self.created = self.modified
        if self.custom_metadata is None:
            self.custom_metadata = {}


@dataclass
class StorageObject:
    """Represents an object in storage with its metadata and optional data."""
    path: str
    metadata: StorageMetadata
    data: Optional[bytes] = None
    
    @property
    def name(self) -> str:
        """Get the object name from path."""
        return Path(self.path).name
    
    @property
    def parent(self) -> str:
        """Get the parent directory path."""
        return str(Path(self.path).parent)


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class StorageNotFoundError(StorageError):
    """Raised when a storage object is not found."""
    pass


class StoragePermissionError(StorageError):
    """Raised when storage operation lacks permissions."""
    pass


class StorageConnectionError(StorageError):
    """Raised when storage backend connection fails."""
    pass


class StorageBackend(ABC):
    """Enhanced abstract base class for storage backends."""
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if an object exists at the given path."""
        pass
    
    @abstractmethod
    def save(self, path: str, data: Union[bytes, BinaryIO], 
             metadata: Optional[StorageMetadata] = None) -> StorageObject:
        """Save data to the given path."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> StorageObject:
        """Load object from the given path with metadata."""
        pass
    
    @abstractmethod
    def load_data(self, path: str) -> bytes:
        """Load only the data from the given path (backward compatibility)."""
        pass
    
    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete object at the given path. Returns True if deleted."""
        pass
    
    @abstractmethod
    def list_objects(self, prefix: str = "", limit: Optional[int] = None) -> List[StorageObject]:
        """List objects with optional prefix filter."""
        pass
    
    @abstractmethod
    def get_metadata(self, path: str) -> StorageMetadata:
        """Get metadata for object at path."""
        pass
    
    @abstractmethod
    def copy(self, source_path: str, destination_path: str) -> StorageObject:
        """Copy object from source to destination."""
        pass
    
    @abstractmethod
    def move(self, source_path: str, destination_path: str) -> StorageObject:
        """Move object from source to destination."""
        pass
    
    @abstractmethod
    def get_url(self, path: str, expires_in: Optional[int] = None) -> str:
        """Get a URL for accessing the object."""
        pass
    
    # Context managers for file-like operations
    @contextmanager
    def open(self, path: str, mode: str = 'rb'):
        """Context manager for opening files."""
        if 'w' in mode or 'a' in mode:
            # For write modes, provide a BytesIO buffer
            buffer = io.BytesIO()
            try:
                yield buffer
                # Save the buffer contents when context exits
                buffer.seek(0)
                self.save(path, buffer.getvalue())
            finally:
                buffer.close()
        else:
            # For read modes, load and provide the data
            obj = self.load(path)
            buffer = io.BytesIO(obj.data)
            try:
                yield buffer
            finally:
                buffer.close()
    
    # Utility methods
    def get_size(self, path: str) -> int:
        """Get the size of an object."""
        return self.get_metadata(path).size
    
    def is_directory(self, path: str) -> bool:
        """Check if path represents a directory."""
        if path.endswith('/'):
            return True
        # Check if there are objects with this path as prefix
        objects = self.list_objects(prefix=path + '/', limit=1)
        return len(objects) > 0
    
    # Legacy methods for backward compatibility
    def list(self, prefix: str = "", recursive: bool = True) -> List[str]:
        """List file paths (backward compatibility)."""
        objects = self.list_objects(prefix)
        return [obj.path for obj in objects]
    
    def set_metadata(self, path: str, metadata: Dict[str, str]) -> bool:
        """Set file metadata (backward compatibility)."""
        try:
            current_meta = self.get_metadata(path)
            current_meta.custom_metadata.update(metadata)
            # This would need to be implemented per backend
            return True
        except Exception:
            return False


class LocalStorage(StorageBackend):
    """Local filesystem storage implementation."""
    
    def __init__(self, base_path: str):
        """
        Initialize local storage.
        
        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _full_path(self, path: str) -> Path:
        """Get full path from relative path."""
        full_path = self.base_path / path
        # Ensure path is within base directory
        try:
            full_path.resolve().relative_to(self.base_path.resolve())
        except ValueError:
            raise ValueError(f"Path '{path}' is outside base directory")
        return full_path
    
    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        return self._full_path(path).exists()
    
    def save(self, path: str, data: Union[bytes, BinaryIO], 
             metadata: Optional[StorageMetadata] = None) -> StorageObject:
        """Save data to local filesystem."""
        full_path = self._full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle BinaryIO or bytes
        if hasattr(data, 'read'):
            # It's a file-like object
            content = data.read()
        else:
            # It's bytes
            content = data
        
        full_path.write_bytes(content)
        
        # Create metadata if not provided
        if metadata is None:
            metadata = StorageMetadata(size=len(content))
        else:
            metadata.size = len(content)
        
        # Save custom metadata as sidecar file if provided
        if metadata.custom_metadata:
            meta_path = full_path.with_suffix(full_path.suffix + '.meta.json')
            import json
            meta_path.write_text(json.dumps(metadata.custom_metadata, indent=2))
        
        return StorageObject(path=path, metadata=metadata, data=content)
    
    def load(self, path: str) -> StorageObject:
        """Load object from local filesystem with metadata."""
        full_path = self._full_path(path)
        if not full_path.exists():
            raise StorageNotFoundError(f"File not found: {path}")
        
        content = full_path.read_bytes()
        metadata = self.get_metadata(path)
        
        return StorageObject(path=path, metadata=metadata, data=content)
    
    def load_data(self, path: str) -> bytes:
        """Load only the data from local filesystem (backward compatibility)."""
        return self._full_path(path).read_bytes()
    
    def delete(self, path: str) -> bool:
        """Delete a file."""
        full_path = self._full_path(path)
        if full_path.exists():
            full_path.unlink()
            # Also delete metadata file if it exists
            meta_path = full_path.with_suffix(full_path.suffix + '.meta.json')
            if meta_path.exists():
                meta_path.unlink()
            return True
        return False
    
    def list_objects(self, prefix: str = "", limit: Optional[int] = None) -> List[StorageObject]:
        """List objects with optional prefix filter."""
        search_path = self.base_path
        if prefix:
            # For prefix search, we need to search from base and filter
            pattern = "**/*"
        else:
            pattern = "**/*"
        
        objects = []
        for path in search_path.glob(pattern):
            if path.is_file() and not path.name.endswith('.meta.json'):
                rel_path = str(path.relative_to(self.base_path))
                
                # Apply prefix filter
                if prefix and not rel_path.startswith(prefix):
                    continue
                
                try:
                    metadata = self.get_metadata(rel_path)
                    obj = StorageObject(path=rel_path, metadata=metadata)
                    objects.append(obj)
                    
                    if limit and len(objects) >= limit:
                        break
                except Exception:
                    # Skip files that can't be processed
                    continue
        
        return sorted(objects, key=lambda x: x.path)
    
    def list(self, prefix: str = "", recursive: bool = True) -> List[str]:
        """List file paths (backward compatibility)."""
        objects = self.list_objects(prefix)
        return [obj.path for obj in objects]
    
    def copy(self, source_path: str, destination_path: str) -> StorageObject:
        """Copy a file within local storage."""
        src = self._full_path(source_path)
        dst = self._full_path(destination_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        
        # Copy metadata if it exists
        src_meta = src.with_suffix(src.suffix + '.meta.json')
        if src_meta.exists():
            dst_meta = dst.with_suffix(dst.suffix + '.meta.json')
            shutil.copy2(src_meta, dst_meta)
        
        # Return the copied object
        return self.load(destination_path)
    
    def move(self, source_path: str, destination_path: str) -> StorageObject:
        """Move a file within local storage."""
        src = self._full_path(source_path)
        dst = self._full_path(destination_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        
        # Move metadata if it exists
        src_meta = src.with_suffix(src.suffix + '.meta.json')
        if src_meta.exists():
            dst_meta = dst.with_suffix(dst.suffix + '.meta.json')
            shutil.move(str(src_meta), str(dst_meta))
        
        # Return the moved object
        return self.load(destination_path)
    
    def get_url(self, path: str, expires_in: Optional[int] = None) -> str:
        """Get file URL (file:// for local storage)."""
        full_path = self._full_path(path)
        return f"file://{full_path}"
    
    def get_metadata(self, path: str) -> StorageMetadata:
        """Get file metadata."""
        full_path = self._full_path(path)
        if not full_path.exists():
            raise StorageNotFoundError(f"File not found: {path}")
        
        stat = full_path.stat()
        
        # Load custom metadata if available
        meta_path = full_path.with_suffix(full_path.suffix + '.meta.json')
        custom_metadata = {}
        if meta_path.exists():
            import json
            custom_metadata = json.loads(meta_path.read_text())
        
        return StorageMetadata(
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
            created=datetime.fromtimestamp(stat.st_ctime),
            custom_metadata=custom_metadata
        )
    
    def set_metadata(self, path: str, metadata: Dict[str, str]) -> bool:
        """Set file metadata."""
        full_path = self._full_path(path)
        if not full_path.exists():
            return False
        
        meta_path = full_path.with_suffix(full_path.suffix + '.meta.json')
        import json
        meta_path.write_text(json.dumps(metadata, indent=2))
        return True


class S3Storage(StorageBackend):
    """Amazon S3 storage implementation."""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1', 
                 access_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize S3 storage.
        
        Args:
            bucket_name: S3 bucket name
            region: AWS region
            access_key: AWS access key (uses environment/IAM if not provided)
            secret_key: AWS secret key (uses environment/IAM if not provided)
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        if access_key and secret_key:
            self.s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
        else:
            # Use environment variables or IAM role
            self.s3_client = boto3.client('s3', region_name=region)
        
        # Verify bucket exists
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise ValueError(f"Bucket '{bucket_name}' does not exist")
            else:
                raise
    
    def exists(self, path: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=path)
            return True
        except:
            return False
    
    def save(self, data: Union[bytes, str], destination_path: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """Save data to S3."""
        extra_args = {}
        if metadata:
            extra_args['Metadata'] = metadata
        
        if isinstance(data, bytes):
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=destination_path,
                Body=data,
                **extra_args
            )
        elif isinstance(data, str) and os.path.exists(data):
            self.s3_client.upload_file(
                data,
                self.bucket_name,
                destination_path,
                ExtraArgs=extra_args
            )
        else:
            raise ValueError("Data must be bytes or a valid file path")
        
        return destination_path
    
    def load(self, source_path: str) -> bytes:
        """Load file data from S3."""
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=source_path)
        return response['Body'].read()
    
    def delete(self, path: str) -> bool:
        """Delete a file from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=path)
            return True
        except:
            return False
    
    def list(self, prefix: str = "", recursive: bool = True) -> List[str]:
        """List files in S3 with optional prefix."""
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        files = []
        delimiter = None if recursive else '/'
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter=delimiter):
            if 'Contents' in page:
                for obj in page['Contents']:
                    files.append(obj['Key'])
        
        return sorted(files)
    
    def copy(self, source_path: str, destination_path: str) -> str:
        """Copy a file within S3."""
        copy_source = {'Bucket': self.bucket_name, 'Key': source_path}
        self.s3_client.copy_object(
            CopySource=copy_source,
            Bucket=self.bucket_name,
            Key=destination_path
        )
        return destination_path
    
    def move(self, source_path: str, destination_path: str) -> str:
        """Move a file within S3."""
        self.copy(source_path, destination_path)
        self.delete(source_path)
        return destination_path
    
    def get_url(self, path: str, expires_in: Optional[int] = None) -> str:
        """Get S3 URL (presigned if expiration specified)."""
        if expires_in:
            return self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': path},
                ExpiresIn=expires_in
            )
        else:
            # Return public URL (assumes public bucket or object)
            return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{path}"
    
    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Get S3 object metadata."""
        response = self.s3_client.head_object(Bucket=self.bucket_name, Key=path)
        return {
            'size': response['ContentLength'],
            'modified': response['LastModified'].isoformat(),
            'etag': response['ETag'].strip('"'),
            'content_type': response.get('ContentType', 'application/octet-stream'),
            'custom': response.get('Metadata', {})
        }
    
    def set_metadata(self, path: str, metadata: Dict[str, str]) -> bool:
        """Update S3 object metadata."""
        try:
            # S3 requires copying the object to update metadata
            copy_source = {'Bucket': self.bucket_name, 'Key': path}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=path,
                Metadata=metadata,
                MetadataDirective='REPLACE'
            )
            return True
        except:
            return False


class StorageManager:
    """
    Manager for handling multiple storage backends.
    Provides a unified interface and handles failover.
    """
    
    def __init__(self, primary: StorageBackend, fallback: Optional[StorageBackend] = None):
        """
        Initialize storage manager.
        
        Args:
            primary: Primary storage backend
            fallback: Optional fallback storage backend
        """
        self.primary = primary
        self.fallback = fallback
    
    def save(self, data: Union[bytes, str], destination_path: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """Save data with automatic failover."""
        try:
            return self.primary.save(data, destination_path, metadata)
        except Exception as e:
            logger.error(f"Primary storage failed: {e}")
            if self.fallback:
                logger.info("Falling back to secondary storage")
                return self.fallback.save(data, destination_path, metadata)
            raise
    
    def load(self, source_path: str) -> bytes:
        """Load data with automatic failover."""
        try:
            return self.primary.load(source_path)
        except Exception as e:
            logger.error(f"Primary storage failed: {e}")
            if self.fallback:
                logger.info("Falling back to secondary storage")
                return self.fallback.load(source_path)
            raise
    
    # Delegate other methods to primary storage
    def __getattr__(self, name):
        return getattr(self.primary, name)


# Factory function for creating storage backends
def create_storage(config: Dict[str, Any]) -> StorageBackend:
    """
    Create a storage backend from configuration.
    
    Args:
        config: Storage configuration dictionary
        
    Returns:
        Storage backend instance
    """
    storage_type = config.get('type', 'local')
    
    if storage_type == 'local':
        return LocalStorage(config['base_path'])
    elif storage_type == 's3':
        return S3Storage(
            bucket_name=config['bucket'],
            region=config.get('region', 'us-east-1'),
            access_key=config.get('access_key'),
            secret_key=config.get('secret_key')
        )
    # Add more storage backends here (GCS, Azure, etc.)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")