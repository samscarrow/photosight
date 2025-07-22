"""
File protection utilities for PhotoSight
Ensures source files are never modified during processing
"""

import os
import stat
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from contextlib import contextmanager
import hashlib
import logging

logger = logging.getLogger(__name__)


class FileProtector:
    """Protects source files from accidental modification"""
    
    def __init__(self, verify_checksums: bool = True, 
                 create_backups: bool = False,
                 read_only_mode: bool = True):
        """
        Initialize file protector
        
        Args:
            verify_checksums: Verify file integrity after operations
            create_backups: Create backups before any operations
            read_only_mode: Set files to read-only during processing
        """
        self.verify_checksums = verify_checksums
        self.create_backups = create_backups
        self.read_only_mode = read_only_mode
        self._protected_files: Dict[Path, Dict] = {}
        
    def protect_file(self, file_path: Path) -> Dict:
        """
        Protect a file by recording its state and optionally making it read-only
        
        Args:
            file_path: Path to file to protect
            
        Returns:
            Protection info dictionary
        """
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Get current file stats
        stat_info = file_path.stat()
        protection_info = {
            'path': file_path,
            'original_mode': stat_info.st_mode,
            'size': stat_info.st_size,
            'mtime': stat_info.st_mtime,
            'checksum': None,
            'backup_path': None,
            'is_protected': False
        }
        
        # Calculate checksum if requested
        if self.verify_checksums:
            protection_info['checksum'] = self._calculate_checksum(file_path)
            
        # Create backup if requested
        if self.create_backups:
            backup_path = self._create_backup(file_path)
            protection_info['backup_path'] = backup_path
            
        # Make file read-only if requested
        if self.read_only_mode and not self._is_read_only(file_path):
            os.chmod(file_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            protection_info['is_protected'] = True
            logger.debug(f"Protected file (read-only): {file_path}")
            
        self._protected_files[file_path] = protection_info
        return protection_info
        
    def unprotect_file(self, file_path: Path) -> None:
        """
        Remove protection from a file
        
        Args:
            file_path: Path to file to unprotect
        """
        file_path = Path(file_path).resolve()
        
        if file_path not in self._protected_files:
            return
            
        info = self._protected_files[file_path]
        
        # Restore original permissions
        if info['is_protected']:
            os.chmod(file_path, info['original_mode'])
            logger.debug(f"Unprotected file: {file_path}")
            
        # Verify checksum if requested
        if self.verify_checksums and info['checksum']:
            current_checksum = self._calculate_checksum(file_path)
            if current_checksum != info['checksum']:
                logger.error(f"File was modified while protected: {file_path}")
                raise RuntimeError(f"Protected file was modified: {file_path}")
                
        del self._protected_files[file_path]
        
    def unprotect_all(self) -> None:
        """Remove protection from all protected files"""
        files_to_unprotect = list(self._protected_files.keys())
        for file_path in files_to_unprotect:
            self.unprotect_file(file_path)
            
    @contextmanager
    def protect_files(self, file_paths: List[Path]):
        """
        Context manager to protect multiple files
        
        Args:
            file_paths: List of files to protect
            
        Yields:
            List of protection info dictionaries
        """
        protected_info = []
        
        try:
            # Protect all files
            for file_path in file_paths:
                info = self.protect_file(file_path)
                protected_info.append(info)
                
            yield protected_info
            
        finally:
            # Unprotect all files
            for file_path in file_paths:
                try:
                    self.unprotect_file(file_path)
                except Exception as e:
                    logger.error(f"Error unprotecting {file_path}: {e}")
                    
    @contextmanager
    def safe_working_copy(self, source_path: Path, 
                         cleanup: bool = True) -> Path:
        """
        Create a safe working copy of a file for processing
        
        Args:
            source_path: Original file path
            cleanup: Whether to delete the copy after use
            
        Yields:
            Path to the working copy
        """
        source_path = Path(source_path).resolve()
        
        # Protect the source file
        self.protect_file(source_path)
        
        # Create temporary copy
        temp_dir = Path(tempfile.mkdtemp(prefix='photosight_safe_'))
        working_copy = temp_dir / source_path.name
        
        try:
            # Copy file with metadata
            shutil.copy2(source_path, working_copy)
            logger.debug(f"Created working copy: {working_copy}")
            
            yield working_copy
            
        finally:
            # Cleanup
            if cleanup and temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up working copy: {working_copy}")
                
            # Unprotect source
            self.unprotect_file(source_path)
            
    def _calculate_checksum(self, file_path: Path, 
                           algorithm: str = 'md5') -> str:
        """
        Calculate file checksum
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            Hex digest of file checksum
        """
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
                
        return hash_func.hexdigest()
        
    def _create_backup(self, file_path: Path) -> Path:
        """
        Create a backup of a file
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Path to backup file
        """
        backup_dir = file_path.parent / '.photosight_backups'
        backup_dir.mkdir(exist_ok=True)
        
        # Create unique backup name
        timestamp = int(file_path.stat().st_mtime)
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        # Copy if backup doesn't exist
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
            
        return backup_path
        
    def _is_read_only(self, file_path: Path) -> bool:
        """Check if a file is read-only"""
        return not os.access(file_path, os.W_OK)
        
    def get_protection_status(self) -> Dict:
        """Get status of all protected files"""
        return {
            'protected_count': len(self._protected_files),
            'files': [str(p) for p in self._protected_files.keys()],
            'total_size': sum(
                info['size'] for info in self._protected_files.values()
            )
        }


class SafeFileOperations:
    """Safe file operations that protect source files"""
    
    def __init__(self, protector: Optional[FileProtector] = None):
        """
        Initialize safe file operations
        
        Args:
            protector: FileProtector instance (creates default if None)
        """
        self.protector = protector or FileProtector(
            verify_checksums=True,
            read_only_mode=True
        )
        
    def safe_copy(self, source: Path, destination: Path,
                  verify: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Safely copy a file with protection
        
        Args:
            source: Source file path
            destination: Destination file path
            verify: Verify copy integrity
            
        Returns:
            (success, error_message)
        """
        try:
            source = Path(source).resolve()
            destination = Path(destination).resolve()
            
            # Protect source file
            with self.protector.protect_files([source]):
                # Ensure destination directory exists
                destination.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source, destination)
                
                # Verify if requested
                if verify:
                    source_checksum = self.protector._calculate_checksum(source)
                    dest_checksum = self.protector._calculate_checksum(destination)
                    
                    if source_checksum != dest_checksum:
                        os.remove(destination)
                        return False, "Copy verification failed"
                        
            logger.info(f"Safely copied: {source} -> {destination}")
            return True, None
            
        except Exception as e:
            logger.error(f"Safe copy failed: {e}")
            return False, str(e)
            
    def safe_move(self, source: Path, destination: Path) -> Tuple[bool, Optional[str]]:
        """
        Safely move a file (copy + verify + delete)
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            (success, error_message)
        """
        try:
            # First copy with verification
            success, error = self.safe_copy(source, destination, verify=True)
            if not success:
                return False, error
                
            # Then remove source
            os.remove(source)
            logger.info(f"Safely moved: {source} -> {destination}")
            return True, None
            
        except Exception as e:
            logger.error(f"Safe move failed: {e}")
            return False, str(e)
            
    def safe_link(self, source: Path, destination: Path) -> Tuple[bool, Optional[str]]:
        """
        Create a hard link with source protection
        
        Args:
            source: Source file path
            destination: Link path
            
        Returns:
            (success, error_message)
        """
        try:
            source = Path(source).resolve()
            destination = Path(destination).resolve()
            
            # Protect source file
            with self.protector.protect_files([source]):
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, destination)
                
            logger.info(f"Created hard link: {source} -> {destination}")
            return True, None
            
        except Exception as e:
            logger.error(f"Safe link failed: {e}")
            return False, str(e)


# Convenience functions
def protect_source_files(file_paths: List[Path]) -> FileProtector:
    """
    Protect a list of source files
    
    Args:
        file_paths: List of files to protect
        
    Returns:
        FileProtector instance managing the files
    """
    protector = FileProtector(
        verify_checksums=True,
        read_only_mode=True
    )
    
    for file_path in file_paths:
        protector.protect_file(file_path)
        
    return protector


@contextmanager
def protected_analysis(file_path: Path):
    """
    Context manager for protected file analysis
    
    Args:
        file_path: File to analyze
        
    Yields:
        Path to read-only protected file
    """
    protector = FileProtector(verify_checksums=True, read_only_mode=True)
    
    try:
        protector.protect_file(file_path)
        yield file_path
    finally:
        protector.unprotect_file(file_path)