"""
File system operations for PhotoSight
Handles finding RAW files, managing output directories, and file operations
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from photosight.utils.file_protection import FileProtector, SafeFileOperations

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations for the PhotoSight pipeline"""
    
    def __init__(self, config: Dict, enable_protection: bool = True):
        """
        Initialize FileManager with configuration
        
        Args:
            config: Configuration dictionary
            enable_protection: Enable source file protection
        """
        self.config = config
        self.raw_extensions = config['processing']['raw_extensions']
        self.operation = config['output']['operation']
        self.include_sidecar = config['output']['include_sidecar_files']
        
        # Initialize file protection
        self.enable_protection = enable_protection
        if enable_protection:
            self.protector = FileProtector(
                verify_checksums=True,
                read_only_mode=True,
                create_backups=False
            )
            self.safe_ops = SafeFileOperations(self.protector)
        else:
            self.protector = None
            self.safe_ops = None
        
    def find_raw_files(self, input_path: str) -> List[Path]:
        """
        Find all RAW files in the input directory
        
        Args:
            input_path: Path to search for RAW files
            
        Returns:
            List of Path objects for found RAW files
        """
        input_path = Path(input_path)
        raw_files = []
        
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
            
        # Search for RAW files
        for ext in self.raw_extensions:
            if input_path.is_file() and input_path.suffix.lower() == ext.lower():
                raw_files.append(input_path)
            else:
                raw_files.extend(input_path.rglob(f"*{ext}"))
                
        # Remove duplicates and sort
        raw_files = sorted(list(set(raw_files)))
        logger.info(f"Found {len(raw_files)} RAW files in {input_path}")
        
        return raw_files
    
    def get_sidecar_files(self, raw_file: Path) -> List[Path]:
        """
        Find sidecar files (.xmp) associated with a RAW file
        
        Args:
            raw_file: Path to the RAW file
            
        Returns:
            List of sidecar file paths
        """
        sidecar_files = []
        base_name = raw_file.stem
        
        # Check for .xmp sidecar
        xmp_file = raw_file.with_suffix('.xmp')
        if xmp_file.exists():
            sidecar_files.append(xmp_file)
            
        # Check for .XMP (uppercase)
        xmp_file_upper = raw_file.with_suffix('.XMP')
        if xmp_file_upper.exists() and xmp_file_upper != xmp_file:
            sidecar_files.append(xmp_file_upper)
            
        return sidecar_files
    
    def setup_output_directories(self, output_base: str) -> Dict[str, Path]:
        """
        Create output directory structure
        
        Args:
            output_base: Base output directory
            
        Returns:
            Dictionary mapping folder names to Path objects
        """
        output_base = Path(output_base)
        folders = {}
        
        # Create base folders
        folders['base'] = output_base
        folders['accepted'] = output_base / self.config['output']['folders']['accepted']
        folders['rejected'] = output_base / self.config['output']['folders']['rejected']
        
        # Create rejection reason subfolders
        for reason, folder in self.config['output']['folders']['rejection_reasons'].items():
            folders[f'rejected_{reason}'] = output_base / folder
            
        # Create all directories
        for name, path in folders.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")
            
        return folders
    
    def get_output_path(self, input_file: Path, input_base: Path, 
                       output_folder: Path, preserve_structure: bool = True) -> Path:
        """
        Calculate output path for a file
        
        Args:
            input_file: Input file path
            input_base: Base input directory
            output_folder: Output folder
            preserve_structure: Whether to preserve folder structure
            
        Returns:
            Output file path
        """
        if preserve_structure and input_base in input_file.parents:
            # Calculate relative path from input base
            relative_path = input_file.relative_to(input_base)
            output_path = output_folder / relative_path
        else:
            # Just use filename
            output_path = output_folder / input_file.name
            
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return output_path
    
    def process_file(self, source: Path, destination: Path, 
                     dry_run: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Copy or move a file to destination with source protection
        
        Args:
            source: Source file path
            destination: Destination file path
            dry_run: If True, don't actually move/copy files
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            if dry_run:
                logger.info(f"[DRY RUN] Would {self.operation} {source} to {destination}")
                return True, None
                
            # Check if destination exists
            if destination.exists():
                if self.config['processing']['skip_existing']:
                    logger.debug(f"Skipping existing file: {destination}")
                    return True, "Already exists"
                else:
                    logger.warning(f"Overwriting existing file: {destination}")
                    
            # Use safe operations if protection is enabled
            if self.enable_protection and self.safe_ops:
                if self.operation == "move":
                    success, error = self.safe_ops.safe_move(source, destination)
                else:  # copy
                    success, error = self.safe_ops.safe_copy(source, destination)
                    
                if not success:
                    return False, error
            else:
                # Standard operations without protection
                if self.operation == "move":
                    shutil.move(str(source), str(destination))
                    logger.debug(f"Moved {source} to {destination}")
                else:  # copy
                    shutil.copy2(str(source), str(destination))
                    logger.debug(f"Copied {source} to {destination}")
                
            # Handle sidecar files if configured
            if self.include_sidecar:
                for sidecar in self.get_sidecar_files(source):
                    sidecar_dest = destination.parent / sidecar.name
                    if self.operation == "move":
                        shutil.move(str(sidecar), str(sidecar_dest))
                    else:
                        shutil.copy2(str(sidecar), str(sidecar_dest))
                    logger.debug(f"Processed sidecar: {sidecar}")
                    
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to {self.operation} {source}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_file_info(self, file_path: Path) -> Dict:
        """
        Get basic file information
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        stat = file_path.stat()
        return {
            'path': str(file_path),
            'name': file_path.name,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': stat.st_mtime,
            'sidecar_files': [str(f) for f in self.get_sidecar_files(file_path)]
        }