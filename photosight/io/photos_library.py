"""
Apple Photos Library scanner for finding RAW files
Scans the Photos Library to find RAW files matching specific criteria
"""

import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import shutil

logger = logging.getLogger(__name__)


class PhotosLibraryScanner:
    """Scans Apple Photos Library for RAW files"""
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize Photos Library scanner
        
        Args:
            library_path: Path to Photos Library, defaults to ~/Pictures/Photos Library.photoslibrary
        """
        if library_path:
            self.library_path = Path(library_path)
        else:
            self.library_path = Path.home() / "Pictures" / "Photos Library.photoslibrary"
            
        if not self.library_path.exists():
            raise ValueError(f"Photos Library not found at {self.library_path}")
            
        self.originals_path = self.library_path / "originals"
        self.database_path = self.library_path / "database" / "photos.db"
        
        # Alternative database locations for different macOS versions
        if not self.database_path.exists():
            self.database_path = self.library_path / "database" / "Photos.sqlite"
        if not self.database_path.exists():
            self.database_path = self.library_path / "database" / "Library.apdb"
            
    def find_raw_files_by_extension(self, extensions: List[str] = ['.ARW', '.arw']) -> List[Path]:
        """
        Find RAW files by extension in the originals folder
        
        Args:
            extensions: List of file extensions to search for
            
        Returns:
            List of paths to RAW files
        """
        raw_files = []
        
        if not self.originals_path.exists():
            logger.warning(f"Originals path not found: {self.originals_path}")
            return raw_files
            
        logger.info(f"Scanning {self.originals_path} for RAW files...")
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.originals_path):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in extensions):
                    file_path = Path(root) / file
                    raw_files.append(file_path)
                    
        logger.info(f"Found {len(raw_files)} RAW files")
        return raw_files
    
    def find_raw_files_by_metadata(self, 
                                  camera_model: Optional[str] = None,
                                  date_range: Optional[Tuple[datetime, datetime]] = None,
                                  min_iso: Optional[int] = None,
                                  max_iso: Optional[int] = None) -> List[Dict]:
        """
        Find RAW files using metadata criteria
        
        Args:
            camera_model: Camera model to filter by (e.g., "ILCE-7M3")
            date_range: Tuple of (start_date, end_date) to filter by
            min_iso: Minimum ISO value
            max_iso: Maximum ISO value
            
        Returns:
            List of dictionaries with file info and metadata
        """
        # First get all RAW files
        raw_files = self.find_raw_files_by_extension()
        
        if not raw_files:
            return []
            
        # Import raw processor for metadata extraction
        from photosight.io.raw_test import TestableRawProcessor as RawProcessor
        
        matching_files = []
        
        with RawProcessor() as processor:
            for i, file_path in enumerate(raw_files):
                if i % 100 == 0:
                    logger.info(f"Processing metadata: {i}/{len(raw_files)}")
                    
                try:
                    metadata = processor.extract_metadata(file_path)
                    
                    # Check camera model
                    if camera_model and metadata.get('camera_model') != camera_model:
                        continue
                        
                    # Check ISO
                    iso = metadata.get('iso')
                    if iso:
                        if min_iso and iso < min_iso:
                            continue
                        if max_iso and iso > max_iso:
                            continue
                            
                    # Check date
                    if date_range and metadata.get('date_taken'):
                        try:
                            date_taken = datetime.strptime(
                                metadata['date_taken'], 
                                "%Y:%m:%d %H:%M:%S"
                            )
                            if date_taken < date_range[0] or date_taken > date_range[1]:
                                continue
                        except:
                            pass
                            
                    # If we got here, file matches criteria
                    matching_files.append({
                        'path': file_path,
                        'metadata': metadata
                    })
                    
                except Exception as e:
                    logger.debug(f"Error processing {file_path}: {e}")
                    
        logger.info(f"Found {len(matching_files)} matching files")
        return matching_files
    
    def export_raw_files(self, 
                        files: List[Path], 
                        output_dir: Path,
                        preserve_structure: bool = True,
                        copy: bool = True) -> List[Tuple[Path, Path]]:
        """
        Export RAW files from Photos Library to a working directory
        
        Args:
            files: List of file paths to export
            output_dir: Directory to export to
            preserve_structure: Whether to preserve folder structure
            copy: If True, copy files. If False, create hard links
            
        Returns:
            List of (source, destination) tuples
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = []
        
        for file_path in files:
            try:
                if preserve_structure:
                    # Calculate relative path from originals
                    rel_path = file_path.relative_to(self.originals_path)
                    dest_path = output_dir / rel_path
                else:
                    # Just use filename
                    dest_path = output_dir / file_path.name
                    
                # Ensure destination directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Skip if already exists
                if dest_path.exists():
                    logger.debug(f"Skipping existing file: {dest_path}")
                    exported.append((file_path, dest_path))
                    continue
                    
                # Export file
                if copy:
                    shutil.copy2(file_path, dest_path)
                    logger.debug(f"Copied {file_path} to {dest_path}")
                else:
                    # Create hard link (saves space)
                    os.link(file_path, dest_path)
                    logger.debug(f"Linked {file_path} to {dest_path}")
                    
                exported.append((file_path, dest_path))
                
            except Exception as e:
                logger.error(f"Failed to export {file_path}: {e}")
                
        logger.info(f"Exported {len(exported)} files to {output_dir}")
        return exported
    
    def create_smart_export(self,
                           output_dir: Path,
                           camera_model: str = "ILCE-7M3",
                           extensions: List[str] = ['.ARW'],
                           last_n_days: Optional[int] = None) -> List[Path]:
        """
        Create a smart export of RAW files based on common criteria
        
        Args:
            output_dir: Directory to export to
            camera_model: Camera model to filter by
            extensions: File extensions to include
            last_n_days: Only include photos from last N days
            
        Returns:
            List of exported file paths
        """
        # Set date range if specified
        date_range = None
        if last_n_days:
            end_date = datetime.now()
            start_date = datetime.now() - timedelta(days=last_n_days)
            date_range = (start_date, end_date)
            
        # Find matching files
        logger.info(f"Searching for {camera_model} RAW files...")
        matching = self.find_raw_files_by_metadata(
            camera_model=camera_model,
            date_range=date_range
        )
        
        # Filter by extension
        filtered = []
        for item in matching:
            if any(str(item['path']).lower().endswith(ext.lower()) for ext in extensions):
                filtered.append(item['path'])
                
        logger.info(f"Found {len(filtered)} files to export")
        
        # Export files
        if filtered:
            exported = self.export_raw_files(filtered, output_dir, preserve_structure=False)
            return [dest for _, dest in exported]
        else:
            return []


def scan_and_tag_for_processing(
    camera_model: str = "ILCE-7M3",
    output_file: str = "raw_files_to_process.txt",
    extensions: List[str] = ['.ARW', '.arw']
) -> int:
    """
    Scan Photos Library and create a list of RAW files to process
    
    Args:
        camera_model: Camera model to search for
        output_file: File to write the list of RAW files to
        extensions: RAW file extensions to search for
        
    Returns:
        Number of files found
    """
    try:
        scanner = PhotosLibraryScanner()
        
        # Find all RAW files from the specified camera
        logger.info(f"Scanning for {camera_model} RAW files...")
        matching_files = scanner.find_raw_files_by_metadata(camera_model=camera_model)
        
        # Filter by extension and write to file
        with open(output_file, 'w') as f:
            count = 0
            for item in matching_files:
                file_path = item['path']
                if any(str(file_path).lower().endswith(ext.lower()) for ext in extensions):
                    f.write(f"{file_path}\n")
                    count += 1
                    
        logger.info(f"Found {count} RAW files, saved list to {output_file}")
        return count
        
    except Exception as e:
        logger.error(f"Error scanning Photos Library: {e}")
        return 0