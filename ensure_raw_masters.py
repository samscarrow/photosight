#!/usr/bin/env python3
"""
Ensure RAW masters exist on pifive0 NFS storage

For each processed JPEG:
1. Check if corresponding RAW exists in master storage
2. If not, search for RAW in common locations
3. Copy RAW to master storage with GUID reference
4. Update metadata to point to master
"""

import os
import sys
import shutil
import hashlib
import json
import uuid
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Master storage configuration
MASTER_BASE = Path("/home/nfs-storage/photosight/masters")
LOCAL_CACHE = Path("~/.photosight/raw_cache").expanduser()
USE_SSH = True  # Use SSH to copy files to pifive0

# Common RAW locations to search
RAW_SEARCH_PATHS = [
    Path("/Volumes/Untitled/DCIM/100MSDCF"),  # SD card
    Path("~/Pictures/Photos Library.photoslibrary/originals").expanduser(),  # macOS Photos
    Path("~/Desktop/RAW_PHOTOS").expanduser(),  # Common local folder
    Path("/home/nfs-storage/photos/raw_archive"),  # Existing NFS archive
]


class RawMasterManager:
    """Manage RAW master files on NFS storage"""
    
    def __init__(self, master_base: Path = MASTER_BASE, dry_run: bool = False):
        self.master_base = master_base
        self.dry_run = dry_run
        self.stats = {
            'checked': 0,
            'found_existing': 0,
            'created_new': 0,
            'missing_raw': 0,
            'errors': 0
        }
        
    def find_existing_master(self, raw_name: str) -> Optional[Path]:
        """Check if RAW already exists in master storage"""
        # Search by filename in all subdirectories
        for master_path in self.master_base.rglob(raw_name):
            if master_path.is_file():
                return master_path
        return None
        
    def find_raw_source(self, jpeg_path: Path) -> Optional[Path]:
        """Find RAW file corresponding to a JPEG"""
        # Get base name without extension
        base_name = jpeg_path.stem
        
        # Common RAW extensions
        raw_extensions = ['.ARW', '.arw', '.RAF', '.raf', '.CR2', '.cr2', 
                         '.NEF', '.nef', '.DNG', '.dng']
        
        # Search in same directory as JPEG first
        for ext in raw_extensions:
            raw_path = jpeg_path.parent / (base_name + ext)
            if raw_path.exists():
                logger.info(f"Found RAW in same directory: {raw_path}")
                return raw_path
                
        # Search in common locations
        for search_dir in RAW_SEARCH_PATHS:
            if not search_dir.exists():
                continue
                
            for ext in raw_extensions:
                raw_name = base_name + ext
                
                # Direct check
                raw_path = search_dir / raw_name
                if raw_path.exists():
                    logger.info(f"Found RAW in {search_dir}: {raw_path}")
                    return raw_path
                    
                # Recursive search (slower but thorough)
                matches = list(search_dir.rglob(raw_name))
                if matches:
                    logger.info(f"Found RAW via recursive search: {matches[0]}")
                    return matches[0]
                    
        return None
        
    def calculate_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
        
    def create_master_structure(self, jpeg_path: Path, metadata: Dict) -> Path:
        """Determine where master should be stored"""
        # Extract date and photoshoot info
        photoshoot = metadata.get('photoshoot', 'uncategorized')
        
        # Try to get date from metadata or file
        if 'date' in metadata:
            date_str = metadata['date']
            year = date_str.split('-')[0]
        elif 'capture_date' in metadata:
            year = metadata['capture_date'].split('-')[0]
        else:
            # Fall back to file modification time
            mtime = datetime.fromtimestamp(jpeg_path.stat().st_mtime)
            year = str(mtime.year)
            
        # Create directory structure
        master_dir = self.master_base / year / photoshoot
        
        if not self.dry_run:
            if USE_SSH:
                # Create directory via SSH
                subprocess.run([
                    'ssh', 'pifive0', 
                    f'mkdir -p "{master_dir}"'
                ], check=True)
            else:
                master_dir.mkdir(parents=True, exist_ok=True)
            
        return master_dir
        
    def ensure_raw_master(self, jpeg_path: Path) -> Tuple[Optional[Path], Dict]:
        """Ensure RAW master exists for a JPEG"""
        self.stats['checked'] += 1
        result = {'jpeg': str(jpeg_path), 'status': 'unknown'}
        
        # Load metadata if available
        metadata = {}
        json_path = jpeg_path.with_suffix('.json')
        if json_path.exists():
            with open(json_path) as f:
                metadata = json.load(f)
                
        # Check for photo GUID
        photo_guid = metadata.get('photo_guid')
        if not photo_guid:
            photo_guid = str(uuid.uuid4())
            result['new_guid'] = photo_guid
            
        # Expected RAW name
        raw_name = jpeg_path.stem + '.ARW'  # Assuming Sony
        
        # Check if master already exists
        existing_master = self.find_existing_master(raw_name)
        if existing_master:
            self.stats['found_existing'] += 1
            result['status'] = 'exists'
            result['master_path'] = str(existing_master)
            
            # Check GUID file
            guid_path = existing_master.with_suffix('.guid')
            if guid_path.exists():
                result['existing_guid'] = guid_path.read_text().strip()
            else:
                # Create GUID file
                if not self.dry_run:
                    guid_path.write_text(photo_guid)
                result['guid_created'] = True
                
            return existing_master, result
            
        # Find RAW source
        raw_source = self.find_raw_source(jpeg_path)
        if not raw_source:
            self.stats['missing_raw'] += 1
            result['status'] = 'raw_not_found'
            logger.warning(f"Could not find RAW for {jpeg_path.name}")
            return None, result
            
        # Create master copy
        master_dir = self.create_master_structure(jpeg_path, metadata)
        master_path = master_dir / raw_name
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would copy {raw_source} to {master_path}")
            result['status'] = 'would_create'
            result['source'] = str(raw_source)
            result['destination'] = str(master_path)
        else:
            try:
                # Copy RAW to master location
                logger.info(f"Copying {raw_source} to {master_path}")
                if USE_SSH:
                    # Create directory on pifive0 first
                    subprocess.run([
                        'ssh', 'pifive0', 
                        f'mkdir -p "{master_dir}"'
                    ], check=True)
                    
                    # Copy file via SCP
                    subprocess.run([
                        'scp', str(raw_source), 
                        f'pifive0:{master_path}'
                    ], check=True)
                else:
                    shutil.copy2(raw_source, master_path)
                
                # Create GUID reference
                guid_path = master_path.with_suffix('.guid')
                if USE_SSH:
                    subprocess.run([
                        'ssh', 'pifive0',
                        f'echo "{photo_guid}" > "{guid_path}"'
                    ], check=True)
                else:
                    guid_path.write_text(photo_guid)
                
                # Calculate hash for verification
                if USE_SSH:
                    result_hash = subprocess.run([
                        'ssh', 'pifive0',
                        f'sha256sum "{master_path}" | cut -d" " -f1'
                    ], capture_output=True, text=True, check=True)
                    file_hash = result_hash.stdout.strip()
                else:
                    file_hash = self.calculate_file_hash(master_path)
                
                # Create metadata file
                meta_path = master_path.with_suffix('.meta.json')
                master_metadata = {
                    'photo_guid': photo_guid,
                    'original_source': str(raw_source),
                    'import_date': datetime.now().isoformat(),
                    'file_size': master_path.stat().st_size,
                    'file_hash': file_hash,
                    'jpeg_reference': str(jpeg_path)
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(master_metadata, f, indent=2)
                    
                self.stats['created_new'] += 1
                result['status'] = 'created'
                result['master_path'] = str(master_path)
                result['file_hash'] = file_hash
                
                # Update JPEG metadata to reference master
                if json_path.exists():
                    metadata['photo_guid'] = photo_guid
                    metadata['raw_master_path'] = str(master_path)
                    metadata['raw_master_hash'] = file_hash
                    
                    with open(json_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                        
            except Exception as e:
                logger.error(f"Failed to create master: {e}")
                self.stats['errors'] += 1
                result['status'] = 'error'
                result['error'] = str(e)
                return None, result
                
        return master_path, result
        
    def process_directory(self, directory: Path) -> Dict:
        """Process all JPEGs in a directory"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing directory: {directory}")
        logger.info(f"Master base: {self.master_base}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"{'='*60}\n")
        
        results = []
        jpeg_files = list(directory.glob("**/*.jpg")) + list(directory.glob("**/*.jpeg"))
        
        logger.info(f"Found {len(jpeg_files)} JPEG files")
        
        for jpeg_path in jpeg_files:
            logger.info(f"\nProcessing: {jpeg_path.name}")
            master_path, result = self.ensure_raw_master(jpeg_path)
            results.append(result)
            
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"Total checked: {self.stats['checked']}")
        logger.info(f"Already had masters: {self.stats['found_existing']}")
        logger.info(f"Created new masters: {self.stats['created_new']}")
        logger.info(f"Missing RAW files: {self.stats['missing_raw']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"{'='*60}\n")
        
        # Save report
        report = {
            'directory': str(directory),
            'timestamp': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'stats': self.stats,
            'results': results
        }
        
        report_path = directory / "master_creation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Report saved: {report_path}")
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensure RAW masters exist on NFS")
    parser.add_argument('directory', type=Path, 
                       help="Directory containing processed JPEGs")
    parser.add_argument('--master-base', type=Path, default=MASTER_BASE,
                       help="Master storage base path")
    parser.add_argument('--dry-run', action='store_true',
                       help="Show what would be done without copying files")
    parser.add_argument('--add-search-path', action='append', dest='search_paths',
                       help="Add additional paths to search for RAW files")
    
    args = parser.parse_args()
    
    # Add any custom search paths
    if args.search_paths:
        for path in args.search_paths:
            RAW_SEARCH_PATHS.append(Path(path))
            
    # Process directory
    manager = RawMasterManager(args.master_base, args.dry_run)
    report = manager.process_directory(args.directory)
    
    return 0 if report['stats']['errors'] == 0 else 1


if __name__ == "__main__":
    # Example usage:
    # python ensure_raw_masters.py /Users/sam/Desktop/photosight_output/enneagram_workshop/accepted --dry-run
    # python ensure_raw_masters.py /Users/sam/Desktop/photosight_output/enneagram_workshop/accepted --add-search-path /Volumes/BackupDrive/Photos
    sys.exit(main())