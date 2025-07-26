"""
Photo file organization utilities.

Provides functionality for organizing photos into structured directories
based on various criteria like date, camera, project, or rating.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from datetime import datetime
import json
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


class PhotoOrganizer:
    """
    Organizes photos into structured directories based on various criteria.
    
    Supports organization by:
    - Date (year/month or year/month/day)
    - Camera make/model
    - Project name
    - Rating/quality score
    - Custom structure
    """
    
    def __init__(self, config: Dict):
        """Initialize the photo organizer."""
        self.config = config
        self.organization_config = config.get('organization', {})
        
    def organize_photos(self, photo_files: List[Path], dest_dir: Path,
                       structure: str = 'date', copy: bool = True,
                       dry_run: bool = False, create_selects: bool = True,
                       selects_threshold: float = 0.8,
                       progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Organize photos into structured directories.
        
        Args:
            photo_files: List of photo file paths
            dest_dir: Destination directory
            structure: Organization structure ('date', 'camera', 'project', 'rating')
            copy: True to copy files, False to move them
            dry_run: If True, don't actually move/copy files
            create_selects: Whether to create a separate selects folder
            selects_threshold: Quality threshold for selects folder
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary with organization results and statistics
        """
        
        results = {
            'processed': 0,
            'organized': 0,
            'errors': 0,
            'selects': 0,
            'error_details': [],
            'organization_map': {},
            'selects_list': []
        }
        
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            if create_selects:
                selects_dir = dest_dir / "selects"
                selects_dir.mkdir(exist_ok=True)
        
        # Initialize quality ranker for selects
        quality_ranker = None
        if create_selects:
            try:
                from ..ranking.quality_ranker import QualityRanker
                quality_ranker = QualityRanker(self.config)
            except Exception as e:
                logger.warning(f"Could not initialize quality ranker: {e}")
        
        for i, photo_file in enumerate(photo_files):
            try:
                results['processed'] += 1
                
                # Determine destination path based on structure
                dest_path = self._get_destination_path(photo_file, dest_dir, structure)
                
                if dest_path:
                    # Create directory structure
                    if not dry_run:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy or move file
                        if copy:
                            shutil.copy2(photo_file, dest_path)
                        else:
                            shutil.move(str(photo_file), str(dest_path))
                    
                    results['organized'] += 1
                    results['organization_map'][str(photo_file)] = str(dest_path)
                    
                    # Check if photo should go in selects
                    if create_selects and quality_ranker:
                        try:
                            quality_score = quality_ranker.rank_photo(photo_file)
                            
                            if quality_score >= selects_threshold:
                                selects_dest = dest_dir / "selects" / photo_file.name
                                
                                if not dry_run:
                                    if copy:
                                        shutil.copy2(photo_file, selects_dest)
                                    else:
                                        # If we moved the original, copy from the new location
                                        shutil.copy2(dest_path, selects_dest)
                                
                                results['selects'] += 1
                                results['selects_list'].append({
                                    'file': str(photo_file),
                                    'score': quality_score,
                                    'selects_path': str(selects_dest)
                                })
                                
                        except Exception as e:
                            logger.warning(f"Could not evaluate {photo_file} for selects: {e}")
                
                if progress_callback:
                    progress_callback(i + 1, len(photo_files))
                    
            except Exception as e:
                logger.error(f"Error organizing {photo_file}: {e}")
                results['errors'] += 1
                results['error_details'].append({
                    'file': str(photo_file),
                    'error': str(e)
                })
        
        # Save organization manifest
        if not dry_run and results['organized'] > 0:
            self._save_organization_manifest(dest_dir, results, structure)
        
        return results
    
    def _get_destination_path(self, photo_file: Path, dest_dir: Path, structure: str) -> Optional[Path]:
        """
        Determine the destination path for a photo based on organization structure.
        
        Args:
            photo_file: Source photo file path
            dest_dir: Destination base directory
            structure: Organization structure type
            
        Returns:
            Destination file path or None if cannot be determined
        """
        try:
            if structure == 'date':
                return self._get_date_based_path(photo_file, dest_dir)
            elif structure == 'camera':
                return self._get_camera_based_path(photo_file, dest_dir)
            elif structure == 'project':
                return self._get_project_based_path(photo_file, dest_dir)
            elif structure == 'rating':
                return self._get_rating_based_path(photo_file, dest_dir)
            else:
                logger.warning(f"Unknown organization structure: {structure}")
                return dest_dir / photo_file.name
                
        except Exception as e:
            logger.error(f"Error determining destination path for {photo_file}: {e}")
            return None
    
    def _get_date_based_path(self, photo_file: Path, dest_dir: Path) -> Path:
        """Get destination path based on photo date."""
        try:
            # Try to get date from EXIF
            date_taken = self._extract_date_from_exif(photo_file)
            
            if not date_taken:
                # Fall back to file modification date
                date_taken = datetime.fromtimestamp(photo_file.stat().st_mtime)
            
            # Create date-based directory structure
            year = date_taken.strftime('%Y')
            month = date_taken.strftime('%m-%B')  # e.g., "03-March"
            
            date_dir = dest_dir / year / month
            return date_dir / photo_file.name
            
        except Exception as e:
            logger.warning(f"Could not extract date from {photo_file}: {e}")
            # Fall back to "Unknown" folder
            return dest_dir / "Unknown_Date" / photo_file.name
    
    def _get_camera_based_path(self, photo_file: Path, dest_dir: Path) -> Path:
        """Get destination path based on camera make/model."""
        try:
            camera_info = self._extract_camera_info(photo_file)
            
            if camera_info['make'] and camera_info['model']:
                camera_dir = f"{camera_info['make']}_{camera_info['model']}"
                camera_dir = self._sanitize_directory_name(camera_dir)
            elif camera_info['make']:
                camera_dir = self._sanitize_directory_name(camera_info['make'])
            else:
                camera_dir = "Unknown_Camera"
            
            return dest_dir / camera_dir / photo_file.name
            
        except Exception as e:
            logger.warning(f"Could not extract camera info from {photo_file}: {e}")
            return dest_dir / "Unknown_Camera" / photo_file.name
    
    def _get_project_based_path(self, photo_file: Path, dest_dir: Path) -> Path:
        """Get destination path based on project."""
        try:
            # Try to determine project from file path or metadata
            project_name = self._determine_project_name(photo_file)
            project_dir = self._sanitize_directory_name(project_name)
            
            return dest_dir / project_dir / photo_file.name
            
        except Exception as e:
            logger.warning(f"Could not determine project for {photo_file}: {e}")
            return dest_dir / "Unknown_Project" / photo_file.name
    
    def _get_rating_based_path(self, photo_file: Path, dest_dir: Path) -> Path:
        """Get destination path based on quality rating."""
        try:
            # Use quality ranker to determine rating
            from ..ranking.quality_ranker import QualityRanker
            ranker = QualityRanker(self.config)
            
            quality_score = ranker.rank_photo(photo_file)
            
            # Determine rating category
            if quality_score >= 0.9:
                rating_dir = "Excellent_(90-100)"
            elif quality_score >= 0.8:
                rating_dir = "Very_Good_(80-89)"
            elif quality_score >= 0.7:
                rating_dir = "Good_(70-79)"
            elif quality_score >= 0.6:
                rating_dir = "Fair_(60-69)"
            elif quality_score >= 0.5:
                rating_dir = "Poor_(50-59)"
            else:
                rating_dir = "Very_Poor_(0-49)"
            
            return dest_dir / rating_dir / photo_file.name
            
        except Exception as e:
            logger.warning(f"Could not rate {photo_file}: {e}")
            return dest_dir / "Unrated" / photo_file.name
    
    def _extract_date_from_exif(self, photo_file: Path) -> Optional[datetime]:
        """Extract date taken from EXIF data."""
        try:
            image = Image.open(photo_file)
            exif = image.getexif()
            
            if exif:
                # Try different date fields
                date_fields = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
                
                for field in date_fields:
                    if field in exif:
                        date_str = str(exif[field])
                        try:
                            return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                        except ValueError:
                            continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting EXIF date from {photo_file}: {e}")
            return None
    
    def _extract_camera_info(self, photo_file: Path) -> Dict[str, Optional[str]]:
        """Extract camera make and model from EXIF data."""
        try:
            image = Image.open(photo_file)
            exif = image.getexif()
            
            camera_info = {'make': None, 'model': None}
            
            if exif:
                # Get camera make
                if 'Make' in exif:
                    camera_info['make'] = str(exif['Make']).strip()
                
                # Get camera model
                if 'Model' in exif:
                    camera_info['model'] = str(exif['Model']).strip()
            
            return camera_info
            
        except Exception as e:
            logger.warning(f"Error extracting camera info from {photo_file}: {e}")
            return {'make': None, 'model': None}
    
    def _determine_project_name(self, photo_file: Path) -> str:
        """Determine project name from file path or other sources."""
        try:
            # Try to extract project name from directory structure
            # Look for common project folder patterns
            path_parts = photo_file.parts
            
            # Look for date-based project folders (e.g., "2024-01-15_Wedding_Smith")
            for part in reversed(path_parts[:-1]):  # Exclude filename
                if '_' in part:
                    # Might be a project folder
                    return part
            
            # Fall back to parent directory name
            if len(path_parts) > 1:
                return path_parts[-2]
            
            return "General"
            
        except Exception as e:
            logger.warning(f"Error determining project for {photo_file}: {e}")
            return "Unknown_Project"
    
    def _sanitize_directory_name(self, name: str) -> str:
        """Sanitize directory name for filesystem compatibility."""
        # Replace problematic characters
        sanitized = name.replace('/', '_').replace('\\', '_').replace(':', '_')
        sanitized = sanitized.replace('*', '_').replace('?', '_').replace('"', '_')
        sanitized = sanitized.replace('<', '_').replace('>', '_').replace('|', '_')
        
        # Remove extra spaces and trim
        sanitized = ' '.join(sanitized.split())
        
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        return sanitized or "Unknown"
    
    def _save_organization_manifest(self, dest_dir: Path, results: Dict, structure: str):
        """Save organization manifest file."""
        try:
            manifest = {
                'organization_date': datetime.now().isoformat(),
                'structure': structure,
                'statistics': {
                    'processed': results['processed'],
                    'organized': results['organized'],
                    'errors': results['errors'],
                    'selects': results['selects']
                },
                'organization_map': results['organization_map'],
                'selects_list': results['selects_list'],
                'error_details': results['error_details']
            }
            
            manifest_file = dest_dir / 'photosight_organization_manifest.json'
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Organization manifest saved to: {manifest_file}")
            
        except Exception as e:
            logger.error(f"Error saving organization manifest: {e}")
    
    def create_directory_structure_preview(self, photo_files: List[Path], 
                                         structure: str) -> Dict[str, List[str]]:
        """
        Create a preview of the directory structure that would be created.
        
        Args:
            photo_files: List of photo file paths
            structure: Organization structure type
            
        Returns:
            Dictionary mapping directory paths to lists of files
        """
        preview = {}
        
        for photo_file in photo_files:
            try:
                # Get destination path (without base directory)
                if structure == 'date':
                    date_taken = self._extract_date_from_exif(photo_file)
                    if not date_taken:
                        date_taken = datetime.fromtimestamp(photo_file.stat().st_mtime)
                    
                    year = date_taken.strftime('%Y')
                    month = date_taken.strftime('%m-%B')
                    dir_path = f"{year}/{month}"
                    
                elif structure == 'camera':
                    camera_info = self._extract_camera_info(photo_file)
                    if camera_info['make'] and camera_info['model']:
                        dir_path = f"{camera_info['make']}_{camera_info['model']}"
                    elif camera_info['make']:
                        dir_path = camera_info['make']
                    else:
                        dir_path = "Unknown_Camera"
                    dir_path = self._sanitize_directory_name(dir_path)
                    
                elif structure == 'project':
                    project_name = self._determine_project_name(photo_file)
                    dir_path = self._sanitize_directory_name(project_name)
                    
                elif structure == 'rating':
                    dir_path = "Rating_Based_Folders"  # Placeholder for preview
                    
                else:
                    dir_path = "Unknown_Structure"
                
                if dir_path not in preview:
                    preview[dir_path] = []
                
                preview[dir_path].append(photo_file.name)
                
            except Exception as e:
                logger.warning(f"Error previewing organization for {photo_file}: {e}")
                if "Errors" not in preview:
                    preview["Errors"] = []
                preview["Errors"].append(f"{photo_file.name}: {e}")
        
        return preview