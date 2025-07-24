#!/usr/bin/env python3
"""
Migrate locally processed photos to unified PhotoSight system

This script:
1. Finds photos processed with old local approach (no GUIDs)
2. Assigns GUIDs to existing photos
3. Extracts metadata from JSON files
4. Optionally syncs to Oracle database
"""

import sys
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add PhotoSight to path
sys.path.insert(0, '/Users/sam/dev/photosight')

from photosight.core.unified_processor import UnifiedPhotoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhotoMigrator:
    """Migrate local photos to unified system"""
    
    def __init__(self, unified_processor: Optional[UnifiedPhotoProcessor] = None):
        self.processor = unified_processor or UnifiedPhotoProcessor(mode="hybrid")
        self.migration_log = []
        
    def find_local_photos(self, directory: Path) -> List[Dict]:
        """Find photos processed with old local approach"""
        local_photos = []
        
        # Look for JPEG + JSON pairs
        for jpg_path in directory.glob("**/*.jpg"):
            json_path = jpg_path.with_suffix('.json')
            
            if json_path.exists():
                with open(json_path) as f:
                    metadata = json.load(f)
                    
                # Check if it has a GUID (new system) or not (old system)
                if 'photo_guid' not in metadata:
                    local_photos.append({
                        'jpg_path': jpg_path,
                        'json_path': json_path,
                        'metadata': metadata,
                        'needs_migration': True
                    })
                    logger.info(f"Found legacy photo: {jpg_path.name}")
                else:
                    logger.debug(f"Photo already has GUID: {jpg_path.name}")
                    
        logger.info(f"Found {len(local_photos)} photos needing migration")
        return local_photos
        
    def migrate_photo(self, photo_info: Dict) -> Dict:
        """Migrate a single photo to unified system"""
        jpg_path = photo_info['jpg_path']
        metadata = photo_info['metadata']
        
        # Generate GUID for this photo
        photo_guid = str(uuid.uuid4())
        logger.info(f"Assigning GUID {photo_guid} to {jpg_path.name}")
        
        # Build unified metadata
        unified_metadata = {
            'photo_guid': photo_guid,
            'file_name': jpg_path.stem + '.ARW',  # Assume original was ARW
            'migrated_from': str(jpg_path),
            'migration_date': datetime.now().isoformat(),
            'original_metadata': metadata
        }
        
        # Extract known fields from old format
        if 'photoshoot' in metadata:
            unified_metadata['photoshoot_tag'] = metadata['photoshoot']
        if 'date' in metadata:
            unified_metadata['capture_date'] = metadata['date']
        if 'people_count' in metadata:
            unified_metadata['people_count'] = metadata['people_count']
        if 'blur_class' in metadata:
            unified_metadata['blur_classification'] = metadata['blur_class']
        if 'objects' in metadata:
            unified_metadata['objects_detected'] = metadata['objects']
            
        # Update JSON file with GUID
        updated_json_path = photo_info['json_path']
        with open(updated_json_path, 'w') as f:
            json.dump(unified_metadata, f, indent=2)
            
        # Optionally sync to Oracle
        if self.processor.oracle_conn:
            try:
                self._sync_to_oracle(unified_metadata)
                unified_metadata['oracle_sync'] = 'success'
            except Exception as e:
                logger.error(f"Oracle sync failed for {jpg_path.name}: {e}")
                unified_metadata['oracle_sync'] = 'failed'
                unified_metadata['oracle_error'] = str(e)
        else:
            unified_metadata['oracle_sync'] = 'skipped'
            
        return unified_metadata
        
    def _sync_to_oracle(self, metadata: Dict) -> None:
        """Sync migrated photo to Oracle database"""
        if not self.processor.oracle_conn:
            return
            
        cursor = self.processor.oracle_conn.cursor()
        
        # Insert photo record
        cursor.execute("""
            INSERT INTO photos (
                photo_guid, file_name, capture_date,
                photoshoot_tag, migration_source
            ) VALUES (
                HEXTORAW(:guid), :file_name, TO_TIMESTAMP(:capture_date, 'YYYY-MM-DD'),
                :photoshoot_tag, :migration_source
            )
        """, {
            'guid': metadata['photo_guid'].replace('-', ''),
            'file_name': metadata['file_name'],
            'capture_date': metadata.get('capture_date', '2025-01-01'),
            'photoshoot_tag': metadata.get('photoshoot_tag'),
            'migration_source': 'local_migration'
        })
        
        # Insert YOLO detections if available
        if metadata.get('objects_detected'):
            for obj in metadata['objects_detected']:
                if obj == 'person':
                    # Create detection records for people
                    for i in range(metadata.get('people_count', 1)):
                        detection_id = str(uuid.uuid4()).replace('-', '')
                        cursor.execute("""
                            INSERT INTO yolo_detections (
                                detection_id, photo_guid, class_name,
                                confidence, migration_flag
                            ) VALUES (
                                HEXTORAW(:detection_id), HEXTORAW(:photo_guid),
                                :class_name, :confidence, 'Y'
                            )
                        """, {
                            'detection_id': detection_id,
                            'photo_guid': metadata['photo_guid'].replace('-', ''),
                            'class_name': 'person',
                            'confidence': 0.95  # Assume high confidence for migrated
                        })
                        
        self.processor.oracle_conn.commit()
        logger.info(f"Synced {metadata['file_name']} to Oracle")
        
    def migrate_directory(self, directory: Path, dry_run: bool = False) -> Dict:
        """Migrate all photos in a directory"""
        logger.info(f"\n{'='*60}")
        logger.info(f"MIGRATION STARTING: {directory}")
        logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        logger.info(f"{'='*60}\n")
        
        # Find photos needing migration
        local_photos = self.find_local_photos(directory)
        
        if not local_photos:
            logger.info("No photos need migration")
            return {'migrated': 0, 'errors': 0}
            
        # Migrate each photo
        results = {
            'migrated': 0,
            'errors': 0,
            'oracle_synced': 0,
            'photos': []
        }
        
        for photo_info in local_photos:
            try:
                if dry_run:
                    logger.info(f"[DRY RUN] Would migrate: {photo_info['jpg_path'].name}")
                    results['migrated'] += 1
                else:
                    migrated = self.migrate_photo(photo_info)
                    results['photos'].append(migrated)
                    results['migrated'] += 1
                    
                    if migrated.get('oracle_sync') == 'success':
                        results['oracle_synced'] += 1
                        
            except Exception as e:
                logger.error(f"Migration failed for {photo_info['jpg_path']}: {e}")
                results['errors'] += 1
                
        # Save migration report
        if not dry_run:
            report_path = directory / "migration_report.json"
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nMigration report saved: {report_path}")
            
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("MIGRATION COMPLETE")
        logger.info(f"Migrated: {results['migrated']} photos")
        logger.info(f"Oracle synced: {results['oracle_synced']} photos")
        logger.info(f"Errors: {results['errors']}")
        logger.info(f"{'='*60}\n")
        
        return results


def main():
    """Run migration for workshop photos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate local photos to unified system")
    parser.add_argument('directory', type=Path, help="Directory containing photos to migrate")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be migrated")
    parser.add_argument('--oracle', action='store_true', help="Sync to Oracle database")
    
    args = parser.parse_args()
    
    # Initialize processor
    mode = 'hybrid' if args.oracle else 'local'
    processor = UnifiedPhotoProcessor(mode=mode)
    
    # Run migration
    migrator = PhotoMigrator(processor)
    results = migrator.migrate_directory(args.directory, dry_run=args.dry_run)
    
    return 0 if results['errors'] == 0 else 1


if __name__ == "__main__":
    # Example: Migrate Enneagram Workshop photos
    # python migrate_local_photos_to_unified.py /Users/sam/Desktop/photosight_output/enneagram_workshop/accepted --dry-run
    sys.exit(main())