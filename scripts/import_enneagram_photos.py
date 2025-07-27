#!/usr/bin/env python3
"""
Import enneagram photos into PhotoSight database.

This script imports the analyzed enneagram photos into the database
so they can be tagged with albums.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from photosight.db.models import Base, Photo, ProcessingStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnneagramPhotoImporter:
    """Import enneagram photos with their analysis results."""
    
    def __init__(self, db_url: str = None):
        """Initialize with database connection."""
        if not db_url:
            # Use environment variable or default
            import os
            db_url = os.getenv('PHOTOSIGHT_DB_URL', 'postgresql://photosight:photosight@localhost:5432/photosight')
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Load analysis results
        self.analysis_data = self._load_analysis_results()
    
    def _load_analysis_results(self) -> dict:
        """Load the enneagram analysis results."""
        results_file = Path("enneagram_analysis_output/enneagram_analysis_results.json")
        if not results_file.exists():
            raise FileNotFoundError(f"Analysis results not found: {results_file}")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def import_photos(self, photo_dir: Path):
        """Import photos from directory."""
        photo_dir = Path(photo_dir)
        if not photo_dir.exists():
            raise ValueError(f"Photo directory not found: {photo_dir}")
        
        imported_count = 0
        existing_count = 0
        
        # Process each analyzed photo
        for filename, result in self.analysis_data['results'].items():
            photo_path = photo_dir / filename
            
            # Check if photo already exists
            existing = self.session.query(Photo).filter_by(filename=filename).first()
            
            if existing:
                existing_count += 1
                logger.debug(f"Photo already exists: {filename}")
                continue
            
            # Create new photo record
            photo = Photo(
                filename=filename,
                file_path=str(photo_path),
                file_size=photo_path.stat().st_size if photo_path.exists() else 0,
                width=4000,  # Default A7R V resolution
                height=2664,
                date_taken=datetime.utcnow(),  # Would extract from EXIF in production
                camera_make='Sony',
                camera_model='A7R V',
                lens_model='Sony FE 24-70mm F2.8 GM',
                focal_length=50,
                aperture=2.8,
                shutter_speed='1/125',
                iso=400,
                processing_status=ProcessingStatus.COMPLETED,
                photo_metadata={
                    'source': 'enneagram_workshop',
                    'import_date': datetime.utcnow().isoformat(),
                    'analysis_scores': result['scores'],
                    'scene_type': result['scene']['type'],
                    'mood': result['aesthetic']['mood']
                }
            )
            
            self.session.add(photo)
            imported_count += 1
            
            if imported_count % 10 == 0:
                self.session.commit()
                logger.info(f"Imported {imported_count} photos...")
        
        # Final commit
        self.session.commit()
        
        logger.info(f"\nImport Summary:")
        logger.info(f"  - New photos imported: {imported_count}")
        logger.info(f"  - Already existing: {existing_count}")
        logger.info(f"  - Total in analysis: {len(self.analysis_data['results'])}")
        
        return imported_count, existing_count
    
    def verify_import(self):
        """Verify all analyzed photos are in database."""
        missing = []
        
        for filename in self.analysis_data['results'].keys():
            photo = self.session.query(Photo).filter_by(filename=filename).first()
            if not photo:
                missing.append(filename)
        
        if missing:
            logger.warning(f"Missing photos in database: {len(missing)}")
            for filename in missing[:5]:
                logger.warning(f"  - {filename}")
            if len(missing) > 5:
                logger.warning(f"  ... and {len(missing) - 5} more")
        else:
            logger.info("✅ All analyzed photos found in database")
        
        return len(missing) == 0
    
    def close(self):
        """Close database connection."""
        self.session.close()
        self.engine.dispose()


def main():
    """Import enneagram photos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import enneagram photos to PhotoSight database')
    parser.add_argument('photo_dir', nargs='?', default='/Volumes/SamsungT7/enneagram',
                       help='Directory containing enneagram photos')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing photos, don\'t import')
    
    args = parser.parse_args()
    
    print("\n📸 PhotoSight Enneagram Photo Import\n")
    
    importer = EnneagramPhotoImporter()
    
    try:
        if args.verify_only:
            print("Verifying photos in database...")
            if importer.verify_import():
                print("\n✅ All photos are in the database")
            else:
                print("\n❌ Some photos are missing from the database")
        else:
            photo_dir = Path(args.photo_dir)
            print(f"Importing photos from: {photo_dir}")
            
            imported, existing = importer.import_photos(photo_dir)
            
            print(f"\n✅ Import complete!")
            print(f"   - Imported: {imported} new photos")
            print(f"   - Existing: {existing} photos")
            
            # Verify import
            importer.verify_import()
    
    finally:
        importer.close()


if __name__ == "__main__":
    main()