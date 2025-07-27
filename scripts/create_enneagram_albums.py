#!/usr/bin/env python3
"""
Create album tags in PhotoSight database for enneagram workshop photos.

Tags the top 30 photos and decisive moments based on the analysis results.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from photosight.db.models import Base, Photo, AlbumTag, PhotoAlbumTag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnneagramAlbumCreator:
    """Creates and populates album tags for enneagram photos."""
    
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
    
    def create_album_tag(self, tag_name: str, display_name: str, 
                        description: str, metadata: dict = None) -> AlbumTag:
        """Create or get an album tag."""
        # Check if tag already exists
        album_tag = self.session.query(AlbumTag).filter_by(tag_name=tag_name).first()
        
        if not album_tag:
            album_tag = AlbumTag(
                tag_name=tag_name,
                display_name=display_name,
                description=description,
                created_by='enneagram_analysis',
                album_metadata=metadata or {}
            )
            self.session.add(album_tag)
            self.session.commit()
            logger.info(f"Created album tag: {tag_name}")
        else:
            # Update metadata if needed
            album_tag.updated_at = datetime.utcnow()
            if metadata:
                album_tag.album_metadata.update(metadata)
            self.session.commit()
            logger.info(f"Updated existing album tag: {tag_name}")
        
        return album_tag
    
    def add_photo_to_album(self, photo: Photo, album_tag: AlbumTag, 
                          sort_order: int = 0, metadata: dict = None):
        """Add a photo to an album."""
        # Check if association already exists
        existing = self.session.query(PhotoAlbumTag).filter_by(
            photo_id=photo.id,
            album_tag_id=album_tag.id
        ).first()
        
        if not existing:
            association = PhotoAlbumTag(
                photo_id=photo.id,
                album_tag_id=album_tag.id,
                added_by='enneagram_analysis',
                sort_order=sort_order,
                association_metadata=metadata or {}
            )
            self.session.add(association)
            
            # Update album statistics
            album_tag.photo_count = (album_tag.photo_count or 0) + 1
            album_tag.last_photo_added = datetime.utcnow()
            
            logger.debug(f"Added {photo.filename} to album {album_tag.tag_name}")
    
    def create_top_30_album(self):
        """Create album for top 30 enneagram photos."""
        logger.info("Creating Top 30 album...")
        
        # Create album tag
        album_metadata = {
            'analysis_date': self.analysis_data['metadata']['timestamp'],
            'selection_criteria': 'overall_score',
            'score_weights': {
                'technical': 0.3,
                'artistic': 0.4,
                'emotional': 0.3
            }
        }
        
        album_tag = self.create_album_tag(
            tag_name='enneagram-top-30',
            display_name='Enneagram Workshop - Top 30',
            description='Top 30 photos from the enneagram workshop based on comprehensive quality analysis',
            metadata=album_metadata
        )
        
        # Get top 30 photos from rankings
        top_30 = self.analysis_data['rankings']['overall'][:30]
        
        added_count = 0
        for rank, entry in enumerate(top_30, 1):
            filename = entry['filename']
            scores = entry['scores']
            
            # Find photo in database
            photo = self.session.query(Photo).filter_by(filename=filename).first()
            
            if photo:
                # Add to album with rank as sort order
                photo_metadata = {
                    'rank': rank,
                    'scores': scores,
                    'scene': entry.get('scene', 'unknown'),
                    'mood': entry.get('mood', 'unknown')
                }
                
                self.add_photo_to_album(
                    photo=photo,
                    album_tag=album_tag,
                    sort_order=rank,
                    metadata=photo_metadata
                )
                added_count += 1
            else:
                logger.warning(f"Photo not found in database: {filename}")
        
        self.session.commit()
        logger.info(f"Added {added_count} photos to Top 30 album")
        
        return album_tag
    
    def create_decisive_moments_album(self):
        """Create album for decisive moment photos."""
        logger.info("Creating Decisive Moments album...")
        
        # Create album tag
        album_metadata = {
            'analysis_date': self.analysis_data['metadata']['timestamp'],
            'selection_criteria': 'decisive_moment_detection',
            'total_moments': len(self.analysis_data['rankings']['decisive_moments'])
        }
        
        album_tag = self.create_album_tag(
            tag_name='enneagram-decisive-moments',
            display_name='Enneagram Workshop - Decisive Moments',
            description='Photos capturing decisive moments with genuine interactions and peak emotions',
            metadata=album_metadata
        )
        
        # Get decisive moments
        decisive_moments = self.analysis_data['rankings']['decisive_moments']
        
        added_count = 0
        for rank, moment in enumerate(decisive_moments, 1):
            filename = moment['filename']
            
            # Find photo in database
            photo = self.session.query(Photo).filter_by(filename=filename).first()
            
            if photo:
                # Add to album with confidence-based sort order
                photo_metadata = {
                    'rank': rank,
                    'confidence': moment['confidence'],
                    'reason': moment.get('reason', ''),
                    'capture_type': 'decisive_moment'
                }
                
                self.add_photo_to_album(
                    photo=photo,
                    album_tag=album_tag,
                    sort_order=rank,
                    metadata=photo_metadata
                )
                added_count += 1
            else:
                logger.warning(f"Photo not found in database: {filename}")
        
        self.session.commit()
        logger.info(f"Added {added_count} photos to Decisive Moments album")
        
        return album_tag
    
    def create_category_albums(self):
        """Create albums for specific categories like scenes and moods."""
        logger.info("Creating category albums...")
        
        albums_created = []
        
        # Create albums for top scene types
        scene_albums = {
            'workshop/discussion': ('enneagram-discussions', 'Enneagram - Discussion Sessions'),
            'workshop/group_activity': ('enneagram-group-activities', 'Enneagram - Group Activities'),
            'portrait/candid': ('enneagram-candid-portraits', 'Enneagram - Candid Portraits')
        }
        
        for scene_type, (tag_name, display_name) in scene_albums.items():
            # Find photos with this scene type
            photos_with_scene = []
            for filename, result in self.analysis_data['results'].items():
                if result.get('scene', {}).get('type') == scene_type:
                    photos_with_scene.append((filename, result['scores']['overall']))
            
            if len(photos_with_scene) >= 5:  # Only create album if we have at least 5 photos
                # Sort by score and take top photos
                photos_with_scene.sort(key=lambda x: x[1], reverse=True)
                
                album_tag = self.create_album_tag(
                    tag_name=tag_name,
                    display_name=display_name,
                    description=f'Best {scene_type.replace("_", " ").title()} photos from the enneagram workshop',
                    metadata={'scene_type': scene_type}
                )
                
                # Add photos to album
                added_count = 0
                for rank, (filename, score) in enumerate(photos_with_scene[:20], 1):
                    photo = self.session.query(Photo).filter_by(filename=filename).first()
                    if photo:
                        self.add_photo_to_album(
                            photo=photo,
                            album_tag=album_tag,
                            sort_order=rank,
                            metadata={'score': score, 'scene': scene_type}
                        )
                        added_count += 1
                
                self.session.commit()
                logger.info(f"Created {display_name} album with {added_count} photos")
                albums_created.append(album_tag)
        
        return albums_created
    
    def list_albums(self):
        """List all created albums with statistics."""
        albums = self.session.query(AlbumTag).filter(
            AlbumTag.tag_name.like('enneagram-%')
        ).order_by(AlbumTag.photo_count.desc()).all()
        
        print("\n" + "="*60)
        print("ENNEAGRAM ALBUMS IN DATABASE")
        print("="*60)
        
        for album in albums:
            print(f"\n📁 {album.display_name}")
            print(f"   Tag: {album.tag_name}")
            print(f"   Photos: {album.photo_count or 0}")
            print(f"   Created: {album.created_at.strftime('%Y-%m-%d %H:%M')}")
            if album.description:
                print(f"   Description: {album.description[:80]}...")
            
            # Show top 5 photos
            associations = self.session.query(PhotoAlbumTag).filter_by(
                album_tag_id=album.id
            ).order_by(PhotoAlbumTag.sort_order).limit(5).all()
            
            if associations:
                print("   Top photos:")
                for assoc in associations:
                    photo = assoc.photo
                    rank = assoc.association_metadata.get('rank', '?')
                    score = assoc.association_metadata.get('scores', {}).get('overall', 0)
                    print(f"     #{rank} {photo.filename} (score: {score:.3f})")
    
    def close(self):
        """Close database connection."""
        self.session.close()
        self.engine.dispose()


def main():
    """Create albums for enneagram photos."""
    print("\n🎯 Creating PhotoSight Albums for Enneagram Photos\n")
    
    creator = EnneagramAlbumCreator()
    
    try:
        # Check if we have photos in the database
        photo_count = creator.session.query(Photo).filter(
            Photo.filename.like('DSC%')
        ).count()
        
        if photo_count == 0:
            print("❌ No photos found in database. Please import photos first.")
            print("   Run: python scripts/import_photos.py /path/to/photos")
            return
        
        print(f"✅ Found {photo_count} photos in database\n")
        
        # Create albums
        top_30_album = creator.create_top_30_album()
        decisive_album = creator.create_decisive_moments_album()
        category_albums = creator.create_category_albums()
        
        # Show summary
        print("\n" + "="*60)
        print("ALBUM CREATION SUMMARY")
        print("="*60)
        
        print(f"\n✅ Created {2 + len(category_albums)} albums:")
        print(f"   • {top_30_album.display_name}: {top_30_album.photo_count} photos")
        print(f"   • {decisive_album.display_name}: {decisive_album.photo_count} photos")
        
        for album in category_albums:
            print(f"   • {album.display_name}: {album.photo_count} photos")
        
        # List all albums
        creator.list_albums()
        
    finally:
        creator.close()


if __name__ == "__main__":
    main()