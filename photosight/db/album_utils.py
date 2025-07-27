"""
Album management utilities for PhotoSight.

Provides functions to create, query, and manage photo albums using tags.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session, joinedload

from .models import Photo, AlbumTag, PhotoAlbumTag, AnalysisResult

logger = logging.getLogger(__name__)


class AlbumManager:
    """Manages photo albums through tags."""
    
    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session
    
    def create_album(self, tag_name: str, display_name: Optional[str] = None,
                    description: Optional[str] = None, 
                    metadata: Optional[Dict] = None) -> AlbumTag:
        """
        Create or update an album tag.
        
        Args:
            tag_name: Unique tag identifier
            display_name: Human-readable name
            description: Album description
            metadata: Additional metadata
            
        Returns:
            AlbumTag instance
        """
        album = self.session.query(AlbumTag).filter_by(tag_name=tag_name).first()
        
        if not album:
            album = AlbumTag(
                tag_name=tag_name,
                display_name=display_name or tag_name.replace('-', ' ').title(),
                description=description,
                album_metadata=metadata or {}
            )
            self.session.add(album)
            logger.info(f"Created album: {tag_name}")
        else:
            # Update existing album
            if display_name:
                album.display_name = display_name
            if description:
                album.description = description
            if metadata:
                album.album_metadata.update(metadata)
            album.updated_at = datetime.utcnow()
            logger.info(f"Updated album: {tag_name}")
        
        self.session.commit()
        return album
    
    def add_photos_to_album(self, album_tag: str, photo_ids: List[int],
                           metadata_list: Optional[List[Dict]] = None) -> int:
        """
        Add multiple photos to an album.
        
        Args:
            album_tag: Album tag name
            photo_ids: List of photo IDs to add
            metadata_list: Optional metadata for each photo
            
        Returns:
            Number of photos added
        """
        # Get or create album
        album = self.session.query(AlbumTag).filter_by(tag_name=album_tag).first()
        if not album:
            album = self.create_album(album_tag)
        
        added_count = 0
        for i, photo_id in enumerate(photo_ids):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            
            # Check if already in album
            existing = self.session.query(PhotoAlbumTag).filter_by(
                photo_id=photo_id,
                album_tag_id=album.id
            ).first()
            
            if not existing:
                association = PhotoAlbumTag(
                    photo_id=photo_id,
                    album_tag_id=album.id,
                    sort_order=i,
                    association_metadata=metadata
                )
                self.session.add(association)
                added_count += 1
        
        # Update album statistics
        album.photo_count = self.session.query(PhotoAlbumTag).filter_by(
            album_tag_id=album.id
        ).count()
        album.last_photo_added = datetime.utcnow()
        
        self.session.commit()
        logger.info(f"Added {added_count} photos to album {album_tag}")
        
        return added_count
    
    def get_album_photos(self, album_tag: str, limit: Optional[int] = None,
                        order_by: str = 'sort_order') -> List[Tuple[Photo, Dict]]:
        """
        Get photos in an album.
        
        Args:
            album_tag: Album tag name
            limit: Maximum number of photos to return
            order_by: Sort field ('sort_order', 'date_taken', 'score')
            
        Returns:
            List of (Photo, metadata) tuples
        """
        album = self.session.query(AlbumTag).filter_by(tag_name=album_tag).first()
        if not album:
            return []
        
        query = self.session.query(Photo, PhotoAlbumTag).join(
            PhotoAlbumTag, Photo.id == PhotoAlbumTag.photo_id
        ).filter(
            PhotoAlbumTag.album_tag_id == album.id
        )
        
        # Apply ordering
        if order_by == 'sort_order':
            query = query.order_by(PhotoAlbumTag.sort_order)
        elif order_by == 'date_taken':
            query = query.order_by(Photo.date_taken.desc())
        elif order_by == 'score':
            # Join with latest analysis results
            subquery = self.session.query(
                AnalysisResult.photo_id,
                func.max(AnalysisResult.created_at).label('latest')
            ).group_by(AnalysisResult.photo_id).subquery()
            
            query = query.outerjoin(
                AnalysisResult,
                and_(
                    AnalysisResult.photo_id == Photo.id,
                    AnalysisResult.created_at == subquery.c.latest
                )
            ).order_by(AnalysisResult.overall_ai_score.desc())
        
        if limit:
            query = query.limit(limit)
        
        results = []
        for photo, association in query.all():
            results.append((photo, association.association_metadata))
        
        return results
    
    def get_photo_albums(self, photo_id: int) -> List[AlbumTag]:
        """
        Get all albums containing a specific photo.
        
        Args:
            photo_id: Photo ID
            
        Returns:
            List of AlbumTag instances
        """
        albums = self.session.query(AlbumTag).join(
            PhotoAlbumTag, AlbumTag.id == PhotoAlbumTag.album_tag_id
        ).filter(
            PhotoAlbumTag.photo_id == photo_id
        ).order_by(AlbumTag.updated_at.desc()).all()
        
        return albums
    
    def list_albums(self, pattern: Optional[str] = None,
                   min_photos: int = 0) -> List[Dict]:
        """
        List all albums with statistics.
        
        Args:
            pattern: Optional tag name pattern (e.g., 'enneagram-%')
            min_photos: Minimum number of photos required
            
        Returns:
            List of album information dictionaries
        """
        query = self.session.query(AlbumTag)
        
        if pattern:
            query = query.filter(AlbumTag.tag_name.like(pattern))
        
        if min_photos > 0:
            query = query.filter(AlbumTag.photo_count >= min_photos)
        
        albums = query.order_by(AlbumTag.photo_count.desc()).all()
        
        results = []
        for album in albums:
            # Get sample photos
            sample_photos = self.session.query(Photo).join(
                PhotoAlbumTag, Photo.id == PhotoAlbumTag.photo_id
            ).filter(
                PhotoAlbumTag.album_tag_id == album.id
            ).order_by(PhotoAlbumTag.sort_order).limit(3).all()
            
            results.append({
                'tag_name': album.tag_name,
                'display_name': album.display_name,
                'description': album.description,
                'photo_count': album.photo_count or 0,
                'created_at': album.created_at,
                'updated_at': album.updated_at,
                'is_public': album.is_public,
                'sample_photos': [p.filename for p in sample_photos],
                'metadata': album.album_metadata
            })
        
        return results
    
    def remove_photo_from_album(self, album_tag: str, photo_id: int) -> bool:
        """
        Remove a photo from an album.
        
        Args:
            album_tag: Album tag name
            photo_id: Photo ID to remove
            
        Returns:
            True if removed, False if not found
        """
        album = self.session.query(AlbumTag).filter_by(tag_name=album_tag).first()
        if not album:
            return False
        
        association = self.session.query(PhotoAlbumTag).filter_by(
            photo_id=photo_id,
            album_tag_id=album.id
        ).first()
        
        if association:
            self.session.delete(association)
            
            # Update album count
            album.photo_count = self.session.query(PhotoAlbumTag).filter_by(
                album_tag_id=album.id
            ).count()
            
            self.session.commit()
            logger.info(f"Removed photo {photo_id} from album {album_tag}")
            return True
        
        return False
    
    def delete_album(self, album_tag: str) -> bool:
        """
        Delete an album (removes all photo associations).
        
        Args:
            album_tag: Album tag name to delete
            
        Returns:
            True if deleted, False if not found
        """
        album = self.session.query(AlbumTag).filter_by(tag_name=album_tag).first()
        if not album:
            return False
        
        # Delete all associations (cascade should handle this)
        self.session.delete(album)
        self.session.commit()
        
        logger.info(f"Deleted album: {album_tag}")
        return True
    
    def merge_albums(self, source_tag: str, target_tag: str) -> int:
        """
        Merge one album into another.
        
        Args:
            source_tag: Source album to merge from
            target_tag: Target album to merge into
            
        Returns:
            Number of photos moved
        """
        source_album = self.session.query(AlbumTag).filter_by(tag_name=source_tag).first()
        target_album = self.session.query(AlbumTag).filter_by(tag_name=target_tag).first()
        
        if not source_album or not target_album:
            raise ValueError("Both albums must exist")
        
        # Get all photos from source album
        associations = self.session.query(PhotoAlbumTag).filter_by(
            album_tag_id=source_album.id
        ).all()
        
        moved_count = 0
        for assoc in associations:
            # Check if photo already in target
            existing = self.session.query(PhotoAlbumTag).filter_by(
                photo_id=assoc.photo_id,
                album_tag_id=target_album.id
            ).first()
            
            if not existing:
                # Create new association
                new_assoc = PhotoAlbumTag(
                    photo_id=assoc.photo_id,
                    album_tag_id=target_album.id,
                    sort_order=assoc.sort_order,
                    association_metadata=assoc.association_metadata,
                    added_by=f"merged_from_{source_tag}"
                )
                self.session.add(new_assoc)
                moved_count += 1
        
        # Update target album count
        target_album.photo_count = self.session.query(PhotoAlbumTag).filter_by(
            album_tag_id=target_album.id
        ).count()
        
        # Delete source album
        self.session.delete(source_album)
        self.session.commit()
        
        logger.info(f"Merged {moved_count} photos from {source_tag} to {target_tag}")
        return moved_count


def get_album_manager(session: Session) -> AlbumManager:
    """Get an AlbumManager instance."""
    return AlbumManager(session)