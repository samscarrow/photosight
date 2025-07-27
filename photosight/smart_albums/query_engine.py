"""
Smart Album Query Engine for PhotoSight.

Provides intelligent photo curation based on semantic analysis criteria.
Enables creation of dynamic albums like "Keynote Moments", "Best Candids", 
"High-Energy Interactions" that automatically update as new photos are analyzed.
"""

from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Lazy imports to avoid dependency issues
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from photosight.db.models import SmartAlbum, SmartAlbumTemplate, Photo, AnalysisResult
    from photosight.api.models_unified import PhotoSightBaseModel

# Dynamic imports for runtime
def _lazy_import_sqlalchemy():
    """Lazy import SQLAlchemy components."""
    try:
        from sqlalchemy import and_, or_, text, func, desc, asc
        return and_, or_, text, func, desc, asc
    except ImportError as e:
        logger.warning(f"SQLAlchemy not available: {e}")
        return None, None, None, None, None, None

def _lazy_import_models():
    """Lazy import database models."""
    try:
        from photosight.db.models import SmartAlbum, SmartAlbumTemplate, Photo, AnalysisResult, smart_album_photos, SmartAlbumRule
        return SmartAlbum, SmartAlbumTemplate, Photo, AnalysisResult, smart_album_photos, SmartAlbumRule
    except ImportError as e:
        logger.warning(f"Database models not available: {e}")
        return None, None, None, None, None, None

def _lazy_import_base_model():
    """Lazy import base model."""
    try:
        from photosight.api.models_unified import PhotoSightBaseModel
        return PhotoSightBaseModel
    except ImportError as e:
        logger.warning(f"Base model not available: {e}")
        return object  # Fallback to basic object

logger = logging.getLogger(__name__)


@dataclass
class SmartAlbumMatch:
    """Result of matching a photo against smart album criteria."""
    photo_id: int
    smart_album_id: int
    match_score: float
    matched_criteria: List[str]
    analysis_data: Dict[str, Any]


class SmartAlbumQueryEngine:
    """
    Engine for evaluating and managing Smart Albums based on semantic analysis.
    
    Key capabilities:
    - Evaluate photos against smart album criteria
    - Update smart albums when new photos are analyzed
    - Create optimized database queries for album contents
    - Suggest new smart albums based on photo patterns
    """
    
    def __init__(self, db_session):
        self.db = db_session
        self._models_cache = None
        self._sqlalchemy_cache = None
    
    @property
    def models(self):
        """Lazy load database models."""
        if self._models_cache is None:
            self._models_cache = _lazy_import_models()
        return self._models_cache
    
    @property
    def sql(self):
        """Lazy load SQLAlchemy components."""
        if self._sqlalchemy_cache is None:
            self._sqlalchemy_cache = _lazy_import_sqlalchemy()
        return self._sqlalchemy_cache
    
    def evaluate_photo_for_smart_albums(self, photo_id: int, analysis_data: Dict[str, Any]) -> List[SmartAlbumMatch]:
        """
        Evaluate a photo against all active smart albums.
        
        Args:
            photo_id: Photo to evaluate
            analysis_data: Latest analysis results for the photo
            
        Returns:
            List of smart albums that match this photo
        """
        matches = []
        
        # Get all active smart albums
        SmartAlbum, _, _, _, _, _ = self.models
        if SmartAlbum is None:
            logger.warning("SmartAlbum model not available")
            return []
            
        smart_albums = self.db.query(SmartAlbum).filter(
            SmartAlbum.is_active == True,
            SmartAlbum.auto_update == True
        ).all()
        
        for album in smart_albums:
            try:
                if album.matches_photo(analysis_data):
                    # Calculate match score based on how well criteria are met
                    match_score = self._calculate_match_score(album, analysis_data)
                    matched_criteria = self._get_matched_criteria(album, analysis_data)
                    
                    matches.append(SmartAlbumMatch(
                        photo_id=photo_id,
                        smart_album_id=album.id,
                        match_score=match_score,
                        matched_criteria=matched_criteria,
                        analysis_data=analysis_data
                    ))
                    
            except Exception as e:
                logger.error(f"Error evaluating photo {photo_id} for smart album {album.id}: {e}")
                continue
        
        return matches
    
    def update_smart_album_memberships(self, photo_id: int, analysis_data: Dict[str, Any]) -> int:
        """
        Update smart album memberships for a photo based on latest analysis.
        
        Args:
            photo_id: Photo to update
            analysis_data: Latest analysis results
            
        Returns:
            Number of smart albums updated
        """
        _, _, _, text, _, _ = self.sql
        if text is None:
            logger.warning("SQLAlchemy not available")
            return 0
            
        # Remove existing smart album associations for this photo
        self.db.execute(
            text("DELETE FROM smart_album_photos WHERE photo_id = :photo_id"),
            {"photo_id": photo_id}
        )
        
        # Find new matches
        matches = self.evaluate_photo_for_smart_albums(photo_id, analysis_data)
        
        # Add new associations
        for match in matches:
            self.db.execute(
                text("""
                    INSERT INTO smart_album_photos (smart_album_id, photo_id, match_score, added_at)
                    VALUES (:album_id, :photo_id, :score, :added_at)
                """),
                {
                    "album_id": match.smart_album_id,
                    "photo_id": match.photo_id,
                    "score": match.match_score,
                    "added_at": datetime.utcnow()
                }
            )
        
        # Update smart album photo counts and last_updated timestamps
        SmartAlbum, _, _, _, _, _ = self.models
        for match in matches:
            if SmartAlbum:
                album = self.db.query(SmartAlbum).get(match.smart_album_id)
                if album:
                    album.photo_count = self.db.execute(
                        text("SELECT COUNT(*) FROM smart_album_photos WHERE smart_album_id = :album_id"),
                        {"album_id": match.smart_album_id}
                    ).scalar()
                    album.last_updated = datetime.utcnow()
        
        self.db.commit()
        
        logger.info(f"Updated {len(matches)} smart album memberships for photo {photo_id}")
        return len(matches)
    
    def get_smart_album_photos(self, album_id: int, limit: Optional[int] = None, 
                              min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get photos in a smart album, sorted by relevance and quality.
        
        Args:
            album_id: Smart album ID
            limit: Maximum number of photos to return
            min_score: Minimum match score threshold
            
        Returns:
            List of photo data with metadata
        """
        album = self.db.query(SmartAlbum).get(album_id)
        if not album:
            return []
        
        # Build query for album photos
        query = self.db.query(
            Photo.id,
            Photo.filename,
            Photo.file_path,
            Photo.date_taken,
            smart_album_photos.c.match_score,
            smart_album_photos.c.added_at,
            AnalysisResult.overall_ai_score,
            AnalysisResult.analysis_data
        ).join(
            smart_album_photos, Photo.id == smart_album_photos.c.photo_id
        ).outerjoin(
            AnalysisResult, Photo.id == AnalysisResult.photo_id
        ).filter(
            smart_album_photos.c.smart_album_id == album_id
        )
        
        # Apply minimum score filter
        if min_score is not None:
            query = query.filter(smart_album_photos.c.match_score >= min_score)
        
        # Apply sorting based on album configuration
        if album.sort_by == 'match_score':
            if album.sort_order == 'desc':
                query = query.order_by(desc(smart_album_photos.c.match_score))
            else:
                query = query.order_by(asc(smart_album_photos.c.match_score))
        elif album.sort_by == 'ai_overall_score':
            if album.sort_order == 'desc':
                query = query.order_by(desc(AnalysisResult.overall_ai_score))
            else:
                query = query.order_by(asc(AnalysisResult.overall_ai_score))
        elif album.sort_by == 'date_taken':
            if album.sort_order == 'desc':
                query = query.order_by(desc(Photo.date_taken))
            else:
                query = query.order_by(asc(Photo.date_taken))
        else:
            # Default: sort by match score descending
            query = query.order_by(desc(smart_album_photos.c.match_score))
        
        # Apply limit
        if limit:
            query = query.limit(limit)
        
        results = query.all()
        
        return [
            {
                "id": row.id,
                "filename": row.filename,
                "file_path": row.file_path,
                "date_taken": row.date_taken.isoformat() if row.date_taken else None,
                "match_score": row.match_score,
                "added_at": row.added_at.isoformat() if row.added_at else None,
                "ai_score": row.overall_ai_score,
                "analysis_data": row.analysis_data or {}
            }
            for row in results
        ]
    
    def create_smart_album_from_template(self, template_id: int, custom_name: Optional[str] = None,
                                       custom_criteria: Optional[Dict[str, Any]] = None):
        """
        Create a new smart album from a template.
        
        Args:
            template_id: Template to use
            custom_name: Override template name
            custom_criteria: Override template criteria
            
        Returns:
            Created smart album
        """
        template = self.db.query(SmartAlbumTemplate).get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Create smart album from template
        album = SmartAlbum(
            name=custom_name or template.suggested_name or template.name,
            description=template.description,
            criteria=custom_criteria or template.criteria_template,
            logical_operator=template.logical_operator,
            icon=template.suggested_icon,
            color=template.suggested_color,
            emotional_weights=template.suggested_weights,
            created_by="system",
            is_active=True,
            auto_update=True
        )
        
        self.db.add(album)
        self.db.flush()  # Get the ID
        
        # Update template usage
        template.usage_count += 1
        
        # Populate the album with existing photos that match
        self._populate_smart_album(album.id)
        
        self.db.commit()
        
        logger.info(f"Created smart album '{album.name}' from template '{template.name}'")
        return album
    
    def suggest_smart_albums(self, analysis_sample_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Analyze photo patterns to suggest new smart albums.
        
        Args:
            analysis_sample_size: Number of recent photos to analyze
            
        Returns:
            List of suggested smart album configurations
        """
        # Get sample of recent analysis results
        recent_analyses = self.db.query(AnalysisResult).filter(
            AnalysisResult.analysis_data.isnot(None)
        ).order_by(desc(AnalysisResult.created_at)).limit(analysis_sample_size).all()
        
        if not recent_analyses:
            return []
        
        suggestions = []
        
        # Analyze patterns in the data
        patterns = self._analyze_semantic_patterns(recent_analyses)
        
        # Generate suggestions based on patterns
        for pattern in patterns:
            if pattern['frequency'] >= 10:  # At least 10 photos match
                suggestion = {
                    "suggested_name": pattern['suggested_name'],
                    "description": pattern['description'],
                    "criteria": pattern['criteria'],
                    "estimated_photo_count": pattern['frequency'],
                    "pattern_strength": pattern['strength'],
                    "category": pattern['category']
                }
                suggestions.append(suggestion)
        
        # Sort by pattern strength and frequency
        suggestions.sort(key=lambda x: (x['pattern_strength'], x['estimated_photo_count']), reverse=True)
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def _calculate_match_score(self, album, analysis_data: Dict[str, Any]) -> float:
        """Calculate how well a photo matches smart album criteria (0.0-1.0)."""
        if not album.criteria:
            return 0.0
        
        total_score = 0.0
        max_score = 0.0
        
        for rule in album.criteria:
            field = rule.get('field')
            operator = SmartAlbumRule(rule.get('operator'))
            target_value = rule.get('value')
            weight = rule.get('weight', 1.0)  # Optional rule weight
            
            field_value = album._get_nested_value(analysis_data, field)
            
            # Calculate rule score
            if album._apply_rule(field_value, operator, target_value):
                if operator in [SmartAlbumRule.GREATER_THAN, SmartAlbumRule.LESS_THAN]:
                    # For numeric comparisons, score based on how far the value exceeds threshold
                    if field_value is not None and target_value is not None:
                        try:
                            field_num = float(field_value)
                            target_num = float(target_value)
                            if operator == SmartAlbumRule.GREATER_THAN:
                                score = min(1.0, (field_num - target_num) / target_num) if target_num > 0 else 1.0
                            else:  # LESS_THAN
                                score = min(1.0, (target_num - field_num) / target_num) if target_num > 0 else 1.0
                            total_score += score * weight
                        except (ValueError, TypeError):
                            total_score += weight
                    else:
                        total_score += weight
                else:
                    # Binary match
                    total_score += weight
            
            max_score += weight
        
        return total_score / max_score if max_score > 0 else 0.0
    
    def _get_matched_criteria(self, album, analysis_data: Dict[str, Any]) -> List[str]:
        """Get list of criteria field names that matched."""
        matched = []
        
        for rule in album.criteria:
            field = rule.get('field')
            operator = SmartAlbumRule(rule.get('operator'))
            target_value = rule.get('value')
            
            field_value = album._get_nested_value(analysis_data, field)
            
            if album._apply_rule(field_value, operator, target_value):
                matched.append(field)
        
        return matched
    
    def _populate_smart_album(self, album_id: int) -> int:
        """Populate a smart album with existing photos that match its criteria."""
        album = self.db.query(SmartAlbum).get(album_id)
        if not album:
            return 0
        
        # Get all photos with analysis data
        photos_with_analysis = self.db.query(
            Photo.id, AnalysisResult.analysis_data
        ).join(
            AnalysisResult, Photo.id == AnalysisResult.photo_id
        ).filter(
            AnalysisResult.analysis_data.isnot(None)
        ).all()
        
        matches = []
        for photo_id, analysis_data in photos_with_analysis:
            if album.matches_photo(analysis_data):
                match_score = self._calculate_match_score(album, analysis_data)
                matches.append((photo_id, match_score))
        
        # Bulk insert matches
        if matches:
            insert_data = [
                {
                    "smart_album_id": album_id,
                    "photo_id": photo_id,
                    "match_score": score,
                    "added_at": datetime.utcnow()
                }
                for photo_id, score in matches
            ]
            
            self.db.execute(
                text("""
                    INSERT INTO smart_album_photos (smart_album_id, photo_id, match_score, added_at)
                    VALUES (:smart_album_id, :photo_id, :match_score, :added_at)
                """),
                insert_data
            )
            
            # Update album photo count
            album.photo_count = len(matches)
            album.last_updated = datetime.utcnow()
        
        return len(matches)
    
    def _analyze_semantic_patterns(self, analyses: List) -> List[Dict[str, Any]]:
        """Analyze semantic patterns in photo analysis data to suggest smart albums."""
        patterns = []
        
        # Group by common semantic features
        emotion_groups = {}
        scene_groups = {}
        moment_groups = {}
        
        for analysis in analyses:
            data = analysis.analysis_data or {}
            
            # Group by dominant emotion
            emotion = data.get('dominant_emotion')
            if emotion:
                if emotion not in emotion_groups:
                    emotion_groups[emotion] = []
                emotion_groups[emotion].append(analysis)
            
            # Group by scene type
            scene = data.get('scene')
            if scene:
                if scene not in scene_groups:
                    scene_groups[scene] = []
                scene_groups[scene].append(analysis)
            
            # Group by decisive moments
            if data.get('is_decisive_moment'):
                if 'decisive_moments' not in moment_groups:
                    moment_groups['decisive_moments'] = []
                moment_groups['decisive_moments'].append(analysis)
        
        # Generate patterns for emotions
        for emotion, analyses_list in emotion_groups.items():
            if len(analyses_list) >= 10:
                patterns.append({
                    'suggested_name': f"{emotion.title()} Moments",
                    'description': f"Photos capturing {emotion} emotions",
                    'criteria': [
                        {
                            'field': 'dominant_emotion',
                            'operator': 'equals',
                            'value': emotion
                        }
                    ],
                    'frequency': len(analyses_list),
                    'strength': min(1.0, len(analyses_list) / 50),
                    'category': 'emotion'
                })
        
        # Generate patterns for scenes
        for scene, analyses_list in scene_groups.items():
            if len(analyses_list) >= 10:
                patterns.append({
                    'suggested_name': f"{scene.title()} Photos",
                    'description': f"Photos from {scene} scenes",
                    'criteria': [
                        {
                            'field': 'scene',
                            'operator': 'equals',
                            'value': scene
                        }
                    ],
                    'frequency': len(analyses_list),
                    'strength': min(1.0, len(analyses_list) / 50),
                    'category': 'scene'
                })
        
        # Generate pattern for decisive moments
        if 'decisive_moments' in moment_groups and len(moment_groups['decisive_moments']) >= 5:
            patterns.append({
                'suggested_name': "Decisive Moments",
                'description': "Key moments that tell the story",
                'criteria': [
                    {
                        'field': 'is_decisive_moment',
                        'operator': 'equals',
                        'value': True
                    }
                ],
                'frequency': len(moment_groups['decisive_moments']),
                'strength': 0.9,  # High strength for decisive moments
                'category': 'moment'
            })
        
        return patterns


def create_default_smart_album_templates(db):
    """Create default smart album templates for common photography scenarios."""
    templates = [
        {
            'name': 'Keynote Moments',
            'description': 'Engaging presentation moments with speaker and attentive audience',
            'category': 'workshop',
            'criteria_template': [
                {'field': 'dominant_emotion', 'operator': 'in', 'value': ['engagement', 'focus', 'insight']},
                {'field': 'scene', 'operator': 'equals', 'value': 'presentation'}
            ],
            'suggested_weights': {'aesthetic_weight': 1.2, 'subject_weight': 1.1}
        },
        {
            'name': 'Best Candids',
            'description': 'Natural, unposed moments capturing authentic interactions',
            'category': 'workshop',
            'criteria_template': [
                {'field': 'is_decisive_moment', 'operator': 'equals', 'value': True},
                {'field': 'scene', 'operator': 'in', 'value': ['workshop/discussion', 'networking', 'break']}
            ],
            'suggested_weights': {'subject_weight': 1.3, 'face_weight': 1.2}
        },
        {
            'name': 'High-Energy Interactions',
            'description': 'Dynamic moments with multiple engaged participants',
            'category': 'workshop',
            'criteria_template': [
                {'field': 'mood', 'operator': 'equals', 'value': 'energetic'},
                {'field': 'participant_count', 'operator': 'greater_than', 'value': 5}
            ],
            'suggested_weights': {'composition_weight': 1.2, 'technical_weight': 1.1}
        },
        {
            'name': 'Portrait Excellence',
            'description': 'High-quality portrait shots with excellent face composition',
            'category': 'portrait',
            'criteria_template': [
                {'field': 'scene', 'operator': 'equals', 'value': 'portrait'},
                {'field': 'face_quality_score', 'operator': 'greater_than', 'value': 0.8}
            ],
            'suggested_weights': {'face_weight': 1.5, 'subject_weight': 1.3}
        },
        {
            'name': 'Emotional Highlights',
            'description': 'Photos capturing strong positive emotions',
            'category': 'emotion',
            'criteria_template': [
                {'field': 'dominant_emotion', 'operator': 'in', 'value': ['joy', 'excitement', 'inspiration', 'breakthrough']}
            ],
            'suggested_weights': {'aesthetic_weight': 1.4, 'face_weight': 1.2}
        }
    ]
    
    created_templates = []
    for template_data in templates:
        template = SmartAlbumTemplate(
            name=template_data['name'],
            description=template_data['description'],
            category=template_data['category'],
            criteria_template=template_data['criteria_template'],
            suggested_weights=template_data['suggested_weights'],
            suggested_name=template_data['name'],
            suggested_icon='star',
            suggested_color='#3B82F6',
            created_by='system',
            is_featured=True
        )
        db.add(template)
        created_templates.append(template)
    
    db.commit()
    logger.info(f"Created {len(created_templates)} default smart album templates")
    return created_templates