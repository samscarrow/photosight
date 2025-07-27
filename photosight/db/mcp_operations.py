"""
MCP-specific database operations using direct connection.

This module provides simplified database operations for the MCP server,
using the direct connection to avoid dual database complexity.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import text

from .direct_connection import get_direct_session, execute_direct_query

logger = logging.getLogger(__name__)


class MCPProjectOperations:
    """Simplified project operations for MCP server."""
    
    @staticmethod
    def list_projects(status: Optional[str] = None, client: Optional[str] = None) -> List[Dict[str, Any]]:
        """List projects with optional filtering."""
        query = """
        SELECT 
            id,
            name,
            description,
            status,
            phase,
            start_date,
            end_date,
            created_at,
            updated_at,
            priority,
            owner_id,
            meta_data
        FROM projects
        WHERE 1=1
        """
        
        params = {}
        
        if status:
            query += " AND status = :status"
            params['status'] = status
            
        # Skip client filter since client_name doesn't exist in PHOTOSIGHT schema
            
        query += " ORDER BY created_at DESC"
        
        return execute_direct_query(query, params)
    
    @staticmethod
    def get_project_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Get project by name."""
        query = """
        SELECT 
            id,
            name,
            description,
            status,
            phase,
            start_date,
            end_date,
            created_at,
            updated_at,
            priority,
            owner_id,
            meta_data
        FROM projects
        WHERE name = :name
        """
        
        results = execute_direct_query(query, {'name': name})
        return results[0] if results else None
    
    @staticmethod
    def create_project(**kwargs) -> Optional[Dict[str, Any]]:
        """Create a new project."""
        query = """
        INSERT INTO projects (
            name, description, status, phase,
            start_date, end_date, priority, owner_id,
            created_at, updated_at, meta_data
        ) VALUES (
            :name, :description, :status, :phase,
            :start_date, :end_date, :priority, :owner_id,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, :meta_data
        )
        """
        
        params = {
            'name': kwargs.get('name'),
            'description': kwargs.get('description'),
            'status': kwargs.get('status', 'Planning'),
            'phase': kwargs.get('phase', 'Capture'),
            'start_date': kwargs.get('start_date'),
            'end_date': kwargs.get('end_date'),
            'priority': kwargs.get('priority', 'P2-Medium'),
            'owner_id': kwargs.get('owner_id'),
            'meta_data': kwargs.get('meta_data')
        }
        
        try:
            execute_direct_query(query, params)
            
            # Get the created project
            return MCPProjectOperations.get_project_by_name(kwargs.get('name'))
                
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None


class MCPTaskOperations:
    """Simplified task operations for MCP server."""
    
    @staticmethod
    def list_tasks(project_name: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""
        query = """
        SELECT 
            t.id,
            t.name,
            t.description,
            t.task_type,
            t.status,
            t.priority,
            t.assigned_to,
            t.estimated_hours,
            t.actual_hours,
            t.created_at,
            t.updated_at,
            p.name as project_name,
            p.id as project_id
        FROM tasks t
        JOIN projects p ON t.project_id = p.id
        WHERE 1=1
        """
        
        params = {}
        
        if project_name:
            query += " AND p.name = :project_name"
            params['project_name'] = project_name
            
        if status:
            query += " AND t.status = :status"
            params['status'] = status
            
        query += " ORDER BY t.priority DESC, t.created_at DESC"
        
        return execute_direct_query(query, params)
    
    @staticmethod
    def create_task(project_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Create a new task for a project."""
        # First get project ID
        project = MCPProjectOperations.get_project_by_name(project_name)
        if not project:
            logger.error(f"Project not found: {project_name}")
            return None
        
        query = """
        INSERT INTO tasks (
            project_id, name, description, task_type,
            status, priority, assigned_to, estimated_hours,
            created_at, updated_at
        ) VALUES (
            :project_id, :name, :description, :task_type,
            :status, :priority, :assigned_to, :estimated_hours,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
        )
        """
        
        params = {
            'project_id': project['ID'],
            'name': kwargs.get('task_name'),
            'description': kwargs.get('description'),
            'task_type': kwargs.get('task_type'),
            'status': 'To Do',
            'priority': kwargs.get('priority', 'P2-Medium'),
            'assigned_to': kwargs.get('assigned_to'),
            'estimated_hours': kwargs.get('estimated_hours')
        }
        
        try:
            execute_direct_query(query, params)
            
            # Return the created task
            tasks = MCPTaskOperations.list_tasks(project_name=project_name)
            for task in tasks:
                if task['NAME'] == kwargs.get('task_name'):
                    return task
            return None
            
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return None


class MCPPhotoOperations:
    """Simplified photo operations for MCP server."""
    
    @staticmethod
    def get_statistics(metric_type: str = 'overview') -> Dict[str, Any]:
        """Get photo library statistics."""
        if metric_type == 'overview':
            query = """
            SELECT 
                COUNT(*) as total_photos,
                COUNT(DISTINCT camera_make || ' ' || camera_model) as unique_cameras,
                COUNT(DISTINCT lens_model) as unique_lenses,
                SUM(CASE WHEN file_status = 'processed' THEN 1 ELSE 0 END) as processed_photos,
                SUM(CASE WHEN file_status = 'rejected' THEN 1 ELSE 0 END) as rejected_photos
            FROM photos
            """
            
            results = execute_direct_query(query)
            return results[0] if results else {
                'total_photos': 0,
                'unique_cameras': 0,
                'unique_lenses': 0,
                'processed_photos': 0,
                'rejected_photos': 0
            }
        
        elif metric_type == 'technical':
            query = """
            SELECT 
                AVG(iso) as avg_iso,
                AVG(aperture) as avg_aperture,
                AVG(focal_length) as avg_focal_length,
                COUNT(DISTINCT iso) as unique_iso_values,
                MIN(date_taken) as earliest_photo,
                MAX(date_taken) as latest_photo
            FROM photos
            WHERE iso IS NOT NULL
            """
            
            results = execute_direct_query(query)
            return results[0] if results else {}
        
        return {'message': f'Statistics for {metric_type} not yet implemented'}


class MCPMetadataOperations:
    """Metadata operations for MCP server."""
    
    @staticmethod
    def get_photo_metadata(photo_id: int) -> Dict[str, Any]:
        """Get complete metadata for a photo."""
        # Get basic photo info
        photo_query = """
        SELECT 
            id, file_path, filename, date_taken, 
            camera_make, camera_model, lens_model,
            iso, aperture, shutter_speed_display, focal_length,
            gps_latitude, gps_longitude, processing_status
        FROM photos
        WHERE id = :photo_id
        """
        
        photo_results = execute_direct_query(photo_query, {'photo_id': photo_id})
        if not photo_results:
            return {'error': 'Photo not found'}
        
        photo_data = photo_results[0]
        
        # Get keywords
        keywords_query = """
        SELECT k.id, k.keyword, k.category, pk.source
        FROM keywords k
        JOIN photo_keywords pk ON k.id = pk.keyword_id
        WHERE pk.photo_id = :photo_id
        ORDER BY k.keyword
        """
        keywords = execute_direct_query(keywords_query, {'photo_id': photo_id})
        
        # Get collections
        collections_query = """
        SELECT c.id, c.name, c.description, c.collection_type
        FROM collections c
        JOIN photo_collections pc ON c.id = pc.collection_id
        WHERE pc.photo_id = :photo_id
        ORDER BY pc.sort_order, c.name
        """
        collections = execute_direct_query(collections_query, {'photo_id': photo_id})
        
        # Get IPTC metadata
        iptc_query = """
        SELECT 
            title, caption, headline, creator, copyright_notice,
            city, region, country, event, sublocation,
            creator_email, creator_website, instructions,
            xmp_sidecar_path, xmp_sync_status
        FROM iptc_metadata
        WHERE photo_id = :photo_id
        """
        iptc_results = execute_direct_query(iptc_query, {'photo_id': photo_id})
        iptc_data = iptc_results[0] if iptc_results else {}
        
        return {
            'photo': photo_data,
            'keywords': keywords,
            'collections': collections,
            'iptc': iptc_data
        }
    
    @staticmethod
    def update_keywords(photo_id: int, keywords: List[str], source: str = 'manual') -> bool:
        """Update keywords for a photo."""
        try:
            # First, remove existing keywords if replacing all
            delete_query = "DELETE FROM photo_keywords WHERE photo_id = :photo_id"
            execute_direct_query(delete_query, {'photo_id': photo_id})
            
            # Process each keyword
            for keyword in keywords:
                if not keyword or not keyword.strip():
                    continue
                    
                keyword = keyword.strip().lower()
                
                # Check if keyword exists
                check_query = "SELECT id FROM keywords WHERE keyword = :keyword"
                existing = execute_direct_query(check_query, {'keyword': keyword})
                
                if existing:
                    keyword_id = existing[0]['ID']
                    # Update usage count
                    update_query = """
                    UPDATE keywords 
                    SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
                    WHERE id = :keyword_id
                    """
                    execute_direct_query(update_query, {'keyword_id': keyword_id})
                else:
                    # Create new keyword
                    insert_query = """
                    INSERT INTO keywords (keyword, usage_count, last_used, created_at, created_by)
                    VALUES (:keyword, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, :source)
                    """
                    execute_direct_query(insert_query, {'keyword': keyword, 'source': source})
                    
                    # Get the new keyword ID
                    existing = execute_direct_query(check_query, {'keyword': keyword})
                    keyword_id = existing[0]['ID'] if existing else None
                
                if keyword_id:
                    # Add photo-keyword association
                    assoc_query = """
                    INSERT INTO photo_keywords (photo_id, keyword_id, added_at, source)
                    VALUES (:photo_id, :keyword_id, CURRENT_TIMESTAMP, :source)
                    """
                    execute_direct_query(assoc_query, {
                        'photo_id': photo_id,
                        'keyword_id': keyword_id,
                        'source': source
                    })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update keywords: {e}")
            return False
    
    @staticmethod
    def update_iptc_metadata(photo_id: int, iptc_data: Dict[str, Any]) -> bool:
        """Update IPTC metadata for a photo."""
        try:
            # Check if IPTC record exists
            check_query = "SELECT id FROM iptc_metadata WHERE photo_id = :photo_id"
            existing = execute_direct_query(check_query, {'photo_id': photo_id})
            
            if existing:
                # Update existing record
                update_fields = []
                params = {'photo_id': photo_id}
                
                # Map of allowed fields
                allowed_fields = {
                    'title', 'caption', 'headline', 'creator', 'creator_job_title',
                    'creator_address', 'creator_city', 'creator_region', 'creator_postal_code',
                    'creator_country', 'creator_phone', 'creator_email', 'creator_website',
                    'copyright_notice', 'rights_usage_terms', 'sublocation', 'city',
                    'region', 'country', 'country_code', 'event', 'instructions',
                    'source', 'credit', 'job_id'
                }
                
                for field, value in iptc_data.items():
                    if field in allowed_fields:
                        update_fields.append(f"{field} = :{field}")
                        params[field] = value
                
                if update_fields:
                    update_query = f"""
                    UPDATE iptc_metadata 
                    SET {', '.join(update_fields)}, 
                        updated_at = CURRENT_TIMESTAMP,
                        modified_by = :modified_by
                    WHERE photo_id = :photo_id
                    """
                    params['modified_by'] = iptc_data.get('modified_by', 'user')
                    execute_direct_query(update_query, params)
            else:
                # Insert new record
                fields = ['photo_id', 'created_at', 'updated_at']
                values = [':photo_id', 'CURRENT_TIMESTAMP', 'CURRENT_TIMESTAMP']
                params = {'photo_id': photo_id}
                
                # Add provided fields
                for field, value in iptc_data.items():
                    if field != 'photo_id':  # Skip photo_id as it's already added
                        fields.append(field)
                        values.append(f':{field}')
                        params[field] = value
                
                insert_query = f"""
                INSERT INTO iptc_metadata ({', '.join(fields)})
                VALUES ({', '.join(values)})
                """
                execute_direct_query(insert_query, params)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update IPTC metadata: {e}")
            return False
    
    @staticmethod
    def add_to_collection(photo_id: int, collection_name: str) -> bool:
        """Add a photo to a collection."""
        try:
            # Get or create collection
            check_query = "SELECT id FROM collections WHERE name = :name"
            existing = execute_direct_query(check_query, {'name': collection_name})
            
            if existing:
                collection_id = existing[0]['ID']
            else:
                # Create new collection
                create_query = """
                INSERT INTO collections (name, collection_type, created_at, updated_at)
                VALUES (:name, 'manual', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
                execute_direct_query(create_query, {'name': collection_name})
                
                # Get the new collection ID
                existing = execute_direct_query(check_query, {'name': collection_name})
                collection_id = existing[0]['ID'] if existing else None
            
            if collection_id:
                # Check if already in collection
                check_assoc = """
                SELECT 1 FROM photo_collections 
                WHERE photo_id = :photo_id AND collection_id = :collection_id
                """
                existing_assoc = execute_direct_query(check_assoc, {
                    'photo_id': photo_id,
                    'collection_id': collection_id
                })
                
                if not existing_assoc:
                    # Add to collection
                    add_query = """
                    INSERT INTO photo_collections (photo_id, collection_id, added_at)
                    VALUES (:photo_id, :collection_id, CURRENT_TIMESTAMP)
                    """
                    execute_direct_query(add_query, {
                        'photo_id': photo_id,
                        'collection_id': collection_id
                    })
                    
                    # Update photo count
                    update_count = """
                    UPDATE collections 
                    SET photo_count = (
                        SELECT COUNT(*) FROM photo_collections WHERE collection_id = :collection_id
                    )
                    WHERE id = :collection_id
                    """
                    execute_direct_query(update_count, {'collection_id': collection_id})
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add to collection: {e}")
            return False
    
    @staticmethod
    def search_by_metadata(search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search photos by metadata criteria."""
        query = """
        SELECT DISTINCT p.id, p.file_path, p.filename, p.date_taken,
               p.camera_model, p.lens_model, p.processing_status
        FROM photos p
        """
        
        joins = []
        where_clauses = []
        params = {}
        
        # Search by keywords
        if 'keywords' in search_params and search_params['keywords']:
            joins.append("JOIN photo_keywords pk ON p.id = pk.photo_id")
            joins.append("JOIN keywords k ON pk.keyword_id = k.id")
            
            keyword_list = search_params['keywords']
            if isinstance(keyword_list, str):
                keyword_list = [keyword_list]
            
            keyword_conditions = []
            for i, keyword in enumerate(keyword_list):
                param_name = f'keyword_{i}'
                keyword_conditions.append(f"k.keyword = :{param_name}")
                params[param_name] = keyword.lower()
            
            where_clauses.append(f"({' OR '.join(keyword_conditions)})")
        
        # Search by collection
        if 'collection' in search_params:
            joins.append("JOIN photo_collections pc ON p.id = pc.photo_id")
            joins.append("JOIN collections c ON pc.collection_id = c.id")
            where_clauses.append("c.name = :collection")
            params['collection'] = search_params['collection']
        
        # Search by IPTC fields
        if any(field in search_params for field in ['creator', 'city', 'event', 'caption']):
            joins.append("JOIN iptc_metadata i ON p.id = i.photo_id")
            
            if 'creator' in search_params:
                where_clauses.append("i.creator ILIKE :creator")
                params['creator'] = f"%{search_params['creator']}%"
            
            if 'city' in search_params:
                where_clauses.append("i.city ILIKE :city")
                params['city'] = f"%{search_params['city']}%"
            
            if 'event' in search_params:
                where_clauses.append("i.event ILIKE :event")
                params['event'] = f"%{search_params['event']}%"
            
            if 'caption' in search_params:
                where_clauses.append("i.caption ILIKE :caption")
                params['caption'] = f"%{search_params['caption']}%"
        
        # Build final query
        if joins:
            query += "\n" + "\n".join(set(joins))  # Remove duplicate joins
        
        if where_clauses:
            query += "\nWHERE " + " AND ".join(where_clauses)
        
        query += "\nORDER BY p.date_taken DESC"
        
        return execute_direct_query(query, params)