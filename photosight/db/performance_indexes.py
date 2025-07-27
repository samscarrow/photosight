"""
Database performance optimization through strategic indexing.

Creates optimized indexes for frequently queried columns and patterns
in PhotoSight database to improve query performance.
"""

import logging
from sqlalchemy import text, Index
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class PerformanceIndexManager:
    """
    Manages performance-optimized database indexes for PhotoSight.
    
    Creates and maintains indexes for:
    - Frequently queried columns
    - Complex WHERE clause combinations
    - JOIN optimization
    - ORDER BY optimization
    - Window function support
    """
    
    def __init__(self, engine):
        """Initialize with database engine."""
        self.engine = engine
        
        # Performance-critical indexes for PhotoSight workloads
        self.performance_indexes = [
            # Photo table performance indexes
            {
                'name': 'idx_photos_date_status_perf',
                'table': 'photos',
                'columns': ['date_taken', 'processing_status', 'id'],
                'description': 'Optimizes date-based queries with status filtering'
            },
            {
                'name': 'idx_photos_camera_settings_perf',
                'table': 'photos',
                'columns': ['camera_model', 'iso', 'aperture', 'id'],
                'description': 'Optimizes camera equipment and settings queries'
            },
            {
                'name': 'idx_photos_quality_ranking',
                'table': 'photos',
                'columns': ['processing_status', 'date_taken DESC', 'id'],
                'description': 'Optimizes quality ranking queries'
            },
            {
                'name': 'idx_photos_sync_status_perf',
                'table': 'photos',
                'columns': ['file_status', 'last_sync_at', 'machine_id'],
                'description': 'Optimizes file sync and status queries'
            },
            {
                'name': 'idx_photos_gps_spatial',
                'table': 'photos',
                'columns': ['gps_latitude', 'gps_longitude'],
                'description': 'Optimizes spatial/location queries',
                'where': 'gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL'
            },
            
            # Analysis results performance indexes
            {
                'name': 'idx_analysis_photo_type_score',
                'table': 'analysis_results',
                'columns': ['photo_id', 'analysis_type', 'overall_ai_score DESC'],
                'description': 'Optimizes analysis result lookups and rankings'
            },
            {
                'name': 'idx_analysis_quality_metrics',
                'table': 'analysis_results',
                'columns': ['overall_ai_score DESC', 'sharpness_score DESC', 'created_at DESC'],
                'description': 'Optimizes quality-based photo rankings'
            },
            {
                'name': 'idx_analysis_wb_temp_fast',
                'table': 'analysis_results',
                'columns': ['wb_estimated_temp', 'wb_confidence DESC'],
                'description': 'Optimizes white balance analysis queries',
                'where': 'wb_estimated_temp IS NOT NULL'
            },
            {
                'name': 'idx_analysis_processing_perf',
                'table': 'analysis_results',
                'columns': ['processing_time_ms', 'created_at DESC'],
                'description': 'Optimizes performance analysis queries'
            },
            
            # Project and task performance indexes
            {
                'name': 'idx_projects_status_due_perf',
                'table': 'projects',
                'columns': ['status', 'due_date', 'priority', 'id'],
                'description': 'Optimizes project management dashboard queries'
            },
            {
                'name': 'idx_tasks_project_status_perf',
                'table': 'tasks',
                'columns': ['project_id', 'status', 'priority', 'due_date'],
                'description': 'Optimizes task tracking and project views'
            },
            {
                'name': 'idx_tasks_assigned_due',
                'table': 'tasks',
                'columns': ['assigned_to', 'status', 'due_date'],
                'description': 'Optimizes assignee task views'
            },
            
            # YOLO detection performance indexes
            {
                'name': 'idx_yolo_detections_perf',
                'table': 'yolo_detections',
                'columns': ['photo_id', 'class_name', 'confidence DESC'],
                'description': 'Optimizes YOLO detection queries and filtering'
            },
            {
                'name': 'idx_yolo_stats_subjects',
                'table': 'yolo_detection_stats',
                'columns': ['primary_subject', 'person_count DESC', 'object_diversity DESC'],
                'description': 'Optimizes subject-based photo discovery'
            },
            {
                'name': 'idx_yolo_runs_photo_model',
                'table': 'yolo_detection_runs',
                'columns': ['photo_id', 'model_name', 'created_at DESC'],
                'description': 'Optimizes detection run tracking'
            },
            
            # Association table performance indexes
            {
                'name': 'idx_project_photos_perf',
                'table': 'project_photos',
                'columns': ['project_id', 'added_at DESC'],
                'description': 'Optimizes project photo listings'
            },
            {
                'name': 'idx_photo_keywords_perf',
                'table': 'photo_keywords',
                'columns': ['keyword_id', 'photo_id'],
                'description': 'Optimizes keyword-based photo searches'
            },
            
            # Camera profile performance indexes
            {
                'name': 'idx_camera_profiles_perf',
                'table': 'camera_profiles',
                'columns': ['camera_make', 'camera_model', 'is_active', 'confidence_score DESC'],
                'description': 'Optimizes camera profile matching'
            },
            
            # Recipe and batch performance indexes
            {
                'name': 'idx_recipes_category_usage',
                'table': 'processing_recipes',
                'columns': ['category', 'is_active', 'times_used DESC'],
                'description': 'Optimizes recipe discovery and popularity'
            },
            
            # Collection performance indexes
            {
                'name': 'idx_collections_type_name',
                'table': 'collections',
                'columns': ['collection_type', 'name', 'updated_at DESC'],
                'description': 'Optimizes collection browsing and search'
            },
            
            # IPTC metadata performance indexes
            {
                'name': 'idx_iptc_creator_location',
                'table': 'iptc_metadata',
                'columns': ['creator', 'city', 'country'],
                'description': 'Optimizes photographer and location searches'
            },
        ]
        
        # Composite function-based indexes for complex queries
        self.function_indexes = [
            {
                'name': 'idx_photos_filename_lower',
                'table': 'photos',
                'expression': 'LOWER(filename)',
                'description': 'Case-insensitive filename searches'
            },
            {
                'name': 'idx_photos_date_trunc_month',
                'table': 'photos',
                'expression': 'DATE_TRUNC(\'month\', date_taken)',
                'description': 'Monthly photo grouping and statistics'
            },
            {
                'name': 'idx_analysis_score_bucket',
                'table': 'analysis_results',
                'expression': 'FLOOR(overall_ai_score * 10)',
                'description': 'Score-based bucketing for quality analysis'
            }
        ]
    
    def create_performance_indexes(self) -> Dict[str, bool]:
        """
        Create all performance-optimized indexes.
        
        Returns:
            Dictionary with index creation results
        """
        results = {}
        
        logger.info("Creating performance-optimized database indexes...")
        
        with self.engine.connect() as conn:
            # Create standard performance indexes
            for index_def in self.performance_indexes:
                success = self._create_index(conn, index_def)
                results[index_def['name']] = success
            
            # Create function-based indexes
            for index_def in self.function_indexes:
                success = self._create_function_index(conn, index_def)
                results[index_def['name']] = success
            
            # Commit all changes
            conn.commit()
        
        successful = sum(results.values())
        total = len(results)
        logger.info(f"Created {successful}/{total} performance indexes")
        
        return results
    
    def _create_index(self, conn, index_def: Dict) -> bool:
        """Create a standard index."""
        try:
            # Parse columns to handle DESC/ASC modifiers
            columns = []
            for col in index_def['columns']:
                if ' DESC' in col:
                    col_name = col.replace(' DESC', '')
                    columns.append(f"{col_name} DESC")
                elif ' ASC' in col:
                    col_name = col.replace(' ASC', '')
                    columns.append(f"{col_name} ASC")
                else:
                    columns.append(col)
            
            # Build CREATE INDEX statement
            columns_str = ', '.join(columns)
            sql = f"CREATE INDEX IF NOT EXISTS {index_def['name']} ON {index_def['table']} ({columns_str})"
            
            # Add WHERE clause if specified
            if 'where' in index_def:
                sql += f" WHERE {index_def['where']}"
            
            conn.execute(text(sql))
            logger.debug(f"Created index: {index_def['name']} - {index_def['description']}")
            return True
            
        except SQLAlchemyError as e:
            logger.warning(f"Failed to create index {index_def['name']}: {e}")
            return False
    
    def _create_function_index(self, conn, index_def: Dict) -> bool:
        """Create a function-based index."""
        try:
            sql = f"CREATE INDEX IF NOT EXISTS {index_def['name']} ON {index_def['table']} ({index_def['expression']})"
            conn.execute(text(sql))
            logger.debug(f"Created function index: {index_def['name']} - {index_def['description']}")
            return True
            
        except SQLAlchemyError as e:
            logger.warning(f"Failed to create function index {index_def['name']}: {e}")
            return False
    
    def analyze_index_usage(self) -> Dict:
        """
        Analyze index usage statistics (PostgreSQL specific).
        
        Returns:
            Dictionary with index usage statistics
        """
        if not self._is_postgresql():
            return {'error': 'Index analysis only supported on PostgreSQL'}
        
        try:
            with self.engine.connect() as conn:
                # Query index usage statistics
                sql = """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_tup_read,
                    idx_tup_fetch,
                    idx_scan,
                    CASE 
                        WHEN idx_scan = 0 THEN 'Never used'
                        WHEN idx_scan < 10 THEN 'Rarely used'
                        WHEN idx_scan < 100 THEN 'Moderately used'
                        ELSE 'Frequently used'
                    END as usage_level
                FROM pg_stat_user_indexes 
                WHERE schemaname = 'public'
                AND indexname LIKE 'idx_%'
                ORDER BY idx_scan DESC;
                """
                
                result = conn.execute(text(sql))
                rows = result.fetchall()
                
                # Convert to dictionary format
                index_stats = []
                for row in rows:
                    index_stats.append({
                        'schema': row[0],
                        'table': row[1], 
                        'index': row[2],
                        'tuples_read': row[3],
                        'tuples_fetched': row[4],
                        'scans': row[5],
                        'usage_level': row[6]
                    })
                
                return {
                    'index_count': len(index_stats),
                    'indexes': index_stats
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to analyze index usage: {e}")
            return {'error': str(e)}
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict]:
        """
        Get slow query statistics (PostgreSQL specific).
        
        Args:
            limit: Number of slow queries to return
            
        Returns:
            List of slow query statistics
        """
        if not self._is_postgresql():
            return []
        
        try:
            with self.engine.connect() as conn:
                # Query slow queries from pg_stat_statements
                sql = """
                SELECT 
                    query,
                    calls,
                    total_exec_time,
                    mean_exec_time,
                    max_exec_time,
                    rows,
                    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                WHERE query NOT LIKE '%pg_stat_statements%'
                AND query NOT LIKE '%pg_stat_user_indexes%'
                ORDER BY mean_exec_time DESC
                LIMIT %s;
                """
                
                result = conn.execute(text(sql), (limit,))
                rows = result.fetchall()
                
                slow_queries = []
                for row in rows:
                    slow_queries.append({
                        'query': row[0][:200] + '...' if len(row[0]) > 200 else row[0],
                        'calls': row[1],
                        'total_time_ms': round(row[2], 2),
                        'avg_time_ms': round(row[3], 2),
                        'max_time_ms': round(row[4], 2),
                        'rows_returned': row[5],
                        'cache_hit_percent': round(row[6] or 0, 2)
                    })
                
                return slow_queries
                
        except SQLAlchemyError as e:
            logger.warning(f"Failed to get slow queries: {e}")
            return []
    
    def _is_postgresql(self) -> bool:
        """Check if the database is PostgreSQL."""
        try:
            return 'postgresql' in str(self.engine.url).lower()
        except:
            return False
    
    def optimize_table_statistics(self) -> Dict[str, bool]:
        """
        Update table statistics for better query planning.
        
        Returns:
            Dictionary with table analyze results
        """
        if not self._is_postgresql():
            return {'error': 'Statistics optimization only supported on PostgreSQL'}
        
        # Critical tables that need fresh statistics
        critical_tables = [
            'photos', 'analysis_results', 'projects', 'tasks',
            'yolo_detections', 'yolo_detection_stats', 'project_photos'
        ]
        
        results = {}
        
        try:
            with self.engine.connect() as conn:
                for table in critical_tables:
                    try:
                        conn.execute(text(f"ANALYZE {table}"))
                        results[table] = True
                        logger.debug(f"Updated statistics for table: {table}")
                    except SQLAlchemyError as e:
                        logger.warning(f"Failed to analyze table {table}: {e}")
                        results[table] = False
                
                conn.commit()
            
            successful = sum(results.values())
            total = len(results)
            logger.info(f"Updated statistics for {successful}/{total} tables")
            
            return results
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to optimize table statistics: {e}")
            return {'error': str(e)}
    
    def get_table_sizes(self) -> List[Dict]:
        """
        Get table size information for capacity planning.
        
        Returns:
            List of table size information
        """
        if not self._is_postgresql():
            return []
        
        try:
            with self.engine.connect() as conn:
                sql = """
                SELECT 
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    correlation,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_stats 
                WHERE schemaname = 'public'
                AND tablename IN ('photos', 'analysis_results', 'yolo_detections', 'projects', 'tasks')
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
                """
                
                result = conn.execute(text(sql))
                rows = result.fetchall()
                
                table_info = []
                for row in rows:
                    table_info.append({
                        'schema': row[0],
                        'table': row[1],
                        'column': row[2],
                        'distinct_values': row[3],
                        'correlation': row[4],
                        'size': row[5]
                    })
                
                return table_info
                
        except SQLAlchemyError as e:
            logger.warning(f"Failed to get table sizes: {e}")
            return []


def create_all_performance_indexes(engine) -> Dict:
    """
    Convenience function to create all performance indexes.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        Dictionary with creation results and statistics
    """
    manager = PerformanceIndexManager(engine)
    
    # Create indexes
    creation_results = manager.create_performance_indexes()
    
    # Update table statistics
    stats_results = manager.optimize_table_statistics()
    
    # Get usage analysis if available
    usage_stats = manager.analyze_index_usage()
    
    return {
        'index_creation': creation_results,
        'statistics_update': stats_results,
        'usage_analysis': usage_stats,
        'summary': {
            'indexes_created': sum(creation_results.values()),
            'total_indexes': len(creation_results),
            'statistics_updated': sum(stats_results.values()) if isinstance(stats_results, dict) else 0
        }
    }