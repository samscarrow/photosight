# PhotoSight MCP Server Documentation

## Overview

The PhotoSight MCP (Model Context Protocol) Server provides a secure, read-only interface for AI assistants to query and analyze your photo database. It enables natural language queries, statistics generation, and photography insights while maintaining strict security controls.

## Features

- **Natural Language Queries**: Search photos using everyday language
- **Project Management**: Query projects, tasks, and workflow status
- **Analytics & Statistics**: Generate insights about gear usage, shooting patterns, and quality
- **Personalized Recommendations**: Get AI-powered suggestions for improvement
- **Read-Only Security**: All operations are strictly read-only to protect your data

## Installation

### Prerequisites

```bash
# Install MCP SDK
pip install mcp

# Install PhotoSight with MCP support
pip install -e ".[mcp]"
```

### Configuration

Add MCP server configuration to your `config.yaml`:

```yaml
database:
  enabled: true
  url: postgresql://user:password@localhost/photosight
  
  # Optional: Dedicated read-only user for MCP
  mcp_server:
    read_only_user: photosight_readonly
    max_query_time: 30000  # 30 seconds
    allowed_operations:
      - SELECT
```

## Running the MCP Server

### Standalone Mode (stdio)

```bash
# Run with default config
python -m photosight.mcp.server

# Run with custom config
python -m photosight.mcp.server --config /path/to/config.yaml
```

### Integration with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "photosight": {
      "command": "python",
      "args": ["-m", "photosight.mcp.server", "--config", "/path/to/config.yaml"],
      "env": {
        "PHOTOSIGHT_CONFIG": "/path/to/config.yaml"
      }
    }
  }
}
```

## Available Tools

### 1. query_photos

Search photos using natural language queries.

**Examples:**
- "Show me sharp portraits with 85mm lens"
- "Find high ISO photos from last month"
- "Wide angle landscapes with GPS from the Smith Wedding project"

**Parameters:**
- `query` (required): Natural language search query
- `limit` (optional): Max results (default: 50, max: 100)
- `project` (optional): Filter by project name

### 2. get_statistics

Generate photography statistics and analytics.

**Types:**
- `gear`: Camera and lens usage statistics
- `shooting`: Shooting patterns (ISO, aperture, etc.)
- `quality`: Acceptance rates and rejection reasons
- `temporal`: Time-based patterns
- `location`: Geographic distribution

**Parameters:**
- `type` (required): Statistics type
- `period` (optional): Time period (e.g., "30days", "6months", "all")
- `project` (optional): Filter by project name

### 3. get_insights

Get personalized photography insights and recommendations.

**Types:**
- `gear_recommendations`: Suggestions for new equipment
- `quality_improvement`: Tips to reduce rejections
- `shooting_patterns`: Analysis of your photography style

**Parameters:**
- `type` (required): Insight type
- `context` (optional): Additional context for insights

### 4. query_projects

Query project management information.

**Actions:**
- `list_projects`: List all projects with optional filters
- `project_status`: Get detailed status of a specific project
- `list_tasks`: List tasks with filtering options
- `project_analytics`: Generate project-specific analytics

**Parameters:**
- `action` (required): Query action
- `project_name` (optional): Specific project name
- `filters` (optional): Additional filters (status, due_days, priority, etc.)

## Natural Language Query Examples

### Photo Searches
```
"Sharp portraits taken with Sony A7III"
"Landscape photos at golden hour with low ISO"
"All photos from the Johnson Wedding project"
"Blurry images that were rejected last month"
"Photos taken at f/1.4 with 85mm lens"
```

### Project Queries
```
"Show me the status of the Smith Wedding project"
"List all active projects due this week"
"What tasks are pending for the Corporate Headshots project?"
"Show project analytics for completed weddings"
```

### Statistics Requests
```
"What cameras do I use most?"
"Show my shooting statistics for the past 6 months"
"What are my most common rejection reasons?"
"Analyze gear usage for portrait projects"
```

## Security Features

### Read-Only Operations
- All database queries are strictly SELECT only
- No INSERT, UPDATE, DELETE, or DDL operations allowed
- Transactions are read-only with automatic rollback

### Query Validation
- SQL injection prevention through parameterized queries
- Natural language queries validated for forbidden operations
- Query timeouts to prevent resource exhaustion

### Access Control
- Optional dedicated read-only database user
- Schema-level permissions enforcement
- Connection encryption with SSL/TLS

## Resources

The MCP server provides access to structured resources:

### Database Schema
- `photosight://schema/tables` - List of all tables
- `photosight://schema/photos` - Photo table structure
- `photosight://schema/projects` - Project table structure

### Metadata Definitions
- `photosight://metadata/exif` - EXIF field definitions
- `photosight://metadata/promoted_fields` - Commonly queried fields

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify database is running
   - Check connection string in config.yaml
   - Ensure database user has proper permissions

2. **Permission Denied Errors**
   - Verify read-only user configuration
   - Check schema permissions: `GRANT SELECT ON ALL TABLES IN SCHEMA public TO photosight_readonly;`

3. **Query Timeout**
   - Adjust `max_query_time` in configuration
   - Optimize queries with proper indexes
   - Consider adding project filters to limit scope

### Debug Mode

Enable debug logging:

```bash
export PHOTOSIGHT_LOG_LEVEL=DEBUG
python -m photosight.mcp.server
```

## Performance Optimization

### Indexing Strategy
Ensure these indexes exist for optimal performance:

```sql
-- Photo searches
CREATE INDEX idx_photos_project_id ON photos(project_id);
CREATE INDEX idx_photos_date_taken ON photos(date_taken);
CREATE INDEX idx_photos_camera_model ON photos(camera_model);
CREATE INDEX idx_photos_processing_status ON photos(processing_status);

-- Project queries
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_due_date ON projects(due_date);
CREATE INDEX idx_tasks_project_id ON tasks(project_id);
CREATE INDEX idx_tasks_status ON tasks(status);
```

### Query Optimization
- Always use project filters when possible
- Limit result sets with reasonable limits
- Use date ranges to reduce data scanned

## Integration Examples

### Python Client

```python
from photosight.mcp.client import PhotoSightMCPClient

# Connect to MCP server
client = PhotoSightMCPClient("stdio://localhost")

# Natural language query
results = await client.query_photos(
    query="sharp portraits from last month",
    project="Smith Wedding"
)

# Get statistics
stats = await client.get_statistics(
    type="gear",
    period="30days"
)
```

### Command Line

```bash
# Query photos
mcp-client call photosight query_photos '{"query": "85mm portraits", "limit": 20}'

# Get project status
mcp-client call photosight query_projects '{"action": "project_status", "project_name": "Smith Wedding"}'
```

## Future Enhancements

- HTTP transport support for remote access
- Webhook notifications for project updates
- Real-time statistics streaming
- Multi-user access control
- Query result caching