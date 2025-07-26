"""
MCP Tools for PhotoSight API Operations

Exposes the new API functionality through MCP tools for AI assistants.
"""

from typing import Dict, Any, List, Optional
import aiohttp
import asyncio
import logging
from pathlib import Path
import base64

from mcp import types

logger = logging.getLogger(__name__)


class APITools:
    """MCP tools for interacting with PhotoSight API"""
    
    def __init__(self, api_base_url: str = "http://localhost:5000/api/v1", 
                 api_token: Optional[str] = None):
        self.api_base_url = api_base_url
        self.api_token = api_token
        self.headers = {}
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"
    
    def get_tools(self) -> List[types.Tool]:
        """Get list of API-related MCP tools"""
        return [
            types.Tool(
                name="upload_photo",
                description="Upload a RAW photo for processing",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the RAW photo file"
                        },
                        "auto_process": {
                            "type": "boolean",
                            "description": "Automatically start processing after upload",
                            "default": False
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional project ID to associate with the photo"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            types.Tool(
                name="process_photo",
                description="Process a photo with custom settings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_id": {
                            "type": "string",
                            "description": "ID of the photo to process"
                        },
                        "recipe": {
                            "type": "object",
                            "description": "Processing recipe",
                            "properties": {
                                "exposure": {"type": "number", "minimum": -5, "maximum": 5},
                                "contrast": {"type": "number", "minimum": -100, "maximum": 100},
                                "highlights": {"type": "number", "minimum": -100, "maximum": 100},
                                "shadows": {"type": "number", "minimum": -100, "maximum": 100},
                                "vibrance": {"type": "number", "minimum": -100, "maximum": 100},
                                "saturation": {"type": "number", "minimum": -100, "maximum": 100},
                                "clarity": {"type": "number", "minimum": -100, "maximum": 100},
                                "white_balance": {
                                    "type": "object",
                                    "properties": {
                                        "temperature": {"type": "integer", "minimum": 2000, "maximum": 10000},
                                        "tint": {"type": "integer", "minimum": -100, "maximum": 100}
                                    }
                                }
                            }
                        }
                    },
                    "required": ["photo_id"]
                }
            ),
            types.Tool(
                name="create_batch_job",
                description="Create a batch processing job for multiple photos",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of photo IDs to process"
                        },
                        "recipe": {
                            "type": "object",
                            "description": "Processing recipe to apply to all photos"
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["jpeg", "png", "tiff", "dng"],
                            "default": "jpeg"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "normal", "high"],
                            "default": "normal"
                        }
                    },
                    "required": ["photo_ids", "recipe"]
                }
            ),
            types.Tool(
                name="get_batch_status",
                description="Get status of a batch processing job",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "Batch job ID"
                        }
                    },
                    "required": ["job_id"]
                }
            ),
            types.Tool(
                name="create_editing_session",
                description="Create an interactive editing session for a photo",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_id": {
                            "type": "string",
                            "description": "ID of the photo to edit"
                        }
                    },
                    "required": ["photo_id"]
                }
            ),
            types.Tool(
                name="update_session",
                description="Update editing session with new settings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Editing session ID"
                        },
                        "recipe": {
                            "type": "object",
                            "description": "Updated processing recipe"
                        },
                        "action": {
                            "type": "string",
                            "enum": ["update", "undo", "redo", "reset"],
                            "description": "Action to perform"
                        }
                    },
                    "required": ["session_id"]
                }
            ),
            types.Tool(
                name="export_photo",
                description="Export a processed photo",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_id": {
                            "type": "string",
                            "description": "ID of the photo to export"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["jpeg", "png", "tiff", "dng"],
                            "default": "jpeg"
                        },
                        "quality": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 90
                        },
                        "resize": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "width": {"type": "integer"},
                                "height": {"type": "integer"}
                            }
                        }
                    },
                    "required": ["photo_id"]
                }
            ),
            types.Tool(
                name="get_processing_stats",
                description="Get processing statistics and queue status",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_queue_status": {
                            "type": "boolean",
                            "description": "Include current queue status",
                            "default": True
                        }
                    }
                }
            ),
            types.Tool(
                name="apply_preset",
                description="Apply a predefined processing preset",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_id": {
                            "type": "string",
                            "description": "ID of the photo"
                        },
                        "preset_name": {
                            "type": "string",
                            "description": "Name of the preset",
                            "enum": ["portrait_natural", "portrait_dramatic", "landscape_vivid", 
                                   "landscape_subtle", "wedding_classic", "wedding_modern",
                                   "street_contrasty", "street_faded", "auto_enhance"]
                        }
                    },
                    "required": ["photo_id", "preset_name"]
                }
            )
        ]
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle API tool calls"""
        try:
            if name == "upload_photo":
                result = await self._upload_photo(arguments)
            elif name == "process_photo":
                result = await self._process_photo(arguments)
            elif name == "create_batch_job":
                result = await self._create_batch_job(arguments)
            elif name == "get_batch_status":
                result = await self._get_batch_status(arguments)
            elif name == "create_editing_session":
                result = await self._create_editing_session(arguments)
            elif name == "update_session":
                result = await self._update_session(arguments)
            elif name == "export_photo":
                result = await self._export_photo(arguments)
            elif name == "get_processing_stats":
                result = await self._get_processing_stats(arguments)
            elif name == "apply_preset":
                result = await self._apply_preset(arguments)
            else:
                return [types.TextContent(
                    type="text",
                    text=f"Unknown API tool: {name}"
                )]
            
            return [types.TextContent(
                type="text",
                text=str(result)
            )]
            
        except Exception as e:
            logger.error(f"API tool error: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    async def _upload_photo(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Upload a photo via API"""
        file_path = Path(args["file_path"])
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', open(file_path, 'rb'), 
                          filename=file_path.name)
            data.add_field('auto_process', str(args.get('auto_process', False)))
            
            if args.get('project_id'):
                data.add_field('project_id', args['project_id'])
            
            async with session.post(
                f"{self.api_base_url}/photos/upload",
                headers=self.headers,
                data=data
            ) as response:
                return await response.json()
    
    async def _process_photo(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Process a photo with given recipe"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base_url}/photos/{args['photo_id']}/process",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"recipe": args.get("recipe", {})}
            ) as response:
                return await response.json()
    
    async def _create_batch_job(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create batch processing job"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base_url}/batch",
                headers={**self.headers, "Content-Type": "application/json"},
                json=args
            ) as response:
                return await response.json()
    
    async def _get_batch_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get batch job status"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.api_base_url}/batch/{args['job_id']}",
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def _create_editing_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create editing session"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base_url}/sessions",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"photo_id": args["photo_id"]}
            ) as response:
                return await response.json()
    
    async def _update_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update editing session"""
        async with aiohttp.ClientSession() as session:
            data = {}
            if "recipe" in args:
                data["recipe"] = args["recipe"]
            if "action" in args:
                data["action"] = args["action"]
            
            async with session.put(
                f"{self.api_base_url}/sessions/{args['session_id']}",
                headers={**self.headers, "Content-Type": "application/json"},
                json=data
            ) as response:
                return await response.json()
    
    async def _export_photo(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Export processed photo"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base_url}/photos/{args['photo_id']}/export",
                headers={**self.headers, "Content-Type": "application/json"},
                json={
                    "format": args.get("format", "jpeg"),
                    "quality": args.get("quality", 90),
                    "resize": args.get("resize")
                }
            ) as response:
                return await response.json()
    
    async def _get_processing_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get processing statistics"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.api_base_url}/system/stats",
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def _apply_preset(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Apply processing preset"""
        # Define presets
        presets = {
            "portrait_natural": {
                "exposure": 0.3,
                "contrast": 5,
                "highlights": -10,
                "shadows": 15,
                "vibrance": 10,
                "clarity": 5
            },
            "portrait_dramatic": {
                "exposure": -0.2,
                "contrast": 20,
                "highlights": -30,
                "shadows": 25,
                "clarity": 15,
                "saturation": -5
            },
            "landscape_vivid": {
                "exposure": 0.2,
                "contrast": 15,
                "highlights": -20,
                "shadows": 20,
                "vibrance": 25,
                "clarity": 20
            },
            "wedding_classic": {
                "exposure": 0.5,
                "contrast": -5,
                "highlights": -15,
                "shadows": 20,
                "vibrance": 8,
                "clarity": 0
            },
            "auto_enhance": {
                "exposure": 0.0,
                "contrast": 10,
                "highlights": -10,
                "shadows": 10,
                "vibrance": 15,
                "clarity": 10
            }
        }
        
        preset = presets.get(args["preset_name"])
        if not preset:
            raise ValueError(f"Unknown preset: {args['preset_name']}")
        
        return await self._process_photo({
            "photo_id": args["photo_id"],
            "recipe": preset
        })