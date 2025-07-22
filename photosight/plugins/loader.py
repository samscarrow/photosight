"""
Plugin loader for PhotoSight plugin system.

Handles discovery and loading of plugins from various sources.
"""

import os
import sys
import json
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import Plugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Loads plugins from filesystem and validates them.
    
    Supports loading from:
    - Individual Python files
    - Plugin directories with metadata
    - Installed packages
    """
    
    PLUGIN_METADATA_FILE = "plugin.json"
    PLUGIN_ENTRY_POINT = "plugin.py"
    
    def __init__(self, plugin_dirs: List[str] = None):
        """
        Initialize plugin loader.
        
        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self.plugin_dirs = plugin_dirs or []
        self._add_default_dirs()
        
    def _add_default_dirs(self):
        """Add default plugin directories."""
        # User plugins directory
        user_plugins = Path.home() / ".photosight" / "plugins"
        if user_plugins not in self.plugin_dirs:
            self.plugin_dirs.append(str(user_plugins))
            
        # System plugins directory
        system_plugins = Path("/usr/local/share/photosight/plugins")
        if system_plugins.exists() and str(system_plugins) not in self.plugin_dirs:
            self.plugin_dirs.append(str(system_plugins))
            
        # Development plugins (relative to package)
        dev_plugins = Path(__file__).parent.parent / "plugins" / "contrib"
        if dev_plugins.exists() and str(dev_plugins) not in self.plugin_dirs:
            self.plugin_dirs.append(str(dev_plugins))
    
    def discover_plugins(self) -> List[Dict[str, any]]:
        """
        Discover all available plugins.
        
        Returns:
            List of plugin info dictionaries
        """
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
                
            # Check each subdirectory
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                
                if os.path.isdir(item_path):
                    plugin_info = self._discover_directory_plugin(item_path)
                    if plugin_info:
                        discovered.append(plugin_info)
                elif item.endswith('.py') and item != '__init__.py':
                    plugin_info = self._discover_file_plugin(item_path)
                    if plugin_info:
                        discovered.append(plugin_info)
        
        # Also check installed packages
        discovered.extend(self._discover_package_plugins())
        
        return discovered
    
    def _discover_directory_plugin(self, path: str) -> Optional[Dict[str, any]]:
        """Discover plugin in a directory."""
        metadata_path = os.path.join(path, self.PLUGIN_METADATA_FILE)
        entry_path = os.path.join(path, self.PLUGIN_ENTRY_POINT)
        
        if not os.path.exists(metadata_path) or not os.path.exists(entry_path):
            return None
            
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            return {
                'path': entry_path,
                'type': 'directory',
                'metadata': metadata,
                'directory': path
            }
        except Exception as e:
            logger.warning(f"Failed to load plugin metadata from {path}: {e}")
            return None
    
    def _discover_file_plugin(self, path: str) -> Optional[Dict[str, any]]:
        """Discover plugin in a single file."""
        try:
            # Try to load the file and check for plugin class
            spec = importlib.util.spec_from_file_location("plugin_temp", path)
            if not spec or not spec.loader:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for Plugin subclass
            plugin_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, Plugin) and 
                    obj != Plugin):
                    plugin_class = obj
                    break
                    
            if not plugin_class:
                return None
                
            # Try to get metadata
            try:
                instance = plugin_class()
                metadata = instance.get_metadata()
                return {
                    'path': path,
                    'type': 'file',
                    'metadata': metadata.to_dict(),
                    'class_name': plugin_class.__name__
                }
            except:
                return None
                
        except Exception as e:
            logger.debug(f"Failed to discover plugin in {path}: {e}")
            return None
    
    def _discover_package_plugins(self) -> List[Dict[str, any]]:
        """Discover plugins installed as packages."""
        discovered = []
        
        # Check for packages with photosight_plugin entry point
        try:
            import pkg_resources
            
            for entry_point in pkg_resources.iter_entry_points('photosight_plugins'):
                try:
                    plugin_class = entry_point.load()
                    instance = plugin_class()
                    metadata = instance.get_metadata()
                    
                    discovered.append({
                        'path': entry_point.module_name,
                        'type': 'package',
                        'metadata': metadata.to_dict(),
                        'entry_point': entry_point.name
                    })
                except Exception as e:
                    logger.warning(f"Failed to load plugin {entry_point.name}: {e}")
                    
        except ImportError:
            # pkg_resources not available
            pass
            
        return discovered
    
    def load_plugin(self, plugin_info: Dict[str, any], config: Dict[str, any] = None) -> Optional[Plugin]:
        """
        Load a plugin instance.
        
        Args:
            plugin_info: Plugin information from discover_plugins
            config: Plugin configuration
            
        Returns:
            Plugin instance or None if loading failed
        """
        try:
            if plugin_info['type'] == 'directory':
                return self._load_directory_plugin(plugin_info, config)
            elif plugin_info['type'] == 'file':
                return self._load_file_plugin(plugin_info, config)
            elif plugin_info['type'] == 'package':
                return self._load_package_plugin(plugin_info, config)
            else:
                logger.error(f"Unknown plugin type: {plugin_info['type']}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load plugin: {e}")
            return None
    
    def _load_directory_plugin(self, plugin_info: Dict[str, any], config: Dict[str, any]) -> Optional[Plugin]:
        """Load plugin from directory."""
        # Add plugin directory to Python path temporarily
        plugin_dir = plugin_info['directory']
        sys.path.insert(0, plugin_dir)
        
        try:
            spec = importlib.util.spec_from_file_location("plugin", plugin_info['path'])
            if not spec or not spec.loader:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for main plugin class
            plugin_class = getattr(module, 'Plugin', None)
            if not plugin_class:
                # Try to find any Plugin subclass
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type) and 
                        issubclass(obj, Plugin) and 
                        obj != Plugin):
                        plugin_class = obj
                        break
                        
            if not plugin_class:
                logger.error(f"No Plugin class found in {plugin_info['path']}")
                return None
                
            # Create instance
            instance = plugin_class(config)
            return instance
            
        finally:
            # Remove from path
            if plugin_dir in sys.path:
                sys.path.remove(plugin_dir)
    
    def _load_file_plugin(self, plugin_info: Dict[str, any], config: Dict[str, any]) -> Optional[Plugin]:
        """Load plugin from single file."""
        spec = importlib.util.spec_from_file_location("plugin", plugin_info['path'])
        if not spec or not spec.loader:
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the plugin class
        plugin_class = getattr(module, plugin_info['class_name'])
        instance = plugin_class(config)
        return instance
    
    def _load_package_plugin(self, plugin_info: Dict[str, any], config: Dict[str, any]) -> Optional[Plugin]:
        """Load plugin from installed package."""
        import pkg_resources
        
        for entry_point in pkg_resources.iter_entry_points('photosight_plugins'):
            if entry_point.name == plugin_info['entry_point']:
                plugin_class = entry_point.load()
                instance = plugin_class(config)
                return instance
                
        return None
    
    def validate_plugin(self, plugin: Plugin) -> bool:
        """
        Validate a loaded plugin.
        
        Args:
            plugin: Plugin instance to validate
            
        Returns:
            True if plugin is valid
        """
        try:
            # Check metadata
            metadata = plugin.get_metadata()
            if not metadata.name or not metadata.version:
                logger.error("Plugin missing required metadata")
                return False
                
            # Check configuration
            if not plugin.validate_config():
                logger.error("Plugin configuration validation failed")
                return False
                
            # Try to initialize
            plugin.initialize()
            plugin._initialized = True
            
            return True
            
        except Exception as e:
            logger.error(f"Plugin validation failed: {e}")
            return False