"""
Plugin manager for PhotoSight plugin system.

Manages plugin lifecycle, registration, and execution.
"""

import logging
from typing import Dict, List, Optional, Any, Type
from collections import defaultdict

from .base import Plugin, PluginType, PluginMetadata
from .loader import PluginLoader

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Central manager for all PhotoSight plugins.
    
    Handles:
    - Plugin discovery and loading
    - Plugin lifecycle management
    - Plugin execution coordination
    - Configuration management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize plugin manager.
        
        Args:
            config: Global configuration including plugin settings
        """
        self.config = config or {}
        self.loader = PluginLoader(self.config.get('plugin_dirs', []))
        
        # Registered plugins by type
        self._plugins: Dict[PluginType, Dict[str, Plugin]] = defaultdict(dict)
        
        # Plugin metadata cache
        self._metadata: Dict[str, PluginMetadata] = {}
        
        # Disabled plugins
        self._disabled: set = set(self.config.get('disabled_plugins', []))
        
    def discover_and_load(self, auto_initialize: bool = True) -> Dict[str, Any]:
        """
        Discover and load all available plugins.
        
        Args:
            auto_initialize: Whether to initialize plugins after loading
            
        Returns:
            Summary of loaded plugins
        """
        discovered = self.loader.discover_plugins()
        loaded = []
        failed = []
        
        for plugin_info in discovered:
            plugin_name = plugin_info['metadata']['name']
            
            # Skip disabled plugins
            if plugin_name in self._disabled:
                logger.info(f"Skipping disabled plugin: {plugin_name}")
                continue
                
            # Get plugin-specific config
            plugin_config = self.config.get('plugin_configs', {}).get(plugin_name, {})
            
            # Try to load
            plugin = self.loader.load_plugin(plugin_info, plugin_config)
            if plugin and self.loader.validate_plugin(plugin):
                self.register_plugin(plugin)
                
                if auto_initialize and not plugin.is_initialized:
                    try:
                        plugin.initialize()
                        plugin._initialized = True
                    except Exception as e:
                        logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
                        failed.append(plugin_name)
                        continue
                        
                loaded.append(plugin_name)
            else:
                failed.append(plugin_name)
                
        return {
            'discovered': len(discovered),
            'loaded': loaded,
            'failed': failed,
            'disabled': list(self._disabled)
        }
    
    def register_plugin(self, plugin: Plugin) -> bool:
        """
        Register a plugin instance.
        
        Args:
            plugin: Plugin to register
            
        Returns:
            True if registration successful
        """
        try:
            metadata = plugin.get_metadata()
            plugin_type = metadata.plugin_type
            plugin_name = metadata.name
            
            # Check for conflicts
            if plugin_name in self._plugins[plugin_type]:
                logger.warning(f"Plugin {plugin_name} already registered, replacing")
                
            # Register
            self._plugins[plugin_type][plugin_name] = plugin
            self._metadata[plugin_name] = metadata
            
            logger.info(f"Registered {plugin_type.value} plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")
            return False
    
    def get_plugin(self, name: str, plugin_type: Optional[PluginType] = None) -> Optional[Plugin]:
        """
        Get a specific plugin by name.
        
        Args:
            name: Plugin name
            plugin_type: Optional plugin type filter
            
        Returns:
            Plugin instance or None
        """
        if plugin_type:
            return self._plugins[plugin_type].get(name)
            
        # Search all types
        for plugins in self._plugins.values():
            if name in plugins:
                return plugins[name]
                
        return None
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """
        Get all plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to retrieve
            
        Returns:
            List of plugin instances
        """
        return list(self._plugins[plugin_type].values())
    
    def get_all_plugins(self) -> List[Plugin]:
        """
        Get all registered plugins.
        
        Returns:
            List of all plugin instances
        """
        all_plugins = []
        for plugins in self._plugins.values():
            all_plugins.extend(plugins.values())
        return all_plugins
    
    def execute_processors(self, 
                          image_path: str,
                          metadata: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all processor plugins on an image.
        
        Args:
            image_path: Path to image
            metadata: Image metadata
            context: Processing context
            
        Returns:
            Combined results from all processors
        """
        results = {}
        processors = self.get_plugins_by_type(PluginType.PROCESSOR)
        
        for processor in processors:
            try:
                plugin_results = processor.process(image_path, metadata, context)
                results[processor.get_metadata().name] = plugin_results
            except Exception as e:
                logger.error(f"Processor {processor.get_metadata().name} failed: {e}")
                
        return results
    
    def execute_analyzers(self,
                         image_path: str,
                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all analyzer plugins on an image.
        
        Args:
            image_path: Path to image
            metadata: Image metadata
            
        Returns:
            Combined analysis results
        """
        results = {}
        analyzers = self.get_plugins_by_type(PluginType.ANALYZER)
        
        for analyzer in analyzers:
            try:
                plugin_results = analyzer.analyze(image_path, metadata)
                results.update(plugin_results)
            except Exception as e:
                logger.error(f"Analyzer {analyzer.get_metadata().name} failed: {e}")
                
        return results
    
    def shutdown_all(self):
        """Shutdown all plugins."""
        for plugin in self.get_all_plugins():
            try:
                plugin.shutdown()
                plugin._initialized = False
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin.get_metadata().name}: {e}")
                
    def enable_plugin(self, name: str):
        """Enable a previously disabled plugin."""
        if name in self._disabled:
            self._disabled.remove(name)
            logger.info(f"Enabled plugin: {name}")
            
    def disable_plugin(self, name: str):
        """Disable a plugin."""
        self._disabled.add(name)
        
        # Shutdown if currently loaded
        plugin = self.get_plugin(name)
        if plugin:
            plugin.shutdown()
            
            # Remove from registry
            metadata = self._metadata.get(name)
            if metadata:
                del self._plugins[metadata.plugin_type][name]
                del self._metadata[name]
                
        logger.info(f"Disabled plugin: {name}")
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Get information about all plugins.
        
        Returns:
            Dictionary with plugin information
        """
        info = {
            'loaded': {},
            'by_type': defaultdict(list)
        }
        
        for name, metadata in self._metadata.items():
            plugin_info = {
                'name': name,
                'version': metadata.version,
                'author': metadata.author,
                'description': metadata.description,
                'type': metadata.plugin_type.value,
                'enabled': name not in self._disabled
            }
            
            info['loaded'][name] = plugin_info
            info['by_type'][metadata.plugin_type.value].append(name)
            
        return info
    
    def reload_plugin(self, name: str) -> bool:
        """
        Reload a plugin.
        
        Args:
            name: Plugin name to reload
            
        Returns:
            True if reload successful
        """
        # Get current plugin
        plugin = self.get_plugin(name)
        if not plugin:
            logger.error(f"Plugin {name} not found")
            return False
            
        metadata = plugin.get_metadata()
        
        # Shutdown current instance
        try:
            plugin.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down plugin {name}: {e}")
            
        # Remove from registry
        del self._plugins[metadata.plugin_type][name]
        del self._metadata[name]
        
        # Re-discover and load
        discovered = self.loader.discover_plugins()
        for plugin_info in discovered:
            if plugin_info['metadata']['name'] == name:
                plugin_config = self.config.get('plugin_configs', {}).get(name, {})
                new_plugin = self.loader.load_plugin(plugin_info, plugin_config)
                
                if new_plugin and self.loader.validate_plugin(new_plugin):
                    self.register_plugin(new_plugin)
                    new_plugin.initialize()
                    new_plugin._initialized = True
                    logger.info(f"Successfully reloaded plugin: {name}")
                    return True
                    
        logger.error(f"Failed to reload plugin: {name}")
        return False