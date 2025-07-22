"""
Base plugin classes and interfaces for PhotoSight plugin system.

Defines the plugin contract that all PhotoSight plugins must implement.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


class PluginType(Enum):
    """Types of plugins supported by PhotoSight."""
    PROCESSOR = "processor"      # Image processing plugins
    ANALYZER = "analyzer"        # Analysis plugins  
    EXPORTER = "exporter"       # Export format plugins
    IMPORTER = "importer"       # Import source plugins
    OPTIMIZER = "optimizer"     # Recipe optimization plugins
    VISUALIZATION = "visualization"  # Data visualization plugins


@dataclass
class PluginMetadata:
    """Metadata for a PhotoSight plugin."""
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = None
    config_schema: Dict[str, Any] = None
    homepage: Optional[str] = None
    license: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'description': self.description,
            'plugin_type': self.plugin_type.value,
            'dependencies': self.dependencies or [],
            'config_schema': self.config_schema or {},
            'homepage': self.homepage,
            'license': self.license
        }


class Plugin(ABC):
    """
    Base class for all PhotoSight plugins.
    
    All plugins must inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin-specific configuration
        """
        self.config = config or {}
        self._initialized = False
        
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            PluginMetadata object describing the plugin
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the plugin.
        
        This method is called once when the plugin is loaded.
        Perform any setup operations here.
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the plugin.
        
        This method is called when the plugin is being unloaded.
        Clean up any resources here.
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate plugin configuration against schema.
        
        Returns:
            True if configuration is valid
        """
        # Basic validation - can be overridden
        metadata = self.get_metadata()
        if not metadata.config_schema:
            return True
            
        # TODO: Implement JSON schema validation
        return True
    
    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized


class ProcessorPlugin(Plugin):
    """
    Base class for image processor plugins.
    
    Processor plugins can modify images during the processing pipeline.
    """
    
    @abstractmethod
    def process(self, 
                image_path: str,
                metadata: Dict[str, Any],
                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an image.
        
        Args:
            image_path: Path to the image file
            metadata: Image metadata
            context: Processing context (recipe, settings, etc.)
            
        Returns:
            Dictionary with processing results
        """
        pass
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported image formats.
        
        Returns:
            List of file extensions (e.g., ['.cr2', '.nef'])
        """
        return []


class AnalyzerPlugin(Plugin):
    """
    Base class for analysis plugins.
    
    Analyzer plugins can extract additional information from images.
    """
    
    @abstractmethod
    def analyze(self,
                image_path: str,
                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an image.
        
        Args:
            image_path: Path to the image file
            metadata: Image metadata
            
        Returns:
            Dictionary with analysis results
        """
        pass
    
    def get_analysis_fields(self) -> List[str]:
        """
        Get list of fields this analyzer produces.
        
        Returns:
            List of field names added to analysis results
        """
        return []


class ExporterPlugin(Plugin):
    """
    Base class for export plugins.
    
    Exporter plugins can export photos and data to various formats.
    """
    
    @abstractmethod
    def export(self,
               photos: List[Dict[str, Any]],
               output_path: str,
               options: Dict[str, Any]) -> bool:
        """
        Export photos to the target format.
        
        Args:
            photos: List of photo data to export
            output_path: Destination path
            options: Export options
            
        Returns:
            True if export was successful
        """
        pass
    
    def get_export_options(self) -> Dict[str, Any]:
        """
        Get available export options.
        
        Returns:
            Dictionary describing available options
        """
        return {}


class ImporterPlugin(Plugin):
    """
    Base class for import plugins.
    
    Importer plugins can import photos from various sources.
    """
    
    @abstractmethod
    def import_photos(self,
                     source: str,
                     options: Dict[str, Any]) -> List[str]:
        """
        Import photos from source.
        
        Args:
            source: Import source (path, URL, etc.)
            options: Import options
            
        Returns:
            List of imported photo paths
        """
        pass
    
    def validate_source(self, source: str) -> bool:
        """
        Validate import source.
        
        Args:
            source: Source to validate
            
        Returns:
            True if source is valid
        """
        return True


class OptimizerPlugin(Plugin):
    """
    Base class for recipe optimizer plugins.
    
    Optimizer plugins can suggest recipe improvements.
    """
    
    @abstractmethod
    def optimize(self,
                 recipe: Dict[str, Any],
                 sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize a processing recipe.
        
        Args:
            recipe: Current recipe
            sample_results: Results from sample images
            
        Returns:
            Optimized recipe
        """
        pass
    
    def get_optimization_metrics(self) -> List[str]:
        """
        Get metrics used for optimization.
        
        Returns:
            List of metric names
        """
        return []


class VisualizationPlugin(Plugin):
    """
    Base class for visualization plugins.
    
    Visualization plugins can create charts and visualizations from photo data.
    """
    
    @abstractmethod
    def visualize(self,
                  data: Dict[str, Any],
                  output_path: str,
                  options: Dict[str, Any]) -> bool:
        """
        Create visualization from data.
        
        Args:
            data: Data to visualize
            output_path: Output file path
            options: Visualization options
            
        Returns:
            True if visualization was created successfully
        """
        pass
    
    def get_visualization_types(self) -> List[str]:
        """
        Get available visualization types.
        
        Returns:
            List of visualization type names
        """
        return []