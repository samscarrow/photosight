"""
PhotoSight Plugin Architecture

Provides extensibility through a plugin system that allows third-party
developers to add custom processors, analyzers, and exporters.
"""

from .base import Plugin, PluginType, PluginMetadata
from .manager import PluginManager
from .loader import PluginLoader

__all__ = [
    'Plugin',
    'PluginType', 
    'PluginMetadata',
    'PluginManager',
    'PluginLoader'
]