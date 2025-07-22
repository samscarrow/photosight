"""
Configuration management for PhotoSight
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values
    
    Returns:
        Default configuration dictionary
    """
    return {
        'protection': {
            'enable_protection': True,
            'verify_checksums': True,
            'read_only_mode': True,
            'create_backups': False,
        },
        'technical_filters': {
            'sharpness': {
                'laplacian_variance_minimum': 65.0,
                'use_region_based_analysis': True,
                'center_region_weight': 0.7,
                'edge_region_weight': 0.3,
                'enable_blur_recovery': True,
                'blur_recovery_threshold': 40.0,
            },
            'exposure': {
                'histogram_black_clip_threshold': 0.01,
                'histogram_white_clip_threshold': 0.01,
                'minimum_mean_brightness': 40,
                'maximum_mean_brightness': 215,
            },
        },
        'scene_classification': {
            'sky_threshold': 0.1,
            'color_temp_threshold': 1.1,
            'brightness_std_threshold': 50,
            'edge_density_threshold': 0.005,
        },
        'raw_processing': {
            'confidence_threshold': 0.7,
            'max_rotation': 10.0,
            'preview_size': 800,
            'enable_scene_aware': True,
        },
        'ai_curation': {
            'enabled': False,
            'yolo_model': 'yolov8n.pt',
            'min_ai_score': 0.5,
            'device': 'cpu',
        },
        'output': {
            'create_folders': True,
            'folder_structure': 'by_date',
            'preserve_timestamps': True,
            'copy_sidecar_files': True,
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'processing.log',
        },
    }

def save_config(config: Dict[str, Any], config_path: Path) -> bool:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the config
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        return False

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'ai_curation.enabled')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def update_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Update a nested configuration value using dot notation
    
    Args:
        config: Configuration dictionary to update
        key_path: Dot-separated key path (e.g., 'ai_curation.enabled')
        value: New value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value