"""
Configuration management for PhotoSight
"""

import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def _expand_env_vars(obj: Union[Dict, Any]) -> Union[Dict, Any]:
    """
    Recursively expand environment variables in config values.
    Supports ${VAR_NAME} syntax.
    """
    if isinstance(obj, dict):
        return {key: _expand_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Replace ${VAR_NAME} with environment variable values
        pattern = r'\$\{([^}]+)\}'
        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # Return original if not found
        return re.sub(pattern, replace_var, obj)
    else:
        return obj

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
        # Expand environment variables in config values
        config = _expand_env_vars(config)
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
        'sync': {
            'machine_id': None,  # Auto-generated if not set
            'verify_checksums': True,
            'auto_detect_conflicts': True,
            'cloud_storage_path': None,
        },
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


def get_machine_id(config: Dict[str, Any]) -> str:
    """
    Get or generate machine ID for sync tracking.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Machine ID string
    """
    machine_id = get_config_value(config, 'sync.machine_id')
    
    if not machine_id:
        # Generate machine ID based on hostname and MAC address
        import socket
        import uuid
        
        hostname = socket.gethostname()
        mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
        machine_id = f"{hostname}-{mac}"
        
        # Save to config
        update_config_value(config, 'sync.machine_id', machine_id)
        
        # Try to save config
        config_path = Path.home() / '.photosight' / 'config.yaml'
        if config_path.parent.exists():
            save_config(config, config_path)
        
        logger.info(f"Generated machine ID: {machine_id}")
    
    return machine_id