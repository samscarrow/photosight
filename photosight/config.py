"""
Configuration management for PhotoSight
Handles loading and validating configuration from YAML files
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation"""
    
    # Default configuration structure
    DEFAULT_CONFIG = {
        'technical_filters': {
            'sharpness': {
                'laplacian_variance_minimum': 100.0,
                'save_debug_heatmaps': False
            },
            'exposure': {
                'histogram_black_clip_threshold': 0.01,
                'histogram_white_clip_threshold': 0.01,
                'histogram_bins': 256,
                'minimum_mean_brightness': 50,
                'maximum_mean_brightness': 205,
                'maximum_shadow_percentage': 0.7,
                'minimum_highlight_percentage': 0.01
            },
            'metadata': {
                'maximum_iso': 12800,
                'minimum_shutter_speed_denominator': 60,
                'apply_focal_length_rule': True,
                'focal_length_rule_multiplier': 1.0
            }
        },
        'output': {
            'folders': {
                'accepted': 'accepted',
                'rejected': 'rejected',
                'rejection_reasons': {
                    'blurry': 'rejected/blurry',
                    'underexposed': 'rejected/underexposed',
                    'overexposed': 'rejected/overexposed',
                    'high_iso': 'rejected/high_iso',
                    'slow_shutter': 'rejected/slow_shutter'
                }
            },
            'preserve_folder_structure': True,
            'operation': 'copy',
            'include_sidecar_files': True
        },
        'processing': {
            'num_threads': 4,
            'batch_size': 10,
            'skip_existing': True,
            'raw_extensions': ['.ARW', '.arw']
        },
        'logging': {
            'level': 'INFO',
            'save_to_file': True,
            'log_file': 'photosight.log'
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file, or None for defaults
        """
        self.config_path = config_path
        self.config = deepcopy(self.DEFAULT_CONFIG)
        
        if config_path:
            self.load_config(config_path)
            
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                
            if user_config:
                # Merge user config with defaults
                self.config = self._merge_configs(self.DEFAULT_CONFIG, user_config)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Configuration file {config_path} is empty, using defaults")
                
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
            
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """
        Recursively merge user configuration with defaults
        
        Args:
            default: Default configuration dictionary
            user: User configuration dictionary
            
        Returns:
            Merged configuration
        """
        result = deepcopy(default)
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override with user value
                result[key] = value
                
        return result
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to YAML file
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Saved configuration to {output_path}")
        
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration key
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        # Set the value
        config[keys[-1]] = value
        
    def validate(self) -> bool:
        """
        Validate configuration values
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Check numeric thresholds
        if self.get('technical_filters.sharpness.laplacian_variance_minimum', 0) <= 0:
            errors.append("Sharpness threshold must be positive")
            
        if not 0 < self.get('technical_filters.exposure.histogram_black_clip_threshold', 0) < 1:
            errors.append("Black clip threshold must be between 0 and 1")
            
        if not 0 < self.get('technical_filters.exposure.histogram_white_clip_threshold', 0) < 1:
            errors.append("White clip threshold must be between 0 and 1")
            
        if self.get('technical_filters.metadata.maximum_iso', 0) <= 0:
            errors.append("Maximum ISO must be positive")
            
        # Check output operation
        if self.get('output.operation') not in ['copy', 'move']:
            errors.append("Output operation must be 'copy' or 'move'")
            
        # Check processing parameters
        if self.get('processing.num_threads', 0) <= 0:
            errors.append("Number of threads must be positive")
            
        if self.get('processing.batch_size', 0) <= 0:
            errors.append("Batch size must be positive")
            
        # Check logging level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.get('logging.level') not in valid_levels:
            errors.append(f"Logging level must be one of {valid_levels}")
            
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
            
        return True
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        
        # Get log level
        level_name = log_config.get('level', 'INFO')
        level = getattr(logging, level_name, logging.INFO)
        
        # Configure logging format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure handlers
        handlers = [logging.StreamHandler()]  # Always log to console
        
        if log_config.get('save_to_file', False):
            log_file = log_config.get('log_file', 'photosight.log')
            file_handler = logging.FileHandler(log_file)
            handlers.append(file_handler)
            
        # Configure root logger
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=handlers
        )
        
        logger.info(f"Logging configured: level={level_name}")
        
    def __str__(self) -> str:
        """String representation of configuration"""
        return yaml.dump(self.config, default_flow_style=False)


# Convenience function to load configuration
def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Load configuration from file or use defaults
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    # Check for default config in standard locations if not specified
    if not config_path:
        default_locations = [
            './config.yaml',
            './photosight.yaml',
            '~/.config/photosight/config.yaml',
            os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        ]
        
        for location in default_locations:
            path = Path(location).expanduser()
            if path.exists():
                config_path = str(path)
                logger.info(f"Found configuration at {path}")
                break
                
    manager = ConfigManager(config_path)
    
    if not manager.validate():
        raise ValueError("Invalid configuration")
        
    return manager