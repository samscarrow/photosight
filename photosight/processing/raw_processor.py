"""
Non-destructive RAW processing module for PhotoSight

Implements recipe-based processing that preserves original RAW files
while allowing iterative adjustments and batch processing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union, Any
from dataclasses import dataclass, asdict, field
import numpy as np
import cv2
import rawpy
import exifread
from datetime import datetime
import hashlib

# Import PhotoSight processing modules
from photosight.processing.scene_classifier import SceneClassifier
from photosight.processing.geometry.auto_straighten import AutoStraightener
from photosight.processing.geometry.smart_crop import SmartCropper, CropSuggestion
from photosight.processing.tone.exposure_optimizer import ExposureOptimizer, ExposureAnalysis
from photosight.processing.color.white_balance import WhiteBalanceCorrector, WhiteBalanceAnalysis, WhiteBalanceMethod
from photosight.processing.color.color_grading import ColorGrader, ColorGradingSettings, ColorGradingPreset
from photosight.processing.yolo_integration import analyze_preview_yolo, YOLOAnalysis
from photosight.processing.local_adjustments import AdjustmentLayer, MaskData
from photosight.processing.sharpening import (
    SharpeningSettings, SharpeningPipeline, create_default_sharpening_settings,
    InputSharpeningSettings, CreativeSharpeningSettings, OutputSharpeningSettings
)

logger = logging.getLogger(__name__)


@dataclass
class ToneMapperSettings:
    """Unified tone mapping settings with parametric curve controls"""
    
    # Global exposure
    exposure_compensation: float = 0.0  # -3.0 to +3.0 EV
    
    # Basic tone controls
    shadows: float = 0.0      # -100 to +100
    highlights: float = 0.0   # -100 to +100
    whites: float = 0.0       # -100 to +100  
    blacks: float = 0.0       # -100 to +100
    
    # Advanced curve controls (Catmull-Rom spline points)
    curve_points: List[Tuple[float, float]] = field(default_factory=list)  # (input, output) pairs
    
    # Parametric curve controls (alternative to curve_points)
    # These follow the standard parametric curve model
    shadows_split: float = 0.25      # 0.0-1.0, shadow/midtone boundary
    highlights_split: float = 0.75   # 0.0-1.0, midtone/highlight boundary
    
    # Contrast controls
    contrast: float = 0.0     # -100 to +100, S-curve strength
    clarity: float = 0.0      # -100 to +100, local contrast
    
    # Advanced tone mapping
    tone_curve_type: str = "parametric"  # "parametric", "spline", or "custom"
    preserve_highlights: bool = True
    recover_shadows: bool = True
    
    # Zone system mapping (Ansel Adams zones)
    zone_mapping_enabled: bool = False
    target_zone_distribution: Optional[List[float]] = None  # 11 zones (0-X)


@dataclass
class EXIFData:
    """
    Comprehensive EXIF data extracted from RAW files
    
    Contains camera settings, lens information, and other metadata
    that can be used to optimize processing parameters.
    """
    # Camera identification
    camera_make: str = "Unknown"
    camera_model: str = "Unknown" 
    lens_model: str = "Unknown"
    
    # Exposure settings
    iso: int = 100
    focal_length: float = 50.0  # mm
    aperture: float = 5.6       # f-stop
    shutter_speed: str = "1/60" # As string (e.g., "1/125", "2", "0.5")
    exposure_compensation: float = 0.0  # EV
    
    # White balance and color
    white_balance: str = "Auto"
    color_temp_kelvin: Optional[int] = None
    color_space: str = "sRGB"
    
    # Metering and focus
    metering_mode: str = "Unknown"
    focus_mode: str = "Unknown"
    flash_fired: bool = False
    
    # RAW-specific data
    black_level: Optional[int] = None
    white_level: Optional[int] = None
    color_matrix: Optional[List[float]] = None
    
    # Image geometry
    image_width: int = 0
    image_height: int = 0
    orientation: int = 1  # EXIF orientation
    
    # GPS location (if available)
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    
    # Timestamps
    datetime_original: Optional[str] = None
    datetime_digitized: Optional[str] = None
    
    # Camera settings
    exposure_mode: str = "Unknown"
    scene_mode: str = "Unknown"
    contrast: Optional[int] = None
    saturation: Optional[int] = None
    sharpness: Optional[int] = None
    
    # Lens corrections applied by camera
    lens_corrections_applied: bool = False
    
    # Software and processing
    software: str = "Unknown"
    camera_profile: Optional[str] = None


@dataclass
class ProcessingRecipe:
    """
    Non-destructive processing recipe for RAW images
    
    Stores all adjustments as parameters without modifying the original file.
    Can be serialized to JSON for storage and batch processing.
    """
    # File identification
    source_path: str
    file_hash: Optional[str] = None
    created_at: Optional[str] = None
    
    # Basic adjustments
    exposure_adjustment: float = 0.0  # EV adjustment (-2.0 to +2.0)
    shadows: float = 0.0  # Shadow lifting (0-100)
    highlights: float = 0.0  # Highlight recovery (-100 to 0)
    contrast: float = 0.0  # Contrast adjustment (-50 to +50)
    
    # White balance
    wb_method: Optional[str] = None  # Auto-selected or manual method
    wb_multipliers: Optional[Tuple[float, float, float]] = None
    temperature_adjustment: int = 0  # Kelvin adjustment (-2000 to +2000)
    tint_adjustment: int = 0  # Green/Magenta adjustment (-50 to +50)
    wb_analysis: Optional[Dict] = None
    
    # Color grading
    color_grading_preset: Optional[str] = None
    vibrance: float = 0.0  # Vibrance (0-100)
    saturation: float = 0.0  # Saturation (-100 to +100)
    color_grading_settings: Optional[Dict] = None
    
    # Tone curve (simplified)
    blacks: float = 0.0  # Black point adjustment (-100 to +100)
    whites: float = 0.0  # White point adjustment (-100 to +100)
    
    # Clarity and texture
    clarity: float = 0.0  # Local contrast (-100 to +100)
    texture: float = 0.0  # Fine detail enhancement (-100 to +100)
    
    # Lens corrections
    vignette_amount: float = 0.0  # Vignette correction (-100 to +100)
    
    # Geometry corrections
    rotation_angle: float = 0.0
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_width: Optional[int] = None  
    crop_height: Optional[int] = None
    
    # Smart crop suggestions
    crop_suggestions: Optional[List[Dict]] = None
    selected_crop_index: Optional[int] = None
    
    # Exposure analysis results
    exposure_analysis: Optional[Dict] = None
    auto_exposure: bool = True
    
    # Scene-aware metadata
    scene_classification: Optional[Dict] = None
    processing_hints: Optional[Dict] = None
    
    # EXIF metadata
    exif_data: Optional[EXIFData] = None
    
    # Processing metadata
    version: str = "1.0"
    notes: Optional[str] = None
    
    # Local adjustments
    adjustment_layers: List[AdjustmentLayer] = field(default_factory=list)
    
    # Noise reduction
    noise_reduction: Optional['NoiseReductionSettings'] = None
    
    # Sharpening settings
    sharpening: Optional[SharpeningSettings] = None
    
    def to_json(self) -> str:
        """Serialize recipe to JSON"""
        data = asdict(self)
        data['created_at'] = data.get('created_at') or datetime.now().isoformat()
        
        # Convert adjustment layers to dictionaries
        if 'adjustment_layers' in data and data['adjustment_layers']:
            data['adjustment_layers'] = [
                layer.to_dict() if hasattr(layer, 'to_dict') else layer
                for layer in self.adjustment_layers
            ]
        
        # Convert noise reduction settings to dict
        if 'noise_reduction' in data and data['noise_reduction']:
            nr_dict = asdict(self.noise_reduction)
            # Convert enum to string for JSON serialization
            if 'edge_preservation_method' in nr_dict:
                nr_dict['edge_preservation_method'] = nr_dict['edge_preservation_method'].value
            data['noise_reduction'] = nr_dict
        
        # Convert sharpening settings to dict
        if 'sharpening' in data and data['sharpening']:
            data['sharpening'] = asdict(self.sharpening)
        
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ProcessingRecipe':
        """Deserialize recipe from JSON"""
        data = json.loads(json_str)
        
        # Convert adjustment layer dictionaries back to objects
        if 'adjustment_layers' in data and data['adjustment_layers']:
            data['adjustment_layers'] = [
                AdjustmentLayer.from_dict(layer) if isinstance(layer, dict) else layer
                for layer in data['adjustment_layers']
            ]
        
        # Convert noise reduction dict back to object
        if 'noise_reduction' in data and data['noise_reduction']:
            from photosight.processing.noise import NoiseReductionSettings, EdgePreservationMethod
            nr_data = data['noise_reduction']
            # Convert string back to enum
            if 'edge_preservation_method' in nr_data and isinstance(nr_data['edge_preservation_method'], str):
                nr_data['edge_preservation_method'] = EdgePreservationMethod(nr_data['edge_preservation_method'])
            data['noise_reduction'] = NoiseReductionSettings(**nr_data)
        
        # Convert sharpening dict back to object
        if 'sharpening' in data and data['sharpening']:
            sharpening_data = data['sharpening']
            # Reconstruct nested dataclasses
            if 'input_sharpening' in sharpening_data:
                sharpening_data['input_sharpening'] = InputSharpeningSettings(**sharpening_data['input_sharpening'])
            if 'creative_sharpening' in sharpening_data:
                sharpening_data['creative_sharpening'] = CreativeSharpeningSettings(**sharpening_data['creative_sharpening'])
            if 'output_sharpening' in sharpening_data:
                sharpening_data['output_sharpening'] = OutputSharpeningSettings(**sharpening_data['output_sharpening'])
            data['sharpening'] = SharpeningSettings(**sharpening_data)
        
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """Save recipe to file"""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: Path) -> 'ProcessingRecipe':
        """Load recipe from file"""
        with open(path, 'r') as f:
            return cls.from_json(f.read())


class RawPostProcessor:
    """
    Non-destructive RAW post-processor
    
    Applies processing recipes to RAW files without modifying originals.
    Supports preview generation and full-resolution export.
    
    Now with database integration for persistent recipe storage!
    """
    
    def __init__(self, 
                 preview_size: int = 800,
                 preview_quality: int = 85,
                 cache_previews: bool = True,
                 auto_analyze: bool = True,
                 use_database: bool = True):
        """
        Initialize RAW post-processor
        
        Args:
            preview_size: Maximum dimension for preview images
            preview_quality: JPEG quality for previews (1-100)
            cache_previews: Whether to cache preview images
            auto_analyze: Whether to automatically analyze images for optimal settings
            use_database: Whether to use database for recipe storage
        """
        self.preview_size = preview_size
        self.preview_quality = preview_quality
        self.cache_previews = cache_previews
        self.auto_analyze = auto_analyze
        self.use_database = use_database
        self._preview_cache: Dict[str, np.ndarray] = {}
        
        # Initialize processing modules
        logger.info("Initializing PhotoSight processing modules...")
        self.scene_classifier = SceneClassifier()
        self.auto_straightener = AutoStraightener()
        self.smart_cropper = SmartCropper()
        self.exposure_optimizer = ExposureOptimizer()
        self.wb_corrector = WhiteBalanceCorrector()
        self.color_grader = ColorGrader()
        logger.info("Processing modules initialized successfully")
        
        # Initialize camera-specific profiles for optimized processing
        self._init_camera_profiles()
        
        # Initialize database recipe manager if enabled
        if self.use_database:
            try:
                from ..db.recipe_manager import RecipeManager
                self.recipe_manager = RecipeManager()
                logger.info("Database recipe manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize recipe manager: {e}")
                self.use_database = False
                self.recipe_manager = None
        else:
            self.recipe_manager = None
    
    def _init_camera_profiles(self):
        """Initialize camera-specific processing profiles"""
        self.camera_profiles = {
            # Sony Alpha 7 Mark III - Primary default camera
            'Sony ILCE-7M3': {
                'high_iso_threshold': 2500,  # Sweet spot for A7 III
                'highlight_headroom': 1.3,    # Good highlight recovery
                'shadow_lift_bias': 1.1,      # Excellent shadow performance
                'color_grading_bias': 'cinematic',  # Works well with this sensor
                'wb_confidence_boost': 0.15   # Sony auto WB is reliable
            },
            
            # Other Sony cameras - known for excellent dynamic range
            'Sony ILCE-7RM5': {
                'high_iso_threshold': 3200,
                'highlight_headroom': 1.5,  # Extra stops for highlight recovery
                'shadow_lift_bias': 1.2,    # Sony sensors handle shadow lifting well
                'color_grading_bias': 'neutral',
                'wb_confidence_boost': 0.1   # Trust Sony's auto WB more
            },
            'Sony ILCE-7M4': {
                'high_iso_threshold': 6400,
                'highlight_headroom': 1.3,
                'shadow_lift_bias': 1.1,
                'color_grading_bias': 'neutral',
                'wb_confidence_boost': 0.1
            },
            'Sony ILCE-7RM4': {
                'high_iso_threshold': 3200,
                'highlight_headroom': 1.4,
                'shadow_lift_bias': 1.2,
                'color_grading_bias': 'neutral',
                'wb_confidence_boost': 0.1
            },
            
            # Canon cameras - known for color science
            'Canon EOS R5': {
                'high_iso_threshold': 6400,
                'highlight_headroom': 1.2,
                'shadow_lift_bias': 1.0,
                'color_grading_bias': 'warm',
                'wb_confidence_boost': 0.15  # Canon's color science is excellent
            },
            'Canon EOS R6': {
                'high_iso_threshold': 12800,  # Excellent high ISO performance
                'highlight_headroom': 1.1,
                'shadow_lift_bias': 0.9,
                'color_grading_bias': 'warm',
                'wb_confidence_boost': 0.15
            },
            
            # Nikon cameras - excellent dynamic range
            'Nikon Z7II': {
                'high_iso_threshold': 3200,
                'highlight_headroom': 1.6,   # Nikon sensors excel at highlight retention
                'shadow_lift_bias': 1.3,
                'color_grading_bias': 'neutral',
                'wb_confidence_boost': 0.1
            },
            'Nikon Z9': {
                'high_iso_threshold': 6400,
                'highlight_headroom': 1.4,
                'shadow_lift_bias': 1.2,
                'color_grading_bias': 'neutral',
                'wb_confidence_boost': 0.1
            },
            
            # Fujifilm cameras - unique color rendition
            'Fujifilm X-T5': {
                'high_iso_threshold': 3200,
                'highlight_headroom': 1.1,
                'shadow_lift_bias': 0.9,
                'color_grading_bias': 'fuji_classic',  # Embrace Fuji's color character
                'wb_confidence_boost': 0.2   # Fuji has distinctive but good color science
            },
            'Fujifilm X-H2S': {
                'high_iso_threshold': 6400,
                'highlight_headroom': 1.0,
                'shadow_lift_bias': 0.8,
                'color_grading_bias': 'fuji_classic',
                'wb_confidence_boost': 0.2
            }
        }
        
        # Default profile for unknown cameras
        self.default_camera_profile = {
            'high_iso_threshold': 1600,
            'highlight_headroom': 1.0,
            'shadow_lift_bias': 1.0,
            'color_grading_bias': 'neutral',
            'wb_confidence_boost': 0.0
        }
        
        logger.info(f"Initialized {len(self.camera_profiles)} camera profiles")
    
    def _get_camera_profile(self, exif_data) -> Dict[str, Any]:
        """Get processing profile for camera based on EXIF data"""
        if exif_data and exif_data.camera_make and exif_data.camera_model:
            # Create camera key from EXIF data
            camera_key = f"{exif_data.camera_make} {exif_data.camera_model}"
            profile = self.camera_profiles.get(camera_key)
            if profile:
                return profile
        
        # Default to Sony A7 III profile (primary camera)
        return self.camera_profiles.get('Sony ILCE-7M3', self.default_camera_profile)
    
    def create_default_recipe(self, raw_path: Path) -> ProcessingRecipe:
        """
        Create a default processing recipe for a RAW file
        
        Args:
            raw_path: Path to RAW file
            
        Returns:
            Default ProcessingRecipe instance
        """
        # Calculate file hash for verification
        file_hash = self._calculate_file_hash(raw_path)
        
        # Extract EXIF data for smart processing
        logger.info(f"Extracting EXIF data from {raw_path.name}...")
        exif_data = self._extract_exif_data(raw_path)
        
        # Determine sharpening style based on image type
        sharpening_style = "standard"
        if exif_data:
            # Try to determine subject type from EXIF
            focal_length = exif_data.get('FOCAL_LENGTH', 0)
            if focal_length and focal_length > 85:  # Likely portrait lens
                sharpening_style = "portrait"
            elif focal_length and focal_length < 35:  # Likely landscape lens
                sharpening_style = "landscape"
        
        recipe = ProcessingRecipe(
            source_path=str(raw_path),
            file_hash=file_hash,
            created_at=datetime.now().isoformat(),
            exif_data=exif_data,
            sharpening=create_default_sharpening_settings(sharpening_style)
        )
        
        # Auto-analyze if enabled
        if self.auto_analyze:
            logger.info(f"Auto-analyzing {raw_path.name}...")
            recipe = self.analyze_and_optimize(raw_path, recipe)
        
        return recipe
    
    def analyze_and_optimize(self, raw_path: Path, recipe: ProcessingRecipe) -> ProcessingRecipe:
        """
        Analyze image and populate recipe with optimal settings
        
        Args:
            raw_path: Path to RAW file
            recipe: Base recipe to populate
            
        Returns:
            Updated recipe with analysis results
        """
        logger.info(f"Running comprehensive analysis on {raw_path.name}")
        
        # Load preview for analysis
        with rawpy.imread(str(raw_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,  # Faster analysis
                no_auto_bright=False,
                output_bps=16
            )
        
        # Convert to float for processing
        rgb_float = rgb.astype(np.float32) / 65535.0
        
        # 1. Scene Classification
        logger.info("  - Classifying scene...")
        scene_result = self.scene_classifier.classify_scene(rgb_float)
        recipe.scene_classification = {
            'classification': scene_result.classification,
            'confidence': scene_result.confidence,
            'processing_hints': {
                'preferred_wb_temp_range': scene_result.processing_hints.get('preferred_wb_temp_range'),
                'suggested_leveling_methods': scene_result.processing_hints.get('suggested_leveling_methods', []),
                'color_grading_hints': scene_result.processing_hints.get('color_grading_hints', {})
            }
        }
        
        # 2. Auto-straightening
        logger.info("  - Detecting horizon/verticals...")
        straighten_result = self.auto_straightener.detect_and_straighten(
            rgb_float, 
            scene_result.processing_hints
        )
        if straighten_result.should_straighten and abs(straighten_result.rotation_angle) > 0.1:
            recipe.rotation_angle = straighten_result.rotation_angle
            logger.info(f"    Auto-straighten: {straighten_result.rotation_angle:.2f}°")
        
        # 3. Smart Cropping
        logger.info("  - Analyzing composition...")
        crop_suggestions = self.smart_cropper.suggest_crops(rgb_float)
        if crop_suggestions:
            recipe.crop_suggestions = [
                {
                    'bbox': suggestion.bbox,
                    'aspect_ratio': suggestion.aspect_ratio,
                    'confidence': suggestion.confidence,
                    'composition_score': suggestion.composition_score,
                    'subject_coverage': suggestion.subject_coverage
                }
                for suggestion in crop_suggestions[:5]  # Top 5 suggestions
            ]
            logger.info(f"    Found {len(crop_suggestions)} crop suggestions")
        
        # 4. Exposure Optimization
        logger.info("  - Optimizing exposure...")
        exposure_analysis = self.exposure_optimizer.analyze_exposure(rgb_float)
        recipe.exposure_analysis = {
            'zone_distribution': exposure_analysis.zone_distribution,
            'clipping_analysis': {
                'shadow_clipping': exposure_analysis.shadow_clipping,
                'highlight_clipping': exposure_analysis.highlight_clipping
            },
            'scene_type': exposure_analysis.scene_type,
            'dynamic_range_score': exposure_analysis.dynamic_range_score
        }
        
        # Apply suggested adjustments with camera profile modifications
        recipe.exposure_adjustment = exposure_analysis.exposure_compensation
        
        # Get camera profile early for exposure adjustments
        camera_profile = self._get_camera_profile(recipe.exif_data)
        
        # Adjust shadow/highlight recovery based on camera capabilities
        recipe.shadows = exposure_analysis.shadow_adjustment + camera_profile.shadow_lift_bias
        recipe.highlights = exposure_analysis.highlight_adjustment - camera_profile.highlight_headroom
        logger.info(f"    Exposure: {exposure_analysis.exposure_compensation:+.2f}EV, "
                   f"Shadows: {exposure_analysis.shadow_adjustment:+.0f}, "
                   f"Highlights: {exposure_analysis.highlight_adjustment:+.0f}")
        
        # 5. White Balance Analysis (Enhanced with EXIF data and camera profiles)
        logger.info("  - Analyzing white balance...")
        
        # Use EXIF white balance as hint for better algorithm selection
        exif_wb_hint = None
        preferred_method = None
        
        if recipe.exif_data:
            if recipe.exif_data.white_balance == "Auto":
                # Camera used auto WB, trust it more but still analyze
                preferred_method = WhiteBalanceMethod.ILLUMINANT_ESTIMATION
                exif_wb_hint = recipe.exif_data.color_temp_kelvin
                logger.info(f"    Camera WB: Auto (estimated {exif_wb_hint}K)")
            elif recipe.exif_data.white_balance == "Manual":
                # Camera used manual WB, analyze to see if it's good
                preferred_method = WhiteBalanceMethod.GRAY_WORLD
                logger.info("    Camera WB: Manual - analyzing scene")
            
            # Log camera and lens for context
            logger.info(f"    Camera: {recipe.exif_data.camera_make} {recipe.exif_data.camera_model}")
            logger.info(f"    Lens: {recipe.exif_data.lens_model}")
            logger.info(f"    Settings: ISO {recipe.exif_data.iso}, "
                       f"{recipe.exif_data.focal_length}mm, f/{recipe.exif_data.aperture}")
        
        # Analyze with method preference if available
        wb_analysis = self.wb_corrector.analyze_white_balance(
            rgb_float, 
            method=preferred_method
        )
        
        # Apply camera profile confidence boost
        wb_analysis.confidence = min(1.0, wb_analysis.confidence * camera_profile.wb_confidence_boost)
        
        # Enhance analysis with EXIF context
        recipe.wb_analysis = {
            'estimated_temp_kelvin': wb_analysis.estimated_temp_kelvin,
            'method_used': wb_analysis.method_used.value,
            'confidence': wb_analysis.confidence,
            'illuminant_type': wb_analysis.illuminant_type,
            'color_cast': {
                'strength': wb_analysis.color_cast_strength,
                'dominant': wb_analysis.dominant_cast
            },
            'exif_wb_mode': recipe.exif_data.white_balance if recipe.exif_data else None,
            'exif_temp_hint': exif_wb_hint
        }
        recipe.wb_multipliers = (
            wb_analysis.red_multiplier,
            wb_analysis.green_multiplier,
            wb_analysis.blue_multiplier
        )
        logger.info(f"    White balance: {wb_analysis.estimated_temp_kelvin:.0f}K "
                   f"({wb_analysis.method_used.value}, confidence: {wb_analysis.confidence:.2f})")
        
        # 6. Color Grading Suggestions (Enhanced with EXIF data and camera profiles)
        logger.info("  - Suggesting color grading...")
        
        # Choose preset based on scene, EXIF settings, shooting conditions, and camera profile
        suggested_preset = ColorGradingPreset.NONE  # Default
        
        # Apply camera profile color grading bias
        if camera_profile.color_grading_bias:
            try:
                suggested_preset = ColorGradingPreset(camera_profile.color_grading_bias)
                logger.info(f"    Camera profile bias: {camera_profile.color_grading_bias}")
            except ValueError:
                pass  # Invalid preset name, continue with analysis
        
        # Consider ISO threshold from camera profile
        if recipe.exif_data and recipe.exif_data.iso >= camera_profile.high_iso_threshold:
            # High ISO shots often benefit from vintage/film looks to embrace grain
            suggested_preset = ColorGradingPreset.FILM_EMULATION
            logger.info(f"    High ISO ({recipe.exif_data.iso} >= {camera_profile.high_iso_threshold}) - suggesting film emulation")
        
        # Consider scene type and shooting conditions
        elif scene_result.classification == 'outdoor':
            if 'landscape' in scene_result.processing_hints.get('scene_features', []):
                # Check time of day from EXIF if available
                if recipe.exif_data and recipe.exif_data.datetime_original:
                    # Simple time extraction (could be enhanced)
                    time_str = recipe.exif_data.datetime_original.split(' ')[-1] if ' ' in recipe.exif_data.datetime_original else ""
                    if time_str:
                        hour = int(time_str.split(':')[0]) if ':' in time_str else 12
                        if hour < 8 or hour > 18:  # Golden hour / blue hour
                            suggested_preset = ColorGradingPreset.CINEMATIC
                            logger.info("    Golden/blue hour landscape - suggesting cinematic")
                        else:
                            suggested_preset = ColorGradingPreset.BRIGHT_AIRY
                            logger.info("    Daytime landscape - suggesting bright & airy")
                    else:
                        suggested_preset = ColorGradingPreset.BRIGHT_AIRY
                else:
                    suggested_preset = ColorGradingPreset.BRIGHT_AIRY
            else:
                # General outdoor
                suggested_preset = ColorGradingPreset.CINEMATIC
        else:  # indoor
            # Consider flash usage for indoor shots
            if recipe.exif_data and recipe.exif_data.flash_fired:
                suggested_preset = ColorGradingPreset.NONE  # Flash shots often need minimal grading
                logger.info("    Indoor with flash - suggesting none")
            else:
                suggested_preset = ColorGradingPreset.VINTAGE
                logger.info("    Indoor ambient - suggesting vintage")
        
        # Consider lens type for specific looks
        if recipe.exif_data and recipe.exif_data.lens_model:
            lens_lower = recipe.exif_data.lens_model.lower()
            if 'macro' in lens_lower:
                suggested_preset = ColorGradingPreset.BRIGHT_AIRY  # Macro often benefits from clean, bright look
                logger.info("    Macro lens detected - suggesting bright & airy")
            elif 'portrait' in lens_lower or (recipe.exif_data.focal_length >= 85):
                suggested_preset = ColorGradingPreset.CINEMATIC  # Portrait lenses often for cinematic work
                logger.info("    Portrait lens/focal length - suggesting cinematic")
        
        recipe.color_grading_preset = suggested_preset.value
        logger.info(f"    Final suggested preset: {suggested_preset.value}")
        
        logger.info(f"Analysis complete for {raw_path.name}")
        return recipe
    
    def generate_preview_with_yolo(self, raw_path: Path, recipe: ProcessingRecipe,
                                  size: Optional[int] = None, config: Optional[Dict] = None) -> Tuple[np.ndarray, Optional[YOLOAnalysis]]:
        """
        Generate a preview image with optional YOLO analysis
        
        Args:
            raw_path: Path to RAW file
            recipe: Processing recipe to apply
            size: Override preview size
            config: PhotoSight configuration dict
            
        Returns:
            Tuple of (RGB numpy array of processed preview, YOLOAnalysis or None)
        """
        # Generate the preview first
        preview = self.generate_preview(raw_path, recipe, size)
        
        # Run YOLO analysis if enabled in config
        yolo_analysis = None
        if config and config.get('raw_processing', {}).get('enable_early_yolo', False):
            try:
                yolo_analysis = analyze_preview_yolo(preview, config)
                logger.info(f"YOLO analysis for {raw_path.name}: {yolo_analysis.object_score:.2f} score, "
                           f"{len(yolo_analysis.detections)} objects, {yolo_analysis.processing_time_ms:.1f}ms")
            except Exception as e:
                logger.warning(f"YOLO analysis failed for {raw_path.name}: {e}")
        
        return preview, yolo_analysis

    def generate_preview(self, raw_path: Path, recipe: ProcessingRecipe,
                        size: Optional[int] = None) -> np.ndarray:
        """
        Generate a preview image with processing applied
        
        Args:
            raw_path: Path to RAW file
            recipe: Processing recipe to apply
            size: Override preview size
            
        Returns:
            RGB numpy array of processed preview
        """
        size = size or self.preview_size
        
        # Check cache
        cache_key = f"{raw_path}_{recipe.file_hash}_{hash(recipe.to_json())}"
        if self.cache_previews and cache_key in self._preview_cache:
            return self._preview_cache[cache_key]
        
        # Load and process RAW
        with rawpy.imread(str(raw_path)) as raw:
            # Apply basic processing
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,  # Faster for preview
                no_auto_bright=False,
                output_bps=16  # 16-bit for better processing headroom
            )
        
        # Convert to float for processing
        rgb = rgb.astype(np.float32) / 65535.0
        
        # Apply comprehensive processing pipeline
        rgb = self._apply_processing_pipeline(rgb, recipe)
        
        # Apply output sharpening before final operations
        if recipe.sharpening and recipe.sharpening.output_sharpening.enabled:
            output_size = rgb.shape[:2]  # (height, width)
            sharpening_pipeline = SharpeningPipeline(recipe.sharpening)
            rgb = sharpening_pipeline.apply_output_sharpening(rgb, output_size)
        
        # Apply rotation if needed
        if abs(recipe.rotation_angle) > 0.1:
            rgb = self._rotate_image(rgb, recipe.rotation_angle)
        
        # Apply crop if specified
        if recipe.crop_x is not None:
            rgb = rgb[
                recipe.crop_y:recipe.crop_y + recipe.crop_height,
                recipe.crop_x:recipe.crop_x + recipe.crop_width
            ]
        
        # Resize for preview
        h, w = rgb.shape[:2]
        if max(h, w) > size:
            scale = size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert back to 8-bit
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        # Cache if enabled
        if self.cache_previews:
            self._preview_cache[cache_key] = rgb
        
        return rgb
    
    def export_full_size(self, raw_path: Path, recipe: ProcessingRecipe,
                        output_path: Path, output_format: str = 'jpeg',
                        quality: int = 95) -> None:
        """
        Export full-resolution processed image
        
        Args:
            raw_path: Path to RAW file
            recipe: Processing recipe to apply
            output_path: Output file path
            output_format: Output format ('jpeg', 'png', 'tiff')
            quality: Output quality for JPEG (1-100)
        """
        logger.info(f"Exporting {raw_path.name} with recipe to {output_path}")
        
        # Load RAW at full resolution
        with rawpy.imread(str(raw_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=16
            )
        
        # Convert to float for processing
        rgb = rgb.astype(np.float32) / 65535.0
        
        # Apply comprehensive processing pipeline
        rgb = self._apply_processing_pipeline(rgb, recipe)
        
        # Apply rotation if needed
        if abs(recipe.rotation_angle) > 0.1:
            rgb = self._rotate_image(rgb, recipe.rotation_angle)
        
        # Apply crop if specified
        if recipe.crop_x is not None:
            rgb = rgb[
                recipe.crop_y:recipe.crop_y + recipe.crop_height,
                recipe.crop_x:recipe.crop_x + recipe.crop_width
            ]
        
        # Apply output sharpening for full resolution export
        if recipe.sharpening and recipe.sharpening.output_sharpening.enabled:
            output_size = rgb.shape[:2]  # (height, width)
            sharpening_pipeline = SharpeningPipeline(recipe.sharpening)
            rgb = sharpening_pipeline.apply_output_sharpening(rgb, output_size)
        
        # Convert to appropriate format and save
        if output_format.lower() in ['jpeg', 'jpg']:
            # Convert to 8-bit BGR for OpenCV
            bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            # PNG supports 16-bit
            bgr = cv2.cvtColor((rgb * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr)
        elif output_format.lower() in ['tiff', 'tif']:
            # TIFF with 16-bit support
            bgr = cv2.cvtColor((rgb * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Save recipe alongside
        recipe_path = output_path.with_suffix('.recipe.json')
        recipe.save(recipe_path)
        
        logger.info(f"Exported to {output_path} and saved recipe to {recipe_path}")
    
    def _apply_processing_pipeline(self, rgb: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        """
        Apply comprehensive processing pipeline to image
        
        This method orchestrates the complete PhotoSight processing pipeline:
        1. White Balance Correction
        2. Exposure Optimization  
        3. Color Grading
        4. Tone Adjustments
        5. Local Adjustments (clarity, texture, vignette)
        6. Layer-Based Local Adjustments
        7. Noise Reduction
        
        Args:
            rgb: Float32 RGB image (0-1 range)
            recipe: Processing recipe with all parameters
            
        Returns:
            Processed RGB image
        """
        logger.debug("Applying comprehensive processing pipeline...")
        
        # 1. INPUT SHARPENING (Early in pipeline to restore lost detail)
        if recipe.sharpening and recipe.sharpening.input_sharpening.enabled:
            logger.debug("  - Applying input sharpening (deconvolution)...")
            sharpening_pipeline = SharpeningPipeline(recipe.sharpening)
            rgb = sharpening_pipeline.apply_input_sharpening(rgb)
        
        # 2. WHITE BALANCE CORRECTION
        if recipe.wb_multipliers:
            logger.debug("  - Applying white balance correction...")
            
            # Convert to uint8 for white balance module
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            
            # Create WhiteBalanceAnalysis from recipe
            wb_analysis = WhiteBalanceAnalysis(
                estimated_temp_kelvin=recipe.wb_analysis.get('estimated_temp_kelvin', 5500) 
                                    if recipe.wb_analysis else 5500,
                confidence=recipe.wb_analysis.get('confidence', 1.0) if recipe.wb_analysis else 1.0,
                red_multiplier=recipe.wb_multipliers[0],
                green_multiplier=recipe.wb_multipliers[1], 
                blue_multiplier=recipe.wb_multipliers[2],
                color_cast_strength=recipe.wb_analysis.get('color_cast', {}).get('strength', 0.0) 
                                   if recipe.wb_analysis else 0.0,
                dominant_cast=recipe.wb_analysis.get('color_cast', {}).get('dominant', 'none')
                             if recipe.wb_analysis else 'none',
                method_used=WhiteBalanceMethod(recipe.wb_analysis.get('method_used', 'auto'))
                           if recipe.wb_analysis else WhiteBalanceMethod.AUTO,
                illuminant_type=recipe.wb_analysis.get('illuminant_type', 'unknown')
                               if recipe.wb_analysis else 'unknown',
                temperature_adjustment=recipe.temperature_adjustment,
                tint_adjustment=recipe.tint_adjustment,
                gray_deviation=0.0,
                highlight_preservation=1.0
            )
            
            # Apply white balance correction
            rgb_corrected = self.wb_corrector.correct_white_balance(
                rgb_uint8,
                wb_analysis,
                recipe.temperature_adjustment,
                recipe.tint_adjustment
            )
            rgb = rgb_corrected.astype(np.float32) / 255.0
        
        # 2. EXPOSURE OPTIMIZATION
        logger.debug("  - Applying exposure optimization...")
        
        # Convert to uint8 for exposure optimizer
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        # Create ExposureAnalysis from recipe if available
        if recipe.exposure_analysis:
            zone_dist = recipe.exposure_analysis.get('zone_distribution', {})
            # Convert zone distribution dict to proper format if needed
            if isinstance(zone_dist, list):
                # Convert list to dict
                zone_dist_dict = {i: val for i, val in enumerate(zone_dist[:11])}
            else:
                zone_dist_dict = zone_dist
                
            exposure_analysis = ExposureAnalysis(
                mean_brightness=0.5,  # Default, will be recalculated
                median_brightness=0.5,  # Default
                histogram_peaks=[],  # Not needed for application
                shadow_clipping=recipe.exposure_analysis.get('clipping_analysis', {}).get('shadow_clipping', 0),
                highlight_clipping=recipe.exposure_analysis.get('clipping_analysis', {}).get('highlight_clipping', 0),
                dynamic_range=recipe.exposure_analysis.get('dynamic_range_score', 0.7),
                contrast_ratio=10.0,  # Default
                zone_distribution=zone_dist_dict,
                exposure_compensation=recipe.exposure_adjustment,
                shadow_lift=recipe.shadows,
                highlight_recovery=recipe.highlights,
                preserve_highlights=True,
                lift_shadows=recipe.shadows > 10,
                increase_contrast=recipe.contrast > 0,
                confidence=0.85,
                quality_score=recipe.exposure_analysis.get('dynamic_range_score', 0.7)
            )
        else:
            # Create basic exposure analysis from recipe parameters
            exposure_analysis = ExposureAnalysis(
                mean_brightness=0.5,
                median_brightness=0.5,
                histogram_peaks=[],
                shadow_clipping=0.0,
                highlight_clipping=0.0,
                dynamic_range=0.7,
                contrast_ratio=10.0,
                zone_distribution={i: 0.091 for i in range(11)},  # Even distribution
                exposure_compensation=recipe.exposure_adjustment,
                shadow_lift=recipe.shadows,
                highlight_recovery=recipe.highlights,
                preserve_highlights=True,
                lift_shadows=recipe.shadows > 10,
                increase_contrast=recipe.contrast > 0,
                confidence=0.85,
                quality_score=0.7
            )
        
        # Apply exposure optimization
        rgb_optimized = self.exposure_optimizer.optimize_exposure(rgb_uint8, exposure_analysis)
        rgb = rgb_optimized.astype(np.float32) / 255.0
        
        # 3. COLOR GRADING
        if recipe.color_grading_preset or recipe.color_grading_settings:
            logger.debug("  - Applying color grading...")
            
            # Create color grading settings
            if recipe.color_grading_settings:
                # Use custom settings from recipe
                settings = ColorGradingSettings(
                    highlights_hue_shift=recipe.color_grading_settings.get('highlights_hue_shift', 0),
                    highlights_saturation=recipe.color_grading_settings.get('highlights_saturation', 0),
                    highlights_luminance=recipe.color_grading_settings.get('highlights_luminance', 0),
                    midtones_hue_shift=recipe.color_grading_settings.get('midtones_hue_shift', 0),
                    midtones_saturation=recipe.color_grading_settings.get('midtones_saturation', 0),
                    midtones_luminance=recipe.color_grading_settings.get('midtones_luminance', 0),
                    shadows_hue_shift=recipe.color_grading_settings.get('shadows_hue_shift', 0),
                    shadows_saturation=recipe.color_grading_settings.get('shadows_saturation', 0),
                    shadows_luminance=recipe.color_grading_settings.get('shadows_luminance', 0),
                    global_hue_shift=recipe.color_grading_settings.get('global_hue_shift', 0),
                    global_saturation=recipe.color_grading_settings.get('global_saturation', 0),
                    global_luminance=recipe.color_grading_settings.get('global_luminance', 0),
                    split_toning_highlights=recipe.color_grading_settings.get('split_toning_highlights', (0, 0, 0)),
                    split_toning_shadows=recipe.color_grading_settings.get('split_toning_shadows', (0, 0, 0)),
                    split_toning_balance=recipe.color_grading_settings.get('split_toning_balance', 0),
                    vibrance=recipe.vibrance,
                    saturation=recipe.saturation
                )
            elif recipe.color_grading_preset:
                # Use preset
                try:
                    preset = ColorGradingPreset(recipe.color_grading_preset)
                    settings = self.color_grader.get_preset(preset)
                except ValueError:
                    logger.warning(f"Unknown color grading preset: {recipe.color_grading_preset}")
                    settings = self.color_grader.get_preset(ColorGradingPreset.NEUTRAL)
                
                # Apply any manual vibrance/saturation overrides
                if recipe.vibrance != 0:
                    settings.vibrance = recipe.vibrance
                if recipe.saturation != 0:
                    settings.saturation = recipe.saturation
            else:
                # Default neutral settings
                settings = self.color_grader.get_preset(ColorGradingPreset.NEUTRAL)
            
            # Apply color grading
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            rgb_graded = self.color_grader.apply_color_grading(rgb_uint8, settings)
            rgb = rgb_graded.astype(np.float32) / 255.0
        else:
            # Apply basic vibrance/saturation if no color grading
            if recipe.vibrance != 0:
                rgb = self._adjust_vibrance(rgb, recipe.vibrance)
            if recipe.saturation != 0:
                rgb = self._adjust_saturation(rgb, recipe.saturation)
        
        # 4. ADDITIONAL TONE ADJUSTMENTS
        # These are fine-tuning adjustments that can be applied after the main modules
        logger.debug("  - Applying fine-tuning adjustments...")
        
        # Additional contrast adjustment (if needed beyond exposure optimizer)
        if recipe.contrast != 0 and not recipe.exposure_analysis:
            # Only apply if not already handled by exposure optimizer
            rgb = self._adjust_contrast(rgb, recipe.contrast)
        
        # 5. LOCAL ADJUSTMENTS
        logger.debug("  - Applying local adjustments...")
        
        # Clarity (local contrast)
        if recipe.clarity != 0:
            rgb = self._adjust_clarity(rgb, recipe.clarity)
        
        # Texture enhancement
        if recipe.texture != 0:
            rgb = self._adjust_texture(rgb, recipe.texture)
        
        # Vignette correction/application
        if recipe.vignette_amount != 0:
            rgb = self._apply_vignette(rgb, recipe.vignette_amount)
        
        # 6. LAYER-BASED LOCAL ADJUSTMENTS
        if recipe.adjustment_layers:
            logger.debug(f"  - Applying {len(recipe.adjustment_layers)} adjustment layers...")
            from photosight.processing.local_adjustments import LocalAdjustmentProcessor
            
            # Initialize processor (with caching enabled by default)
            local_processor = LocalAdjustmentProcessor()
            
            # Apply all adjustment layers
            rgb = local_processor.apply_adjustment_layers(rgb, recipe.adjustment_layers)
        
        # 7. CREATIVE SHARPENING (After local adjustments)
        if recipe.sharpening and recipe.sharpening.creative_sharpening.enabled:
            logger.debug("  - Applying creative sharpening (unsharp mask + clarity)...")
            sharpening_pipeline = SharpeningPipeline(recipe.sharpening)
            rgb = sharpening_pipeline.apply_creative_sharpening(rgb)
        
        # 8. NOISE REDUCTION
        if recipe.noise_reduction:
            logger.debug("  - Applying noise reduction...")
            from photosight.processing.noise import NoiseReducer
            
            # Initialize noise reducer
            noise_reducer = NoiseReducer()
            
            # Apply noise reduction with ISO and camera info if available
            iso = recipe.exif_data.iso if recipe.exif_data else None
            camera_model = recipe.exif_data.camera_model if recipe.exif_data else None
            
            rgb = noise_reducer.reduce_noise(
                rgb, 
                recipe.noise_reduction,
                iso=iso,
                camera_model=camera_model
            )
        
        # Final clipping to ensure valid range
        rgb = np.clip(rgb, 0, 1)
        
        logger.debug("Processing pipeline complete")
        return rgb
    
    def _apply_advanced_shadow_highlight(self, img: np.ndarray, shadows: float, 
                                        highlights: float) -> np.ndarray:
        """Advanced shadow/highlight recovery using luminosity masks"""
        if shadows == 0 and highlights == 0:
            return img
            
        # Create luminosity masks for targeted adjustments
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        
        # Shadow mask (targets dark areas)
        if shadows > 0:
            shadow_mask = np.maximum(0, 1 - luminance * 2)
            shadow_mask = shadow_mask ** 2  # Smooth falloff
            shadow_mask = shadow_mask[:, :, np.newaxis]
            
            # Lift shadows with natural falloff
            shadow_strength = shadows / 100.0 * 0.5  # Scale down for natural look
            img = img + img * shadow_mask * shadow_strength
        
        # Highlight mask (targets bright areas)
        if highlights < 0:
            highlight_mask = np.maximum(0, luminance * 2 - 1)
            highlight_mask = highlight_mask ** 2  # Smooth falloff
            highlight_mask = highlight_mask[:, :, np.newaxis]
            
            # Recover highlights
            highlight_strength = abs(highlights) / 100.0 * 0.5
            img = img - img * highlight_mask * highlight_strength
        
        return np.clip(img, 0, 1)
    
    def _adjust_blacks_whites(self, img: np.ndarray, blacks: float, whites: float) -> np.ndarray:
        """Adjust black and white points"""
        if blacks == 0 and whites == 0:
            return img
            
        # Create tone curve adjustment
        # Blacks adjustment affects the lower part of the curve
        if blacks != 0:
            black_factor = blacks / 100.0 * 0.3  # Scale for natural adjustment
            img = img + black_factor * (1 - img)  # Lift or lower blacks
        
        # Whites adjustment affects the upper part of the curve  
        if whites != 0:
            white_factor = whites / 100.0 * 0.3
            img = img * (1 + white_factor)  # Boost or reduce whites
            
        return np.clip(img, 0, 1)
    
    def _adjust_texture(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust fine texture details"""
        if amount == 0:
            return img
            
        # Create high-frequency enhancement
        # Use smaller kernel for fine details vs clarity
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=0.5)
        high_freq = img - blurred
        
        # Apply texture enhancement
        factor = amount / 100.0 * 0.3  # Subtle effect
        result = img + high_freq * factor
        
        return np.clip(result, 0, 1)
    
    def _adjust_contrast(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust contrast"""
        # S-curve contrast adjustment
        factor = (amount + 100) / 100
        img = (img - 0.5) * factor + 0.5
        return np.clip(img, 0, 1)
    
    def _adjust_vibrance(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust vibrance (smart saturation)"""
        # Convert to HSV
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1 + (amount / 100)  # Adjust saturation channel
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to RGB
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255
    
    def _adjust_saturation(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust saturation"""
        # Simple saturation adjustment
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        factor = (amount + 100) / 100
        img = gray + (img - gray) * factor
        return np.clip(img, 0, 1)
    
    def _adjust_clarity(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust clarity (local contrast)"""
        if amount == 0:
            return img
        
        # Create blurred version
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        
        # High-pass filter
        high_pass = img - blurred
        
        # Add back with strength
        factor = amount / 100
        result = img + high_pass * factor
        
        return np.clip(result, 0, 1)
    
    def _apply_vignette(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Apply or correct vignetting"""
        h, w = img.shape[:2]
        
        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize and create vignette mask
        vignette = 1 - (dist_from_center / max_dist) ** 2
        vignette = vignette[:, :, np.newaxis]
        
        # Apply vignette
        factor = abs(amount) / 100
        if amount < 0:
            # Darken edges
            img = img * (1 - factor * (1 - vignette))
        else:
            # Brighten edges (vignette correction)
            img = img / (1 - factor * (1 - vignette))
        
        return np.clip(img, 0, 1)
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for new dimensions
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # Rotate
        rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def _extract_exif_data(self, raw_path: Path) -> EXIFData:
        """
        Extract comprehensive EXIF data from RAW file
        
        Args:
            raw_path: Path to RAW file
            
        Returns:
            EXIFData object with extracted metadata
        """
        exif_data = EXIFData()
        
        try:
            with open(raw_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            
            # Camera identification
            if 'Image Make' in tags:
                exif_data.camera_make = str(tags['Image Make']).strip()
            if 'Image Model' in tags:
                exif_data.camera_model = str(tags['Image Model']).strip()
            if 'EXIF LensModel' in tags:
                exif_data.lens_model = str(tags['EXIF LensModel']).strip()
            
            # Exposure settings
            if 'EXIF ISOSpeedRatings' in tags:
                exif_data.iso = int(str(tags['EXIF ISOSpeedRatings']))
            
            if 'EXIF FocalLength' in tags:
                focal_str = str(tags['EXIF FocalLength'])
                # Handle fractional values like "50/1"
                if '/' in focal_str:
                    num, denom = focal_str.split('/')
                    exif_data.focal_length = float(num) / float(denom)
                else:
                    exif_data.focal_length = float(focal_str)
            
            if 'EXIF FNumber' in tags:
                f_str = str(tags['EXIF FNumber'])
                if '/' in f_str:
                    num, denom = f_str.split('/')
                    exif_data.aperture = float(num) / float(denom)
                else:
                    exif_data.aperture = float(f_str)
            
            if 'EXIF ExposureTime' in tags:
                exif_data.shutter_speed = str(tags['EXIF ExposureTime'])
            
            if 'EXIF ExposureBiasValue' in tags:
                exp_bias_str = str(tags['EXIF ExposureBiasValue'])
                if '/' in exp_bias_str:
                    num, denom = exp_bias_str.split('/')
                    exif_data.exposure_compensation = float(num) / float(denom)
                else:
                    exif_data.exposure_compensation = float(exp_bias_str)
            
            # White balance and color
            if 'EXIF WhiteBalance' in tags:
                wb_mode = str(tags['EXIF WhiteBalance'])
                exif_data.white_balance = "Auto" if wb_mode == "0" else "Manual"
            
            if 'EXIF ColorSpace' in tags:
                color_space = str(tags['EXIF ColorSpace'])
                exif_data.color_space = "sRGB" if color_space == "1" else "Adobe RGB"
            
            # Metering and focus
            if 'EXIF MeteringMode' in tags:
                metering_modes = {
                    '0': 'Unknown', '1': 'Average', '2': 'Center-weighted',
                    '3': 'Spot', '4': 'Multi-spot', '5': 'Pattern', '6': 'Partial'
                }
                exif_data.metering_mode = metering_modes.get(
                    str(tags['EXIF MeteringMode']), 'Unknown'
                )
            
            if 'EXIF Flash' in tags:
                flash_value = int(str(tags['EXIF Flash']))
                exif_data.flash_fired = bool(flash_value & 1)  # Check first bit
            
            # Image dimensions
            if 'EXIF ExifImageWidth' in tags:
                exif_data.image_width = int(str(tags['EXIF ExifImageWidth']))
            elif 'Image ImageWidth' in tags:
                exif_data.image_width = int(str(tags['Image ImageWidth']))
            
            if 'EXIF ExifImageLength' in tags:
                exif_data.image_height = int(str(tags['EXIF ExifImageLength']))
            elif 'Image ImageLength' in tags:
                exif_data.image_height = int(str(tags['Image ImageLength']))
            
            if 'Image Orientation' in tags:
                exif_data.orientation = int(str(tags['Image Orientation']))
            
            # GPS data
            if 'GPS GPSLatitude' in tags and 'GPS GPSLatitudeRef' in tags:
                lat_ref = str(tags['GPS GPSLatitudeRef'])
                lat_parts = str(tags['GPS GPSLatitude']).replace('[', '').replace(']', '').split(', ')
                if len(lat_parts) >= 3:
                    # Convert DMS to decimal degrees
                    degrees = self._parse_gps_coord(lat_parts[0])
                    minutes = self._parse_gps_coord(lat_parts[1])
                    seconds = self._parse_gps_coord(lat_parts[2])
                    lat = degrees + minutes/60 + seconds/3600
                    if lat_ref == 'S':
                        lat = -lat
                    exif_data.gps_latitude = lat
            
            if 'GPS GPSLongitude' in tags and 'GPS GPSLongitudeRef' in tags:
                lon_ref = str(tags['GPS GPSLongitudeRef'])
                lon_parts = str(tags['GPS GPSLongitude']).replace('[', '').replace(']', '').split(', ')
                if len(lon_parts) >= 3:
                    degrees = self._parse_gps_coord(lon_parts[0])
                    minutes = self._parse_gps_coord(lon_parts[1])
                    seconds = self._parse_gps_coord(lon_parts[2])
                    lon = degrees + minutes/60 + seconds/3600
                    if lon_ref == 'W':
                        lon = -lon
                    exif_data.gps_longitude = lon
            
            # Timestamps
            if 'EXIF DateTimeOriginal' in tags:
                exif_data.datetime_original = str(tags['EXIF DateTimeOriginal'])
            
            if 'EXIF DateTimeDigitized' in tags:
                exif_data.datetime_digitized = str(tags['EXIF DateTimeDigitized'])
            
            # Camera settings
            if 'EXIF ExposureMode' in tags:
                exposure_modes = {'0': 'Auto', '1': 'Manual', '2': 'Auto bracket'}
                exif_data.exposure_mode = exposure_modes.get(
                    str(tags['EXIF ExposureMode']), 'Unknown'
                )
            
            if 'EXIF SceneCaptureType' in tags:
                scene_modes = {
                    '0': 'Standard', '1': 'Landscape', '2': 'Portrait', '3': 'Night'
                }
                exif_data.scene_mode = scene_modes.get(
                    str(tags['EXIF SceneCaptureType']), 'Unknown'
                )
            
            # Software
            if 'Image Software' in tags:
                exif_data.software = str(tags['Image Software']).strip()
            
            logger.debug(f"Extracted EXIF data: {exif_data.camera_make} {exif_data.camera_model}, "
                        f"ISO {exif_data.iso}, {exif_data.focal_length}mm, f/{exif_data.aperture}")
            
        except Exception as e:
            logger.warning(f"Failed to extract EXIF data from {raw_path}: {e}")
            # Return default EXIFData with at least the filename info
            exif_data.camera_model = f"Unknown ({raw_path.name})"
        
        return exif_data
    
    def _parse_gps_coord(self, coord_str: str) -> float:
        """Parse GPS coordinate from EXIF string format"""
        if '/' in coord_str:
            num, denom = coord_str.split('/')
            return float(num) / float(denom)
        return float(coord_str)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]  # Use first 16 chars
    
    def clear_preview_cache(self) -> None:
        """Clear preview cache"""
        self._preview_cache.clear()
        logger.info("Preview cache cleared")
    
    # Database integration methods
    def save_recipe_to_db(self, recipe: ProcessingRecipe, name: str,
                         description: Optional[str] = None,
                         created_by: Optional[str] = None) -> bool:
        """
        Save a recipe to the database.
        
        Args:
            recipe: ProcessingRecipe to save
            name: Unique name for the recipe
            description: Optional description
            created_by: Username/identifier of creator
            
        Returns:
            True if saved successfully
        """
        if not self.recipe_manager:
            logger.warning("Database not available for recipe storage")
            return False
        
        try:
            db_recipe = self.recipe_manager.save_recipe(
                recipe, name, description, created_by
            )
            logger.info(f"Recipe '{name}' saved to database (ID: {db_recipe.id})")
            return True
        except Exception as e:
            logger.error(f"Failed to save recipe to database: {e}")
            return False
    
    def load_recipe_from_db(self, recipe_name: str, 
                           source_path: Optional[Path] = None) -> Optional[ProcessingRecipe]:
        """
        Load a recipe from the database by name.
        
        Args:
            recipe_name: Name of the recipe to load
            source_path: Optional source path for the recipe
            
        Returns:
            ProcessingRecipe or None if not found
        """
        if not self.recipe_manager:
            logger.warning("Database not available for recipe loading")
            return None
        
        try:
            recipe = self.recipe_manager.load_recipe(
                recipe_name, str(source_path) if source_path else None
            )
            if recipe:
                logger.info(f"Recipe '{recipe_name}' loaded from database")
            return recipe
        except Exception as e:
            logger.error(f"Failed to load recipe from database: {e}")
            return None
    
    def list_available_recipes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List available recipes from the database.
        
        Args:
            limit: Maximum number of recipes to return
            
        Returns:
            List of recipe summaries
        """
        if not self.recipe_manager:
            return []
        
        try:
            return self.recipe_manager.list_recipes(limit)
        except Exception as e:
            logger.error(f"Failed to list recipes: {e}")
            return []
    
    def save_photo_recipe(self, photo_id: int, recipe: ProcessingRecipe,
                         is_applied: bool = False) -> bool:
        """
        Save a recipe associated with a specific photo.
        
        Args:
            photo_id: Database ID of the photo
            recipe: ProcessingRecipe to save
            is_applied: Whether this recipe has been applied
            
        Returns:
            True if saved successfully
        """
        if not self.recipe_manager:
            logger.warning("Database not available for photo recipe storage")
            return False
        
        try:
            self.recipe_manager.save_photo_recipe(photo_id, recipe, is_applied)
            logger.info(f"Recipe saved for photo ID {photo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save photo recipe: {e}")
            return False
    
    def load_photo_recipe(self, photo_id: int) -> Optional[ProcessingRecipe]:
        """
        Load the recipe associated with a specific photo.
        
        Args:
            photo_id: Database ID of the photo
            
        Returns:
            ProcessingRecipe or None if not found
        """
        if not self.recipe_manager:
            logger.warning("Database not available for photo recipe loading")
            return None
        
        try:
            recipe = self.recipe_manager.load_photo_recipe(photo_id)
            if recipe:
                logger.info(f"Recipe loaded for photo ID {photo_id}")
            return recipe
        except Exception as e:
            logger.error(f"Failed to load photo recipe: {e}")
            return None
    
    def find_similar_recipes(self, recipe: ProcessingRecipe, 
                           threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Find recipes with similar parameters.
        
        Args:
            recipe: Recipe to compare against
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar recipes
        """
        if not self.recipe_manager:
            return []
        
        try:
            return self.recipe_manager.find_similar_recipes(recipe, threshold)
        except Exception as e:
            logger.error(f"Failed to find similar recipes: {e}")
            return []


class ToneMapper:
    """
    Unified tone mapping system with parametric curves and Catmull-Rom spline interpolation
    
    Features:
    - Parametric tone curves (shadows/midtones/highlights)
    - Custom spline interpolation with Catmull-Rom curves
    - Zone system mapping (Ansel Adams zones)
    - Advanced highlight recovery and shadow lifting
    """
    
    def __init__(self):
        """Initialize tone mapper"""
        self.zone_boundaries = self._compute_zone_boundaries()
    
    def apply_tone_mapping(self, image: np.ndarray, 
                          settings: ToneMapperSettings) -> np.ndarray:
        """
        Apply tone mapping to image
        
        Args:
            image: Input image (RGB, 0-1 range)
            settings: Tone mapping settings
            
        Returns:
            Tone mapped image
        """
        # Convert to luminance for tone curve calculation
        luminance = self._rgb_to_luminance(image)
        
        # Apply exposure compensation first
        if settings.exposure_compensation != 0:
            exposure_factor = 2 ** settings.exposure_compensation
            image = image * exposure_factor
            luminance = luminance * exposure_factor
        
        # Generate tone curve based on settings
        if settings.tone_curve_type == "spline" and settings.curve_points:
            tone_curve = self._create_spline_curve(settings.curve_points)
        else:
            tone_curve = self._create_parametric_curve(settings)
        
        # Apply tone curve to luminance
        mapped_luminance = self._apply_curve(luminance, tone_curve)
        
        # Preserve color by scaling RGB channels
        result = self._preserve_color_tone_mapping(image, luminance, mapped_luminance)
        
        # Apply local contrast (clarity) if enabled
        if settings.clarity != 0:
            result = self._apply_clarity(result, settings.clarity)
        
        # Zone system mapping if enabled
        if settings.zone_mapping_enabled and settings.target_zone_distribution:
            result = self._apply_zone_mapping(result, settings.target_zone_distribution)
        
        return np.clip(result, 0, 1)
    
    def _rgb_to_luminance(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to luminance using Rec. 709 weights"""
        return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    
    def _create_parametric_curve(self, settings: ToneMapperSettings) -> np.ndarray:
        """
        Create parametric tone curve
        
        Uses a five-region parametric curve:
        - Blacks (0 to shadows_split/2)
        - Shadows (shadows_split/2 to shadows_split)  
        - Midtones (shadows_split to highlights_split)
        - Highlights (highlights_split to (1+highlights_split)/2)
        - Whites ((1+highlights_split)/2 to 1)
        """
        curve_size = 1024
        input_values = np.linspace(0, 1, curve_size)
        output_values = np.zeros(curve_size)
        
        # Define region boundaries
        blacks_end = settings.shadows_split / 2
        shadows_end = settings.shadows_split
        highlights_start = settings.highlights_split
        whites_start = (1 + settings.highlights_split) / 2
        
        for i, x in enumerate(input_values):
            if x <= blacks_end:
                # Blacks region
                t = x / blacks_end if blacks_end > 0 else 0
                adjustment = settings.blacks / 100.0
                output_values[i] = x * (1 + adjustment * (1 - t))
                
            elif x <= shadows_end:
                # Shadows region  
                t = (x - blacks_end) / (shadows_end - blacks_end) if shadows_end > blacks_end else 0
                adjustment = settings.shadows / 100.0
                base_curve = x
                shadow_lift = adjustment * t * (1 - t) * 4  # Bell curve for smooth transition
                output_values[i] = base_curve + shadow_lift
                
            elif x <= highlights_start:
                # Midtones region with contrast
                t = (x - shadows_end) / (highlights_start - shadows_end) if highlights_start > shadows_end else 0
                contrast_adjustment = settings.contrast / 100.0
                # S-curve for contrast
                s_curve = 0.5 + np.sin((t - 0.5) * np.pi) * 0.5 * contrast_adjustment
                output_values[i] = x + (s_curve - 0.5) * 0.2
                
            elif x <= whites_start:
                # Highlights region
                t = (x - highlights_start) / (whites_start - highlights_start) if whites_start > highlights_start else 0
                adjustment = settings.highlights / 100.0
                base_curve = x
                highlight_adjust = adjustment * t * (1 - t) * 4
                output_values[i] = base_curve + highlight_adjust
                
            else:
                # Whites region
                t = (x - whites_start) / (1 - whites_start) if whites_start < 1 else 0
                adjustment = settings.whites / 100.0
                output_values[i] = x * (1 + adjustment * t)
        
        return np.clip(output_values, 0, 1)
    
    def _create_spline_curve(self, control_points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Create tone curve using Catmull-Rom spline interpolation
        
        Args:
            control_points: List of (input, output) control points
            
        Returns:
            Interpolated curve as numpy array
        """
        curve_size = 1024
        input_values = np.linspace(0, 1, curve_size)
        
        # Ensure we have enough points for Catmull-Rom
        if len(control_points) < 2:
            return input_values  # Identity curve
        
        # Sort control points by input value
        control_points = sorted(control_points, key=lambda p: p[0])
        
        # Add phantom points for proper Catmull-Rom behavior
        if control_points[0][0] > 0:
            control_points.insert(0, (0, 0))
        if control_points[-1][0] < 1:
            control_points.append((1, 1))
        
        # Extract x and y coordinates
        xs = np.array([p[0] for p in control_points])
        ys = np.array([p[1] for p in control_points])
        
        # Catmull-Rom spline interpolation
        output_values = np.zeros(curve_size)
        
        for i, x in enumerate(input_values):
            # Find the segment
            if x <= xs[0]:
                output_values[i] = ys[0]
            elif x >= xs[-1]:
                output_values[i] = ys[-1]
            else:
                # Find surrounding points
                for j in range(len(xs) - 1):
                    if xs[j] <= x <= xs[j + 1]:
                        # Get four control points for Catmull-Rom
                        p0_idx = max(0, j - 1)
                        p1_idx = j
                        p2_idx = j + 1
                        p3_idx = min(len(xs) - 1, j + 2)
                        
                        x0, y0 = xs[p0_idx], ys[p0_idx]
                        x1, y1 = xs[p1_idx], ys[p1_idx] 
                        x2, y2 = xs[p2_idx], ys[p2_idx]
                        x3, y3 = xs[p3_idx], ys[p3_idx]
                        
                        # Parameterize by arc length for better results
                        t = (x - x1) / (x2 - x1) if x2 != x1 else 0
                        
                        # Catmull-Rom formula
                        output_values[i] = self._catmull_rom_interpolate(
                            y0, y1, y2, y3, t
                        )
                        break
        
        return np.clip(output_values, 0, 1)
    
    def _catmull_rom_interpolate(self, p0: float, p1: float, p2: float, p3: float, t: float) -> float:
        """Catmull-Rom spline interpolation between p1 and p2"""
        return (
            0.5 * (
                2 * p1 +
                (-p0 + p2) * t +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
            )
        )
    
    def _apply_curve(self, luminance: np.ndarray, curve: np.ndarray) -> np.ndarray:
        """Apply tone curve to luminance values"""
        # Map luminance values to curve indices
        indices = (luminance * (len(curve) - 1)).astype(np.int32)
        indices = np.clip(indices, 0, len(curve) - 1)
        
        return curve[indices]
    
    def _preserve_color_tone_mapping(self, original_rgb: np.ndarray, 
                                   original_lum: np.ndarray,
                                   mapped_lum: np.ndarray) -> np.ndarray:
        """
        Preserve color saturation during tone mapping
        
        Uses luminance-preserving color scaling
        """
        result = original_rgb.copy()
        
        # Avoid division by zero
        safe_original_lum = np.maximum(original_lum, 1e-8)
        ratio = mapped_lum / safe_original_lum
        
        # Scale each channel by the luminance ratio
        for c in range(3):
            result[:, :, c] *= ratio
        
        return result
    
    def _apply_clarity(self, image: np.ndarray, clarity_amount: float) -> np.ndarray:
        """Apply local contrast enhancement (clarity)"""
        # Convert to LAB for better clarity processing
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Create unsharp mask for local contrast
        gaussian_blur = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=10, sigmaY=10)
        unsharp_mask = l_channel - gaussian_blur
        
        # Apply clarity
        clarity_factor = clarity_amount / 100.0
        enhanced_l = l_channel + unsharp_mask * clarity_factor * 0.5
        
        # Update LAB and convert back
        lab[:, :, 0] = np.clip(enhanced_l, 0, 255).astype(np.uint8)
        rgb_result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return rgb_result.astype(np.float32) / 255.0
    
    def _compute_zone_boundaries(self) -> np.ndarray:
        """Compute Ansel Adams zone system boundaries (11 zones)"""
        # Zone 0: Pure black, Zone X: Pure white
        # Each zone represents 1 stop difference
        zones = np.zeros(11)
        for i in range(11):
            zones[i] = i / 10.0  # Linear spacing for now, could be logarithmic
        return zones
    
    def _apply_zone_mapping(self, image: np.ndarray, 
                          target_distribution: List[float]) -> np.ndarray:
        """Apply zone system tone mapping"""
        if len(target_distribution) != 11:
            logger.warning("Zone distribution must have 11 values, skipping zone mapping")
            return image
        
        # Convert to luminance
        luminance = self._rgb_to_luminance(image)
        
        # Map current zones to target zones
        mapped_lum = np.zeros_like(luminance)
        
        for i in range(10):  # 10 intervals between 11 zones
            zone_start = self.zone_boundaries[i]
            zone_end = self.zone_boundaries[i + 1]
            
            # Find pixels in this zone
            in_zone = (luminance >= zone_start) & (luminance < zone_end)
            
            if np.any(in_zone):
                # Map to target zone
                target_start = target_distribution[i]
                target_end = target_distribution[i + 1]
                
                # Linear interpolation within zone
                zone_progress = (luminance[in_zone] - zone_start) / (zone_end - zone_start)
                mapped_lum[in_zone] = target_start + zone_progress * (target_end - target_start)
        
        # Preserve color
        return self._preserve_color_tone_mapping(image, luminance, mapped_lum)


@dataclass
class NoiseReductionSettings:
    """Advanced noise reduction settings with multiple algorithms"""
    
    # Global settings
    noise_reduction_strength: float = 50.0  # 0-100, overall noise reduction amount
    preserve_details: bool = True           # Protect fine details during reduction
    
    # Luminance noise reduction
    luminance_amount: float = 50.0          # 0-100, luminance noise strength
    luminance_detail: float = 50.0          # 0-100, detail preservation in luminance
    luminance_contrast: float = 0.0         # -100 to +100, contrast in smooth areas
    
    # Chrominance (color) noise reduction  
    chrominance_amount: float = 50.0        # 0-100, color noise strength
    chrominance_detail: float = 50.0        # 0-100, color detail preservation
    
    # Advanced wavelet settings
    wavelet_levels: int = 5                 # Number of wavelet decomposition levels
    wavelet_type: str = "db8"              # Wavelet type: db4, db8, haar, bior4.4
    
    # Adaptive settings
    iso_adaptive: bool = True               # Adjust based on ISO
    edge_preservation: float = 50.0         # 0-100, edge protection strength
    
    # Sony A7 III specific
    sensor_pattern_suppression: bool = True  # Suppress sensor pattern noise
    a7m3_optimized: bool = True             # Use A7 III specific optimizations


class NoiseReducer:
    """
    Advanced noise reduction system with wavelet decomposition
    
    Features:
    - Multi-scale wavelet decomposition
    - Adaptive thresholding based on local statistics
    - Edge-preserving smoothing
    - Sony A7 III sensor-specific optimizations
    - ISO-adaptive processing
    """
    
    def __init__(self):
        """Initialize noise reducer"""
        try:
            import pywt
            self.pywt = pywt
            self.wavelets_available = True
            logger.info("PyWavelets available - using wavelet-based noise reduction")
        except ImportError:
            self.pywt = None
            self.wavelets_available = False
            logger.warning("PyWavelets not available - using bilateral filtering fallback")
            
        # Sony A7 III noise characteristics
        self.a7m3_noise_profile = {
            100: {'luminance_noise': 0.001, 'chroma_noise': 0.0005, 'pattern_noise': 0.0001},
            200: {'luminance_noise': 0.002, 'chroma_noise': 0.001, 'pattern_noise': 0.0002},
            400: {'luminance_noise': 0.003, 'chroma_noise': 0.002, 'pattern_noise': 0.0003},
            800: {'luminance_noise': 0.006, 'chroma_noise': 0.004, 'pattern_noise': 0.0005},
            1600: {'luminance_noise': 0.012, 'chroma_noise': 0.008, 'pattern_noise': 0.001},
            3200: {'luminance_noise': 0.025, 'chroma_noise': 0.016, 'pattern_noise': 0.002},
            6400: {'luminance_noise': 0.050, 'chroma_noise': 0.032, 'pattern_noise': 0.004},
            12800: {'luminance_noise': 0.100, 'chroma_noise': 0.064, 'pattern_noise': 0.008}
        }
    
    def reduce_noise(self, image: np.ndarray, settings: NoiseReductionSettings,
                    iso: int = None) -> np.ndarray:
        """
        Apply advanced noise reduction to image
        
        Args:
            image: Input image (RGB, 0-1 range)
            settings: Noise reduction settings
            iso: ISO value for adaptive processing
            
        Returns:
            Noise reduced image
        """
        if settings.iso_adaptive and iso:
            # Adjust settings based on ISO
            settings = self._adapt_settings_for_iso(settings, iso)
            logger.info(f"Adapted noise reduction for ISO {iso}")
        
        # Convert to appropriate color space for noise reduction
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab_float = lab.astype(np.float32) / 255.0
        
        # Separate luminance and chrominance channels
        luminance = lab_float[:, :, 0]
        a_channel = lab_float[:, :, 1] 
        b_channel = lab_float[:, :, 2]
        
        # Apply luminance noise reduction
        if settings.luminance_amount > 0:
            if self.wavelets_available:
                luminance_denoised = self._wavelet_denoise(
                    luminance, settings, channel_type='luminance'
                )
            else:
                luminance_denoised = self._bilateral_denoise(
                    luminance, settings.luminance_amount / 100.0
                )
        else:
            luminance_denoised = luminance
        
        # Apply chrominance noise reduction
        if settings.chrominance_amount > 0:
            if self.wavelets_available:
                a_denoised = self._wavelet_denoise(
                    a_channel, settings, channel_type='chrominance'
                )
                b_denoised = self._wavelet_denoise(
                    b_channel, settings, channel_type='chrominance'
                )
            else:
                a_denoised = self._bilateral_denoise(
                    a_channel, settings.chrominance_amount / 100.0
                )
                b_denoised = self._bilateral_denoise(
                    b_channel, settings.chrominance_amount / 100.0
                )
        else:
            a_denoised = a_channel
            b_denoised = b_channel
        
        # Reconstruct LAB image
        lab_denoised = np.stack([luminance_denoised, a_denoised, b_denoised], axis=2)
        lab_denoised = np.clip(lab_denoised * 255, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        rgb_denoised = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2RGB)
        result = rgb_denoised.astype(np.float32) / 255.0
        
        # Apply edge preservation if enabled
        if settings.preserve_details and settings.edge_preservation > 0:
            result = self._preserve_edges(image, result, settings.edge_preservation / 100.0)
        
        # Sony A7 III specific pattern noise suppression
        if settings.a7m3_optimized and settings.sensor_pattern_suppression:
            result = self._suppress_sensor_pattern(result, iso)
        
        return np.clip(result, 0, 1)
    
    def _wavelet_denoise(self, channel: np.ndarray, settings: NoiseReductionSettings,
                        channel_type: str = 'luminance') -> np.ndarray:
        """Apply wavelet-based denoising to a single channel"""
        if not self.wavelets_available:
            return channel
        
        # Get appropriate settings for channel type
        if channel_type == 'luminance':
            strength = settings.luminance_amount / 100.0
            detail_preservation = settings.luminance_detail / 100.0
        else:  # chrominance
            strength = settings.chrominance_amount / 100.0
            detail_preservation = settings.chrominance_detail / 100.0
        
        try:
            # Wavelet decomposition
            coeffs = self.pywt.wavedec2(
                channel, 
                settings.wavelet_type, 
                level=settings.wavelet_levels,
                mode='symmetric'
            )
            
            # Adaptive thresholding
            denoised_coeffs = []
            
            # Process approximation coefficients (lowest frequency)
            denoised_coeffs.append(coeffs[0])
            
            # Process detail coefficients (higher frequencies contain noise)
            for i, (cH, cV, cD) in enumerate(coeffs[1:]):
                # Calculate adaptive threshold based on level and settings
                level_factor = (i + 1) / len(coeffs[1:])  # Higher levels = more aggressive
                base_threshold = strength * 0.1 * (1 + level_factor)
                
                # Estimate noise standard deviation using robust MAD estimator
                sigma_H = np.median(np.abs(cH)) / 0.6745
                sigma_V = np.median(np.abs(cV)) / 0.6745  
                sigma_D = np.median(np.abs(cD)) / 0.6745
                
                # Adaptive soft thresholding
                threshold_H = base_threshold * sigma_H
                threshold_V = base_threshold * sigma_V
                threshold_D = base_threshold * sigma_D
                
                # Apply soft thresholding with detail preservation
                preservation_factor = detail_preservation
                
                cH_denoised = self._soft_threshold(cH, threshold_H * (1 - preservation_factor))
                cV_denoised = self._soft_threshold(cV, threshold_V * (1 - preservation_factor))
                cD_denoised = self._soft_threshold(cD, threshold_D * (1 - preservation_factor))
                
                denoised_coeffs.append((cH_denoised, cV_denoised, cD_denoised))
            
            # Wavelet reconstruction
            denoised = self.pywt.waverec2(denoised_coeffs, settings.wavelet_type, mode='symmetric')
            
            # Ensure same shape as input (wavelet reconstruction can change size slightly)
            if denoised.shape != channel.shape:
                denoised = cv2.resize(denoised, (channel.shape[1], channel.shape[0]))
            
            return np.clip(denoised, 0, 1)
            
        except Exception as e:
            logger.warning(f"Wavelet denoising failed: {e}, falling back to bilateral")
            return self._bilateral_denoise(channel, strength)
    
    def _soft_threshold(self, coeffs: np.ndarray, threshold: float) -> np.ndarray:
        """Apply soft thresholding to wavelet coefficients"""
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    
    def _bilateral_denoise(self, channel: np.ndarray, strength: float) -> np.ndarray:
        """Fallback bilateral filtering for noise reduction"""
        # Convert to uint8 for cv2.bilateralFilter
        channel_uint8 = (channel * 255).astype(np.uint8)
        
        # Bilateral filter parameters based on strength
        d = int(5 + strength * 5)  # Neighborhood diameter
        sigma_color = strength * 80  # Color similarity threshold
        sigma_space = strength * 80  # Spatial similarity threshold
        
        denoised = cv2.bilateralFilter(channel_uint8, d, sigma_color, sigma_space)
        return denoised.astype(np.float32) / 255.0
    
    def _preserve_edges(self, original: np.ndarray, denoised: np.ndarray,
                       preservation_strength: float) -> np.ndarray:
        """Preserve edges by blending original and denoised based on edge strength"""
        # Calculate edge mask using Sobel operator
        gray_orig = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize edge magnitude and create mask
        edge_magnitude = edge_magnitude / edge_magnitude.max()
        edge_mask = edge_magnitude ** (1 / preservation_strength)  # Stronger preservation = lower power
        edge_mask = edge_mask[:, :, np.newaxis]  # Add channel dimension
        
        # Blend original and denoised based on edge strength
        result = denoised * (1 - edge_mask) + original * edge_mask
        return result
    
    def _suppress_sensor_pattern(self, image: np.ndarray, iso: int) -> np.ndarray:
        """Suppress Sony A7 III specific sensor pattern noise"""
        if iso is None or iso <= 400:
            return image  # No pattern suppression needed at low ISOs
        
        # Sony A7 III tends to have slight banding patterns at high ISOs
        # Apply very gentle horizontal pattern suppression
        pattern_strength = min((iso - 400) / 6000.0, 0.3)  # Max 30% strength
        
        # Create subtle horizontal smoothing kernel
        kernel_size = 3
        kernel = np.ones((1, kernel_size)) / kernel_size
        kernel = kernel[:, :, np.newaxis]
        
        # Apply pattern suppression only to highly noisy areas
        # Convert to grayscale for noise detection
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Detect high-frequency noise areas
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_map = np.abs(gray.astype(np.float32) - blur.astype(np.float32))
        noise_threshold = np.percentile(noise_map, 75)  # Top 25% noisiest areas
        
        pattern_mask = (noise_map > noise_threshold).astype(np.float32)
        pattern_mask = cv2.GaussianBlur(pattern_mask, (5, 5), 0)  # Smooth transitions
        pattern_mask = pattern_mask[:, :, np.newaxis]
        
        # Apply gentle horizontal smoothing only in noisy areas
        smoothed = cv2.filter2D(image, -1, kernel.squeeze())
        result = image * (1 - pattern_mask * pattern_strength) + smoothed * (pattern_mask * pattern_strength)
        
        return result
    
    def _adapt_settings_for_iso(self, settings: NoiseReductionSettings, iso: int) -> NoiseReductionSettings:
        """Adapt noise reduction settings based on ISO value"""
        # Create a copy to avoid modifying original
        adapted = NoiseReductionSettings(**settings.__dict__)
        
        # Get Sony A7 III noise characteristics for this ISO
        noise_profile = self._get_noise_profile_for_iso(iso)
        
        # Adjust luminance noise reduction based on ISO
        if iso <= 400:
            adapted.luminance_amount = settings.luminance_amount * 0.5  # Gentle
            adapted.chrominance_amount = settings.chrominance_amount * 0.3
        elif iso <= 1600:
            adapted.luminance_amount = settings.luminance_amount * 0.8  # Moderate
            adapted.chrominance_amount = settings.chrominance_amount * 0.6
        elif iso <= 3200:
            adapted.luminance_amount = settings.luminance_amount * 1.0  # Full strength
            adapted.chrominance_amount = settings.chrominance_amount * 0.8
        elif iso <= 6400:
            adapted.luminance_amount = settings.luminance_amount * 1.2  # Aggressive
            adapted.chrominance_amount = settings.chrominance_amount * 1.0
        else:  # Very high ISO
            adapted.luminance_amount = settings.luminance_amount * 1.5  # Very aggressive
            adapted.chrominance_amount = settings.chrominance_amount * 1.2
        
        # Adjust detail preservation - less preservation at higher ISOs
        if iso > 3200:
            adapted.luminance_detail *= 0.8
            adapted.chrominance_detail *= 0.7
            adapted.edge_preservation *= 0.9
        
        # Clamp values to valid ranges
        adapted.luminance_amount = min(adapted.luminance_amount, 100.0)
        adapted.chrominance_amount = min(adapted.chrominance_amount, 100.0)
        
        return adapted
    
    def _get_noise_profile_for_iso(self, iso: int) -> Dict[str, float]:
        """Get noise characteristics for given ISO (with interpolation)"""
        iso_values = sorted(self.a7m3_noise_profile.keys())
        
        if iso <= iso_values[0]:
            return self.a7m3_noise_profile[iso_values[0]]
        if iso >= iso_values[-1]:
            return self.a7m3_noise_profile[iso_values[-1]]
        
        # Find surrounding ISO values for interpolation
        for i in range(len(iso_values) - 1):
            if iso_values[i] <= iso <= iso_values[i + 1]:
                low_iso = iso_values[i]
                high_iso = iso_values[i + 1]
                
                # Linear interpolation
                factor = (iso - low_iso) / (high_iso - low_iso)
                
                low_profile = self.a7m3_noise_profile[low_iso]
                high_profile = self.a7m3_noise_profile[high_iso]
                
                return {
                    'luminance_noise': low_profile['luminance_noise'] + 
                                     factor * (high_profile['luminance_noise'] - low_profile['luminance_noise']),
                    'chroma_noise': low_profile['chroma_noise'] + 
                                  factor * (high_profile['chroma_noise'] - low_profile['chroma_noise']),
                    'pattern_noise': low_profile['pattern_noise'] + 
                                   factor * (high_profile['pattern_noise'] - low_profile['pattern_noise'])
                }
        
        # Fallback
        return self.a7m3_noise_profile[1600]


@dataclass
class LocalAdjustment:
    """Represents a local adjustment with mask and parameters"""
    
    # Adjustment identification
    name: str = "Local Adjustment"
    enabled: bool = True
    
    # Mask definition
    mask_type: str = "luminosity"  # "luminosity", "color", "radial", "linear", "brush"
    mask_data: Dict[str, Any] = field(default_factory=dict)
    invert_mask: bool = False
    
    # Adjustment parameters
    exposure: float = 0.0       # -3 to +3 EV
    shadows: float = 0.0        # -100 to +100
    highlights: float = 0.0     # -100 to +100  
    whites: float = 0.0         # -100 to +100
    blacks: float = 0.0         # -100 to +100
    
    # Color adjustments
    vibrance: float = 0.0       # -100 to +100
    saturation: float = 0.0     # -100 to +100
    hue_shift: float = 0.0      # -180 to +180 degrees
    
    # Local contrast  
    clarity: float = 0.0        # -100 to +100
    structure: float = 0.0      # -100 to +100
    
    # Mask refinement
    feather: float = 50.0       # 0-100, edge softness
    opacity: float = 100.0      # 0-100, adjustment strength
    
    # Color range selection (for color masks)
    target_color: Optional[Tuple[float, float, float]] = None  # RGB 0-1
    color_tolerance: float = 20.0  # 0-100


@dataclass 
class LocalAdjustmentSettings:
    """Settings for the local adjustment system"""
    
    # Global settings
    max_adjustments: int = 10          # Maximum number of local adjustments
    mask_resolution_factor: float = 1.0  # 1.0 = full resolution masks
    
    # Mask generation
    auto_mask_refinement: bool = True   # Automatically refine mask edges
    edge_aware_feathering: bool = True  # Use image edges for smarter feathering
    
    # Performance
    use_gpu_acceleration: bool = False  # GPU acceleration if available
    cache_masks: bool = True           # Cache generated masks


class LocalAdjustmentProcessor:
    """
    Advanced local adjustment system with intelligent mask generation
    
    Features:
    - Multiple mask types (luminosity, color, geometric, brush)
    - Edge-aware mask refinement
    - Non-destructive layer system
    - Batch mask generation
    - Sony A7 III optimized processing
    """
    
    def __init__(self, settings: LocalAdjustmentSettings = None):
        """Initialize local adjustment processor"""
        self.settings = settings or LocalAdjustmentSettings()
        self.mask_cache = {}
        
        logger.info("Local adjustment processor initialized")
    
    def apply_local_adjustments(self, image: np.ndarray, 
                               adjustments: List[LocalAdjustment]) -> np.ndarray:
        """
        Apply a list of local adjustments to an image
        
        Args:
            image: Input image (RGB, 0-1 range)
            adjustments: List of local adjustments to apply
            
        Returns:
            Image with local adjustments applied
        """
        if not adjustments:
            return image
        
        # Start with original image
        result = image.copy()
        
        # Apply each adjustment in order
        for i, adjustment in enumerate(adjustments):
            if not adjustment.enabled:
                continue
                
            logger.info(f"Applying local adjustment {i+1}/{len(adjustments)}: {adjustment.name}")
            
            # Generate or retrieve mask
            mask = self._generate_mask(image, adjustment)
            
            # Apply the adjustment
            result = self._apply_single_adjustment(result, adjustment, mask)
        
        return result
    
    def _generate_mask(self, image: np.ndarray, adjustment: LocalAdjustment) -> np.ndarray:
        """Generate mask for local adjustment"""
        mask_key = self._get_mask_cache_key(adjustment)
        
        if self.settings.cache_masks and mask_key in self.mask_cache:
            return self.mask_cache[mask_key]
        
        mask_type = adjustment.mask_type
        
        if mask_type == "luminosity":
            mask = self._create_luminosity_mask(image, adjustment.mask_data)
        elif mask_type == "color":
            mask = self._create_color_mask(image, adjustment)
        elif mask_type == "radial":
            mask = self._create_radial_mask(image, adjustment.mask_data)
        elif mask_type == "linear":
            mask = self._create_linear_mask(image, adjustment.mask_data)
        elif mask_type == "brush":
            mask = self._create_brush_mask(image, adjustment.mask_data)
        else:
            logger.warning(f"Unknown mask type: {mask_type}, using full image")
            mask = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply mask refinements
        if adjustment.feather > 0:
            mask = self._apply_feathering(mask, adjustment.feather, image)
        
        if adjustment.invert_mask:
            mask = 1.0 - mask
        
        # Apply opacity
        mask = mask * (adjustment.opacity / 100.0)
        
        # Cache the mask
        if self.settings.cache_masks:
            self.mask_cache[mask_key] = mask
        
        return mask
    
    def _create_luminosity_mask(self, image: np.ndarray, 
                               mask_data: Dict[str, Any]) -> np.ndarray:
        """Create luminosity-based mask"""
        # Convert to luminance
        luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
        
        # Get parameters
        target_range = mask_data.get('target_range', 'midtones')  # 'shadows', 'midtones', 'highlights'
        range_width = mask_data.get('range_width', 0.3)  # 0.1-1.0
        smooth_falloff = mask_data.get('smooth_falloff', True)
        
        if target_range == 'shadows':
            # Shadow mask - bright in shadows, dark in highlights
            center = 0.2
            mask = np.exp(-((luminance - center) / (range_width / 2)) ** 2)
        elif target_range == 'highlights':
            # Highlight mask - bright in highlights, dark in shadows
            center = 0.8
            mask = np.exp(-((luminance - center) / (range_width / 2)) ** 2)
        else:  # midtones
            # Midtone mask - bright in midtones, dark at extremes
            center = 0.5
            mask = np.exp(-((luminance - center) / (range_width / 2)) ** 2)
        
        # Apply smooth falloff
        if smooth_falloff:
            mask = np.power(mask, 0.7)  # Softer transitions
        
        return mask.astype(np.float32)
    
    def _create_color_mask(self, image: np.ndarray, adjustment: LocalAdjustment) -> np.ndarray:
        """Create color-based mask"""
        if not adjustment.target_color:
            logger.warning("No target color specified for color mask")
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        
        target_rgb = np.array(adjustment.target_color)
        tolerance = adjustment.color_tolerance / 100.0
        
        # Convert to LAB for better color matching
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor((target_rgb * 255).astype(np.uint8).reshape(1, 1, 3), 
                                 cv2.COLOR_RGB2LAB)[0, 0]
        
        # Calculate color distance in LAB space
        lab_float = lab.astype(np.float32)
        
        # Weighted distance (L*, a*, b*)
        l_diff = (lab_float[:, :, 0] - target_lab[0]) / 100.0  # L* range 0-100
        a_diff = (lab_float[:, :, 1] - target_lab[1]) / 128.0  # a* range -128 to 127
        b_diff = (lab_float[:, :, 2] - target_lab[2]) / 128.0  # b* range -128 to 127
        
        # Perceptually weighted distance
        color_distance = np.sqrt(2 * l_diff**2 + 4 * a_diff**2 + 3 * b_diff**2)
        
        # Create mask based on distance and tolerance
        max_distance = tolerance * 2.0  # Scale tolerance appropriately
        mask = np.maximum(0, 1 - color_distance / max_distance)
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask.astype(np.float32)
    
    def _create_radial_mask(self, image: np.ndarray, mask_data: Dict[str, Any]) -> np.ndarray:
        """Create radial (circular) mask"""
        height, width = image.shape[:2]
        
        # Get parameters
        center_x = mask_data.get('center_x', 0.5) * width   # 0-1 normalized to pixels
        center_y = mask_data.get('center_y', 0.5) * height
        radius = mask_data.get('radius', 0.3) * min(width, height)
        falloff = mask_data.get('falloff', 2.0)  # Higher = sharper edge
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Calculate distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create radial mask with smooth falloff
        mask = np.exp(-(distance / radius) ** falloff)
        
        return mask.astype(np.float32)
    
    def _create_linear_mask(self, image: np.ndarray, mask_data: Dict[str, Any]) -> np.ndarray:
        """Create linear gradient mask"""
        height, width = image.shape[:2]
        
        # Get parameters
        angle = mask_data.get('angle', 0.0)  # Degrees, 0 = horizontal
        center_x = mask_data.get('center_x', 0.5) * width
        center_y = mask_data.get('center_y', 0.5) * height
        gradient_width = mask_data.get('gradient_width', 0.5)  # 0-1
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Rotate coordinates based on angle
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Translate to center
        x_centered = x - center_x
        y_centered = y - center_y
        
        # Rotate
        x_rot = x_centered * cos_a - y_centered * sin_a
        
        # Create linear gradient
        gradient_range = gradient_width * min(width, height) / 2
        mask = 0.5 + 0.5 * np.tanh(x_rot / gradient_range)
        
        return mask.astype(np.float32)
    
    def _create_brush_mask(self, image: np.ndarray, mask_data: Dict[str, Any]) -> np.ndarray:
        """Create brush-style mask from stroke data"""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Get brush strokes
        strokes = mask_data.get('strokes', [])
        brush_size = mask_data.get('brush_size', 50)  # pixels
        brush_hardness = mask_data.get('brush_hardness', 0.5)  # 0-1
        
        for stroke in strokes:
            points = stroke.get('points', [])
            opacity = stroke.get('opacity', 1.0)
            
            # Draw brush stroke
            for i in range(len(points) - 1):
                x1, y1 = int(points[i][0] * width), int(points[i][1] * height)
                x2, y2 = int(points[i+1][0] * width), int(points[i+1][1] * height)
                
                # Simple brush implementation - could be enhanced
                cv2.line(mask, (x1, y1), (x2, y2), opacity, thickness=brush_size)
        
        # Apply brush hardness (blur for soft brushes)
        if brush_hardness < 1.0:
            blur_amount = int((1 - brush_hardness) * 10)
            if blur_amount > 0:
                mask = cv2.GaussianBlur(mask, (blur_amount*2+1, blur_amount*2+1), 0)
        
        return np.clip(mask, 0, 1).astype(np.float32)
    
    def _apply_feathering(self, mask: np.ndarray, feather_amount: float,
                         image: np.ndarray = None) -> np.ndarray:
        """Apply edge feathering to mask"""
        if feather_amount <= 0:
            return mask
        
        # Basic Gaussian feathering
        feather_sigma = feather_amount / 20.0  # Scale to reasonable sigma
        feathered = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
        
        # Edge-aware feathering if image provided and enabled
        if (image is not None and self.settings.edge_aware_feathering):
            # Use image edges to guide feathering
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_mask = edges.astype(np.float32) / 255.0
            
            # Reduce feathering near edges
            edge_preservation = 0.7  # How much to preserve edges
            adaptive_feather = feathered * (1 - edge_mask * edge_preservation) + mask * (edge_mask * edge_preservation)
            return adaptive_feather
        
        return feathered
    
    def _apply_single_adjustment(self, image: np.ndarray, adjustment: LocalAdjustment,
                                mask: np.ndarray) -> np.ndarray:
        """Apply a single local adjustment using the provided mask"""
        # Expand mask to 3 channels
        mask_3d = mask[:, :, np.newaxis]
        
        # Start with original image
        adjusted = image.copy()
        
        # Apply exposure adjustment
        if adjustment.exposure != 0:
            exposure_factor = 2 ** adjustment.exposure
            adjusted = adjusted * exposure_factor
        
        # Apply tone adjustments (simplified - could use ToneMapper for more precision)
        if any([adjustment.shadows, adjustment.highlights, adjustment.whites, adjustment.blacks]):
            adjusted = self._apply_tone_adjustments(adjusted, adjustment)
        
        # Apply color adjustments
        if any([adjustment.vibrance, adjustment.saturation, adjustment.hue_shift]):
            adjusted = self._apply_color_adjustments(adjusted, adjustment)
        
        # Apply local contrast (clarity/structure)  
        if adjustment.clarity != 0 or adjustment.structure != 0:
            adjusted = self._apply_local_contrast(adjusted, adjustment)
        
        # Blend with original using mask
        result = image * (1 - mask_3d) + adjusted * mask_3d
        
        return np.clip(result, 0, 1)
    
    def _apply_tone_adjustments(self, image: np.ndarray, adjustment: LocalAdjustment) -> np.ndarray:
        """Apply tone adjustments (simplified implementation)"""
        # Convert to luminance for tone curve
        luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
        
        # Simple tone curve adjustments
        result = image.copy()
        
        # Shadows/highlights (basic implementation)
        if adjustment.shadows != 0:
            shadow_mask = np.maximum(0, 1 - luminance * 2)  # Strong in shadows
            shadow_factor = 1 + (adjustment.shadows / 100.0) * 0.5
            result = result * (1 + shadow_mask[:, :, np.newaxis] * (shadow_factor - 1))
        
        if adjustment.highlights != 0:
            highlight_mask = np.maximum(0, luminance * 2 - 1)  # Strong in highlights  
            highlight_factor = 1 + (adjustment.highlights / 100.0) * 0.5
            result = result * (1 + highlight_mask[:, :, np.newaxis] * (highlight_factor - 1))
        
        return result
    
    def _apply_color_adjustments(self, image: np.ndarray, adjustment: LocalAdjustment) -> np.ndarray:
        """Apply color adjustments"""
        result = image.copy()
        
        # Convert to HSV for color manipulation
        hsv = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Apply hue shift
        if adjustment.hue_shift != 0:
            hsv[:, :, 0] = np.mod(hsv[:, :, 0] + adjustment.hue_shift, 180)
        
        # Apply saturation
        if adjustment.saturation != 0:
            sat_factor = 1 + adjustment.saturation / 100.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        
        # Convert back to RGB
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        # Apply vibrance (smart saturation)
        if adjustment.vibrance != 0:
            # Simple vibrance implementation - enhance less saturated colors more
            saturation = hsv[:, :, 1] / 255.0
            vibrance_mask = 1.0 - saturation  # More effect on less saturated areas
            
            vibrance_factor = 1 + (adjustment.vibrance / 100.0) * vibrance_mask[:, :, np.newaxis]
            result = image + (result - image) * vibrance_factor
        
        return result
    
    def _apply_local_contrast(self, image: np.ndarray, adjustment: LocalAdjustment) -> np.ndarray:
        """Apply local contrast adjustments (clarity/structure)"""
        if adjustment.clarity == 0 and adjustment.structure == 0:
            return image
        
        # Convert to LAB for better contrast processing
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        result = image.copy()
        
        # Clarity (medium frequency contrast)
        if adjustment.clarity != 0:
            blur_radius = 10
            blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=blur_radius)
            unsharp_mask = l_channel - blurred
            
            clarity_factor = adjustment.clarity / 100.0 * 0.5
            enhanced_l = l_channel + unsharp_mask * clarity_factor
            
            # Apply back to image
            lab[:, :, 0] = np.clip(enhanced_l, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        return result
    
    def _get_mask_cache_key(self, adjustment: LocalAdjustment) -> str:
        """Generate cache key for mask"""
        # Simple cache key based on mask parameters
        key_data = {
            'type': adjustment.mask_type,
            'data': adjustment.mask_data,
            'feather': adjustment.feather,
            'invert': adjustment.invert_mask,
            'opacity': adjustment.opacity
        }
        return str(hash(str(sorted(key_data.items()))))
    
    def create_luminosity_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Create a set of standard luminosity masks"""
        luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
        
        masks = {}
        
        # Basic luminosity masks
        masks['shadows'] = np.maximum(0, 1 - luminance * 2)
        masks['midtones'] = 4 * luminance * (1 - luminance)  # Bell curve
        masks['highlights'] = np.maximum(0, luminance * 2 - 1)
        
        # Refined luminosity masks
        masks['dark_shadows'] = masks['shadows'] ** 2
        masks['light_shadows'] = masks['shadows'] ** 0.5
        masks['dark_highlights'] = masks['highlights'] ** 2  
        masks['light_highlights'] = masks['highlights'] ** 0.5
        
        return {k: v.astype(np.float32) for k, v in masks.items()}