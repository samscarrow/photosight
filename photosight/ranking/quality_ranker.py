"""
Quality-based photo ranking system.

Implements comprehensive photo quality assessment using multiple AI models
and technical analysis to provide accurate photo rankings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)


class QualityRanker:
    """
    Advanced photo quality ranking system.
    
    Combines multiple quality assessment techniques:
    - Technical quality (sharpness, exposure, noise)
    - Composition analysis (rule of thirds, leading lines)
    - AI-powered aesthetic scoring
    - YOLO-based subject detection and analysis
    """
    
    def __init__(self, config: Dict):
        """Initialize the quality ranker with configuration."""
        self.config = config
        self.ranking_config = config.get('ranking', {})
        
        # Weight factors for different quality aspects
        self.weights = self.ranking_config.get('weights', {
            'technical': 0.3,
            'composition': 0.25,
            'aesthetic': 0.25,
            'subject': 0.2
        })
        
        # Initialize analyzers lazily
        self._technical_analyzer = None
        self._composition_analyzer = None
        self._aesthetic_analyzer = None
        self._subject_analyzer = None
        self._vision_llm_analyzer = None
        
    @property
    def technical_analyzer(self):
        """Lazy loading of technical analyzer."""
        if self._technical_analyzer is None:
            from ..analysis.technical_analyzer import TechnicalAnalyzer
            self._technical_analyzer = TechnicalAnalyzer(self.config)
        return self._technical_analyzer
    
    @property
    def composition_analyzer(self):
        """Lazy loading of composition analyzer."""
        if self._composition_analyzer is None:
            from ..analysis.composition_analyzer import CompositionAnalyzer
            self._composition_analyzer = CompositionAnalyzer(self.config)
        return self._composition_analyzer
    
    @property
    def aesthetic_analyzer(self):
        """Lazy loading of aesthetic analyzer."""
        if self._aesthetic_analyzer is None:
            from ..analysis.aesthetic_analyzer import AestheticAnalyzer
            self._aesthetic_analyzer = AestheticAnalyzer(self.config)
        return self._aesthetic_analyzer
    
    @property
    def subject_analyzer(self):
        """Lazy loading of subject analyzer."""
        if self._subject_analyzer is None:
            from ..processing.production_yolo_blur_processor import ProductionYOLOBlurProcessor
            self._subject_analyzer = ProductionYOLOBlurProcessor(self.config)
        return self._subject_analyzer
    
    @property
    def vision_llm_analyzer(self):
        """Lazy loading of vision LLM analyzer."""
        if self._vision_llm_analyzer is None:
            try:
                from ..analysis.vision_llm_analyzer import VisionLLMAnalyzer
                self._vision_llm_analyzer = VisionLLMAnalyzer(self.config)
            except Exception as e:
                logger.warning(f"Could not initialize vision LLM analyzer: {e}")
                self._vision_llm_analyzer = None
        return self._vision_llm_analyzer
    
    def rank_photo(self, photo_path: Union[str, Path]) -> float:
        """
        Rank a single photo and return a quality score (0.0-1.0).
        
        Args:
            photo_path: Path to the photo file
            
        Returns:
            Quality score between 0.0 and 1.0 (higher is better)
        """
        photo_path = Path(photo_path)
        
        if not photo_path.exists():
            logger.error(f"Photo not found: {photo_path}")
            return 0.0
        
        try:
            # Load image for analysis - handle RAW files
            image = self._load_image(photo_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create unified analysis context - single source of truth for image data
            import numpy as np
            from ..processing.analysis_context import AnalysisContext
            
            # Convert to numpy array and create analysis context
            image_array = np.array(image, dtype=np.uint8)
            # Ensure proper shape and contiguous memory layout for OpenCV
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = np.ascontiguousarray(image_array, dtype=np.uint8)
            
            # Create analysis context with standardized data pipeline
            context = AnalysisContext(image_array, str(photo_path))
            
            scores = {}
            
            # Get emotional impact from Vision LLM first (if available)
            emotional_impact_score = 0.5  # Default fallback
            try:
                if self.vision_llm_analyzer and self.vision_llm_analyzer.enabled:
                    vision_result = self.vision_llm_analyzer.analyze_emotional_impact(photo_path)
                    vision_analysis = vision_result.get('parsed', {})
                    if vision_analysis:
                        emotional_impact_score = vision_analysis.get('emotion_score', 0.5)
                        emotional_impact_score = max(0.0, min(1.0, float(emotional_impact_score)))
                        logger.debug(f"Vision LLM emotional impact: {emotional_impact_score:.3f}")
            except Exception as e:
                logger.warning(f"Vision LLM emotional analysis failed for {photo_path}: {e}")
            
            # Technical quality analysis
            try:
                technical_score = self._analyze_technical_quality(image, photo_path)
                scores['technical'] = technical_score
            except Exception as e:
                logger.warning(f"Technical analysis failed for {photo_path}: {e}")
                scores['technical'] = 0.5  # Default score
            
            # Composition analysis (context provides appropriate data types)
            try:
                composition_score = self._analyze_composition(context)
                scores['composition'] = composition_score
            except Exception as e:
                logger.warning(f"Composition analysis failed for {photo_path}: {e}")
                scores['composition'] = 0.5
            
            # Aesthetic analysis (context provides appropriate data types)
            # Pass emotional impact score to avoid redundant vision LLM calls
            try:
                logger.debug(f"Aesthetic analysis - image size: {context.width}x{context.height}, channels: {context.channels}")
                aesthetic_score = self._analyze_aesthetics(context, emotional_impact_score)
                scores['aesthetic'] = aesthetic_score
            except Exception as e:
                logger.warning(f"Aesthetic analysis failed for {photo_path}: {e}")
                scores['aesthetic'] = 0.5
            
            # Subject analysis (context provides appropriate data types)
            try:
                subject_score = self._analyze_subjects(context, photo_path)
                scores['subject'] = subject_score
            except Exception as e:
                logger.warning(f"Subject analysis failed for {photo_path}: {e}")
                scores['subject'] = 0.5
            
            # Calculate weighted overall score
            overall_score = sum(
                scores[aspect] * self.weights[aspect]
                for aspect in scores.keys()
                if aspect in self.weights
            )
            
            # Decisive moment detection and bonus (vision LLM analysis)
            is_decisive_moment = False
            decisive_moment_score = 0.0
            try:
                if self.vision_llm_analyzer and self.vision_llm_analyzer.enabled:
                    moment_analysis = self.vision_llm_analyzer.detect_moment(photo_path)
                    if moment_analysis and 'parsed' in moment_analysis:
                        parsed = moment_analysis['parsed']
                        is_decisive_moment = parsed.get('is_decisive_moment', False)
                        decisive_moment_score = parsed.get('decisive_moment_score', 0.0)
                        
                        # Apply decisive moment bonus - 20% bonus for true decisive moments
                        if is_decisive_moment and decisive_moment_score > 0.7:
                            overall_score *= 1.2
                            logger.debug(f"Applied decisive moment bonus to {photo_path.name}")
                        
                        # Store decisive moment info for later use
                        scores['is_decisive_moment'] = is_decisive_moment
                        scores['decisive_moment_score'] = decisive_moment_score
            except Exception as e:
                logger.warning(f"Decisive moment analysis failed for {photo_path}: {e}")
                scores['is_decisive_moment'] = False
                scores['decisive_moment_score'] = 0.0
            
            # Ensure score is in valid range
            overall_score = max(0.0, min(1.0, overall_score))
            
            # Log results including decisive moment info
            decisive_info = ""
            if scores.get('is_decisive_moment', False):
                decisive_info = f", DM:{scores.get('decisive_moment_score', 0.0):.2f}✓"
            
            logger.debug(f"Ranked {photo_path.name}: {overall_score:.3f} "
                        f"(T:{scores['technical']:.2f}, C:{scores['composition']:.2f}, "
                        f"A:{scores['aesthetic']:.2f}, S:{scores['subject']:.2f}{decisive_info})")
            
            return overall_score
            
        except Exception as e:
            logger.error(f"Error ranking photo {photo_path}: {e}")
            return 0.0
    
    def rank_photos_batch(self, photo_paths: List[Union[str, Path]], 
                         progress_callback: Optional[callable] = None) -> List[Tuple[Path, float]]:
        """
        Rank multiple photos and return sorted results.
        
        Args:
            photo_paths: List of photo file paths
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of (path, score) tuples sorted by score (descending)
        """
        results = []
        
        for i, photo_path in enumerate(photo_paths):
            score = self.rank_photo(photo_path)
            results.append((Path(photo_path), score))
            
            if progress_callback:
                progress_callback(i + 1, len(photo_paths))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _analyze_technical_quality(self, image: Image.Image, photo_path: Path) -> float:
        """Analyze technical quality aspects."""
        try:
            analysis = self.technical_analyzer.analyze_photo(image, photo_path)
            
            # Combine technical metrics
            sharpness = analysis.get('sharpness', 0) / 1000.0  # Normalize
            sharpness = min(1.0, sharpness)  # Cap at 1.0
            
            exposure_quality = analysis.get('exposure_quality', 0.5)
            noise_quality = 1.0 - analysis.get('noise_level', 0.5)  # Invert noise
            
            # Weight technical aspects
            technical_score = (
                sharpness * 0.4 +
                exposure_quality * 0.4 +
                noise_quality * 0.2
            )
            
            return max(0.0, min(1.0, technical_score))
            
        except Exception as e:
            logger.warning(f"Technical analysis error: {e}")
            return 0.5
    
    def _analyze_composition(self, context) -> float:
        """Analyze composition quality using analysis context."""
        try:
            # Let the composition analyzer request the specific data format it needs
            analysis = self.composition_analyzer.analyze_composition(context)
            
            # Extract composition scores
            rule_of_thirds = analysis.get('rule_of_thirds_score', 0.5)
            balance = analysis.get('balance_score', 0.5)
            leading_lines = analysis.get('leading_lines_score', 0.5)
            symmetry = analysis.get('symmetry_score', 0.5)
            
            # Weight composition aspects
            composition_score = (
                rule_of_thirds * 0.3 +
                balance * 0.3 +
                leading_lines * 0.2 +
                symmetry * 0.2
            )
            
            return max(0.0, min(1.0, composition_score))
            
        except Exception as e:
            logger.warning(f"Composition analysis error: {e}")
            return 0.5
    
    def _analyze_aesthetics(self, context, emotional_impact_score: float) -> float:
        """Analyze aesthetic quality using analysis context and pre-computed emotional impact."""
        try:
            # Let the aesthetic analyzer request the specific data format it needs
            # Pass emotional impact score to avoid redundant vision LLM calls
            analysis = self.aesthetic_analyzer.analyze_aesthetics(context, emotional_impact_score)
            
            # Extract aesthetic scores
            color_harmony = analysis.get('color_harmony', 0.5)
            contrast = analysis.get('contrast_score', 0.5)
            saturation = analysis.get('saturation_score', 0.5)
            overall_appeal = analysis.get('overall_appeal', 0.5)
            
            # Use pre-computed emotional impact score instead of mood_score
            # This replaces the traditional mood-based scoring with LLM-powered emotional analysis
            
            # Weight aesthetic aspects (updated to use single emotional score)
            aesthetic_score = (
                color_harmony * 0.30 +
                contrast * 0.25 +
                saturation * 0.20 +
                emotional_impact_score * 0.25  # Use pre-computed emotional impact
            )
            
            return max(0.0, min(1.0, aesthetic_score))
            
        except Exception as e:
            logger.warning(f"Aesthetic analysis error: {e}")
            return 0.5
    
    def _analyze_subjects(self, context, photo_path: Path) -> float:
        """Analyze subject quality and focus using analysis context."""
        try:
            # The production YOLO processor expects image file paths, not context
            # For now, provide a simplified subject analysis based on context data
            
            # Use edge detection and contrast as proxies for subject clarity
            edges = context.get_edge_map(50, 150)
            edge_density = np.mean(edges)
            
            # Use gradient magnitude for sharpness estimation
            gradient_mag = context.get_gradient_magnitude()
            sharpness_score = np.mean(gradient_mag)
            
            # Simple subject analysis based on image statistics
            subject_clarity = min(sharpness_score * 2.0, 1.0)  # Scale gradient magnitude
            subject_prominence = min(edge_density * 3.0, 1.0)  # Scale edge density
            background_quality = 0.5  # Default neutral score
            
            # General photo weighting (no face detection in simplified version)
            subject_score = (
                subject_clarity * 0.4 +
                subject_prominence * 0.3 +
                background_quality * 0.3
            )
            
            return max(0.0, min(1.0, subject_score))
            
        except Exception as e:
            logger.warning(f"Subject analysis error: {e}")
            return 0.5
    
    def get_detailed_analysis(self, photo_path: Union[str, Path]) -> Dict:
        """
        Get detailed quality analysis for a photo.
        
        Returns:
            Dictionary with detailed scores and analysis
        """
        photo_path = Path(photo_path)
        
        try:
            # Load image
            image = self._load_image(photo_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create analysis context
            image_array = np.array(image, dtype=np.uint8)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = np.ascontiguousarray(image_array, dtype=np.uint8)
            
            from ..processing.analysis_context import AnalysisContext
            context = AnalysisContext(image_array, str(photo_path))
            
            # Get emotional impact from Vision LLM first
            emotional_impact_score = 0.5
            emotional_analysis = {}
            try:
                if self.vision_llm_analyzer and self.vision_llm_analyzer.enabled:
                    vision_result = self.vision_llm_analyzer.analyze_emotional_impact(photo_path)
                    emotional_analysis = vision_result.get('parsed', {})
                    if emotional_analysis:
                        emotional_impact_score = emotional_analysis.get('emotion_score', 0.5)
                        emotional_impact_score = max(0.0, min(1.0, float(emotional_impact_score)))
            except Exception as e:
                logger.warning(f"Vision LLM emotional analysis failed: {e}")
            
            # Get detailed analysis from each component
            technical = self.technical_analyzer.analyze_photo(image, photo_path)
            composition = self.composition_analyzer.analyze_composition(context)
            aesthetic = self.aesthetic_analyzer.analyze_aesthetics(context, emotional_impact_score)
            
            # Simplified subject analysis
            subject = {
                'subject_clarity': 0.5,
                'subject_prominence': 0.5,
                'background_quality': 0.5
            }
            
            # Calculate component scores
            technical_score = self._analyze_technical_quality(image, photo_path)
            composition_score = self._analyze_composition(context)
            aesthetic_score = self._analyze_aesthetics(context, emotional_impact_score)
            subject_score = self._analyze_subjects(context, photo_path)
            
            # Calculate base overall score
            overall_score = sum(
                score * self.weights[aspect]
                for aspect, score in [
                    ('technical', technical_score),
                    ('composition', composition_score),
                    ('aesthetic', aesthetic_score),
                    ('subject', subject_score)
                ]
            )
            
            # Decisive moment analysis
            decisive_moment_data = {
                'is_decisive_moment': False,
                'decisive_moment_score': 0.0,
                'moment_analysis': {}
            }
            
            try:
                if self.vision_llm_analyzer and self.vision_llm_analyzer.enabled:
                    moment_result = self.vision_llm_analyzer.detect_moment(photo_path)
                    if moment_result and 'parsed' in moment_result:
                        decisive_moment_data['moment_analysis'] = moment_result['parsed']
                        decisive_moment_data['is_decisive_moment'] = moment_result['parsed'].get('is_decisive_moment', False)
                        decisive_moment_data['decisive_moment_score'] = moment_result['parsed'].get('decisive_moment_score', 0.0)
                        
                        # Apply decisive moment bonus
                        if decisive_moment_data['is_decisive_moment'] and decisive_moment_data['decisive_moment_score'] > 0.7:
                            overall_score *= 1.2
            except Exception as e:
                logger.warning(f"Decisive moment analysis failed: {e}")
            
            # Ensure score is in valid range
            overall_score = max(0.0, min(1.0, overall_score))
            
            return {
                'overall_score': overall_score,
                'component_scores': {
                    'technical': technical_score,
                    'composition': composition_score,
                    'aesthetic': aesthetic_score,
                    'subject': subject_score
                },
                'detailed_analysis': {
                    'technical': technical,
                    'composition': composition,
                    'aesthetic': aesthetic,
                    'subject': subject,
                    'emotional_impact': emotional_analysis,
                    'decisive_moment': decisive_moment_data
                },
                'weights': self.weights,
                'file_path': str(photo_path),
                'file_name': photo_path.name,
                'is_decisive_moment': decisive_moment_data['is_decisive_moment'],
                'decisive_moment_score': decisive_moment_data['decisive_moment_score']
            }
            
        except Exception as e:
            logger.error(f"Error in detailed analysis for {photo_path}: {e}")
            return {
                'overall_score': 0.0,
                'error': str(e),
                'file_path': str(photo_path),
                'file_name': photo_path.name
            }
    
    def _load_image(self, photo_path: Path) -> Image.Image:
        """
        Load image file handling both standard formats and RAW files.
        
        Args:
            photo_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        file_ext = photo_path.suffix.lower()
        
        # Check if it's a RAW file
        raw_extensions = {'.arw', '.cr2', '.nef', '.dng', '.raf', '.orf', '.rw2', '.pef', '.srw'}
        
        if file_ext in raw_extensions:
            try:
                import rawpy
                logger.debug(f"Processing RAW file: {photo_path}")
                
                # Open RAW file
                with rawpy.imread(str(photo_path)) as raw:
                    # Process to RGB array
                    rgb_array = raw.postprocess(
                        use_camera_wb=True,        # Use camera white balance
                        half_size=True,            # Use half-size for faster processing
                        no_auto_bright=True,       # Disable auto brightness
                        output_bps=8               # 8-bit output for compatibility
                    )
                
                # Convert numpy array to PIL Image
                image = Image.fromarray(rgb_array)
                logger.debug(f"Successfully processed RAW file: {photo_path}")
                return image
                
            except ImportError:
                logger.warning("rawpy not available, trying to extract embedded JPEG from RAW")
                return self._extract_raw_preview(photo_path)
            except Exception as e:
                logger.warning(f"RAW processing failed for {photo_path}: {e}, trying preview extraction")
                return self._extract_raw_preview(photo_path)
        else:
            # Standard image file
            return Image.open(photo_path)
    
    def _extract_raw_preview(self, photo_path: Path) -> Image.Image:
        """
        Extract embedded JPEG preview from RAW file as fallback.
        
        Args:
            photo_path: Path to the RAW file
            
        Returns:
            PIL Image object from embedded preview
        """
        try:
            # Try to open as TIFF and extract embedded JPEG
            from PIL import TiffImagePlugin
            
            img = Image.open(photo_path)
            
            # Try to get the JPEG interchange format
            if hasattr(img, 'tag') and img.tag:
                # Look for JPEG interchange format in TIFF tags
                jpeg_offset = img.tag.get(513)  # JPEGInterchangeFormat
                jpeg_length = img.tag.get(514)  # JPEGInterchangeFormatLength
                
                if jpeg_offset and jpeg_length:
                    # Extract JPEG data
                    with open(photo_path, 'rb') as f:
                        f.seek(jpeg_offset[0] if isinstance(jpeg_offset, tuple) else jpeg_offset)
                        jpeg_data = f.read(jpeg_length[0] if isinstance(jpeg_length, tuple) else jpeg_length)
                    
                    # Create PIL Image from JPEG data
                    from io import BytesIO
                    jpeg_image = Image.open(BytesIO(jpeg_data))
                    logger.debug(f"Extracted embedded JPEG from RAW: {photo_path}")
                    return jpeg_image
            
            # If that fails, try to use the image as-is (might work for some RAW formats)
            if img.mode in ['RGB', 'L', 'P']:
                return img
            else:
                # Convert to RGB
                return img.convert('RGB')
                
        except Exception as e:
            logger.error(f"Failed to extract preview from RAW file {photo_path}: {e}")
            # Return a black image as last resort
            return Image.new('RGB', (640, 480), color='black')