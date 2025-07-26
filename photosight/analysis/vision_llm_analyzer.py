"""
Vision LLM analyzer for PhotoSight.

This module provides integration with vision-capable Large Language Models
to enhance photo analysis with semantic understanding beyond traditional
computer vision techniques.
"""

import logging
import base64
import json
from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


class VisionLLMProvider:
    """Abstract base class for vision-capable LLM providers."""
    
    def __init__(self, config: Dict):
        """Initialize the provider with configuration."""
        self.config = config
        self.name = "base"
        self.max_image_size = config.get('max_image_size', 5 * 1024 * 1024)  # 5MB default
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    
    def prepare_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Prepare image for vision LLM analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prepared image data
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {image_path.suffix}")
        
        # Check file size
        file_size = image_path.stat().st_size
        if file_size > self.max_image_size:
            logger.warning(f"Image size ({file_size} bytes) exceeds limit, resizing")
            return self._resize_image(image_path)
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        return {
            'path': str(image_path),
            'data': base64.b64encode(image_data).decode('utf-8'),
            'format': image_path.suffix.lower()[1:],  # Remove dot
            'size': file_size
        }
    
    def _resize_image(self, image_path: Path, max_size: int = 1024) -> Dict:
        """Resize image to fit within size limits."""
        img = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # Resize maintaining aspect ratio
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        return {
            'path': str(image_path),
            'data': base64.b64encode(buffer.getvalue()).decode('utf-8'),
            'format': 'jpeg',
            'size': buffer.tell(),
            'resized': True
        }
    
    def analyze_image(self, image_path: Union[str, Path], prompt: str) -> Dict:
        """
        Analyze an image with a specific prompt.
        
        Args:
            image_path: Path to the image
            prompt: Analysis prompt
            
        Returns:
            Analysis results dictionary
        """
        raise NotImplementedError("Subclasses must implement analyze_image")
    
    def batch_analyze(self, images: List[Union[str, Path]], prompt: str) -> List[Dict]:
        """
        Analyze multiple images efficiently.
        
        Args:
            images: List of image paths
            prompt: Analysis prompt for all images
            
        Returns:
            List of analysis results
        """
        # Default implementation: sequential processing
        results = []
        for image_path in images:
            try:
                result = self.analyze_image(image_path, prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {image_path}: {e}")
                results.append({
                    'error': str(e),
                    'image': str(image_path)
                })
        return results


class PhotoAnalysisPrompts:
    """Standard prompts for photo analysis tasks."""
    
    SCENE_CLASSIFICATION = """Analyze this photograph and provide:
1. Primary scene type (e.g., landscape, portrait, street, wedding, nature)
2. Setting/location context
3. Time of day and lighting conditions
4. Key subjects or focal points
5. Overall mood or atmosphere

Format as JSON with keys: scene_type, setting, lighting, subjects, mood"""
    
    COMPOSITION_ANALYSIS = """Evaluate the photographic composition:
1. How well does it follow compositional rules (rule of thirds, leading lines, symmetry)?
2. What is the visual flow and focal point?
3. Balance and visual weight distribution
4. Use of negative space
5. Suggested improvements for better composition

Format as JSON with keys: rules_followed, focal_point, balance, negative_space, suggestions"""
    
    QUALITY_ASSESSMENT = """Assess the overall quality and impact of this photograph:
1. Technical quality (focus, exposure, color)
2. Artistic merit and emotional impact
3. Storytelling effectiveness
4. What makes this photo stand out (or not)?
5. Overall rating (1-10) with justification

Format as JSON with keys: technical_quality, artistic_merit, story_impact, standout_features, rating, justification"""
    
    MOMENT_DETECTION = """Identify if this captures a significant moment:
1. Is this a decisive/peak moment?
2. Emotional significance (if any)
3. Context of the moment
4. Why this moment matters
5. Related moments to look for

Format as JSON with keys: is_decisive_moment, emotion_level, context, significance, related_moments"""
    
    CROP_SUGGESTIONS = """Suggest optimal crops for this image:
1. Identify the main subject(s)
2. Suggest 2-3 different crop ratios and their purposes
3. What elements to include/exclude
4. How each crop changes the story/impact
5. Recommended primary crop

Format as JSON with keys: main_subjects, crop_options, elements_analysis, impact_analysis, recommended_crop"""


class VisionLLMAnalyzer:
    """
    Main analyzer class that coordinates vision LLM analysis.
    """
    
    def __init__(self, config: Dict):
        """Initialize the analyzer with configuration."""
        self.config = config
        self.vision_config = config.get('vision_llm', {})
        self.enabled = self.vision_config.get('enabled', False)
        self.provider = self._init_provider()
        self.cache = {}  # Simple in-memory cache
        
    def _init_provider(self) -> Optional[VisionLLMProvider]:
        """Initialize the appropriate vision LLM provider."""
        if not self.enabled:
            return None
            
        provider_name = self.vision_config.get('provider', 'gemini')
        
        # Import providers dynamically to avoid dependencies if not used
        if provider_name == 'gemini':
            from .vision_providers.gemini_vision import GeminiVisionProvider
            return GeminiVisionProvider(self.vision_config.get('gemini', {}))
        elif provider_name == 'claude':
            from .vision_providers.claude_vision import ClaudeVisionProvider
            return ClaudeVisionProvider(self.vision_config.get('claude', {}))
        elif provider_name == 'gpt4v':
            from .vision_providers.gpt4_vision import GPT4VisionProvider
            return GPT4VisionProvider(self.vision_config.get('gpt4v', {}))
        else:
            raise ValueError(f"Unknown vision provider: {provider_name}")
    
    def analyze_scene(self, image_path: Union[str, Path]) -> Dict:
        """
        Analyze scene content using vision LLM.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Scene analysis results
        """
        if not self.enabled or not self.provider:
            return {}
        
        cache_key = f"scene_{Path(image_path).name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            result = self.provider.analyze_image(
                image_path, 
                PhotoAnalysisPrompts.SCENE_CLASSIFICATION
            )
            
            # Parse JSON response
            if isinstance(result.get('content'), str):
                try:
                    result['parsed'] = json.loads(result['content'])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response")
                    result['parsed'] = {}
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            return {'error': str(e)}
    
    def analyze_composition(self, image_path: Union[str, Path]) -> Dict:
        """
        Analyze composition using vision LLM.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Composition analysis results
        """
        if not self.enabled or not self.provider:
            return {}
        
        cache_key = f"comp_{Path(image_path).name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            result = self.provider.analyze_image(
                image_path,
                PhotoAnalysisPrompts.COMPOSITION_ANALYSIS
            )
            
            # Parse JSON response
            if isinstance(result.get('content'), str):
                try:
                    result['parsed'] = json.loads(result['content'])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response")
                    result['parsed'] = {}
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Composition analysis failed: {e}")
            return {'error': str(e)}
    
    def assess_quality(self, image_path: Union[str, Path]) -> Dict:
        """
        Assess overall image quality using vision LLM.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Quality assessment results
        """
        if not self.enabled or not self.provider:
            return {}
        
        cache_key = f"quality_{Path(image_path).name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            result = self.provider.analyze_image(
                image_path,
                PhotoAnalysisPrompts.QUALITY_ASSESSMENT
            )
            
            # Parse JSON response
            if isinstance(result.get('content'), str):
                try:
                    result['parsed'] = json.loads(result['content'])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response")
                    result['parsed'] = {}
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {'error': str(e)}
    
    def detect_moment(self, image_path: Union[str, Path]) -> Dict:
        """
        Detect if image captures a significant moment.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Moment detection results
        """
        if not self.enabled or not self.provider:
            return {}
        
        try:
            result = self.provider.analyze_image(
                image_path,
                PhotoAnalysisPrompts.MOMENT_DETECTION
            )
            
            # Parse JSON response
            if isinstance(result.get('content'), str):
                try:
                    result['parsed'] = json.loads(result['content'])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response")
                    result['parsed'] = {}
            
            return result
            
        except Exception as e:
            logger.error(f"Moment detection failed: {e}")
            return {'error': str(e)}
    
    def suggest_crops(self, image_path: Union[str, Path]) -> Dict:
        """
        Get intelligent crop suggestions.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Crop suggestion results
        """
        if not self.enabled or not self.provider:
            return {}
        
        try:
            result = self.provider.analyze_image(
                image_path,
                PhotoAnalysisPrompts.CROP_SUGGESTIONS
            )
            
            # Parse JSON response
            if isinstance(result.get('content'), str):
                try:
                    result['parsed'] = json.loads(result['content'])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response")
                    result['parsed'] = {}
            
            return result
            
        except Exception as e:
            logger.error(f"Crop suggestion failed: {e}")
            return {'error': str(e)}
    
    def analyze_burst_sequence(self, image_paths: List[Union[str, Path]]) -> Dict:
        """
        Analyze a burst sequence to find the best shot.
        
        Args:
            image_paths: List of paths in the burst sequence
            
        Returns:
            Burst analysis results with recommended shots
        """
        if not self.enabled or not self.provider:
            return {}
        
        prompt = """Analyze this sequence of photos (likely a burst):
1. What moment/action is being captured?
2. Which frame(s) best capture the peak moment?
3. Technical quality comparison
4. Rank all frames from best to worst
5. Why did you choose the top frame?

Format as JSON with keys: moment_description, best_frames, quality_comparison, ranking, justification"""
        
        try:
            results = self.provider.batch_analyze(image_paths, prompt)
            
            # Aggregate results
            return {
                'sequence_analysis': results,
                'count': len(image_paths),
                'provider': self.provider.name
            }
            
        except Exception as e:
            logger.error(f"Burst analysis failed: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self.cache.clear()
        logger.info("Vision LLM cache cleared")