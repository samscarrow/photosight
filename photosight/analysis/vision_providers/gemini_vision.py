"""
Google Gemini Vision provider for PhotoSight.

Implements vision analysis using Google's Gemini Pro Vision model.
"""

import logging
import os
from typing import Dict, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI package not installed")


class GeminiVisionProvider:
    """Google Gemini Vision implementation."""
    
    def __init__(self, config: Dict):
        """
        Initialize Gemini Vision provider.
        
        Args:
            config: Provider-specific configuration
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        self.config = config
        self.name = "gemini"
        
        # Configure API
        api_key = config.get('api_key') or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key not provided")
        
        genai.configure(api_key=api_key)
        
        # Initialize model
        model_name = config.get('model', 'gemini-1.5-flash')
        self.model = genai.GenerativeModel(model_name)
        
        # Safety settings
        self.safety_settings = self._configure_safety()
        
        # Processing options
        self.max_image_size = config.get('max_image_size', 4 * 1024 * 1024)  # 4MB
        self.generation_config = {
            'temperature': config.get('temperature', 0.4),
            'top_p': config.get('top_p', 1),
            'top_k': config.get('top_k', 32),
            'max_output_tokens': config.get('max_output_tokens', 2048),
        }
    
    def _configure_safety(self) -> List[Dict]:
        """Configure safety settings based on config."""
        safety_level = self.config.get('safety_settings', 'medium')
        
        # Map safety levels
        level_map = {
            'low': 'BLOCK_ONLY_HIGH',
            'medium': 'BLOCK_MEDIUM_AND_ABOVE',
            'high': 'BLOCK_LOW_AND_ABOVE'
        }
        
        threshold = level_map.get(safety_level, 'BLOCK_MEDIUM_AND_ABOVE')
        
        return [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": threshold
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": threshold
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": threshold
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": threshold
            }
        ]
    
    def prepare_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Prepare image for Gemini Vision.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Prepared image data
        """
        from PIL import Image
        
        image_path = Path(image_path)
        
        # Load image
        img = Image.open(image_path)
        
        # Gemini accepts PIL images directly
        return {
            'image': img,
            'path': str(image_path),
            'format': img.format,
            'size': image_path.stat().st_size
        }
    
    def analyze_image(self, image_path: Union[str, Path], prompt: str) -> Dict:
        """
        Analyze an image using Gemini Vision.
        
        Args:
            image_path: Path to the image
            prompt: Analysis prompt
            
        Returns:
            Analysis results
        """
        try:
            # Prepare image
            image_data = self.prepare_image(image_path)
            
            # Generate content
            response = self.model.generate_content(
                [prompt, image_data['image']],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Extract response
            result = {
                'provider': 'gemini',
                'model': self.model.model_name,
                'content': response.text,
                'image_path': str(image_path),
                'prompt': prompt
            }
            
            # Add usage data if available
            if hasattr(response, 'usage_metadata'):
                result['usage'] = {
                    'prompt_tokens': response.usage_metadata.prompt_token_count,
                    'completion_tokens': response.usage_metadata.candidates_token_count,
                    'total_tokens': response.usage_metadata.total_token_count
                }
            
            # Check for safety blocks
            if response.prompt_feedback:
                result['safety_feedback'] = str(response.prompt_feedback)
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini Vision analysis failed: {e}")
            return {
                'provider': 'gemini',
                'error': str(e),
                'image_path': str(image_path)
            }
    
    def batch_analyze(self, images: List[Union[str, Path]], prompt: str) -> List[Dict]:
        """
        Analyze multiple images with Gemini Vision.
        
        Note: Gemini can process multiple images in a single prompt,
        making this more efficient than sequential processing.
        
        Args:
            images: List of image paths
            prompt: Analysis prompt
            
        Returns:
            List of analysis results
        """
        try:
            # Prepare all images
            prepared_images = []
            for image_path in images:
                image_data = self.prepare_image(image_path)
                prepared_images.append(image_data['image'])
            
            # Create content list
            content = [prompt]
            content.extend(prepared_images)
            
            # Add context about multiple images
            enhanced_prompt = f"{prompt}\n\nNote: Analyzing {len(images)} images. Please provide analysis for each image separately."
            content[0] = enhanced_prompt
            
            # Generate response
            response = self.model.generate_content(
                content,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Return consolidated result
            result = {
                'provider': 'gemini',
                'model': self.model.model_name,
                'content': response.text,
                'image_count': len(images),
                'image_paths': [str(p) for p in images],
                'prompt': prompt,
                'batch_mode': True
            }
            
            # Add usage data
            if hasattr(response, 'usage_metadata'):
                result['usage'] = {
                    'prompt_tokens': response.usage_metadata.prompt_token_count,
                    'completion_tokens': response.usage_metadata.candidates_token_count,
                    'total_tokens': response.usage_metadata.total_token_count
                }
            
            return [result]  # Return as list for consistency
            
        except Exception as e:
            logger.error(f"Gemini batch analysis failed: {e}")
            # Fall back to sequential processing
            results = []
            for image_path in images:
                result = self.analyze_image(image_path, prompt)
                results.append(result)
            return results