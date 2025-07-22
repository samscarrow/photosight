"""
Perceptual similarity detection for identifying near-duplicate photos
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Set
import logging
from pathlib import Path
import hashlib
from collections import defaultdict
import imagehash
from PIL import Image
import rawpy

logger = logging.getLogger(__name__)


class SimilarityDetector:
    """
    Detect perceptually similar photos using multiple hash algorithms
    and metadata comparison for robust duplicate detection
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.99,
                 hash_size: int = 16,
                 enable_metadata_check: bool = True):
        """
        Initialize similarity detector
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider duplicates
            hash_size: Size of perceptual hashes (larger = more precise)
            enable_metadata_check: Use metadata to pre-filter candidates
        """
        self.similarity_threshold = similarity_threshold
        self.hash_size = hash_size
        self.enable_metadata_check = enable_metadata_check
        
        # Convert similarity threshold to hamming distance
        # For 99% similarity with 16x16 hash: allow ~2-3 bit differences
        max_bits = hash_size * hash_size
        self.max_hamming_distance = int((1.0 - similarity_threshold) * max_bits)
        
        logger.info(f"Similarity detector initialized: threshold={similarity_threshold}, "
                   f"max_hamming={self.max_hamming_distance}")
        
    def compute_image_hashes(self, image_path: Path) -> Dict[str, str]:
        """
        Compute multiple perceptual hashes for an image
        
        Args:
            image_path: Path to RAW image file
            
        Returns:
            Dictionary of hash algorithm names to hash strings
        """
        try:
            # Load RAW image
            if str(image_path).lower().endswith('.arw'):
                with rawpy.imread(str(image_path)) as raw:
                    # Use embedded JPEG for speed, or postprocess at low quality
                    try:
                        rgb = raw.extract_thumb()
                        if rgb.format == rawpy.ThumbFormat.JPEG:
                            # Convert JPEG bytes to PIL Image
                            from io import BytesIO
                            pil_image = Image.open(BytesIO(rgb.data))
                        else:
                            # Fallback to postprocessing
                            rgb_array = raw.postprocess(
                                use_camera_wb=True,
                                no_auto_bright=True,
                                output_bps=8,
                                bright=1.0,  # Lower quality for speed
                                gamma=(2.222, 4.5)
                            )
                            pil_image = Image.fromarray(rgb_array)
                    except:
                        # Fallback to full postprocessing
                        rgb_array = raw.postprocess(
                            use_camera_wb=True,
                            no_auto_bright=True,
                            output_bps=8
                        )
                        pil_image = Image.fromarray(rgb_array)
            else:
                pil_image = Image.open(image_path)
            
            # Compute multiple types of perceptual hashes
            hashes = {}
            
            # pHash - most robust for similar images with slight differences
            hashes['phash'] = str(imagehash.phash(pil_image, hash_size=self.hash_size))
            
            # dHash - good for detecting crops and minor edits
            hashes['dhash'] = str(imagehash.dhash(pil_image, hash_size=self.hash_size))
            
            # Average hash - fast, good for exact duplicates
            hashes['ahash'] = str(imagehash.average_hash(pil_image, hash_size=self.hash_size))
            
            # Wavelet hash - good for different compression/quality
            hashes['whash'] = str(imagehash.whash(pil_image, hash_size=self.hash_size))
            
            return hashes
            
        except Exception as e:
            logger.error(f"Error computing hashes for {image_path}: {e}")
            return {}
            
    def compute_metadata_fingerprint(self, image_path: Path) -> Optional[str]:
        """
        Compute metadata fingerprint for pre-filtering candidates
        
        Args:
            image_path: Path to RAW image file
            
        Returns:
            Metadata fingerprint string or None if unavailable
        """
        if not self.enable_metadata_check:
            return None
            
        try:
            with rawpy.imread(str(image_path)) as raw:
                # Use camera metadata that should be identical for burst shots
                metadata_items = []
                
                # Camera model and settings
                if hasattr(raw, 'camera_whitebalance'):
                    metadata_items.extend([str(x) for x in raw.camera_whitebalance])
                
                # Image dimensions (should be identical)
                metadata_items.extend([
                    str(raw.sizes.width),
                    str(raw.sizes.height),
                    str(raw.sizes.iwidth), 
                    str(raw.sizes.iheight)
                ])
                
                # Color matrix (should be identical for same camera/settings)
                if hasattr(raw, 'color_matrix'):
                    metadata_items.extend([str(x) for x in raw.color_matrix.flatten()])
                    
                # Create fingerprint
                fingerprint_data = '|'.join(metadata_items)
                return hashlib.md5(fingerprint_data.encode()).hexdigest()
                
        except Exception as e:
            logger.debug(f"Could not extract metadata fingerprint from {image_path}: {e}")
            return None
    
    def compare_hashes(self, hashes1: Dict[str, str], hashes2: Dict[str, str]) -> float:
        """
        Compare two sets of perceptual hashes and return similarity score
        
        Args:
            hashes1: First image's hashes
            hashes2: Second image's hashes
            
        Returns:
            Similarity score (0-1), where 1 is identical
        """
        if not hashes1 or not hashes2:
            return 0.0
            
        similarities = []
        
        for hash_type in ['phash', 'dhash', 'ahash', 'whash']:
            if hash_type in hashes1 and hash_type in hashes2:
                # Convert hex strings to imagehash objects
                try:
                    hash1 = imagehash.hex_to_hash(hashes1[hash_type])
                    hash2 = imagehash.hex_to_hash(hashes2[hash_type])
                    
                    # Compute hamming distance
                    distance = hash1 - hash2
                    max_distance = self.hash_size * self.hash_size
                    
                    # Convert to similarity score
                    similarity = 1.0 - (distance / max_distance)
                    similarities.append(similarity)
                    
                except Exception as e:
                    logger.debug(f"Error comparing {hash_type} hashes: {e}")
                    continue
        
        if not similarities:
            return 0.0
            
        # Use weighted average, prioritizing pHash for photographic content
        weights = {
            0: 0.4,  # phash (most important for photos)
            1: 0.3,  # dhash (good for minor changes) 
            2: 0.2,  # ahash (fast comparison)
            3: 0.1   # whash (compression robustness)
        }
        
        weighted_similarity = sum(
            sim * weights.get(i, 1.0/len(similarities))
            for i, sim in enumerate(similarities)
        )
        
        return weighted_similarity
    
    def find_similar_groups(self, image_paths: List[Path]) -> List[List[Path]]:
        """
        Group images by similarity
        
        Args:
            image_paths: List of image paths to analyze
            
        Returns:
            List of groups, where each group contains similar images
        """
        logger.info(f"Finding similar groups among {len(image_paths)} images")
        
        # Compute hashes and metadata for all images
        image_data = {}
        metadata_groups = defaultdict(list)
        
        for path in image_paths:
            hashes = self.compute_image_hashes(path)
            if not hashes:
                continue
                
            metadata_fp = self.compute_metadata_fingerprint(path)
            
            image_data[path] = {
                'hashes': hashes,
                'metadata_fingerprint': metadata_fp
            }
            
            # Group by metadata fingerprint for efficiency
            if metadata_fp:
                metadata_groups[metadata_fp].append(path)
            else:
                metadata_groups['unknown'].append(path)
        
        logger.info(f"Computed hashes for {len(image_data)} images")
        logger.info(f"Found {len(metadata_groups)} metadata groups")
        
        # Find similar images within each metadata group
        similar_groups = []
        
        for group_paths in metadata_groups.values():
            if len(group_paths) < 2:
                continue
                
            # Compare all pairs within the metadata group
            used_images = set()
            
            for i, path1 in enumerate(group_paths):
                if path1 in used_images or path1 not in image_data:
                    continue
                    
                current_group = [path1]
                used_images.add(path1)
                
                for path2 in group_paths[i+1:]:
                    if path2 in used_images or path2 not in image_data:
                        continue
                    
                    # Compare hashes
                    similarity = self.compare_hashes(
                        image_data[path1]['hashes'],
                        image_data[path2]['hashes']
                    )
                    
                    if similarity >= self.similarity_threshold:
                        current_group.append(path2)
                        used_images.add(path2)
                        logger.debug(f"Similar images found: {path1.name} <-> {path2.name} "
                                   f"(similarity: {similarity:.3f})")
                
                # Only keep groups with multiple images
                if len(current_group) > 1:
                    similar_groups.append(current_group)
        
        logger.info(f"Found {len(similar_groups)} similarity groups")
        for i, group in enumerate(similar_groups):
            logger.info(f"  Group {i+1}: {len(group)} images")
            
        return similar_groups
    
    def analyze_temporal_sequence(self, image_paths: List[Path]) -> Dict[str, any]:
        """
        Analyze temporal patterns in image sequences to identify burst shots
        
        Args:
            image_paths: List of image paths (should be sorted by filename/timestamp)
            
        Returns:
            Analysis results including burst detection
        """
        if len(image_paths) < 2:
            return {'is_burst_sequence': False, 'burst_groups': []}
            
        # Extract sequence numbers from filenames (e.g., DSC04123.ARW -> 4123)
        sequence_numbers = []
        for path in image_paths:
            try:
                # Extract number from filename like DSC04123.ARW
                filename = path.stem
                number = int(''.join(filter(str.isdigit, filename)))
                sequence_numbers.append(number)
            except:
                sequence_numbers.append(0)
        
        # Identify consecutive sequences (likely burst shots)
        burst_groups = []
        current_burst = []
        
        for i, (path, seq_num) in enumerate(zip(image_paths, sequence_numbers)):
            if not current_burst:
                current_burst = [path]
            elif (seq_num - sequence_numbers[i-1]) <= 3:  # Allow small gaps
                current_burst.append(path)
            else:
                if len(current_burst) > 1:
                    burst_groups.append(current_burst)
                current_burst = [path]
        
        # Don't forget the last group
        if len(current_burst) > 1:
            burst_groups.append(current_burst)
            
        return {
            'is_burst_sequence': len(burst_groups) > 0,
            'burst_groups': burst_groups,
            'total_bursts': len(burst_groups)
        }