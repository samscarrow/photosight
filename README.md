# PhotoSight

An intelligent RAW photo processing pipeline with scene-aware analysis, AI curation, and non-destructive workflow.

## Overview

PhotoSight is a comprehensive RAW photo processing system that combines technical analysis with AI-powered curation to automatically process and enhance your photos. It features scene-aware processing that adapts to indoor vs outdoor scenes, intelligent straightening based on architectural or horizon references, and a non-destructive workflow that preserves your original files.

## Core Features

### ✅ Subject-Aware Smart Cropping
- **Person & Face Detection**: Automatically detects people and faces in photos
- **Intelligent Composition**: Suggests crops using rule of thirds and other composition principles
- **Multiple Aspect Ratios**: Supports various formats from square to cinematic
- **Confidence Scoring**: Ranks suggestions based on composition quality

### ✅ Intelligent Exposure Optimization
- **Histogram Analysis**: Multi-zone analysis with shadow/highlight detection
- **Dynamic Range Enhancement**: Optimizes tonal range while preserving detail
- **Scene-Aware Adjustments**: Adapts to low-key, high-key, and backlit scenes
- **Zone System Integration**: Based on Ansel Adams' zone system for precise control

### ✅ White Balance Correction
- **Multiple Algorithms**: Gray world, white patch, retinex, illuminant estimation, face detection
- **Scene-Aware Selection**: Automatically chooses best method based on content
- **Temperature & Tint Controls**: Fine-tune adjustments from -2000K to +2000K
- **Skin Tone Preservation**: Protects natural skin tones during correction

### ✅ Color Grading Engine
- **Creative Presets**: Cinematic, vintage, moody, bright & airy, teal-orange, and more
- **Three-Way Color Wheels**: Independent shadows/midtones/highlights control
- **HSL Color Mixer**: Selective hue, saturation, and luminance per color channel
- **Split Toning**: Different colors for highlights and shadows with balance control
- **Vibrance & Saturation**: Smart saturation that protects skin tones

### ✅ Scene-Aware Processing
- **Scene Classification**: Automatically detects indoor vs outdoor scenes
- **Adaptive Leveling**: Prioritizes architectural features for indoor scenes, horizon lines for outdoor
- **Processing Hints**: Scene-specific recommendations for color grading and exposure

### ✅ Intelligent Straightening
- **Multi-Method Detection**: Horizon lines, vertical references, grid patterns, and image moments
- **Confidence-Based**: Only applies corrections when highly confident
- **Architectural Priority**: Enhanced detection for picture frames, doors, and building lines

### ✅ Non-Destructive Processing
- **Recipe System**: All adjustments stored as JSON recipes, RAW files never modified
- **Iterative Previews**: Generate small preview JPEGs for quick approval
- **Full Export**: Only render full-size outputs when satisfied with preview

### ✅ AI-Powered Analysis
- **Regional Blur Detection**: Face-aware sharpness analysis avoiding global averaging
- **Composition Scoring**: Rule of thirds, symmetry, and visual balance analysis
- **Expression Detection**: Eyes open/closed, smile detection for portrait optimization
- **Similarity Clustering**: Groups similar photos and selects the best from each cluster

### ✅ Technical Quality Assessment
- **Advanced Sharpness**: Multi-region analysis with subject prioritization
- **Exposure Analysis**: Histogram-based clipping detection and dynamic range assessment
- **Metadata Integration**: Camera settings, lens information, and shooting conditions

### ✅ MCP Server Integration
- **Natural Language Queries**: AI assistants can search photos using everyday language
- **Project Management**: Query projects, tasks, and workflow status via AI
- **Analytics Access**: Generate insights about gear usage and shooting patterns
- **Secure Read-Only**: All AI operations are strictly read-only for data protection

## Installation

```bash
# Clone the repository
git clone https://github.com/samscarrow/photosight.git
cd photosight

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Processing
```bash
# Process photos with scene-aware detection
python -m photosight.processing.raw_processor --input ~/Pictures/RAW --output ~/Pictures/Processed

# Generate iterative previews
python process_raw_interactive.py ~/Pictures/RAW/photo.ARW

# Run AI curation on a folder
python -m photosight.analysis.ai.curator --input ~/Pictures/RAW --output ~/Pictures/Curated
```

### Scene-Aware Demo
```bash
# See the complete scene-aware processing pipeline in action
python scene_aware_processing_demo.py
```

### Smart Cropping Demo
```bash
# Demonstrate intelligent subject-aware cropping
python demo_smart_crop.py ~/Pictures/photo.jpg

# Process multiple images with comparison grid
python demo_smart_crop.py ~/Pictures --comparison
```

### Exposure Optimization Demo
```bash
# Analyze and optimize exposure for a single image
python demo_exposure_optimization.py ~/Pictures/photo.jpg

# Batch analyze exposure for multiple images
python demo_exposure_optimization.py ~/Pictures --batch
```

### Color Processing Demo
```bash
# Demonstrate white balance and color grading
python demo_color_processing.py ~/Pictures/photo.jpg

# Show all color adjustments and presets
python demo_color_processing.py ~/Pictures/photo.jpg --all

# Only test white balance methods
python demo_color_processing.py ~/Pictures/photo.jpg --wb-only
```

### MCP Server (AI Assistant Integration)
```bash
# Run the MCP server for AI assistant access
python -m photosight.mcp.server

# Configure for Claude Desktop - see docs/MCP_SERVER.md
```

## Architecture

### Scene-Aware Processing Pipeline
```
RAW Image → Scene Classification → Processing Hints → Adaptive Processing → Recipe → Preview/Export
     ↓              ↓                    ↓                   ↓              ↓         ↓
  ARW/DNG     Indoor/Outdoor      Leveling Methods    Geometry + Color    JSON    JPEG Output
```

### Core Modules

```
photosight/
├── processing/
│   ├── scene_classifier.py      # Indoor/outdoor scene detection
│   ├── raw_processor.py         # Non-destructive RAW processing
│   ├── geometry/
│   │   ├── horizon_detector.py  # Multi-method horizon/reference detection
│   │   ├── auto_straighten.py   # Scene-aware straightening
│   │   └── smart_crop.py        # Subject-aware intelligent cropping
│   ├── tone/
│   │   └── exposure_optimizer.py # Histogram-based exposure optimization
│   └── color/
│       ├── white_balance.py     # Multi-algorithm white balance correction
│       └── color_grading.py     # Creative color grading with presets
├── analysis/
│   ├── improved_blur_detection.py  # Regional sharpness with face priority
│   ├── technical.py             # Exposure and quality analysis
│   └── ai/
│       ├── curator.py           # AI-powered photo selection
│       ├── face_analysis.py     # Expression and quality detection
│       └── composition.py       # Rule of thirds and balance scoring
├── io/
│   ├── raw.py                   # RAW file handling and metadata
│   ├── photos_library.py        # macOS Photos Library integration
│   └── filesystem.py           # Safe file operations
└── utils/
    ├── logging.py               # Comprehensive logging system
    └── file_protection.py      # File safety and backup utilities
```

## Scene-Aware Processing

PhotoSight's key innovation is scene-aware processing that adapts its algorithms based on the content of your photos:

### Indoor Scenes
- **Leveling**: Prioritizes vertical references (door frames, picture frames, architectural lines)
- **Color**: Optimized for tungsten/fluorescent lighting (2700-4000K)
- **Exposure**: Enhanced shadow lifting for indoor lighting conditions
- **Features**: Skin tone protection, architectural detail enhancement

### Outdoor Scenes  
- **Leveling**: Prioritizes horizon line detection
- **Color**: Optimized for daylight conditions (5000-7000K)
- **Exposure**: Enhanced highlight recovery for bright outdoor scenes
- **Features**: Landscape color enhancement, natural contrast preservation

## Processing Recipes

All adjustments are stored as JSON recipes that preserve complete processing history:

```json
{
  "rotation_angle": -1.2,
  "exposure_adjustment": 0.3,
  "shadows": 25.0,
  "highlights": -15.0,
  "temperature_adjustment": -200,
  "scene_classification": {
    "classification": "indoor",
    "confidence": 0.85,
    "processing_hints": {...}
  }
}
```

## Configuration

Customize processing in `config.yaml`:

```yaml
# Scene classification
scene_classification:
  sky_threshold: 0.1
  edge_density_threshold: 0.005
  
# Straightening detection
geometry:
  confidence_threshold: 0.7
  max_rotation: 10.0
  
# AI curation
ai_curation:
  enabled: true
  min_face_size: 50
  composition_weight: 0.3
```

## Performance

- **Scene Classification**: ~50ms per image
- **Horizon Detection**: ~100ms per image  
- **Preview Generation**: ~200ms per image
- **AI Curation**: ~500ms per image
- **Batch Processing**: Supports parallel processing of large photo sets

## Development

### Running Tests
```bash
# Test scene classification
python test_scene_aware_detection.py

# Test horizon detection
python test_updated_detection.py

# Run comprehensive demo
python scene_aware_processing_demo.py
```

### Integration with Photos Library (macOS)
```bash
# Process directly from Photos Library exports
python -m photosight.io.photos_library --library ~/Pictures/Photos\ Library.photoslibrary
```

## Roadmap

- ✅ Scene-aware processing with indoor/outdoor classification
- ✅ Multi-method horizon and reference line detection  
- ✅ Non-destructive recipe-based processing
- ✅ Regional blur detection with face prioritization
- ✅ Subject-aware intelligent cropping with rule of thirds
- ✅ Advanced exposure optimization with shadow/highlight recovery
- ✅ White balance and color grading modules with creative presets
- 🚧 Batch processor for large photo collections

## License

MIT License - see LICENSE file for details

## Credits

Built with scene-aware intelligence and non-destructive processing principles. Designed for photographers who want intelligent automation without losing control over their creative process.

🤖 Generated with [Claude Code](https://claude.ai/code)