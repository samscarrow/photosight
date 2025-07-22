# PhotoSight

An intelligent RAW photo processing pipeline for automatically culling and curating photos from Sony A7III cameras.

## Overview

PhotoSight uses a two-stage approach to automatically process your RAW photos:

1. **Technical Culling**: Fast, objective filtering based on exposure, sharpness, and metadata
2. **AI Curation**: Intelligent analysis using computer vision to identify the best shots

## Features

### Milestone 1 ✅
- ✅ Automatic detection and processing of Sony .ARW files
- ✅ Technical quality filtering (exposure, sharpness, ISO)
- ✅ Configurable thresholds via YAML
- ✅ Non-destructive workflow (moves files to organized folders)
- ✅ Dry-run mode for safe testing
- ✅ Progress tracking and statistics

### Milestone 2 ✅
- ✅ AI-powered person detection (YOLOv8)
- ✅ Face quality analysis (MediaPipe)
- ✅ Advanced composition analysis (rule of thirds, symmetry, balance)
- ✅ Expression detection (eyes open, smiles)
- ✅ Visual balance and color harmony analysis

### Planned Features
- 🚧 NIMA aesthetic scoring
- 🚧 GPU acceleration optimization
- 🚧 Lightroom integration
- 🚧 Custom model training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/photosight.git
cd photosight

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For AI features, also install AI dependencies
pip install -r requirements-ai.txt
```

## Quick Start

```bash
# Process photos with default settings
python cli.py --input ~/Pictures/RAW --output ~/Pictures/Processed

# Dry run to preview what would happen
python cli.py --input ~/Pictures/RAW --output ~/Pictures/Processed --dry-run

# Use custom configuration
python cli.py --input ~/Pictures/RAW --output ~/Pictures/Processed --config my-config.yaml

# Enable AI curation
python cli.py --input ~/Pictures/RAW --output ~/Pictures/Processed --enable-ai
```

## Configuration

Edit `config.yaml` to customize processing thresholds:

```yaml
# Technical filtering
technical_filters:
  sharpness:
    laplacian_variance_minimum: 100.0
  exposure:
    histogram_clip_threshold: 0.01
  metadata:
    maximum_iso: 12800

# AI curation (optional)
ai_curation:
  enabled: false  # Set to true to enable AI
  yolo_model: "yolov8n.pt"  # Faster: yolov8n.pt, More accurate: yolov8l.pt
  min_ai_score: 0.5
```

See [AI_CURATION.md](docs/AI_CURATION.md) for detailed AI configuration options.

## Project Structure

```
photosight/
├── cli.py              # Command-line interface
├── config.yaml         # Default configuration
├── io/                 # File I/O operations
│   ├── filesystem.py   # File management
│   └── raw.py         # RAW file processing
├── analysis/          # Image analysis
│   ├── technical.py   # Technical quality checks
│   └── ai/           # AI-powered analysis
│       ├── curator.py    # Main AI curator
│       ├── person_detection.py  # YOLO person detection
│       ├── face_analysis.py     # MediaPipe face analysis
│       └── composition.py       # Composition analysis
└── utils/             # Utilities
    └── logging.py     # Logging configuration
```

## Development

### Running Tests

```bash
pytest tests/
```

### Phase 0: Baseline Research

Before using PhotoSight, you should:

1. Create a test dataset of 100-200 representative photos
2. Run the baseline analysis script to determine optimal thresholds
3. Update `config.yaml` with your camera-specific values

```bash
python -m photosight.research.analyze_baseline --input ./test_photos
```

## Performance

- Processes ~100 images in 30 seconds (Stage 1 only)
- Non-destructive: original files are never modified
- Preserves .xmp sidecar files with moved images

## License

MIT License - see LICENSE file for details