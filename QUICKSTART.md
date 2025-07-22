# PhotoSight Quick Start Guide

## Installation

1. **Install System Dependencies**
   ```bash
   # macOS
   brew install exiftool
   
   # Linux (Ubuntu/Debian)
   sudo apt-get install libimage-exiftool-perl
   ```

2. **Clone and Install PhotoSight**
   ```bash
   cd /Users/sam/dev/photosight
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Basic Usage

### Process RAW files with default settings
```bash
python cli.py --input ~/Pictures/RAW --output ~/Pictures/Processed
```

### Dry run to preview actions
```bash
python cli.py --input ~/Pictures/RAW --output ~/Pictures/Processed --dry-run
```

### Use custom configuration
```bash
python cli.py --input ~/Pictures/RAW --output ~/Pictures/Processed --config my-config.yaml
```

### Enable verbose logging
```bash
python cli.py --input ~/Pictures/RAW --output ~/Pictures/Processed --verbose
```

## Configuration

The default `config.yaml` contains sensible defaults, but you should adjust thresholds based on your camera and shooting style:

1. **Run baseline analysis** (when implemented):
   ```bash
   python -m photosight.research.analyze_baseline --input ./test_photos
   ```

2. **Update thresholds** in `config.yaml`:
   - `laplacian_variance_minimum`: Higher = stricter sharpness requirement
   - `maximum_iso`: Set based on your camera's acceptable noise level
   - `histogram_clip_threshold`: Lower = less tolerance for over/underexposure

## Output Structure

PhotoSight organizes your photos:
```
output/
├── accepted/          # Photos that passed all checks
└── rejected/
    ├── blurry/       # Failed sharpness check
    ├── underexposed/ # Too dark
    ├── overexposed/  # Too bright
    ├── high_iso/     # ISO too high
    └── slow_shutter/ # Shutter speed too slow
```

## Testing

Run the test script:
```bash
# Basic functionality test
python test_photosight.py

# Test with a real RAW file
python test_photosight.py /path/to/your/photo.ARW
```

## Tips

1. **Start with dry-run**: Always test with `--dry-run` first
2. **Use copy mode**: Keep `operation: "copy"` in config until confident
3. **Check logs**: Enable `--verbose` to see detailed processing info
4. **Adjust thresholds**: Run on a test set and adjust config based on results
5. **Preserve structure**: Keep `preserve_folder_structure: true` to maintain organization

## Next Steps

After Milestone 1 is working well:
1. Collect training data for AI models
2. Integrate YOLO for subject detection
3. Add MediaPipe for face/eye detection
4. Implement aesthetic scoring with NIMA

## Troubleshooting

- **"No RAW files found"**: Check file extensions in config match your camera
- **"ExifTool not found"**: Install exiftool via package manager
- **High rejection rate**: Lower thresholds in config.yaml
- **Slow processing**: Reduce `num_threads` if system is struggling