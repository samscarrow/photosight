# Smart Git Push with Claude Code Review

Execute the photosight code review workflow:
1. Create feature branch from current changes
2. Commit with descriptive message
3. Push branch and create PR for Claude Code Review
4. Provide PR link for review results

This ensures all pushes get automated code review for computer vision and AI model code.

## Photosight-Specific Checks
- AI model file validation (.pkl, .pt, .onnx)
- Image processing pipeline safety
- Privacy-preserving crop suggestions
- No test images or personal photos in commits