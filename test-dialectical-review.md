# Test Dialectical Review Integration

This file is created to test the Dialectical PR Reviewer GitHub App.

## Features to Review
- Image metadata extraction
- Performance optimization
- Security considerations
- User experience improvements

```javascript
// Example code for review
async function extractMetadata(imagePath) {
  const metadata = await exif.read(imagePath);
  return {
    camera: metadata.Make + ' ' + metadata.Model,
    settings: {
      iso: metadata.ISO,
      aperture: metadata.FNumber,
      shutter: metadata.ExposureTime
    }
  };
}
```

This PR tests the integration with the Team of Rivals review methodology.