# Image Handling System

```yaml
---
type: feature
tags: [images, pillow, document-generation, media]
created: 2025-07-22
updated: 2025-07-22
status: todo
up: "[[Document Generation System.md]]"
related: "[[Document Generation Pipeline.md]], [[AI Knowledge Base Query System.md]]"
dependencies: "[[Document Generation System.md]]"
---
```

## Purpose

This feature implements comprehensive image handling capabilities for the document generation system, enabling the processing, optimization, and embedding of images from various sources. Using Pillow for image manipulation, the system supports user-uploaded images, GitLab repository images, and potentially AI-generated visuals, ensuring they are properly formatted and optimized for inclusion in DOCX and PDF documents.

## Scope

- Image upload and validation from user interface
- Image retrieval from GitLab repositories
- Format conversion between common image types
- Automatic resizing and optimization for documents
- Image compression with quality preservation
- Metadata extraction and preservation
- Thumbnail generation for previews
- Batch image processing capabilities
- Error handling for corrupted images

## User Flow

1. User uploads image through chat or document interface
2. System validates image format and size
3. Image is processed and optimized for document use
4. Thumbnail generated for preview display
5. User positions image in document with caption
6. System embeds image maintaining aspect ratio
7. Final document includes properly formatted image
8. Original image stored for future reference

**Success State**: Images display correctly in documents with optimal quality/size balance

**Error Handling**: Clear format errors, automatic conversion attempts, fallback options

## Data Models

```yaml
ImageAsset:
  id: str
  original_filename: str
  format: str  # 'png' | 'jpg' | 'gif' | 'bmp' | 'tiff'
  source: str  # 'upload' | 'gitlab' | 'generated'
  source_url: str
  file_size: int
  dimensions: dict
    width: int
    height: int
  metadata: dict
    dpi: int
    color_mode: str
    exif: dict

ProcessedImage:
  asset_id: str
  format: str
  dimensions: dict
  compression: str
  file_path: str
  thumbnail_path: str
  processing_time: float

ImagePlacement:
  image_id: str
  document_id: str
  position: str  # 'inline' | 'float-left' | 'float-right' | 'center'
  size: str  # 'original' | 'fit-width' | 'custom'
  custom_dimensions: dict
  caption: str
  alt_text: str
```

## API Specification

```yaml
# Image Processing Service
class ImageService:
  async def process_upload(file: bytes, filename: str) -> ImageAsset:
    """Process uploaded image file"""
  
  async def retrieve_from_gitlab(project_id: int, path: str) -> ImageAsset:
    """Fetch image from GitLab repository"""
  
  async def optimize_for_document(asset: ImageAsset, target_format: str) -> ProcessedImage:
    """Optimize image for document embedding"""
  
  async def generate_thumbnail(asset: ImageAsset, size: tuple) -> str:
    """Create thumbnail for preview"""
  
  async def batch_process(assets: List[ImageAsset], options: dict) -> List[ProcessedImage]:
    """Process multiple images efficiently"""

# Pillow-based processors
class ImageProcessor:
  def convert_format(image: Image, target_format: str) -> Image:
    """Convert between image formats"""
  
  def resize_image(image: Image, max_size: tuple, maintain_aspect: bool) -> Image:
    """Resize with aspect ratio preservation"""
  
  def optimize_compression(image: Image, target_size: int) -> bytes:
    """Compress while maintaining quality"""
```

## Technical Implementation

### Core Components

- **ImageValidator**: `src/images/image_validator.py` - Format and size validation
- **ImageProcessor**: `src/images/image_processor.py` - Pillow-based processing
- **ThumbnailGenerator**: `src/images/thumbnail_generator.py` - Preview generation
- **ImageOptimizer**: `src/images/image_optimizer.py` - Compression and optimization
- **ImageStorage**: `src/images/image_storage.py` - File system management
- **MetadataExtractor**: `src/images/metadata_extractor.py` - EXIF and metadata handling

### Integration Points

- **Document Generation Pipeline**: Provides processed images for embedding
- **GitLab Repository Integration**: Retrieves images from repositories
- **Streamlit Interface**: Handles image uploads and previews

### Implementation Patterns

- **Pipeline Pattern**: Sequential image processing steps
- **Strategy Pattern**: Format-specific processing strategies
- **Cache Pattern**: Processed image caching for performance
- **Observer Pattern**: Progress updates for batch processing

## Examples

### Implementation References

- **[image-processing-example/](Examples/image-processing-example/)** - Complete image pipeline
- **[pillow-operations.py](Examples/pillow-operations.py)** - Common Pillow operations
- **[image-optimization.py](Examples/image-optimization.py)** - Optimization techniques
- **[batch-processing.py](Examples/batch-processing.py)** - Efficient batch handling

### Example Content Guidelines

- Show complete image processing pipeline
- Demonstrate format conversion examples
- Include optimization strategies
- Provide error handling patterns
- Show integration with documents

## Error Scenarios

- **Invalid Format**: Unsupported type → Attempt conversion → Error message
- **Corrupted File**: Can't read image → Show error → Suggest re-upload
- **Too Large**: Exceeds limits → Auto-resize → Warn about quality
- **Missing Image**: GitLab 404 → Placeholder → Log for review
- **Processing Failed**: Pillow error → Fallback processing → Basic embed

## Acceptance Criteria

- [ ] Support PNG, JPG, GIF, BMP, TIFF formats with conversion
- [ ] Automatically resize images over 2000px to document-friendly sizes
- [ ] Compress images to < 500KB while maintaining visual quality
- [ ] Generate thumbnails within 500ms for preview display
- [ ] Preserve image metadata including EXIF when relevant
- [ ] Batch process 50 images in under 30 seconds
- [ ] Handle corrupted images gracefully with clear errors
- [ ] Support transparency in PNG images for documents

## Validation

### Testing Strategy

- **Unit Tests**: Test individual processing functions, format conversions
- **Integration Tests**: Full pipeline with various image types
- **Performance Tests**: Processing speed and memory usage
- **Visual Tests**: Quality comparison before/after processing

### Verification Commands

```bash
# Test image processing
python -m src.images.image_processor test --all-formats

# Run visual quality tests
python -m tests.visual.image_quality_test

# Performance benchmark
python -m tests.performance.image_benchmark

# Memory profiling
mprof run python -m src.images.batch_processor test

# Format compatibility test
pytest tests/images/test_format_support.py
```

### Success Metrics

- Processing Speed: < 1s per image average
- Compression Ratio: 60-80% size reduction
- Quality Score: > 0.95 SSIM after optimization
- Format Support: 100% common formats
- Batch Efficiency: Linear scaling to 100 images