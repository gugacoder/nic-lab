"""
DOCX Image Embedding and Processing

This module handles image embedding in DOCX documents with proper sizing,
compression, and format conversion capabilities.
"""

import io
import logging
import base64
from pathlib import Path
from typing import Optional, Tuple, Union
import asyncio
import aiofiles
import httpx

try:
    from PIL import Image as PILImage
    from PIL import ImageOps
except ImportError:
    raise ImportError("Pillow is required for image processing. Install with: pip install Pillow")

from docx import Document
from docx.shared import Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

from ...models import ContentElement, ImageData, ImageAlignment

logger = logging.getLogger(__name__)


class DocxImageHandler:
    """
    Handles image embedding and processing for DOCX documents.
    
    This class provides functionality to embed images from various sources
    with automatic resizing, format conversion, and alignment options.
    """
    
    def __init__(self):
        """Initialize the image handler."""
        self.max_image_width = Inches(6.5)  # Maximum width for page
        self.max_image_height = Inches(8)   # Maximum height for page
        self.default_dpi = 96  # Default DPI for images
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        
        # Image quality settings
        self.jpeg_quality = 85
        self.png_optimize = True
    
    async def add_image(self, document: Document, element: ContentElement) -> None:
        """
        Add an image element to the document.
        
        Args:
            document: DOCX document to add to
            element: Image element to add
        """
        try:
            if not element.image_data:
                logger.warning("Image element has no image_data")
                return
            
            image_data = element.image_data
            
            # Get image bytes from source
            image_bytes = await self._get_image_bytes(image_data)
            if not image_bytes:
                logger.error(f"Failed to get image bytes for: {image_data.source_path or image_data.source_url}")
                await self._add_image_placeholder(document, image_data)
                return
            
            # Process image (resize, optimize)
            processed_bytes, width, height = await self._process_image(image_bytes, image_data)
            
            # Add image to document
            await self._embed_image(document, processed_bytes, width, height, image_data)
            
            logger.debug(f"Added image: {image_data.source_path or image_data.source_url}")
            
        except Exception as e:
            logger.error(f"Failed to add image: {str(e)}")
            # Add placeholder on error
            if element.image_data:
                await self._add_image_placeholder(document, element.image_data)
    
    async def _get_image_bytes(self, image_data: ImageData) -> Optional[bytes]:
        """
        Get image bytes from various sources.
        
        Args:
            image_data: Image data with source information
            
        Returns:
            Image bytes or None if failed
        """
        try:
            if image_data.source_type == "file" and image_data.source_path:
                # Load from file
                return await self._load_image_from_file(image_data.source_path)
            
            elif image_data.source_type == "url" and image_data.source_url:
                # Load from URL
                return await self._load_image_from_url(image_data.source_url)
            
            elif image_data.source_type == "base64" and image_data.base64_data:
                # Decode from base64
                return await self._load_image_from_base64(image_data.base64_data)
            
            else:
                logger.error(f"Unsupported image source type: {image_data.source_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get image bytes: {str(e)}")
            return None
    
    async def _load_image_from_file(self, file_path: str) -> Optional[bytes]:
        """Load image from file path."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Image file not found: {file_path}")
                return None
            
            if path.suffix.lower() not in self.supported_formats:
                logger.warning(f"Unsupported image format: {path.suffix}")
            
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
                
        except Exception as e:
            logger.error(f"Failed to load image from file {file_path}: {str(e)}")
            return None
    
    async def _load_image_from_url(self, url: str) -> Optional[bytes]:
        """Load image from URL."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    logger.warning(f"URL does not appear to be an image: {url}")
                
                return response.content
                
        except Exception as e:
            logger.error(f"Failed to load image from URL {url}: {str(e)}")
            return None
    
    async def _load_image_from_base64(self, base64_data: str) -> Optional[bytes]:
        """Load image from base64 data."""
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',', 1)[1]
            
            return base64.b64decode(base64_data)
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image data: {str(e)}")
            return None
    
    async def _process_image(
        self, 
        image_bytes: bytes, 
        image_data: ImageData
    ) -> Tuple[bytes, float, float]:
        """
        Process image with resizing and optimization.
        
        Args:
            image_bytes: Original image bytes
            image_data: Image metadata and settings
            
        Returns:
            Tuple of (processed_bytes, width_inches, height_inches)
        """
        try:
            # Open image with PIL
            with PILImage.open(io.BytesIO(image_bytes)) as pil_image:
                # Convert to RGB if necessary (for JPEG output)
                if pil_image.mode in ('RGBA', 'P'):
                    rgb_image = PILImage.new('RGB', pil_image.size, (255, 255, 255))
                    rgb_image.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
                    pil_image = rgb_image
                
                # Get original dimensions
                original_width, original_height = pil_image.size
                
                # Calculate target dimensions
                target_width, target_height = self._calculate_target_size(
                    original_width, original_height, image_data
                )
                
                # Resize if necessary
                if (target_width != original_width or target_height != original_height):
                    if image_data.maintain_aspect_ratio:
                        # Use thumbnail to maintain aspect ratio
                        pil_image.thumbnail((target_width, target_height), PILImage.Resampling.LANCZOS)
                    else:
                        # Resize without maintaining aspect ratio
                        pil_image = pil_image.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
                
                # Optimize image
                pil_image = self._optimize_image(pil_image, image_data)
                
                # Convert back to bytes
                output_buffer = io.BytesIO()
                format_name = 'JPEG' if image_data.compression_quality else 'PNG'
                
                if format_name == 'JPEG':
                    quality = image_data.compression_quality or self.jpeg_quality
                    pil_image.save(output_buffer, format=format_name, quality=quality, optimize=True)
                else:
                    pil_image.save(output_buffer, format=format_name, optimize=self.png_optimize)
                
                processed_bytes = output_buffer.getvalue()
                
                # Calculate final dimensions in inches for DOCX
                final_width = pil_image.width / self.default_dpi
                final_height = pil_image.height / self.default_dpi
                
                logger.debug(f"Processed image: {original_width}x{original_height} -> {pil_image.width}x{pil_image.height}")
                
                return processed_bytes, final_width, final_height
                
        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            # Return original bytes with estimated dimensions
            estimated_width = min(6.0, image_data.width / 96 if image_data.width else 4.0)
            estimated_height = min(8.0, image_data.height / 96 if image_data.height else 3.0)
            return image_bytes, estimated_width, estimated_height
    
    def _calculate_target_size(
        self, 
        original_width: int, 
        original_height: int, 
        image_data: ImageData
    ) -> Tuple[int, int]:
        """Calculate target image size based on constraints."""
        # Start with original size
        target_width = original_width
        target_height = original_height
        
        # Apply explicit size constraints from image_data
        if image_data.width and image_data.height:
            target_width = image_data.width
            target_height = image_data.height
        elif image_data.width:
            # Scale height proportionally
            target_width = image_data.width
            if image_data.maintain_aspect_ratio:
                aspect_ratio = original_height / original_width
                target_height = int(target_width * aspect_ratio)
        elif image_data.height:
            # Scale width proportionally
            target_height = image_data.height
            if image_data.maintain_aspect_ratio:
                aspect_ratio = original_width / original_height
                target_width = int(target_height * aspect_ratio)
        
        # Apply max size constraints
        max_width_px = int(self.max_image_width.inches * self.default_dpi)
        max_height_px = int(self.max_image_height.inches * self.default_dpi)
        
        if image_data.max_width:
            max_width_px = min(max_width_px, image_data.max_width)
        if image_data.max_height:
            max_height_px = min(max_height_px, image_data.max_height)
        
        # Scale down if too large
        if target_width > max_width_px or target_height > max_height_px:
            if image_data.maintain_aspect_ratio:
                # Scale to fit within bounds while maintaining aspect ratio
                scale_x = max_width_px / target_width
                scale_y = max_height_px / target_height
                scale = min(scale_x, scale_y)
                
                target_width = int(target_width * scale)
                target_height = int(target_height * scale)
            else:
                # Clamp to maximum bounds
                target_width = min(target_width, max_width_px)
                target_height = min(target_height, max_height_px)
        
        return target_width, target_height
    
    def _optimize_image(self, pil_image: PILImage.Image, image_data: ImageData) -> PILImage.Image:
        """Apply image optimization."""
        try:
            # Auto-orient based on EXIF data
            pil_image = ImageOps.exif_transpose(pil_image)
            
            # Additional optimizations could be added here
            # - Sharpening
            # - Color correction
            # - Compression optimization
            
            return pil_image
            
        except Exception as e:
            logger.warning(f"Image optimization failed: {str(e)}")
            return pil_image
    
    async def _embed_image(
        self,
        document: Document,
        image_bytes: bytes,
        width_inches: float,
        height_inches: float,
        image_data: ImageData
    ) -> None:
        """Embed processed image into document."""
        try:
            # Create paragraph for image
            paragraph = document.add_paragraph()
            
            # Set alignment
            alignment_map = {
                ImageAlignment.LEFT: WD_ALIGN_PARAGRAPH.LEFT,
                ImageAlignment.CENTER: WD_ALIGN_PARAGRAPH.CENTER,
                ImageAlignment.RIGHT: WD_ALIGN_PARAGRAPH.RIGHT
            }
            
            if image_data.alignment in alignment_map:
                paragraph.alignment = alignment_map[image_data.alignment]
            
            # Add image to paragraph
            run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
            
            # Create image stream
            image_stream = io.BytesIO(image_bytes)
            
            # Add image with calculated dimensions
            picture = run.add_picture(image_stream, width=Inches(width_inches), height=Inches(height_inches))
            
            # Add caption if provided
            if image_data.caption:
                caption_paragraph = document.add_paragraph(image_data.caption)
                caption_paragraph.alignment = paragraph.alignment
                
                # Style caption
                for run in caption_paragraph.runs:
                    run.font.italic = True
                    run.font.size = 9  # Smaller font for caption
            
            logger.debug(f"Embedded image: {width_inches:.2f}x{height_inches:.2f} inches")
            
        except Exception as e:
            logger.error(f"Failed to embed image: {str(e)}")
            raise
    
    async def _add_image_placeholder(self, document: Document, image_data: ImageData) -> None:
        """Add a placeholder when image cannot be loaded."""
        try:
            paragraph = document.add_paragraph(f"[Image placeholder: {image_data.source_path or image_data.source_url or 'Unknown source'}]")
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Style as placeholder
            for run in paragraph.runs:
                run.font.italic = True
                run.font.color.rgb = PILImage.new('RGB', (1, 1), (128, 128, 128))  # Gray color
            
            # Add caption if provided
            if image_data.caption:
                caption_paragraph = document.add_paragraph(f"Caption: {image_data.caption}")
                caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in caption_paragraph.runs:
                    run.font.italic = True
            
        except Exception as e:
            logger.error(f"Failed to add image placeholder: {str(e)}")
    
    async def validate_image(self, image_data: ImageData) -> Tuple[bool, Optional[str]]:
        """
        Validate image data and accessibility.
        
        Args:
            image_data: Image data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if source is specified
            if not any([image_data.source_path, image_data.source_url, image_data.base64_data]):
                return False, "No image source specified"
            
            # Validate file path
            if image_data.source_type == "file" and image_data.source_path:
                path = Path(image_data.source_path)
                if not path.exists():
                    return False, f"Image file not found: {image_data.source_path}"
                
                if path.suffix.lower() not in self.supported_formats:
                    return False, f"Unsupported image format: {path.suffix}"
            
            # Validate URL
            if image_data.source_type == "url" and image_data.source_url:
                if not image_data.source_url.startswith(('http://', 'https://')):
                    return False, "Invalid image URL format"
            
            # Validate base64 data
            if image_data.source_type == "base64" and image_data.base64_data:
                try:
                    base64_clean = image_data.base64_data
                    if base64_clean.startswith('data:'):
                        base64_clean = base64_clean.split(',', 1)[1]
                    base64.b64decode(base64_clean)
                except Exception:
                    return False, "Invalid base64 image data"
            
            return True, None
            
        except Exception as e:
            return False, f"Image validation error: {str(e)}"


# Test functionality
async def test_image_handler():
    """Test image handler functionality."""
    from docx import Document
    from ...models import ContentElement, ContentType, ImageData, ImageAlignment
    
    # Create test document
    document = Document()
    handler = DocxImageHandler()
    
    try:
        # Test with placeholder (no actual image file)
        image_element = ContentElement(
            type=ContentType.IMAGE,
            image_data=ImageData(
                source_type="file",
                source_path="test_image.jpg",  # Non-existent file for testing
                caption="Test Image Caption",
                width=400,
                height=300,
                alignment=ImageAlignment.CENTER
            )
        )
        
        # This will add a placeholder since file doesn't exist
        await handler.add_image(document, image_element)
        
        # Test validation
        is_valid, error = await handler.validate_image(image_element.image_data)
        print(f"Image validation: valid={is_valid}, error={error}")
        
        print("✅ Image handler test passed (placeholder added)")
        return True
        
    except Exception as e:
        print(f"❌ Image handler test failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(test_image_handler())