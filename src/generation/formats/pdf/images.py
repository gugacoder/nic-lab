"""
PDF Image Handler

This module handles image processing and embedding for PDF generation
using ReportLab, including image compression, sizing, and positioning.
"""

import logging
import io
import base64
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse
import asyncio

try:
    from PIL import Image as PILImage
    from PIL import ImageOps
except ImportError:
    raise ImportError("Pillow is required for image processing. Install with: pip install Pillow")

from reportlab.platypus import Image, Spacer, KeepTogether, Paragraph
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib import colors

from ...models import ContentElement, ImageData, ImageAlignment

logger = logging.getLogger(__name__)


class PdfImageHandler:
    """
    Handles image processing and embedding for PDF generation.
    
    This class manages image loading, compression, sizing, and positioning
    for optimal PDF output while maintaining quality and file size balance.
    """
    
    def __init__(self):
        """Initialize the PDF image handler."""
        # Default compression settings
        self.default_quality = 85
        self.max_width = 6 * inch  # Maximum image width in PDFs
        self.max_height = 8 * inch  # Maximum image height in PDFs
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        # Image cache for optimization
        self._image_cache = {}
    
    async def add_image(self, story: List, element: ContentElement) -> None:
        """
        Add an image element to the PDF story.
        
        Args:
            story: List of ReportLab flowables
            element: Image content element
        """
        try:
            if not element.image_data:
                logger.warning("Image element has no image data")
                await self._add_placeholder_image(story, "Missing image data")
                return
            
            # Load and process the image
            image_path, processed_image = await self._load_and_process_image(element.image_data)
            
            if not processed_image:
                logger.warning(f"Failed to load image from {element.image_data.source_path or element.image_data.source_url}")
                await self._add_placeholder_image(story, "Failed to load image")
                return
            
            # Calculate dimensions
            width, height = await self._calculate_dimensions(processed_image, element.image_data)
            
            # Create ReportLab Image
            if image_path:
                # Use file path if available (more efficient)
                rl_image = Image(image_path, width=width, height=height)
            else:
                # Use PIL image data
                img_buffer = io.BytesIO()
                processed_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                rl_image = Image(img_buffer, width=width, height=height)
            
            # Apply alignment
            image_flowable = await self._apply_alignment(rl_image, element.image_data.alignment)
            
            # Add caption if provided
            flowables = []
            
            # Add spacing before image
            flowables.append(Spacer(1, 6))
            
            # Add the image
            flowables.append(image_flowable)
            
            # Add caption if provided
            if element.image_data.caption:
                caption_style = ParagraphStyle(
                    'Caption',
                    fontName='Helvetica',
                    fontSize=9,
                    alignment=self._get_alignment_enum(element.image_data.alignment),
                    spaceBefore=3,
                    spaceAfter=6,
                    textColor=colors.grey
                )
                caption = Paragraph(f"<i>{self._escape_text(element.image_data.caption)}</i>", caption_style)
                flowables.append(caption)
            
            # Add spacing after image
            flowables.append(Spacer(1, 6))
            
            # Keep image and caption together
            story.append(KeepTogether(flowables))
            
            logger.debug(f"Added image: {width:.1f}x{height:.1f} points")
            
        except Exception as e:
            logger.error(f"Failed to add image: {e}")
            await self._add_placeholder_image(story, f"Error: {str(e)}")
    
    async def _load_and_process_image(self, image_data: ImageData) -> Tuple[Optional[str], Optional[PILImage.Image]]:
        """
        Load and process an image from various sources.
        
        Args:
            image_data: Image data configuration
            
        Returns:
            Tuple of (file_path, PIL_Image) - file_path is None for non-file sources
        """
        try:
            pil_image = None
            file_path = None
            
            # Load image based on source type
            if image_data.source_type == "file" and image_data.source_path:
                file_path = image_data.source_path
                if Path(file_path).exists():
                    pil_image = PILImage.open(file_path)
                else:
                    logger.error(f"Image file not found: {file_path}")
                    return None, None
            
            elif image_data.source_type == "url" and image_data.source_url:
                # For URLs, we'd need to download the image
                # For now, we'll handle this as a TODO
                logger.warning("URL image loading not yet implemented")
                return None, None
            
            elif image_data.source_type == "base64" and image_data.base64_data:
                # Decode base64 image
                try:
                    image_bytes = base64.b64decode(image_data.base64_data)
                    pil_image = PILImage.open(io.BytesIO(image_bytes))
                except Exception as e:
                    logger.error(f"Failed to decode base64 image: {e}")
                    return None, None
            
            elif image_data.source_type == "generated":
                # Handle AI-generated images
                logger.warning("Generated image handling not yet implemented")
                return None, None
            
            else:
                logger.error(f"Unsupported image source type: {image_data.source_type}")
                return None, None
            
            if not pil_image:
                return None, None
            
            # Process the image
            processed_image = await self._process_image(pil_image, image_data)
            
            return file_path, processed_image
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None, None
    
    async def _process_image(self, pil_image: PILImage.Image, image_data: ImageData) -> PILImage.Image:
        """
        Process the PIL image for optimal PDF embedding.
        
        Args:
            pil_image: Original PIL image
            image_data: Image configuration
            
        Returns:
            Processed PIL image
        """
        try:
            # Convert to RGB if necessary (PDFs work best with RGB)
            if pil_image.mode not in ('RGB', 'RGBA'):
                if pil_image.mode == 'P':
                    # Convert palette images to RGB
                    pil_image = pil_image.convert('RGB')
                else:
                    pil_image = pil_image.convert('RGB')
            
            # Handle transparency for RGBA images
            if pil_image.mode == 'RGBA':
                # Create a white background and paste the image onto it
                background = PILImage.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[-1])  # Use alpha channel as mask
                pil_image = background
            
            # Apply image compression if specified
            quality = image_data.compression_quality or self.default_quality
            if quality < 100:
                # Re-compress the image to reduce file size
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
                buffer.seek(0)
                pil_image = PILImage.open(buffer)
            
            # Auto-rotate based on EXIF data
            try:
                pil_image = ImageOps.exif_transpose(pil_image)
            except Exception:
                # Ignore EXIF errors
                pass
            
            return pil_image
            
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return pil_image  # Return original on error
    
    async def _calculate_dimensions(
        self, 
        pil_image: PILImage.Image, 
        image_data: ImageData
    ) -> Tuple[float, float]:
        """
        Calculate optimal dimensions for the image in the PDF.
        
        Args:
            pil_image: PIL image
            image_data: Image configuration
            
        Returns:
            Tuple of (width, height) in points
        """
        # Get original dimensions
        orig_width, orig_height = pil_image.size
        
        # Convert pixels to points (assuming 72 DPI)
        dpi = 72
        orig_width_pts = (orig_width / dpi) * inch
        orig_height_pts = (orig_height / dpi) * inch
        
        # Start with specified dimensions or original
        target_width = image_data.width or orig_width_pts
        target_height = image_data.height or orig_height_pts
        
        # Convert specified dimensions to points if they're in pixels
        if image_data.width and image_data.width < 50:  # Assume inches if < 50
            target_width = image_data.width * inch
        elif image_data.width:  # Assume pixels
            target_width = (image_data.width / dpi) * inch
        
        if image_data.height and image_data.height < 50:  # Assume inches if < 50
            target_height = image_data.height * inch
        elif image_data.height:  # Assume pixels
            target_height = (image_data.height / dpi) * inch
        
        # Apply maximum constraints
        if image_data.max_width:
            max_width_pts = image_data.max_width * inch if image_data.max_width < 50 else (image_data.max_width / dpi) * inch
            target_width = min(target_width, max_width_pts)
        
        if image_data.max_height:
            max_height_pts = image_data.max_height * inch if image_data.max_height < 50 else (image_data.max_height / dpi) * inch
            target_height = min(target_height, max_height_pts)
        
        # Apply global maximum constraints
        target_width = min(target_width, self.max_width)
        target_height = min(target_height, self.max_height)
        
        # Maintain aspect ratio if requested
        if image_data.maintain_aspect_ratio:
            aspect_ratio = orig_width / orig_height
            
            # Calculate dimensions that fit within constraints while maintaining aspect ratio
            width_by_height = target_height * aspect_ratio
            height_by_width = target_width / aspect_ratio
            
            if width_by_height <= target_width:
                # Use height constraint
                target_width = width_by_height
            else:
                # Use width constraint
                target_height = height_by_width
        
        return target_width, target_height
    
    async def _apply_alignment(self, image: Image, alignment: ImageAlignment) -> Union[Image, List]:
        """
        Apply alignment to the image.
        
        Args:
            image: ReportLab Image object
            alignment: Image alignment setting
            
        Returns:
            Aligned image or list of flowables for complex alignment
        """
        if alignment == ImageAlignment.CENTER:
            # Center alignment using a paragraph
            from reportlab.platypus import Table
            table = Table([[image]], colWidths=[None])
            table.setStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ])
            return table
        
        elif alignment == ImageAlignment.RIGHT:
            # Right alignment using a table
            from reportlab.platypus import Table
            table = Table([[image]], colWidths=[None])
            table.setStyle([
                ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ])
            return table
        
        elif alignment == ImageAlignment.LEFT:
            # Left alignment (default)
            return image
        
        elif alignment == ImageAlignment.INLINE:
            # Inline alignment (treat as left for now)
            return image
        
        else:
            # Default to left alignment
            return image
    
    def _get_alignment_enum(self, alignment: ImageAlignment) -> int:
        """Convert ImageAlignment to ReportLab alignment enum."""
        alignment_map = {
            ImageAlignment.LEFT: TA_LEFT,
            ImageAlignment.CENTER: TA_CENTER,
            ImageAlignment.RIGHT: TA_RIGHT,
            ImageAlignment.INLINE: TA_LEFT
        }
        return alignment_map.get(alignment, TA_CENTER)
    
    async def _add_placeholder_image(self, story: List, message: str) -> None:
        """
        Add a placeholder when image loading fails.
        
        Args:
            story: List of ReportLab flowables
            message: Error message to display
        """
        try:
            # Create a simple placeholder using a paragraph
            placeholder_style = ParagraphStyle(
                'ImagePlaceholder',
                fontName='Helvetica',
                fontSize=10,
                alignment=TA_CENTER,
                spaceBefore=6,
                spaceAfter=6,
                borderColor=colors.grey,
                borderWidth=1,
                borderRadius=3,
                leftIndent=10,
                rightIndent=10,
                topPadding=10,
                bottomPadding=10,
                backgroundColor=colors.lightgrey
            )
            
            placeholder_text = f"[Image Placeholder]<br/>{message}"
            placeholder = Paragraph(placeholder_text, placeholder_style)
            
            story.append(Spacer(1, 6))
            story.append(placeholder)
            story.append(Spacer(1, 6))
            
        except Exception as e:
            logger.error(f"Failed to add placeholder image: {e}")
            # Ultra-simple fallback
            from reportlab.lib.styles import getSampleStyleSheet
            styles = getSampleStyleSheet()
            story.append(Paragraph(f"[Image: {message}]", styles['Normal']))
    
    def _escape_text(self, text: str) -> str:
        """Escape text for safe use in ReportLab paragraphs."""
        if not text:
            return ""
        
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        return text
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats."""
        return list(self.supported_formats)
    
    def set_compression_defaults(self, quality: int, max_width: float, max_height: float) -> None:
        """
        Set default compression settings.
        
        Args:
            quality: JPEG compression quality (1-100)
            max_width: Maximum image width in inches
            max_height: Maximum image height in inches
        """
        self.default_quality = max(1, min(100, quality))
        self.max_width = max_width * inch
        self.max_height = max_height * inch
        
        logger.info(f"Updated compression defaults: quality={quality}, max_size={max_width}x{max_height} inches")
    
    async def optimize_for_pdf(self, image_path: str, output_path: str, quality: int = 85) -> bool:
        """
        Optimize an image file for PDF embedding.
        
        Args:
            image_path: Path to source image
            output_path: Path for optimized output
            quality: Compression quality (1-100)
            
        Returns:
            True if optimization successful, False otherwise
        """
        try:
            with PILImage.open(image_path) as img:
                # Process the image
                processed_img = await self._process_image(img, ImageData(compression_quality=quality))
                
                # Save optimized image
                processed_img.save(output_path, format='JPEG', quality=quality, optimize=True)
                
                logger.info(f"Optimized image: {image_path} -> {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to optimize image {image_path}: {e}")
            return False