"""
PDF Layout Manager

This module handles page layout configuration and template application
for PDF generation using ReportLab, including page sizes, margins, and templates.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from reportlab.platypus import SimpleDocTemplate, PageTemplate, Frame
from reportlab.lib.pagesizes import A4, LETTER, LEGAL, landscape, portrait
from reportlab.lib.units import inch, cm, mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas

from ...models import TemplateLayout, DocumentTemplate

logger = logging.getLogger(__name__)


class PdfLayoutManager:
    """
    Manages PDF page layouts and templates using ReportLab.
    
    This class handles page size configuration, margin settings, and the
    application of template-based layout configurations to PDF documents.
    """
    
    def __init__(self):
        """Initialize the PDF layout manager."""
        # Page size mappings
        self.page_sizes = {
            'A4': A4,
            'Letter': LETTER,
            'Legal': LEGAL,
            'A3': (297*mm, 420*mm),
            'A5': (148*mm, 210*mm),
            'Tabloid': (11*inch, 17*inch),
            'Executive': (7.25*inch, 10.5*inch)
        }
        
        # Default margins (in cm)
        self.default_margins = {
            'top': 2.54,
            'bottom': 2.54,
            'left': 2.54,
            'right': 2.54
        }
        
        # Header/footer heights (in cm)
        self.default_header_height = 1.27
        self.default_footer_height = 1.27
    
    async def apply_template_layout(
        self, 
        doc: SimpleDocTemplate, 
        layout: TemplateLayout
    ) -> None:
        """
        Apply template layout configuration to a PDF document.
        
        Args:
            doc: ReportLab SimpleDocTemplate
            layout: Template layout configuration
        """
        try:
            logger.info("Applying template layout configuration")
            
            # Apply page size and orientation
            page_size = self._get_page_size(layout.page_size, layout.page_orientation)
            doc.pagesize = page_size
            
            # Apply margins
            margins = layout.margins or self.default_margins
            doc.topMargin = cm * margins.get('top', self.default_margins['top'])
            doc.bottomMargin = cm * margins.get('bottom', self.default_margins['bottom'])
            doc.leftMargin = cm * margins.get('left', self.default_margins['left'])
            doc.rightMargin = cm * margins.get('right', self.default_margins['right'])
            
            # Configure header and footer space
            if hasattr(layout, 'header_height'):
                doc.topMargin += cm * (layout.header_height or self.default_header_height)
            
            if hasattr(layout, 'footer_height'):
                doc.bottomMargin += cm * (layout.footer_height or self.default_footer_height)
            
            # Handle multi-column layout if specified
            if hasattr(layout, 'column_count') and layout.column_count > 1:
                await self._configure_multi_column_layout(doc, layout)
            
            logger.debug(f"Applied layout: {layout.page_size} {layout.page_orientation}, margins: {margins}")
            
        except Exception as e:
            logger.error(f"Failed to apply template layout: {e}")
            # Use default layout as fallback
            doc.pagesize = A4
            doc.topMargin = cm * self.default_margins['top']
            doc.bottomMargin = cm * self.default_margins['bottom']
            doc.leftMargin = cm * self.default_margins['left']
            doc.rightMargin = cm * self.default_margins['right']
    
    def _get_page_size(self, page_size: str, orientation: str) -> Tuple[float, float]:
        """
        Get ReportLab page size tuple with orientation.
        
        Args:
            page_size: Page size name
            orientation: 'portrait' or 'landscape'
            
        Returns:
            Tuple of (width, height) in points
        """
        # Get base page size
        base_size = self.page_sizes.get(page_size, A4)
        
        # Apply orientation
        if orientation.lower() == 'landscape':
            return landscape(base_size)
        else:
            return portrait(base_size)
    
    async def _configure_multi_column_layout(
        self, 
        doc: SimpleDocTemplate, 
        layout: TemplateLayout
    ) -> None:
        """
        Configure multi-column layout for the document.
        
        Args:
            doc: ReportLab SimpleDocTemplate
            layout: Template layout configuration
        """
        try:
            column_count = layout.column_count
            column_spacing = cm * (layout.column_spacing if hasattr(layout, 'column_spacing') else 1.27)
            
            # Calculate column width
            page_width = doc.width
            total_spacing = column_spacing * (column_count - 1)
            column_width = (page_width - total_spacing) / column_count
            
            # Create frames for each column
            frames = []
            for i in range(column_count):
                x_offset = doc.leftMargin + i * (column_width + column_spacing)
                frame = Frame(
                    x_offset, 
                    doc.bottomMargin, 
                    column_width, 
                    doc.height,
                    id=f'column_{i+1}',
                    showBoundary=0
                )
                frames.append(frame)
            
            # Create page template with multiple frames
            page_template = PageTemplate(id='MultiColumn', frames=frames)
            doc.addPageTemplates([page_template])
            
            logger.debug(f"Configured {column_count}-column layout")
            
        except Exception as e:
            logger.error(f"Failed to configure multi-column layout: {e}")
            # Fall back to single column
            pass
    
    def create_custom_page_template(
        self, 
        template_id: str,
        page_size: Tuple[float, float],
        margins: Dict[str, float],
        header_height: float = 0,
        footer_height: float = 0,
        column_count: int = 1,
        column_spacing: float = 1.27
    ) -> PageTemplate:
        """
        Create a custom page template with specified layout.
        
        Args:
            template_id: Unique identifier for the template
            page_size: Tuple of (width, height) in points
            margins: Dictionary of margin values in cm
            header_height: Header height in cm
            footer_height: Footer height in cm
            column_count: Number of columns (default 1)
            column_spacing: Spacing between columns in cm
            
        Returns:
            ReportLab PageTemplate
        """
        try:
            # Convert margins to points
            left_margin = cm * margins.get('left', self.default_margins['left'])
            right_margin = cm * margins.get('right', self.default_margins['right'])
            top_margin = cm * (margins.get('top', self.default_margins['top']) + header_height)
            bottom_margin = cm * (margins.get('bottom', self.default_margins['bottom']) + footer_height)
            
            # Calculate content area
            content_width = page_size[0] - left_margin - right_margin
            content_height = page_size[1] - top_margin - bottom_margin
            
            # Create frames
            frames = []
            
            if column_count == 1:
                # Single column layout
                frame = Frame(
                    left_margin,
                    bottom_margin,
                    content_width,
                    content_height,
                    id='main'
                )
                frames.append(frame)
            else:
                # Multi-column layout
                spacing = cm * column_spacing
                total_spacing = spacing * (column_count - 1)
                column_width = (content_width - total_spacing) / column_count
                
                for i in range(column_count):
                    x_offset = left_margin + i * (column_width + spacing)
                    frame = Frame(
                        x_offset,
                        bottom_margin,
                        column_width,
                        content_height,
                        id=f'column_{i+1}'
                    )
                    frames.append(frame)
            
            # Create page template
            page_template = PageTemplate(id=template_id, frames=frames)
            
            logger.debug(f"Created custom page template: {template_id}")
            return page_template
            
        except Exception as e:
            logger.error(f"Failed to create custom page template: {e}")
            # Return basic single-frame template as fallback
            frame = Frame(cm * 2.54, cm * 2.54, A4[0] - cm * 5.08, A4[1] - cm * 5.08)
            return PageTemplate(id=template_id, frames=[frame])
    
    def calculate_content_area(
        self, 
        page_size: Tuple[float, float], 
        margins: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate the available content area given page size and margins.
        
        Args:
            page_size: Tuple of (width, height) in points
            margins: Dictionary of margin values in cm
            
        Returns:
            Dictionary with content area dimensions
        """
        left_margin = cm * margins.get('left', self.default_margins['left'])
        right_margin = cm * margins.get('right', self.default_margins['right'])
        top_margin = cm * margins.get('top', self.default_margins['top'])
        bottom_margin = cm * margins.get('bottom', self.default_margins['bottom'])
        
        content_width = page_size[0] - left_margin - right_margin
        content_height = page_size[1] - top_margin - bottom_margin
        
        return {
            'width': content_width,
            'height': content_height,
            'left': left_margin,
            'bottom': bottom_margin,
            'right': page_size[0] - right_margin,
            'top': page_size[1] - top_margin
        }
    
    def get_available_page_sizes(self) -> List[str]:
        """Get list of available page size names."""
        return list(self.page_sizes.keys())
    
    def add_custom_page_size(self, name: str, width: float, height: float, unit: str = 'mm') -> None:
        """
        Add a custom page size to the available sizes.
        
        Args:
            name: Name for the custom page size
            width: Page width
            height: Page height
            unit: Unit of measurement ('mm', 'cm', 'inch', 'pt')
        """
        try:
            # Convert to points
            unit_multipliers = {
                'mm': mm,
                'cm': cm,
                'inch': inch,
                'pt': 1.0
            }
            
            multiplier = unit_multipliers.get(unit.lower(), mm)
            page_size = (width * multiplier, height * multiplier)
            
            self.page_sizes[name] = page_size
            logger.info(f"Added custom page size: {name} ({width}x{height} {unit})")
            
        except Exception as e:
            logger.error(f"Failed to add custom page size {name}: {e}")
    
    def validate_layout_config(self, layout: TemplateLayout) -> List[str]:
        """
        Validate layout configuration.
        
        Args:
            layout: Template layout to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check page size
        if layout.page_size not in self.page_sizes:
            errors.append(f"Unknown page size: {layout.page_size}")
        
        # Check orientation
        if layout.page_orientation not in ['portrait', 'landscape']:
            errors.append(f"Invalid orientation: {layout.page_orientation}")
        
        # Check margins
        if layout.margins:
            for margin_name, value in layout.margins.items():
                if value < 0:
                    errors.append(f"Negative margin not allowed: {margin_name}={value}")
                elif value > 10:  # Arbitrary limit of 10cm
                    errors.append(f"Margin too large: {margin_name}={value}cm")
        
        # Check column configuration
        if hasattr(layout, 'column_count'):
            if layout.column_count < 1 or layout.column_count > 6:
                errors.append(f"Invalid column count: {layout.column_count} (must be 1-6)")
            
            if hasattr(layout, 'column_spacing') and layout.column_spacing < 0:
                errors.append(f"Negative column spacing not allowed: {layout.column_spacing}")
        
        return errors
    
    def get_layout_recommendations(self, content_type: str) -> Dict[str, Any]:
        """
        Get layout recommendations based on content type.
        
        Args:
            content_type: Type of document content
            
        Returns:
            Dictionary with recommended layout settings
        """
        recommendations = {
            'report': {
                'page_size': 'A4',
                'orientation': 'portrait',
                'margins': {'top': 2.5, 'bottom': 2.5, 'left': 2.5, 'right': 2.5},
                'column_count': 1
            },
            'newsletter': {
                'page_size': 'A4',
                'orientation': 'portrait',
                'margins': {'top': 2.0, 'bottom': 2.0, 'left': 2.0, 'right': 2.0},
                'column_count': 2,
                'column_spacing': 1.0
            },
            'brochure': {
                'page_size': 'A4',
                'orientation': 'landscape',
                'margins': {'top': 1.5, 'bottom': 1.5, 'left': 1.5, 'right': 1.5},
                'column_count': 3,
                'column_spacing': 0.8
            },
            'letter': {
                'page_size': 'Letter',
                'orientation': 'portrait',
                'margins': {'top': 2.54, 'bottom': 2.54, 'left': 2.54, 'right': 2.54},
                'column_count': 1
            },
            'presentation': {
                'page_size': 'A4',
                'orientation': 'landscape',
                'margins': {'top': 2.0, 'bottom': 2.0, 'left': 2.0, 'right': 2.0},
                'column_count': 1
            }
        }
        
        return recommendations.get(content_type.lower(), recommendations['report'])
    
    def estimate_page_count(
        self, 
        content_length: int, 
        layout: TemplateLayout,
        average_chars_per_page: int = 3000
    ) -> int:
        """
        Estimate page count based on content length and layout.
        
        Args:
            content_length: Total character count of content
            layout: Layout configuration
            average_chars_per_page: Estimated characters per page
            
        Returns:
            Estimated number of pages
        """
        # Adjust characters per page based on layout
        chars_per_page = average_chars_per_page
        
        # Multi-column layouts typically fit more text
        if hasattr(layout, 'column_count') and layout.column_count > 1:
            chars_per_page *= 1.2  # 20% more text with columns
        
        # Smaller margins fit more text
        if layout.margins:
            avg_margin = sum(layout.margins.values()) / len(layout.margins)
            if avg_margin < 2.0:  # Small margins
                chars_per_page *= 1.1
            elif avg_margin > 3.0:  # Large margins
                chars_per_page *= 0.9
        
        # Landscape orientation typically fits more text per page
        if layout.page_orientation == 'landscape':
            chars_per_page *= 1.3
        
        estimated_pages = max(1, content_length // int(chars_per_page))
        return estimated_pages