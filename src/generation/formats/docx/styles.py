"""
DOCX Style Management

This module handles style definitions and formatting for DOCX documents,
providing consistent styling and template application capabilities.
"""

import logging
from typing import Dict, Any, Optional, List
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.styles.style import _CharacterStyle, _ParagraphStyle

from ...models import DocumentTemplate, TemplateStyle, TextStyle, StyleWeight, StyleEmphasis

logger = logging.getLogger(__name__)


class DocxStyleManager:
    """
    Manages styles and formatting for DOCX documents.
    
    This class handles the application of text styles, template styles,
    and corporate branding to DOCX documents.
    """
    
    def __init__(self):
        """Initialize the style manager."""
        self._style_cache: Dict[str, Any] = {}
        
        # Default style mappings
        self._weight_mappings = {
            StyleWeight.NORMAL: False,
            StyleWeight.BOLD: True,
            StyleWeight.LIGHT: False  # Not directly supported in DOCX
        }
        
        self._emphasis_mappings = {
            StyleEmphasis.NONE: {'italic': False, 'underline': False},
            StyleEmphasis.ITALIC: {'italic': True, 'underline': False},
            StyleEmphasis.UNDERLINE: {'italic': False, 'underline': True},
            StyleEmphasis.STRIKETHROUGH: {'italic': False, 'underline': False}  # Special handling needed
        }
    
    async def apply_template_styles(self, document: Document, template: DocumentTemplate) -> None:
        """
        Apply template styles to a document.
        
        Args:
            document: The DOCX document to style
            template: Template containing style definitions
        """
        try:
            logger.info(f"Applying template styles from: {template.name}")
            
            # Apply brand colors if available
            if template.brand_colors:
                self._apply_brand_colors(document, template.brand_colors)
            
            # Apply template styles
            for template_style in template.styles:
                await self._apply_template_style(document, template_style)
            
            # Apply default font settings
            if template.layout:
                await self._apply_default_fonts(document, template)
            
            logger.info("Template styles applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply template styles: {str(e)}")
            raise
    
    async def _apply_template_style(self, document: Document, template_style: TemplateStyle) -> None:
        """Apply a single template style to the document."""
        try:
            # Get or create style in document
            style_name = template_style.name
            
            # Check if style already exists
            existing_style = None
            for style in document.styles:
                if style.name == style_name:
                    existing_style = style
                    break
            
            if existing_style is None:
                # Create new style based on type
                if any(content_type.value in ['heading', 'paragraph', 'text'] for content_type in template_style.applies_to):
                    style = document.styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
                else:
                    style = document.styles.add_style(style_name, WD_STYLE_TYPE.CHARACTER)
            else:
                style = existing_style
            
            # Apply text formatting
            await self._apply_text_style_to_docx_style(style, template_style.style)
            
        except Exception as e:
            logger.warning(f"Failed to apply template style '{template_style.name}': {str(e)}")
    
    async def _apply_text_style_to_docx_style(self, docx_style, text_style: TextStyle) -> None:
        """Apply TextStyle formatting to a DOCX style."""
        if not text_style:
            return
        
        # Font family
        if text_style.font_family:
            docx_style.font.name = text_style.font_family
        
        # Font size
        if text_style.font_size:
            docx_style.font.size = Pt(text_style.font_size)
        
        # Font weight (bold)
        if text_style.weight in self._weight_mappings:
            docx_style.font.bold = self._weight_mappings[text_style.weight]
        
        # Font emphasis
        if text_style.emphasis in self._emphasis_mappings:
            emphasis_settings = self._emphasis_mappings[text_style.emphasis]
            docx_style.font.italic = emphasis_settings['italic']
            docx_style.font.underline = emphasis_settings['underline']
            
            # Special handling for strikethrough
            if text_style.emphasis == StyleEmphasis.STRIKETHROUGH:
                docx_style.font.strike = True
        
        # Font color
        if text_style.color:
            color = self._parse_color(text_style.color)
            if color:
                docx_style.font.color.rgb = color
        
        # Background color (highlighting)
        if text_style.background_color:
            # DOCX doesn't directly support background color on styles
            # This would need to be applied at the run level
            pass
        
        # Line height (paragraph-level style)
        if hasattr(docx_style, 'paragraph_format') and text_style.line_height:
            docx_style.paragraph_format.line_spacing = text_style.line_height
    
    def apply_text_style_to_run(self, run, text_style: TextStyle) -> None:
        """
        Apply text style to a specific run (character formatting).
        
        Args:
            run: DOCX run object
            text_style: TextStyle to apply
        """
        if not text_style:
            return
        
        # Font family
        if text_style.font_family:
            run.font.name = text_style.font_family
        
        # Font size
        if text_style.font_size:
            run.font.size = Pt(text_style.font_size)
        
        # Font weight
        if text_style.weight == StyleWeight.BOLD:
            run.font.bold = True
        elif text_style.weight == StyleWeight.LIGHT:
            run.font.bold = False
        
        # Font emphasis
        if text_style.emphasis == StyleEmphasis.ITALIC:
            run.font.italic = True
        elif text_style.emphasis == StyleEmphasis.UNDERLINE:
            run.font.underline = True
        elif text_style.emphasis == StyleEmphasis.STRIKETHROUGH:
            run.font.strike = True
        
        # Font color
        if text_style.color:
            color = self._parse_color(text_style.color)
            if color:
                run.font.color.rgb = color
        
        # Background color (highlighting)
        if text_style.background_color:
            highlight_color = self._parse_highlight_color(text_style.background_color)
            if highlight_color:
                run.font.highlight_color = highlight_color
    
    def apply_paragraph_style(self, paragraph, text_style: TextStyle, alignment: Optional[str] = None) -> None:
        """
        Apply paragraph-level formatting.
        
        Args:
            paragraph: DOCX paragraph object
            text_style: TextStyle to apply
            alignment: Text alignment ('left', 'center', 'right', 'justify')
        """
        # Paragraph alignment
        if alignment:
            alignment_map = {
                'left': WD_ALIGN_PARAGRAPH.LEFT,
                'center': WD_ALIGN_PARAGRAPH.CENTER,
                'right': WD_ALIGN_PARAGRAPH.RIGHT,
                'justify': WD_ALIGN_PARAGRAPH.JUSTIFY
            }
            if alignment in alignment_map:
                paragraph.alignment = alignment_map[alignment]
        
        # Line spacing
        if text_style and text_style.line_height:
            paragraph.paragraph_format.line_spacing = text_style.line_height
        
        # Paragraph spacing
        if text_style:
            # Add some default spacing
            paragraph.paragraph_format.space_after = Pt(6)
    
    def _parse_color(self, color_string: str) -> Optional[RGBColor]:
        """
        Parse color string to RGBColor.
        
        Args:
            color_string: Color in hex (#RRGGBB) or named format
            
        Returns:
            RGBColor object or None if parsing fails
        """
        try:
            if color_string.startswith('#') and len(color_string) == 7:
                # Hex color
                r = int(color_string[1:3], 16)
                g = int(color_string[3:5], 16)
                b = int(color_string[5:7], 16)
                return RGBColor(r, g, b)
            else:
                # Named colors (basic support)
                color_map = {
                    'black': RGBColor(0, 0, 0),
                    'white': RGBColor(255, 255, 255),
                    'red': RGBColor(255, 0, 0),
                    'green': RGBColor(0, 128, 0),
                    'blue': RGBColor(0, 0, 255),
                    'gray': RGBColor(128, 128, 128),
                    'grey': RGBColor(128, 128, 128)
                }
                return color_map.get(color_string.lower())
        except Exception:
            logger.warning(f"Failed to parse color: {color_string}")
            return None
    
    def _parse_highlight_color(self, color_string: str) -> Optional[Any]:
        """Parse highlight color (limited support in python-docx)."""
        # python-docx has limited highlight color support
        # This is a simplified implementation
        highlight_map = {
            'yellow': 7,  # WD_COLOR_INDEX.YELLOW
            'green': 11,  # WD_COLOR_INDEX.BRIGHT_GREEN
            'cyan': 10,   # WD_COLOR_INDEX.TURQUOISE
            'pink': 13    # WD_COLOR_INDEX.PINK
        }
        return highlight_map.get(color_string.lower())
    
    def _apply_brand_colors(self, document: Document, brand_colors: Dict[str, str]) -> None:
        """Apply brand colors to document theme (if supported)."""
        # This is a placeholder for brand color application
        # Full implementation would require more advanced theme manipulation
        logger.info(f"Brand colors available: {list(brand_colors.keys())}")
    
    async def _apply_default_fonts(self, document: Document, template: DocumentTemplate) -> None:
        """Apply default font settings from template."""
        try:
            # Apply to Normal style
            normal_style = document.styles['Normal']
            
            # Set default font (if not already set by template)
            if not any(style.name == 'DefaultFont' for style in template.styles):
                normal_style.font.name = 'Calibri'  # Default professional font
                normal_style.font.size = Pt(11)
                
        except Exception as e:
            logger.warning(f"Failed to apply default fonts: {str(e)}")
    
    def create_heading_style(self, document: Document, level: int, text_style: Optional[TextStyle] = None) -> str:
        """
        Create or get a heading style for the specified level.
        
        Args:
            document: DOCX document
            level: Heading level (1-6)
            text_style: Optional custom styling
            
        Returns:
            Style name to use
        """
        style_name = f'Heading {level}'
        
        try:
            # Check if style exists
            style = document.styles[style_name]
        except KeyError:
            # Create heading style if it doesn't exist
            style = document.styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
            
            # Set default heading formatting
            font_sizes = {1: 16, 2: 14, 3: 13, 4: 12, 5: 11, 6: 10}
            style.font.size = Pt(font_sizes.get(level, 11))
            style.font.bold = True
            style.font.color.rgb = RGBColor(0, 0, 0)
            
            # Add spacing
            style.paragraph_format.space_before = Pt(12)
            style.paragraph_format.space_after = Pt(6)
        
        # Apply custom styling if provided
        if text_style:
            self.apply_text_style_to_run(style, text_style)
        
        return style_name
    
    def get_list_style(self, document: Document, list_type: str) -> Optional[str]:
        """
        Get appropriate list style name.
        
        Args:
            document: DOCX document
            list_type: Type of list ('bullet', 'numbered', 'checklist')
            
        Returns:
            Style name or None
        """
        style_map = {
            'bullet': 'List Bullet',
            'numbered': 'List Number',
            'checklist': 'List Bullet'  # Use bullet style for checklists
        }
        
        style_name = style_map.get(list_type)
        if style_name:
            try:
                # Check if style exists
                document.styles[style_name]
                return style_name
            except KeyError:
                # Style doesn't exist, return None to use default
                return None
        
        return None


# Test functionality
def test_style_manager():
    """Test style manager functionality."""
    from docx import Document
    from ...models import DocumentTemplate, TemplateStyle, TextStyle, StyleWeight, StyleEmphasis, ContentType
    
    # Create test document
    document = Document()
    
    # Create test template
    template = DocumentTemplate(
        name="Test Template",
        format_type="docx",
        styles=[
            TemplateStyle(
                name="Custom Heading",
                style=TextStyle(
                    font_family="Arial",
                    font_size=14,
                    weight=StyleWeight.BOLD,
                    color="#0066CC"
                ),
                applies_to=[ContentType.HEADING]
            )
        ],
        brand_colors={
            "primary": "#0066CC",
            "secondary": "#666666"
        }
    )
    
    # Test style manager
    import asyncio
    
    async def run_test():
        style_manager = DocxStyleManager()
        
        try:
            await style_manager.apply_template_styles(document, template)
            print("✅ Style manager test passed")
            return True
        except Exception as e:
            print(f"❌ Style manager test failed: {e}")
            return False
    
    return asyncio.run(run_test())


if __name__ == "__main__":
    test_style_manager()