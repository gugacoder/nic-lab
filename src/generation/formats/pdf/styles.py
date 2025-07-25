"""
PDF Style Management

This module handles style configuration and application for PDF generation
using ReportLab, including paragraph styles, font management, and color handling.
"""

import logging
from typing import Dict, List, Optional, Any
from reportlab.lib.styles import StyleSheet1, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import inch, cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from ...models import DocumentTemplate, TemplateStyle, TextStyle, StyleWeight, StyleEmphasis, ContentType

logger = logging.getLogger(__name__)


class PdfStyleManager:
    """
    Manages PDF styles and formatting using ReportLab.
    
    This class handles the conversion of abstract text styles to ReportLab
    paragraph styles, manages font loading, and applies template-based styling.
    """
    
    def __init__(self):
        """Initialize the PDF style manager."""
        self._custom_fonts: Dict[str, str] = {}
        self._color_cache: Dict[str, Any] = {}
        
        # Default style mappings
        self._weight_map = {
            StyleWeight.NORMAL: 'Helvetica',
            StyleWeight.BOLD: 'Helvetica-Bold',
            StyleWeight.LIGHT: 'Helvetica'
        }
        
        self._emphasis_map = {
            StyleEmphasis.NONE: '',
            StyleEmphasis.ITALIC: '-Oblique',
            StyleEmphasis.UNDERLINE: '',  # Handled separately
            StyleEmphasis.STRIKETHROUGH: ''  # Not directly supported in ReportLab
        }
        
        # Alignment mappings
        self._alignment_map = {
            'left': TA_LEFT,
            'center': TA_CENTER,
            'right': TA_RIGHT,
            'justify': TA_JUSTIFY
        }
    
    async def apply_template_styles(self, styles: StyleSheet1, template: DocumentTemplate) -> None:
        """
        Apply template styles to the ReportLab stylesheet.
        
        Args:
            styles: ReportLab stylesheet to modify
            template: Template containing style definitions
        """
        logger.info(f"Applying {len(template.styles)} template styles")
        
        for template_style in template.styles:
            try:
                # Convert template style to ReportLab style
                reportlab_style = await self._convert_template_style(template_style, styles)
                
                # Add or update style in stylesheet
                styles.add(reportlab_style)
                
                logger.debug(f"Applied style: {template_style.name}")
                
            except Exception as e:
                logger.warning(f"Failed to apply style '{template_style.name}': {e}")
    
    async def _convert_template_style(
        self, 
        template_style: TemplateStyle, 
        base_styles: StyleSheet1
    ) -> ParagraphStyle:
        """
        Convert a template style to a ReportLab ParagraphStyle.
        
        Args:
            template_style: Template style to convert
            base_styles: Base stylesheet for parent styles
            
        Returns:
            ReportLab ParagraphStyle
        """
        text_style = template_style.style
        
        # Determine parent style
        parent_style = self._get_parent_style(template_style.applies_to, base_styles)
        
        # Build style parameters
        style_params = {
            'name': template_style.name,
            'parent': parent_style
        }
        
        # Apply font family
        if text_style.font_family:
            font_name = await self._get_font_name(text_style)
            if font_name:
                style_params['fontName'] = font_name
        
        # Apply font size
        if text_style.font_size:
            style_params['fontSize'] = text_style.font_size
            style_params['leading'] = text_style.font_size * (text_style.line_height or 1.2)
        
        # Apply colors
        if text_style.color:
            style_params['textColor'] = self._parse_color(text_style.color)
        
        if text_style.background_color:
            style_params['backColor'] = self._parse_color(text_style.background_color)
        
        # Apply spacing
        if text_style.letter_spacing:
            # ReportLab doesn't directly support letter spacing in ParagraphStyle
            # This would need to be handled at the text level
            pass
        
        # Apply line height
        if text_style.line_height and text_style.font_size:
            style_params['leading'] = text_style.font_size * text_style.line_height
        
        # Create and return the style
        return ParagraphStyle(**style_params)
    
    def _get_parent_style(self, applies_to: List[ContentType], base_styles: StyleSheet1) -> ParagraphStyle:
        """Get appropriate parent style based on content types."""
        if not applies_to:
            return base_styles['Normal']
        
        # Map content types to base styles
        primary_type = applies_to[0]
        
        if primary_type == ContentType.HEADING:
            return base_styles.get('Heading1', base_styles['Normal'])
        elif primary_type == ContentType.PARAGRAPH:
            return base_styles['Normal']
        elif primary_type == ContentType.LIST:
            return base_styles.get('Bullet', base_styles['Normal'])
        elif primary_type == ContentType.CODE:
            return base_styles.get('Code', base_styles['Normal'])
        elif primary_type == ContentType.QUOTE:
            return base_styles.get('Quote', base_styles['Normal'])
        else:
            return base_styles['Normal']
    
    async def _get_font_name(self, text_style: TextStyle) -> Optional[str]:
        """
        Get the appropriate ReportLab font name for the text style.
        
        Args:
            text_style: Text style configuration
            
        Returns:
            ReportLab font name or None if default should be used
        """
        if not text_style.font_family:
            return None
        
        # Handle common font families
        base_font = self._map_font_family(text_style.font_family)
        
        # Apply weight and emphasis
        font_name = base_font
        
        if text_style.weight == StyleWeight.BOLD:
            if 'Bold' not in font_name:
                font_name += '-Bold'
        
        if text_style.emphasis == StyleEmphasis.ITALIC:
            if 'Oblique' not in font_name and 'Italic' not in font_name:
                font_name += '-Oblique'
        
        return font_name
    
    def _map_font_family(self, font_family: str) -> str:
        """Map common font family names to ReportLab font names."""
        font_mapping = {
            'arial': 'Helvetica',
            'helvetica': 'Helvetica',
            'sans-serif': 'Helvetica',
            'times': 'Times-Roman',
            'times new roman': 'Times-Roman',
            'serif': 'Times-Roman',
            'courier': 'Courier',
            'courier new': 'Courier',
            'monospace': 'Courier',
            'calibri': 'Helvetica',  # Fallback
            'verdana': 'Helvetica',  # Fallback
        }
        
        return font_mapping.get(font_family.lower(), font_family)
    
    def _parse_color(self, color_str: str) -> Any:
        """
        Parse color string to ReportLab color object.
        
        Args:
            color_str: Color in hex (#RRGGBB) or named format
            
        Returns:
            ReportLab color object
        """
        if color_str in self._color_cache:
            return self._color_cache[color_str]
        
        color_obj = None
        
        try:
            if color_str.startswith('#'):
                # Hex color
                if len(color_str) == 7:  # #RRGGBB
                    r = int(color_str[1:3], 16) / 255.0
                    g = int(color_str[3:5], 16) / 255.0
                    b = int(color_str[5:7], 16) / 255.0
                    color_obj = colors.Color(r, g, b)
                elif len(color_str) == 4:  # #RGB
                    r = int(color_str[1], 16) / 15.0
                    g = int(color_str[2], 16) / 15.0
                    b = int(color_str[3], 16) / 15.0
                    color_obj = colors.Color(r, g, b)
            else:
                # Named color
                color_obj = getattr(colors, color_str.lower(), None)
                if color_obj is None:
                    # Try common color names
                    color_map = {
                        'black': colors.black,
                        'white': colors.white,
                        'red': colors.red,
                        'green': colors.green,
                        'blue': colors.blue,
                        'yellow': colors.yellow,
                        'cyan': colors.cyan,
                        'magenta': colors.magenta,
                        'gray': colors.gray,
                        'grey': colors.gray,
                        'darkgray': colors.darkgray,
                        'darkgrey': colors.darkgray,
                        'lightgray': colors.lightgray,
                        'lightgrey': colors.lightgray
                    }
                    color_obj = color_map.get(color_str.lower(), colors.black)
        
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse color: {color_str}, using black")
            color_obj = colors.black
        
        # Cache the result
        self._color_cache[color_str] = color_obj
        return color_obj
    
    def register_custom_font(self, font_name: str, font_path: str) -> None:
        """
        Register a custom font for use in PDFs.
        
        Args:
            font_name: Name to register the font as
            font_path: Path to the TTF font file
        """
        try:
            # Register the font with ReportLab
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            self._custom_fonts[font_name] = font_path
            logger.info(f"Registered custom font: {font_name}")
            
        except Exception as e:
            logger.error(f"Failed to register font {font_name}: {e}")
    
    def create_heading_style(
        self, 
        level: int, 
        base_styles: StyleSheet1,
        custom_style: Optional[TextStyle] = None
    ) -> ParagraphStyle:
        """
        Create a heading style for the specified level.
        
        Args:
            level: Heading level (1-6)
            base_styles: Base stylesheet
            custom_style: Optional custom styling
            
        Returns:
            ParagraphStyle for the heading
        """
        # Default heading sizes
        size_map = {1: 20, 2: 16, 3: 14, 4: 12, 5: 11, 6: 10}
        font_size = size_map.get(level, 10)
        
        style_params = {
            'name': f'CustomHeading{level}',
            'parent': base_styles['Heading1'],
            'fontSize': font_size,
            'leading': font_size * 1.2,
            'spaceAfter': 12,
            'spaceBefore': 12,
            'fontName': 'Helvetica-Bold'
        }
        
        # Apply custom styling if provided
        if custom_style:
            if custom_style.font_size:
                style_params['fontSize'] = custom_style.font_size
                style_params['leading'] = custom_style.font_size * (custom_style.line_height or 1.2)
            
            if custom_style.color:
                style_params['textColor'] = self._parse_color(custom_style.color)
            
            if custom_style.font_family:
                font_name = self._map_font_family(custom_style.font_family)
                if custom_style.weight == StyleWeight.BOLD:
                    font_name += '-Bold'
                style_params['fontName'] = font_name
        
        return ParagraphStyle(**style_params)
    
    def create_paragraph_style(
        self, 
        base_styles: StyleSheet1,
        custom_style: Optional[TextStyle] = None,
        alignment: str = 'left'
    ) -> ParagraphStyle:
        """
        Create a paragraph style with optional customization.
        
        Args:
            base_styles: Base stylesheet
            custom_style: Optional custom styling
            alignment: Text alignment ('left', 'center', 'right', 'justify')
            
        Returns:
            ParagraphStyle for paragraphs
        """
        style_params = {
            'name': 'CustomParagraph',
            'parent': base_styles['Normal'],
            'alignment': self._alignment_map.get(alignment, TA_LEFT),
            'spaceAfter': 6
        }
        
        # Apply custom styling if provided
        if custom_style:
            if custom_style.font_size:
                style_params['fontSize'] = custom_style.font_size
                style_params['leading'] = custom_style.font_size * (custom_style.line_height or 1.15)
            
            if custom_style.color:
                style_params['textColor'] = self._parse_color(custom_style.color)
            
            if custom_style.background_color:
                style_params['backColor'] = self._parse_color(custom_style.background_color)
            
            if custom_style.font_family:
                font_name = self._map_font_family(custom_style.font_family)
                if custom_style.weight == StyleWeight.BOLD:
                    font_name += '-Bold'
                if custom_style.emphasis == StyleEmphasis.ITALIC:
                    font_name += '-Oblique'
                style_params['fontName'] = font_name
        
        return ParagraphStyle(**style_params)
    
    def create_list_style(
        self, 
        base_styles: StyleSheet1,
        custom_style: Optional[TextStyle] = None
    ) -> ParagraphStyle:
        """
        Create a list item style.
        
        Args:
            base_styles: Base stylesheet
            custom_style: Optional custom styling
            
        Returns:
            ParagraphStyle for list items
        """
        style_params = {
            'name': 'CustomListItem',
            'parent': base_styles.get('Bullet', base_styles['Normal']),
            'leftIndent': 20,
            'bulletIndent': 10,
            'spaceAfter': 3
        }
        
        # Apply custom styling if provided
        if custom_style:
            if custom_style.font_size:
                style_params['fontSize'] = custom_style.font_size
                style_params['leading'] = custom_style.font_size * (custom_style.line_height or 1.15)
            
            if custom_style.color:
                style_params['textColor'] = self._parse_color(custom_style.color)
        
        return ParagraphStyle(**style_params)
    
    def get_available_fonts(self) -> List[str]:
        """Get list of available fonts."""
        # Standard ReportLab fonts
        standard_fonts = [
            'Helvetica', 'Helvetica-Bold', 'Helvetica-Oblique', 'Helvetica-BoldOblique',
            'Times-Roman', 'Times-Bold', 'Times-Italic', 'Times-BoldItalic',
            'Courier', 'Courier-Bold', 'Courier-Oblique', 'Courier-BoldOblique',
            'Symbol', 'ZapfDingbats'
        ]
        
        # Add custom fonts
        custom_fonts = list(self._custom_fonts.keys())
        
        return standard_fonts + custom_fonts