"""
PDF Element Handler

This module handles the creation and formatting of various content elements
for PDF generation using ReportLab, including headings, paragraphs, lists, and code blocks.
"""

import logging
from typing import List, Optional, Any
from reportlab.platypus import Paragraph, Spacer, KeepTogether
from reportlab.lib.styles import StyleSheet1, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import inch, cm

from ...models import ContentElement, ContentType, ListType, TextStyle

logger = logging.getLogger(__name__)


class PdfElementHandler:
    """
    Handles the creation of PDF content elements using ReportLab.
    
    This class converts structured content elements into ReportLab flowables
    that can be added to the PDF document story.
    """
    
    def __init__(self):
        """Initialize the PDF element handler."""
        self._bullet_symbols = {
            'bullet': '•',
            'circle': '○',
            'square': '■',
            'dash': '–'
        }
    
    async def add_heading(
        self, 
        story: List, 
        styles: StyleSheet1, 
        element: ContentElement
    ) -> None:
        """
        Add a heading element to the PDF story.
        
        Args:
            story: List of ReportLab flowables
            styles: ReportLab stylesheet
            element: Heading content element
        """
        try:
            # Get or create heading style
            style_name = f'Heading{element.level}'
            if style_name in styles:
                heading_style = styles[style_name]
            else:
                heading_style = self._create_heading_style(element.level, styles, element.style)
            
            # Create heading paragraph
            heading_text = self._escape_text(element.content)
            heading_paragraph = Paragraph(heading_text, heading_style)
            
            # Add to story with appropriate spacing
            if element.level <= 2:
                # Major headings get more spacing
                story.append(Spacer(1, 0.2 * inch))
                story.append(heading_paragraph)
                story.append(Spacer(1, 0.15 * inch))
            else:
                # Minor headings get less spacing
                story.append(Spacer(1, 0.1 * inch))
                story.append(heading_paragraph)
                story.append(Spacer(1, 0.1 * inch))
            
            logger.debug(f"Added heading: {element.content[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to add heading: {e}")
            # Fallback to plain paragraph
            await self.add_paragraph(story, styles, element)
    
    async def add_paragraph(
        self, 
        story: List, 
        styles: StyleSheet1, 
        element: ContentElement
    ) -> None:
        """
        Add a paragraph element to the PDF story.
        
        Args:
            story: List of ReportLab flowables
            styles: ReportLab stylesheet
            element: Paragraph content element
        """
        try:
            # Get paragraph style
            if element.style:
                paragraph_style = self._create_custom_paragraph_style(styles, element.style)
            else:
                paragraph_style = styles['Normal']
            
            # Create paragraph
            paragraph_text = self._escape_text(element.content)
            paragraph = Paragraph(paragraph_text, paragraph_style)
            
            # Add to story
            story.append(paragraph)
            story.append(Spacer(1, 6))  # Small spacing after paragraph
            
            logger.debug(f"Added paragraph: {element.content[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to add paragraph: {e}")
            # Add as plain text
            story.append(Paragraph(self._escape_text(element.content), styles['Normal']))
    
    async def add_list(
        self, 
        story: List, 
        styles: StyleSheet1, 
        element: ContentElement
    ) -> None:
        """
        Add a list element to the PDF story.
        
        Args:
            story: List of ReportLab flowables
            styles: ReportLab stylesheet
            element: List content element
        """
        try:
            if not element.list_data or not element.list_data.items:
                logger.warning("List element has no items")
                return
            
            list_data = element.list_data
            
            # Create list style
            if element.style:
                list_style = self._create_custom_list_style(styles, element.style)
            else:
                list_style = styles.get('Bullet', styles['Normal'])
            
            # Add spacing before list
            story.append(Spacer(1, 6))
            
            # Process list items
            for i, item in enumerate(list_data.items):
                item_text = self._escape_text(item.content)
                
                if list_data.list_type == ListType.BULLET:
                    # Bullet list
                    bullet_text = f"{self._bullet_symbols.get('bullet', '•')} {item_text}"
                    
                elif list_data.list_type == ListType.NUMBERED:
                    # Numbered list
                    item_number = list_data.start_number + i if hasattr(list_data, 'start_number') else i + 1
                    bullet_text = f"{item_number}. {item_text}"
                    
                elif list_data.list_type == ListType.CHECKLIST:
                    # Checklist
                    check = "☑" if item.checked else "☐"
                    bullet_text = f"{check} {item_text}"
                    
                else:
                    # Default to bullet
                    bullet_text = f"• {item_text}"
                
                # Handle nested levels
                if item.level > 0:
                    # Create nested style with additional indentation
                    nested_style = ParagraphStyle(
                        f'NestedList{item.level}',
                        parent=list_style,
                        leftIndent=list_style.leftIndent + (item.level * 20),
                        bulletIndent=list_style.bulletIndent + (item.level * 20)
                    )
                    list_paragraph = Paragraph(bullet_text, nested_style)
                else:
                    list_paragraph = Paragraph(bullet_text, list_style)
                
                story.append(list_paragraph)
            
            # Add spacing after list
            story.append(Spacer(1, 6))
            
            logger.debug(f"Added list with {len(list_data.items)} items")
            
        except Exception as e:
            logger.error(f"Failed to add list: {e}")
            # Fallback to simple paragraph
            if element.list_data and element.list_data.items:
                text = "\n".join([f"• {item.content}" for item in element.list_data.items])
                await self.add_paragraph(story, styles, ContentElement(
                    type=ContentType.PARAGRAPH,
                    content=text
                ))
    
    async def add_code_block(
        self, 
        story: List, 
        styles: StyleSheet1, 
        element: ContentElement
    ) -> None:
        """
        Add a code block element to the PDF story.
        
        Args:
            story: List of ReportLab flowables
            styles: ReportLab stylesheet
            element: Code content element
        """
        try:
            # Create code style
            code_style = self._create_code_style(styles, element.style)
            
            # Preserve code formatting
            code_text = self._escape_code_text(element.content)
            code_paragraph = Paragraph(code_text, code_style)
            
            # Add with spacing and keep together
            story.append(Spacer(1, 6))
            story.append(KeepTogether([code_paragraph]))
            story.append(Spacer(1, 6))
            
            logger.debug(f"Added code block: {len(element.content)} characters")
            
        except Exception as e:
            logger.error(f"Failed to add code block: {e}")
            # Fallback to paragraph with monospace font
            fallback_style = ParagraphStyle(
                'CodeFallback',
                parent=styles['Normal'],
                fontName='Courier',
                fontSize=9,
                backgroundColor=colors.lightgrey,
                borderColor=colors.grey,
                borderWidth=1,
                leftIndent=10,
                rightIndent=10,
                spaceBefore=6,
                spaceAfter=6
            )
            story.append(Paragraph(self._escape_text(element.content), fallback_style))
    
    async def add_quote(
        self, 
        story: List, 
        styles: StyleSheet1, 
        element: ContentElement
    ) -> None:
        """
        Add a quote element to the PDF story.
        
        Args:
            story: List of ReportLab flowables
            styles: ReportLab stylesheet
            element: Quote content element
        """
        try:
            # Create quote style
            quote_style = self._create_quote_style(styles, element.style)
            
            # Add quote markers if not already present
            quote_text = element.content
            if not quote_text.startswith('"') and not quote_text.startswith('"'):
                quote_text = f'"{quote_text}"'
            
            quote_paragraph = Paragraph(self._escape_text(quote_text), quote_style)
            
            # Add with appropriate spacing
            story.append(Spacer(1, 12))
            story.append(quote_paragraph)
            story.append(Spacer(1, 12))
            
            logger.debug(f"Added quote: {element.content[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to add quote: {e}")
            # Fallback to italic paragraph
            fallback_style = ParagraphStyle(
                'QuoteFallback',
                parent=styles['Normal'],
                fontName='Helvetica-Oblique',
                leftIndent=20,
                rightIndent=20,
                spaceBefore=6,
                spaceAfter=6
            )
            story.append(Paragraph(self._escape_text(element.content), fallback_style))
    
    async def add_divider(self, story: List, element: ContentElement) -> None:
        """
        Add a divider element to the PDF story.
        
        Args:
            story: List of ReportLab flowables
            element: Divider content element
        """
        try:
            from reportlab.platypus.flowables import HRFlowable
            
            # Add spacing before divider
            story.append(Spacer(1, 12))
            
            # Create horizontal rule
            divider = HRFlowable(
                width="100%",
                thickness=1,
                color=colors.black,
                spaceBefore=6,
                spaceAfter=6
            )
            story.append(divider)
            
            # Add spacing after divider
            story.append(Spacer(1, 12))
            
            logger.debug("Added divider")
            
        except Exception as e:
            logger.error(f"Failed to add divider: {e}")
            # Fallback to dashes
            dash_style = ParagraphStyle(
                'DividerFallback',
                parent=styles['Normal'],
                alignment=TA_CENTER,
                spaceBefore=12,
                spaceAfter=12
            )
            story.append(Paragraph("— — — — —", dash_style))
    
    def _create_heading_style(
        self, 
        level: int, 
        styles: StyleSheet1, 
        custom_style: Optional[TextStyle] = None
    ) -> ParagraphStyle:
        """Create a heading style for the specified level."""
        size_map = {1: 20, 2: 16, 3: 14, 4: 12, 5: 11, 6: 10}
        font_size = size_map.get(level, 10)
        
        style_params = {
            'name': f'CustomHeading{level}',
            'parent': styles.get('Heading1', styles['Normal']),
            'fontSize': font_size,
            'leading': font_size * 1.2,
            'spaceAfter': 12,
            'spaceBefore': 12,
            'fontName': 'Helvetica-Bold',
            'keepWithNext': True  # Keep heading with following content
        }
        
        # Apply custom styling
        if custom_style:
            if custom_style.font_size:
                style_params['fontSize'] = custom_style.font_size
                style_params['leading'] = custom_style.font_size * 1.2
            
            if custom_style.color:
                style_params['textColor'] = self._parse_color(custom_style.color)
        
        return ParagraphStyle(**style_params)
    
    def _create_custom_paragraph_style(
        self, 
        styles: StyleSheet1, 
        custom_style: TextStyle
    ) -> ParagraphStyle:
        """Create a custom paragraph style."""
        style_params = {
            'name': 'CustomParagraph',
            'parent': styles['Normal'],
            'spaceAfter': 6
        }
        
        if custom_style.font_size:
            style_params['fontSize'] = custom_style.font_size
            style_params['leading'] = custom_style.font_size * (custom_style.line_height or 1.15)
        
        if custom_style.color:
            style_params['textColor'] = self._parse_color(custom_style.color)
        
        if custom_style.background_color:
            style_params['backColor'] = self._parse_color(custom_style.background_color)
        
        return ParagraphStyle(**style_params)
    
    def _create_custom_list_style(
        self, 
        styles: StyleSheet1, 
        custom_style: TextStyle
    ) -> ParagraphStyle:
        """Create a custom list style."""
        base_style = styles.get('Bullet', styles['Normal'])
        
        style_params = {
            'name': 'CustomListItem',
            'parent': base_style,
            'leftIndent': 20,
            'bulletIndent': 10,
            'spaceAfter': 3
        }
        
        if custom_style.font_size:
            style_params['fontSize'] = custom_style.font_size
            style_params['leading'] = custom_style.font_size * 1.15
        
        if custom_style.color:
            style_params['textColor'] = self._parse_color(custom_style.color)
        
        return ParagraphStyle(**style_params)
    
    def _create_code_style(
        self, 
        styles: StyleSheet1, 
        custom_style: Optional[TextStyle] = None
    ) -> ParagraphStyle:
        """Create a code block style."""
        style_params = {
            'name': 'CustomCode',
            'parent': styles.get('Code', styles['Normal']),
            'fontName': 'Courier',
            'fontSize': 9,
            'backgroundColor': colors.lightgrey,
            'borderColor': colors.grey,
            'borderWidth': 1,
            'leftIndent': 10,
            'rightIndent': 10,
            'spaceBefore': 6,
            'spaceAfter': 6,
            'leading': 11
        }
        
        if custom_style:
            if custom_style.font_size:
                style_params['fontSize'] = custom_style.font_size
                style_params['leading'] = custom_style.font_size * 1.2
            
            if custom_style.background_color:
                style_params['backgroundColor'] = self._parse_color(custom_style.background_color)
        
        return ParagraphStyle(**style_params)
    
    def _create_quote_style(
        self, 
        styles: StyleSheet1, 
        custom_style: Optional[TextStyle] = None
    ) -> ParagraphStyle:
        """Create a quote style."""
        style_params = {
            'name': 'CustomQuote',
            'parent': styles['Normal'],
            'fontName': 'Helvetica-Oblique',
            'leftIndent': 20,
            'rightIndent': 20,
            'spaceBefore': 6,
            'spaceAfter': 6,
            'borderColor': colors.grey,
            'borderWidth': 0,
            'leftIndent': 30,
            'bulletText': None
        }
        
        if custom_style:
            if custom_style.font_size:
                style_params['fontSize'] = custom_style.font_size
                style_params['leading'] = custom_style.font_size * 1.15
            
            if custom_style.color:
                style_params['textColor'] = self._parse_color(custom_style.color)
        
        return ParagraphStyle(**style_params)
    
    def _escape_text(self, text: str) -> str:
        """Escape text for safe use in ReportLab paragraphs."""
        if not text:
            return ""
        
        # Escape HTML/XML characters that ReportLab might interpret
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Convert line breaks to ReportLab paragraph breaks
        text = text.replace('\n', '<br/>')
        
        return text
    
    def _escape_code_text(self, text: str) -> str:
        """Escape code text while preserving formatting."""
        if not text:
            return ""
        
        # Escape HTML/XML characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Preserve spaces and line breaks in code blocks
        text = text.replace(' ', '&nbsp;')
        text = text.replace('\n', '<br/>')
        text = text.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')  # Tab to 4 spaces
        
        return text
    
    def _parse_color(self, color_str: str) -> Any:
        """Parse color string to ReportLab color object."""
        try:
            if color_str.startswith('#'):
                # Hex color
                if len(color_str) == 7:  # #RRGGBB
                    r = int(color_str[1:3], 16) / 255.0
                    g = int(color_str[3:5], 16) / 255.0
                    b = int(color_str[5:7], 16) / 255.0
                    return colors.Color(r, g, b)
                elif len(color_str) == 4:  # #RGB
                    r = int(color_str[1], 16) / 15.0
                    g = int(color_str[2], 16) / 15.0
                    b = int(color_str[3], 16) / 15.0
                    return colors.Color(r, g, b)
            else:
                # Named color
                return getattr(colors, color_str.lower(), colors.black)
        
        except (ValueError, AttributeError):
            return colors.black