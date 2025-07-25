"""
DOCX Document Elements Handler

This module handles the creation and formatting of document elements
like paragraphs, headings, lists, and other content types in DOCX format.
"""

import logging
from typing import Optional, List
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn

from ...models import ContentElement, ContentType, ListType, TextStyle
from .styles import DocxStyleManager

logger = logging.getLogger(__name__)


class DocxElementHandler:
    """
    Handles creation and formatting of document elements in DOCX format.
    
    This class provides methods to add various content elements to DOCX documents
    with proper formatting and styling.
    """
    
    def __init__(self):
        """Initialize the element handler."""
        self.style_manager = DocxStyleManager()
    
    async def add_heading(self, document: Document, element: ContentElement) -> None:
        """
        Add a heading element to the document.
        
        Args:
            document: DOCX document to add to
            element: Heading element to add
        """
        try:
            # Determine heading level (1-6, with 1 being highest)
            level = max(1, min(6, element.level))
            
            # Add heading to document
            heading = document.add_heading(element.content, level=level)
            
            # Apply custom styling if provided
            if element.style:
                # Apply style to the first run in the heading
                if heading.runs:
                    self.style_manager.apply_text_style_to_run(heading.runs[0], element.style)
                
                # Apply paragraph-level formatting
                self.style_manager.apply_paragraph_style(heading, element.style)
            
            logger.debug(f"Added heading (level {level}): {element.content[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to add heading: {str(e)}")
            # Fallback to paragraph
            await self.add_paragraph(document, element)
    
    async def add_paragraph(self, document: Document, element: ContentElement) -> None:
        """
        Add a paragraph element to the document.
        
        Args:
            document: DOCX document to add to
            element: Paragraph element to add
        """
        try:
            # Add paragraph to document
            paragraph = document.add_paragraph(element.content)
            
            # Apply styling if provided
            if element.style:
                # Apply style to the first run
                if paragraph.runs:
                    self.style_manager.apply_text_style_to_run(paragraph.runs[0], element.style)
                
                # Apply paragraph-level formatting
                self.style_manager.apply_paragraph_style(paragraph, element.style)
            
            # Handle custom attributes
            if element.custom_attributes:
                await self._apply_custom_attributes(paragraph, element.custom_attributes)
            
            logger.debug(f"Added paragraph: {element.content[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to add paragraph: {str(e)}")
            # Add as plain text paragraph
            document.add_paragraph(element.content or "")
    
    async def add_list(self, document: Document, element: ContentElement) -> None:
        """
        Add a list element to the document.
        
        Args:
            document: DOCX document to add to
            element: List element to add
        """
        try:
            if not element.list_data or not element.list_data.items:
                logger.warning("List element has no items")
                return
            
            list_data = element.list_data
            
            # Determine list style
            style_name = self.style_manager.get_list_style(document, list_data.list_type.value)
            
            # Add list items
            for i, list_item in enumerate(list_data.items):
                if list_data.list_type == ListType.NUMBERED:
                    # For numbered lists, add number prefix
                    item_text = f"{list_data.start_number + i}. {list_item.content}"
                    paragraph = document.add_paragraph(item_text)
                elif list_data.list_type == ListType.CHECKLIST:
                    # For checklists, add checkbox
                    checkbox = "☑" if list_item.checked else "☐"
                    item_text = f"{checkbox} {list_item.content}"
                    paragraph = document.add_paragraph(item_text)
                else:
                    # Default bullet list
                    paragraph = document.add_paragraph(list_item.content, style=style_name)
                
                # Apply custom styling to list item
                if list_item.style:
                    if paragraph.runs:
                        self.style_manager.apply_text_style_to_run(paragraph.runs[0], list_item.style)
                    self.style_manager.apply_paragraph_style(paragraph, list_item.style)
                elif element.style:
                    # Apply list-level styling
                    if paragraph.runs:
                        self.style_manager.apply_text_style_to_run(paragraph.runs[0], element.style)
                    self.style_manager.apply_paragraph_style(paragraph, element.style)
                
                # Handle nested levels
                if list_item.level > 0:
                    await self._apply_list_indentation(paragraph, list_item.level)
            
            logger.debug(f"Added list with {len(list_data.items)} items")
            
        except Exception as e:
            logger.error(f"Failed to add list: {str(e)}")
            # Fallback to simple paragraphs
            if element.list_data and element.list_data.items:
                for item in element.list_data.items:
                    document.add_paragraph(f"• {item.content}")
    
    async def add_code_block(self, document: Document, element: ContentElement) -> None:
        """
        Add a code block element to the document.
        
        Args:
            document: DOCX document to add to
            element: Code element to add
        """
        try:
            # Add code block with monospace font
            paragraph = document.add_paragraph(element.content)
            
            # Apply monospace formatting
            for run in paragraph.runs:
                run.font.name = 'Consolas'  # Monospace font
                run.font.size = Pt(9)
                
            # Add background shading (if supported)
            await self._add_paragraph_shading(paragraph, "#F5F5F5")
            
            # Apply custom styling if provided
            if element.style:
                # Override with custom styles but keep monospace
                for run in paragraph.runs:
                    if element.style.font_family:
                        # Use custom font if it's monospace, otherwise keep Consolas
                        if 'mono' in element.style.font_family.lower() or 'consol' in element.style.font_family.lower():
                            run.font.name = element.style.font_family
                    
                    if element.style.font_size:
                        run.font.size = Pt(element.style.font_size)
            
            logger.debug(f"Added code block: {element.content[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to add code block: {str(e)}")
            # Fallback to regular paragraph
            await self.add_paragraph(document, element)
    
    async def add_quote(self, document: Document, element: ContentElement) -> None:
        """
        Add a quote element to the document.
        
        Args:
            document: DOCX document to add to
            element: Quote element to add
        """
        try:
            # Add quote with special formatting
            paragraph = document.add_paragraph(f'"{element.content}"')
            
            # Apply quote styling
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            for run in paragraph.runs:
                run.font.italic = True
                
            # Add some spacing
            paragraph.paragraph_format.space_before = Pt(6)
            paragraph.paragraph_format.space_after = Pt(6)
            
            # Apply custom styling if provided
            if element.style:
                if paragraph.runs:
                    self.style_manager.apply_text_style_to_run(paragraph.runs[0], element.style)
                self.style_manager.apply_paragraph_style(paragraph, element.style, alignment='center')
            
            logger.debug(f"Added quote: {element.content[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to add quote: {str(e)}")
            # Fallback to regular paragraph
            await self.add_paragraph(document, element)
    
    async def add_divider(self, document: Document, element: ContentElement) -> None:
        """
        Add a divider element to the document.
        
        Args:
            document: DOCX document to add to
            element: Divider element to add
        """
        try:
            # Add horizontal line as divider
            paragraph = document.add_paragraph()
            
            # Create horizontal line using borders
            paragraph_format = paragraph.paragraph_format
            paragraph_format.space_before = Pt(6)
            paragraph_format.space_after = Pt(6)
            
            # Add border bottom to create line effect
            await self._add_paragraph_border(paragraph, 'bottom')
            
            logger.debug("Added divider")
            
        except Exception as e:
            logger.error(f"Failed to add divider: {str(e)}")
            # Fallback to paragraph with dashes
            document.add_paragraph("─" * 50)
    
    async def _apply_custom_attributes(self, paragraph, attributes: dict) -> None:
        """Apply custom attributes to a paragraph."""
        try:
            if 'alignment' in attributes:
                alignment_map = {
                    'left': WD_ALIGN_PARAGRAPH.LEFT,
                    'center': WD_ALIGN_PARAGRAPH.CENTER,
                    'right': WD_ALIGN_PARAGRAPH.RIGHT,
                    'justify': WD_ALIGN_PARAGRAPH.JUSTIFY
                }
                alignment = attributes['alignment']
                if alignment in alignment_map:
                    paragraph.alignment = alignment_map[alignment]
            
            if 'indent' in attributes:
                # Apply indentation
                indent_value = attributes['indent']
                if isinstance(indent_value, (int, float)):
                    paragraph.paragraph_format.left_indent = Pt(indent_value)
            
        except Exception as e:
            logger.warning(f"Failed to apply custom attributes: {str(e)}")
    
    async def _apply_list_indentation(self, paragraph, level: int) -> None:
        """Apply indentation for nested list items."""
        try:
            # Each level adds 0.5 inches of indentation
            indent_amount = level * 0.5
            paragraph.paragraph_format.left_indent = Pt(indent_amount * 72)  # Convert inches to points
            
        except Exception as e:
            logger.warning(f"Failed to apply list indentation: {str(e)}")
    
    async def _add_paragraph_shading(self, paragraph, color: str) -> None:
        """Add background shading to a paragraph."""
        try:
            # This is a complex operation in python-docx
            # Simplified implementation - would need more advanced XML manipulation for full support
            logger.debug(f"Background shading requested: {color}")
            
        except Exception as e:
            logger.warning(f"Failed to add paragraph shading: {str(e)}")
    
    async def _add_paragraph_border(self, paragraph, position: str) -> None:
        """Add border to a paragraph."""
        try:
            # Add border using XML manipulation
            # This is a simplified implementation
            pPr = paragraph._element.get_or_add_pPr()
            
            # Create border element
            pBdr = OxmlElement('w:pBdr')
            
            if position == 'bottom':
                bottom_border = OxmlElement('w:bottom')
                bottom_border.set(qn('w:val'), 'single')
                bottom_border.set(qn('w:sz'), '6')  # Border size
                bottom_border.set(qn('w:space'), '1')
                bottom_border.set(qn('w:color'), '000000')  # Black border
                pBdr.append(bottom_border)
            
            pPr.append(pBdr)
            
        except Exception as e:
            logger.warning(f"Failed to add paragraph border: {str(e)}")
    
    def apply_element_formatting(self, element, docx_element, element_type: ContentType) -> None:
        """
        Apply general formatting to any document element.
        
        Args:
            element: Source ContentElement
            docx_element: DOCX element (paragraph, run, etc.)
            element_type: Type of content element
        """
        try:
            # Apply metadata-based formatting
            if element.metadata:
                # Apply any metadata-based styling
                pass
            
            # Apply custom attributes
            if element.custom_attributes:
                # Apply custom formatting based on attributes
                pass
                
        except Exception as e:
            logger.warning(f"Failed to apply element formatting: {str(e)}")


# Test functionality
async def test_element_handler():
    """Test element handler functionality."""
    from docx import Document
    from ...models import ContentElement, ContentType, ListData, ListItem, ListType, TextStyle, StyleWeight
    
    # Create test document
    document = Document()
    handler = DocxElementHandler()
    
    try:
        # Test heading
        heading_element = ContentElement(
            type=ContentType.HEADING,
            content="Test Heading",
            level=1,
            style=TextStyle(font_size=16, weight=StyleWeight.BOLD)
        )
        await handler.add_heading(document, heading_element)
        
        # Test paragraph
        paragraph_element = ContentElement(
            type=ContentType.PARAGRAPH,
            content="This is a test paragraph with some content."
        )
        await handler.add_paragraph(document, paragraph_element)
        
        # Test list
        list_element = ContentElement(
            type=ContentType.LIST,
            list_data=ListData(
                items=[
                    ListItem(content="First item"),
                    ListItem(content="Second item"),
                    ListItem(content="Third item")
                ],
                list_type=ListType.BULLET
            )
        )
        await handler.add_list(document, list_element)
        
        # Test code block
        code_element = ContentElement(
            type=ContentType.CODE,
            content="def hello_world():\n    print('Hello, World!')"
        )
        await handler.add_code_block(document, code_element)
        
        print("✅ Element handler test passed")
        return True
        
    except Exception as e:
        print(f"❌ Element handler test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_element_handler())