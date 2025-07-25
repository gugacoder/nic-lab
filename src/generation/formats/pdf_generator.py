"""
PDF Document Generator

This module implements the PDF document generator using ReportLab,
providing high-fidelity PDF output with precise layout control.
"""

import io
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import asyncio

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.platypus.flowables import Flowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4, LETTER, landscape
    from reportlab.lib.units import inch, cm, mm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")

from ..base import BaseDocumentGenerator, DocumentFormat, GenerationError
from ..models import DocumentContent, DocumentTemplate, GenerationOptions, ContentType, ContentElement
from .pdf.styles import PdfStyleManager
from .pdf.elements import PdfElementHandler
from .pdf.images import PdfImageHandler
from .pdf.layouts import PdfLayoutManager
from .pdf.flowables import PdfFlowableHandler

logger = logging.getLogger(__name__)


class PdfGenerator(BaseDocumentGenerator):
    """
    PDF document generator using ReportLab.
    
    This generator creates PDF documents from structured content with support
    for precise layouts, images, tables, headers/footers, and bookmarks.
    """
    
    def __init__(self):
        """Initialize the PDF generator."""
        super().__init__(DocumentFormat.PDF)
        
        # Initialize component handlers
        self.style_manager = PdfStyleManager()
        self.element_handler = PdfElementHandler()
        self.image_handler = PdfImageHandler()
        self.layout_manager = PdfLayoutManager()
        self.flowable_handler = PdfFlowableHandler()
        
        # Document creation settings
        self._current_doc: Optional[SimpleDocTemplate] = None
        self._story: List = []
        self._styles = getSampleStyleSheet()
        self._page_templates = []
        
    @property
    def supported_features(self) -> List[str]:
        """Return list of features supported by this generator."""
        return [
            "text", "paragraphs", "headings", "lists", "tables", 
            "images", "styles", "templates", "metadata", "headers_footers",
            "page_breaks", "bookmarks", "hyperlinks", "fonts", "colors",
            "precise_layout", "vector_graphics", "compression", "password_protection"
        ]
    
    @property
    def file_extension(self) -> str:
        """Return the file extension for PDF format."""
        return "pdf"
    
    @property
    def mime_type(self) -> str:
        """Return the MIME type for PDF format."""
        return "application/pdf"
    
    async def generate_document(
        self,
        content: DocumentContent,
        template: Optional[DocumentTemplate] = None,
        options: Optional[GenerationOptions] = None
    ) -> bytes:
        """
        Generate a PDF document from content and template.
        
        Args:
            content: Structured document content
            template: Optional template to apply
            options: Generation options and settings
            
        Returns:
            Generated PDF document as bytes
            
        Raises:
            GenerationError: If document generation fails
        """
        try:
            logger.info("Starting PDF document generation")
            
            # Validate inputs
            validation_errors = await self.validate_content(content)
            if validation_errors:
                raise GenerationError(f"Content validation failed: {', '.join(validation_errors)}")
            
            # Set generation options
            if options:
                self.set_generation_options(options)
            
            # Create PDF buffer
            pdf_buffer = io.BytesIO()
            
            # Configure page size and layout
            page_size = self._get_page_size(content.page_size, content.page_orientation)
            
            # Create document template
            self._current_doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=page_size,
                topMargin=cm * content.margins.get('top', 2.54),
                bottomMargin=cm * content.margins.get('bottom', 2.54),
                leftMargin=cm * content.margins.get('left', 2.54),
                rightMargin=cm * content.margins.get('right', 2.54),
                title=content.metadata.title or "Generated Document",
                author=content.metadata.author or "Document Generator",
                subject=content.metadata.subject or ""
            )
            
            # Apply template if provided
            if template:
                await self._apply_template(template)
            
            # Initialize story (document content)
            self._story = []
            
            # Generate document content
            await self._generate_content(content)
            
            # Build the PDF
            self._current_doc.build(
                self._story,
                onFirstPage=self._create_page_template(content, template, True),
                onLaterPages=self._create_page_template(content, template, False)
            )
            
            # Get PDF bytes
            pdf_bytes = pdf_buffer.getvalue()
            pdf_buffer.close()
            
            logger.info(f"PDF document generated successfully ({len(pdf_bytes)} bytes)")
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            raise GenerationError(f"Failed to generate PDF document: {str(e)}") from e
        finally:
            self._current_doc = None
            self._story = []
    
    async def generate_preview(
        self,
        content: DocumentContent,
        template: Optional[DocumentTemplate] = None,
        options: Optional[GenerationOptions] = None
    ) -> Dict[str, Any]:
        """
        Generate a preview representation of the PDF document.
        
        Args:
            content: Structured document content
            template: Optional template to apply
            options: Generation options and settings
            
        Returns:
            Preview data with structure information
        """
        try:
            # Create document structure preview without full rendering
            preview_data = {
                "format": "pdf",
                "page_count": self._estimate_page_count(content),
                "sections": [],
                "images": [],
                "tables": [],
                "metadata": {
                    "title": content.metadata.title,
                    "author": content.metadata.author,
                    "word_count": self._estimate_word_count(content),
                    "created": content.created_timestamp.isoformat(),
                    "file_size_estimate": self._estimate_file_size(content)
                },
                "template": template.name if template else "Default",
                "features_used": self._analyze_content_features(content),
                "page_layout": {
                    "size": content.page_size,
                    "orientation": content.page_orientation,
                    "margins": content.margins
                }
            }
            
            # Analyze sections for preview
            for i, section in enumerate(content.sections):
                section_preview = {
                    "index": i,
                    "title": section.title,
                    "level": section.level,
                    "element_count": len(section.elements),
                    "elements": []
                }
                
                # Preview first few elements
                for j, element in enumerate(section.elements[:5]):  # Preview first 5 elements
                    element_preview = {
                        "type": element.type.value,
                        "content_preview": element.content[:100] + "..." if len(element.content) > 100 else element.content
                    }
                    
                    if element.type == ContentType.IMAGE and element.image_data:
                        preview_data["images"].append({
                            "caption": element.image_data.caption,
                            "source": element.image_data.source_path or element.image_data.source_url,
                            "alignment": element.image_data.alignment.value
                        })
                    elif element.type == ContentType.TABLE and element.table_data:
                        preview_data["tables"].append({
                            "rows": len(element.table_data.rows),
                            "columns": len(element.table_data.rows[0].cells) if element.table_data.rows else 0,
                            "has_header": element.table_data.has_header
                        })
                    
                    section_preview["elements"].append(element_preview)
                
                preview_data["sections"].append(section_preview)
            
            return preview_data
            
        except Exception as e:
            logger.error(f"Preview generation failed: {str(e)}")
            return {
                "format": "pdf",
                "error": str(e),
                "page_count": 0,
                "sections": []
            }
    
    async def validate_content(self, content: DocumentContent) -> List[str]:
        """
        Validate content compatibility with PDF generator.
        
        Args:
            content: Document content to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic content validation
        if not content.sections:
            errors.append("Document must contain at least one section")
        
        # Check for unsupported elements or missing data
        for i, section in enumerate(content.sections):
            for j, element in enumerate(section.elements):
                if element.type == ContentType.IMAGE:
                    if not element.image_data:
                        errors.append(f"Section {i}, Element {j}: Image element missing image_data")
                    elif not any([
                        element.image_data.source_path,
                        element.image_data.source_url,
                        element.image_data.base64_data
                    ]):
                        errors.append(f"Section {i}, Element {j}: Image element missing valid source")
                
                elif element.type == ContentType.TABLE:
                    if not element.table_data or not element.table_data.rows:
                        errors.append(f"Section {i}, Element {j}: Table missing data or rows")
                    else:
                        # Check for consistent column counts
                        if element.table_data.rows:
                            first_row_cols = len(element.table_data.rows[0].cells)
                            for k, row in enumerate(element.table_data.rows[1:], 1):
                                if len(row.cells) != first_row_cols:
                                    errors.append(f"Section {i}, Element {j}: Table row {k} has inconsistent column count")
        
        # Validate page settings
        if content.page_size not in ["A4", "Letter", "Legal"]:
            errors.append(f"Unsupported page size: {content.page_size}")
        
        if content.page_orientation not in ["portrait", "landscape"]:
            errors.append(f"Unsupported page orientation: {content.page_orientation}")
        
        return errors
    
    async def estimate_generation_time(self, content: DocumentContent) -> float:
        """
        Estimate PDF generation time in seconds.
        
        Args:
            content: Document content to analyze
            
        Returns:
            Estimated generation time in seconds
        """
        # Base time for PDF creation
        base_time = 1.0
        
        # Time per element (rough estimates)
        element_count = sum(len(section.elements) for section in content.sections)
        element_time = element_count * 0.05  # 50ms per element (PDFs are more complex)
        
        # Additional time for images (PDF image embedding is expensive)
        image_count = sum(
            1 for section in content.sections
            for element in section.elements
            if element.type == ContentType.IMAGE
        )
        image_time = image_count * 1.0  # 1s per image
        
        # Additional time for tables (complex layout calculations)
        table_count = sum(
            1 for section in content.sections
            for element in section.elements
            if element.type == ContentType.TABLE
        )
        table_time = table_count * 0.5  # 500ms per table
        
        # Additional time for large documents (pagination overhead)
        estimated_pages = self._estimate_page_count(content)
        page_time = max(0, (estimated_pages - 10) * 0.1)  # Extra time for docs > 10 pages
        
        total_time = base_time + element_time + image_time + table_time + page_time
        return max(total_time, 0.2)  # Minimum 200ms
    
    def _get_page_size(self, page_size: str, orientation: str) -> tuple:
        """Get ReportLab page size tuple."""
        size_map = {
            "A4": A4,
            "Letter": LETTER,
            "Legal": (8.5*inch, 14*inch)
        }
        
        size = size_map.get(page_size, A4)
        
        if orientation == "landscape":
            return landscape(size)
        return size
    
    async def _apply_template(self, template: DocumentTemplate) -> None:
        """Apply template styles and configuration to document."""
        logger.info(f"Applying PDF template: {template.name}")
        
        # Apply styles through style manager
        await self.style_manager.apply_template_styles(self._styles, template)
        
        # Configure page layout through layout manager
        if template.layout:
            await self.layout_manager.apply_template_layout(self._current_doc, template.layout)
    
    async def _generate_content(self, content: DocumentContent) -> None:
        """Generate the main document content."""
        # Add document title if provided
        if content.metadata.title:
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=self._styles['Title'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER
            )
            self._story.append(Paragraph(content.metadata.title, title_style))
            self._story.append(Spacer(1, 20))
        
        # Process each section
        for section in content.sections:
            await self._generate_section(section)
    
    async def _generate_section(self, section) -> None:
        """Generate content for a document section."""
        # Add section title if provided
        if section.title:
            heading_style = self._get_heading_style(section.level)
            self._story.append(Paragraph(section.title, heading_style))
            self._story.append(Spacer(1, 12))
        
        # Add page break before if specified
        if section.page_break_before:
            self._story.append(PageBreak())
        
        # Process section elements
        for element in section.elements:
            await self._generate_element(element)
        
        # Process subsections recursively
        for subsection in section.subsections:
            await self._generate_section(subsection)
        
        # Add page break after if specified
        if section.page_break_after:
            self._story.append(PageBreak())
    
    async def _generate_element(self, element: ContentElement) -> None:
        """Generate content for a single element."""
        if element.type == ContentType.HEADING:
            await self.element_handler.add_heading(self._story, self._styles, element)
        elif element.type == ContentType.PARAGRAPH:
            await self.element_handler.add_paragraph(self._story, self._styles, element)
        elif element.type == ContentType.LIST:
            await self.element_handler.add_list(self._story, self._styles, element)
        elif element.type == ContentType.TABLE:
            await self.flowable_handler.add_table(self._story, element)
        elif element.type == ContentType.IMAGE:
            await self.image_handler.add_image(self._story, element)
        elif element.type == ContentType.CODE:
            await self.element_handler.add_code_block(self._story, self._styles, element)
        elif element.type == ContentType.QUOTE:
            await self.element_handler.add_quote(self._story, self._styles, element)
        elif element.type == ContentType.DIVIDER:
            await self.element_handler.add_divider(self._story, element)
        else:
            # Default to paragraph for unknown types
            await self.element_handler.add_paragraph(self._story, self._styles, element)
    
    def _get_heading_style(self, level: int) -> ParagraphStyle:
        """Get appropriate heading style for level."""
        style_names = ['Heading1', 'Heading2', 'Heading3', 'Heading4', 'Heading5', 'Heading6']
        style_name = style_names[min(level - 1, len(style_names) - 1)]
        
        if style_name in self._styles:
            return self._styles[style_name]
        
        # Create custom heading style if not available
        base_size = 16 - (level * 2)
        return ParagraphStyle(
            f'CustomHeading{level}',
            parent=self._styles['Normal'],
            fontSize=max(base_size, 10),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
    
    def _create_page_template(self, content: DocumentContent, template: Optional[DocumentTemplate], is_first_page: bool):
        """Create page template function for headers/footers."""
        def page_template(canvas_obj, doc):
            canvas_obj.saveState()
            
            # Add header if specified
            if content.header_content or (template and template.header_template):
                header_text = content.header_content or template.header_template
                canvas_obj.setFont('Helvetica', 9)
                canvas_obj.drawString(doc.leftMargin, doc.height + doc.topMargin - 10, header_text)
            
            # Add footer if specified
            if content.footer_content or (template and template.footer_template):
                footer_text = content.footer_content or template.footer_template
                canvas_obj.setFont('Helvetica', 9)
                canvas_obj.drawString(doc.leftMargin, doc.bottomMargin - 10, footer_text)
            
            # Add page number
            page_num = canvas_obj.getPageNumber()
            canvas_obj.setFont('Helvetica', 9)
            canvas_obj.drawRightString(
                doc.width + doc.leftMargin, 
                doc.bottomMargin - 10, 
                f"Page {page_num}"
            )
            
            canvas_obj.restoreState()
        
        return page_template
    
    def _estimate_page_count(self, content: DocumentContent) -> int:
        """Estimate page count for preview."""
        # More conservative estimation for PDF (tighter layouts)
        total_chars = sum(
            len(element.content) 
            for section in content.sections 
            for element in section.elements
        )
        
        # PDF typically fits more text per page than Word
        estimated_pages = max(1, total_chars // 3000)
        
        # Add pages for images and tables (PDFs handle these more efficiently)
        image_count = sum(
            1 for section in content.sections
            for element in section.elements
            if element.type == ContentType.IMAGE
        )
        table_count = sum(
            1 for section in content.sections
            for element in section.elements
            if element.type == ContentType.TABLE
        )
        
        # Images and tables in PDF are more space-efficient
        estimated_pages += (image_count // 3) + (table_count // 4)
        
        return estimated_pages
    
    def _estimate_word_count(self, content: DocumentContent) -> int:
        """Estimate word count for preview."""
        total_words = 0
        for section in content.sections:
            for element in section.elements:
                if element.content:
                    total_words += len(element.content.split())
        return total_words
    
    def _estimate_file_size(self, content: DocumentContent) -> str:
        """Estimate PDF file size."""
        # Base PDF overhead
        base_size = 50000  # ~50KB base
        
        # Size per page
        page_count = self._estimate_page_count(content)
        page_size = page_count * 15000  # ~15KB per page
        
        # Size per image (compressed)
        image_count = sum(
            1 for section in content.sections
            for element in section.elements
            if element.type == ContentType.IMAGE
        )
        image_size = image_count * 200000  # ~200KB per image (compressed)
        
        total_size = base_size + page_size + image_size
        
        # Convert to human readable
        if total_size < 1024 * 1024:  # Less than 1MB
            return f"{total_size // 1024} KB"
        else:
            return f"{total_size / (1024 * 1024):.1f} MB"
    
    def _analyze_content_features(self, content: DocumentContent) -> List[str]:
        """Analyze what features are used in the content."""
        features = set()
        
        for section in content.sections:
            for element in section.elements:
                features.add(element.type.value)
        
        # Add PDF-specific features
        features.add("precise_layout")
        if content.header_content or content.footer_content:
            features.add("headers_footers")
        if content.page_orientation == "landscape":
            features.add("landscape_orientation")
        
        return list(features)


# Test functionality
async def test_pdf_generator():
    """Test PDF generator functionality."""
    from ..models import DocumentContent, DocumentMetadata, DocumentSection, create_heading, create_paragraph
    
    # Create test content
    metadata = DocumentMetadata(
        title="Test PDF Document",
        author="Test Author",
        subject="Testing PDF Generation"
    )
    
    section = DocumentSection(
        title="Test Section",
        elements=[
            create_heading("Test Heading", level=1),
            create_paragraph("This is a test paragraph for PDF generation.")
        ]
    )
    
    content = DocumentContent(
        metadata=metadata,
        sections=[section]
    )
    
    # Test generator
    generator = PdfGenerator()
    
    # Test validation
    errors = await generator.validate_content(content)
    if errors:
        print(f"❌ Validation failed: {errors}")
        return False
    
    # Test preview generation
    preview = await generator.generate_preview(content)
    print(f"✅ Preview generated: {preview['page_count']} pages, {preview['metadata']['word_count']} words")
    
    # Test document generation
    try:
        doc_bytes = await generator.generate_document(content)
        print(f"✅ Document generated: {len(doc_bytes)} bytes")
        return True
    except Exception as e:
        print(f"❌ Document generation failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_pdf_generator())
    else:
        print("Usage: python -m src.generation.formats.pdf_generator test")