"""
Comprehensive Tests for DOCX Generator

This module provides thorough testing of the DOCX document generation
functionality including content elements, styling, and template application.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from io import BytesIO

from src.generation.formats.docx_generator import DocxGenerator
from src.generation.models import (
    DocumentContent, DocumentMetadata, DocumentSection, ContentElement,
    ContentType, ImageData, ImageAlignment, TableData, TableRow, TableCell,
    ListData, ListItem, ListType, TextStyle, StyleWeight, StyleEmphasis,
    create_heading, create_paragraph, create_image, create_table, create_list
)
from templates.docx.default_template import get_docx_template


class TestDocxGenerator:
    """Test suite for DOCX generator functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create DOCX generator instance."""
        return DocxGenerator()
    
    @pytest.fixture
    def sample_content(self):
        """Create sample document content for testing."""
        metadata = DocumentMetadata(
            title="Test Document",
            author="Test Author", 
            subject="DOCX Generation Testing",
            description="Comprehensive test of DOCX generation capabilities",
            keywords=["test", "docx", "generation"],
            language="en"
        )
        
        # Create content elements
        elements = [
            create_heading("Introduction", level=1),
            create_paragraph("This is a test document generated to verify DOCX functionality."),
            create_heading("Features", level=2),
            create_paragraph("The following features are being tested:"),
            create_list([
                "Text formatting and styling",
                "Heading hierarchy",
                "Table generation",
                "Image embedding"
            ], ListType.BULLET),
            create_heading("Sample Table", level=2),
            create_table([
                ["Feature", "Status", "Notes"],
                ["Text", "✓", "Working correctly"],
                ["Images", "✓", "Placeholder support"],
                ["Tables", "✓", "Full formatting"]
            ], has_header=True)
        ]
        
        section = DocumentSection(
            title="Main Content",
            elements=elements
        )
        
        return DocumentContent(
            metadata=metadata,
            sections=[section]
        )
    
    def test_generator_properties(self, generator):
        """Test generator basic properties."""
        assert generator.file_extension == "docx"
        assert generator.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert "text" in generator.supported_features
        assert "images" in generator.supported_features
        assert "tables" in generator.supported_features
    
    @pytest.mark.asyncio
    async def test_content_validation(self, generator, sample_content):
        """Test content validation functionality."""
        # Valid content should pass
        errors = await generator.validate_content(sample_content)
        assert len(errors) == 0
        
        # Empty content should fail
        empty_content = DocumentContent(sections=[])
        errors = await generator.validate_content(empty_content)
        assert len(errors) > 0
        assert any("at least one section" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_generation_time_estimation(self, generator, sample_content):
        """Test generation time estimation."""
        estimated_time = await generator.estimate_generation_time(sample_content)
        assert isinstance(estimated_time, float)
        assert estimated_time > 0
        assert estimated_time < 60  # Should be reasonable
    
    @pytest.mark.asyncio
    async def test_preview_generation(self, generator, sample_content):
        """Test preview generation functionality."""
        preview = await generator.generate_preview(sample_content)
        
        assert preview["format"] == "docx"
        assert preview["page_count"] > 0
        assert len(preview["sections"]) > 0
        assert preview["metadata"]["title"] == "Test Document"
        assert preview["metadata"]["word_count"] > 0
        assert "paragraph" in preview["features_used"]
        assert "table" in preview["features_used"]
    
    @pytest.mark.asyncio
    async def test_basic_document_generation(self, generator, sample_content):
        """Test basic document generation without template."""
        doc_bytes = await generator.generate_document(sample_content)
        
        assert isinstance(doc_bytes, bytes)
        assert len(doc_bytes) > 1000  # Should be substantial
        
        # Verify it's a valid ZIP file (DOCX is ZIP-based)
        from zipfile import ZipFile, is_zipfile
        assert is_zipfile(BytesIO(doc_bytes))
        
        # Verify DOCX structure
        with ZipFile(BytesIO(doc_bytes)) as docx_zip:
            files = docx_zip.namelist()
            assert 'word/document.xml' in files
            assert '[Content_Types].xml' in files
    
    @pytest.mark.asyncio
    async def test_template_application(self, generator, sample_content):
        """Test document generation with template."""
        template = get_docx_template("default")
        
        doc_bytes = await generator.generate_document(
            sample_content, 
            template=template
        )
        
        assert isinstance(doc_bytes, bytes)
        assert len(doc_bytes) > 1000
        
        # Template application should still produce valid DOCX
        from zipfile import is_zipfile
        assert is_zipfile(BytesIO(doc_bytes))
    
    @pytest.mark.asyncio
    async def test_image_handling(self, generator):
        """Test image element handling."""
        # Create content with image (placeholder test)
        image_element = ContentElement(
            type=ContentType.IMAGE,
            image_data=ImageData(
                source_type="file",
                source_path="nonexistent.jpg",  # Will create placeholder
                caption="Test Image",
                width=400,
                height=300,
                alignment=ImageAlignment.CENTER
            )
        )
        
        section = DocumentSection(
            title="Image Test",
            elements=[image_element]
        )
        
        content = DocumentContent(
            metadata=DocumentMetadata(title="Image Test"),
            sections=[section]
        )
        
        # Should handle missing image gracefully
        doc_bytes = await generator.generate_document(content)
        assert isinstance(doc_bytes, bytes)
        assert len(doc_bytes) > 500
    
    @pytest.mark.asyncio
    async def test_table_generation(self, generator):
        """Test table generation with various configurations."""
        # Complex table with merged cells
        table_data = TableData(
            rows=[
                TableRow(
                    cells=[
                        TableCell(content="Header 1", background_color="#E6E6FA"),
                        TableCell(content="Header 2", background_color="#E6E6FA"),
                        TableCell(content="Header 3", background_color="#E6E6FA")
                    ],
                    is_header=True
                ),
                TableRow(
                    cells=[
                        TableCell(content="Row 1, Col 1"),
                        TableCell(content="Merged Cell", colspan=2, alignment="center")
                    ]
                ),
                TableRow(
                    cells=[
                        TableCell(content="Row 2, Col 1"),
                        TableCell(content="Row 2, Col 2"),
                        TableCell(content="Row 2, Col 3")
                    ]
                )
            ],
            caption="Complex Test Table",
            has_header=True,
            border_width=2,
            border_color="#000000"
        )
        
        table_element = ContentElement(
            type=ContentType.TABLE,
            table_data=table_data
        )
        
        section = DocumentSection(
            title="Table Test",
            elements=[table_element]
        )
        
        content = DocumentContent(
            metadata=DocumentMetadata(title="Table Test"),
            sections=[section]
        )
        
        doc_bytes = await generator.generate_document(content)
        assert isinstance(doc_bytes, bytes)
        assert len(doc_bytes) > 1000
    
    @pytest.mark.asyncio
    async def test_list_generation(self, generator):
        """Test list generation with different types."""
        elements = [
            # Bullet list
            ContentElement(
                type=ContentType.LIST,
                list_data=ListData(
                    items=[
                        ListItem(content="Bullet item 1"),
                        ListItem(content="Bullet item 2", level=1),  # Nested
                        ListItem(content="Bullet item 3")
                    ],
                    list_type=ListType.BULLET
                )
            ),
            
            # Numbered list
            ContentElement(
                type=ContentType.LIST,
                list_data=ListData(
                    items=[
                        ListItem(content="Numbered item 1"),
                        ListItem(content="Numbered item 2"),
                        ListItem(content="Numbered item 3")
                    ],
                    list_type=ListType.NUMBERED,
                    start_number=1
                )
            ),
            
            # Checklist
            ContentElement(
                type=ContentType.LIST,
                list_data=ListData(
                    items=[
                        ListItem(content="Completed task", checked=True),
                        ListItem(content="Pending task", checked=False),
                        ListItem(content="Another task", checked=True)
                    ],
                    list_type=ListType.CHECKLIST
                )
            )
        ]
        
        section = DocumentSection(
            title="List Test",
            elements=elements
        )
        
        content = DocumentContent(
            metadata=DocumentMetadata(title="List Test"),
            sections=[section]
        )
        
        doc_bytes = await generator.generate_document(content)
        assert isinstance(doc_bytes, bytes)
        assert len(doc_bytes) > 800
    
    @pytest.mark.asyncio
    async def test_styled_content(self, generator):
        """Test content with custom styling."""
        styled_elements = [
            ContentElement(
                type=ContentType.HEADING,
                content="Styled Heading",
                level=1,
                style=TextStyle(
                    font_family="Arial",
                    font_size=18,
                    weight=StyleWeight.BOLD,
                    color="#0066CC"
                )
            ),
            
            ContentElement(
                type=ContentType.PARAGRAPH,
                content="This paragraph has custom styling with italic emphasis.",
                style=TextStyle(
                    font_family="Georgia",
                    font_size=12,
                    emphasis=StyleEmphasis.ITALIC,
                    color="#333333"
                )
            ),
            
            ContentElement(
                type=ContentType.CODE,
                content="def test_function():\n    return 'Hello, World!'",
                style=TextStyle(
                    font_family="Consolas",
                    font_size=10,
                    background_color="#F0F0F0"
                )
            ),
            
            ContentElement(
                type=ContentType.QUOTE,
                content="This is a styled quote block with custom formatting.",
                style=TextStyle(
                    emphasis=StyleEmphasis.ITALIC,
                    color="#666666"
                )
            )
        ]
        
        section = DocumentSection(
            title="Styling Test",
            elements=styled_elements
        )
        
        content = DocumentContent(
            metadata=DocumentMetadata(title="Styling Test"),
            sections=[section]
        )
        
        doc_bytes = await generator.generate_document(content)
        assert isinstance(doc_bytes, bytes)
        assert len(doc_bytes) > 1000
    
    @pytest.mark.asyncio
    async def test_error_handling(self, generator):
        """Test error handling and recovery."""
        # Test with invalid image
        invalid_image = ContentElement(
            type=ContentType.IMAGE,
            image_data=ImageData(
                source_type="invalid",
                source_path=None,
                source_url=None,
                base64_data=None
            )
        )
        
        section = DocumentSection(
            title="Error Test",
            elements=[invalid_image]
        )
        
        content = DocumentContent(
            metadata=DocumentMetadata(title="Error Test"),
            sections=[section]
        )
        
        # Should handle errors gracefully
        doc_bytes = await generator.generate_document(content)
        assert isinstance(doc_bytes, bytes)
        assert len(doc_bytes) > 500  # Should still generate document
    
    @pytest.mark.asyncio
    async def test_large_document(self, generator):
        """Test generation of larger documents."""
        # Create document with many sections and elements
        sections = []
        
        for i in range(10):  # 10 sections
            elements = [
                create_heading(f"Section {i+1}", level=2),
                create_paragraph(f"This is section {i+1} content. " * 20),  # Long paragraph
                create_list([f"Item {j+1}" for j in range(5)], ListType.BULLET)
            ]
            
            sections.append(DocumentSection(
                title=f"Section {i+1}",
                elements=elements
            ))
        
        content = DocumentContent(
            metadata=DocumentMetadata(title="Large Document Test"),
            sections=sections
        )
        
        start_time = asyncio.get_event_loop().time()
        doc_bytes = await generator.generate_document(content)
        generation_time = asyncio.get_event_loop().time() - start_time
        
        assert isinstance(doc_bytes, bytes)
        assert len(doc_bytes) > 5000  # Should be substantial
        assert generation_time < 30  # Should complete in reasonable time
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self, generator, sample_content):
        """Test generation with progress tracking."""
        progress_updates = []
        
        async for progress in generator.generate_with_progress(sample_content):
            progress_updates.append(progress)
        
        assert len(progress_updates) >= 2  # Should have multiple updates
        assert progress_updates[0].progress_percent == 0.0
        assert progress_updates[-1].progress_percent == 100.0
        assert progress_updates[-1].status.value == "completed"
    
    def test_validation_errors(self, generator):
        """Test content validation error detection."""
        # Content with validation issues - no sections
        invalid_content = DocumentContent(sections=[])
        
        # Should detect validation errors
        async def run_validation():
            errors = await generator.validate_content(invalid_content)
            return errors
        
        errors = asyncio.run(run_validation())
        assert len(errors) > 0
        assert any("at least one section" in error.lower() for error in errors)
        
        # Test with image missing image_data
        invalid_content2 = DocumentContent(
            sections=[
                DocumentSection(
                    title="Invalid Section",
                    elements=[
                        ContentElement(
                            type=ContentType.IMAGE,
                            image_data=None  # Missing image_data
                        )
                    ]
                )
            ]
        )
        
        errors2 = asyncio.run(run_validation())
        async def run_validation2():
            errors = await generator.validate_content(invalid_content2)
            return errors
        
        errors2 = asyncio.run(run_validation2())
        assert len(errors2) > 0
        assert any("missing image_data" in error.lower() for error in errors2)


# Integration tests
class TestDocxIntegration:
    """Integration tests for DOCX generator with other components."""
    
    @pytest.mark.asyncio
    async def test_template_integration(self):
        """Test integration with template system."""
        generator = DocxGenerator()
        
        # Test all available templates
        from templates.docx.default_template import list_docx_templates, get_docx_template
        
        templates = list_docx_templates()
        assert len(templates) > 0
        
        for template_name in templates:
            template = get_docx_template(template_name)
            assert template.name is not None
            assert template.format_type == "docx"
    
    @pytest.mark.asyncio
    async def test_file_output(self):
        """Test saving generated document to file."""
        generator = DocxGenerator()
        
        content = DocumentContent(
            metadata=DocumentMetadata(title="File Output Test"),
            sections=[
                DocumentSection(
                    title="Test Section",
                    elements=[create_paragraph("Test content for file output.")]
                )
            ]
        )
        
        doc_bytes = await generator.generate_document(content)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
            tmp_file.write(doc_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Verify file was created and is valid
            assert Path(tmp_path).exists()
            assert Path(tmp_path).stat().st_size > 1000
            
            # Verify it's a valid DOCX file
            from zipfile import is_zipfile
            assert is_zipfile(tmp_path)
            
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


# Performance tests
class TestDocxPerformance:
    """Performance tests for DOCX generation."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_generation_performance(self):
        """Test generation performance with various document sizes."""
        generator = DocxGenerator()
        
        # Test different document sizes
        test_cases = [
            ("small", 1, 5),     # 1 section, 5 elements
            ("medium", 5, 20),   # 5 sections, 20 elements each
            ("large", 10, 50)    # 10 sections, 50 elements each
        ]
        
        for test_name, section_count, element_count in test_cases:
            # Create test content
            sections = []
            for i in range(section_count):
                elements = []
                for j in range(element_count):
                    elements.append(create_paragraph(f"Test paragraph {j+1} content. " * 5))
                
                sections.append(DocumentSection(
                    title=f"Section {i+1}",
                    elements=elements
                ))
            
            content = DocumentContent(
                metadata=DocumentMetadata(title=f"Performance Test - {test_name}"),
                sections=sections
            )
            
            # Measure generation time
            start_time = asyncio.get_event_loop().time()
            doc_bytes = await generator.generate_document(content)
            generation_time = asyncio.get_event_loop().time() - start_time
            
            # Verify results
            assert isinstance(doc_bytes, bytes)
            assert len(doc_bytes) > 1000
            
            # Performance assertions (adjust as needed)
            max_times = {"small": 2.0, "medium": 10.0, "large": 30.0}
            assert generation_time < max_times[test_name], f"{test_name} generation took {generation_time:.2f}s"
            
            print(f"✅ {test_name.capitalize()} document: {generation_time:.2f}s, {len(doc_bytes)} bytes")


# Run specific tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])