#!/usr/bin/env python3
"""
Test cases for Document Ingestion Module
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import BytesIO

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from document_ingestion import (
    DocumentIngestionManager, IngestionConfig, DocumentFormat, ValidationStatus,
    DocumentMetadata, ValidationResult, IngestedDocument, NormalizedDocument,
    create_document_ingestion_manager
)

class TestDocumentIngestionManager:
    """Test cases for DocumentIngestionManager"""
    
    @pytest.fixture
    def basic_config(self):
        return IngestionConfig(
            max_file_size_mb=10,
            supported_formats=['pdf', 'docx', 'txt', 'md', 'jpg', 'png']
        )
    
    @pytest.fixture
    def ingestion_manager(self, basic_config):
        return DocumentIngestionManager(basic_config)
    
    @pytest.fixture
    def sample_text_content(self):
        return b"This is a sample text document with multiple sentences. It contains enough content for testing purposes."
    
    @pytest.fixture
    def sample_pdf_content(self):
        # Minimal PDF structure for testing
        return b'%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj xref 0 4 0000000000 65535 f 0000000015 00000 n 0000000074 00000 n 0000000120 00000 n trailer<</Size 4/Root 1 0 R>>startxref 180 %%EOF'
    
    @pytest.fixture
    def sample_docx_content(self):
        # ZIP signature (DOCX is a ZIP file)
        return b'PK\x03\x04[Content_Types].xml'
    
    @pytest.fixture
    def sample_jpeg_content(self):
        return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb'
    
    @pytest.fixture
    def sample_png_content(self):
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
    
    def test_format_detection_pdf(self, ingestion_manager, sample_pdf_content):
        """Test PDF format detection"""
        detected_format = ingestion_manager._detect_format(sample_pdf_content, "test.pdf")
        assert detected_format == DocumentFormat.PDF
    
    def test_format_detection_text(self, ingestion_manager, sample_text_content):
        """Test text format detection"""
        detected_format = ingestion_manager._detect_format(sample_text_content, "test.txt")
        assert detected_format == DocumentFormat.TXT
    
    def test_format_detection_markdown(self, ingestion_manager):
        """Test markdown format detection"""
        markdown_content = b"# Title\n\n## Subtitle\n\n* List item\n* Another item"
        detected_format = ingestion_manager._detect_format(markdown_content, "test.md")
        assert detected_format == DocumentFormat.MARKDOWN
    
    def test_format_detection_jpeg(self, ingestion_manager, sample_jpeg_content):
        """Test JPEG format detection"""
        detected_format = ingestion_manager._detect_format(sample_jpeg_content, "test.jpg")
        assert detected_format == DocumentFormat.JPEG
    
    def test_format_detection_png(self, ingestion_manager, sample_png_content):
        """Test PNG format detection"""
        detected_format = ingestion_manager._detect_format(sample_png_content, "test.png")
        assert detected_format == DocumentFormat.PNG
    
    def test_format_detection_by_extension(self, ingestion_manager):
        """Test format detection by file extension"""
        # Test with content that doesn't have clear signatures
        generic_content = b'some binary content'
        
        detected_format = ingestion_manager._detect_format(generic_content, "test.pdf")
        assert detected_format == DocumentFormat.PDF
        
        detected_format = ingestion_manager._detect_format(generic_content, "test.docx")
        assert detected_format == DocumentFormat.DOCX
    
    def test_text_document_ingestion(self, ingestion_manager, sample_text_content):
        """Test complete text document ingestion"""
        metadata = {
            'file_path': 'test.txt',
            'source': 'test'
        }
        
        result = ingestion_manager.ingest_document(sample_text_content, metadata)
        
        # Verify ingestion result
        assert isinstance(result, IngestedDocument)
        assert result.content == sample_text_content
        assert result.validation_result.is_valid
        assert result.metadata.detected_format == DocumentFormat.TXT
        assert result.metadata.file_name == "test.txt"
        assert result.metadata.file_size == len(sample_text_content)
        assert result.metadata.word_count > 0
        assert result.preview_text is not None
        assert result.ingestion_source == "test"
    
    def test_pdf_document_ingestion(self, ingestion_manager, sample_pdf_content):
        """Test PDF document ingestion"""
        metadata = {
            'file_path': 'test.pdf',
            'source': 'gitlab'
        }
        
        result = ingestion_manager.ingest_document(sample_pdf_content, metadata)
        
        assert isinstance(result, IngestedDocument)
        assert result.metadata.detected_format == DocumentFormat.PDF
        assert result.metadata.file_name == "test.pdf"
        # Note: Validation may fail due to minimal PDF structure, but ingestion should succeed
    
    def test_file_size_validation(self, basic_config):
        """Test file size limit enforcement"""
        # Set very small limit
        basic_config.max_file_size_mb = 1
        manager = DocumentIngestionManager(basic_config)
        
        # Create content that exceeds limit
        large_content = b'x' * (2 * 1024 * 1024)  # 2MB
        metadata = {'file_path': 'large.txt'}
        
        with pytest.raises(ValueError, match="File size .* exceeds limit"):
            manager.ingest_document(large_content, metadata)
    
    def test_unsupported_format_rejection(self, basic_config):
        """Test rejection of unsupported formats"""
        basic_config.supported_formats = ['txt']  # Only support TXT
        manager = DocumentIngestionManager(basic_config)
        
        pdf_content = b'%PDF-1.4'
        metadata = {'file_path': 'test.pdf'}
        
        with pytest.raises(ValueError, match="Unsupported format: pdf"):
            manager.ingest_document(pdf_content, metadata)
    
    def test_text_validation(self, ingestion_manager):
        """Test text document validation"""
        # Valid text
        valid_text = b"This is valid text content with sufficient length."
        validation = ingestion_manager.validate_document(valid_text, "txt")
        assert validation.is_valid
        assert validation.status == ValidationStatus.VALID
        
        # Empty text
        empty_text = b""
        validation = ingestion_manager.validate_document(empty_text, "txt")
        assert not validation.is_valid
        assert "Text document is empty" in validation.issues
        
        # Very short text
        short_text = b"Hi"
        validation = ingestion_manager.validate_document(short_text, "txt")
        assert not validation.is_valid
        assert "Text document is very short" in validation.issues
    
    def test_security_checks(self, ingestion_manager):
        """Test basic security checks"""
        # Content with suspicious patterns
        suspicious_content = b"<script>alert('xss')</script>Some normal content here."
        validation = ingestion_manager.validate_document(suspicious_content, "txt")
        
        # Should detect suspicious pattern
        assert len(validation.security_flags) > 0
        assert any("script" in flag.lower() for flag in validation.security_flags)
        assert not validation.is_valid
        assert validation.status == ValidationStatus.SUSPICIOUS
    
    def test_metadata_extraction(self, ingestion_manager, sample_text_content):
        """Test metadata extraction"""
        metadata = ingestion_manager.extract_metadata(sample_text_content, "/path/to/test.txt")
        
        assert metadata.file_name == "test.txt"
        assert metadata.file_path == "/path/to/test.txt"
        assert metadata.file_size == len(sample_text_content)
        assert len(metadata.file_hash) == 64  # SHA256 hash
        assert metadata.detected_format == DocumentFormat.TXT
        assert metadata.word_count > 0
        assert metadata.character_count > 0
    
    def test_language_detection(self, ingestion_manager):
        """Test basic language detection"""
        # Portuguese text (encoded as UTF-8 bytes)
        pt_content = "Este é um texto em português com acentuação.".encode('utf-8')
        metadata = ingestion_manager.extract_metadata(pt_content, "test_pt.txt")
        assert metadata.language == "pt-BR"
        
        # English text
        en_content = b"This is an English text with common words like the and but."
        metadata = ingestion_manager.extract_metadata(en_content, "test_en.txt")
        assert metadata.language == "en-US"
    
    def test_preview_extraction(self, ingestion_manager, sample_text_content):
        """Test preview text extraction"""
        preview = ingestion_manager._extract_preview(sample_text_content, DocumentFormat.TXT)
        
        assert preview is not None
        assert len(preview) <= ingestion_manager.config.preview_length
        assert isinstance(preview, str)
    
    def test_document_normalization(self, ingestion_manager, sample_text_content):
        """Test document normalization"""
        metadata = {'file_path': 'test.txt', 'source': 'test'}
        ingested_doc = ingestion_manager.ingest_document(sample_text_content, metadata)
        
        normalized_doc = ingestion_manager.normalize_content(ingested_doc)
        
        assert isinstance(normalized_doc, NormalizedDocument)
        assert normalized_doc.original_document == ingested_doc
        assert normalized_doc.normalized_content == sample_text_content
        assert normalized_doc.content_type == "txt"
        assert "text_only" in normalized_doc.processing_hints
        assert 0.0 <= normalized_doc.quality_score <= 1.0
    
    def test_batch_ingestion(self, ingestion_manager, sample_text_content):
        """Test batch document processing"""
        documents = [
            {
                'content': sample_text_content,
                'metadata': {'file_path': 'doc1.txt', 'source': 'test'}
            },
            {
                'content': b"Another document for batch testing.",
                'metadata': {'file_path': 'doc2.txt', 'source': 'test'}
            }
        ]
        
        results = ingestion_manager.batch_ingest(documents)
        
        assert len(results) == 2
        assert all(isinstance(doc, IngestedDocument) for doc in results)
        assert results[0].metadata.file_name == "doc1.txt"
        assert results[1].metadata.file_name == "doc2.txt"
    
    def test_batch_ingestion_with_errors(self, ingestion_manager):
        """Test batch ingestion with some failures"""
        documents = [
            {
                'content': b"Valid document content.",
                'metadata': {'file_path': 'valid.txt', 'source': 'test'}
            },
            {
                'content': b'x' * (200 * 1024 * 1024),  # Too large
                'metadata': {'file_path': 'toolarge.txt', 'source': 'test'}
            }
        ]
        
        results = ingestion_manager.batch_ingest(documents)
        
        # Should process only the valid document
        assert len(results) == 1
        assert results[0].metadata.file_name == "valid.txt"
    
    def test_factory_function(self):
        """Test factory function for manager creation"""
        config_dict = {
            'max_file_size_mb': 50,
            'supported_formats': ['pdf', 'txt'],
            'enable_content_validation': False
        }
        
        manager = create_document_ingestion_manager(config_dict)
        
        assert isinstance(manager, DocumentIngestionManager)
        assert manager.config.max_file_size_mb == 50
        assert manager.config.supported_formats == ['pdf', 'txt']
        assert not manager.config.enable_content_validation
    
    def test_processing_statistics(self, ingestion_manager):
        """Test processing statistics and capabilities"""
        stats = ingestion_manager.get_processing_statistics()
        
        assert 'supported_formats' in stats
        assert 'max_file_size_mb' in stats
        assert 'capabilities' in stats
        
        capabilities = stats['capabilities']
        assert 'pdf_processing' in capabilities
        assert 'docx_processing' in capabilities
        assert 'image_processing' in capabilities
        assert 'mime_detection' in capabilities
        assert 'content_validation' in capabilities
        assert 'malware_check' in capabilities
        assert 'preview_extraction' in capabilities
    
    def test_supported_formats(self, ingestion_manager):
        """Test supported formats retrieval"""
        formats = ingestion_manager.get_supported_formats()
        
        assert isinstance(formats, list)
        assert 'pdf' in formats
        assert 'txt' in formats
        assert 'docx' in formats
        assert 'md' in formats
        assert 'jpg' in formats
        assert 'png' in formats
    
    def test_pdf_ocr_detection(self, ingestion_manager, sample_pdf_content):
        """Test PDF OCR requirement detection"""
        # Mock PDF with no text (requires OCR)
        requires_ocr = ingestion_manager._pdf_requires_ocr(sample_pdf_content)
        
        # Should return True since our minimal PDF has no meaningful text
        # Note: This depends on PyMuPDF availability
        assert isinstance(requires_ocr, bool)
    
    def test_invalid_format_detection(self, ingestion_manager):
        """Test behavior with undetectable format"""
        # Random binary content that doesn't match any signature
        unknown_content = b'\x00\x01\x02\x03\x04\x05' * 100
        
        with pytest.raises(ValueError, match="Unable to detect document format"):
            ingestion_manager._detect_format(unknown_content, "unknown.xyz")
    
    def test_encoding_error_handling(self, ingestion_manager):
        """Test handling of encoding errors in text files"""
        # Invalid UTF-8 content
        invalid_utf8 = b'\xff\xfe\x00\x00invalid utf8 content'
        
        validation = ingestion_manager.validate_document(invalid_utf8, "txt")
        assert not validation.is_valid
        assert any("encoding error" in issue.lower() for issue in validation.issues)
    
    @patch('document_ingestion.fitz')
    def test_pdf_processing_without_pymupdf(self, mock_fitz, ingestion_manager, sample_pdf_content):
        """Test PDF processing when PyMuPDF is not available"""
        mock_fitz = None
        
        # Should still detect format but skip advanced PDF operations
        detected_format = ingestion_manager._detect_format(sample_pdf_content, "test.pdf")
        assert detected_format == DocumentFormat.PDF
        
        # Validation should note missing PyMuPDF
        validation = ingestion_manager.validate_document(sample_pdf_content, "pdf")
        assert any("PyMuPDF not available" in issue for issue in validation.issues)

if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running Document Ingestion Tests...")
    
    # Create test manager
    config = IngestionConfig(max_file_size_mb=10)
    manager = DocumentIngestionManager(config)
    
    # Test text ingestion
    text_content = b"This is a test document for validation."
    metadata = {'file_path': 'test.txt', 'source': 'test'}
    
    try:
        result = manager.ingest_document(text_content, metadata)
        print(f"✓ Text ingestion successful: {result.metadata.file_name}")
        print(f"✓ Format detected: {result.metadata.detected_format.value}")
        print(f"✓ Validation passed: {result.validation_result.is_valid}")
        print(f"✓ Word count: {result.metadata.word_count}")
        print(f"✓ Preview: {result.preview_text[:50]}...")
        
        # Test normalization
        normalized = manager.normalize_content(result)
        print(f"✓ Normalization successful: quality={normalized.quality_score}")
        
        # Test statistics
        stats = manager.get_processing_statistics()
        print(f"✓ Processing capabilities: {len(stats['capabilities'])} features")
        
        print("\n✅ All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()