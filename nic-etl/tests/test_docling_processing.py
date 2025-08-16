import pytest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from docling_processing import (
    DoclingProcessor, DocumentResult, ProcessingMetrics,
    create_docling_processor, validate_docling_config,
    DocumentProcessingContext
)

class TestDoclingProcessor:
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for Docling processing"""
        return {
            'ocr_engine': 'easyocr',
            'confidence_threshold': 0.8,
            'max_file_size_mb': 100,
            'enable_table_extraction': True,
            'enable_figure_extraction': True,
            'output_format': 'json',
            'quality_gates_enabled': True,
            'cache_processed_documents': True
        }
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing"""
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    
    @pytest.fixture
    def temp_document(self, sample_pdf_content):
        """Create temporary document file"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(sample_pdf_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_docling_processor_initialization(self, mock_config):
        """Test DoclingProcessor initialization"""
        processor = DoclingProcessor(mock_config)
        
        assert processor.config == mock_config
        assert processor.ocr_engine == 'easyocr'
        assert processor.confidence_threshold == 0.8
        assert processor.max_file_size_mb == 100
        assert processor.enable_table_extraction is True
        assert processor.quality_gates_enabled is True
    
    def test_config_validation(self, mock_config):
        """Test configuration validation"""
        # Valid config
        errors = validate_docling_config(mock_config)
        assert len(errors) == 0
        
        # Invalid config - missing required fields
        invalid_config = {}
        errors = validate_docling_config(invalid_config)
        assert len(errors) > 0
        
        # Invalid config - bad values
        bad_config = mock_config.copy()
        bad_config['confidence_threshold'] = 1.5  # Invalid threshold
        errors = validate_docling_config(bad_config)
        assert len(errors) > 0
    
    @patch('docling_processing.DoclingProcessor._process_with_docling')
    def test_process_document_success(self, mock_docling, mock_config, temp_document):
        """Test successful document processing"""
        # Mock Docling response
        mock_docling_result = {
            'content': 'Sample document content',
            'metadata': {'pages': 1, 'language': 'en'},
            'structure': {'paragraphs': ['Sample content']},
            'tables': [],
            'figures': []
        }
        mock_docling.return_value = mock_docling_result
        
        processor = DoclingProcessor(mock_config)
        result = processor.process_document(temp_document)
        
        assert isinstance(result, DocumentResult)
        assert result.success is True
        assert result.content == 'Sample document content'
        assert result.metadata['pages'] == 1
        assert result.processing_time > 0
        assert len(result.quality_metrics) > 0
    
    @patch('docling_processing.DoclingProcessor._process_with_docling')
    def test_process_document_failure(self, mock_docling, mock_config, temp_document):
        """Test document processing failure"""
        # Mock Docling failure
        mock_docling.side_effect = Exception("Processing failed")
        
        processor = DoclingProcessor(mock_config)
        result = processor.process_document(temp_document)
        
        assert isinstance(result, DocumentResult)
        assert result.success is False
        assert result.error is not None
        assert "Processing failed" in result.error
    
    def test_file_size_validation(self, mock_config):
        """Test file size validation"""
        # Create processor with small max file size
        config = mock_config.copy()
        config['max_file_size_mb'] = 1  # 1MB limit
        processor = DoclingProcessor(config)
        
        # Create large temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write 2MB of data
            f.write(b'x' * (2 * 1024 * 1024))
            large_file = f.name
        
        try:
            result = processor.process_document(large_file)
            assert result.success is False
            assert "file size" in result.error.lower()
        finally:
            Path(large_file).unlink(missing_ok=True)
    
    def test_unsupported_file_format(self, mock_config):
        """Test handling of unsupported file formats"""
        processor = DoclingProcessor(mock_config)
        
        # Create file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b'test content')
            unsupported_file = f.name
        
        try:
            result = processor.process_document(unsupported_file)
            assert result.success is False
            assert "unsupported" in result.error.lower() or "format" in result.error.lower()
        finally:
            Path(unsupported_file).unlink(missing_ok=True)
    
    @patch('docling_processing.DoclingProcessor._process_with_docling')
    def test_quality_gates(self, mock_docling, mock_config, temp_document):
        """Test quality gate validation"""
        # Mock low-quality result
        mock_docling_result = {
            'content': 'ab',  # Very short content
            'metadata': {'confidence': 0.3},  # Low confidence
            'structure': {'paragraphs': []},
            'tables': [],
            'figures': []
        }
        mock_docling.return_value = mock_docling_result
        
        processor = DoclingProcessor(mock_config)
        result = processor.process_document(temp_document)
        
        # Should pass through but flag quality issues
        assert isinstance(result, DocumentResult)
        # Quality gates should detect issues
        assert any('content_length' in metric for metric in result.quality_metrics)
    
    def test_cache_functionality(self, mock_config, temp_document):
        """Test document caching functionality"""
        config = mock_config.copy()
        config['cache_processed_documents'] = True
        
        processor = DoclingProcessor(config)
        
        # Mock cache directory
        with patch('docling_processing.DoclingProcessor._get_cache_path') as mock_cache:
            cache_dir = Path(tempfile.mkdtemp())
            mock_cache.return_value = cache_dir / "test_cache.json"
            
            # Mock the actual docling processing
            with patch.object(processor, '_process_with_docling') as mock_process:
                mock_process.return_value = {
                    'content': 'cached content',
                    'metadata': {},
                    'structure': {},
                    'tables': [],
                    'figures': []
                }
                
                # First call should process and cache
                result1 = processor.process_document(temp_document)
                assert mock_process.call_count == 1
                
                # Second call should use cache
                result2 = processor.process_document(temp_document)
                assert mock_process.call_count == 1  # Should not increase
    
    def test_processing_metrics(self, mock_config):
        """Test processing metrics calculation"""
        processor = DoclingProcessor(mock_config)
        
        # Test metrics calculation
        mock_result = DocumentResult(
            file_path='/test/doc.pdf',
            success=True,
            content='Test content',
            metadata={'pages': 2},
            structure={'paragraphs': ['p1', 'p2']},
            processing_time=1.5
        )
        
        metrics = processor.calculate_processing_metrics([mock_result])
        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.total_documents == 1
        assert metrics.successful_documents == 1
        assert metrics.failed_documents == 0
        assert metrics.average_processing_time == 1.5
        assert metrics.total_pages == 2
    
    def test_context_manager(self, mock_config):
        """Test DocumentProcessingContext context manager"""
        with DocumentProcessingContext(mock_config) as processor:
            assert isinstance(processor, DoclingProcessor)
            assert processor.config == mock_config
    
    def test_factory_function(self, mock_config):
        """Test factory function for creating processor"""
        processor = create_docling_processor(mock_config)
        assert isinstance(processor, DoclingProcessor)
        assert processor.config == mock_config
    
    def test_batch_processing(self, mock_config):
        """Test batch document processing"""
        processor = DoclingProcessor(mock_config)
        
        # Create multiple temp files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                f.write(f'Document {i} content'.encode())
                temp_files.append(f.name)
        
        try:
            with patch.object(processor, '_process_with_docling') as mock_process:
                mock_process.return_value = {
                    'content': 'processed content',
                    'metadata': {},
                    'structure': {},
                    'tables': [],
                    'figures': []
                }
                
                results = processor.process_documents(temp_files)
                assert len(results) == 3
                assert all(isinstance(r, DocumentResult) for r in results)
                assert mock_process.call_count == 3
        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)
    
    def test_error_recovery(self, mock_config, temp_document):
        """Test error recovery and fallback processing"""
        processor = DoclingProcessor(mock_config)
        
        # Mock primary processing failure, fallback success
        with patch.object(processor, '_process_with_docling') as mock_docling:
            mock_docling.side_effect = Exception("Primary processing failed")
            
            with patch.object(processor, '_fallback_text_extraction') as mock_fallback:
                mock_fallback.return_value = "Fallback extracted text"
                
                result = processor.process_document(temp_document)
                
                # Should succeed with fallback
                assert result.success is True
                assert result.content == "Fallback extracted text"
                assert "fallback" in result.processing_method.lower()

class TestDoclingProcessingIntegration:
    """Integration tests for Docling processing"""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end document processing"""
        config = {
            'ocr_engine': 'easyocr',
            'confidence_threshold': 0.7,
            'max_file_size_mb': 50,
            'enable_table_extraction': True,
            'enable_figure_extraction': False,
            'output_format': 'json',
            'quality_gates_enabled': True,
            'cache_processed_documents': False
        }
        
        # Create test document
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as f:
            f.write("This is a test document with some content for processing.")
            test_file = f.name
        
        try:
            processor = create_docling_processor(config)
            
            # Mock the actual Docling call since we don't have it installed
            with patch.object(processor, '_process_with_docling') as mock_process:
                mock_process.return_value = {
                    'content': 'This is a test document with some content for processing.',
                    'metadata': {'pages': 1, 'language': 'en', 'confidence': 0.95},
                    'structure': {
                        'paragraphs': ['This is a test document with some content for processing.']
                    },
                    'tables': [],
                    'figures': []
                }
                
                result = processor.process_document(test_file)
                
                assert result.success is True
                assert len(result.content) > 0
                assert result.metadata['pages'] == 1
                assert result.processing_time > 0
                assert len(result.quality_metrics) > 0
                
        finally:
            Path(test_file).unlink(missing_ok=True)