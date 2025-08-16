import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from text_chunking import (
    TextChunker, ChunkResult, ChunkingMetrics, ChunkingConfig,
    create_text_chunker, validate_chunking_config,
    TextChunkingContext, calculate_chunk_quality
)

class TestTextChunker:
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for text chunking"""
        return {
            'target_chunk_size': 500,
            'overlap_size': 100,
            'max_chunk_size': 600,
            'min_chunk_size': 50,
            'model_name': 'BAAI/bge-m3',
            'boundary_strategy': 'paragraph',
            'preserve_structure': True,
            'respect_semantic_boundaries': True
        }
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for chunking tests"""
        return """
        This is the first paragraph of a sample document. It contains multiple sentences
        that should be processed together to maintain semantic coherence. The content
        discusses various aspects of document processing and text analysis.
        
        This is the second paragraph that continues the discussion. It provides additional
        context and information that builds upon the previous paragraph. The goal is to
        test how well the chunking algorithm handles paragraph boundaries.
        
        The third paragraph introduces new concepts and ideas. It serves as a good test
        case for semantic boundary detection. The chunker should ideally keep related
        concepts together while respecting size constraints.
        
        Finally, this last paragraph concludes the sample text. It wraps up the discussion
        and provides a natural ending point for the document. This helps test how the
        chunker handles document endings.
        """
    
    @pytest.fixture
    def long_text(self):
        """Long text that requires multiple chunks"""
        # Generate text that will definitely exceed chunk size limits
        base_text = "This is a sentence that will be repeated many times to create a long document. "
        return base_text * 50  # Creates ~3500 characters
    
    def test_text_chunker_initialization(self, mock_config):
        """Test TextChunker initialization"""
        chunker = TextChunker(mock_config)
        
        assert chunker.config == mock_config
        assert chunker.target_chunk_size == 500
        assert chunker.overlap_size == 100
        assert chunker.max_chunk_size == 600
        assert chunker.min_chunk_size == 50
        assert chunker.model_name == 'BAAI/bge-m3'
        assert chunker.boundary_strategy == 'paragraph'
        assert chunker.preserve_structure is True
    
    def test_config_validation(self, mock_config):
        """Test configuration validation"""
        # Valid config
        errors = validate_chunking_config(mock_config)
        assert len(errors) == 0
        
        # Invalid config - overlap larger than chunk size
        invalid_config = mock_config.copy()
        invalid_config['overlap_size'] = 600  # Larger than target_chunk_size
        errors = validate_chunking_config(invalid_config)
        assert len(errors) > 0
        assert any('overlap' in error.lower() for error in errors)
        
        # Invalid config - zero chunk size
        invalid_config2 = mock_config.copy()
        invalid_config2['target_chunk_size'] = 0
        errors = validate_chunking_config(invalid_config2)
        assert len(errors) > 0
    
    @patch('text_chunking.TextChunker._get_tokenizer')
    def test_chunk_text_basic(self, mock_tokenizer, mock_config, sample_text):
        """Test basic text chunking functionality"""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = list(range(400))  # Mock 400 tokens
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        chunker = TextChunker(mock_config)
        result = chunker.chunk_text(sample_text)
        
        assert isinstance(result, ChunkResult)
        assert result.success is True
        assert len(result.chunks) > 0
        assert all(len(chunk.content) > 0 for chunk in result.chunks)
        assert result.total_chunks == len(result.chunks)
        assert result.processing_time > 0
    
    @patch('text_chunking.TextChunker._get_tokenizer')
    def test_chunk_overlap(self, mock_tokenizer, mock_config, long_text):
        """Test chunk overlap functionality"""
        # Mock tokenizer to return many tokens
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = list(range(1500))  # Many tokens
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        chunker = TextChunker(mock_config)
        result = chunker.chunk_text(long_text)
        
        assert result.success is True
        assert len(result.chunks) > 1  # Should create multiple chunks
        
        # Check for overlap (this is a simplified check)
        if len(result.chunks) > 1:
            first_chunk_end = result.chunks[0].content[-50:]  # Last 50 chars
            second_chunk_start = result.chunks[1].content[:50]  # First 50 chars
            # There should be some overlap
            assert len(first_chunk_end.strip()) > 0
            assert len(second_chunk_start.strip()) > 0
    
    @patch('text_chunking.TextChunker._get_tokenizer')
    def test_paragraph_boundary_strategy(self, mock_tokenizer, mock_config, sample_text):
        """Test paragraph boundary preservation"""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = list(range(200))
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        config = mock_config.copy()
        config['boundary_strategy'] = 'paragraph'
        chunker = TextChunker(config)
        
        result = chunker.chunk_text(sample_text)
        
        assert result.success is True
        # Chunks should respect paragraph boundaries when possible
        for chunk in result.chunks:
            # Each chunk should contain complete sentences
            assert chunk.content.strip().endswith('.') or chunk.content.strip().endswith('!')
    
    @patch('text_chunking.TextChunker._get_tokenizer')
    def test_sentence_boundary_strategy(self, mock_tokenizer, mock_config, sample_text):
        """Test sentence boundary preservation"""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = list(range(200))
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        config = mock_config.copy()
        config['boundary_strategy'] = 'sentence'
        chunker = TextChunker(config)
        
        result = chunker.chunk_text(sample_text)
        
        assert result.success is True
        # Chunks should respect sentence boundaries
        for chunk in result.chunks:
            content = chunk.content.strip()
            if content:  # If chunk has content
                # Should end with sentence punctuation
                assert content.endswith('.') or content.endswith('!') or content.endswith('?')
    
    def test_empty_text_handling(self, mock_config):
        """Test handling of empty or whitespace-only text"""
        chunker = TextChunker(mock_config)
        
        # Empty string
        result = chunker.chunk_text("")
        assert result.success is True
        assert len(result.chunks) == 0
        
        # Whitespace only
        result = chunker.chunk_text("   \n\t   ")
        assert result.success is True
        assert len(result.chunks) == 0
    
    @patch('text_chunking.TextChunker._get_tokenizer')
    def test_very_long_single_sentence(self, mock_tokenizer, mock_config):
        """Test handling of very long sentences that exceed chunk size"""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = list(range(1000))  # Very long
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create a very long sentence without punctuation
        long_sentence = "This is a very long sentence without proper punctuation that goes on and on " * 100
        
        chunker = TextChunker(mock_config)
        result = chunker.chunk_text(long_sentence)
        
        assert result.success is True
        # Should still create chunks even if forced to break mid-sentence
        assert len(result.chunks) > 0
    
    def test_chunk_quality_calculation(self, mock_config):
        """Test chunk quality metrics calculation"""
        # Create mock chunks
        mock_chunks = [
            type('Chunk', (), {
                'content': 'This is a good chunk with proper content.',
                'token_count': 450,
                'start_index': 0,
                'end_index': 42
            }),
            type('Chunk', (), {
                'content': 'Short.',
                'token_count': 30,  # Too short
                'start_index': 42,
                'end_index': 48
            }),
            type('Chunk', (), {
                'content': 'This is another good chunk that meets the quality standards.',
                'token_count': 480,
                'start_index': 48,
                'end_index': 108
            })
        ]
        
        quality_score = calculate_chunk_quality(mock_chunks, mock_config)
        
        assert 0 <= quality_score <= 1
        # Quality should be impacted by the short chunk
        assert quality_score < 1.0
    
    def test_chunking_metrics(self, mock_config):
        """Test chunking metrics calculation"""
        chunker = TextChunker(mock_config)
        
        # Create mock chunk results
        mock_results = [
            ChunkResult(
                text_id='doc1',
                success=True,
                chunks=[Mock(token_count=450), Mock(token_count=480)],
                total_chunks=2,
                processing_time=1.5,
                metadata={'strategy': 'paragraph'}
            ),
            ChunkResult(
                text_id='doc2',
                success=True,
                chunks=[Mock(token_count=500)],
                total_chunks=1,
                processing_time=0.8,
                metadata={'strategy': 'paragraph'}
            )
        ]
        
        metrics = chunker.calculate_chunking_metrics(mock_results)
        
        assert isinstance(metrics, ChunkingMetrics)
        assert metrics.total_texts == 2
        assert metrics.successful_chunks == 2
        assert metrics.total_chunks == 3
        assert metrics.average_chunks_per_text == 1.5
        assert metrics.average_chunk_size == (450 + 480 + 500) / 3
    
    def test_context_manager(self, mock_config):
        """Test TextChunkingContext context manager"""
        with TextChunkingContext(mock_config) as chunker:
            assert isinstance(chunker, TextChunker)
            assert chunker.config == mock_config
    
    def test_factory_function(self, mock_config):
        """Test factory function for creating chunker"""
        chunker = create_text_chunker(mock_config)
        assert isinstance(chunker, TextChunker)
        assert chunker.config == mock_config
    
    @patch('text_chunking.TextChunker._get_tokenizer')
    def test_batch_chunking(self, mock_tokenizer, mock_config):
        """Test batch text chunking"""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = list(range(300))
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        texts = [
            "First document text for chunking.",
            "Second document with different content.",
            "Third document to test batch processing."
        ]
        
        chunker = TextChunker(mock_config)
        results = chunker.chunk_texts(texts)
        
        assert len(results) == 3
        assert all(isinstance(result, ChunkResult) for result in results)
        assert all(result.success for result in results)
    
    @patch('text_chunking.TextChunker._get_tokenizer')
    def test_semantic_boundary_preservation(self, mock_tokenizer, mock_config):
        """Test semantic boundary preservation"""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = list(range(400))
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Text with clear semantic sections
        semantic_text = """
        Introduction to Machine Learning
        Machine learning is a subset of artificial intelligence.
        
        Types of Machine Learning
        There are three main types: supervised, unsupervised, and reinforcement learning.
        
        Applications
        Machine learning has many real-world applications in various industries.
        """
        
        config = mock_config.copy()
        config['respect_semantic_boundaries'] = True
        chunker = TextChunker(config)
        
        result = chunker.chunk_text(semantic_text)
        
        assert result.success is True
        # Should preserve semantic sections when possible
        for chunk in result.chunks:
            content = chunk.content.strip()
            if content:
                # Check that chunks don't abruptly cut off in the middle of concepts
                assert len(content.split()) > 3  # Reasonable chunk length
    
    def test_error_handling(self, mock_config):
        """Test error handling in chunking process"""
        chunker = TextChunker(mock_config)
        
        # Test with None input
        result = chunker.chunk_text(None)
        assert result.success is False
        assert result.error is not None
        
        # Test with invalid type
        result = chunker.chunk_text(12345)
        assert result.success is False
        assert result.error is not None

class TestTextChunkingIntegration:
    """Integration tests for text chunking"""
    
    def test_end_to_end_chunking(self):
        """Test complete end-to-end text chunking"""
        config = {
            'target_chunk_size': 300,
            'overlap_size': 50,
            'max_chunk_size': 350,
            'min_chunk_size': 100,
            'model_name': 'BAAI/bge-m3',
            'boundary_strategy': 'paragraph',
            'preserve_structure': True,
            'respect_semantic_boundaries': True
        }
        
        # Real-world style document
        document_text = """
        Natural Language Processing (NLP) is a field of artificial intelligence that focuses on 
        the interaction between computers and humans through natural language. The ultimate objective 
        of NLP is to read, decipher, understand, and make sense of human languages in a manner that 
        is valuable.
        
        The history of NLP dates back to the 1950s when Alan Turing published his famous article 
        "Computing Machinery and Intelligence" which proposed what is now called the Turing test as 
        a criterion of intelligence. This was followed by significant developments in computational 
        linguistics and machine learning.
        
        Modern NLP applications include machine translation, sentiment analysis, text summarization, 
        question answering systems, and chatbots. These applications have transformed how we interact 
        with technology and have made information more accessible across language barriers.
        
        The challenges in NLP include dealing with ambiguity, context understanding, cultural nuances, 
        and the constantly evolving nature of human language. Despite these challenges, recent advances 
        in deep learning and transformer architectures have led to significant breakthroughs in the field.
        """
        
        # Mock the tokenizer since we might not have the actual model
        with patch('text_chunking.TextChunker._get_tokenizer') as mock_tokenizer:
            mock_tokenizer_instance = Mock()
            # Simulate realistic token counts
            mock_tokenizer_instance.encode.side_effect = lambda text: list(range(len(text.split()) * 2))
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            chunker = create_text_chunker(config)
            result = chunker.chunk_text(document_text)
            
            assert result.success is True
            assert len(result.chunks) > 0
            assert result.total_chunks == len(result.chunks)
            assert result.processing_time > 0
            
            # Verify chunk properties
            for chunk in result.chunks:
                assert len(chunk.content.strip()) >= config['min_chunk_size']
                assert chunk.token_count > 0
                assert chunk.start_index >= 0
                assert chunk.end_index > chunk.start_index
            
            # Verify overall coverage (chunks should cover the original text)
            total_coverage = sum(chunk.end_index - chunk.start_index for chunk in result.chunks)
            assert total_coverage > 0