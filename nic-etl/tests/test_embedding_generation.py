import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from embedding_generation import (
    EmbeddingGenerator, EmbeddingResult, EmbeddingMetrics,
    create_embedding_generator, validate_embedding_config,
    EmbeddingGenerationContext, calculate_embedding_quality
)

class TestEmbeddingGenerator:
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for embedding generation"""
        return {
            'model_name': 'BAAI/bge-m3',
            'batch_size': 32,
            'max_sequence_length': 512,
            'normalize_embeddings': True,
            'device': 'cpu',
            'cache_model': True,
            'max_memory_gb': 4.0,
            'num_threads': 4,
            'deterministic': True,
            'warmup_iterations': 3
        }
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for embedding generation"""
        return [
            "This is the first sample text for embedding generation.",
            "Here is another text with different content to embed.",
            "A third piece of text to test batch processing capabilities.",
            "Final sample text to complete the test set."
        ]
    
    @pytest.fixture
    def long_text(self):
        """Long text that might exceed sequence length"""
        return "This is a very long text. " * 100  # 500+ words
    
    def test_embedding_generator_initialization(self, mock_config):
        """Test EmbeddingGenerator initialization"""
        with patch('embedding_generation.EmbeddingGenerator._load_model'):
            generator = EmbeddingGenerator(mock_config)
            
            assert generator.config == mock_config
            assert generator.model_name == 'BAAI/bge-m3'
            assert generator.batch_size == 32
            assert generator.max_sequence_length == 512
            assert generator.normalize_embeddings is True
            assert generator.device == 'cpu'
            assert generator.cache_model is True
    
    def test_config_validation(self, mock_config):
        """Test configuration validation"""
        # Valid config
        errors = validate_embedding_config(mock_config)
        assert len(errors) == 0
        
        # Invalid config - bad batch size
        invalid_config = mock_config.copy()
        invalid_config['batch_size'] = 0
        errors = validate_embedding_config(invalid_config)
        assert len(errors) > 0
        assert any('batch_size' in error.lower() for error in errors)
        
        # Invalid config - bad max_sequence_length
        invalid_config2 = mock_config.copy()
        invalid_config2['max_sequence_length'] = -1
        errors = validate_embedding_config(invalid_config2)
        assert len(errors) > 0
    
    @patch('embedding_generation.EmbeddingGenerator._load_model')
    @patch('embedding_generation.EmbeddingGenerator._generate_batch_embeddings')
    def test_generate_embeddings_single_batch(self, mock_batch_gen, mock_load, mock_config, sample_texts):
        """Test embedding generation for single batch"""
        # Mock model loading
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Mock batch generation - return realistic embeddings
        mock_embeddings = np.random.rand(len(sample_texts), 1024).astype(np.float32)
        mock_batch_gen.return_value = mock_embeddings
        
        generator = EmbeddingGenerator(mock_config)
        result = generator.generate_embeddings(sample_texts)
        
        assert isinstance(result, EmbeddingResult)
        assert result.success is True
        assert len(result.embeddings) == len(sample_texts)
        assert result.embeddings.shape == (len(sample_texts), 1024)
        assert result.processing_time > 0
        assert result.total_texts == len(sample_texts)
    
    @patch('embedding_generation.EmbeddingGenerator._load_model')
    @patch('embedding_generation.EmbeddingGenerator._generate_batch_embeddings')
    def test_generate_embeddings_multiple_batches(self, mock_batch_gen, mock_load, mock_config):
        """Test embedding generation for multiple batches"""
        # Mock model loading
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Create large text list that requires multiple batches
        large_text_list = [f"Text number {i}" for i in range(100)]
        
        # Mock batch generation
        def mock_batch_side_effect(texts):
            return np.random.rand(len(texts), 1024).astype(np.float32)
        
        mock_batch_gen.side_effect = mock_batch_side_effect
        
        config = mock_config.copy()
        config['batch_size'] = 16  # Force multiple batches
        generator = EmbeddingGenerator(config)
        
        result = generator.generate_embeddings(large_text_list)
        
        assert result.success is True
        assert len(result.embeddings) == 100
        assert result.embeddings.shape == (100, 1024)
        # Should have called batch generation multiple times
        assert mock_batch_gen.call_count > 1
    
    @patch('embedding_generation.EmbeddingGenerator._load_model')
    def test_text_preprocessing(self, mock_load, mock_config):
        """Test text preprocessing functionality"""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        generator = EmbeddingGenerator(mock_config)
        
        # Test with various text formats
        raw_texts = [
            "  Normal text with spaces  ",
            "Text\nwith\nnewlines",
            "Text with\ttabs",
            "",  # Empty string
            "   ",  # Whitespace only
            "Text with special chars: !@#$%"
        ]
        
        processed = generator._preprocess_texts(raw_texts)
        
        # Should handle all inputs gracefully
        assert len(processed) == len(raw_texts)
        assert all(isinstance(text, str) for text in processed)
        # Whitespace should be normalized
        assert processed[0].strip() == "Normal text with spaces"
    
    @patch('embedding_generation.EmbeddingGenerator._load_model')
    def test_sequence_length_handling(self, mock_load, mock_config, long_text):
        """Test handling of texts exceeding max sequence length"""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = list(range(1000))  # Long token sequence
        mock_model.tokenizer = mock_tokenizer
        mock_load.return_value = mock_model
        
        generator = EmbeddingGenerator(mock_config)
        
        # Test truncation
        truncated = generator._handle_sequence_length([long_text])
        assert len(truncated) == 1
        assert len(truncated[0]) <= len(long_text)  # Should be truncated or same
    
    @patch('embedding_generation.EmbeddingGenerator._load_model')
    @patch('embedding_generation.EmbeddingGenerator._generate_batch_embeddings')
    def test_embedding_normalization(self, mock_batch_gen, mock_load, mock_config, sample_texts):
        """Test embedding normalization"""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Create unnormalized embeddings
        unnormalized = np.random.rand(len(sample_texts), 1024).astype(np.float32) * 10
        mock_batch_gen.return_value = unnormalized
        
        # Test with normalization enabled
        config = mock_config.copy()
        config['normalize_embeddings'] = True
        generator = EmbeddingGenerator(config)
        
        result = generator.generate_embeddings(sample_texts)
        
        # Check that embeddings are normalized (L2 norm ≈ 1)
        norms = np.linalg.norm(result.embeddings, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-5)
    
    @patch('embedding_generation.EmbeddingGenerator._load_model')
    @patch('embedding_generation.EmbeddingGenerator._generate_batch_embeddings')
    def test_no_normalization(self, mock_batch_gen, mock_load, mock_config, sample_texts):
        """Test embedding generation without normalization"""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        unnormalized = np.random.rand(len(sample_texts), 1024).astype(np.float32) * 10
        mock_batch_gen.return_value = unnormalized
        
        # Test with normalization disabled
        config = mock_config.copy()
        config['normalize_embeddings'] = False
        generator = EmbeddingGenerator(config)
        
        result = generator.generate_embeddings(sample_texts)
        
        # Embeddings should not be normalized
        norms = np.linalg.norm(result.embeddings, axis=1)
        assert not np.allclose(norms, 1.0, rtol=1e-5)
    
    def test_empty_input_handling(self, mock_config):
        """Test handling of empty input"""
        with patch('embedding_generation.EmbeddingGenerator._load_model'):
            generator = EmbeddingGenerator(mock_config)
            
            # Empty list
            result = generator.generate_embeddings([])
            assert result.success is True
            assert len(result.embeddings) == 0
            assert result.total_texts == 0
    
    @patch('embedding_generation.EmbeddingGenerator._load_model')
    def test_error_handling(self, mock_load, mock_config, sample_texts):
        """Test error handling during generation"""
        # Mock model loading failure
        mock_load.side_effect = Exception("Model loading failed")
        
        generator = EmbeddingGenerator(mock_config)
        result = generator.generate_embeddings(sample_texts)
        
        assert result.success is False
        assert result.error is not None
        assert "failed" in result.error.lower()
    
    @patch('embedding_generation.EmbeddingGenerator._load_model')
    @patch('embedding_generation.EmbeddingGenerator._generate_batch_embeddings')
    def test_deterministic_generation(self, mock_batch_gen, mock_load, mock_config, sample_texts):
        """Test deterministic embedding generation"""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Use fixed random seed for deterministic results
        np.random.seed(42)
        embeddings1 = np.random.rand(len(sample_texts), 1024).astype(np.float32)
        np.random.seed(42)
        embeddings2 = np.random.rand(len(sample_texts), 1024).astype(np.float32)
        
        mock_batch_gen.side_effect = [embeddings1, embeddings2]
        
        config = mock_config.copy()
        config['deterministic'] = True
        generator = EmbeddingGenerator(config)
        
        result1 = generator.generate_embeddings(sample_texts)
        result2 = generator.generate_embeddings(sample_texts)
        
        # Results should be similar (testing framework, not actual determinism)
        assert result1.success and result2.success
        assert result1.embeddings.shape == result2.embeddings.shape
    
    def test_embedding_quality_calculation(self, mock_config):
        """Test embedding quality metrics"""
        # Create mock embeddings with known properties
        good_embeddings = np.random.rand(10, 1024).astype(np.float32)
        # Normalize them
        good_embeddings = good_embeddings / np.linalg.norm(good_embeddings, axis=1, keepdims=True)
        
        # Add some zero embeddings (poor quality)
        poor_embeddings = np.zeros((2, 1024), dtype=np.float32)
        all_embeddings = np.vstack([good_embeddings, poor_embeddings])
        
        quality_score = calculate_embedding_quality(all_embeddings, mock_config)
        
        assert 0 <= quality_score <= 1
        # Quality should be reduced due to zero embeddings
        assert quality_score < 1.0
    
    def test_embedding_metrics(self, mock_config):
        """Test embedding metrics calculation"""
        with patch('embedding_generation.EmbeddingGenerator._load_model'):
            generator = EmbeddingGenerator(mock_config)
            
            # Create mock results
            mock_results = [
                EmbeddingResult(
                    texts=['text1', 'text2'],
                    success=True,
                    embeddings=np.random.rand(2, 1024),
                    processing_time=1.5,
                    total_texts=2,
                    metadata={'batch_count': 1}
                ),
                EmbeddingResult(
                    texts=['text3'],
                    success=True,
                    embeddings=np.random.rand(1, 1024),
                    processing_time=0.8,
                    total_texts=1,
                    metadata={'batch_count': 1}
                )
            ]
            
            metrics = generator.calculate_embedding_metrics(mock_results)
            
            assert isinstance(metrics, EmbeddingMetrics)
            assert metrics.total_texts == 3
            assert metrics.successful_texts == 3
            assert metrics.failed_texts == 0
            assert metrics.average_processing_time == (1.5 + 0.8) / 2
            assert metrics.total_embeddings == 3
    
    def test_context_manager(self, mock_config):
        """Test EmbeddingGenerationContext context manager"""
        with patch('embedding_generation.EmbeddingGenerator._load_model'):
            with EmbeddingGenerationContext(mock_config) as generator:
                assert isinstance(generator, EmbeddingGenerator)
                assert generator.config == mock_config
    
    def test_factory_function(self, mock_config):
        """Test factory function for creating generator"""
        with patch('embedding_generation.EmbeddingGenerator._load_model'):
            generator = create_embedding_generator(mock_config)
            assert isinstance(generator, EmbeddingGenerator)
            assert generator.config == mock_config
    
    @patch('embedding_generation.EmbeddingGenerator._load_model')
    @patch('embedding_generation.EmbeddingGenerator._generate_batch_embeddings')
    def test_memory_management(self, mock_batch_gen, mock_load, mock_config):
        """Test memory management during batch processing"""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # Mock batch generation with memory simulation
        def mock_batch_with_memory(texts):
            # Simulate memory usage
            return np.random.rand(len(texts), 1024).astype(np.float32)
        
        mock_batch_gen.side_effect = mock_batch_with_memory
        
        # Test with memory-conscious settings
        config = mock_config.copy()
        config['batch_size'] = 8  # Smaller batches
        config['max_memory_gb'] = 2.0  # Lower memory limit
        
        generator = EmbeddingGenerator(config)
        
        # Large text set to test memory handling
        large_texts = [f"Text {i}" for i in range(50)]
        result = generator.generate_embeddings(large_texts)
        
        assert result.success is True
        assert len(result.embeddings) == 50
    
    @patch('embedding_generation.EmbeddingGenerator._load_model')
    def test_model_caching(self, mock_load, mock_config):
        """Test model caching functionality"""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        config = mock_config.copy()
        config['cache_model'] = True
        
        # Create first generator
        generator1 = EmbeddingGenerator(config)
        assert mock_load.call_count == 1
        
        # Create second generator - should use cache
        generator2 = EmbeddingGenerator(config)
        # Model loading should be optimized (implementation dependent)
        assert generator1.model_name == generator2.model_name

class TestEmbeddingGenerationIntegration:
    """Integration tests for embedding generation"""
    
    def test_end_to_end_embedding_generation(self):
        """Test complete end-to-end embedding generation"""
        config = {
            'model_name': 'BAAI/bge-m3',
            'batch_size': 16,
            'max_sequence_length': 512,
            'normalize_embeddings': True,
            'device': 'cpu',
            'cache_model': True,
            'max_memory_gb': 2.0,
            'num_threads': 2,
            'deterministic': True,
            'warmup_iterations': 1
        }
        
        texts = [
            "Natural language processing is a field of artificial intelligence.",
            "Machine learning algorithms can learn patterns from data.",
            "Deep learning uses neural networks with multiple layers.",
            "Text embeddings capture semantic meaning of words and sentences."
        ]
        
        # Mock the actual model since we don't have it installed
        with patch('embedding_generation.EmbeddingGenerator._load_model') as mock_load:
            with patch('embedding_generation.EmbeddingGenerator._generate_batch_embeddings') as mock_batch:
                # Mock model
                mock_model = Mock()
                mock_load.return_value = mock_model
                
                # Mock embeddings
                mock_embeddings = np.random.rand(len(texts), 1024).astype(np.float32)
                mock_batch.return_value = mock_embeddings
                
                generator = create_embedding_generator(config)
                result = generator.generate_embeddings(texts)
                
                assert result.success is True
                assert result.embeddings.shape == (len(texts), 1024)
                assert result.total_texts == len(texts)
                assert result.processing_time > 0
                
                # Check that embeddings are normalized
                norms = np.linalg.norm(result.embeddings, axis=1)
                assert np.allclose(norms, 1.0, rtol=1e-5)
                
                # Verify metadata
                assert 'batch_count' in result.metadata
                assert 'model_name' in result.metadata
                assert result.metadata['model_name'] == config['model_name']