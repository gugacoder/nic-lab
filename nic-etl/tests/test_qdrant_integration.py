import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from qdrant_integration import (
    QdrantVectorStore, SearchResult, StorageResult, StorageMetrics,
    create_qdrant_vector_store, validate_qdrant_config,
    QdrantConnectionContext, calculate_storage_quality
)

class TestQdrantVectorStore:
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for Qdrant integration"""
        return {
            'url': 'https://qdrant.codrstudio.dev/',
            'api_key': 'test_api_key',
            'collection_name': 'test_nic',
            'vector_size': 1024,
            'distance_metric': 'COSINE',
            'timeout': 30.0,
            'retry_attempts': 3,
            'batch_size': 100,
            'enable_payload_validation': True,
            'optimize_collection': True
        }
    
    @pytest.fixture
    def sample_vectors(self):
        """Sample vectors for testing"""
        np.random.seed(42)  # For reproducible tests
        return np.random.rand(5, 1024).astype(np.float32)
    
    @pytest.fixture
    def sample_payloads(self):
        """Sample payloads for testing"""
        return [
            {
                'document_id': 'doc1',
                'content': 'First document content',
                'chunk_id': 'chunk1',
                'metadata': {'source': 'test'}
            },
            {
                'document_id': 'doc2',
                'content': 'Second document content',
                'chunk_id': 'chunk2',
                'metadata': {'source': 'test'}
            },
            {
                'document_id': 'doc3',
                'content': 'Third document content',
                'chunk_id': 'chunk3',
                'metadata': {'source': 'test'}
            },
            {
                'document_id': 'doc4',
                'content': 'Fourth document content',
                'chunk_id': 'chunk4',
                'metadata': {'source': 'test'}
            },
            {
                'document_id': 'doc5',
                'content': 'Fifth document content',
                'chunk_id': 'chunk5',
                'metadata': {'source': 'test'}
            }
        ]
    
    def test_qdrant_vector_store_initialization(self, mock_config):
        """Test QdrantVectorStore initialization"""
        with patch('qdrant_integration.QdrantClient') as mock_client:
            store = QdrantVectorStore(mock_config)
            
            assert store.config == mock_config
            assert store.url == 'https://qdrant.codrstudio.dev/'
            assert store.api_key == 'test_api_key'
            assert store.collection_name == 'test_nic'
            assert store.vector_size == 1024
            assert store.distance_metric == 'COSINE'
            assert store.batch_size == 100
    
    def test_config_validation(self, mock_config):
        """Test configuration validation"""
        # Valid config
        errors = validate_qdrant_config(mock_config)
        assert len(errors) == 0
        
        # Invalid config - missing URL
        invalid_config = mock_config.copy()
        del invalid_config['url']
        errors = validate_qdrant_config(invalid_config)
        assert len(errors) > 0
        assert any('url' in error.lower() for error in errors)
        
        # Invalid config - wrong vector size
        invalid_config2 = mock_config.copy()
        invalid_config2['vector_size'] = -1
        errors = validate_qdrant_config(invalid_config2)
        assert len(errors) > 0
        
        # Invalid config - bad distance metric
        invalid_config3 = mock_config.copy()
        invalid_config3['distance_metric'] = 'INVALID'
        errors = validate_qdrant_config(invalid_config3)
        assert len(errors) > 0
    
    @patch('qdrant_integration.QdrantClient')
    def test_collection_creation(self, mock_client_class, mock_config):
        """Test collection creation"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock collection info - collection doesn't exist
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = True
        
        store = QdrantVectorStore(mock_config)
        result = store.create_collection()
        
        assert result is True
        mock_client.create_collection.assert_called_once()
    
    @patch('qdrant_integration.QdrantClient')
    def test_collection_exists(self, mock_client_class, mock_config):
        """Test checking if collection exists"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock collection exists
        mock_client.get_collection.return_value = Mock(status='green')
        
        store = QdrantVectorStore(mock_config)
        exists = store.collection_exists()
        
        assert exists is True
        mock_client.get_collection.assert_called_once()
    
    @patch('qdrant_integration.QdrantClient')
    def test_store_vectors_single_batch(self, mock_client_class, mock_config, sample_vectors, sample_payloads):
        """Test storing vectors in single batch"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.upsert.return_value = Mock(status='completed')
        
        store = QdrantVectorStore(mock_config)
        result = store.store_vectors(sample_vectors, sample_payloads)
        
        assert isinstance(result, StorageResult)
        assert result.success is True
        assert result.stored_count == len(sample_vectors)
        assert result.failed_count == 0
        assert result.processing_time > 0
        
        # Verify upsert was called
        mock_client.upsert.assert_called_once()
    
    @patch('qdrant_integration.QdrantClient')
    def test_store_vectors_multiple_batches(self, mock_client_class, mock_config):
        """Test storing vectors in multiple batches"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.upsert.return_value = Mock(status='completed')
        
        # Create large dataset that requires batching
        large_vectors = np.random.rand(250, 1024).astype(np.float32)
        large_payloads = [{'id': f'doc{i}'} for i in range(250)]
        
        config = mock_config.copy()
        config['batch_size'] = 100  # Force multiple batches
        
        store = QdrantVectorStore(config)
        result = store.store_vectors(large_vectors, large_payloads)
        
        assert result.success is True
        assert result.stored_count == 250
        # Should have called upsert multiple times
        assert mock_client.upsert.call_count >= 3
    
    @patch('qdrant_integration.QdrantClient')
    def test_search_similar_vectors(self, mock_client_class, mock_config, sample_vectors):
        """Test searching for similar vectors"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock search results
        mock_search_results = [
            Mock(id='1', score=0.95, payload={'content': 'Similar content 1'}),
            Mock(id='2', score=0.87, payload={'content': 'Similar content 2'}),
            Mock(id='3', score=0.82, payload={'content': 'Similar content 3'})
        ]
        mock_client.search.return_value = mock_search_results
        
        store = QdrantVectorStore(mock_config)
        query_vector = sample_vectors[0]
        
        results = store.search_similar_vectors(query_vector, limit=3)
        
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score == 0.95
        assert results[0].payload['content'] == 'Similar content 1'
        
        mock_client.search.assert_called_once()
    
    @patch('qdrant_integration.QdrantClient')
    def test_search_with_filters(self, mock_client_class, mock_config, sample_vectors):
        """Test searching with payload filters"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search.return_value = []
        
        store = QdrantVectorStore(mock_config)
        query_vector = sample_vectors[0]
        
        # Search with filters
        filters = {'source': 'test', 'category': 'document'}
        results = store.search_similar_vectors(
            query_vector, 
            limit=5, 
            filters=filters
        )
        
        assert isinstance(results, list)
        mock_client.search.assert_called_once()
        
        # Verify filters were passed
        call_args = mock_client.search.call_args
        assert call_args is not None
    
    @patch('qdrant_integration.QdrantClient')
    def test_delete_vectors(self, mock_client_class, mock_config):
        """Test deleting vectors"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.delete.return_value = Mock(status='completed')
        
        store = QdrantVectorStore(mock_config)
        
        # Delete by IDs
        vector_ids = ['1', '2', '3']
        result = store.delete_vectors(vector_ids)
        
        assert result is True
        mock_client.delete.assert_called_once()
    
    @patch('qdrant_integration.QdrantClient')
    def test_get_collection_info(self, mock_client_class, mock_config):
        """Test getting collection information"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock collection info
        mock_info = Mock(
            vectors_count=1000,
            status='green',
            optimizer_status={'status': 'ok'},
            config=Mock(params=Mock(vectors=Mock(size=1024)))
        )
        mock_client.get_collection.return_value = mock_info
        
        store = QdrantVectorStore(mock_config)
        info = store.get_collection_info()
        
        assert info is not None
        assert hasattr(info, 'vectors_count')
        mock_client.get_collection.assert_called_once()
    
    @patch('qdrant_integration.QdrantClient')
    def test_optimize_collection(self, mock_client_class, mock_config):
        """Test collection optimization"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.update_collection.return_value = True
        
        store = QdrantVectorStore(mock_config)
        result = store.optimize_collection()
        
        assert result is True
        mock_client.update_collection.assert_called_once()
    
    @patch('qdrant_integration.QdrantClient')
    def test_error_handling_connection_failure(self, mock_client_class, mock_config):
        """Test error handling for connection failures"""
        # Mock connection failure
        mock_client_class.side_effect = Exception("Connection failed")
        
        store = QdrantVectorStore(mock_config)
        # Store should handle initialization gracefully
        assert store is not None
    
    @patch('qdrant_integration.QdrantClient')
    def test_error_handling_storage_failure(self, mock_client_class, mock_config, sample_vectors, sample_payloads):
        """Test error handling for storage failures"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.upsert.side_effect = Exception("Storage failed")
        
        store = QdrantVectorStore(mock_config)
        result = store.store_vectors(sample_vectors, sample_payloads)
        
        assert result.success is False
        assert result.error is not None
        assert "failed" in result.error.lower()
    
    @patch('qdrant_integration.QdrantClient')
    def test_error_handling_search_failure(self, mock_client_class, mock_config, sample_vectors):
        """Test error handling for search failures"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search.side_effect = Exception("Search failed")
        
        store = QdrantVectorStore(mock_config)
        results = store.search_similar_vectors(sample_vectors[0])
        
        # Should return empty list on error
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_payload_validation(self, mock_config):
        """Test payload validation"""
        with patch('qdrant_integration.QdrantClient'):
            store = QdrantVectorStore(mock_config)
            
            # Valid payload
            valid_payload = {
                'document_id': 'doc1',
                'content': 'Valid content',
                'chunk_id': 'chunk1'
            }
            assert store._validate_payload(valid_payload) is True
            
            # Invalid payload - missing required fields
            invalid_payload = {'content': 'Missing required fields'}
            assert store._validate_payload(invalid_payload) is False
    
    def test_vector_validation(self, mock_config, sample_vectors):
        """Test vector validation"""
        with patch('qdrant_integration.QdrantClient'):
            store = QdrantVectorStore(mock_config)
            
            # Valid vectors
            assert store._validate_vectors(sample_vectors) is True
            
            # Invalid vectors - wrong dimension
            wrong_dim_vectors = np.random.rand(5, 512).astype(np.float32)
            assert store._validate_vectors(wrong_dim_vectors) is False
            
            # Invalid vectors - not numpy array
            assert store._validate_vectors([[1, 2, 3]]) is False
    
    def test_storage_quality_calculation(self, mock_config):
        """Test storage quality metrics"""
        # Create mock storage results
        results = [
            StorageResult(
                success=True,
                stored_count=100,
                failed_count=0,
                processing_time=2.5,
                metadata={'batch_count': 1}
            ),
            StorageResult(
                success=True,
                stored_count=80,
                failed_count=5,  # Some failures
                processing_time=3.0,
                metadata={'batch_count': 1}
            )
        ]
        
        quality_score = calculate_storage_quality(results, mock_config)
        
        assert 0 <= quality_score <= 1
        # Quality should be reduced due to failures
        assert quality_score < 1.0
    
    def test_storage_metrics(self, mock_config):
        """Test storage metrics calculation"""
        with patch('qdrant_integration.QdrantClient'):
            store = QdrantVectorStore(mock_config)
            
            # Create mock results
            results = [
                StorageResult(
                    success=True,
                    stored_count=100,
                    failed_count=0,
                    processing_time=2.0,
                    metadata={'batch_count': 1}
                ),
                StorageResult(
                    success=True,
                    stored_count=150,
                    failed_count=5,
                    processing_time=3.0,
                    metadata={'batch_count': 2}
                )
            ]
            
            metrics = store.calculate_storage_metrics(results)
            
            assert isinstance(metrics, StorageMetrics)
            assert metrics.total_operations == 2
            assert metrics.successful_operations == 2
            assert metrics.total_vectors_stored == 250
            assert metrics.total_failed_vectors == 5
            assert metrics.average_processing_time == 2.5
    
    def test_context_manager(self, mock_config):
        """Test QdrantConnectionContext context manager"""
        with patch('qdrant_integration.QdrantClient'):
            with QdrantConnectionContext(mock_config) as store:
                assert isinstance(store, QdrantVectorStore)
                assert store.config == mock_config
    
    def test_factory_function(self, mock_config):
        """Test factory function for creating vector store"""
        with patch('qdrant_integration.QdrantClient'):
            store = create_qdrant_vector_store(mock_config)
            assert isinstance(store, QdrantVectorStore)
            assert store.config == mock_config
    
    @patch('qdrant_integration.QdrantClient')
    def test_retry_mechanism(self, mock_client_class, mock_config, sample_vectors, sample_payloads):
        """Test retry mechanism for failed operations"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock failure then success
        mock_client.upsert.side_effect = [
            Exception("Temporary failure"),
            Exception("Another failure"),
            Mock(status='completed')  # Success on third try
        ]
        
        config = mock_config.copy()
        config['retry_attempts'] = 3
        
        store = QdrantVectorStore(config)
        result = store.store_vectors(sample_vectors, sample_payloads)
        
        # Should eventually succeed
        assert result.success is True
        assert mock_client.upsert.call_count == 3
    
    @patch('qdrant_integration.QdrantClient')
    def test_batch_size_optimization(self, mock_client_class, mock_config):
        """Test batch size optimization"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.upsert.return_value = Mock(status='completed')
        
        # Test with different batch sizes
        large_vectors = np.random.rand(1000, 1024).astype(np.float32)
        large_payloads = [{'id': f'doc{i}'} for i in range(1000)]
        
        config = mock_config.copy()
        config['batch_size'] = 50  # Small batches
        
        store = QdrantVectorStore(config)
        result = store.store_vectors(large_vectors, large_payloads)
        
        assert result.success is True
        assert result.stored_count == 1000
        # Should have made multiple batch calls
        assert mock_client.upsert.call_count >= 20

class TestQdrantIntegrationIntegration:
    """Integration tests for Qdrant integration"""
    
    def test_end_to_end_vector_operations(self):
        """Test complete end-to-end vector operations"""
        config = {
            'url': 'https://test.qdrant.dev/',
            'api_key': 'test_key',
            'collection_name': 'integration_test',
            'vector_size': 1024,
            'distance_metric': 'COSINE',
            'timeout': 30.0,
            'retry_attempts': 3,
            'batch_size': 50,
            'enable_payload_validation': True,
            'optimize_collection': True
        }
        
        # Test data
        vectors = np.random.rand(100, 1024).astype(np.float32)
        payloads = [
            {
                'document_id': f'doc{i}',
                'content': f'Content for document {i}',
                'chunk_id': f'chunk{i}',
                'metadata': {'category': 'test', 'index': i}
            }
            for i in range(100)
        ]
        
        # Mock the Qdrant client since we don't have actual connection
        with patch('qdrant_integration.QdrantClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock successful operations
            mock_client.get_collection.side_effect = Exception("Not found")  # First call
            mock_client.create_collection.return_value = True
            mock_client.upsert.return_value = Mock(status='completed')
            mock_client.search.return_value = [
                Mock(id='1', score=0.95, payload=payloads[0]),
                Mock(id='2', score=0.87, payload=payloads[1])
            ]
            
            store = create_qdrant_vector_store(config)
            
            # 1. Create collection
            created = store.create_collection()
            assert created is True
            
            # 2. Store vectors
            storage_result = store.store_vectors(vectors, payloads)
            assert storage_result.success is True
            assert storage_result.stored_count == 100
            
            # 3. Search vectors
            query_vector = vectors[0]
            search_results = store.search_similar_vectors(query_vector, limit=2)
            assert len(search_results) == 2
            assert all(isinstance(r, SearchResult) for r in search_results)
            
            # 4. Verify all operations were called
            assert mock_client.create_collection.called
            assert mock_client.upsert.called
            assert mock_client.search.called