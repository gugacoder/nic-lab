# Qdrant Integration - PRP

## ROLE
**Vector Database Engineer with Qdrant and Semantic Search expertise**

Specialized in Qdrant vector database architecture, high-performance vector operations, and semantic search optimization. Responsible for implementing robust vector storage, retrieval, and collection management with optimal indexing strategies and payload schema compliance for production semantic search applications.

## OBJECTIVE
**Production-Ready Qdrant Vector Database Integration**

Deliver a production-ready Python module that:
- Manages Qdrant collections with 1024-dimensional vectors and COSINE distance metrics
- Implements idempotent vector insertion preventing duplicates through deterministic ID generation
- Supports high-performance batch operations with optimal payload structuring
- Provides comprehensive error handling, connection management, and retry mechanisms
- Validates payloads against NIC Schema requirements
- Enables efficient vector search and retrieval operations
- Implements collection monitoring, health checks, and performance optimization

## MOTIVATION
**Scalable Semantic Search Infrastructure**

Reliable vector database integration is the foundation of effective semantic search and information retrieval systems. By implementing optimized Qdrant operations with idempotent processing, comprehensive error handling, and NIC Schema compliance, this module ensures consistent vector storage and retrieval performance that scales with document processing demands while maintaining data integrity and search quality.

## CONTEXT
**Qdrant Production Deployment Architecture**

- **Qdrant Instance**: `https://qdrant.codrstudio.dev/`
- **Authentication**: API Key (`93f0c9d6b9a53758f2376decf318b3ae300e9bdb50be2d0e9c893ee4469fd857`)
- **Collection Name**: `nic`
- **Vector Configuration**: 1024 dimensions, COSINE distance
- **Input Source**: ChunkEmbedding objects from embedding generation
- **Payload Schema**: NIC Schema with comprehensive metadata
- **Performance Requirements**: Support 10K+ vectors with sub-second search response

## IMPLEMENTATION BLUEPRINT
**Comprehensive Qdrant Integration Module**

### Architecture Overview
```python
# Module Structure: modules/qdrant_integration.py
class QdrantVectorStore:
    """Production-ready Qdrant vector database integration"""
    
    def __init__(self, config: QdrantConfig)
    def ensure_collection(self, collection_name: str) -> bool
    def insert_vectors(self, embeddings: List[ChunkEmbedding]) -> InsertionResult
    def search_vectors(self, query_vector: np.ndarray, limit: int, filters: Optional[Dict]) -> List[SearchResult]
    def get_collection_info(self, collection_name: str) -> CollectionInfo
    def validate_payloads(self, payloads: List[Dict[str, Any]]) -> ValidationResult
    def health_check(self) -> HealthStatus
```

### Code Structure
**File Organization**: `modules/qdrant_integration.py`
```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CollectionStatus, PointStruct, 
    Filter, FieldCondition, SearchRequest, SearchParams,
    UpdateStatus, OptimizersConfigDiff, CollectionParams
)
from qdrant_client.http import models
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib
import uuid
from uuid import UUID
import time
import json
from pathlib import Path

@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database"""
    url: str = "https://qdrant.codrstudio.dev/"
    api_key: Optional[str] = None
    collection_name: str = "nic"
    vector_size: int = 1024
    distance_metric: Distance = Distance.COSINE
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    enable_payload_validation: bool = True
    optimize_collection: bool = True

@dataclass
class InsertionResult:
    """Result of vector insertion operation"""
    total_inserted: int
    successful_insertions: int
    failed_insertions: int
    duplicate_skipped: int
    processing_time_seconds: float
    errors: List[str] = field(default_factory=list)

@dataclass
class SearchResult:
    """Individual search result with metadata"""
    point_id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[np.ndarray] = None

@dataclass
class CollectionInfo:
    """Collection status and statistics"""
    name: str
    status: str
    vector_count: int
    segments_count: int
    disk_usage_bytes: int
    memory_usage_bytes: int
    indexed_vectors_count: int
    configuration: Dict[str, Any]

@dataclass
class HealthStatus:
    """Qdrant instance health information"""
    is_healthy: bool
    response_time_ms: float
    collections_accessible: List[str]
    errors: List[str] = field(default_factory=list)

class QdrantVectorStore:
    """Production-ready Qdrant vector database integration"""
    
    # NIC Schema payload structure
    NIC_SCHEMA_FIELDS = {
        'chunk_id': str,
        'document_title': str,
        'document_path': str,
        'section_title': str,
        'content': str,
        'token_count': int,
        'chunk_index': int,
        'total_chunks': int,
        'page_number': int,
        'chunk_type': str,
        'embedding_model': str,
        'processing_timestamp': str,
        'gitlab_commit': str,
        'gitlab_branch': str,
        'ocr_applied': bool,
        'is_latest': bool
    }
    
    def __init__(self, config: QdrantConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = None
        
        # Initialize Qdrant client
        self._initialize_client()
        
        # Ensure collection exists
        self.ensure_collection(config.collection_name)
        
    def _initialize_client(self):
        """Initialize Qdrant client with authentication and configuration"""
        try:
            self.client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            
            # Test connection
            collections = self.client.get_collections()
            self.logger.info(f"Connected to Qdrant at {self.config.url}")
            self.logger.info(f"Available collections: {[c.name for c in collections.collections]}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    def ensure_collection(self, collection_name: str) -> bool:
        """Ensure collection exists with proper configuration"""
        try:
            # Check if collection exists
            try:
                collection_info = self.client.get_collection(collection_name)
                self.logger.info(f"Collection '{collection_name}' already exists")
                
                # Validate collection configuration
                config = collection_info.config
                if (config.params.vectors.size != self.config.vector_size or
                    config.params.vectors.distance != self.config.distance_metric):
                    
                    self.logger.warning(
                        f"Collection configuration mismatch. "
                        f"Expected: size={self.config.vector_size}, distance={self.config.distance_metric}. "
                        f"Actual: size={config.params.vectors.size}, distance={config.params.vectors.distance}"
                    )
                    return False
                
                return True
                
            except Exception:
                # Collection doesn't exist, create it
                self.logger.info(f"Creating collection '{collection_name}'")
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=self.config.distance_metric
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=None,
                        memmap_threshold=None,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=1
                    ) if self.config.optimize_collection else None
                )
                
                self.logger.info(f"Collection '{collection_name}' created successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to ensure collection '{collection_name}': {e}")
            raise
    
    def insert_vectors(self, embeddings: List[Any]) -> InsertionResult:
        """Insert embeddings with comprehensive error handling and idempotency"""
        
        if not embeddings:
            return InsertionResult(
                total_inserted=0,
                successful_insertions=0,
                failed_insertions=0,
                duplicate_skipped=0,
                processing_time_seconds=0.0
            )
        
        start_time = time.time()
        self.logger.info(f"Inserting {len(embeddings)} vectors into Qdrant")
        
        try:
            # Prepare points for insertion
            points = []
            payload_validation_errors = []
            
            for embedding in embeddings:
                try:
                    # Generate deterministic point ID
                    point_id = self._generate_point_id(embedding)
                    
                    # Create payload following NIC Schema
                    payload = self._create_nic_payload(embedding)
                    
                    # Validate payload if enabled
                    if self.config.enable_payload_validation:
                        validation_result = self._validate_single_payload(payload)
                        if not validation_result['is_valid']:
                            payload_validation_errors.extend(validation_result['errors'])
                            continue
                    
                    # Create point structure
                    point = PointStruct(
                        id=point_id,
                        vector=embedding.embedding_vector.tolist(),
                        payload=payload
                    )
                    points.append(point)
                    
                except Exception as e:
                    self.logger.error(f"Failed to prepare point for embedding {embedding.chunk_id}: {e}")
                    payload_validation_errors.append(f"Point preparation failed: {e}")
            
            if payload_validation_errors:
                self.logger.warning(f"Payload validation errors: {payload_validation_errors}")
            
            # Insert points in batches
            insertion_results = self._batch_insert_points(points)
            
            processing_time = time.time() - start_time
            
            result = InsertionResult(
                total_inserted=len(points),
                successful_insertions=insertion_results['successful'],
                failed_insertions=insertion_results['failed'],
                duplicate_skipped=insertion_results['duplicates'],
                processing_time_seconds=processing_time,
                errors=payload_validation_errors + insertion_results['errors']
            )
            
            self.logger.info(
                f"Insertion completed: {result.successful_insertions}/{result.total_inserted} successful "
                f"in {result.processing_time_seconds:.2f}s"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Vector insertion failed: {e}")
            
            return InsertionResult(
                total_inserted=len(embeddings),
                successful_insertions=0,
                failed_insertions=len(embeddings),
                duplicate_skipped=0,
                processing_time_seconds=processing_time,
                errors=[str(e)]
            )
    
    def _batch_insert_points(self, points: List[PointStruct]) -> Dict[str, Any]:
        """Insert points in batches with retry logic"""
        
        total_points = len(points)
        successful = 0
        failed = 0
        duplicates = 0
        errors = []
        
        # Process in batches
        for i in range(0, total_points, self.config.batch_size):
            batch = points[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            total_batches = (total_points + self.config.batch_size - 1) // self.config.batch_size
            
            self.logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} points)")
            
            # Retry logic for batch insertion
            for attempt in range(self.config.retry_attempts):
                try:
                    # Check for existing points to implement idempotency
                    existing_ids = self._check_existing_points([p.id for p in batch])
                    new_points = [p for p in batch if p.id not in existing_ids]
                    
                    duplicates += len(existing_ids)
                    
                    if new_points:
                        # Insert new points
                        operation_result = self.client.upsert(
                            collection_name=self.config.collection_name,
                            points=new_points
                        )
                        
                        if operation_result.status == UpdateStatus.COMPLETED:
                            successful += len(new_points)
                            self.logger.debug(f"Batch {batch_num} inserted successfully ({len(new_points)} new points)")
                        else:
                            failed += len(new_points)
                            errors.append(f"Batch {batch_num} failed with status: {operation_result.status}")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < self.config.retry_attempts - 1:
                        wait_time = self.config.retry_delay * (attempt + 1)
                        self.logger.warning(f"Batch {batch_num} attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        failed += len(batch)
                        errors.append(f"Batch {batch_num} failed after {self.config.retry_attempts} attempts: {e}")
        
        return {
            'successful': successful,
            'failed': failed,
            'duplicates': duplicates,
            'errors': errors
        }
    
    def _check_existing_points(self, point_ids: List[str]) -> List[str]:
        """Check which point IDs already exist in the collection"""
        try:
            # Retrieve points by IDs
            response = self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=point_ids,
                with_payload=False,
                with_vectors=False
            )
            
            existing_ids = [str(point.id) for point in response]
            return existing_ids
            
        except Exception as e:
            self.logger.warning(f"Failed to check existing points: {e}")
            return []  # Assume no duplicates on error
    
    def _generate_point_id(self, embedding: Any) -> str:
        """Generate deterministic point ID for idempotent operations"""
        
        # Create deterministic ID based on content hash and chunk metadata
        content_components = [
            embedding.text_content,
            embedding.chunk_id,
            str(embedding.metadata.token_count),
            embedding.metadata.model_version
        ]
        
        content_string = "|".join(content_components)
        content_hash = hashlib.sha256(content_string.encode()).hexdigest()
        
        # Use UUID5 for deterministic UUID generation
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
        point_uuid = uuid.uuid5(namespace, content_hash)
        
        return str(point_uuid)
    
    def _create_nic_payload(self, embedding: Any) -> Dict[str, Any]:
        """Create payload following NIC Schema requirements"""
        
        # Extract document metadata
        doc_metadata = embedding.chunk_metadata.get('document_metadata', {})
        chunk_metadata = embedding.metadata
        
        payload = {
            # Core identifiers
            'chunk_id': embedding.chunk_id,
            'document_title': doc_metadata.get('title', 'Unknown Document'),
            'document_path': doc_metadata.get('file_path', ''),
            
            # Content information
            'content': embedding.text_content,
            'section_title': embedding.chunk_metadata.get('section_title', ''),
            'token_count': chunk_metadata.token_count,
            'chunk_index': chunk_metadata.chunk_index,
            'total_chunks': chunk_metadata.total_chunks,
            'page_number': embedding.chunk_metadata.get('page_number', 0),
            'chunk_type': chunk_metadata.chunk_type.value if hasattr(chunk_metadata.chunk_type, 'value') else str(chunk_metadata.chunk_type),
            
            # Processing metadata
            'embedding_model': chunk_metadata.model_version,
            'processing_timestamp': chunk_metadata.generation_timestamp.isoformat(),
            
            # GitLab lineage (from document metadata)
            'gitlab_commit': doc_metadata.get('commit_sha', ''),
            'gitlab_branch': doc_metadata.get('branch', 'main'),
            'gitlab_url': doc_metadata.get('gitlab_url', ''),
            
            # Processing lineage
            'ocr_applied': doc_metadata.get('ocr_applied', False),
            'is_latest': doc_metadata.get('is_latest', True),
            
            # Quality metrics
            'quality_score': chunk_metadata.quality_score,
            'was_truncated': chunk_metadata.was_truncated,
            'semantic_coherence_score': getattr(chunk_metadata, 'semantic_coherence_score', 0.0),
            
            # Additional metadata for search and filtering
            'hierarchy_path': embedding.chunk_metadata.get('hierarchy_path', []),
            'processing_pipeline_version': '1.0'
        }
        
        return payload
    
    def _validate_single_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single payload against NIC Schema"""
        
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['chunk_id', 'content', 'document_title', 'token_count']
        for field in required_fields:
            if field not in payload or payload[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate field types
        for field, expected_type in self.NIC_SCHEMA_FIELDS.items():
            if field in payload and payload[field] is not None:
                if not isinstance(payload[field], expected_type):
                    try:
                        # Attempt type conversion
                        payload[field] = expected_type(payload[field])
                        warnings.append(f"Converted {field} to {expected_type.__name__}")
                    except (ValueError, TypeError):
                        errors.append(f"Field {field} has invalid type: expected {expected_type.__name__}, got {type(payload[field]).__name__}")
        
        # Validate content length
        if 'content' in payload and len(str(payload['content'])) > 10000:
            warnings.append("Content field is very long (>10000 chars)")
        
        # Validate token count reasonableness
        if 'token_count' in payload and payload['token_count'] > 2000:
            warnings.append(f"Token count seems high: {payload['token_count']}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def search_vectors(self, query_vector: np.ndarray, limit: int = 10, 
                      filters: Optional[Dict[str, Any]] = None,
                      score_threshold: Optional[float] = None) -> List[SearchResult]:
        """Search for similar vectors with optional filtering"""
        
        try:
            # Prepare search parameters
            search_params = SearchParams(
                hnsw_ef=128,  # Higher values = better accuracy, slower search
                exact=False   # Use approximate search for better performance
            )
            
            # Prepare filter conditions
            filter_conditions = None
            if filters:
                filter_conditions = self._build_filter_conditions(filters)
            
            # Execute search
            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=filter_conditions,
                limit=limit,
                params=search_params,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                search_result = SearchResult(
                    point_id=str(result.id),
                    score=result.score,
                    payload=result.payload or {}
                )
                results.append(search_result)
            
            self.logger.debug(f"Found {len(results)} similar vectors")
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            raise
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """Build Qdrant filter conditions from filter dictionary"""
        
        conditions = []
        
        for field, value in filters.items():
            if isinstance(value, str):
                condition = FieldCondition(
                    key=field,
                    match=models.MatchValue(value=value)
                )
            elif isinstance(value, (int, float)):
                condition = FieldCondition(
                    key=field,
                    match=models.MatchValue(value=value)
                )
            elif isinstance(value, list):
                condition = FieldCondition(
                    key=field,
                    match=models.MatchAny(any=value)
                )
            else:
                continue  # Skip unsupported filter types
            
            conditions.append(condition)
        
        if conditions:
            return Filter(must=conditions)
        
        return None
    
    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """Get comprehensive collection information"""
        
        try:
            collection = self.client.get_collection(collection_name)
            
            # Get collection statistics
            collection_info = CollectionInfo(
                name=collection_name,
                status=collection.status.value,
                vector_count=collection.points_count or 0,
                segments_count=len(collection.segments),
                disk_usage_bytes=sum(seg.disk_usage_bytes or 0 for seg in collection.segments),
                memory_usage_bytes=sum(seg.ram_usage_bytes or 0 for seg in collection.segments),
                indexed_vectors_count=collection.indexed_vectors_count or 0,
                configuration={
                    'vector_size': collection.config.params.vectors.size,
                    'distance_metric': collection.config.params.vectors.distance.value,
                    'optimizer_config': str(collection.config.optimizer_config)
                }
            )
            
            return collection_info
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            raise
    
    def health_check(self) -> HealthStatus:
        """Perform comprehensive health check"""
        
        start_time = time.time()
        errors = []
        accessible_collections = []
        
        try:
            # Test basic connectivity
            collections = self.client.get_collections()
            response_time = (time.time() - start_time) * 1000
            
            # Check collection accessibility
            for collection in collections.collections:
                try:
                    self.client.get_collection(collection.name)
                    accessible_collections.append(collection.name)
                except Exception as e:
                    errors.append(f"Collection {collection.name} not accessible: {e}")
            
            # Check target collection specifically
            if self.config.collection_name not in accessible_collections:
                errors.append(f"Target collection '{self.config.collection_name}' not accessible")
            
            is_healthy = len(errors) == 0
            
            return HealthStatus(
                is_healthy=is_healthy,
                response_time_ms=response_time,
                collections_accessible=accessible_collections,
                errors=errors
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                is_healthy=False,
                response_time_ms=response_time,
                collections_accessible=[],
                errors=[f"Health check failed: {e}"]
            )
    
    def optimize_collection(self, collection_name: Optional[str] = None) -> bool:
        """Trigger collection optimization for better performance"""
        
        target_collection = collection_name or self.config.collection_name
        
        try:
            # Trigger optimization
            operation_result = self.client.create_snapshot(collection_name=target_collection)
            self.logger.info(f"Collection optimization triggered for '{target_collection}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Collection optimization failed: {e}")
            return False

def create_qdrant_vector_store(config_dict: Dict[str, Any]) -> QdrantVectorStore:
    """Factory function for Qdrant vector store creation"""
    config = QdrantConfig(**config_dict)
    return QdrantVectorStore(config)
```

### Error Handling
**Comprehensive Qdrant Integration Error Management**
```python
class QdrantIntegrationError(Exception):
    """Base exception for Qdrant integration errors"""
    pass

class ConnectionError(QdrantIntegrationError):
    """Qdrant connection and authentication errors"""
    pass

class CollectionError(QdrantIntegrationError):
    """Collection management errors"""
    pass

class InsertionError(QdrantIntegrationError):
    """Vector insertion and update errors"""
    pass

class SearchError(QdrantIntegrationError):
    """Vector search and retrieval errors"""
    pass

# Retry decorators and circuit breaker patterns
def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt)
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        raise
            return None
        return wrapper
    return decorator
```

## VALIDATION LOOP
**Comprehensive Testing Strategy**

### Unit Testing
```python
# tests/test_qdrant_integration.py
import pytest
import numpy as np
from unittest.mock import Mock, patch
from modules.qdrant_integration import QdrantVectorStore, QdrantConfig

class TestQdrantVectorStore:
    
    @pytest.fixture
    def default_config(self):
        return QdrantConfig(
            url="http://localhost:6333",
            collection_name="test_collection",
            api_key=None  # For testing
        )
    
    @pytest.fixture
    def mock_embeddings(self):
        from modules.embedding_generation import ChunkEmbedding, EmbeddingMetadata
        
        embeddings = []
        for i in range(3):
            metadata = EmbeddingMetadata(
                chunk_id=f"test_chunk_{i}",
                embedding_hash="test_hash",
                model_version="BAAI/bge-m3",
                generation_timestamp=datetime.utcnow(),
                processing_time_ms=10.0,
                sequence_length=100,
                was_truncated=False,
                quality_score=0.9
            )
            
            embedding = ChunkEmbedding(
                chunk_id=f"test_chunk_{i}",
                text_content=f"Test content {i}",
                embedding_vector=np.random.rand(1024),
                metadata=metadata,
                chunk_metadata={'document_metadata': {}}
            )
            embeddings.append(embedding)
        
        return embeddings
    
    @patch('modules.qdrant_integration.QdrantClient')
    def test_client_initialization(self, mock_client_class, default_config):
        """Test Qdrant client initialization"""
        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client
        
        vector_store = QdrantVectorStore(default_config)
        
        assert vector_store.client is not None
        mock_client_class.assert_called_once()
        mock_client.get_collections.assert_called_once()
    
    def test_point_id_generation(self, default_config):
        """Test deterministic point ID generation"""
        vector_store = QdrantVectorStore(default_config)
        
        # Mock embedding
        embedding = Mock()
        embedding.text_content = "Test content"
        embedding.chunk_id = "test_chunk_1"
        embedding.metadata.token_count = 50
        embedding.metadata.model_version = "BAAI/bge-m3"
        
        id1 = vector_store._generate_point_id(embedding)
        id2 = vector_store._generate_point_id(embedding)
        
        assert id1 == id2  # Should be deterministic
        assert len(id1) == 36  # UUID format
    
    def test_nic_payload_creation(self, default_config, mock_embeddings):
        """Test NIC Schema payload creation"""
        vector_store = QdrantVectorStore(default_config)
        
        embedding = mock_embeddings[0]
        payload = vector_store._create_nic_payload(embedding)
        
        # Check required fields
        required_fields = ['chunk_id', 'content', 'document_title', 'token_count']
        for field in required_fields:
            assert field in payload
        
        assert payload['chunk_id'] == embedding.chunk_id
        assert payload['content'] == embedding.text_content
        assert payload['token_count'] == embedding.metadata.token_count
    
    def test_payload_validation(self, default_config):
        """Test payload validation against NIC Schema"""
        vector_store = QdrantVectorStore(default_config)
        
        # Valid payload
        valid_payload = {
            'chunk_id': 'test_chunk',
            'content': 'Test content',
            'document_title': 'Test Document',
            'token_count': 50
        }
        
        result = vector_store._validate_single_payload(valid_payload)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        
        # Invalid payload (missing required field)
        invalid_payload = {
            'content': 'Test content'
        }
        
        result = vector_store._validate_single_payload(invalid_payload)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
```

### Integration Testing
```python
# tests/integration/test_qdrant_live.py
@pytest.mark.integration
def test_live_qdrant_operations():
    """Integration test with live Qdrant instance"""
    
    # Requires live Qdrant instance
    config = QdrantConfig(
        url=os.getenv('QDRANT_URL', 'http://localhost:6333'),
        collection_name='test_integration'
    )
    
    vector_store = QdrantVectorStore(config)
    
    # Test collection creation
    assert vector_store.ensure_collection('test_integration') is True
    
    # Test health check
    health = vector_store.health_check()
    assert health.is_healthy is True
    
    # Test vector insertion and search
    test_vector = np.random.rand(1024)
    mock_embedding = create_mock_embedding(test_vector)
    
    result = vector_store.insert_vectors([mock_embedding])
    assert result.successful_insertions == 1
    
    # Test search
    search_results = vector_store.search_vectors(test_vector, limit=5)
    assert len(search_results) > 0
    assert search_results[0].score > 0.9  # Should find exact match
```

### Performance Testing
- **Insertion Speed**: Target >1000 vectors/second for batch operations
- **Search Latency**: Sub-second response times for typical queries
- **Memory Usage**: Monitor memory consumption during large batch operations
- **Collection Optimization**: Test optimization effects on search performance

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
- **API Key Management**: Secure storage and rotation of Qdrant API keys
- **Network Security**: HTTPS enforcement for all Qdrant communications
- **Access Control**: Implement collection-level access controls where possible
- **Data Privacy**: Ensure sensitive payload data is appropriately handled

### Performance Optimization
- **Batch Processing**: Optimize batch sizes based on network and memory constraints
- **Index Configuration**: Tune HNSW parameters for optimal search performance
- **Collection Optimization**: Regular optimization for improved search speed
- **Connection Pooling**: Reuse client connections for better performance

### Maintenance Requirements
- **Health Monitoring**: Regular health checks and performance monitoring
- **Index Maintenance**: Periodic optimization and compaction operations
- **Backup Strategy**: Implement collection backup and recovery procedures
- **Version Compatibility**: Monitor Qdrant updates and compatibility requirements