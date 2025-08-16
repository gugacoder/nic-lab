import logging
from typing import List, Dict, Optional, Any, Union, Tuple

# Optional numpy import with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create simple mock numpy for basic operations
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def linalg_norm(vector):
            if hasattr(vector, '__iter__'):
                return sum(x*x for x in vector) ** 0.5
            return abs(vector)
        
        @staticmethod
        def dot(v1, v2):
            return sum(a*b for a, b in zip(v1, v2))
        
        @staticmethod
        def any(condition):
            return any(condition)
        
        @staticmethod
        def isnan(x):
            return x != x  # NaN check
        
        @staticmethod
        def isinf(x):
            return x == float('inf') or x == float('-inf')
        
        class ndarray:
            def __init__(self, data):
                self.data = list(data) if hasattr(data, '__iter__') else [data]
                self.shape = (len(self.data),) if hasattr(data, '__iter__') else (1,)
            
            def tolist(self):
                return self.data
            
            def __iter__(self):
                return iter(self.data)
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, index):
                return self.data[index]
    
    np = MockNumpy()
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import uuid
from uuid import UUID
import time
import json
from pathlib import Path

# Optional qdrant-client import with fallback
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, CollectionStatus, PointStruct, 
        Filter, FieldCondition, SearchRequest, SearchParams,
        UpdateStatus, OptimizersConfigDiff, CollectionParams
    )
    from qdrant_client.http import models
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False
    
    # Create mock classes for development
    class Distance:
        COSINE = "cosine"
        EUCLIDEAN = "euclidean"
        DOT = "dot"
    
    class UpdateStatus:
        COMPLETED = "completed"
        ACKNOWLEDGED = "acknowledged"
    
    class VectorParams:
        def __init__(self, size: int, distance: str):
            self.size = size
            self.distance = distance
    
    class PointStruct:
        def __init__(self, id: str, vector: List[float], payload: Dict[str, Any]):
            self.id = id
            self.vector = vector
            self.payload = payload
    
    class OptimizersConfigDiff:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class MockCollection:
        def __init__(self, name: str):
            self.name = name
            self.status = UpdateStatus.COMPLETED
            self.points_count = 0
            self.segments = []
            self.indexed_vectors_count = 0
            self.config = type('Config', (), {
                'params': type('Params', (), {
                    'vectors': VectorParams(1024, Distance.COSINE)
                })(),
                'optimizer_config': {}
            })()
    
    class MockCollectionsList:
        def __init__(self):
            self.collections = []
    
    class MockQdrantClient:
        def __init__(self, url: str, api_key: Optional[str] = None, timeout: float = 30.0):
            self.url = url
            self.api_key = api_key
            self.timeout = timeout
            self.collections_db = {}
            self.points_db = {}
        
        def get_collections(self):
            return MockCollectionsList()
        
        def get_collection(self, collection_name: str):
            if collection_name not in self.collections_db:
                raise Exception(f"Collection {collection_name} not found")
            return self.collections_db[collection_name]
        
        def create_collection(self, collection_name: str, vectors_config: VectorParams, 
                            optimizers_config: Optional[OptimizersConfigDiff] = None):
            collection = MockCollection(collection_name)
            self.collections_db[collection_name] = collection
            self.points_db[collection_name] = {}
            return True
        
        def upsert(self, collection_name: str, points: List[PointStruct]):
            if collection_name not in self.points_db:
                self.points_db[collection_name] = {}
            
            for point in points:
                self.points_db[collection_name][point.id] = point
            
            return type('Result', (), {'status': UpdateStatus.COMPLETED})()
        
        def retrieve(self, collection_name: str, ids: List[str], 
                    with_payload: bool = True, with_vectors: bool = True):
            if collection_name not in self.points_db:
                return []
            
            results = []
            for point_id in ids:
                if point_id in self.points_db[collection_name]:
                    point = self.points_db[collection_name][point_id]
                    results.append(type('Point', (), {'id': point.id})())
            
            return results
        
        def search(self, collection_name: str, query_vector: List[float], 
                  query_filter: Optional[Any] = None, limit: int = 10, 
                  params: Optional[Any] = None, score_threshold: Optional[float] = None,
                  with_payload: bool = True, with_vectors: bool = False):
            # Mock search - return random results
            results = []
            if collection_name in self.points_db:
                points = list(self.points_db[collection_name].values())[:limit]
                for i, point in enumerate(points):
                    results.append(type('SearchResult', (), {
                        'id': point.id,
                        'score': 0.9 - (i * 0.1),
                        'payload': point.payload if with_payload else {}
                    })())
            
            return results
        
        def create_snapshot(self, collection_name: str):
            return True
    
    # Use mock client when real client not available
    QdrantClient = MockQdrantClient
    
    # Mock models module
    class MockModels:
        class MatchValue:
            def __init__(self, value: Any):
                self.value = value
        
        class MatchAny:
            def __init__(self, any: List[Any]):
                self.any = any
    
    models = MockModels()

@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database"""
    url: str = "https://qdrant.codrstudio.dev/"
    api_key: Optional[str] = None
    collection_name: str = "nic"
    vector_size: int = 1024
    distance_metric: str = "cosine"  # Using string instead of Distance enum for flexibility
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
        self.qdrant_available = QDRANT_CLIENT_AVAILABLE
        
        # Initialize Qdrant client
        self._initialize_client()
        
        # Ensure collection exists
        self.ensure_collection(config.collection_name)
        
    def _initialize_client(self):
        """Initialize Qdrant client with authentication and configuration"""
        try:
            if not self.qdrant_available:
                self.logger.warning("Qdrant client not available, using mock client")
            
            self.client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            
            # Test connection
            collections = self.client.get_collections()
            self.logger.info(f"Connected to Qdrant at {self.config.url}")
            
            if self.qdrant_available and hasattr(collections, 'collections'):
                self.logger.info(f"Available collections: {[c.name for c in collections.collections]}")
            else:
                self.logger.info("Using mock Qdrant client for development")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant client: {e}")
            raise QdrantIntegrationError(f"Client initialization failed: {e}")
    
    def ensure_collection(self, collection_name: str) -> bool:
        """Ensure collection exists with proper configuration"""
        try:
            # Check if collection exists
            try:
                collection_info = self.client.get_collection(collection_name)
                self.logger.info(f"Collection '{collection_name}' already exists")
                
                # Validate collection configuration if using real client
                if self.qdrant_available and hasattr(collection_info, 'config'):
                    config = collection_info.config
                    actual_size = config.params.vectors.size
                    actual_distance = config.params.vectors.distance
                    
                    if (actual_size != self.config.vector_size or
                        str(actual_distance).lower() != self.config.distance_metric.lower()):
                        
                        self.logger.warning(
                            f"Collection configuration mismatch. "
                            f"Expected: size={self.config.vector_size}, distance={self.config.distance_metric}. "
                            f"Actual: size={actual_size}, distance={actual_distance}"
                        )
                        return False
                
                return True
                
            except Exception:
                # Collection doesn't exist, create it
                self.logger.info(f"Creating collection '{collection_name}'")
                
                # Map string distance to appropriate format
                distance_value = self.config.distance_metric.upper()
                if hasattr(Distance, distance_value):
                    distance_enum = getattr(Distance, distance_value)
                else:
                    distance_enum = Distance.COSINE
                
                vectors_config = VectorParams(
                    size=self.config.vector_size,
                    distance=distance_enum
                )
                
                optimizers_config = None
                if self.config.optimize_collection:
                    optimizers_config = OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=None,
                        memmap_threshold=None,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=1
                    )
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                    optimizers_config=optimizers_config
                )
                
                self.logger.info(f"Collection '{collection_name}' created successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to ensure collection '{collection_name}': {e}")
            raise CollectionError(f"Collection setup failed: {e}")
    
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
                    
                    # Ensure vector is the right format and type
                    if hasattr(embedding, 'embedding_vector'):
                        vector = embedding.embedding_vector
                        if isinstance(vector, np.ndarray):
                            vector = vector.tolist()
                    else:
                        self.logger.error(f"Embedding missing vector data: {embedding}")
                        continue
                    
                    # Create point structure
                    point = PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                    points.append(point)
                    
                except Exception as e:
                    chunk_id = getattr(embedding, 'chunk_id', 'unknown')
                    self.logger.error(f"Failed to prepare point for embedding {chunk_id}: {e}")
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
                        
                        # Check operation status
                        if hasattr(operation_result, 'status'):
                            if operation_result.status == UpdateStatus.COMPLETED:
                                successful += len(new_points)
                                self.logger.debug(f"Batch {batch_num} inserted successfully ({len(new_points)} new points)")
                            else:
                                failed += len(new_points)
                                errors.append(f"Batch {batch_num} failed with status: {operation_result.status}")
                        else:
                            # Assume success if no status (mock client)
                            successful += len(new_points)
                            self.logger.debug(f"Batch {batch_num} inserted successfully ({len(new_points)} new points)")
                    
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
        chunk_id = getattr(embedding, 'chunk_id', 'unknown')
        text_content = getattr(embedding, 'text_content', '')
        
        # Get metadata attributes safely
        metadata = getattr(embedding, 'metadata', None)
        if metadata:
            token_count = getattr(metadata, 'token_count', 0)
            model_version = getattr(metadata, 'model_version', 'unknown')
        else:
            token_count = 0
            model_version = 'unknown'
        
        content_components = [
            text_content,
            chunk_id,
            str(token_count),
            model_version
        ]
        
        content_string = "|".join(content_components)
        content_hash = hashlib.sha256(content_string.encode()).hexdigest()
        
        # Use UUID5 for deterministic UUID generation
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
        point_uuid = uuid.uuid5(namespace, content_hash)
        
        return str(point_uuid)
    
    def _create_nic_payload(self, embedding: Any) -> Dict[str, Any]:
        """Create payload following NIC Schema requirements"""
        
        # Extract document metadata safely
        chunk_metadata = getattr(embedding, 'chunk_metadata', {})
        if isinstance(chunk_metadata, dict):
            doc_metadata = chunk_metadata.get('document_metadata', {})
        else:
            doc_metadata = {}
        
        # Extract embedding metadata safely
        metadata = getattr(embedding, 'metadata', None)
        chunk_id = getattr(embedding, 'chunk_id', 'unknown')
        text_content = getattr(embedding, 'text_content', '')
        
        payload = {
            # Core identifiers
            'chunk_id': chunk_id,
            'document_title': doc_metadata.get('title', 'Unknown Document'),
            'document_path': doc_metadata.get('file_path', ''),
            
            # Content information
            'content': text_content,
            'section_title': chunk_metadata.get('section_title', ''),
            'token_count': getattr(metadata, 'token_count', 0) if metadata else 0,
            'chunk_index': getattr(metadata, 'chunk_index', 0) if metadata else 0,
            'total_chunks': getattr(metadata, 'total_chunks', 1) if metadata else 1,
            'page_number': chunk_metadata.get('page_number', 0),
            'chunk_type': self._safe_get_chunk_type(metadata),
            
            # Processing metadata
            'embedding_model': getattr(metadata, 'model_version', 'unknown') if metadata else 'unknown',
            'processing_timestamp': self._safe_get_timestamp(metadata),
            
            # GitLab lineage (from document metadata)
            'gitlab_commit': doc_metadata.get('commit_sha', ''),
            'gitlab_branch': doc_metadata.get('branch', 'main'),
            'gitlab_url': doc_metadata.get('gitlab_url', ''),
            
            # Processing lineage
            'ocr_applied': doc_metadata.get('ocr_applied', False),
            'is_latest': doc_metadata.get('is_latest', True),
            
            # Quality metrics
            'quality_score': getattr(metadata, 'quality_score', 0.0) if metadata else 0.0,
            'was_truncated': getattr(metadata, 'was_truncated', False) if metadata else False,
            'semantic_coherence_score': getattr(metadata, 'semantic_coherence_score', 0.0) if metadata else 0.0,
            
            # Additional metadata for search and filtering
            'hierarchy_path': chunk_metadata.get('hierarchy_path', []),
            'processing_pipeline_version': '1.0'
        }
        
        return payload
    
    def _safe_get_chunk_type(self, metadata) -> str:
        """Safely extract chunk type from metadata"""
        if not metadata:
            return 'unknown'
        
        chunk_type = getattr(metadata, 'chunk_type', None)
        if chunk_type:
            if hasattr(chunk_type, 'value'):
                return chunk_type.value
            else:
                return str(chunk_type)
        
        return 'paragraph'
    
    def _safe_get_timestamp(self, metadata) -> str:
        """Safely extract timestamp from metadata"""
        if not metadata:
            return datetime.utcnow().isoformat()
        
        timestamp = getattr(metadata, 'generation_timestamp', None)
        if timestamp:
            if hasattr(timestamp, 'isoformat'):
                return timestamp.isoformat()
            else:
                return str(timestamp)
        
        return datetime.utcnow().isoformat()
    
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
            search_params = None
            if self.qdrant_available:
                search_params = SearchParams(
                    hnsw_ef=128,  # Higher values = better accuracy, slower search
                    exact=False   # Use approximate search for better performance
                )
            
            # Prepare filter conditions
            filter_conditions = None
            if filters:
                filter_conditions = self._build_filter_conditions(filters)
            
            # Convert numpy array to list
            if isinstance(query_vector, np.ndarray):
                query_vector_list = query_vector.tolist()
            else:
                query_vector_list = query_vector
            
            # Execute search
            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector_list,
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
                    payload=getattr(result, 'payload', {}) or {}
                )
                results.append(search_result)
            
            self.logger.debug(f"Found {len(results)} similar vectors")
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            raise SearchError(f"Search operation failed: {e}")
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> Optional[Any]:
        """Build Qdrant filter conditions from filter dictionary"""
        
        if not self.qdrant_available:
            # Return None for mock client
            return None
        
        conditions = []
        
        for field, value in filters.items():
            condition = None
            
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
            
            if condition:
                conditions.append(condition)
        
        if conditions:
            return Filter(must=conditions)
        
        return None
    
    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """Get comprehensive collection information"""
        
        try:
            collection = self.client.get_collection(collection_name)
            
            # Extract information safely (handle both real and mock clients)
            if hasattr(collection, 'status'):
                status = collection.status.value if hasattr(collection.status, 'value') else str(collection.status)
            else:
                status = 'unknown'
            
            points_count = getattr(collection, 'points_count', 0) or 0
            segments = getattr(collection, 'segments', [])
            indexed_vectors_count = getattr(collection, 'indexed_vectors_count', 0) or 0
            
            # Calculate disk and memory usage
            disk_usage = sum(getattr(seg, 'disk_usage_bytes', 0) or 0 for seg in segments)
            memory_usage = sum(getattr(seg, 'ram_usage_bytes', 0) or 0 for seg in segments)
            
            # Extract configuration
            configuration = {}
            if hasattr(collection, 'config') and collection.config:
                config = collection.config
                if hasattr(config, 'params') and hasattr(config.params, 'vectors'):
                    vectors_config = config.params.vectors
                    configuration = {
                        'vector_size': getattr(vectors_config, 'size', self.config.vector_size),
                        'distance_metric': str(getattr(vectors_config, 'distance', self.config.distance_metric)),
                        'optimizer_config': str(getattr(config, 'optimizer_config', {}))
                    }
            
            collection_info = CollectionInfo(
                name=collection_name,
                status=status,
                vector_count=points_count,
                segments_count=len(segments),
                disk_usage_bytes=disk_usage,
                memory_usage_bytes=memory_usage,
                indexed_vectors_count=indexed_vectors_count,
                configuration=configuration
            )
            
            return collection_info
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            raise CollectionError(f"Collection info retrieval failed: {e}")
    
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
            if hasattr(collections, 'collections'):
                for collection in collections.collections:
                    try:
                        self.client.get_collection(collection.name)
                        accessible_collections.append(collection.name)
                    except Exception as e:
                        errors.append(f"Collection {collection.name} not accessible: {e}")
            
            # Check target collection specifically
            if self.config.collection_name not in accessible_collections:
                try:
                    self.client.get_collection(self.config.collection_name)
                    accessible_collections.append(self.config.collection_name)
                except Exception as e:
                    errors.append(f"Target collection '{self.config.collection_name}' not accessible: {e}")
            
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
            # Trigger optimization via snapshot creation (as per PRP spec)
            operation_result = self.client.create_snapshot(collection_name=target_collection)
            self.logger.info(f"Collection optimization triggered for '{target_collection}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Collection optimization failed: {e}")
            return False
    
    def get_store_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics and capabilities"""
        return {
            'qdrant_available': self.qdrant_available,
            'collection_name': self.config.collection_name,
            'vector_size': self.config.vector_size,
            'distance_metric': self.config.distance_metric,
            'batch_size': self.config.batch_size,
            'url': self.config.url,
            'payload_validation_enabled': self.config.enable_payload_validation,
            'retry_attempts': self.config.retry_attempts,
            'supported_schema_fields': list(self.NIC_SCHEMA_FIELDS.keys())
        }

# Context manager for Qdrant vector store
class QdrantVectorStoreContext:
    """Context manager for Qdrant operations with automatic cleanup"""
    
    def __init__(self, config: QdrantConfig):
        self.config = config
        self.store = None
    
    def __enter__(self) -> QdrantVectorStore:
        self.store = QdrantVectorStore(self.config)
        return self.store
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass

def create_qdrant_vector_store(config_dict: Dict[str, Any]) -> QdrantVectorStore:
    """Factory function for Qdrant vector store creation"""
    config = QdrantConfig(
        url=config_dict.get('url', 'https://qdrant.codrstudio.dev/'),
        api_key=config_dict.get('api_key'),
        collection_name=config_dict.get('collection_name', 'nic'),
        vector_size=config_dict.get('vector_size', 1024),
        distance_metric=config_dict.get('distance_metric', 'cosine'),
        timeout=config_dict.get('timeout', 30.0),
        retry_attempts=config_dict.get('retry_attempts', 3),
        retry_delay=config_dict.get('retry_delay', 1.0),
        batch_size=config_dict.get('batch_size', 100),
        enable_payload_validation=config_dict.get('enable_payload_validation', True),
        optimize_collection=config_dict.get('optimize_collection', True)
    )
    return QdrantVectorStore(config)

# Error classes
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

# Retry decorator
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

# Utility functions
def safe_vector_insertion(store: QdrantVectorStore, embeddings: List[Any]) -> InsertionResult:
    """Safe vector insertion with fallback strategies"""
    try:
        return store.insert_vectors(embeddings)
    except Exception as e:
        if "memory" in str(e).lower() or "timeout" in str(e).lower():
            # Reduce batch size and retry
            original_batch_size = store.config.batch_size
            store.config.batch_size = max(1, original_batch_size // 2)
            logging.warning(f"Reducing batch size to {store.config.batch_size} due to: {e}")
            try:
                result = store.insert_vectors(embeddings)
                store.config.batch_size = original_batch_size  # Restore original
                return result
            except Exception:
                store.config.batch_size = original_batch_size  # Restore original
                pass
        
        logging.error(f"Vector insertion failed completely: {e}")
        return InsertionResult(
            total_inserted=len(embeddings),
            successful_insertions=0,
            failed_insertions=len(embeddings),
            duplicate_skipped=0,
            processing_time_seconds=0.0,
            errors=[str(e)]
        )

def validate_vector_format(vector: Any, expected_dim: int = 1024) -> bool:
    """Validate vector format and dimensions"""
    try:
        # Handle both real numpy arrays and mock arrays
        if NUMPY_AVAILABLE:
            if not isinstance(vector, np.ndarray):
                return False
            
            if len(vector.shape) != 1:
                return False
            
            if vector.shape[0] != expected_dim:
                return False
            
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                return False
        else:
            # Handle mock numpy case
            if hasattr(vector, 'shape'):
                if len(vector.shape) != 1:
                    return False
                if vector.shape[0] != expected_dim:
                    return False
            elif hasattr(vector, '__len__'):
                if len(vector) != expected_dim:
                    return False
            else:
                return False
            
            # Check for NaN/Inf in mock case
            try:
                for val in vector:
                    if np.isnan(val) or np.isinf(val):
                        return False
            except (TypeError, AttributeError):
                pass
        
        return True
        
    except Exception:
        return False

def calculate_vector_similarity(vector1: Any, vector2: Any) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        # Normalize vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(vector1, vector2) / (norm1 * norm2)
        return float(similarity)
        
    except Exception:
        return 0.0