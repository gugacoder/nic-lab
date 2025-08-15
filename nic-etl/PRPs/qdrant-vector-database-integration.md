# QDrant Vector Database Integration - PRP

## ROLE
**Backend Developer with Vector Database and Search Engine expertise**

Responsible for implementing comprehensive QDrant vector database integration for storing and managing document embeddings. Must have experience with vector databases, similarity search, collection management, and high-performance data insertion workflows.

## OBJECTIVE
**Implement robust QDrant integration for vector storage and semantic search**

Develop a comprehensive QDrant integration system that:
- Creates and manages QDrant collections with optimal configurations
- Implements efficient batch upsert operations for embeddings and metadata
- Provides collection existence checking and automatic creation
- Generates stable, consistent IDs for document chunks
- Implements error handling and retry mechanisms for database operations
- Provides search functionality and collection statistics
- Ensures data consistency and integrity across operations

Success criteria: Successfully store 100% of generated embeddings with stable IDs, achieve >500 insertions/second throughput, and provide <100ms average search response times.

## MOTIVATION
**Enable fast, scalable semantic search across document corpus**

QDrant vector database serves as the foundation for semantic search capabilities, storing high-dimensional embeddings alongside rich metadata. Efficient integration ensures fast similarity search, scalable storage, and reliable data persistence for the NIC knowledge base.

## CONTEXT
**NIC ETL Pipeline - Vector Database Integration Phase**

Technology Stack:
- Python 3.8+ with jupyter notebook environment
- QDrant vector database (hosted at https://qdrant.codrstudio.dev/)
- qdrant-client Python library
- Input from embedding generation pipeline
- Collection name: "nic"

QDrant Configuration:
- Vector size: 1024 dimensions
- Distance metric: COSINE similarity
- API endpoint: https://qdrant.codrstudio.dev/
- API key: 93f0c9d6b9a53758f2376decf318b3ae300e9bdb50be2d0e9c893ee4469fd857

## IMPLEMENTATION BLUEPRINT

### Architecture Overview
```
Embedding Results → Collection Management → ID Generation → Batch Upsert → Search Interface → Collection Monitoring
```

### Code Structure
```python
# File organization
src/
├── qdrant_integration/
│   ├── __init__.py
│   ├── qdrant_client_wrapper.py   # QDrant client management
│   ├── collection_manager.py      # Collection creation and management
│   ├── data_uploader.py           # Batch upload operations
│   ├── id_generator.py            # Stable ID generation
│   ├── search_interface.py        # Search functionality
│   └── qdrant_orchestrator.py     # Main QDrant pipeline
├── models/
│   └── qdrant_models.py          # Data models for QDrant operations
└── notebooks/
    └── 08_qdrant_integration.ipynb
```

### QDrant Client Wrapper
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionInfo, PointStruct
from qdrant_client.http import models
from typing import List, Dict, Any, Optional, Union
import logging
import time
from dataclasses import dataclass

@dataclass
class QdrantConfig:
    """Configuration for QDrant connection"""
    url: str = "https://qdrant.codrstudio.dev/"
    api_key: str = "93f0c9d6b9a53758f2376decf318b3ae300e9bdb50be2d0e9c893ee4469fd857"
    collection_name: str = "nic"
    vector_size: int = 1024
    distance: Distance = Distance.COSINE
    timeout: float = 30.0
    retry_count: int = 3

class QdrantClientWrapper:
    """Enhanced QDrant client with error handling and retries"""
    
    def __init__(self, config: QdrantConfig = None):
        self.config = config or QdrantConfig()
        self.client = None
        self.logger = logging.getLogger(__name__)
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize QDrant client with configuration"""
        try:
            self.client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            
            # Test connection
            self._test_connection()
            self.logger.info("QDrant client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"QDrant client initialization failed: {e}")
            raise
    
    def _test_connection(self):
        """Test QDrant connection"""
        try:
            collections = self.client.get_collections()
            self.logger.info(f"Connection test successful. Found {len(collections.collections)} collections")
        except Exception as e:
            self.logger.error(f"QDrant connection test failed: {e}")
            raise
    
    def execute_with_retry(self, operation, *args, **kwargs):
        """Execute QDrant operation with retry mechanism"""
        last_exception = None
        
        for attempt in range(self.config.retry_count):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Operation failed (attempt {attempt + 1}/{self.config.retry_count}): {e}")
                
                if attempt < self.config.retry_count - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 1.0
                    time.sleep(wait_time)
        
        # All retries failed
        self.logger.error(f"Operation failed after {self.config.retry_count} attempts")
        raise last_exception
    
    def get_collection_info(self, collection_name: str = None) -> Optional[CollectionInfo]:
        """Get collection information"""
        collection_name = collection_name or self.config.collection_name
        
        try:
            return self.execute_with_retry(
                self.client.get_collection,
                collection_name=collection_name
            )
        except Exception as e:
            self.logger.warning(f"Failed to get collection info: {e}")
            return None
    
    def collection_exists(self, collection_name: str = None) -> bool:
        """Check if collection exists"""
        collection_name = collection_name or self.config.collection_name
        
        try:
            collections = self.execute_with_retry(self.client.get_collections)
            return any(col.name == collection_name for col in collections.collections)
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {e}")
            return False

class CollectionManager:
    """Manage QDrant collection lifecycle"""
    
    def __init__(self, client_wrapper: QdrantClientWrapper):
        self.client_wrapper = client_wrapper
        self.client = client_wrapper.client
        self.config = client_wrapper.config
        self.logger = logging.getLogger(__name__)
    
    def ensure_collection_exists(self, collection_name: str = None) -> Dict[str, Any]:
        """Ensure collection exists, create if necessary"""
        collection_name = collection_name or self.config.collection_name
        
        try:
            # Check if collection exists
            if self.client_wrapper.collection_exists(collection_name):
                self.logger.info(f"Collection '{collection_name}' already exists")
                collection_info = self.client_wrapper.get_collection_info(collection_name)
                return {
                    'success': True,
                    'action': 'found_existing',
                    'collection_info': self._serialize_collection_info(collection_info)
                }
            
            # Create collection
            self.logger.info(f"Creating collection '{collection_name}'")
            return self._create_collection(collection_name)
            
        except Exception as e:
            self.logger.error(f"Collection management failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_collection(self, collection_name: str) -> Dict[str, Any]:
        """Create new collection with optimal configuration"""
        try:
            # Define collection configuration
            vector_config = VectorParams(
                size=self.config.vector_size,
                distance=self.config.distance
            )
            
            # Create collection
            self.client_wrapper.execute_with_retry(
                self.client.create_collection,
                collection_name=collection_name,
                vectors_config=vector_config
            )
            
            # Verify creation
            collection_info = self.client_wrapper.get_collection_info(collection_name)
            
            self.logger.info(f"Collection '{collection_name}' created successfully")
            
            return {
                'success': True,
                'action': 'created_new',
                'collection_info': self._serialize_collection_info(collection_info)
            }
            
        except Exception as e:
            self.logger.error(f"Collection creation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_collection_statistics(self, collection_name: str = None) -> Dict[str, Any]:
        """Get comprehensive collection statistics"""
        collection_name = collection_name or self.config.collection_name
        
        try:
            collection_info = self.client_wrapper.get_collection_info(collection_name)
            
            if not collection_info:
                return {'error': 'Collection not found'}
            
            return {
                'collection_name': collection_name,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.value,
                'status': collection_info.status.value,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'payload_schema': collection_info.payload_schema
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection statistics: {e}")
            return {'error': str(e)}
    
    def _serialize_collection_info(self, collection_info) -> Dict[str, Any]:
        """Serialize collection info for JSON compatibility"""
        if not collection_info:
            return {}
        
        return {
            'name': collection_info.name,
            'points_count': collection_info.points_count,
            'segments_count': collection_info.segments_count,
            'vector_size': collection_info.config.params.vectors.size,
            'distance_metric': collection_info.config.params.vectors.distance.value,
            'status': collection_info.status.value
        }
```

### Stable ID Generation
```python
import hashlib
from typing import Dict, Any
import uuid

class StableIDGenerator:
    """Generate stable, consistent IDs for document chunks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """Generate stable ID for document chunk"""
        try:
            # Collect identifying information
            id_components = self._extract_id_components(chunk)
            
            # Create hash-based stable ID
            stable_id = self._create_hash_id(id_components)
            
            return stable_id
            
        except Exception as e:
            self.logger.warning(f"Stable ID generation failed, using fallback: {e}")
            return self._generate_fallback_id(chunk)
    
    def _extract_id_components(self, chunk: Dict[str, Any]) -> Dict[str, str]:
        """Extract components for ID generation"""
        components = {}
        
        # Document-level identifiers
        doc_metadata = chunk.get('document_metadata', {})
        components['doc_title'] = doc_metadata.get('title', '')
        
        # Source lineage identifiers
        lineage = chunk.get('lineage_metadata', {})
        source_lineage = lineage.get('source_lineage', {})
        repo_info = source_lineage.get('repository', {})
        
        components['repo_path'] = repo_info.get('repository_path', '')
        components['file_path'] = repo_info.get('file_path', '')
        components['commit_id'] = repo_info.get('commit_id', '')
        
        # Chunk-specific identifiers
        nic_metadata = chunk.get('nic_metadata', {})
        chunk_position = nic_metadata.get('chunk_position', {})
        
        components['start_paragraph'] = str(chunk_position.get('start_paragraph', 0))
        components['end_paragraph'] = str(chunk_position.get('end_paragraph', 0))
        components['section_path'] = nic_metadata.get('section_path', '')
        
        # Content hash for uniqueness
        text = chunk.get('text', '').strip()
        components['content_hash'] = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        
        return components
    
    def _create_hash_id(self, components: Dict[str, str]) -> str:
        """Create hash-based stable ID from components"""
        # Combine components in a consistent order
        id_string = '|'.join([
            components.get('repo_path', ''),
            components.get('file_path', ''),
            components.get('commit_id', ''),
            components.get('section_path', ''),
            components.get('start_paragraph', ''),
            components.get('end_paragraph', ''),
            components.get('content_hash', '')
        ])
        
        # Generate SHA-256 hash and take first 16 characters
        hash_id = hashlib.sha256(id_string.encode('utf-8')).hexdigest()[:16]
        
        return f"nic_{hash_id}"
    
    def _generate_fallback_id(self, chunk: Dict[str, Any]) -> str:
        """Generate fallback ID if stable ID creation fails"""
        # Use chunk_id if available
        if 'chunk_id' in chunk:
            return f"nic_{chunk['chunk_id']}"
        
        # Use UUID as last resort
        return f"nic_{uuid.uuid4().hex[:16]}"

class DataUploader:
    """Handle batch upload operations to QDrant"""
    
    def __init__(self, client_wrapper: QdrantClientWrapper):
        self.client_wrapper = client_wrapper
        self.client = client_wrapper.client
        self.config = client_wrapper.config
        self.id_generator = StableIDGenerator()
        self.logger = logging.getLogger(__name__)
    
    def upload_embeddings_batch(self, embedding_results: List[Dict[str, Any]], 
                              collection_name: str = None,
                              batch_size: int = 100) -> Dict[str, Any]:
        """Upload embeddings to QDrant in batches"""
        collection_name = collection_name or self.config.collection_name
        
        try:
            # Filter successful embeddings
            successful_results = [r for r in embedding_results if r['embedding_successful']]
            
            if not successful_results:
                return {
                    'success': False,
                    'error': 'No successful embeddings to upload'
                }
            
            self.logger.info(f"Uploading {len(successful_results)} embeddings to collection '{collection_name}'")
            
            # Process in batches
            upload_stats = {
                'total_points': len(successful_results),
                'successful_uploads': 0,
                'failed_uploads': 0,
                'batch_results': []
            }
            
            for batch_start in range(0, len(successful_results), batch_size):
                batch_end = min(batch_start + batch_size, len(successful_results))
                batch_results = successful_results[batch_start:batch_end]
                
                batch_result = self._upload_single_batch(batch_results, collection_name)
                upload_stats['batch_results'].append(batch_result)
                
                if batch_result['success']:
                    upload_stats['successful_uploads'] += batch_result['points_uploaded']
                else:
                    upload_stats['failed_uploads'] += len(batch_results)
            
            upload_stats['success'] = upload_stats['failed_uploads'] == 0
            upload_stats['success_rate'] = upload_stats['successful_uploads'] / upload_stats['total_points']
            
            return upload_stats
            
        except Exception as e:
            self.logger.error(f"Batch upload failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _upload_single_batch(self, batch_results: List[Dict[str, Any]], 
                           collection_name: str) -> Dict[str, Any]:
        """Upload single batch of points to QDrant"""
        try:
            # Prepare points for upload
            points = []
            
            for result in batch_results:
                chunk = result['original_chunk']
                embedding = result['embedding']
                
                # Generate stable ID
                point_id = self.id_generator.generate_chunk_id(chunk)
                
                # Prepare payload
                payload = self._prepare_payload(chunk)
                
                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
                points.append(point)
            
            # Upload batch
            self.client_wrapper.execute_with_retry(
                self.client.upsert,
                collection_name=collection_name,
                points=points
            )
            
            self.logger.info(f"Successfully uploaded batch of {len(points)} points")
            
            return {
                'success': True,
                'points_uploaded': len(points),
                'point_ids': [p.id for p in points]
            }
            
        except Exception as e:
            self.logger.error(f"Single batch upload failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'points_uploaded': 0
            }
    
    def _prepare_payload(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare payload for QDrant point"""
        # Extract key metadata for payload
        doc_metadata = chunk.get('document_metadata', {})
        section_metadata = chunk.get('section_metadata', {})
        nic_metadata = chunk.get('nic_metadata', {})
        
        payload = {
            # Document information
            'document_title': doc_metadata.get('title', ''),
            'document_description': doc_metadata.get('description', ''),
            'document_author': doc_metadata.get('author', []),
            'document_tags': doc_metadata.get('tags', []),
            'document_status': doc_metadata.get('status', ''),
            'document_created': doc_metadata.get('created', ''),
            
            # Section information
            'section_title': section_metadata.get('title', ''),
            'section_path': nic_metadata.get('section_path', ''),
            'section_level': section_metadata.get('level', 0),
            
            # Chunk information
            'chunk_text': chunk.get('text', ''),
            'token_count': nic_metadata.get('chunk_position', {}).get('token_count', 0),
            'start_paragraph': nic_metadata.get('chunk_position', {}).get('start_paragraph', 0),
            'end_paragraph': nic_metadata.get('chunk_position', {}).get('end_paragraph', 0),
            
            # Quality metrics
            'chunk_quality_score': nic_metadata.get('quality_indicators', {}).get('chunk_quality_score', 0),
            'processing_confidence': nic_metadata.get('quality_indicators', {}).get('processing_confidence', 0),
            
            # Processing flags
            'ocr_applied': nic_metadata.get('processing_flags', {}).get('ocr_applied', False),
            'structure_analyzed': nic_metadata.get('processing_flags', {}).get('structure_analyzed', False),
            'is_latest_version': nic_metadata.get('processing_flags', {}).get('is_latest_version', True)
        }
        
        # Clean payload (remove None values and ensure JSON serializable)
        cleaned_payload = self._clean_payload(payload)
        
        return cleaned_payload
    
    def _clean_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Clean payload for QDrant compatibility"""
        cleaned = {}
        
        for key, value in payload.items():
            if value is not None:
                # Ensure JSON serializable types
                if isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                elif isinstance(value, list):
                    # Filter None values from lists
                    cleaned_list = [item for item in value if item is not None]
                    if cleaned_list:
                        cleaned[key] = cleaned_list
                else:
                    # Convert to string for other types
                    cleaned[key] = str(value)
        
        return cleaned
```

### Search Interface
```python
from typing import List, Dict, Any, Optional
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

class QdrantSearchInterface:
    """Search interface for QDrant vector database"""
    
    def __init__(self, client_wrapper: QdrantClientWrapper):
        self.client_wrapper = client_wrapper
        self.client = client_wrapper.client
        self.config = client_wrapper.config
        self.logger = logging.getLogger(__name__)
    
    def semantic_search(self, query_vector: List[float], 
                       limit: int = 10,
                       filters: Dict[str, Any] = None,
                       collection_name: str = None) -> Dict[str, Any]:
        """Perform semantic search using query vector"""
        collection_name = collection_name or self.config.collection_name
        
        try:
            # Prepare search filters
            search_filter = self._build_search_filter(filters) if filters else None
            
            # Perform search
            search_results = self.client_wrapper.execute_with_retry(
                self.client.search,
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format results
            formatted_results = self._format_search_results(search_results)
            
            return {
                'success': True,
                'results': formatted_results,
                'total_found': len(search_results),
                'query_info': {
                    'limit': limit,
                    'filters_applied': filters is not None,
                    'collection_name': collection_name
                }
            }
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return {
                'success': False,
                'results': [],
                'error': str(e)
            }
    
    def text_search(self, query_text: str, 
                   text_fields: List[str] = None,
                   limit: int = 10,
                   collection_name: str = None) -> Dict[str, Any]:
        """Perform text-based search in payload fields"""
        collection_name = collection_name or self.config.collection_name
        text_fields = text_fields or ['chunk_text', 'document_title', 'section_title']
        
        try:
            # Build text search filter
            text_conditions = []
            for field in text_fields:
                condition = FieldCondition(
                    key=field,
                    match=MatchValue(value=query_text)
                )
                text_conditions.append(condition)
            
            # Create OR filter for text fields
            text_filter = Filter(
                should=text_conditions
            )
            
            # Perform scroll search (no vector needed for text search)
            search_results = self.client_wrapper.execute_with_retry(
                self.client.scroll,
                collection_name=collection_name,
                scroll_filter=text_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            formatted_results = self._format_scroll_results(search_results)
            
            return {
                'success': True,
                'results': formatted_results,
                'total_found': len(formatted_results),
                'query_info': {
                    'query_text': query_text,
                    'fields_searched': text_fields,
                    'limit': limit
                }
            }
            
        except Exception as e:
            self.logger.error(f"Text search failed: {e}")
            return {
                'success': False,
                'results': [],
                'error': str(e)
            }
    
    def _build_search_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """Build QDrant filter from filter dictionary"""
        conditions = []
        
        for field, value in filters.items():
            if isinstance(value, str):
                condition = FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            elif isinstance(value, list):
                condition = FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            elif isinstance(value, dict) and 'range' in value:
                # Range filter
                range_filter = value['range']
                condition = FieldCondition(
                    key=field,
                    range=Range(
                        gte=range_filter.get('gte'),
                        lte=range_filter.get('lte')
                    )
                )
            else:
                continue  # Skip unsupported filter types
            
            conditions.append(condition)
        
        if conditions:
            return Filter(must=conditions)
        
        return None
    
    def _format_search_results(self, search_results) -> List[Dict[str, Any]]:
        """Format search results for response"""
        formatted_results = []
        
        for result in search_results:
            formatted_result = {
                'id': result.id,
                'score': result.score,
                'payload': result.payload
            }
            
            # Extract key information for easier access
            payload = result.payload
            formatted_result['document_title'] = payload.get('document_title', '')
            formatted_result['section_title'] = payload.get('section_title', '')
            formatted_result['chunk_text'] = payload.get('chunk_text', '')[:200] + '...' if len(payload.get('chunk_text', '')) > 200 else payload.get('chunk_text', '')
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _format_scroll_results(self, scroll_results) -> List[Dict[str, Any]]:
        """Format scroll results for response"""
        points = scroll_results[0]  # scroll returns (points, next_page_offset)
        formatted_results = []
        
        for point in points:
            formatted_result = {
                'id': point.id,
                'payload': point.payload
            }
            
            # Extract key information
            payload = point.payload
            formatted_result['document_title'] = payload.get('document_title', '')
            formatted_result['section_title'] = payload.get('section_title', '')
            formatted_result['chunk_text'] = payload.get('chunk_text', '')[:200] + '...' if len(payload.get('chunk_text', '')) > 200 else payload.get('chunk_text', '')
            
            formatted_results.append(formatted_result)
        
        return formatted_results

class QdrantOrchestrator:
    """Main orchestrator for QDrant integration pipeline"""
    
    def __init__(self, config: QdrantConfig = None):
        self.config = config or QdrantConfig()
        self.client_wrapper = QdrantClientWrapper(self.config)
        self.collection_manager = CollectionManager(self.client_wrapper)
        self.data_uploader = DataUploader(self.client_wrapper)
        self.search_interface = QdrantSearchInterface(self.client_wrapper)
        self.logger = logging.getLogger(__name__)
    
    def process_embeddings_to_qdrant(self, embedding_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Complete pipeline: ensure collection exists and upload embeddings"""
        try:
            self.logger.info("Starting QDrant integration pipeline")
            
            # Step 1: Ensure collection exists
            collection_result = self.collection_manager.ensure_collection_exists()
            if not collection_result['success']:
                return {
                    'success': False,
                    'error': f"Collection management failed: {collection_result.get('error', 'Unknown error')}"
                }
            
            # Step 2: Upload embeddings
            upload_result = self.data_uploader.upload_embeddings_batch(embedding_results)
            if not upload_result['success']:
                return {
                    'success': False,
                    'error': f"Upload failed: {upload_result.get('error', 'Unknown error')}"
                }
            
            # Step 3: Get final statistics
            final_stats = self.collection_manager.get_collection_statistics()
            
            return {
                'success': True,
                'collection_management': collection_result,
                'upload_results': upload_result,
                'final_collection_stats': final_stats,
                'pipeline_summary': {
                    'total_embeddings_processed': len(embedding_results),
                    'successful_uploads': upload_result.get('successful_uploads', 0),
                    'success_rate': upload_result.get('success_rate', 0.0),
                    'collection_name': self.config.collection_name
                }
            }
            
        except Exception as e:
            self.logger.error(f"QDrant integration pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
```

## VALIDATION LOOP

### Unit Testing
```python
import pytest
from unittest.mock import Mock, patch
from src.qdrant_integration.qdrant_client_wrapper import QdrantClientWrapper, QdrantConfig

class TestQdrantIntegration:
    def test_client_initialization(self):
        """Test QDrant client initialization"""
        with patch('qdrant_client.QdrantClient') as mock_client:
            mock_client.return_value.get_collections.return_value.collections = []
            
            config = QdrantConfig()
            wrapper = QdrantClientWrapper(config)
            
            assert wrapper.client is not None
            mock_client.assert_called_once()
    
    def test_collection_creation(self):
        """Test collection creation workflow"""
        with patch('qdrant_client.QdrantClient') as mock_client:
            # Mock collection doesn't exist initially
            mock_client.return_value.get_collections.return_value.collections = []
            
            wrapper = QdrantClientWrapper()
            manager = CollectionManager(wrapper)
            
            result = manager.ensure_collection_exists()
            assert result['success'] == True
            assert result['action'] in ['created_new', 'found_existing']
    
    def test_stable_id_generation(self):
        """Test stable ID generation"""
        from src.qdrant_integration.id_generator import StableIDGenerator
        
        generator = StableIDGenerator()
        
        chunk = {
            'text': 'Test chunk content',
            'document_metadata': {'title': 'Test Document'},
            'lineage_metadata': {
                'source_lineage': {
                    'repository': {
                        'repository_path': 'test/repo',
                        'file_path': 'test/file.pdf'
                    }
                }
            }
        }
        
        id1 = generator.generate_chunk_id(chunk)
        id2 = generator.generate_chunk_id(chunk)  # Same chunk
        
        assert id1 == id2  # Should be stable
        assert id1.startswith('nic_')
    
    def test_payload_preparation(self):
        """Test payload preparation for QDrant"""
        from src.qdrant_integration.data_uploader import DataUploader
        
        mock_wrapper = Mock()
        uploader = DataUploader(mock_wrapper)
        
        chunk = {
            'text': 'Test chunk text',
            'document_metadata': {
                'title': 'Test Document',
                'tags': ['test', 'document']
            },
            'nic_metadata': {
                'chunk_position': {'token_count': 100}
            }
        }
        
        payload = uploader._prepare_payload(chunk)
        
        assert payload['document_title'] == 'Test Document'
        assert payload['chunk_text'] == 'Test chunk text'
        assert payload['token_count'] == 100
        assert 'test' in payload['document_tags']
```

### Integration Testing
- End-to-end pipeline from embeddings to QDrant storage
- Search functionality testing with real queries
- Collection management and data persistence validation

### Performance Testing
- Batch upload performance with 1000+ embeddings
- Search response time optimization
- Concurrent access and data integrity testing

## ADDITIONAL NOTES

### Security Considerations
- API key security and rotation procedures
- Payload sanitization to prevent injection attacks
- Access control and authentication for search operations
- Audit logging for database operations

### Performance Optimization
- Batch size optimization for uploads
- Connection pooling and retry strategies
- Search result caching for common queries
- Index optimization for frequently filtered fields

### Maintenance Requirements
- Regular collection statistics monitoring
- Data backup and recovery procedures
- QDrant version updates and compatibility testing
- Query performance analysis and optimization