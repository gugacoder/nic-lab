# Embedding Generation - PRP

## ROLE
**ML Engineering Specialist with Transformer Model Optimization expertise**

Specialized in transformer model deployment, CPU-optimized inference, and embedding generation pipelines. Responsible for implementing high-performance embedding generation using BAAI/bge-m3 model with optimal memory usage, batch processing, and quality validation for production vector search applications.

## OBJECTIVE
**High-Performance CPU-Based Embedding Generation Module**

Deliver a production-ready Python module that:
- Generates 1024-dimensional embeddings using BAAI/bge-m3 model on CPU
- Implements efficient batch processing with memory optimization
- Provides deterministic, reproducible embedding generation
- Handles model loading, caching, and warm-up for optimal performance
- Validates embedding quality and consistency across processing sessions
- Supports concurrent processing while managing resource consumption
- Implements comprehensive error handling and fallback mechanisms

## MOTIVATION
**Optimized Vector Generation for Semantic Search**

High-quality embeddings are the foundation of effective semantic search and retrieval systems. By implementing CPU-optimized BAAI/bge-m3 inference with intelligent batch processing and quality validation, this module ensures consistent, performant vector generation that scales with document processing demands while maintaining embedding quality and system stability.

## CONTEXT
**BAAI/bge-m3 CPU Inference Architecture**

- **Model**: BAAI/bge-m3 (multilingual, 1024 dimensions)
- **Deployment**: CPU-based inference (no GPU dependency)
- **Input Source**: Text chunks from chunking module
- **Output Format**: 1024-dimensional float32 vectors
- **Performance Goals**: Process 1000+ chunks efficiently with <4GB memory usage
- **Integration Pattern**: Modular Python module with batch processing capabilities
- **Quality Requirements**: Deterministic outputs, embedding validation, error resilience

## IMPLEMENTATION BLUEPRINT
**Comprehensive Embedding Generation Module**

### Architecture Overview
```python
# Module Structure: modules/embedding_generation.py
class EmbeddingGenerator:
    """CPU-optimized BAAI/bge-m3 embedding generation"""
    
    def __init__(self, config: EmbeddingConfig)
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[ChunkEmbedding]
    def generate_single_embedding(self, text: str) -> np.ndarray
    def validate_embedding_quality(self, embeddings: List[ChunkEmbedding]) -> QualityReport
    def batch_process(self, texts: List[str], batch_size: int) -> List[np.ndarray]
    def warm_up_model(self) -> None
```

### Code Structure
**File Organization**: `modules/embedding_generation.py`
```python
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib
import pickle
from pathlib import Path
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 32
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    device: str = "cpu"
    cache_model: bool = True
    cache_embeddings: bool = False  # Optional embedding caching
    max_memory_gb: float = 4.0
    num_threads: int = 4
    deterministic: bool = True
    warmup_iterations: int = 3

@dataclass
class EmbeddingMetadata:
    """Metadata for individual embeddings"""
    chunk_id: str
    embedding_hash: str
    model_version: str
    generation_timestamp: datetime
    processing_time_ms: float
    sequence_length: int
    was_truncated: bool
    quality_score: float

@dataclass
class ChunkEmbedding:
    """Embedding with comprehensive metadata"""
    chunk_id: str
    text_content: str
    embedding_vector: np.ndarray
    metadata: EmbeddingMetadata
    chunk_metadata: Dict[str, Any]  # From original chunk

@dataclass
class QualityReport:
    """Embedding quality assessment report"""
    total_embeddings: int
    avg_quality_score: float
    dimension_consistency: bool
    normalization_status: bool
    outlier_count: int
    processing_issues: List[str]
    performance_metrics: Dict[str, float]

class EmbeddingGenerator:
    """Production-ready embedding generation with BAAI/bge-m3"""
    
    MODEL_DIMENSION = 1024
    DEFAULT_QUALITY_THRESHOLD = 0.8
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_lock = threading.Lock()
        self._model_loaded = False
        
        # Performance monitoring
        self.processing_stats = {
            'total_embeddings': 0,
            'total_processing_time': 0.0,
            'average_batch_size': 0.0,
            'memory_usage_peak': 0.0
        }
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize and configure BAAI/bge-m3 model"""
        try:
            with self.model_lock:
                self.logger.info(f"Loading BAAI/bge-m3 model: {self.config.model_name}")
                
                # Configure device and threading
                if self.config.deterministic:
                    torch.manual_seed(42)
                    np.random.seed(42)
                
                torch.set_num_threads(self.config.num_threads)
                
                # Load model with CPU optimization
                self.model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device
                )
                
                # Configure model settings
                self.model.max_seq_length = self.config.max_sequence_length
                
                # Verify model dimensions
                test_embedding = self.model.encode("Test sentence", convert_to_numpy=True)
                if test_embedding.shape[0] != self.MODEL_DIMENSION:
                    raise ValueError(f"Model dimension mismatch: expected {self.MODEL_DIMENSION}, got {test_embedding.shape[0]}")
                
                self._model_loaded = True
                self.logger.info(f"Model loaded successfully. Dimension: {test_embedding.shape[0]}")
                
                # Warm up model
                if self.config.warmup_iterations > 0:
                    self.warm_up_model()
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def warm_up_model(self):
        """Warm up model with dummy inference to optimize performance"""
        if not self._model_loaded:
            return
            
        self.logger.info("Warming up embedding model...")
        warmup_texts = [
            "This is a warmup sentence for model optimization.",
            "Model warmup helps reduce first inference latency.",
            "Warmup process prepares the model for optimal performance."
        ]
        
        try:
            for i in range(self.config.warmup_iterations):
                _ = self.model.encode(
                    warmup_texts,
                    batch_size=min(len(warmup_texts), self.config.batch_size),
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings
                )
                
            self.logger.info("Model warmup completed successfully")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    def generate_embeddings(self, chunks: List[Any]) -> List[ChunkEmbedding]:
        """Generate embeddings for document chunks with batch processing"""
        
        if not chunks:
            return []
        
        if not self._model_loaded:
            raise RuntimeError("Embedding model not loaded")
        
        start_time = datetime.utcnow()
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        try:
            # Extract text content from chunks
            texts = []
            chunk_metadata_list = []
            
            for chunk in chunks:
                if hasattr(chunk, 'text_chunk'):
                    # DocumentChunk format
                    text = chunk.text_chunk.content
                    metadata = {
                        'chunk_id': chunk.text_chunk.metadata.chunk_id,
                        'token_count': chunk.text_chunk.metadata.token_count,
                        'chunk_type': chunk.text_chunk.metadata.chunk_type.value,
                        'document_metadata': chunk.document_metadata
                    }
                else:
                    # Direct text chunk format
                    text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    metadata = {'chunk_id': hashlib.md5(text.encode()).hexdigest()[:16]}
                
                texts.append(text)
                chunk_metadata_list.append(metadata)
            
            # Generate embeddings in batches
            all_embeddings = self.batch_process(texts, self.config.batch_size)
            
            # Create ChunkEmbedding objects
            chunk_embeddings = []
            for i, (text, embedding, chunk_meta) in enumerate(zip(texts, all_embeddings, chunk_metadata_list)):
                
                # Calculate processing metrics
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000 / len(chunks)
                
                # Generate embedding metadata
                embedding_hash = hashlib.sha256(embedding.tobytes()).hexdigest()[:16]
                quality_score = self._calculate_quality_score(embedding, text)
                
                embedding_metadata = EmbeddingMetadata(
                    chunk_id=chunk_meta['chunk_id'],
                    embedding_hash=embedding_hash,
                    model_version=self.config.model_name,
                    generation_timestamp=datetime.utcnow(),
                    processing_time_ms=processing_time,
                    sequence_length=len(text),
                    was_truncated=len(text) > self.config.max_sequence_length * 4,  # Rough estimate
                    quality_score=quality_score
                )
                
                chunk_embedding = ChunkEmbedding(
                    chunk_id=chunk_meta['chunk_id'],
                    text_content=text,
                    embedding_vector=embedding,
                    metadata=embedding_metadata,
                    chunk_metadata=chunk_meta
                )
                
                chunk_embeddings.append(chunk_embedding)
            
            # Update processing statistics
            total_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_processing_stats(len(chunks), total_time)
            
            self.logger.info(f"Generated {len(chunk_embeddings)} embeddings in {total_time:.2f}s")
            return chunk_embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text input"""
        
        if not self._model_loaded:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode(
                [text],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False
            )[0]
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Single embedding generation failed: {e}")
            raise
    
    def batch_process(self, texts: List[str], batch_size: int) -> List[np.ndarray]:
        """Process texts in batches with memory management"""
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # Monitor memory usage
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
                
                if memory_before > self.config.max_memory_gb * 0.8:
                    self.logger.warning(f"High memory usage detected: {memory_before:.2f}GB")
                    gc.collect()  # Force garbage collection
                
                self.logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
                
                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False
                )
                
                # Validate batch results
                if len(batch_embeddings) != len(batch_texts):
                    raise ValueError(f"Batch processing mismatch: {len(batch_embeddings)} embeddings for {len(batch_texts)} texts")
                
                for embedding in batch_embeddings:
                    if embedding.shape[0] != self.MODEL_DIMENSION:
                        raise ValueError(f"Embedding dimension error: {embedding.shape[0]} != {self.MODEL_DIMENSION}")
                
                all_embeddings.extend(batch_embeddings)
                
                # Monitor memory after processing
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
                self.processing_stats['memory_usage_peak'] = max(
                    self.processing_stats['memory_usage_peak'], memory_after
                )
                
            except Exception as e:
                self.logger.error(f"Batch {batch_num} processing failed: {e}")
                # Create zero embeddings for failed batch to maintain consistency
                zero_embeddings = [np.zeros(self.MODEL_DIMENSION) for _ in batch_texts]
                all_embeddings.extend(zero_embeddings)
        
        return all_embeddings
    
    def validate_embedding_quality(self, embeddings: List[ChunkEmbedding]) -> QualityReport:
        """Comprehensive embedding quality validation"""
        
        if not embeddings:
            return QualityReport(
                total_embeddings=0,
                avg_quality_score=0.0,
                dimension_consistency=True,
                normalization_status=True,
                outlier_count=0,
                processing_issues=[],
                performance_metrics={}
            )
        
        quality_scores = []
        dimension_issues = 0
        normalization_issues = 0
        outliers = 0
        processing_issues = []
        
        # Collect all embedding vectors
        vectors = np.array([emb.embedding_vector for emb in embeddings])
        
        for i, embedding in enumerate(embeddings):
            vector = embedding.embedding_vector
            
            # Check dimensions
            if vector.shape[0] != self.MODEL_DIMENSION:
                dimension_issues += 1
                processing_issues.append(f"Embedding {i}: dimension {vector.shape[0]} != {self.MODEL_DIMENSION}")
            
            # Check normalization if enabled
            if self.config.normalize_embeddings:
                norm = np.linalg.norm(vector)
                if abs(norm - 1.0) > 0.01:  # Allow small tolerance
                    normalization_issues += 1
                    processing_issues.append(f"Embedding {i}: norm {norm:.4f} != 1.0")
            
            # Check for outliers (vectors with unusual characteristics)
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                outliers += 1
                processing_issues.append(f"Embedding {i}: contains NaN or Inf values")
            
            quality_scores.append(embedding.metadata.quality_score)
        
        # Calculate performance metrics
        processing_times = [emb.metadata.processing_time_ms for emb in embeddings]
        
        performance_metrics = {
            'avg_processing_time_ms': np.mean(processing_times),
            'median_processing_time_ms': np.median(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'total_processing_time_s': np.sum(processing_times) / 1000
        }
        
        return QualityReport(
            total_embeddings=len(embeddings),
            avg_quality_score=np.mean(quality_scores) if quality_scores else 0.0,
            dimension_consistency=(dimension_issues == 0),
            normalization_status=(normalization_issues == 0),
            outlier_count=outliers,
            processing_issues=processing_issues,
            performance_metrics=performance_metrics
        )
    
    def _calculate_quality_score(self, embedding: np.ndarray, text: str) -> float:
        """Calculate quality score for individual embedding"""
        
        score = 1.0
        
        # Check for zero vectors (failed embeddings)
        if np.allclose(embedding, 0.0):
            return 0.0
        
        # Check vector magnitude
        magnitude = np.linalg.norm(embedding)
        if magnitude < 0.1:  # Very small magnitude indicates poor embedding
            score -= 0.5
        
        # Check for unusual patterns
        if np.std(embedding) < 0.01:  # Very low variance indicates poor embedding
            score -= 0.3
        
        # Text length correlation (very short or very long text may have lower quality)
        text_length = len(text)
        if text_length < 10:
            score -= 0.2
        elif text_length > self.config.max_sequence_length * 4:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _update_processing_stats(self, batch_size: int, processing_time: float):
        """Update processing statistics"""
        
        self.processing_stats['total_embeddings'] += batch_size
        self.processing_stats['total_processing_time'] += processing_time
        
        # Update average batch size (running average)
        current_avg = self.processing_stats['average_batch_size']
        new_avg = (current_avg + batch_size) / 2 if current_avg > 0 else batch_size
        self.processing_stats['average_batch_size'] = new_avg
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        
        stats = self.processing_stats.copy()
        
        if stats['total_embeddings'] > 0:
            stats['avg_time_per_embedding'] = stats['total_processing_time'] / stats['total_embeddings']
        else:
            stats['avg_time_per_embedding'] = 0.0
        
        return stats
    
    def clear_cache(self):
        """Clear any cached data and free memory"""
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Cache cleared and memory freed")

# Context manager for embedding generation
class EmbeddingGeneratorContext:
    """Context manager for embedding generation with automatic cleanup"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.generator = None
    
    def __enter__(self) -> EmbeddingGenerator:
        self.generator = EmbeddingGenerator(self.config)
        return self.generator
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.generator:
            self.generator.clear_cache()

def create_embedding_generator(config_dict: Dict[str, Any]) -> EmbeddingGenerator:
    """Factory function for embedding generator creation"""
    config = EmbeddingConfig(**config_dict)
    return EmbeddingGenerator(config)
```

### Error Handling
**Comprehensive Embedding Generation Error Management**
```python
class EmbeddingGenerationError(Exception):
    """Base exception for embedding generation errors"""
    pass

class ModelLoadingError(EmbeddingGenerationError):
    """Model loading and initialization errors"""
    pass

class InferenceError(EmbeddingGenerationError):
    """Model inference and processing errors"""
    pass

class QualityValidationError(EmbeddingGenerationError):
    """Embedding quality validation errors"""
    pass

# Error recovery patterns
def safe_embedding_generation(generator: EmbeddingGenerator, texts: List[str]) -> List[np.ndarray]:
    """Safe embedding generation with fallback strategies"""
    try:
        return generator.batch_process(texts, generator.config.batch_size)
    except torch.cuda.OutOfMemoryError:
        # Reduce batch size and retry
        reduced_batch_size = max(1, generator.config.batch_size // 2)
        logging.warning(f"Reducing batch size to {reduced_batch_size} due to memory constraints")
        return generator.batch_process(texts, reduced_batch_size)
    except Exception as e:
        logging.error(f"Embedding generation failed, using zero vectors: {e}")
        return [np.zeros(EmbeddingGenerator.MODEL_DIMENSION) for _ in texts]
```

## VALIDATION LOOP
**Comprehensive Testing Strategy**

### Unit Testing
```python
# tests/test_embedding_generation.py
import pytest
import numpy as np
from modules.embedding_generation import EmbeddingGenerator, EmbeddingConfig

class TestEmbeddingGenerator:
    
    @pytest.fixture
    def default_config(self):
        return EmbeddingConfig(
            model_name="BAAI/bge-m3",
            batch_size=4,
            deterministic=True
        )
    
    @pytest.fixture
    def sample_texts(self):
        return [
            "This is a sample sentence for testing.",
            "Another sentence with different content.",
            "A third sentence to test batch processing.",
            "Final sentence for the test batch."
        ]
    
    def test_model_initialization(self, default_config):
        """Test successful model initialization"""
        generator = EmbeddingGenerator(default_config)
        
        assert generator._model_loaded is True
        assert generator.model is not None
        assert generator.MODEL_DIMENSION == 1024
    
    def test_single_embedding_generation(self, default_config):
        """Test single text embedding generation"""
        generator = EmbeddingGenerator(default_config)
        
        text = "Test sentence for embedding generation."
        embedding = generator.generate_single_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
        assert not np.allclose(embedding, 0.0)  # Should not be zero vector
    
    def test_batch_processing(self, default_config, sample_texts):
        """Test batch embedding generation"""
        generator = EmbeddingGenerator(default_config)
        
        embeddings = generator.batch_process(sample_texts, batch_size=2)
        
        assert len(embeddings) == len(sample_texts)
        assert all(emb.shape == (1024,) for emb in embeddings)
        assert all(not np.allclose(emb, 0.0) for emb in embeddings)
    
    def test_deterministic_output(self, default_config):
        """Test deterministic embedding generation"""
        generator1 = EmbeddingGenerator(default_config)
        generator2 = EmbeddingGenerator(default_config)
        
        text = "Deterministic test sentence."
        
        embedding1 = generator1.generate_single_embedding(text)
        embedding2 = generator2.generate_single_embedding(text)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)
    
    def test_normalization(self, default_config):
        """Test embedding normalization"""
        config = default_config
        config.normalize_embeddings = True
        
        generator = EmbeddingGenerator(config)
        
        text = "Test normalization sentence."
        embedding = generator.generate_single_embedding(text)
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be approximately unit norm
    
    def test_quality_validation(self, default_config, sample_texts):
        """Test embedding quality validation"""
        generator = EmbeddingGenerator(default_config)
        
        # Create mock chunk embeddings
        embeddings = generator.batch_process(sample_texts, batch_size=4)
        
        from modules.embedding_generation import ChunkEmbedding, EmbeddingMetadata
        chunk_embeddings = []
        
        for i, (text, emb) in enumerate(zip(sample_texts, embeddings)):
            metadata = EmbeddingMetadata(
                chunk_id=f"test_chunk_{i}",
                embedding_hash="test_hash",
                model_version="BAAI/bge-m3",
                generation_timestamp=datetime.utcnow(),
                processing_time_ms=10.0,
                sequence_length=len(text),
                was_truncated=False,
                quality_score=0.9
            )
            
            chunk_embedding = ChunkEmbedding(
                chunk_id=f"test_chunk_{i}",
                text_content=text,
                embedding_vector=emb,
                metadata=metadata,
                chunk_metadata={}
            )
            chunk_embeddings.append(chunk_embedding)
        
        quality_report = generator.validate_embedding_quality(chunk_embeddings)
        
        assert quality_report.total_embeddings == len(sample_texts)
        assert quality_report.dimension_consistency is True
        assert quality_report.avg_quality_score > 0.0
```

### Integration Testing
```python
# tests/integration/test_embedding_pipeline.py
@pytest.mark.integration
def test_full_embedding_pipeline():
    """Integration test with text chunking module"""
    from modules.text_chunking import TextChunker, ChunkingConfig
    
    # Create sample document chunks
    chunking_config = ChunkingConfig(target_chunk_size=100)
    chunker = TextChunker(chunking_config)
    
    sample_text = "This is a sample document. " * 100
    chunks = chunker.chunk_text(sample_text, ChunkContext())
    
    # Generate embeddings
    embedding_config = EmbeddingConfig(batch_size=4)
    generator = EmbeddingGenerator(embedding_config)
    
    embeddings = generator.generate_embeddings(chunks)
    
    assert len(embeddings) == len(chunks)
    assert all(emb.embedding_vector.shape == (1024,) for emb in embeddings)
    
    # Validate quality
    quality_report = generator.validate_embedding_quality(embeddings)
    assert quality_report.dimension_consistency is True
```

### Performance Testing
- **Inference Speed**: Target <100ms per embedding on CPU
- **Batch Processing**: Verify optimal batch sizes for different text lengths
- **Memory Usage**: Monitor memory consumption and prevent leaks
- **Model Loading**: Test model initialization time and caching effectiveness

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
- **Model Integrity**: Verify model checksum and authenticity before loading
- **Input Sanitization**: Validate text inputs to prevent injection attacks
- **Memory Protection**: Implement safeguards against memory exhaustion attacks
- **Resource Limits**: Enforce processing limits to prevent resource abuse

### Performance Optimization
- **CPU Optimization**: Use optimized BLAS libraries and threading configuration
- **Memory Management**: Implement efficient memory usage patterns and garbage collection
- **Batch Size Tuning**: Dynamically adjust batch sizes based on available memory
- **Model Caching**: Cache model instances for reuse across processing sessions

### Maintenance Requirements
- **Model Updates**: Monitor BAAI/bge-m3 model updates and compatibility
- **Performance Monitoring**: Track inference speed and quality metrics over time
- **Resource Usage**: Monitor CPU and memory usage patterns for optimization
- **Quality Assurance**: Implement automated quality checks for embedding consistency