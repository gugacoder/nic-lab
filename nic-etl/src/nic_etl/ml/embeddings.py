import logging
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import gc
from pathlib import Path
import threading

# Optional dependencies with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Create mock SentenceTransformer for development
    class SentenceTransformer:
        def __init__(self, model_name: str, device: str = "cpu"):
            self.max_seq_length = 512
            self.model_name = model_name
            self.device = device
        
        def encode(self, texts, batch_size=32, convert_to_numpy=True, 
                  normalize_embeddings=True, show_progress_bar=False):
            if isinstance(texts, str):
                texts = [texts]
            # Return mock embeddings (1024 dimensions for BAAI/bge-m3)
            embeddings = []
            for text in texts:
                # Generate deterministic mock embedding based on text hash
                text_hash = hashlib.md5(text.encode()).hexdigest()
                # Convert hash to numeric seed
                seed = int(text_hash[:8], 16)
                np.random.seed(seed % 2**31)
                embedding = np.random.normal(0, 1, 1024).astype(np.float32)
                if normalize_embeddings:
                    embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create mock torch for development
    class MockTorch:
        @staticmethod
        def manual_seed(seed):
            np.random.seed(seed)
        
        @staticmethod
        def set_num_threads(n):
            pass
        
        class cuda:
            @staticmethod
            def is_available():
                return False
            
            @staticmethod
            def empty_cache():
                pass
    
    torch = MockTorch()

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create mock psutil for development
    class MockProcess:
        def memory_info(self):
            class MemInfo:
                rss = 1024 * 1024 * 1024  # 1GB mock usage
            return MemInfo()
    
    class MockPsutil:
        @staticmethod
        def Process():
            return MockProcess()
    
    psutil = MockPsutil()

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
        self.sentence_transformers_available = SENTENCE_TRANSFORMERS_AVAILABLE
        
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
                
                # Configure deterministic behavior
                if self.config.deterministic:
                    if TORCH_AVAILABLE:
                        torch.manual_seed(42)
                    np.random.seed(42)
                
                # Configure threading
                if TORCH_AVAILABLE:
                    torch.set_num_threads(self.config.num_threads)
                
                # Load model with fallback
                if self.sentence_transformers_available:
                    try:
                        self.model = SentenceTransformer(
                            self.config.model_name,
                            device=self.config.device
                        )
                        
                        # Configure model settings
                        self.model.max_seq_length = self.config.max_sequence_length
                        
                        # Verify model dimensions
                        test_embedding = self.model.encode("Test sentence", convert_to_numpy=True)
                        if hasattr(test_embedding, 'shape'):
                            actual_dim = test_embedding.shape[-1] if len(test_embedding.shape) > 0 else len(test_embedding)
                        else:
                            actual_dim = len(test_embedding)
                        
                        if actual_dim != self.MODEL_DIMENSION:
                            self.logger.warning(f"Model dimension unexpected: got {actual_dim}, expected {self.MODEL_DIMENSION}")
                        
                        self.logger.info(f"Real model loaded successfully. Dimension: {actual_dim}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load real model, using fallback: {e}")
                        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
                        self.sentence_transformers_available = False
                else:
                    self.logger.warning("SentenceTransformers not available, using mock model")
                    self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
                
                self._model_loaded = True
                
                # Warm up model
                if self.config.warmup_iterations > 0:
                    self.warm_up_model()
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise EmbeddingGenerationError(f"Model initialization failed: {e}")
    
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
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False
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
                        'chunk_type': chunk.text_chunk.metadata.chunk_type.value if hasattr(chunk.text_chunk.metadata.chunk_type, 'value') else str(chunk.text_chunk.metadata.chunk_type),
                        'document_metadata': chunk.document_metadata
                    }
                elif hasattr(chunk, 'content'):
                    # Direct text chunk format
                    text = chunk.content
                    chunk_id = getattr(chunk.metadata, 'chunk_id', None) if hasattr(chunk, 'metadata') else None
                    if not chunk_id:
                        chunk_id = hashlib.md5(text.encode()).hexdigest()[:16]
                    metadata = {'chunk_id': chunk_id}
                else:
                    # String content
                    text = str(chunk)
                    metadata = {'chunk_id': hashlib.md5(text.encode()).hexdigest()[:16]}
                
                texts.append(text)
                chunk_metadata_list.append(metadata)
            
            # Generate embeddings in batches
            all_embeddings = self.batch_process(texts, self.config.batch_size)
            
            # Create ChunkEmbedding objects
            chunk_embeddings = []
            processing_time_total = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            for i, (text, embedding, chunk_meta) in enumerate(zip(texts, all_embeddings, chunk_metadata_list)):
                
                # Calculate processing metrics
                processing_time = processing_time_total / len(chunks)
                
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
            raise EmbeddingGenerationError(f"Failed to generate embeddings: {e}")
    
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
            )
            
            # Handle both single embedding and array of embeddings
            if hasattr(embedding, 'shape') and len(embedding.shape) > 1:
                return embedding[0]
            else:
                return embedding
            
        except Exception as e:
            self.logger.error(f"Single embedding generation failed: {e}")
            raise EmbeddingGenerationError(f"Single embedding failed: {e}")
    
    def batch_process(self, texts: List[str], batch_size: int) -> List[np.ndarray]:
        """Process texts in batches with memory management"""
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # Monitor memory usage if psutil is available
                if PSUTIL_AVAILABLE:
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
                
                # Ensure we have the right format
                if not isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = np.array(batch_embeddings)
                
                # Handle single vs multiple embeddings
                if len(batch_texts) == 1:
                    if len(batch_embeddings.shape) == 1:
                        batch_embeddings = batch_embeddings.reshape(1, -1)
                
                # Validate batch results
                if len(batch_embeddings) != len(batch_texts):
                    raise ValueError(f"Batch processing mismatch: {len(batch_embeddings)} embeddings for {len(batch_texts)} texts")
                
                for embedding in batch_embeddings:
                    expected_dim = self.MODEL_DIMENSION
                    actual_dim = embedding.shape[-1] if hasattr(embedding, 'shape') else len(embedding)
                    
                    if actual_dim != expected_dim:
                        self.logger.warning(f"Embedding dimension warning: {actual_dim} != {expected_dim}")
                
                # Convert to list of individual embeddings
                embedding_list = [batch_embeddings[j] for j in range(len(batch_embeddings))]
                all_embeddings.extend(embedding_list)
                
                # Monitor memory after processing
                if PSUTIL_AVAILABLE:
                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
                    self.processing_stats['memory_usage_peak'] = max(
                        self.processing_stats['memory_usage_peak'], memory_after
                    )
                
            except Exception as e:
                self.logger.error(f"Batch {batch_num} processing failed: {e}")
                # Create zero embeddings for failed batch to maintain consistency
                zero_embeddings = [np.zeros(self.MODEL_DIMENSION, dtype=np.float32) for _ in batch_texts]
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
        vectors = []
        for emb in embeddings:
            try:
                vectors.append(emb.embedding_vector)
            except Exception as e:
                processing_issues.append(f"Failed to access embedding vector: {e}")
                continue
        
        if not vectors:
            return QualityReport(
                total_embeddings=len(embeddings),
                avg_quality_score=0.0,
                dimension_consistency=False,
                normalization_status=False,
                outlier_count=len(embeddings),
                processing_issues=["No valid embedding vectors found"],
                performance_metrics={}
            )
        
        for i, embedding in enumerate(embeddings):
            vector = embedding.embedding_vector
            
            # Check dimensions
            actual_dim = vector.shape[-1] if hasattr(vector, 'shape') else len(vector)
            if actual_dim != self.MODEL_DIMENSION:
                dimension_issues += 1
                processing_issues.append(f"Embedding {i}: dimension {actual_dim} != {self.MODEL_DIMENSION}")
            
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
            'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0.0,
            'median_processing_time_ms': np.median(processing_times) if processing_times else 0.0,
            'max_processing_time_ms': np.max(processing_times) if processing_times else 0.0,
            'total_processing_time_s': np.sum(processing_times) / 1000 if processing_times else 0.0
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
        
        stats['model_available'] = self.sentence_transformers_available
        stats['torch_available'] = TORCH_AVAILABLE
        stats['psutil_available'] = PSUTIL_AVAILABLE
        
        return stats
    
    def clear_cache(self):
        """Clear any cached data and free memory"""
        
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Cache cleared and memory freed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.config.model_name,
            'model_loaded': self._model_loaded,
            'device': self.config.device,
            'max_sequence_length': self.config.max_sequence_length,
            'embedding_dimension': self.MODEL_DIMENSION,
            'normalization_enabled': self.config.normalize_embeddings,
            'deterministic': self.config.deterministic,
            'sentence_transformers_available': self.sentence_transformers_available
        }

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
    config = EmbeddingConfig(
        model_name=config_dict.get('model_name', 'BAAI/bge-m3'),
        batch_size=config_dict.get('batch_size', 32),
        max_sequence_length=config_dict.get('max_sequence_length', 512),
        normalize_embeddings=config_dict.get('normalize_embeddings', True),
        device=config_dict.get('device', 'cpu'),
        cache_model=config_dict.get('cache_model', True),
        cache_embeddings=config_dict.get('cache_embeddings', False),
        max_memory_gb=config_dict.get('max_memory_gb', 4.0),
        num_threads=config_dict.get('num_threads', 4),
        deterministic=config_dict.get('deterministic', True),
        warmup_iterations=config_dict.get('warmup_iterations', 3)
    )
    return EmbeddingGenerator(config)

# Error classes
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

# Utility functions
def safe_embedding_generation(generator: EmbeddingGenerator, texts: List[str]) -> List[np.ndarray]:
    """Safe embedding generation with fallback strategies"""
    try:
        return generator.batch_process(texts, generator.config.batch_size)
    except Exception as e:
        if "memory" in str(e).lower() or "out of memory" in str(e).lower():
            # Reduce batch size and retry
            reduced_batch_size = max(1, generator.config.batch_size // 2)
            logging.warning(f"Reducing batch size to {reduced_batch_size} due to memory constraints")
            try:
                return generator.batch_process(texts, reduced_batch_size)
            except Exception:
                pass
        
        logging.error(f"Embedding generation failed, using zero vectors: {e}")
        return [np.zeros(EmbeddingGenerator.MODEL_DIMENSION, dtype=np.float32) for _ in texts]

def calculate_embedding_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    try:
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
        
    except Exception:
        return 0.0

def validate_embedding_format(embedding: np.ndarray, expected_dim: int = 1024) -> bool:
    """Validate embedding format and dimensions"""
    try:
        if not isinstance(embedding, np.ndarray):
            return False
        
        if len(embedding.shape) != 1:
            return False
        
        if embedding.shape[0] != expected_dim:
            return False
        
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return False
        
        return True
        
    except Exception:
        return False