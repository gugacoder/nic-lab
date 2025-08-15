# Embedding Generation - PRP

## ROLE
**Machine Learning Engineer with Embedding Model Expertise**

Specialist in transformer-based embedding models, vector representations, and semantic similarity computation. Expert in implementing BAAI/bge-m3 model for multilingual embeddings, optimizing CPU-based inference, and managing batch processing for large-scale document collections. Proficient in handling 1024-dimensional dense vectors and ensuring embedding quality for retrieval tasks.

## OBJECTIVE
**Generate High-Quality Multilingual Embeddings**

Build an embedding generation module within Jupyter Notebook cells that:
* Implements BAAI/bge-m3 model for 1024-dimensional embeddings
* Runs efficiently on CPU with optimized batch processing
* Handles multilingual content (Portuguese and English)
* Generates deterministic embeddings for reproducibility
* Implements caching for processed chunks
* Monitors embedding quality and generation performance
* Provides batch and streaming generation modes

## MOTIVATION
**Semantic Understanding for Intelligent Retrieval**

Embeddings transform text into dense vector representations that capture semantic meaning, enabling powerful similarity-based search and retrieval. The BAAI/bge-m3 model provides state-of-the-art multilingual understanding crucial for the Portuguese and English documents in the NIC knowledge base. High-quality embeddings directly impact search relevance, user experience, and the overall effectiveness of the knowledge management system.

## CONTEXT
**CPU-Based Embedding Generation Environment**

Operating specifications:
* Model: BAAI/bge-m3 (multilingual, 1024 dimensions)
* Infrastructure: CPU-only deployment
* Input: Text chunks from chunking pipeline
* Batch size: Optimized for CPU memory constraints
* Languages: Portuguese and English support
* Output: 1024-dimensional float32 vectors
* Constraints: Jupyter Notebook implementation
* Performance: Balance between speed and resource usage

## IMPLEMENTATION BLUEPRINT
**Complete Embedding Generation Architecture**

### Architecture Overview
```
Cell 7: Embedding Generation
├── EmbeddingGenerator class
│   ├── Model initialization
│   ├── Batch processing
│   ├── CPU optimization
│   ├── Cache management
│   └── Quality monitoring
├── Vector normalization
├── Similarity computation
└── Performance tracking
```

### Code Structure
```python
# Cell 7: Embedding Generation Functions
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
import hashlib
from pathlib import Path
import pickle
import time
from dataclasses import dataclass
import json

@dataclass
class EmbeddingResult:
    """Embedding generation result"""
    chunk_id: str
    embedding: np.ndarray
    dimension: int
    model: str
    generation_time: float
    cached: bool
    metadata: Dict[str, Any]

class EmbeddingGenerator:
    def __init__(self,
                 model_name: str = "BAAI/bge-m3",
                 cache_dir: Path = Path("./cache/embeddings"),
                 batch_size: int = 32,
                 device: str = None):
        """Initialize embedding generator with BGE-M3 model"""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device (CPU by default)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize model
        print(f"Loading {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Optimize for CPU if needed
        if self.device == 'cpu':
            self._optimize_for_cpu()
        
        # Model properties
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'generation_time': 0.0
        }
    
    def _optimize_for_cpu(self):
        """Optimize model for CPU inference"""
        # Set number of threads for optimal CPU performance
        torch.set_num_threads(torch.get_num_threads())
        
        # Enable CPU optimizations
        if hasattr(torch, 'set_flush_denormal'):
            torch.set_flush_denormal(True)
    
    def generate_embeddings(self, 
                           chunks: List[TextChunk],
                           use_cache: bool = True) -> List[EmbeddingResult]:
        """Generate embeddings for text chunks"""
        results = []
        chunks_to_process = []
        
        # Check cache for existing embeddings
        for chunk in chunks:
            if use_cache:
                cached_embedding = self._load_from_cache(chunk.id)
                if cached_embedding is not None:
                    results.append(EmbeddingResult(
                        chunk_id=chunk.id,
                        embedding=cached_embedding,
                        dimension=self.embedding_dimension,
                        model=self.model_name,
                        generation_time=0.0,
                        cached=True,
                        metadata={'from_cache': True}
                    ))
                    self.stats['cache_hits'] += 1
                    continue
            
            chunks_to_process.append(chunk)
        
        # Process remaining chunks in batches
        if chunks_to_process:
            new_results = self._process_chunks_in_batches(chunks_to_process)
            
            # Cache new embeddings
            if use_cache:
                for result in new_results:
                    self._save_to_cache(result.chunk_id, result.embedding)
            
            results.extend(new_results)
        
        self.stats['total_processed'] += len(chunks)
        
        return results
    
    def _process_chunks_in_batches(self, chunks: List[TextChunk]) -> List[EmbeddingResult]:
        """Process chunks in optimized batches"""
        results = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            # Extract texts from chunks
            texts = [chunk.content for chunk in batch]
            
            # Generate embeddings
            start_time = time.time()
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,  # L2 normalization
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            
            generation_time = time.time() - start_time
            self.stats['generation_time'] += generation_time
            
            # Create results
            for chunk, embedding in zip(batch, embeddings):
                results.append(EmbeddingResult(
                    chunk_id=chunk.id,
                    embedding=embedding.astype(np.float32),
                    dimension=self.embedding_dimension,
                    model=self.model_name,
                    generation_time=generation_time / len(batch),
                    cached=False,
                    metadata={
                        'token_count': chunk.token_count,
                        'batch_size': len(batch)
                    }
                ))
        
        return results
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        with torch.no_grad():
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        
        return embedding.astype(np.float32)
    
    def compute_similarity(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        # Embeddings are already normalized, so dot product = cosine similarity
        return float(np.dot(embedding1, embedding2))
    
    def find_similar_chunks(self,
                           query_embedding: np.ndarray,
                           chunk_embeddings: List[EmbeddingResult],
                           top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar chunks to query"""
        similarities = []
        
        for result in chunk_embeddings:
            similarity = self.compute_similarity(query_embedding, result.embedding)
            similarities.append((result.chunk_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _generate_cache_key(self, chunk_id: str) -> str:
        """Generate cache key for chunk"""
        return f"{self.model_name}_{chunk_id}".replace('/', '_')
    
    def _save_to_cache(self, chunk_id: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_key = self._generate_cache_key(chunk_id)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        
        np.save(cache_path, embedding)
    
    def _load_from_cache(self, chunk_id: str) -> Optional[np.ndarray]:
        """Load embedding from cache"""
        cache_key = self._generate_cache_key(chunk_id)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        
        if cache_path.exists():
            return np.load(cache_path)
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        stats = self.stats.copy()
        
        # Calculate additional metrics
        if stats['total_processed'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_processed']
            stats['avg_generation_time'] = stats['generation_time'] / (
                stats['total_processed'] - stats['cache_hits']
            ) if stats['total_processed'] > stats['cache_hits'] else 0
        
        return stats

class EmbeddingQualityAnalyzer:
    """Analyze embedding quality and distribution"""
    
    @staticmethod
    def analyze_embeddings(embeddings: List[EmbeddingResult]) -> Dict[str, Any]:
        """Analyze embedding quality metrics"""
        if not embeddings:
            return {'error': 'No embeddings to analyze'}
        
        # Convert to numpy array
        embedding_matrix = np.array([e.embedding for e in embeddings])
        
        analysis = {
            'count': len(embeddings),
            'dimension': embeddings[0].dimension,
            'statistics': {},
            'quality_metrics': {}
        }
        
        # Basic statistics
        analysis['statistics'] = {
            'mean': float(np.mean(embedding_matrix)),
            'std': float(np.std(embedding_matrix)),
            'min': float(np.min(embedding_matrix)),
            'max': float(np.max(embedding_matrix))
        }
        
        # Check normalization (should be close to 1.0 for normalized embeddings)
        norms = np.linalg.norm(embedding_matrix, axis=1)
        analysis['quality_metrics']['norm_stats'] = {
            'mean': float(np.mean(norms)),
            'std': float(np.std(norms)),
            'properly_normalized': bool(np.allclose(norms, 1.0, atol=1e-5))
        }
        
        # Compute pairwise similarity distribution
        if len(embeddings) > 1 and len(embeddings) <= 1000:  # Limit for memory
            similarities = np.dot(embedding_matrix, embedding_matrix.T)
            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(similarities, k=1)
            similarity_values = upper_triangle[upper_triangle != 0]
            
            analysis['quality_metrics']['similarity_distribution'] = {
                'mean': float(np.mean(similarity_values)),
                'std': float(np.std(similarity_values)),
                'min': float(np.min(similarity_values)),
                'max': float(np.max(similarity_values)),
                'median': float(np.median(similarity_values))
            }
        
        # Check for duplicate embeddings
        unique_embeddings = len(np.unique(embedding_matrix, axis=0))
        analysis['quality_metrics']['duplicates'] = len(embeddings) - unique_embeddings
        
        return analysis

class BatchEmbeddingProcessor:
    """Process large document collections efficiently"""
    
    def __init__(self, generator: EmbeddingGenerator):
        self.generator = generator
    
    def process_document_collection(self,
                                   documents: List[Dict[str, Any]],
                                   chunking_strategy: ChunkingStrategy,
                                   progress_callback=None) -> Dict[str, List[EmbeddingResult]]:
        """Process entire document collection"""
        all_results = {}
        
        for i, doc in enumerate(documents):
            doc_id = doc['document_id']
            
            # Chunk document
            chunks = chunking_strategy.chunk_document(doc, doc_id)
            
            # Generate embeddings
            embeddings = self.generator.generate_embeddings(chunks)
            
            all_results[doc_id] = embeddings
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(documents), doc_id)
        
        return all_results
```

### Error Handling
```python
class EmbeddingGenerationError(Exception):
    """Base exception for embedding generation errors"""
    pass

class ModelLoadError(EmbeddingGenerationError):
    """Raised when model fails to load"""
    pass

class EmbeddingDimensionError(EmbeddingGenerationError):
    """Raised when embedding dimension is unexpected"""
    pass

def safe_generate_with_fallback(generator: EmbeddingGenerator,
                               chunks: List[TextChunk],
                               max_retries: int = 3) -> List[EmbeddingResult]:
    """Generate embeddings with retry logic"""
    for attempt in range(max_retries):
        try:
            return generator.generate_embeddings(chunks)
        except torch.cuda.OutOfMemoryError:
            # Reduce batch size and retry
            generator.batch_size = max(1, generator.batch_size // 2)
            print(f"Reducing batch size to {generator.batch_size}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise EmbeddingGenerationError(f"Failed after {max_retries} attempts: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
```

## VALIDATION LOOP
**Comprehensive Embedding Generation Testing**

### Unit Testing
```python
def test_model_initialization():
    """Test model loading and initialization"""
    generator = EmbeddingGenerator(device='cpu')
    
    assert generator.model is not None
    assert generator.embedding_dimension == 1024
    assert generator.device == 'cpu'

def test_single_embedding_generation():
    """Test single text embedding"""
    generator = EmbeddingGenerator()
    
    text = "This is a test sentence for embedding generation."
    embedding = generator.generate_single_embedding(text)
    
    assert embedding.shape == (1024,)
    assert embedding.dtype == np.float32
    # Check normalization
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

def test_batch_processing():
    """Test batch embedding generation"""
    generator = EmbeddingGenerator(batch_size=4)
    
    # Create test chunks
    chunks = create_test_chunks(10)
    
    results = generator.generate_embeddings(chunks, use_cache=False)
    
    assert len(results) == 10
    assert all(r.dimension == 1024 for r in results)
    assert all(r.embedding.shape == (1024,) for r in results)

def test_similarity_computation():
    """Test cosine similarity calculation"""
    generator = EmbeddingGenerator()
    
    # Generate embeddings for similar texts
    text1 = "Machine learning is a subset of artificial intelligence."
    text2 = "AI includes machine learning as a key component."
    text3 = "The weather is sunny today."
    
    emb1 = generator.generate_single_embedding(text1)
    emb2 = generator.generate_single_embedding(text2)
    emb3 = generator.generate_single_embedding(text3)
    
    sim12 = generator.compute_similarity(emb1, emb2)
    sim13 = generator.compute_similarity(emb1, emb3)
    
    assert sim12 > sim13  # Similar texts should have higher similarity
    assert 0 <= sim12 <= 1
    assert 0 <= sim13 <= 1
```

### Integration Testing
```python
def test_full_embedding_pipeline():
    """Test complete embedding generation pipeline"""
    generator = EmbeddingGenerator()
    chunking_strategy = ChunkingStrategy()
    
    # Process test document
    test_doc = get_sample_processed_document()
    chunks = chunking_strategy.chunk_document(test_doc, "test_doc")
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(chunks)
    
    assert len(embeddings) == len(chunks)
    
    # Analyze quality
    analysis = EmbeddingQualityAnalyzer.analyze_embeddings(embeddings)
    assert analysis['quality_metrics']['properly_normalized']

def test_caching_mechanism():
    """Test embedding caching"""
    generator = EmbeddingGenerator()
    
    chunks = create_test_chunks(5)
    
    # First generation (no cache)
    results1 = generator.generate_embeddings(chunks, use_cache=True)
    assert all(not r.cached for r in results1)
    
    # Second generation (from cache)
    results2 = generator.generate_embeddings(chunks, use_cache=True)
    assert all(r.cached for r in results2)
    
    # Verify embeddings are identical
    for r1, r2 in zip(results1, results2):
        assert np.allclose(r1.embedding, r2.embedding)

def test_multilingual_embeddings():
    """Test Portuguese and English embedding generation"""
    generator = EmbeddingGenerator()
    
    texts = [
        "This is an English sentence.",
        "Esta é uma frase em português.",
        "Mixed: English e português juntos."
    ]
    
    embeddings = [generator.generate_single_embedding(t) for t in texts]
    
    assert all(e.shape == (1024,) for e in embeddings)
    assert all(np.allclose(np.linalg.norm(e), 1.0) for e in embeddings)
```

### Performance Testing
* Generation speed: > 100 chunks/second on CPU
* Memory usage: < 2GB for model + 1000 embeddings
* Cache performance: < 1ms for cached embedding retrieval
* Batch efficiency: Linear scaling up to batch size 32

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
* **Input Validation**: Sanitize text input before embedding
* **Resource Limits**: Set maximum text length to prevent DoS
* **Cache Security**: Protect cached embeddings from unauthorized access
* **Model Integrity**: Verify model checksum before loading
* **Memory Protection**: Limit batch sizes to prevent OOM attacks

### Performance Optimization
* **CPU Optimization**: Use Intel MKL or OpenBLAS for matrix operations
* **Batch Sizing**: Dynamic batch size based on available memory
* **Caching Strategy**: LRU cache for frequently accessed embeddings
* **Parallel Processing**: Use multiprocessing for large collections
* **Quantization**: Consider INT8 quantization for faster inference

### Maintenance Requirements
* **Model Updates**: Regular updates to latest BGE-M3 versions
* **Performance Monitoring**: Track generation times and cache hit rates
* **Quality Checks**: Periodic embedding quality analysis
* **Cache Management**: Regular cleanup of old cached embeddings
* **Documentation**: Maintain embedding dimension and model specifications