# Embedding Generation - PRP

## ROLE
**ML Engineer with Text Embedding and Vector Processing expertise**

Responsible for implementing high-quality embedding generation using BAAI/bge-m3 model for document chunks. Must have experience with transformer models, vector embeddings, batch processing, and embedding optimization for semantic search applications.

## OBJECTIVE
**Generate high-quality 1024-dimensional embeddings using BAAI/bge-m3 model**

Develop a robust embedding generation system that:
- Loads and optimizes BAAI/bge-m3 model for local CPU inference
- Processes document chunks to generate 1024-dimensional embeddings
- Implements efficient batch processing for large document sets
- Provides embedding quality assessment and validation
- Handles multilingual content (Portuguese/English)
- Optimizes memory usage and processing performance
- Generates stable, reproducible embeddings for consistent search results

Success criteria: Generate embeddings for 95%+ of input chunks with consistent quality, processing throughput >100 chunks/minute, and embedding similarity >0.95 for identical content.

## MOTIVATION
**Enable semantic search and knowledge retrieval through vector representations**

High-quality embeddings are the foundation of semantic search systems. The BAAI/bge-m3 model provides state-of-the-art multilingual embeddings that capture semantic meaning, enabling accurate document retrieval, similarity search, and knowledge discovery across the NIC document corpus.

## CONTEXT
**NIC ETL Pipeline - Embedding Generation Phase**

Technology Stack:
- Python 3.8+ with jupyter notebook environment
- BAAI/bge-m3 model (local CPU deployment)
- Transformers library for model loading and inference
- Input from metadata-enriched chunks
- Output to QDrant vector database integration

Model Configuration:
- Model: BAAI/bge-m3
- Dimensions: 1024
- Deployment: Local CPU inference
- Languages: Portuguese, English
- Max sequence length: 8192 tokens (BGE-M3 capacity)

## IMPLEMENTATION BLUEPRINT

### Architecture Overview
```
Enriched Chunks → Model Loading → Text Preprocessing → Batch Embedding Generation → Quality Validation → Embedding Storage Preparation
```

### Code Structure
```python
# File organization
src/
├── embeddings/
│   ├── __init__.py
│   ├── bge_m3_model.py           # BGE-M3 model wrapper
│   ├── text_preprocessor.py      # Text preprocessing for embeddings
│   ├── batch_processor.py        # Batch processing optimization
│   ├── embedding_validator.py    # Embedding quality validation
│   ├── embedding_cache.py        # Caching for performance
│   └── embedding_orchestrator.py # Main embedding pipeline
├── models/
│   └── embedding_models.py       # Data models for embeddings
└── notebooks/
    └── 07_embedding_generation.ipynb
```

### BGE-M3 Model Integration
```python
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class BGEM3EmbeddingModel:
    """BGE-M3 embedding model wrapper for local CPU inference"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", 
                 device: str = "cpu", 
                 cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.max_length = 8192  # BGE-M3 max sequence length
        self.embedding_dim = 1024
        self.logger = logging.getLogger(__name__)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize BGE-M3 model and tokenizer"""
        try:
            self.logger.info(f"Loading BGE-M3 model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.float32  # Use float32 for CPU
            )
            
            # Move to specified device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.logger.info(f"BGE-M3 model loaded successfully on {self.device}")
            
            # Test model with dummy input
            self._test_model_functionality()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BGE-M3 model: {e}")
            raise
    
    def _test_model_functionality(self):
        """Test model with dummy input to ensure functionality"""
        try:
            test_text = "This is a test sentence for model validation."
            embedding = self.generate_embedding(test_text)
            
            if embedding is None or len(embedding) != self.embedding_dim:
                raise ValueError(f"Model test failed: Expected {self.embedding_dim} dimensions, got {len(embedding) if embedding else None}")
            
            self.logger.info("Model functionality test passed")
            
        except Exception as e:
            self.logger.error(f"Model functionality test failed: {e}")
            raise
    
    def generate_embedding(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """Generate embedding for single text input"""
        try:
            if not text or not text.strip():
                self.logger.warning("Empty text provided for embedding generation")
                return None
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use mean pooling on token embeddings
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to numpy
                embedding = embeddings.cpu().numpy().flatten()
                
                return embedding
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed for text: {e}")
            return None
    
    def generate_batch_embeddings(self, texts: List[str], 
                                 batch_size: int = 8,
                                 normalize: bool = True) -> List[Optional[np.ndarray]]:
        """Generate embeddings for batch of texts"""
        embeddings = []
        
        try:
            # Process in batches for memory efficiency
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._process_batch(batch_texts, normalize)
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {e}")
            return [None] * len(texts)
    
    def _process_batch(self, batch_texts: List[str], normalize: bool = True) -> List[Optional[np.ndarray]]:
        """Process single batch of texts"""
        try:
            # Filter empty texts
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(batch_texts):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i)
            
            if not valid_texts:
                return [None] * len(batch_texts)
            
            # Tokenize batch
            inputs = self.tokenizer(
                valid_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to numpy
                embeddings_np = embeddings.cpu().numpy()
            
            # Map back to original indices
            result_embeddings = [None] * len(batch_texts)
            for i, valid_idx in enumerate(valid_indices):
                result_embeddings[valid_idx] = embeddings_np[i]
            
            return result_embeddings
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return [None] * len(batch_texts)
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to token embeddings"""
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Apply mask and sum
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Calculate mean (avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'max_sequence_length': self.max_length,
            'device': self.device,
            'model_size_mb': self._estimate_model_size(),
            'tokenizer_vocab_size': len(self.tokenizer.vocab) if self.tokenizer else 0
        }
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB"""
        if self.model is None:
            return 0.0
        
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return round(size_mb, 2)
```

### Text Preprocessing for Embeddings
```python
import re
from typing import List, Dict, Any
import unicodedata

class EmbeddingTextPreprocessor:
    """Preprocess text for optimal embedding generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_chunk_for_embedding(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess chunk text for embedding generation"""
        try:
            original_text = chunk.get('text', '')
            
            if not original_text.strip():
                return {
                    'success': False,
                    'processed_text': '',
                    'error': 'Empty text content'
                }
            
            # Apply preprocessing pipeline
            processed_text = self._normalize_unicode(original_text)
            processed_text = self._clean_ocr_artifacts(processed_text)
            processed_text = self._normalize_whitespace(processed_text)
            processed_text = self._enhance_context_with_metadata(processed_text, chunk)
            processed_text = self._truncate_if_necessary(processed_text)
            
            # Calculate preprocessing statistics
            preprocessing_stats = {
                'original_length': len(original_text),
                'processed_length': len(processed_text),
                'reduction_ratio': 1 - (len(processed_text) / len(original_text)) if original_text else 0,
                'preprocessing_applied': [
                    'unicode_normalization',
                    'ocr_cleanup',
                    'whitespace_normalization',
                    'metadata_enhancement'
                ]
            }
            
            return {
                'success': True,
                'processed_text': processed_text,
                'preprocessing_stats': preprocessing_stats,
                'token_estimate': self._estimate_token_count(processed_text)
            }
            
        except Exception as e:
            self.logger.error(f"Text preprocessing failed: {e}")
            return {
                'success': False,
                'processed_text': original_text,  # Fallback to original
                'error': str(e)
            }
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        # Normalize to NFC form (canonical decomposition followed by canonical composition)
        text = unicodedata.normalize('NFC', text)
        
        # Remove or replace problematic Unicode characters
        # Replace em-dash and en-dash with regular dash
        text = text.replace('—', '-').replace('–', '-')
        
        # Replace smart quotes with regular quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)
        
        return text
    
    def _clean_ocr_artifacts(self, text: str) -> str:
        """Clean common OCR artifacts"""
        # Remove excessive repetition of special characters
        text = re.sub(r'[-_=]{5,}', '---', text)
        text = re.sub(r'[.]{4,}', '...', text)
        
        # Fix common OCR character substitutions
        ocr_corrections = {
            r'\b0\b': 'o',     # Zero to letter O in words
            r'\b1\b': 'I',     # One to letter I in words  
            r'rn\b': 'm',      # rn to m at word endings
            r'\bm\b': 'in',    # Standalone m to in
        }
        
        for pattern, replacement in ocr_corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove orphaned punctuation
        text = re.sub(r'(?<!\w)[.,;:!?](?!\w)', '', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks"""
        # Replace multiple consecutive whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple line breaks with double line break
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Clean up spacing around common punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s*', r'\1 ', text)
        
        return text.strip()
    
    def _enhance_context_with_metadata(self, text: str, chunk: Dict[str, Any]) -> str:
        """Enhance text with contextual metadata for better embeddings"""
        # Extract relevant metadata
        doc_metadata = chunk.get('document_metadata', {})
        section_metadata = chunk.get('section_metadata', {})
        nic_metadata = chunk.get('nic_metadata', {})
        
        # Create context prefix
        context_parts = []
        
        # Document context
        doc_title = doc_metadata.get('title', '')
        if doc_title and len(doc_title.strip()) > 3:
            context_parts.append(f"Documento: {doc_title}")
        
        # Section context
        section_title = section_metadata.get('title', '')
        section_path = nic_metadata.get('section_path', '')
        
        if section_title and len(section_title.strip()) > 3:
            context_parts.append(f"Seção: {section_title}")
        elif section_path:
            context_parts.append(f"Seção: {section_path}")
        
        # Tags context
        tags = doc_metadata.get('tags', [])
        if tags and isinstance(tags, list):
            tag_text = ', '.join(tags[:3])  # Limit to top 3 tags
            context_parts.append(f"Tags: {tag_text}")
        
        # Build enhanced text
        if context_parts:
            context_prefix = ' | '.join(context_parts)
            enhanced_text = f"{context_prefix}\n\n{text}"
        else:
            enhanced_text = text
        
        return enhanced_text
    
    def _truncate_if_necessary(self, text: str, max_chars: int = 30000) -> str:
        """Truncate text if it exceeds reasonable limits"""
        if len(text) <= max_chars:
            return text
        
        # Truncate at sentence boundary if possible
        truncated = text[:max_chars]
        last_sentence = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence > max_chars * 0.8:  # If we can keep most content
            return truncated[:last_sentence + 1]
        else:
            return truncated + "..."
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count"""
        # Simple heuristic: ~1.3 tokens per word for Portuguese/English
        word_count = len(text.split())
        return int(word_count * 1.3)
```

### Batch Processing Optimization
```python
from typing import List, Dict, Any, Generator
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

class BatchEmbeddingProcessor:
    """Optimized batch processing for embedding generation"""
    
    def __init__(self, model: BGEM3EmbeddingModel, 
                 batch_size: int = 8,
                 max_workers: int = 1):  # CPU-based, limit concurrency
        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def process_chunks_to_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process chunks to generate embeddings with optimization"""
        try:
            start_time = time.time()
            preprocessor = EmbeddingTextPreprocessor()
            
            # Preprocess all chunks
            self.logger.info(f"Preprocessing {len(chunks)} chunks for embedding generation")
            preprocessed_chunks = []
            
            for chunk in chunks:
                processed = preprocessor.preprocess_chunk_for_embedding(chunk)
                if processed['success']:
                    chunk_copy = chunk.copy()
                    chunk_copy['processed_text'] = processed['processed_text']
                    chunk_copy['preprocessing_stats'] = processed['preprocessing_stats']
                    preprocessed_chunks.append(chunk_copy)
                else:
                    self.logger.warning(f"Preprocessing failed for chunk {chunk.get('chunk_id', 'unknown')}")
            
            # Generate embeddings in batches
            self.logger.info(f"Generating embeddings for {len(preprocessed_chunks)} preprocessed chunks")
            embedding_results = self._generate_embeddings_batch(preprocessed_chunks)
            
            # Calculate statistics
            end_time = time.time()
            processing_stats = self._calculate_processing_statistics(
                chunks, embedding_results, start_time, end_time
            )
            
            return {
                'success': True,
                'embeddings': embedding_results,
                'processing_stats': processing_stats,
                'total_processed': len(embedding_results)
            }
            
        except Exception as e:
            self.logger.error(f"Batch embedding processing failed: {e}")
            return {
                'success': False,
                'embeddings': [],
                'error': str(e)
            }
    
    def _generate_embeddings_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings using optimized batching"""
        embedding_results = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        for batch_idx, batch in enumerate(self._create_batches(chunks, self.batch_size)):
            self.logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            try:
                # Extract texts for embedding generation
                batch_texts = [chunk.get('processed_text', chunk.get('text', '')) for chunk in batch]
                
                # Generate embeddings
                batch_embeddings = self.model.generate_batch_embeddings(
                    batch_texts, 
                    batch_size=len(batch_texts)
                )
                
                # Package results
                for i, (chunk, embedding) in enumerate(zip(batch, batch_embeddings)):
                    result = {
                        'chunk_id': chunk.get('chunk_id', f'chunk_{batch_idx}_{i}'),
                        'embedding': embedding,
                        'embedding_successful': embedding is not None,
                        'original_chunk': chunk
                    }
                    
                    if embedding is not None:
                        result['embedding_stats'] = {
                            'dimension': len(embedding),
                            'norm': float(np.linalg.norm(embedding)),
                            'mean': float(np.mean(embedding)),
                            'std': float(np.std(embedding))
                        }
                    
                    embedding_results.append(result)
                
                # Memory management
                self._monitor_memory_usage()
                
            except Exception as e:
                self.logger.error(f"Batch {batch_idx + 1} processing failed: {e}")
                # Add failed entries
                for i, chunk in enumerate(batch):
                    embedding_results.append({
                        'chunk_id': chunk.get('chunk_id', f'chunk_{batch_idx}_{i}'),
                        'embedding': None,
                        'embedding_successful': False,
                        'error': str(e),
                        'original_chunk': chunk
                    })
        
        return embedding_results
    
    def _create_batches(self, chunks: List[Dict[str, Any]], batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """Create batches from chunks list"""
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]
    
    def _monitor_memory_usage(self):
        """Monitor and log memory usage"""
        memory_info = psutil.virtual_memory()
        memory_usage_mb = (memory_info.total - memory_info.available) / 1024 / 1024
        
        if memory_usage_mb > 4000:  # Warn if using more than 4GB
            self.logger.warning(f"High memory usage detected: {memory_usage_mb:.1f} MB")
    
    def _calculate_processing_statistics(self, original_chunks: List[Dict[str, Any]], 
                                       embedding_results: List[Dict[str, Any]], 
                                       start_time: float, end_time: float) -> Dict[str, Any]:
        """Calculate comprehensive processing statistics"""
        total_time = end_time - start_time
        successful_embeddings = len([r for r in embedding_results if r['embedding_successful']])
        failed_embeddings = len(embedding_results) - successful_embeddings
        
        # Calculate processing rates
        chunks_per_second = len(original_chunks) / total_time if total_time > 0 else 0
        chunks_per_minute = chunks_per_second * 60
        
        # Calculate embedding statistics
        successful_results = [r for r in embedding_results if r['embedding_successful']]
        embedding_stats = {}
        
        if successful_results:
            norms = [r['embedding_stats']['norm'] for r in successful_results]
            dimensions = [r['embedding_stats']['dimension'] for r in successful_results]
            
            embedding_stats = {
                'mean_norm': float(np.mean(norms)),
                'std_norm': float(np.std(norms)),
                'min_norm': float(np.min(norms)),
                'max_norm': float(np.max(norms)),
                'consistent_dimensions': len(set(dimensions)) == 1,
                'dimension': dimensions[0] if dimensions else 0
            }
        
        return {
            'total_chunks': len(original_chunks),
            'successful_embeddings': successful_embeddings,
            'failed_embeddings': failed_embeddings,
            'success_rate': successful_embeddings / len(original_chunks) if original_chunks else 0,
            'processing_time_seconds': total_time,
            'chunks_per_second': chunks_per_second,
            'chunks_per_minute': chunks_per_minute,
            'embedding_statistics': embedding_stats
        }
```

### Embedding Quality Validation
```python
import numpy as np
from typing import List, Dict, Any
from scipy.spatial.distance import cosine
import statistics

class EmbeddingQualityValidator:
    """Validate embedding quality and consistency"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_embeddings(self, embedding_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive embedding quality validation"""
        try:
            validation_result = {
                'overall_quality_score': 0.0,
                'validation_passed': False,
                'quality_metrics': {},
                'recommendations': [],
                'detailed_analysis': {}
            }
            
            # Filter successful embeddings
            successful_embeddings = [r for r in embedding_results if r['embedding_successful']]
            
            if not successful_embeddings:
                validation_result['recommendations'].append("No successful embeddings found - check model and preprocessing")
                return validation_result
            
            # Run quality checks
            dimensional_consistency = self._check_dimensional_consistency(successful_embeddings)
            norm_consistency = self._check_norm_consistency(successful_embeddings)
            distribution_health = self._check_distribution_health(successful_embeddings)
            similarity_coherence = self._check_similarity_coherence(successful_embeddings)
            
            # Compile quality metrics
            validation_result['quality_metrics'] = {
                'dimensional_consistency': dimensional_consistency,
                'norm_consistency': norm_consistency,
                'distribution_health': distribution_health,
                'similarity_coherence': similarity_coherence
            }
            
            # Calculate overall quality score
            scores = [
                dimensional_consistency.get('score', 0),
                norm_consistency.get('score', 0),
                distribution_health.get('score', 0),
                similarity_coherence.get('score', 0)
            ]
            
            validation_result['overall_quality_score'] = sum(scores) / len(scores)
            validation_result['validation_passed'] = validation_result['overall_quality_score'] >= 75
            
            # Generate recommendations
            validation_result['recommendations'] = self._generate_quality_recommendations(
                validation_result['quality_metrics']
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Embedding quality validation failed: {e}")
            return {
                'overall_quality_score': 0.0,
                'validation_passed': False,
                'error': str(e)
            }
    
    def _check_dimensional_consistency(self, embedding_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check that all embeddings have consistent dimensions"""
        dimensions = [r['embedding_stats']['dimension'] for r in embedding_results]
        
        unique_dimensions = set(dimensions)
        is_consistent = len(unique_dimensions) == 1
        
        return {
            'score': 100.0 if is_consistent else 0.0,
            'is_consistent': is_consistent,
            'unique_dimensions': list(unique_dimensions),
            'expected_dimension': 1024,
            'dimension_matches_expected': 1024 in unique_dimensions
        }
    
    def _check_norm_consistency(self, embedding_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check embedding norm consistency (should be ~1.0 for normalized embeddings)"""
        norms = [r['embedding_stats']['norm'] for r in embedding_results]
        
        mean_norm = statistics.mean(norms)
        std_norm = statistics.stdev(norms) if len(norms) > 1 else 0
        
        # Score based on how close to 1.0 and how consistent
        norm_score = max(0, 100 - abs(mean_norm - 1.0) * 100)  # Penalty for deviation from 1.0
        consistency_score = max(0, 100 - std_norm * 100)  # Penalty for high variance
        
        overall_score = (norm_score + consistency_score) / 2
        
        return {
            'score': overall_score,
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'min_norm': min(norms),
            'max_norm': max(norms),
            'norm_range': max(norms) - min(norms),
            'properly_normalized': 0.95 <= mean_norm <= 1.05
        }
    
    def _check_distribution_health(self, embedding_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check embedding value distribution health"""
        # Collect all embedding values for statistical analysis
        all_values = []
        for result in embedding_results:
            if result['embedding'] is not None:
                all_values.extend(result['embedding'].tolist())
        
        if not all_values:
            return {'score': 0.0, 'error': 'No embedding values found'}
        
        mean_value = statistics.mean(all_values)
        std_value = statistics.stdev(all_values)
        
        # Check for reasonable distribution (not too concentrated, not too sparse)
        score = 100.0
        
        # Penalize if mean is too far from 0 (indicates bias)
        if abs(mean_value) > 0.1:
            score -= 20
        
        # Penalize if standard deviation is too low (under-dispersed) or too high (over-dispersed)
        if std_value < 0.1:
            score -= 30  # Under-dispersed
        elif std_value > 1.0:
            score -= 20  # Over-dispersed
        
        return {
            'score': max(0, score),
            'mean_value': mean_value,
            'std_value': std_value,
            'value_range': max(all_values) - min(all_values),
            'distribution_healthy': abs(mean_value) < 0.1 and 0.1 <= std_value <= 1.0
        }
    
    def _check_similarity_coherence(self, embedding_results: List[Dict[str, Any]], 
                                  sample_size: int = 20) -> Dict[str, Any]:
        """Check similarity coherence by comparing embeddings of similar/different content"""
        if len(embedding_results) < 10:
            return {'score': 50.0, 'warning': 'Insufficient embeddings for coherence check'}
        
        # Sample embeddings for comparison
        sample_results = embedding_results[:min(sample_size, len(embedding_results))]
        
        similarities = []
        
        # Calculate pairwise similarities
        for i in range(len(sample_results)):
            for j in range(i + 1, len(sample_results)):
                emb1 = sample_results[i]['embedding']
                emb2 = sample_results[j]['embedding']
                
                if emb1 is not None and emb2 is not None:
                    similarity = 1 - cosine(emb1, emb2)  # Cosine similarity
                    similarities.append(similarity)
        
        if not similarities:
            return {'score': 0.0, 'error': 'No valid similarity calculations'}
        
        # Analyze similarity distribution
        mean_similarity = statistics.mean(similarities)
        std_similarity = statistics.stdev(similarities) if len(similarities) > 1 else 0
        
        # Score based on reasonable similarity distribution
        score = 100.0
        
        # Expect some variation in similarities (not all identical, not all orthogonal)
        if mean_similarity > 0.95:  # Too similar
            score -= 30
        elif mean_similarity < 0.1:  # Too dissimilar
            score -= 40
        
        if std_similarity < 0.05:  # Too uniform
            score -= 20
        
        return {
            'score': max(0, score),
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'min_similarity': min(similarities),
            'max_similarity': max(similarities),
            'similarity_distribution_healthy': 0.2 <= mean_similarity <= 0.8 and std_similarity >= 0.05
        }
    
    def _generate_quality_recommendations(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality metrics"""
        recommendations = []
        
        # Dimensional consistency recommendations
        if not quality_metrics.get('dimensional_consistency', {}).get('is_consistent', True):
            recommendations.append("Fix dimensional inconsistency - ensure all embeddings have 1024 dimensions")
        
        # Norm consistency recommendations
        norm_metrics = quality_metrics.get('norm_consistency', {})
        if not norm_metrics.get('properly_normalized', True):
            recommendations.append("Enable embedding normalization - embeddings should have unit norm (~1.0)")
        
        # Distribution health recommendations
        dist_metrics = quality_metrics.get('distribution_health', {})
        if not dist_metrics.get('distribution_healthy', True):
            recommendations.append("Review embedding distribution - values may be biased or poorly dispersed")
        
        # Similarity coherence recommendations
        sim_metrics = quality_metrics.get('similarity_coherence', {})
        if not sim_metrics.get('similarity_distribution_healthy', True):
            recommendations.append("Check similarity patterns - embeddings may be too similar or too dissimilar")
        
        return recommendations if recommendations else ["Embedding quality is satisfactory"]
```

## VALIDATION LOOP

### Unit Testing
```python
import pytest
import numpy as np
from src.embeddings.bge_m3_model import BGEM3EmbeddingModel

class TestEmbeddingGeneration:
    def test_model_initialization(self):
        """Test BGE-M3 model loading and initialization"""
        model = BGEM3EmbeddingModel()
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.embedding_dim == 1024
    
    def test_single_embedding_generation(self):
        """Test single text embedding generation"""
        model = BGEM3EmbeddingModel()
        test_text = "Este é um teste de geração de embeddings em português."
        
        embedding = model.generate_embedding(test_text)
        assert embedding is not None
        assert len(embedding) == 1024
        assert abs(np.linalg.norm(embedding) - 1.0) < 0.01  # Normalized
    
    def test_batch_embedding_generation(self):
        """Test batch embedding generation"""
        model = BGEM3EmbeddingModel()
        texts = [
            "Primeiro texto para teste",
            "Segundo texto em português", 
            "Third text in English"
        ]
        
        embeddings = model.generate_batch_embeddings(texts, batch_size=2)
        assert len(embeddings) == 3
        assert all(emb is not None for emb in embeddings)
        assert all(len(emb) == 1024 for emb in embeddings)
    
    def test_preprocessing_quality(self):
        """Test text preprocessing for embeddings"""
        from src.embeddings.text_preprocessor import EmbeddingTextPreprocessor
        
        preprocessor = EmbeddingTextPreprocessor()
        chunk = {
            'text': 'Test   text  with   irregular   spacing.',
            'document_metadata': {'title': 'Test Document'},
            'section_metadata': {'title': 'Test Section'}
        }
        
        result = preprocessor.preprocess_chunk_for_embedding(chunk)
        assert result['success'] == True
        assert 'Test Document' in result['processed_text']  # Context added
        assert '   ' not in result['processed_text']  # Whitespace normalized
```

### Integration Testing
- End-to-end embedding generation pipeline
- Performance testing with large document batches
- Quality consistency validation across different content types

### Performance Testing
- Process 1000+ chunks within 10 minutes on CPU
- Memory usage under 2GB during batch processing
- Embedding consistency >95% for identical inputs

## ADDITIONAL NOTES

### Security Considerations
- Model file integrity verification
- Secure handling of document content during processing
- Memory cleanup to prevent data leakage
- Access control for embedding generation services

### Performance Optimization
- Model quantization for reduced memory usage
- Batch size optimization based on available memory
- GPU acceleration support (optional enhancement)
- Embedding caching for duplicate content

### Maintenance Requirements
- Regular model updates and compatibility testing
- Embedding quality monitoring and alerting
- Performance metrics collection and analysis
- Integration testing with QDrant vector database