# Text Chunking Engine - PRP

## ROLE
**Python NLP Developer with Text Processing expertise**

Responsible for implementing intelligent text chunking system using paragraph-based strategy with BGE tokenizer for accurate token counting. Must have experience with tokenization, text segmentation, overlap strategies, and chunk optimization for embedding generation.

## OBJECTIVE
**Implement paragraph-based text chunking with BGE tokenizer for optimal embeddings**

Develop a sophisticated chunking system that:
- Uses paragraph boundaries as primary chunking strategy
- Employs BGE tokenizer for accurate token counting (500 tokens per chunk)
- Implements configurable chunk overlap (100 tokens)
- Respects document structure and section boundaries
- Maintains semantic coherence within chunks
- Generates chunk metadata for lineage tracking
- Optimizes chunk quality for embedding generation

Success criteria: Generate chunks with 95%+ adherence to token limits, maintain semantic coherence, and ensure proper overlap handling across all document types.

## MOTIVATION
**Enable high-quality embedding generation through optimal text segmentation**

Proper text chunking is crucial for embedding quality and downstream semantic search performance. By using paragraph-based chunking that respects document structure and employs accurate tokenization, the system ensures that each chunk contains semantically coherent content that translates to meaningful embeddings for search and retrieval.

## CONTEXT
**NIC ETL Pipeline - Text Chunking Phase**

Technology Stack:
- Python 3.8+ with jupyter notebook environment
- BGE (BAAI/bge-m3) tokenizer for accurate token counting
- Transformers library for tokenizer integration
- Input from document structure analysis pipeline
- Output to metadata extraction and embedding generation

Chunking Requirements:
- Chunk strategy: Paragraph-based with structure awareness
- Chunk size: 500 tokens (measured with BGE tokenizer)
- Chunk overlap: 100 tokens
- Language support: Portuguese and English
- Document structure preservation

## IMPLEMENTATION BLUEPRINT

### Architecture Overview
```
Structured Documents → Paragraph Detection → Token-based Segmentation → Overlap Management → Chunk Optimization → Metadata Enrichment
```

### Code Structure
```python
# File organization
src/
├── chunking/
│   ├── __init__.py
│   ├── bge_tokenizer.py           # BGE tokenizer wrapper
│   ├── paragraph_chunker.py       # Paragraph-based chunking logic
│   ├── structure_aware_chunker.py # Structure-aware chunking
│   ├── overlap_manager.py         # Chunk overlap handling
│   ├── chunk_optimizer.py         # Chunk quality optimization
│   ├── chunk_metadata.py          # Chunk metadata generation
│   └── chunking_orchestrator.py   # Main chunking pipeline
├── models/
│   └── chunk_models.py           # Data models for chunks
└── notebooks/
    └── 05_text_chunking.ipynb
```

### BGE Tokenizer Integration
```python
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple
import logging

class BGETokenizer:
    """BGE tokenizer wrapper for accurate token counting"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.tokenizer = None
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize BGE tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.logger.info(f"BGE tokenizer initialized: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize BGE tokenizer: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using BGE tokenizer"""
        try:
            if not text.strip():
                return 0
            
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False
            )
            return len(tokens)
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}")
            # Fallback to approximate counting
            return len(text.split()) * 1.3  # Rough approximation
    
    def tokenize_with_offsets(self, text: str) -> Dict[str, Any]:
        """Tokenize text and return tokens with character offsets"""
        try:
            encoding = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=False
            )
            
            return {
                'tokens': encoding['input_ids'],
                'token_count': len(encoding['input_ids']),
                'offsets': encoding['offset_mapping'],
                'attention_mask': encoding['attention_mask']
            }
        except Exception as e:
            self.logger.error(f"Tokenization with offsets failed: {e}")
            return {
                'tokens': [],
                'token_count': 0,
                'offsets': [],
                'attention_mask': []
            }
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        try:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        except Exception as e:
            self.logger.error(f"Token decoding failed: {e}")
            return ""
    
    def find_token_boundary(self, text: str, target_token_count: int) -> Tuple[int, int]:
        """Find character position that corresponds to target token count"""
        try:
            encoding = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=False
            )
            
            if len(encoding['input_ids']) <= target_token_count:
                return len(text), len(encoding['input_ids'])
            
            # Find the character offset for the target token
            if target_token_count < len(encoding['offset_mapping']):
                char_end = encoding['offset_mapping'][target_token_count][0]
                return char_end, target_token_count
            
            return len(text), len(encoding['input_ids'])
            
        except Exception as e:
            self.logger.warning(f"Token boundary detection failed: {e}")
            # Fallback: estimate based on average tokens per character
            chars_per_token = len(text) / self.count_tokens(text) if self.count_tokens(text) > 0 else 4
            estimated_chars = int(target_token_count * chars_per_token)
            return min(estimated_chars, len(text)), target_token_count
```

### Paragraph-based Chunking Implementation
```python
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class ChunkCandidate:
    """Represents a potential chunk during processing"""
    text: str
    token_count: int
    start_paragraph: int
    end_paragraph: int
    quality_score: float
    metadata: Dict[str, Any]

class ParagraphChunker:
    """Paragraph-based text chunking with structure awareness"""
    
    def __init__(self, tokenizer: BGETokenizer, 
                 target_chunk_size: int = 500,
                 overlap_size: int = 100):
        self.tokenizer = tokenizer
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
        self.logger = logging.getLogger(__name__)
    
    def chunk_document(self, structured_document: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk document using paragraph-based strategy"""
        try:
            # Extract paragraphs from structured document
            paragraphs = self._extract_paragraphs(structured_document)
            
            if not paragraphs:
                return {
                    'success': False,
                    'chunks': [],
                    'error': 'No paragraphs found in document'
                }
            
            # Generate chunks
            chunks = self._generate_paragraph_chunks(paragraphs, structured_document)
            
            # Apply overlap strategy
            overlapped_chunks = self._apply_overlap_strategy(chunks)
            
            # Optimize chunk quality
            optimized_chunks = self._optimize_chunks(overlapped_chunks)
            
            # Generate chunk metadata
            final_chunks = self._enrich_chunk_metadata(optimized_chunks, structured_document)
            
            return {
                'success': True,
                'chunks': final_chunks,
                'statistics': self._generate_chunking_statistics(final_chunks),
                'total_chunks': len(final_chunks)
            }
            
        except Exception as e:
            self.logger.error(f"Document chunking failed: {e}")
            return {
                'success': False,
                'chunks': [],
                'error': str(e)
            }
    
    def _extract_paragraphs(self, structured_document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract paragraphs with structure information"""
        paragraphs = []
        
        try:
            # Handle different input formats
            if 'document_sections' in structured_document:
                # From structure analysis
                paragraphs = self._extract_from_sections(structured_document['document_sections'])
            elif 'elements' in structured_document:
                # From element list
                paragraphs = self._extract_from_elements(structured_document['elements'])
            elif 'normalized_text' in structured_document:
                # From plain text
                paragraphs = self._extract_from_text(structured_document['normalized_text'])
            
            # Filter and validate paragraphs
            valid_paragraphs = []
            for i, para in enumerate(paragraphs):
                if self._is_valid_paragraph(para):
                    para['paragraph_index'] = i
                    valid_paragraphs.append(para)
            
            return valid_paragraphs
            
        except Exception as e:
            self.logger.warning(f"Paragraph extraction failed: {e}")
            return []
    
    def _extract_from_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract paragraphs from structured sections"""
        paragraphs = []
        
        def process_section(section_data, section_path=""):
            current_path = f"{section_path}/{section_data.get('title', '')}" if section_path else section_data.get('title', '')
            
            # Process content elements
            for element in section_data.get('content_elements', []):
                if element.get('type') == 'paragraph':
                    paragraphs.append({
                        'text': element['content'],
                        'section_path': current_path,
                        'section_level': section_data.get('level', 0),
                        'element_type': 'paragraph',
                        'metadata': element.get('metadata', {})
                    })
                elif element.get('type') == 'list':
                    # Treat list items as paragraphs
                    paragraphs.append({
                        'text': element['content'],
                        'section_path': current_path,
                        'section_level': section_data.get('level', 0),
                        'element_type': 'list',
                        'metadata': element.get('metadata', {})
                    })
            
            # Process subsections recursively
            for subsection in section_data.get('subsections', []):
                process_section(subsection, current_path)
        
        # Process all sections
        for section in sections:
            process_section(section)
        
        return paragraphs
    
    def _generate_paragraph_chunks(self, paragraphs: List[Dict[str, Any]], 
                                 structured_document: Dict[str, Any]) -> List[ChunkCandidate]:
        """Generate chunks from paragraphs"""
        chunks = []
        current_chunk_text = ""
        current_chunk_paragraphs = []
        current_token_count = 0
        
        for i, paragraph in enumerate(paragraphs):
            para_text = paragraph['text']
            para_tokens = self.tokenizer.count_tokens(para_text)
            
            # Handle large paragraphs that exceed chunk size
            if para_tokens > self.target_chunk_size:
                # Save current chunk if it has content
                if current_chunk_text:
                    chunks.append(self._create_chunk_candidate(
                        current_chunk_text, current_chunk_paragraphs, current_token_count
                    ))
                    current_chunk_text = ""
                    current_chunk_paragraphs = []
                    current_token_count = 0
                
                # Split large paragraph
                sub_chunks = self._split_large_paragraph(paragraph, para_tokens)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this paragraph would exceed target size
            if current_token_count + para_tokens > self.target_chunk_size and current_chunk_text:
                # Create chunk with current content
                chunks.append(self._create_chunk_candidate(
                    current_chunk_text, current_chunk_paragraphs, current_token_count
                ))
                
                # Start new chunk
                current_chunk_text = para_text
                current_chunk_paragraphs = [i]
                current_token_count = para_tokens
            else:
                # Add paragraph to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para_text
                    current_token_count += para_tokens + 2  # Account for newlines
                else:
                    current_chunk_text = para_text
                    current_token_count = para_tokens
                
                current_chunk_paragraphs.append(i)
        
        # Add final chunk if it has content
        if current_chunk_text:
            chunks.append(self._create_chunk_candidate(
                current_chunk_text, current_chunk_paragraphs, current_token_count
            ))
        
        return chunks
    
    def _split_large_paragraph(self, paragraph: Dict[str, Any], 
                              token_count: int) -> List[ChunkCandidate]:
        """Split paragraphs that are larger than target chunk size"""
        chunks = []
        text = paragraph['text']
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+\s+', text)
        
        current_text = ""
        current_tokens = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            sentence = sentence.strip() + ". "  # Restore punctuation
            sentence_tokens = self.tokenizer.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.target_chunk_size and current_text:
                # Create chunk
                chunk = ChunkCandidate(
                    text=current_text.strip(),
                    token_count=current_tokens,
                    start_paragraph=paragraph.get('paragraph_index', 0),
                    end_paragraph=paragraph.get('paragraph_index', 0),
                    quality_score=0.8,  # Lower score for split paragraphs
                    metadata={
                        'split_paragraph': True,
                        'original_paragraph_tokens': token_count,
                        'section_path': paragraph.get('section_path', ''),
                        'element_type': paragraph.get('element_type', 'paragraph')
                    }
                )
                chunks.append(chunk)
                
                current_text = sentence
                current_tokens = sentence_tokens
            else:
                current_text += sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_text.strip():
            chunk = ChunkCandidate(
                text=current_text.strip(),
                token_count=current_tokens,
                start_paragraph=paragraph.get('paragraph_index', 0),
                end_paragraph=paragraph.get('paragraph_index', 0),
                quality_score=0.8,
                metadata={
                    'split_paragraph': True,
                    'original_paragraph_tokens': token_count,
                    'section_path': paragraph.get('section_path', ''),
                    'element_type': paragraph.get('element_type', 'paragraph')
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_candidate(self, text: str, paragraph_indices: List[int], 
                              token_count: int) -> ChunkCandidate:
        """Create a chunk candidate with quality assessment"""
        
        # Calculate quality score based on various factors
        quality_score = self._calculate_chunk_quality(text, token_count)
        
        return ChunkCandidate(
            text=text,
            token_count=token_count,
            start_paragraph=min(paragraph_indices),
            end_paragraph=max(paragraph_indices),
            quality_score=quality_score,
            metadata={
                'paragraph_count': len(paragraph_indices),
                'paragraph_indices': paragraph_indices,
                'token_efficiency': token_count / self.target_chunk_size
            }
        )
    
    def _calculate_chunk_quality(self, text: str, token_count: int) -> float:
        """Calculate quality score for chunk"""
        score = 100.0
        
        # Token utilization efficiency
        utilization = token_count / self.target_chunk_size
        if utilization < 0.5:
            score -= 20  # Too small
        elif utilization > 1.1:
            score -= 10  # Too large
        
        # Text coherence indicators
        sentence_count = len(re.split(r'[.!?]+', text))
        if sentence_count < 2:
            score -= 15  # Too few sentences
        
        # Paragraph completeness (prefer complete paragraphs)
        if text.endswith('.') or text.endswith('!') or text.endswith('?'):
            score += 10  # Complete sentences
        
        # Length consistency
        if len(text) < 100:
            score -= 20  # Too short
        
        return max(0, min(100, score))
```

### Chunk Overlap Management
```python
from typing import List, Dict, Any

class OverlapManager:
    """Manage chunk overlap for better context preservation"""
    
    def __init__(self, tokenizer: BGETokenizer, overlap_size: int = 100):
        self.tokenizer = tokenizer
        self.overlap_size = overlap_size
        self.logger = logging.getLogger(__name__)
    
    def apply_overlap_strategy(self, chunks: List[ChunkCandidate]) -> List[Dict[str, Any]]:
        """Apply overlap strategy to chunk candidates"""
        if len(chunks) <= 1:
            return [self._chunk_to_dict(chunks[0])] if chunks else []
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_dict = self._chunk_to_dict(chunk)
            
            # Add overlap from previous chunk
            if i > 0:
                overlap_text = self._generate_overlap_from_previous(chunks[i-1], chunk)
                if overlap_text:
                    chunk_dict['text'] = overlap_text + "\n\n" + chunk_dict['text']
                    chunk_dict['token_count'] = self.tokenizer.count_tokens(chunk_dict['text'])
                    chunk_dict['has_previous_overlap'] = True
                    chunk_dict['overlap_metadata'] = {
                        'previous_chunk_id': f"chunk_{i-1}",
                        'overlap_tokens': self.tokenizer.count_tokens(overlap_text)
                    }
            
            # Prepare overlap for next chunk
            if i < len(chunks) - 1:
                chunk_dict['overlap_for_next'] = self._prepare_overlap_for_next(chunk)
            
            chunk_dict['chunk_id'] = f"chunk_{i}"
            overlapped_chunks.append(chunk_dict)
        
        return overlapped_chunks
    
    def _generate_overlap_from_previous(self, previous_chunk: ChunkCandidate, 
                                      current_chunk: ChunkCandidate) -> str:
        """Generate overlap text from previous chunk"""
        try:
            prev_text = previous_chunk.text
            
            # Find the last N tokens from previous chunk
            tokens_data = self.tokenizer.tokenize_with_offsets(prev_text)
            
            if len(tokens_data['tokens']) < self.overlap_size:
                # Use entire previous chunk if it's smaller than overlap size
                return prev_text
            
            # Find character position for overlap
            start_token_idx = max(0, len(tokens_data['tokens']) - self.overlap_size)
            
            if start_token_idx < len(tokens_data['offsets']):
                start_char = tokens_data['offsets'][start_token_idx][0]
                overlap_text = prev_text[start_char:].strip()
                
                # Try to find a natural break (sentence boundary)
                sentences = re.split(r'([.!?]+\s+)', overlap_text)
                if len(sentences) > 1:
                    # Start from first complete sentence
                    for i in range(1, len(sentences), 2):  # Skip punctuation
                        if sentences[i-1].strip():
                            return ''.join(sentences[i-1:]).strip()
                
                return overlap_text
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Overlap generation failed: {e}")
            return ""
    
    def _prepare_overlap_for_next(self, chunk: ChunkCandidate) -> str:
        """Prepare overlap content for next chunk"""
        try:
            # Get last portion of current chunk for next overlap
            text = chunk.text
            tokens_data = self.tokenizer.tokenize_with_offsets(text)
            
            if len(tokens_data['tokens']) < self.overlap_size:
                return text
            
            start_token_idx = max(0, len(tokens_data['tokens']) - self.overlap_size)
            
            if start_token_idx < len(tokens_data['offsets']):
                start_char = tokens_data['offsets'][start_token_idx][0]
                return text[start_char:].strip()
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Overlap preparation failed: {e}")
            return ""
    
    def _chunk_to_dict(self, chunk: ChunkCandidate) -> Dict[str, Any]:
        """Convert ChunkCandidate to dictionary"""
        return {
            'text': chunk.text,
            'token_count': chunk.token_count,
            'start_paragraph': chunk.start_paragraph,
            'end_paragraph': chunk.end_paragraph,
            'quality_score': chunk.quality_score,
            'metadata': chunk.metadata.copy(),
            'has_previous_overlap': False,
            'overlap_metadata': {}
        }
```

### Chunking Statistics and Quality Metrics
```python
from typing import List, Dict, Any
import statistics

class ChunkingStatistics:
    """Generate comprehensive statistics for chunking results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive chunking statistics"""
        if not chunks:
            return {'total_chunks': 0, 'error': 'No chunks provided'}
        
        try:
            stats = {
                'total_chunks': len(chunks),
                'token_statistics': self._calculate_token_statistics(chunks),
                'quality_statistics': self._calculate_quality_statistics(chunks),
                'overlap_statistics': self._calculate_overlap_statistics(chunks),
                'content_distribution': self._calculate_content_distribution(chunks),
                'recommendations': self._generate_recommendations(chunks)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Statistics generation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_token_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate token-related statistics"""
        token_counts = [chunk['token_count'] for chunk in chunks]
        
        return {
            'mean_tokens': statistics.mean(token_counts),
            'median_tokens': statistics.median(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'std_deviation': statistics.stdev(token_counts) if len(token_counts) > 1 else 0,
            'total_tokens': sum(token_counts),
            'target_compliance': len([t for t in token_counts if 400 <= t <= 600]) / len(token_counts)
        }
    
    def _calculate_quality_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality-related statistics"""
        quality_scores = [chunk.get('quality_score', 0) for chunk in chunks]
        
        return {
            'mean_quality': statistics.mean(quality_scores),
            'median_quality': statistics.median(quality_scores),
            'min_quality': min(quality_scores),
            'max_quality': max(quality_scores),
            'high_quality_chunks': len([q for q in quality_scores if q >= 80]) / len(quality_scores),
            'low_quality_chunks': len([q for q in quality_scores if q < 60]) / len(quality_scores)
        }
    
    def _generate_recommendations(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for chunking improvement"""
        recommendations = []
        
        token_counts = [chunk['token_count'] for chunk in chunks]
        quality_scores = [chunk.get('quality_score', 0) for chunk in chunks]
        
        # Token size recommendations
        oversized_chunks = len([t for t in token_counts if t > 600])
        undersized_chunks = len([t for t in token_counts if t < 300])
        
        if oversized_chunks > len(chunks) * 0.1:
            recommendations.append("Consider reducing target chunk size - many chunks exceed 600 tokens")
        
        if undersized_chunks > len(chunks) * 0.2:
            recommendations.append("Consider increasing minimum chunk size - many chunks are too small")
        
        # Quality recommendations
        low_quality_ratio = len([q for q in quality_scores if q < 60]) / len(quality_scores)
        if low_quality_ratio > 0.2:
            recommendations.append("High proportion of low-quality chunks - review paragraph splitting strategy")
        
        # Overlap recommendations
        overlapped_chunks = len([c for c in chunks if c.get('has_previous_overlap', False)])
        if overlapped_chunks / len(chunks) < 0.8:
            recommendations.append("Consider increasing overlap to improve context preservation")
        
        return recommendations
```

## VALIDATION LOOP

### Unit Testing
```python
import pytest
from src.chunking.paragraph_chunker import ParagraphChunker
from src.chunking.bge_tokenizer import BGETokenizer

class TestTextChunking:
    def test_bge_tokenizer_accuracy(self):
        tokenizer = BGETokenizer()
        test_text = "Este é um teste de contagem de tokens em português."
        token_count = tokenizer.count_tokens(test_text)
        assert token_count > 0
        assert token_count < len(test_text.split()) * 2  # Reasonable upper bound
    
    def test_paragraph_chunking(self):
        tokenizer = BGETokenizer()
        chunker = ParagraphChunker(tokenizer, target_chunk_size=100, overlap_size=20)
        
        # Test document with known structure
        structured_doc = {
            'document_sections': [{
                'title': 'Test Section',
                'content_elements': [
                    {'type': 'paragraph', 'content': 'First paragraph with sufficient content for testing.'},
                    {'type': 'paragraph', 'content': 'Second paragraph with more content for chunk testing.'}
                ]
            }]
        }
        
        result = chunker.chunk_document(structured_doc)
        assert result['success'] == True
        assert len(result['chunks']) > 0
        assert all(chunk['token_count'] <= 120 for chunk in result['chunks'])  # Allow some tolerance
    
    def test_overlap_functionality(self):
        tokenizer = BGETokenizer()
        chunker = ParagraphChunker(tokenizer, target_chunk_size=50, overlap_size=10)
        
        # Create test with multiple chunks
        long_text = "This is a test paragraph. " * 20
        structured_doc = {
            'normalized_text': long_text
        }
        
        result = chunker.chunk_document(structured_doc)
        chunks = result['chunks']
        
        if len(chunks) > 1:
            # Check that chunks have overlap
            overlapped_chunks = [c for c in chunks if c.get('has_previous_overlap', False)]
            assert len(overlapped_chunks) > 0
```

### Integration Testing
- End-to-end chunking with various document types and structures
- Validation of token counting accuracy against BGE model
- Performance testing with large documents

### Performance Testing
- Process 10,000 paragraphs within 5 minutes
- Memory usage under 500MB for large documents
- Consistent chunk quality scores >80%

## ADDITIONAL NOTES

### Security Considerations
- Input validation for chunk size parameters
- Memory limits to prevent resource exhaustion
- Secure handling of document content during processing
- Rate limiting for large document processing

### Performance Optimization
- Batch tokenization for efficiency
- Caching of frequently used tokenization results
- Parallel processing for large document sets
- Memory-efficient chunk generation

### Maintenance Requirements
- Regular BGE model updates and compatibility testing
- Chunking quality monitoring and metrics collection
- Parameter tuning based on embedding performance feedback
- Integration monitoring with embedding generation pipeline