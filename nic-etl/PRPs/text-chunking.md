# Text Chunking - PRP

## ROLE
**NLP Engineer with Text Segmentation Expertise**

Specialist in natural language processing, text segmentation algorithms, and semantic chunking strategies. Expert in implementing token-based chunking using transformer tokenizers, maintaining semantic coherence across chunk boundaries, and optimizing chunk sizes for embedding models. Proficient in handling multilingual content, preserving document structure, and implementing overlap strategies for context preservation.

## OBJECTIVE
**Implement Intelligent Paragraph-Based Text Chunking**

Create a sophisticated text chunking module within Jupyter Notebook cells that:
* Implements paragraph-based chunking with 500-token chunks and 100-token overlap
* Uses BAAI/bge-m3 tokenizer for accurate token boundary measurement
* Preserves semantic coherence and document structure
* Maintains metadata and section context for each chunk
* Handles multilingual content (Portuguese and English)
* Generates deterministic chunk IDs for deduplication
* Tracks chunk relationships and document position

## MOTIVATION
**Optimal Text Segmentation for Vector Search**

Text chunking directly impacts the quality of semantic search and retrieval. Properly sized chunks ensure that embeddings capture meaningful semantic units while staying within model token limits. The overlap strategy preserves context across boundaries, improving retrieval accuracy. This systematic chunking approach enables efficient vector storage, faster similarity searches, and more relevant results for end users querying the knowledge base.

## CONTEXT
**Token-Based Chunking for Embedding Pipeline**

Operating environment:
* Tokenizer: BAAI/bge-m3 model tokenizer
* Chunk size: 500 tokens (optimal for bge-m3 model)
* Overlap: 100 tokens for context preservation
* Strategy: Paragraph-based with smart splitting
* Input: Structured content from Docling processing
* Languages: Portuguese and English support
* Constraints: Jupyter Notebook implementation
* Output: Chunks with full metadata and relationships

## IMPLEMENTATION BLUEPRINT
**Complete Text Chunking Architecture**

### Architecture Overview
```
Cell 6: Text Chunking
├── ChunkingStrategy class
│   ├── Tokenizer initialization
│   ├── Paragraph detection
│   ├── Smart splitting
│   ├── Overlap management
│   └── Metadata preservation
├── Chunk generation
├── ID creation
└── Relationship tracking
```

### Code Structure
```python
# Cell 6: Text Chunking Functions
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import re
from collections import deque

@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    id: str
    content: str
    tokens: List[int]
    token_count: int
    document_id: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    section_title: Optional[str]
    page_numbers: List[int]
    overlap_prev: int  # Token overlap with previous chunk
    overlap_next: int  # Token overlap with next chunk
    metadata: Dict[str, Any]

class ChunkingStrategy:
    def __init__(self, 
                 model_name: str = "BAAI/bge-m3",
                 chunk_size: int = 500,
                 overlap_size: int = 100,
                 min_chunk_size: int = 100):
        """Initialize chunking strategy with tokenizer"""
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Paragraph detection patterns
        self.paragraph_patterns = [
            r'\n\n+',  # Double newlines
            r'\n(?=[A-Z])',  # Newline before capital letter
            r'\.\s+(?=[A-Z])',  # Period followed by capital
        ]
    
    def chunk_document(self, 
                      processed_doc: Dict[str, Any],
                      document_id: str) -> List[TextChunk]:
        """Chunk processed document into semantic units"""
        chunks = []
        
        # Extract structured content
        structured = processed_doc['structured_content']
        
        # Build document text with structure preservation
        document_sections = self._build_document_sections(structured)
        
        # Process each section
        for section in document_sections:
            section_chunks = self._chunk_section(
                section['text'],
                section['title'],
                section['pages'],
                document_id
            )
            chunks.extend(section_chunks)
        
        # Assign global chunk indices
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = total_chunks
        
        return chunks
    
    def _build_document_sections(self, structured: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build document sections from structured content"""
        sections = []
        current_section = {
            'title': structured.get('title', 'Document'),
            'text': '',
            'pages': set()
        }
        
        # Process by page order
        for page_data in structured['page_structure']:
            page_num = page_data['page_number']
            
            for element in page_data['elements']:
                if element['type'] == 'heading':
                    # Save current section if has content
                    if current_section['text'].strip():
                        current_section['pages'] = list(current_section['pages'])
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        'title': element['text'],
                        'text': '',
                        'pages': {page_num}
                    }
                elif element['type'] in ['paragraph', 'list']:
                    current_section['text'] += element['text'] + '\n\n'
                    current_section['pages'].add(page_num)
                elif element['type'] == 'table':
                    # Include table as text representation
                    current_section['text'] += '[TABLE]\n' + element['text'] + '\n\n'
                    current_section['pages'].add(page_num)
        
        # Add final section
        if current_section['text'].strip():
            current_section['pages'] = list(current_section['pages'])
            sections.append(current_section)
        
        return sections
    
    def _chunk_section(self,
                      text: str,
                      section_title: str,
                      page_numbers: List[int],
                      document_id: str) -> List[TextChunk]:
        """Chunk a section of text with overlap"""
        if not text.strip():
            return []
        
        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        # Tokenize paragraphs
        tokenized_paragraphs = []
        for para in paragraphs:
            tokens = self.tokenizer.encode(para, add_special_tokens=False)
            if tokens:  # Skip empty paragraphs
                tokenized_paragraphs.append({
                    'text': para,
                    'tokens': tokens,
                    'length': len(tokens)
                })
        
        # Group paragraphs into chunks
        chunks = self._group_paragraphs_into_chunks(
            tokenized_paragraphs,
            section_title,
            page_numbers,
            document_id
        )
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using multiple strategies"""
        # Try double newline split first
        paragraphs = text.split('\n\n')
        
        # Further split very long paragraphs
        final_paragraphs = []
        for para in paragraphs:
            if len(para) > 2000:  # Long paragraph, try to split
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = []
                current_len = 0
                
                for sent in sentences:
                    sent_len = len(sent)
                    if current_len + sent_len > 1500 and current:
                        final_paragraphs.append(' '.join(current))
                        current = [sent]
                        current_len = sent_len
                    else:
                        current.append(sent)
                        current_len += sent_len
                
                if current:
                    final_paragraphs.append(' '.join(current))
            else:
                if para.strip():
                    final_paragraphs.append(para)
        
        return final_paragraphs
    
    def _group_paragraphs_into_chunks(self,
                                     tokenized_paragraphs: List[Dict],
                                     section_title: str,
                                     page_numbers: List[int],
                                     document_id: str) -> List[TextChunk]:
        """Group paragraphs into chunks with overlap"""
        chunks = []
        current_chunk_tokens = []
        current_chunk_text = []
        char_position = 0
        
        for para_data in tokenized_paragraphs:
            para_tokens = para_data['tokens']
            para_text = para_data['text']
            para_length = para_data['length']
            
            # Check if adding this paragraph exceeds chunk size
            if current_chunk_tokens and len(current_chunk_tokens) + para_length > self.chunk_size:
                # Create chunk from current content
                chunk = self._create_chunk(
                    tokens=current_chunk_tokens,
                    text=' '.join(current_chunk_text),
                    document_id=document_id,
                    section_title=section_title,
                    page_numbers=page_numbers,
                    start_char=char_position - len(' '.join(current_chunk_text)),
                    end_char=char_position
                )
                chunks.append(chunk)
                
                # Prepare overlap for next chunk
                overlap_tokens = current_chunk_tokens[-self.overlap_size:] if len(current_chunk_tokens) > self.overlap_size else current_chunk_tokens
                overlap_text = self.tokenizer.decode(overlap_tokens)
                
                # Start new chunk with overlap
                current_chunk_tokens = overlap_tokens + para_tokens
                current_chunk_text = [overlap_text, para_text]
            else:
                # Add paragraph to current chunk
                current_chunk_tokens.extend(para_tokens)
                current_chunk_text.append(para_text)
            
            char_position += len(para_text) + 1
        
        # Create final chunk
        if current_chunk_tokens and len(current_chunk_tokens) >= self.min_chunk_size:
            chunk = self._create_chunk(
                tokens=current_chunk_tokens,
                text=' '.join(current_chunk_text),
                document_id=document_id,
                section_title=section_title,
                page_numbers=page_numbers,
                start_char=char_position - len(' '.join(current_chunk_text)),
                end_char=char_position
            )
            chunks.append(chunk)
        
        # Calculate overlap sizes
        for i in range(len(chunks)):
            if i > 0:
                chunks[i].overlap_prev = self._calculate_overlap(
                    chunks[i-1].tokens,
                    chunks[i].tokens
                )
            if i < len(chunks) - 1:
                chunks[i].overlap_next = self._calculate_overlap(
                    chunks[i].tokens,
                    chunks[i+1].tokens
                )
        
        return chunks
    
    def _create_chunk(self,
                     tokens: List[int],
                     text: str,
                     document_id: str,
                     section_title: str,
                     page_numbers: List[int],
                     start_char: int,
                     end_char: int) -> TextChunk:
        """Create a text chunk with metadata"""
        # Generate deterministic chunk ID
        chunk_id = self._generate_chunk_id(document_id, text)
        
        return TextChunk(
            id=chunk_id,
            content=text,
            tokens=tokens,
            token_count=len(tokens),
            document_id=document_id,
            chunk_index=0,  # Will be set later
            total_chunks=0,  # Will be set later
            start_char=start_char,
            end_char=end_char,
            section_title=section_title,
            page_numbers=page_numbers,
            overlap_prev=0,  # Will be calculated later
            overlap_next=0,  # Will be calculated later
            metadata={
                'chunking_strategy': 'paragraph-based',
                'model': self.model_name,
                'chunk_size': self.chunk_size,
                'overlap_size': self.overlap_size
            }
        )
    
    def _generate_chunk_id(self, document_id: str, content: str) -> str:
        """Generate deterministic chunk ID"""
        # Use document ID and content hash for uniqueness
        id_string = f"{document_id}:{hashlib.md5(content.encode()).hexdigest()[:8]}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    def _calculate_overlap(self, tokens1: List[int], tokens2: List[int]) -> int:
        """Calculate token overlap between two chunks"""
        if not tokens1 or not tokens2:
            return 0
        
        # Find longest common subsequence at boundaries
        overlap = 0
        min_len = min(len(tokens1), len(tokens2), self.overlap_size)
        
        for i in range(1, min_len + 1):
            if tokens1[-i:] == tokens2[:i]:
                overlap = i
        
        return overlap

class ChunkOptimizer:
    """Optimize chunk quality and boundaries"""
    
    @staticmethod
    def optimize_chunks(chunks: List[TextChunk]) -> List[TextChunk]:
        """Optimize chunk boundaries for better semantic coherence"""
        optimized = []
        
        for i, chunk in enumerate(chunks):
            # Check if chunk is too small and can be merged
            if chunk.token_count < 200 and i > 0:
                # Try to merge with previous chunk
                prev_chunk = optimized[-1] if optimized else None
                if prev_chunk and prev_chunk.token_count + chunk.token_count <= 600:
                    # Merge chunks
                    merged = ChunkOptimizer._merge_chunks(prev_chunk, chunk)
                    optimized[-1] = merged
                    continue
            
            optimized.append(chunk)
        
        return optimized
    
    @staticmethod
    def _merge_chunks(chunk1: TextChunk, chunk2: TextChunk) -> TextChunk:
        """Merge two chunks into one"""
        merged = TextChunk(
            id=chunk1.id,  # Keep first chunk's ID
            content=chunk1.content + " " + chunk2.content,
            tokens=chunk1.tokens + chunk2.tokens,
            token_count=chunk1.token_count + chunk2.token_count,
            document_id=chunk1.document_id,
            chunk_index=chunk1.chunk_index,
            total_chunks=chunk1.total_chunks,
            start_char=chunk1.start_char,
            end_char=chunk2.end_char,
            section_title=chunk1.section_title,
            page_numbers=list(set(chunk1.page_numbers + chunk2.page_numbers)),
            overlap_prev=chunk1.overlap_prev,
            overlap_next=chunk2.overlap_next,
            metadata={**chunk1.metadata, 'merged': True}
        )
        return merged

class ChunkValidator:
    """Validate chunk quality and completeness"""
    
    @staticmethod
    def validate_chunks(chunks: List[TextChunk], original_text: str) -> Dict[str, Any]:
        """Validate chunking results"""
        validation = {
            'valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Calculate statistics
        token_counts = [c.token_count for c in chunks]
        validation['statistics'] = {
            'total_chunks': len(chunks),
            'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'total_tokens': sum(token_counts)
        }
        
        # Check chunk sizes
        for i, chunk in enumerate(chunks):
            if chunk.token_count > 600:  # Allow some flexibility
                validation['issues'].append(f"Chunk {i} exceeds max size: {chunk.token_count} tokens")
            if chunk.token_count < 50:
                validation['issues'].append(f"Chunk {i} is too small: {chunk.token_count} tokens")
        
        # Check overlap consistency
        for i in range(1, len(chunks)):
            if chunks[i].overlap_prev == 0:
                validation['issues'].append(f"No overlap between chunks {i-1} and {i}")
        
        # Check coverage (simplified)
        reconstructed_length = sum(len(c.content) for c in chunks)
        original_length = len(original_text)
        coverage = reconstructed_length / original_length if original_length > 0 else 0
        
        if coverage < 0.9:
            validation['issues'].append(f"Low text coverage: {coverage:.2%}")
            validation['valid'] = False
        
        return validation
```

### Error Handling
```python
class ChunkingError(Exception):
    """Base exception for chunking errors"""
    pass

class TokenizationError(ChunkingError):
    """Raised when tokenization fails"""
    pass

class ChunkSizeError(ChunkingError):
    """Raised when chunk size constraints are violated"""
    pass

def safe_chunk_with_fallback(strategy: ChunkingStrategy, 
                            document: Dict[str, Any],
                            document_id: str) -> List[TextChunk]:
    """Chunk document with fallback strategies"""
    try:
        # Try primary strategy
        chunks = strategy.chunk_document(document, document_id)
        
        # Optimize if needed
        if any(c.token_count < 100 for c in chunks):
            chunks = ChunkOptimizer.optimize_chunks(chunks)
        
        return chunks
        
    except Exception as e:
        # Fallback to simple splitting
        print(f"Primary chunking failed, using fallback: {str(e)}")
        
        # Simple character-based splitting as fallback
        text = document.get('markdown', '')
        simple_chunks = []
        
        for i in range(0, len(text), 2000):
            chunk_text = text[i:i+2000]
            chunk = TextChunk(
                id=hashlib.md5(f"{document_id}:{i}".encode()).hexdigest()[:16],
                content=chunk_text,
                tokens=[],
                token_count=len(strategy.tokenizer.encode(chunk_text)),
                document_id=document_id,
                chunk_index=i // 2000,
                total_chunks=0,
                start_char=i,
                end_char=i + len(chunk_text),
                section_title="Document",
                page_numbers=[],
                overlap_prev=0,
                overlap_next=0,
                metadata={'fallback': True}
            )
            simple_chunks.append(chunk)
        
        return simple_chunks
```

## VALIDATION LOOP
**Comprehensive Text Chunking Testing**

### Unit Testing
```python
def test_tokenization():
    """Test tokenizer initialization and encoding"""
    strategy = ChunkingStrategy(chunk_size=500, overlap_size=100)
    
    test_text = "This is a test sentence for tokenization."
    tokens = strategy.tokenizer.encode(test_text, add_special_tokens=False)
    
    assert len(tokens) > 0
    assert len(tokens) < 50  # Reasonable token count

def test_paragraph_splitting():
    """Test paragraph detection and splitting"""
    strategy = ChunkingStrategy()
    
    text = """First paragraph here.
    
    Second paragraph with more content.
    And continues on the same paragraph.
    
    Third paragraph."""
    
    paragraphs = strategy._split_into_paragraphs(text)
    assert len(paragraphs) == 3

def test_chunk_id_generation():
    """Test deterministic chunk ID generation"""
    strategy = ChunkingStrategy()
    
    id1 = strategy._generate_chunk_id("doc1", "content")
    id2 = strategy._generate_chunk_id("doc1", "content")
    id3 = strategy._generate_chunk_id("doc1", "different")
    
    assert id1 == id2  # Same input produces same ID
    assert id1 != id3  # Different content produces different ID

def test_overlap_calculation():
    """Test overlap calculation between chunks"""
    strategy = ChunkingStrategy()
    
    tokens1 = [1, 2, 3, 4, 5]
    tokens2 = [4, 5, 6, 7, 8]
    
    overlap = strategy._calculate_overlap(tokens1, tokens2)
    assert overlap == 2
```

### Integration Testing
```python
def test_full_chunking_pipeline():
    """Test complete chunking pipeline"""
    strategy = ChunkingStrategy(chunk_size=500, overlap_size=100)
    
    # Get processed document
    processed_doc = get_sample_processed_document()
    
    chunks = strategy.chunk_document(processed_doc, "test_doc_id")
    
    assert len(chunks) > 0
    assert all(c.token_count <= 600 for c in chunks)  # Allow some flexibility
    assert all(c.token_count >= 50 for c in chunks)  # Minimum size

def test_chunk_optimization():
    """Test chunk boundary optimization"""
    strategy = ChunkingStrategy()
    
    # Create test chunks with varying sizes
    chunks = create_test_chunks_with_sizes([150, 100, 450, 80, 500])
    
    optimized = ChunkOptimizer.optimize_chunks(chunks)
    
    # Should merge small chunks
    assert len(optimized) < len(chunks)
    assert all(c.token_count >= 180 or c.token_count == 500 for c in optimized)

def test_multilingual_chunking():
    """Test chunking with Portuguese and English content"""
    strategy = ChunkingStrategy()
    
    multilingual_text = """
    This is English content that should be chunked properly.
    
    Este é conteúdo em português que também deve ser processado corretamente.
    Incluindo acentuação e caracteres especiais.
    """
    
    doc = create_document_from_text(multilingual_text)
    chunks = strategy.chunk_document(doc, "multilingual_doc")
    
    assert len(chunks) > 0
    assert all(c.content for c in chunks)
```

### Performance Testing
* Chunking speed: > 1000 chunks per second
* Memory usage: < 100MB for 1000 chunks
* Token accuracy: 100% alignment with model tokenizer
* Overlap consistency: 100% overlap preservation

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
* **Input Sanitization**: Clean text from potential injection patterns
* **Size Limits**: Enforce maximum document size to prevent DoS
* **Token Validation**: Verify token counts match actual encoding
* **ID Collision**: Use strong hashing to prevent ID collisions

### Performance Optimization
* **Batch Tokenization**: Process multiple paragraphs in batch
* **Caching**: Cache tokenization results for repeated text
* **Parallel Processing**: Chunk multiple sections concurrently
* **Memory Efficiency**: Use generators for large documents
* **Pre-compilation**: Pre-compile regex patterns

### Maintenance Requirements
* **Tokenizer Updates**: Update when model tokenizer changes
* **Strategy Tuning**: Adjust chunk sizes based on retrieval performance
* **Quality Monitoring**: Track chunk quality metrics
* **Language Support**: Add tokenizers for new languages
* **Documentation**: Maintain chunking strategy documentation