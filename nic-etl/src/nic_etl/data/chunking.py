import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
from pathlib import Path
import hashlib
from enum import Enum

# Optional transformers import with fallback
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create mock tokenizer for development
    class MockTokenizer:
        def encode(self, text: str) -> List[int]:
            # Rough approximation: 1 token ≈ 4 characters
            return list(range(len(text) // 4))
    
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str):
            return MockTokenizer()

class ChunkType(Enum):
    """Types of content chunks"""
    PARAGRAPH = "paragraph"
    SECTION = "section"
    LIST_ITEM = "list_item"
    TABLE_CONTENT = "table_content"
    MIXED_CONTENT = "mixed_content"

class BoundaryStrategy(Enum):
    """Chunk boundary detection strategies"""
    SENTENCE_BOUNDARY = "sentence"
    PARAGRAPH_BOUNDARY = "paragraph"
    SEMANTIC_BOUNDARY = "semantic"

@dataclass
class ChunkingConfig:
    """Configuration for text chunking operations"""
    target_chunk_size: int = 500  # tokens
    overlap_size: int = 100  # tokens
    max_chunk_size: int = 600  # hard limit
    min_chunk_size: int = 50   # minimum viable chunk
    model_name: str = "BAAI/bge-m3"
    boundary_strategy: BoundaryStrategy = BoundaryStrategy.PARAGRAPH_BOUNDARY
    preserve_structure: bool = True
    respect_semantic_boundaries: bool = True

@dataclass
class ChunkContext:
    """Context information for chunk generation"""
    document_title: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    section_hierarchy: List[str] = field(default_factory=list)
    source_element_type: str = "paragraph"
    document_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkMetadata:
    """Comprehensive metadata for text chunks"""
    chunk_id: str
    chunk_index: int
    total_chunks: int
    token_count: int
    character_count: int
    chunk_type: ChunkType
    source_page: Optional[int]
    source_section: Optional[str]
    hierarchy_path: List[str]
    overlap_start: int
    overlap_end: int
    semantic_coherence_score: float
    processing_timestamp: datetime

@dataclass
class TextChunk:
    """Individual text chunk with metadata"""
    content: str
    metadata: ChunkMetadata
    context: ChunkContext
    token_boundaries: Tuple[int, int]  # Start and end token positions
    character_boundaries: Tuple[int, int]  # Start and end character positions

@dataclass
class DocumentChunk:
    """Document-level chunk with enhanced metadata"""
    text_chunk: TextChunk
    document_metadata: Dict[str, Any]
    processing_lineage: Dict[str, Any]
    chunk_hash: str

class TextChunker:
    """Production-ready text chunking with BAAI/bge-m3 optimization"""
    
    # Sentence boundary patterns
    SENTENCE_BOUNDARIES = re.compile(r'[.!?]+\s+')
    PARAGRAPH_BOUNDARIES = re.compile(r'\n\s*\n')
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.transformers_available = TRANSFORMERS_AVAILABLE
        
        # Initialize BAAI/bge-m3 tokenizer
        try:
            if self.transformers_available:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                self.logger.info(f"Initialized tokenizer: {config.model_name}")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                self.logger.warning(f"Using mock tokenizer - transformers library not available")
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer {config.model_name}, using fallback: {e}")
            self.tokenizer = MockTokenizer()
        
        # Pre-compile regex patterns for efficiency
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for text processing"""
        self.list_pattern = re.compile(r'^\s*[-*•]\s+', re.MULTILINE)
        self.numbered_list_pattern = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
        self.whitespace_pattern = re.compile(r'\s+')
        self.heading_pattern = re.compile(r'^#+\s+', re.MULTILINE)
        
    def chunk_document(self, structured_content: Any, 
                      document_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk structured document content with metadata preservation"""
        
        document_chunks = []
        chunk_index = 0
        
        try:
            self.logger.info(f"Starting document chunking")
            
            # Process sections with hierarchy awareness
            if hasattr(structured_content, 'sections') and structured_content.sections:
                for section in structured_content.sections:
                    section_context = ChunkContext(
                        document_title=getattr(structured_content, 'title', None),
                        section_title=section.get('title', ''),
                        section_hierarchy=[section.get('title', '')],
                        source_element_type='section',
                        document_metadata=document_metadata
                    )
                    
                    # Combine section content
                    section_text = self._combine_section_content(section)
                    
                    if section_text.strip():
                        chunks = self.chunk_text(section_text, section_context)
                        
                        for chunk in chunks:
                            chunk.metadata.chunk_index = chunk_index
                            document_chunk = self._create_document_chunk(
                                chunk, document_metadata, chunk_index
                            )
                            document_chunks.append(document_chunk)
                            chunk_index += 1
            
            # Process standalone paragraphs
            if hasattr(structured_content, 'paragraphs') and structured_content.paragraphs:
                processed_sections = set()
                if hasattr(structured_content, 'sections'):
                    for section in structured_content.sections:
                        for content_item in section.get('content', []):
                            if isinstance(content_item, dict) and 'text' in content_item:
                                processed_sections.add(content_item['text'])
                
                for paragraph in structured_content.paragraphs:
                    para_text = paragraph.get('text', '')
                    
                    # Skip if already processed in sections
                    if para_text not in processed_sections:
                        para_context = ChunkContext(
                            document_title=getattr(structured_content, 'title', None),
                            page_number=paragraph.get('page_number'),
                            source_element_type='paragraph',
                            document_metadata=document_metadata
                        )
                        
                        chunks = self.chunk_text(para_text, para_context)
                        
                        for chunk in chunks:
                            chunk.metadata.chunk_index = chunk_index
                            document_chunk = self._create_document_chunk(
                                chunk, document_metadata, chunk_index
                            )
                            document_chunks.append(document_chunk)
                            chunk_index += 1
            
            # Process lists
            if hasattr(structured_content, 'lists') and structured_content.lists:
                list_chunks = self._process_lists(
                    structured_content.lists, document_metadata, chunk_index
                )
                document_chunks.extend(list_chunks)
                chunk_index += len(list_chunks)
            
            # Process tables
            if hasattr(structured_content, 'tables') and structured_content.tables:
                table_chunks = self._process_tables(
                    structured_content.tables, document_metadata, chunk_index
                )
                document_chunks.extend(table_chunks)
                chunk_index += len(table_chunks)
            
            # If no structured content found, process as plain text
            if not document_chunks and hasattr(structured_content, 'paragraphs'):
                all_text = ""
                for paragraph in structured_content.paragraphs:
                    all_text += paragraph.get('text', '') + "\n\n"
                
                if all_text.strip():
                    plain_context = ChunkContext(
                        document_title=getattr(structured_content, 'title', None),
                        source_element_type='document',
                        document_metadata=document_metadata
                    )
                    
                    chunks = self.chunk_text(all_text.strip(), plain_context)
                    
                    for chunk in chunks:
                        chunk.metadata.chunk_index = chunk_index
                        document_chunk = self._create_document_chunk(
                            chunk, document_metadata, chunk_index
                        )
                        document_chunks.append(document_chunk)
                        chunk_index += 1
            
            # Update total chunk counts
            total_chunks = len(document_chunks)
            for doc_chunk in document_chunks:
                doc_chunk.text_chunk.metadata.total_chunks = total_chunks
            
            self.logger.info(f"Generated {total_chunks} chunks from document")
            return document_chunks
            
        except Exception as e:
            self.logger.error(f"Document chunking failed: {e}")
            raise ChunkingError(f"Document chunking failed: {e}")
    
    def chunk_text(self, text: str, context: ChunkContext) -> List[TextChunk]:
        """Chunk text with semantic boundary preservation"""
        
        if not text or not text.strip():
            return []
        
        try:
            # Normalize whitespace
            text = self.whitespace_pattern.sub(' ', text.strip())
            
            # Calculate token count for entire text
            total_tokens = len(self.safe_tokenize(text))
            
            # If text fits in single chunk, return as-is
            if total_tokens <= self.config.target_chunk_size:
                return [self._create_single_chunk(text, context, 0)]
            
            # Find optimal chunk boundaries
            boundaries = self.calculate_optimal_boundaries(text, self.config.target_chunk_size)
            
            chunks = []
            for i, (start, end) in enumerate(boundaries):
                chunk_text = text[start:end]
                
                # Calculate overlap for non-first chunks
                overlap_start = start
                if i > 0 and self.config.overlap_size > 0:
                    overlap_start = self._calculate_overlap_start(text, start, self.config.overlap_size)
                    if overlap_start < start:
                        chunk_text = text[overlap_start:end]
                
                # Calculate overlap for non-last chunks
                overlap_end = end
                if i < len(boundaries) - 1 and self.config.overlap_size > 0:
                    overlap_end = min(len(text), end + self._calculate_overlap_chars(self.config.overlap_size))
                    chunk_text = text[overlap_start:overlap_end]
                
                chunk = self._create_text_chunk(
                    chunk_text, context, i, 
                    (overlap_start, end), (start - overlap_start, end - overlap_start)
                )
                chunks.append(chunk)
            
            # Apply semantic coherence optimization
            if self.config.respect_semantic_boundaries:
                chunks = self.preserve_semantic_coherence(chunks)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Text chunking failed: {e}")
            # Return single chunk as fallback
            return [self._create_single_chunk(text, context, 0)]
    
    def calculate_optimal_boundaries(self, text: str, max_tokens: int) -> List[Tuple[int, int]]:
        """Calculate optimal chunk boundaries respecting semantic structure"""
        
        boundaries = []
        current_start = 0
        
        while current_start < len(text):
            # Find the optimal end position
            optimal_end = self._find_optimal_boundary(
                text, current_start, max_tokens
            )
            
            boundaries.append((current_start, optimal_end))
            
            # Move to next chunk start (accounting for overlap)
            next_start = optimal_end
            if self.config.overlap_size > 0 and len(boundaries) > 0:
                overlap_chars = self._calculate_overlap_chars(self.config.overlap_size)
                next_start = max(current_start + 1, optimal_end - overlap_chars)
            
            current_start = next_start
            
            if current_start >= optimal_end:
                break
        
        return boundaries
    
    def _find_optimal_boundary(self, text: str, start: int, max_tokens: int) -> int:
        """Find optimal boundary respecting semantic structure"""
        
        if start >= len(text):
            return len(text)
        
        # Calculate rough character estimate for target tokens
        remaining_text = text[start:]
        if not remaining_text:
            return len(text)
        
        # Use rough approximation: 1 token ≈ 4 characters for initial estimate
        chars_per_token = 4
        target_chars = min(max_tokens * chars_per_token, len(remaining_text))
        
        # Initial rough position
        rough_end = start + target_chars
        rough_end = min(rough_end, len(text))
        
        # Fine-tune with actual tokenization
        search_window = target_chars // 4
        search_start = max(start + target_chars // 2, start + 10)
        search_end = min(len(text), rough_end + search_window)
        
        best_position = rough_end
        best_score = 0
        
        # Search for optimal boundary
        search_positions = list(range(search_start, search_end, max(1, search_window // 10)))
        search_positions.append(search_end)  # Always include the end
        
        for pos in search_positions:
            if pos <= start:
                continue
                
            test_text = text[start:pos]
            if not test_text.strip():
                continue
                
            token_count = len(self.safe_tokenize(test_text))
            
            if token_count <= max_tokens:
                # Calculate boundary quality score
                boundary_score = self._calculate_boundary_score(text, pos, token_count, max_tokens)
                
                if boundary_score > best_score:
                    best_position = pos
                    best_score = boundary_score
            elif token_count > max_tokens:
                # If we exceed tokens, try to find a good boundary before this position
                for backup_pos in range(pos - search_window, pos, max(1, search_window // 5)):
                    if backup_pos <= start:
                        continue
                    backup_text = text[start:backup_pos]
                    backup_tokens = len(self.safe_tokenize(backup_text))
                    if backup_tokens <= max_tokens:
                        backup_score = self._calculate_boundary_score(text, backup_pos, backup_tokens, max_tokens)
                        if backup_score > best_score:
                            best_position = backup_pos
                            best_score = backup_score
                        break
                break
        
        # Ensure we don't go backwards
        best_position = max(best_position, start + 1)
        return min(best_position, len(text))
    
    def _calculate_boundary_score(self, text: str, position: int, token_count: int, target_tokens: int) -> float:
        """Calculate quality score for a potential boundary"""
        
        score = 0.0
        
        # Prefer positions closer to target token count
        token_distance = abs(token_count - target_tokens)
        token_score = 1.0 - (token_distance / target_tokens)
        score += token_score * 0.3
        
        # Check for good semantic boundaries
        if self._is_good_boundary(text, position):
            score += 0.7
        
        # Penalize positions that cut through words
        if position < len(text) and position > 0:
            if text[position-1].isalnum() and text[position].isalnum():
                score -= 0.5
        
        return score
    
    def _is_good_boundary(self, text: str, position: int) -> bool:
        """Check if position is a good semantic boundary"""
        
        if position >= len(text) or position <= 0:
            return True
        
        # Check for paragraph boundaries (double newline)
        if position >= 2 and text[position-2:position] == '\n\n':
            return True
        
        # Check for sentence boundaries
        if position >= 2 and text[position-2:position] in ['. ', '! ', '? ']:
            return True
        
        # Check for heading boundaries (# at start of line)
        if position > 0 and text[position-1] == '\n' and position < len(text):
            line_ahead = text[position:position+10]
            if self.heading_pattern.match(line_ahead):
                return True
        
        # Check for list item boundaries
        if position > 0 and text[position-1] == '\n':
            line_ahead = text[position:position+20]
            if self.list_pattern.match(line_ahead) or self.numbered_list_pattern.match(line_ahead):
                return True
        
        # Check for natural line breaks
        if position > 0 and text[position-1] == '\n':
            return True
        
        return False
    
    def _create_text_chunk(self, content: str, context: ChunkContext, 
                          index: int, char_bounds: Tuple[int, int], 
                          overlap_bounds: Tuple[int, int]) -> TextChunk:
        """Create text chunk with comprehensive metadata"""
        
        token_count = len(self.safe_tokenize(content))
        
        # Calculate semantic coherence score
        coherence_score = self._calculate_coherence_score(content)
        
        # Generate unique chunk ID
        chunk_id_source = f"{context.document_title or 'doc'}_{index}_{content[:50]}"
        chunk_id = hashlib.md5(chunk_id_source.encode()).hexdigest()[:16]
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            chunk_index=index,
            total_chunks=0,  # Will be updated later
            token_count=token_count,
            character_count=len(content),
            chunk_type=self._determine_chunk_type(content),
            source_page=context.page_number,
            source_section=context.section_title,
            hierarchy_path=context.section_hierarchy.copy(),
            overlap_start=overlap_bounds[0],
            overlap_end=overlap_bounds[1],
            semantic_coherence_score=coherence_score,
            processing_timestamp=datetime.utcnow()
        )
        
        return TextChunk(
            content=content,
            metadata=metadata,
            context=context,
            token_boundaries=(0, token_count),  # Relative to chunk
            character_boundaries=char_bounds
        )
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate semantic coherence score for chunk"""
        
        score = 1.0
        text_stripped = text.strip()
        
        if not text_stripped:
            return 0.0
        
        # Check if chunk starts with capital letter (good sentence start)
        if text_stripped[0].isupper():
            score += 0.1
        else:
            score -= 0.2
        
        # Check if chunk ends with proper punctuation
        if text_stripped[-1] in '.!?':
            score += 0.1
        else:
            score -= 0.1
        
        # Reward complete paragraphs (presence of multiple sentences)
        sentence_count = len(self.SENTENCE_BOUNDARIES.findall(text))
        if sentence_count >= 2:
            score += 0.2
        
        # Reward paragraph breaks (suggests complete thoughts)
        if '\n\n' in text:
            score += 0.1
        
        # Penalize very short chunks
        if len(text_stripped) < 50:
            score -= 0.3
        elif len(text_stripped) < 100:
            score -= 0.1
        
        # Penalize chunks that appear to cut through sentences
        if '...' in text or text.count('.') != text.count('! ') + text.count('? ') + text.count('. '):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _determine_chunk_type(self, content: str) -> ChunkType:
        """Determine the type of content in the chunk"""
        
        # Check for list content
        if self.list_pattern.search(content) or self.numbered_list_pattern.search(content):
            return ChunkType.LIST_ITEM
        
        # Check for table content (simple heuristic)
        if '|' in content and content.count('|') > 4:
            return ChunkType.TABLE_CONTENT
        
        # Check for heading content
        if self.heading_pattern.search(content):
            return ChunkType.SECTION
        
        # Check for mixed content (multiple paragraphs)
        if content.count('\n\n') >= 2:
            return ChunkType.MIXED_CONTENT
        
        # Default to paragraph
        return ChunkType.PARAGRAPH
    
    def _create_document_chunk(self, text_chunk: TextChunk, 
                              document_metadata: Dict[str, Any], 
                              global_index: int) -> DocumentChunk:
        """Create document chunk with processing lineage"""
        
        # Create chunk content with context
        chunk_content_parts = []
        
        if text_chunk.context.document_title:
            chunk_content_parts.append(text_chunk.context.document_title)
        
        if text_chunk.context.section_title:
            chunk_content_parts.append(text_chunk.context.section_title)
        
        chunk_content_parts.append(text_chunk.content)
        
        chunk_content = " - ".join(chunk_content_parts)
        chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()
        
        processing_lineage = {
            'chunking_config': {
                'target_size': self.config.target_chunk_size,
                'overlap_size': self.config.overlap_size,
                'strategy': self.config.boundary_strategy.value,
                'preserve_structure': self.config.preserve_structure
            },
            'tokenizer_model': self.config.model_name,
            'tokenizer_available': self.transformers_available,
            'processing_timestamp': datetime.utcnow().isoformat(),
            'global_chunk_index': global_index,
            'chunk_type': text_chunk.metadata.chunk_type.value
        }
        
        return DocumentChunk(
            text_chunk=text_chunk,
            document_metadata=document_metadata,
            processing_lineage=processing_lineage,
            chunk_hash=chunk_hash
        )
    
    def preserve_semantic_coherence(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Optimize chunks for semantic coherence"""
        
        if len(chunks) <= 1:
            return chunks
        
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Check if chunk should be merged with previous chunk
            if (i > 0 and 
                chunk.metadata.token_count < self.config.min_chunk_size and
                optimized_chunks and
                optimized_chunks[-1].metadata.token_count + chunk.metadata.token_count <= self.config.max_chunk_size):
                
                # Merge with previous chunk
                merged_content = optimized_chunks[-1].content + " " + chunk.content
                
                # Use the previous chunk's context but update the boundaries
                merged_chunk = self._create_text_chunk(
                    merged_content, 
                    optimized_chunks[-1].context, 
                    optimized_chunks[-1].metadata.chunk_index,
                    optimized_chunks[-1].character_boundaries, 
                    (0, len(merged_content))
                )
                optimized_chunks[-1] = merged_chunk
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _combine_section_content(self, section: Dict[str, Any]) -> str:
        """Combine section content respecting structure"""
        
        content_parts = []
        
        # Add section title
        title = section.get('title', '')
        if title:
            content_parts.append(title)
        
        # Add section content
        for content_item in section.get('content', []):
            if isinstance(content_item, dict) and 'text' in content_item:
                content_parts.append(content_item['text'])
            elif isinstance(content_item, str):
                content_parts.append(content_item)
        
        return '\n\n'.join(filter(None, content_parts))
    
    def _process_lists(self, lists: List[Dict[str, Any]], 
                      document_metadata: Dict[str, Any], 
                      start_index: int) -> List[DocumentChunk]:
        """Process list content into chunks"""
        
        list_chunks = []
        current_index = start_index
        
        for list_item in lists:
            # Convert list to text representation
            list_text = self._serialize_list(list_item)
            
            if list_text.strip():
                context = ChunkContext(
                    source_element_type='list',
                    document_metadata=document_metadata
                )
                
                chunks = self.chunk_text(list_text, context)
                
                for chunk in chunks:
                    chunk.metadata.chunk_index = current_index
                    doc_chunk = self._create_document_chunk(chunk, document_metadata, current_index)
                    list_chunks.append(doc_chunk)
                    current_index += 1
        
        return list_chunks
    
    def _process_tables(self, tables: List[Dict[str, Any]], 
                       document_metadata: Dict[str, Any], 
                       start_index: int) -> List[DocumentChunk]:
        """Process table content into chunks"""
        
        table_chunks = []
        current_index = start_index
        
        for table in tables:
            # Convert table to text representation
            table_text = self._serialize_table(table)
            
            if table_text.strip():
                context = ChunkContext(
                    source_element_type='table',
                    page_number=table.get('page_number'),
                    document_metadata=document_metadata
                )
                
                chunks = self.chunk_text(table_text, context)
                
                for chunk in chunks:
                    chunk.metadata.chunk_index = current_index
                    doc_chunk = self._create_document_chunk(chunk, document_metadata, current_index)
                    table_chunks.append(doc_chunk)
                    current_index += 1
        
        return table_chunks
    
    def _serialize_list(self, list_item: Dict[str, Any]) -> str:
        """Convert list structure to text"""
        
        if isinstance(list_item, dict):
            if 'item' in list_item:
                list_type = list_item.get('type', 'unordered')
                prefix = "• " if list_type == 'unordered' else "1. "
                return f"{prefix}{list_item['item']}"
            elif 'text' in list_item:
                return str(list_item['text'])
            else:
                return str(list_item)
        
        return str(list_item)
    
    def _serialize_table(self, table: Dict[str, Any]) -> str:
        """Convert table structure to text"""
        
        if isinstance(table, dict) and 'data' in table:
            try:
                rows = []
                for row in table['data']:
                    if isinstance(row, list):
                        row_text = ' | '.join(str(cell) for cell in row)
                        rows.append(row_text)
                    else:
                        rows.append(str(row))
                return '\n'.join(rows)
            except Exception:
                return str(table)
        
        return str(table)
    
    def _create_single_chunk(self, text: str, context: ChunkContext, index: int) -> TextChunk:
        """Create single chunk for short text"""
        return self._create_text_chunk(
            text, context, index, (0, len(text)), (0, len(text))
        )
    
    def _calculate_overlap_start(self, text: str, position: int, overlap_tokens: int) -> int:
        """Calculate optimal overlap start position"""
        overlap_chars = self._calculate_overlap_chars(overlap_tokens)
        return max(0, position - overlap_chars)
    
    def _calculate_overlap_chars(self, overlap_tokens: int) -> int:
        """Calculate character count for overlap tokens"""
        # Rough approximation: 1 token ≈ 4 characters
        return overlap_tokens * 4
    
    def safe_tokenize(self, text: str) -> List[int]:
        """Safe tokenization with fallback"""
        try:
            return self.tokenizer.encode(text)
        except Exception as e:
            self.logger.warning(f"Tokenization failed, using character count estimate: {e}")
            # Fallback: rough approximation
            return list(range(len(text) // 4))
    
    def get_chunking_statistics(self) -> Dict[str, Any]:
        """Get current chunking capabilities and statistics"""
        return {
            'transformers_available': self.transformers_available,
            'model_name': self.config.model_name,
            'target_chunk_size': self.config.target_chunk_size,
            'overlap_size': self.config.overlap_size,
            'boundary_strategy': self.config.boundary_strategy.value,
            'preserve_structure': self.config.preserve_structure,
            'respect_semantic_boundaries': self.config.respect_semantic_boundaries,
            'supported_chunk_types': [ct.value for ct in ChunkType]
        }

def create_text_chunker(config_dict: Dict[str, Any]) -> TextChunker:
    """Factory function for text chunker creation"""
    
    # Convert string enum values to actual enums if needed
    if 'boundary_strategy' in config_dict:
        if isinstance(config_dict['boundary_strategy'], str):
            config_dict['boundary_strategy'] = BoundaryStrategy(config_dict['boundary_strategy'])
    
    config = ChunkingConfig(
        target_chunk_size=config_dict.get('target_chunk_size', 500),
        overlap_size=config_dict.get('overlap_size', 100),
        max_chunk_size=config_dict.get('max_chunk_size', 600),
        min_chunk_size=config_dict.get('min_chunk_size', 50),
        model_name=config_dict.get('model_name', 'BAAI/bge-m3'),
        boundary_strategy=config_dict.get('boundary_strategy', BoundaryStrategy.PARAGRAPH_BOUNDARY),
        preserve_structure=config_dict.get('preserve_structure', True),
        respect_semantic_boundaries=config_dict.get('respect_semantic_boundaries', True)
    )
    
    return TextChunker(config)

# Context manager for text chunking
class TextChunkingContext:
    """Context manager for text chunking with automatic cleanup"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config_dict = config_dict
        self.chunker = None
    
    def __enter__(self) -> TextChunker:
        self.chunker = create_text_chunker(self.config_dict)
        return self.chunker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass

# Error classes
class ChunkingError(Exception):
    """Base exception for text chunking errors"""
    pass

class TokenizationError(ChunkingError):
    """Tokenizer-related errors"""
    pass

class BoundaryDetectionError(ChunkingError):
    """Chunk boundary detection errors"""
    pass

class SemanticCoherenceError(ChunkingError):
    """Semantic coherence optimization errors"""
    pass

# Utility functions
def safe_tokenize(tokenizer, text: str) -> List[int]:
    """Safe tokenization with fallback"""
    try:
        return tokenizer.encode(text)
    except Exception as e:
        logging.warning(f"Tokenization failed, using character count estimate: {e}")
        return list(range(len(text) // 4))  # Rough estimate

def calculate_chunk_overlap(chunk1: TextChunk, chunk2: TextChunk) -> int:
    """Calculate overlap between two chunks in characters"""
    
    if chunk1.metadata.chunk_index >= chunk2.metadata.chunk_index:
        return 0
    
    chunk1_end = chunk1.character_boundaries[1]
    chunk2_start = chunk2.character_boundaries[0]
    
    if chunk1_end > chunk2_start:
        return chunk1_end - chunk2_start
    
    return 0

def validate_chunk_quality(chunks: List[TextChunk], config: ChunkingConfig) -> Dict[str, Any]:
    """Validate chunk quality against configuration"""
    
    validation_results = {
        'total_chunks': len(chunks),
        'valid_chunks': 0,
        'oversized_chunks': 0,
        'undersized_chunks': 0,
        'average_coherence': 0.0,
        'token_distribution': {'min': 0, 'max': 0, 'avg': 0.0},
        'issues': []
    }
    
    if not chunks:
        validation_results['issues'].append("No chunks generated")
        return validation_results
    
    token_counts = []
    coherence_scores = []
    
    for chunk in chunks:
        token_count = chunk.metadata.token_count
        token_counts.append(token_count)
        coherence_scores.append(chunk.metadata.semantic_coherence_score)
        
        if token_count <= config.max_chunk_size:
            validation_results['valid_chunks'] += 1
        else:
            validation_results['oversized_chunks'] += 1
            validation_results['issues'].append(f"Chunk {chunk.metadata.chunk_index} exceeds max size: {token_count} > {config.max_chunk_size}")
        
        if token_count < config.min_chunk_size:
            validation_results['undersized_chunks'] += 1
            validation_results['issues'].append(f"Chunk {chunk.metadata.chunk_index} below min size: {token_count} < {config.min_chunk_size}")
    
    # Calculate statistics
    validation_results['average_coherence'] = sum(coherence_scores) / len(coherence_scores)
    validation_results['token_distribution'] = {
        'min': min(token_counts),
        'max': max(token_counts),
        'avg': sum(token_counts) / len(token_counts)
    }
    
    return validation_results