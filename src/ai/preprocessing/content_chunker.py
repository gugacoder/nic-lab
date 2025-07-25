"""
Content Chunking for LLM Processing

This module provides intelligent content chunking strategies to optimize
text for LLM consumption while preserving semantic coherence and context.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SENTENCE_AWARE = "sentence_aware"
    PARAGRAPH_AWARE = "paragraph_aware"
    SEMANTIC_AWARE = "semantic_aware"
    CODE_AWARE = "code_aware"
    MARKDOWN_AWARE = "markdown_aware"


@dataclass
class ChunkingConfig:
    """Configuration for content chunking"""
    max_chunk_size: int = 2000  # Maximum characters per chunk
    min_chunk_size: int = 100   # Minimum characters per chunk
    overlap_size: int = 200     # Overlap between chunks
    strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_AWARE
    preserve_structure: bool = True  # Preserve document structure markers
    include_metadata: bool = True    # Include chunk metadata
    merge_short_chunks: bool = True  # Merge chunks shorter than min_size
    max_chunks_per_document: int = 50  # Maximum chunks per document


@dataclass
class ContentChunk:
    """Represents a chunk of content with metadata"""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_type: str = "text"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def size(self) -> int:
        """Get chunk size in characters"""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Get approximate word count"""
        return len(self.content.split())
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to chunk"""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            'content': self.content,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'chunk_type': self.chunk_type,
            'size': self.size,
            'word_count': self.word_count,
            'metadata': self.metadata
        }


class ContentChunker:
    """Intelligent content chunking for LLM processing"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize content chunker
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        
        # Precompile regex patterns for efficiency
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.code_block_pattern = re.compile(r'```[\s\S]*?```|`[^`]*`')
        self.markdown_header_pattern = re.compile(r'^#+\s+.*$', re.MULTILINE)
        self.list_pattern = re.compile(r'^[\s]*[-*+]\s+.*$', re.MULTILINE)
    
    def chunk_content(
        self,
        content: str,
        content_type: Optional[str] = None,
        source_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContentChunk]:
        """Chunk content using the configured strategy
        
        Args:
            content: Content to chunk
            content_type: Type of content (markdown, code, text, etc.)
            source_metadata: Metadata about the source document
            
        Returns:
            List of content chunks
        """
        if not content or not content.strip():
            return []
        
        logger.debug(f"Chunking {len(content)} characters using {self.config.strategy.value} strategy")
        
        # Detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(content)
        
        # Select chunking strategy based on content type and config
        if content_type == 'markdown' or self.config.strategy == ChunkingStrategy.MARKDOWN_AWARE:
            chunks = self._chunk_markdown_aware(content)
        elif content_type == 'code' or self.config.strategy == ChunkingStrategy.CODE_AWARE:
            chunks = self._chunk_code_aware(content)
        elif self.config.strategy == ChunkingStrategy.PARAGRAPH_AWARE:
            chunks = self._chunk_paragraph_aware(content)
        elif self.config.strategy == ChunkingStrategy.SENTENCE_AWARE:
            chunks = self._chunk_sentence_aware(content)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC_AWARE:
            chunks = self._chunk_semantic_aware(content)
        else:  # FIXED_SIZE
            chunks = self._chunk_fixed_size(content)
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks, source_metadata)
        
        logger.debug(f"Created {len(chunks)} chunks")
        return chunks
    
    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content
        
        Args:
            content: Content to analyze
            
        Returns:
            Detected content type
        """
        content_lower = content.lower()
        
        # Check for markdown indicators
        if ('```' in content or content.count('#') > 2 or 
            re.search(r'\[.*\]\(.*\)', content) or 
            content.count('**') > 2):
            return 'markdown'
        
        # Check for code indicators
        if (content.count('{') + content.count('}') > 10 or
            content.count('def ') > 0 or content.count('function ') > 0 or
            content.count('class ') > 0 or content.count('import ') > 0):
            return 'code'
        
        # Check for structured text
        if content.count('\n\n') > len(content) / 500:  # Many paragraph breaks
            return 'structured_text'
        
        return 'text'
    
    def _chunk_fixed_size(self, content: str) -> List[ContentChunk]:
        """Chunk content using fixed size strategy
        
        Args:
            content: Content to chunk
            
        Returns:
            List of fixed-size chunks
        """
        chunks = []
        chunk_size = self.config.max_chunk_size
        overlap = self.config.overlap_size
        
        for i in range(0, len(content), chunk_size - overlap):
            end_pos = min(i + chunk_size, len(content))
            chunk_content = content[i:end_pos]
            
            if chunk_content.strip():
                chunk = ContentChunk(
                    content=chunk_content,
                    chunk_index=len(chunks),
                    start_char=i,
                    end_char=end_pos,
                    chunk_type="fixed"
                )
                chunks.append(chunk)
            
            if end_pos >= len(content):
                break
        
        return chunks
    
    def _chunk_sentence_aware(self, content: str) -> List[ContentChunk]:
        """Chunk content preserving sentence boundaries
        
        Args:
            content: Content to chunk
            
        Returns:
            List of sentence-aware chunks
        """
        # Split into sentences
        sentences = self.sentence_pattern.split(content)
        if not sentences:
            return [ContentChunk(content, 0, 0, len(content), "sentence")]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        current_sentences = []
        
        char_pos = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) > self.config.max_chunk_size and current_chunk:
                # Create chunk with current sentences
                chunk = ContentChunk(
                    content=current_chunk.strip(),
                    chunk_index=len(chunks),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    chunk_type="sentence"
                )
                chunk.add_metadata("sentence_count", len(current_sentences))
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) > 2 else current_sentences
                current_chunk = " ".join(overlap_sentences) + (" " + sentence if overlap_sentences else sentence)
                current_start = char_pos - len(" ".join(overlap_sentences))
                current_sentences = overlap_sentences + [sentence]
            else:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
                if not current_chunk.strip():
                    current_start = char_pos
            
            char_pos += len(sentence) + 1
        
        # Add final chunk
        if current_chunk.strip():
            chunk = ContentChunk(
                content=current_chunk.strip(),
                chunk_index=len(chunks),
                start_char=current_start,
                end_char=char_pos,
                chunk_type="sentence"
            )
            chunk.add_metadata("sentence_count", len(current_sentences))
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_paragraph_aware(self, content: str) -> List[ContentChunk]:
        """Chunk content preserving paragraph boundaries
        
        Args:
            content: Content to chunk
            
        Returns:
            List of paragraph-aware chunks
        """
        paragraphs = self.paragraph_pattern.split(content)
        chunks = []
        current_chunk = ""
        current_start = 0
        current_paragraphs = []
        
        char_pos = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if len(potential_chunk) > self.config.max_chunk_size and current_chunk:
                # Create chunk
                chunk = ContentChunk(
                    content=current_chunk.strip(),
                    chunk_index=len(chunks),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    chunk_type="paragraph"
                )
                chunk.add_metadata("paragraph_count", len(current_paragraphs))
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = paragraph
                current_start = char_pos
                current_paragraphs = [paragraph]
            else:
                current_chunk = potential_chunk
                current_paragraphs.append(paragraph)
                if not current_chunk.strip():
                    current_start = char_pos
            
            char_pos += len(paragraph) + 2  # +2 for \n\n
        
        # Add final chunk
        if current_chunk.strip():
            chunk = ContentChunk(
                content=current_chunk.strip(),
                chunk_index=len(chunks),
                start_char=current_start,
                end_char=char_pos,
                chunk_type="paragraph"
            )
            chunk.add_metadata("paragraph_count", len(current_paragraphs))
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_markdown_aware(self, content: str) -> List[ContentChunk]:
        """Chunk markdown content preserving structure
        
        Args:
            content: Markdown content to chunk
            
        Returns:
            List of markdown-aware chunks
        """
        chunks = []
        lines = content.split('\n')
        current_chunk = ""
        current_start = 0
        current_section = ""
        
        line_pos = 0
        for i, line in enumerate(lines):
            # Check for headers
            header_match = self.markdown_header_pattern.match(line)
            if header_match:
                # If we have a current chunk, finalize it
                if current_chunk.strip() and len(current_chunk) > self.config.min_chunk_size:
                    chunk = ContentChunk(
                        content=current_chunk.strip(),
                        chunk_index=len(chunks),
                        start_char=current_start,
                        end_char=line_pos,
                        chunk_type="markdown_section"
                    )
                    chunk.add_metadata("section", current_section)
                    chunks.append(chunk)
                
                # Start new section
                current_section = line.strip()
                current_chunk = line
                current_start = line_pos
            else:
                # Add line to current chunk
                current_chunk += "\n" + line
                
                # Check if chunk is getting too large
                if len(current_chunk) > self.config.max_chunk_size:
                    # Try to find a good break point
                    break_point = self._find_markdown_break_point(current_chunk)
                    if break_point > 0:
                        chunk_content = current_chunk[:break_point].strip()
                        chunk = ContentChunk(
                            content=chunk_content,
                            chunk_index=len(chunks),
                            start_char=current_start,
                            end_char=current_start + break_point,
                            chunk_type="markdown_section"
                        )
                        chunk.add_metadata("section", current_section)
                        chunks.append(chunk)
                        
                        # Continue with remainder
                        current_chunk = current_chunk[break_point:].strip()
                        current_start = current_start + break_point
            
            line_pos += len(line) + 1  # +1 for newline
        
        # Add final chunk
        if current_chunk.strip():
            chunk = ContentChunk(
                content=current_chunk.strip(),
                chunk_index=len(chunks),
                start_char=current_start,
                end_char=line_pos,
                chunk_type="markdown_section"
            )
            chunk.add_metadata("section", current_section)
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_code_aware(self, content: str) -> List[ContentChunk]:
        """Chunk code content preserving structure
        
        Args:
            content: Code content to chunk
            
        Returns:
            List of code-aware chunks
        """
        chunks = []
        lines = content.split('\n')
        current_chunk = ""
        current_start = 0
        current_function = ""
        brace_count = 0
        
        line_pos = 0
        for line in lines:
            # Detect function/class definitions
            if ('def ' in line or 'function ' in line or 'class ' in line):
                # If we have a current chunk and it's substantial, finalize it
                if current_chunk.strip() and len(current_chunk) > self.config.min_chunk_size:
                    chunk = ContentChunk(
                        content=current_chunk.strip(),
                        chunk_index=len(chunks),
                        start_char=current_start,
                        end_char=line_pos,
                        chunk_type="code_block"
                    )
                    chunk.add_metadata("function", current_function)
                    chunks.append(chunk)
                
                # Start new function block
                current_function = line.strip()
                current_chunk = line
                current_start = line_pos
                brace_count = line.count('{') - line.count('}')
            else:
                current_chunk += "\n" + line
                brace_count += line.count('{') - line.count('}')
                
                # For languages like Python, use indentation
                if brace_count == 0 and not line.strip().startswith((' ', '\t')) and current_chunk:
                    # End of function/class block
                    if len(current_chunk) > self.config.max_chunk_size:
                        chunk = ContentChunk(
                            content=current_chunk.strip(),
                            chunk_index=len(chunks),
                            start_char=current_start,
                            end_char=line_pos,
                            chunk_type="code_block"
                        )
                        chunk.add_metadata("function", current_function)
                        chunks.append(chunk)
                        
                        current_chunk = ""
                        current_start = line_pos
                        current_function = ""
            
            line_pos += len(line) + 1
        
        # Add final chunk
        if current_chunk.strip():
            chunk = ContentChunk(
                content=current_chunk.strip(),
                chunk_index=len(chunks),
                start_char=current_start,
                end_char=line_pos,
                chunk_type="code_block"
            )
            chunk.add_metadata("function", current_function)
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_semantic_aware(self, content: str) -> List[ContentChunk]:
        """Chunk content using semantic boundaries (simplified implementation)
        
        Args:
            content: Content to chunk
            
        Returns:
            List of semantically-aware chunks
        """
        # This is a simplified semantic chunking
        # A full implementation would use NLP techniques
        
        # For now, combine paragraph and sentence awareness
        paragraphs = self.paragraph_pattern.split(content)
        chunks = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            if len(paragraph) <= self.config.max_chunk_size:
                # Paragraph fits in one chunk
                chunk = ContentChunk(
                    content=paragraph.strip(),
                    chunk_index=len(chunks),
                    start_char=0,  # Would need proper tracking
                    end_char=len(paragraph),
                    chunk_type="semantic"
                )
                chunks.append(chunk)
            else:
                # Split paragraph using sentence awareness
                paragraph_chunks = self._chunk_sentence_aware(paragraph)
                chunks.extend(paragraph_chunks)
        
        return chunks
    
    def _find_markdown_break_point(self, content: str) -> int:
        """Find a good break point in markdown content
        
        Args:
            content: Markdown content
            
        Returns:
            Character position for break point
        """
        # Look for natural break points in reverse order of preference
        
        # 1. Header
        for match in re.finditer(r'\n#+\s+', content):
            if match.start() > self.config.max_chunk_size * 0.5:
                return match.start()
        
        # 2. Paragraph break
        for match in re.finditer(r'\n\s*\n', content):
            if match.start() > self.config.max_chunk_size * 0.7:
                return match.start()
        
        # 3. List item
        for match in re.finditer(r'\n[-*+]\s+', content):
            if match.start() > self.config.max_chunk_size * 0.8:
                return match.start()
        
        # 4. Sentence boundary
        for match in re.finditer(r'[.!?]\s+', content):
            if match.end() > self.config.max_chunk_size * 0.8:
                return match.end()
        
        # Default: 80% of max size
        return int(self.config.max_chunk_size * 0.8)
    
    def _post_process_chunks(
        self,
        chunks: List[ContentChunk],
        source_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContentChunk]:
        """Post-process chunks for optimization
        
        Args:
            chunks: List of chunks to process
            source_metadata: Source document metadata
            
        Returns:
            Processed chunks
        """
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        # Merge short chunks if enabled
        if self.config.merge_short_chunks:
            chunks = self._merge_short_chunks(chunks)
        
        # Add metadata and finalize
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            
            if self.config.include_metadata and source_metadata:
                for key, value in source_metadata.items():
                    chunk.add_metadata(f"source_{key}", value)
            
            # Add chunk statistics
            if self.config.include_metadata:
                chunk.add_metadata("total_chunks", len(chunks))
                chunk.add_metadata("chunk_ratio", (i + 1) / len(chunks))
                chunk.add_metadata("chunking_strategy", self.config.strategy.value)
            
            processed_chunks.append(chunk)
        
        # Limit number of chunks if configured
        if len(processed_chunks) > self.config.max_chunks_per_document:
            # Keep the most substantial chunks
            processed_chunks.sort(key=lambda x: x.size, reverse=True)
            processed_chunks = processed_chunks[:self.config.max_chunks_per_document]
            # Re-sort by original order
            processed_chunks.sort(key=lambda x: x.start_char)
            # Update indices
            for i, chunk in enumerate(processed_chunks):
                chunk.chunk_index = i
                chunk.add_metadata("total_chunks", len(processed_chunks))
        
        return processed_chunks
    
    def _merge_short_chunks(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        """Merge chunks that are shorter than minimum size
        
        Args:
            chunks: List of chunks to process
            
        Returns:
            List with short chunks merged
        """
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if chunk.size < self.config.min_chunk_size:
                if current_chunk is None:
                    current_chunk = chunk
                else:
                    # Merge with current chunk
                    merged_content = current_chunk.content + "\n\n" + chunk.content
                    current_chunk.content = merged_content
                    current_chunk.end_char = chunk.end_char
                    
                    # If merged chunk is now large enough, add it
                    if current_chunk.size >= self.config.min_chunk_size:
                        merged_chunks.append(current_chunk)
                        current_chunk = None
            else:
                # Add any pending merged chunk
                if current_chunk is not None:
                    merged_chunks.append(current_chunk)
                    current_chunk = None
                
                # Add current chunk
                merged_chunks.append(chunk)
        
        # Add any remaining merged chunk
        if current_chunk is not None:
            merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content to provide chunking recommendations
        
        Args:
            content: Content to analyze
            
        Returns:
            Analysis results with recommendations
        """
        analysis = {
            'content_length': len(content),
            'word_count': len(content.split()),
            'line_count': content.count('\n') + 1,
            'paragraph_count': len(self.paragraph_pattern.split(content)),
            'detected_type': self._detect_content_type(content),
            'estimated_chunks': math.ceil(len(content) / self.config.max_chunk_size)
        }
        
        # Provide recommendations
        recommendations = []
        
        if analysis['content_length'] > 10000:
            recommendations.append("Consider using paragraph or sentence-aware chunking for better context preservation")
        
        if analysis['detected_type'] == 'markdown':
            recommendations.append("Use markdown-aware chunking to preserve document structure")
        
        if analysis['detected_type'] == 'code':
            recommendations.append("Use code-aware chunking to keep functions/classes together")
        
        if analysis['paragraph_count'] > 20:
            recommendations.append("Paragraph-aware chunking would work well for this content")
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def test_chunking(
        self,
        content: str,
        strategy: Optional[ChunkingStrategy] = None
    ) -> Dict[str, Any]:
        """Test chunking with different strategies
        
        Args:
            content: Content to test
            strategy: Specific strategy to test
            
        Returns:
            Test results
        """
        original_strategy = self.config.strategy
        
        try:
            if strategy:
                self.config.strategy = strategy
            
            chunks = self.chunk_content(content)
            
            return {
                'success': True,
                'strategy': self.config.strategy.value,
                'chunk_count': len(chunks),
                'content_length': len(content),
                'chunks': [chunk.to_dict() for chunk in chunks[:3]],  # Sample first 3
                'stats': {
                    'avg_chunk_size': sum(c.size for c in chunks) / len(chunks) if chunks else 0,
                    'min_chunk_size': min(c.size for c in chunks) if chunks else 0,
                    'max_chunk_size': max(c.size for c in chunks) if chunks else 0,
                    'total_size': sum(c.size for c in chunks)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'strategy': self.config.strategy.value
            }
        finally:
            self.config.strategy = original_strategy


def create_content_chunker(
    max_chunk_size: int = 2000,
    overlap_size: int = 200,
    strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_AWARE,
    preserve_structure: bool = True
) -> ContentChunker:
    """Factory function to create content chunker
    
    Args:
        max_chunk_size: Maximum characters per chunk
        overlap_size: Overlap between chunks  
        strategy: Chunking strategy to use
        preserve_structure: Whether to preserve document structure
        
    Returns:
        Configured content chunker
    """
    config = ChunkingConfig(
        max_chunk_size=max_chunk_size,
        overlap_size=overlap_size,
        strategy=strategy,
        preserve_structure=preserve_structure
    )
    
    return ContentChunker(config)


if __name__ == "__main__":
    # Test content chunking
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.ai.preprocessing.content_chunker <content_file>")
        sys.exit(1)
    
    content_file = sys.argv[1]
    
    try:
        with open(content_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Testing content chunker with file: {content_file}")
        
        chunker = create_content_chunker()
        
        # Analyze content first
        analysis = chunker.analyze_content(content)
        print(f"Content analysis:")
        print(f"  Length: {analysis['content_length']} chars")
        print(f"  Type: {analysis['detected_type']}")
        print(f"  Estimated chunks: {analysis['estimated_chunks']}")
        print(f"  Recommendations: {analysis['recommendations']}")
        
        # Test chunking
        results = chunker.test_chunking(content)
        print(f"\nChunking results:")
        print(f"  Success: {results['success']}")
        if results['success']:
            print(f"  Strategy: {results['strategy']}")
            print(f"  Chunks created: {results['chunk_count']}")
            print(f"  Average chunk size: {results['stats']['avg_chunk_size']:.0f} chars")
            print(f"  Size range: {results['stats']['min_chunk_size']}-{results['stats']['max_chunk_size']} chars")
        else:
            print(f"  Error: {results['error']}")
    
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)