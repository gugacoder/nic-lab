"""
Chunk Overlap Manager

This module manages intelligent chunk overlap to ensure context continuity
across chunks while minimizing redundancy and optimizing for retrieval.
"""

import logging
import re
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OverlapStrategy:
    """Configuration for overlap strategy"""
    strategy_type: str = "adaptive"    # adaptive, fixed, semantic, structural
    base_overlap_ratio: float = 0.1    # Base overlap as ratio of chunk size
    min_overlap_chars: int = 100       # Minimum overlap in characters
    max_overlap_chars: int = 300       # Maximum overlap in characters
    preserve_sentences: bool = True    # Ensure overlap preserves sentence boundaries
    preserve_paragraphs: bool = True   # Prefer paragraph boundaries for overlap
    context_window: int = 50           # Characters to analyze for overlap quality


class OverlapManager:
    """
    Manages intelligent overlap between content chunks to ensure
    proper context continuity for optimal retrieval and assembly.
    """
    
    def __init__(self, config):
        """Initialize overlap manager
        
        Args:
            config: ChunkingConfig instance
        """
        self.config = config
        
        # Calculate overlap parameters
        self.overlap_strategy = OverlapStrategy(
            base_overlap_ratio=config.overlap_ratio,
            min_overlap_chars=config.min_overlap_size,
            max_overlap_chars=config.max_overlap_size
        )
        
        # Regex patterns for boundary detection
        self.sentence_boundary = re.compile(r'[.!?]+\s+')
        self.paragraph_boundary = re.compile(r'\n\s*\n')
        self.word_boundary = re.compile(r'\b')
        
        logger.debug("OverlapManager initialized")
    
    async def apply_overlap(
        self, 
        initial_chunks: List, 
        original_content: str,
        doc_structure
    ) -> List:
        """
        Apply intelligent overlap to chunks
        
        Args:
            initial_chunks: List of ContentChunk objects without overlap
            original_content: Original document content
            doc_structure: DocumentStructure instance
            
        Returns:
            List of chunks with optimized overlap applied
        """
        if not initial_chunks or len(initial_chunks) <= 1:
            return initial_chunks
        
        logger.debug(f"Applying overlap to {len(initial_chunks)} chunks")
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(initial_chunks):
            if i == 0:
                # First chunk - no leading overlap needed
                overlapped_chunks.append(chunk)
                continue
            
            # Calculate optimal overlap with previous chunk
            previous_chunk = overlapped_chunks[i - 1]
            overlap_content = await self._calculate_optimal_overlap(
                previous_chunk, chunk, original_content, doc_structure
            )
            
            # Create new chunk with overlap
            overlapped_chunk = await self._apply_overlap_to_chunk(
                chunk, overlap_content, original_content
            )
            
            overlapped_chunks.append(overlapped_chunk)
        
        # Validate and adjust overlaps
        final_chunks = await self._validate_overlaps(
            overlapped_chunks, original_content
        )
        
        logger.debug(f"Applied overlap to {len(final_chunks)} chunks")
        return final_chunks
    
    async def _calculate_optimal_overlap(
        self, 
        previous_chunk, 
        current_chunk, 
        original_content: str,
        doc_structure
    ) -> str:
        """Calculate optimal overlap content between two chunks"""
        
        # Determine overlap size based on chunk characteristics
        target_overlap_size = self._calculate_target_overlap_size(
            previous_chunk, current_chunk
        )
        
        # Find the best overlap boundary in the previous chunk
        overlap_start_pos = max(
            previous_chunk.end_char - target_overlap_size,
            previous_chunk.start_char
        )
        
        # Find the best boundary position
        optimal_start = await self._find_optimal_overlap_start(
            original_content, overlap_start_pos, previous_chunk.end_char,
            target_overlap_size
        )
        
        # Extract overlap content
        overlap_content = original_content[optimal_start:previous_chunk.end_char]
        
        # Ensure overlap quality
        overlap_content = self._ensure_overlap_quality(
            overlap_content, original_content, optimal_start
        )
        
        return overlap_content
    
    def _calculate_target_overlap_size(self, previous_chunk, current_chunk) -> int:
        """Calculate target overlap size based on chunk characteristics"""
        
        # Base overlap from configuration
        base_size = min(
            int(previous_chunk.size * self.overlap_strategy.base_overlap_ratio),
            int(current_chunk.size * self.overlap_strategy.base_overlap_ratio)
        )
        
        # Adjust based on chunk types
        if previous_chunk.chunk_type == 'code' or current_chunk.chunk_type == 'code':
            # Smaller overlap for code chunks
            base_size = int(base_size * 0.7)
        elif previous_chunk.chunk_type == 'table' or current_chunk.chunk_type == 'table':
            # Minimal overlap for tables
            base_size = int(base_size * 0.5)
        elif (previous_chunk.section_title and current_chunk.section_title and
              previous_chunk.section_title != current_chunk.section_title):
            # Different sections - smaller overlap
            base_size = int(base_size * 0.8)
        
        # Apply bounds
        target_size = max(
            self.overlap_strategy.min_overlap_chars,
            min(base_size, self.overlap_strategy.max_overlap_chars)
        )
        
        return target_size
    
    async def _find_optimal_overlap_start(
        self, 
        content: str, 
        start_pos: int, 
        end_pos: int,
        target_size: int
    ) -> int:
        """Find the optimal starting position for overlap"""
        
        # Search for the best boundary within the target range
        search_start = max(0, end_pos - target_size - 50)
        search_end = min(len(content), end_pos)
        search_content = content[search_start:search_end]
        
        # Find boundary candidates
        candidates = []
        
        # Paragraph boundaries (highest priority)
        if self.overlap_strategy.preserve_paragraphs:
            for match in self.paragraph_boundary.finditer(search_content):
                abs_pos = search_start + match.end()
                if search_start < abs_pos < search_end:
                    distance_from_target = abs(abs_pos - (end_pos - target_size))
                    candidates.append({
                        'position': abs_pos,
                        'type': 'paragraph',
                        'priority': 1,
                        'distance': distance_from_target
                    })
        
        # Sentence boundaries (medium priority)
        if self.overlap_strategy.preserve_sentences:
            for match in self.sentence_boundary.finditer(search_content):
                abs_pos = search_start + match.end()
                if search_start < abs_pos < search_end:
                    distance_from_target = abs(abs_pos - (end_pos - target_size))
                    candidates.append({
                        'position': abs_pos,
                        'type': 'sentence',
                        'priority': 2,
                        'distance': distance_from_target
                    })
        
        # Word boundaries (low priority)
        for match in self.word_boundary.finditer(search_content):
            abs_pos = search_start + match.start()
            if (search_start < abs_pos < search_end and 
                content[abs_pos:abs_pos+1].isalpha()):
                distance_from_target = abs(abs_pos - (end_pos - target_size))
                candidates.append({
                    'position': abs_pos,
                    'type': 'word',
                    'priority': 3,
                    'distance': distance_from_target
                })
        
        # Select best candidate
        if candidates:
            # Sort by priority first, then by distance from target
            candidates.sort(key=lambda x: (x['priority'], x['distance']))
            return candidates[0]['position']
        
        # Fallback to target position
        return max(search_start, end_pos - target_size)
    
    def _ensure_overlap_quality(
        self, 
        overlap_content: str, 
        original_content: str, 
        overlap_start: int
    ) -> str:
        """Ensure overlap content quality and coherence"""
        
        if not overlap_content or not overlap_content.strip():
            return overlap_content
        
        # Remove leading/trailing whitespace while preserving structure
        cleaned_overlap = overlap_content.strip()
        
        # Ensure overlap doesn't start mid-word
        if (overlap_start > 0 and 
            not original_content[overlap_start - 1].isspace() and
            not original_content[overlap_start].isspace()):
            
            # Find the next word boundary
            next_space = cleaned_overlap.find(' ')
            if next_space > 0:
                cleaned_overlap = cleaned_overlap[next_space + 1:]
        
        # Ensure overlap ends at a reasonable boundary
        if not cleaned_overlap.endswith(('.', '!', '?', '\n', ':')):
            # Try to end at sentence boundary
            last_sentence_end = max(
                cleaned_overlap.rfind('.'),
                cleaned_overlap.rfind('!'),
                cleaned_overlap.rfind('?')
            )
            
            if last_sentence_end > len(cleaned_overlap) * 0.5:
                cleaned_overlap = cleaned_overlap[:last_sentence_end + 1]
        
        # Ensure minimum meaningful length
        if len(cleaned_overlap) < self.overlap_strategy.min_overlap_chars // 2:
            return overlap_content  # Return original if cleaning made it too short
        
        return cleaned_overlap
    
    async def _apply_overlap_to_chunk(
        self, 
        chunk, 
        overlap_content: str, 
        original_content: str
    ):
        """Apply overlap content to a chunk"""
        
        if not overlap_content:
            return chunk
        
        # Prepend overlap to chunk content
        new_content = overlap_content + '\n\n' + chunk.content
        
        # Update chunk properties
        overlap_chars = len(overlap_content) + 2  # +2 for \n\n
        new_start_char = chunk.start_char - overlap_chars
        
        # Create new chunk with overlap
        chunk.content = new_content
        chunk.start_char = new_start_char
        
        # Update metadata
        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
            chunk.metadata = {}
        
        chunk.metadata.update({
            'has_overlap': True,
            'overlap_size': len(overlap_content),
            'overlap_type': 'leading'
        })
        
        return chunk
    
    async def _validate_overlaps(
        self, 
        chunks: List, 
        original_content: str
    ) -> List:
        """Validate and adjust overlaps to ensure quality"""
        
        validated_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Check for excessive overlap
            if (hasattr(chunk, 'metadata') and 
                chunk.metadata.get('has_overlap', False)):
                
                overlap_size = chunk.metadata.get('overlap_size', 0)
                
                # Check if overlap is too large relative to chunk
                if overlap_size > chunk.size * 0.5:
                    # Reduce overlap
                    chunk = await self._reduce_chunk_overlap(chunk, overlap_size // 2)
            
            # Check for duplicate content between adjacent chunks
            if i > 0:
                chunk = await self._deduplicate_adjacent_chunks(
                    validated_chunks[i - 1], chunk
                )
            
            validated_chunks.append(chunk)
        
        return validated_chunks
    
    async def _reduce_chunk_overlap(self, chunk, target_overlap_size: int):
        """Reduce overlap size for a chunk"""
        
        if not chunk.metadata.get('has_overlap', False):
            return chunk
        
        current_overlap_size = chunk.metadata.get('overlap_size', 0)
        if current_overlap_size <= target_overlap_size:
            return chunk
        
        # Find where to cut the overlap
        content_lines = chunk.content.split('\n')
        overlap_lines = []
        remaining_lines = []
        current_size = 0
        
        for line in content_lines:
            if current_size < target_overlap_size:
                overlap_lines.append(line)
                current_size += len(line) + 1  # +1 for newline
            else:
                remaining_lines.append(line)
        
        # Reconstruct content
        if remaining_lines:
            chunk.content = '\n'.join(overlap_lines + remaining_lines)
            chunk.metadata['overlap_size'] = current_size - 1  # -1 for last newline
        
        return chunk
    
    async def _deduplicate_adjacent_chunks(self, previous_chunk, current_chunk):
        """Remove duplicate content between adjacent chunks"""
        
        if not current_chunk.metadata.get('has_overlap', False):
            return current_chunk
        
        # Get the tail of previous chunk
        prev_tail = previous_chunk.content[-200:] if len(previous_chunk.content) > 200 else previous_chunk.content
        
        # Get the head of current chunk
        curr_head = current_chunk.content[:200] if len(current_chunk.content) > 200 else current_chunk.content
        
        # Find common substring
        common_length = self._find_common_suffix_prefix(prev_tail, curr_head)
        
        if common_length > 50:  # Significant duplication
            # Remove duplication from current chunk
            chunk_lines = current_chunk.content.split('\n')
            duplicate_chars = 0
            deduplicated_lines = []
            
            for line in chunk_lines:
                if duplicate_chars < common_length:
                    duplicate_chars += len(line) + 1
                else:
                    deduplicated_lines.append(line)
            
            if deduplicated_lines:
                current_chunk.content = '\n'.join(deduplicated_lines)
                current_chunk.metadata['deduplication_applied'] = True
                current_chunk.metadata['removed_chars'] = duplicate_chars
        
        return current_chunk
    
    def _find_common_suffix_prefix(self, text1: str, text2: str) -> int:
        """Find length of common suffix of text1 and prefix of text2"""
        
        max_check = min(len(text1), len(text2), 200)
        common_length = 0
        
        # Check character by character from the end of text1 and start of text2
        for i in range(1, max_check + 1):
            if text1[-i] == text2[i - 1]:
                common_length = i
            else:
                break
        
        return common_length
    
    def calculate_overlap_quality(self, chunk1, chunk2) -> float:
        """Calculate quality score for overlap between two chunks"""
        
        if not chunk2.metadata.get('has_overlap', False):
            return 0.0
        
        overlap_size = chunk2.metadata.get('overlap_size', 0)
        if overlap_size == 0:
            return 0.0
        
        # Extract overlap content
        overlap_content = chunk2.content[:overlap_size]
        
        # Quality factors
        quality_score = 0.0
        
        # 1. Appropriate size (not too small, not too large)
        size_ratio = overlap_size / len(chunk2.content)
        if 0.05 <= size_ratio <= 0.3:  # 5-30% is good
            quality_score += 0.3
        
        # 2. Ends at sentence boundary
        if overlap_content.strip().endswith(('.', '!', '?')):
            quality_score += 0.2
        
        # 3. Starts at word boundary
        if overlap_content.strip().split()[0].isalpha():
            quality_score += 0.1
        
        # 4. Contains complete thoughts (simple heuristic)
        sentence_count = len(re.findall(r'[.!?]+', overlap_content))
        if sentence_count >= 1:
            quality_score += 0.2
        
        # 5. No excessive duplication
        if not chunk2.metadata.get('deduplication_applied', False):
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def get_overlap_statistics(self, chunks: List) -> Dict[str, Any]:
        """Get statistics about overlap application"""
        
        stats = {
            'total_chunks': len(chunks),
            'chunks_with_overlap': 0,
            'total_overlap_chars': 0,
            'average_overlap_size': 0,
            'overlap_quality_scores': [],
            'deduplication_applied': 0
        }
        
        overlap_sizes = []
        
        for i, chunk in enumerate(chunks):
            if chunk.metadata.get('has_overlap', False):
                stats['chunks_with_overlap'] += 1
                
                overlap_size = chunk.metadata.get('overlap_size', 0)
                stats['total_overlap_chars'] += overlap_size
                overlap_sizes.append(overlap_size)
                
                if i > 0:
                    quality = self.calculate_overlap_quality(chunks[i-1], chunk)
                    stats['overlap_quality_scores'].append(quality)
                
                if chunk.metadata.get('deduplication_applied', False):
                    stats['deduplication_applied'] += 1
        
        if overlap_sizes:
            stats['average_overlap_size'] = sum(overlap_sizes) / len(overlap_sizes)
            stats['min_overlap_size'] = min(overlap_sizes)
            stats['max_overlap_size'] = max(overlap_sizes)
        
        if stats['overlap_quality_scores']:
            stats['average_quality_score'] = sum(stats['overlap_quality_scores']) / len(stats['overlap_quality_scores'])
        
        return stats


if __name__ == "__main__":
    # Test overlap management
    import asyncio
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        overlap_ratio: float = 0.1
        min_overlap_size: int = 100
        max_overlap_size: int = 300
    
    @dataclass
    class MockChunk:
        content: str
        chunk_id: str
        chunk_index: int
        start_char: int
        end_char: int
        chunk_type: str = "text"
        section_title: str = None
        section_level: int = 0
        metadata: dict = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
        
        @property
        def size(self) -> int:
            return len(self.content)
    
    @dataclass
    class MockDocStructure:
        pass
    
    async def test_overlap_management():
        # Create test content and chunks
        content = """
        This is the first paragraph of our test document. It contains important
        information that provides context for the following sections.
        
        This is the second paragraph that continues the discussion. It builds
        upon the concepts introduced in the first paragraph and adds new details.
        
        This is the third paragraph that concludes our discussion. It summarizes
        the key points and provides final thoughts on the topic.
        """
        
        # Create mock chunks
        chunks = [
            MockChunk(
                content="This is the first paragraph of our test document. It contains important information that provides context for the following sections.",
                chunk_id="chunk_0",
                chunk_index=0,
                start_char=0,
                end_char=140,
                chunk_type="text"
            ),
            MockChunk(
                content="This is the second paragraph that continues the discussion. It builds upon the concepts introduced in the first paragraph and adds new details.",
                chunk_id="chunk_1", 
                chunk_index=1,
                start_char=140,
                end_char=290,
                chunk_type="text"
            ),
            MockChunk(
                content="This is the third paragraph that concludes our discussion. It summarizes the key points and provides final thoughts on the topic.",
                chunk_id="chunk_2",
                chunk_index=2,
                start_char=290,
                end_char=430,
                chunk_type="text"
            )
        ]
        
        config = MockConfig()
        doc_structure = MockDocStructure()
        
        overlap_manager = OverlapManager(config)
        
        # Apply overlap
        overlapped_chunks = await overlap_manager.apply_overlap(
            chunks, content, doc_structure
        )
        
        print(f"Applied overlap to {len(overlapped_chunks)} chunks:")
        
        for i, chunk in enumerate(overlapped_chunks):
            print(f"\nChunk {i} ({chunk.chunk_id}):")
            print(f"  Size: {chunk.size} chars")
            print(f"  Has overlap: {chunk.metadata.get('has_overlap', False)}")
            if chunk.metadata.get('has_overlap', False):
                print(f"  Overlap size: {chunk.metadata.get('overlap_size', 0)} chars")
            print(f"  Content preview: {chunk.content[:100]}...")
        
        # Get statistics
        stats = overlap_manager.get_overlap_statistics(overlapped_chunks)
        print(f"\nOverlap Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    asyncio.run(test_overlap_management())