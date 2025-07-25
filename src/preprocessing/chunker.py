"""
Main Content Chunking Implementation

This module provides the primary interface for intelligent content chunking
optimized for the Context Assembly Engine. It coordinates different chunking
strategies and ensures optimal chunk creation for RAG performance.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .strategies.semantic_chunker import SemanticChunker
from .strategies.structural_chunker import StructuralChunker
from .analyzers.document_analyzer import DocumentAnalyzer, DocumentStructure
from .utils.overlap_manager import OverlapManager
from .metadata_preservers import MetadataPreserver

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies optimized for Context Assembly"""
    SEMANTIC = "semantic"           # Semantic boundary detection
    STRUCTURAL = "structural"       # Document structure-aware
    HYBRID = "hybrid"              # Combination of semantic and structural
    ADAPTIVE = "adaptive"          # Auto-select based on content analysis


@dataclass
class ChunkingConfig:
    """Configuration for content chunking optimized for RAG"""
    # Core chunking parameters
    target_chunk_size: int = 1500      # Target characters per chunk (optimized for LLM context)
    max_chunk_size: int = 2000         # Maximum characters per chunk
    min_chunk_size: int = 200          # Minimum characters per chunk
    
    # Overlap configuration
    overlap_ratio: float = 0.1         # Overlap as ratio of chunk size
    min_overlap_size: int = 100        # Minimum overlap characters
    max_overlap_size: int = 300        # Maximum overlap characters
    
    # Strategy selection
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    preserve_structure: bool = True     # Preserve document structure markers
    
    # Semantic chunking options
    semantic_similarity_threshold: float = 0.75
    use_sentence_embeddings: bool = False  # For advanced semantic chunking
    
    # Structure preservation
    respect_headers: bool = True        # Don't split across headers
    respect_code_blocks: bool = True    # Keep code blocks intact
    respect_tables: bool = True         # Keep tables intact
    respect_lists: bool = True          # Keep list items together when possible
    
    # Metadata and optimization
    include_metadata: bool = True       # Include chunk metadata
    merge_short_chunks: bool = True     # Merge chunks below min_size
    optimize_for_tokens: bool = True    # Optimize chunk sizes for token efficiency
    max_chunks_per_document: int = 100  # Limit chunks per document
    
    # Performance options
    enable_caching: bool = True         # Cache analysis results
    parallel_processing: bool = True    # Process chunks in parallel when possible


@dataclass
class ContentChunk:
    """Enhanced content chunk with rich metadata for Context Assembly"""
    # Core content
    content: str
    chunk_id: str
    chunk_index: int
    
    # Position tracking
    start_char: int
    end_char: int
    start_token: Optional[int] = None
    end_token: Optional[int] = None
    
    # Content characteristics
    chunk_type: str = "text"            # text, code, table, list, header
    content_hash: Optional[str] = None   # For deduplication
    language: Optional[str] = None       # Programming/markup language
    
    # Structure context
    section_title: Optional[str] = None  # Parent section/header
    section_level: int = 0               # Header level (0 = no header)
    document_position: float = 0.0       # Position as ratio (0.0 to 1.0)
    
    # Quality metrics
    semantic_coherence_score: float = 0.0    # Semantic coherence (0-1)
    structural_completeness: float = 0.0     # Structural completeness (0-1)
    information_density: float = 0.0         # Information density score
    
    # Context relationships
    preceding_chunks: List[str] = field(default_factory=list)  # Chunk IDs
    following_chunks: List[str] = field(default_factory=list)  # Chunk IDs
    related_chunks: List[str] = field(default_factory=list)    # Semantically related
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        """Get chunk size in characters"""
        return len(self.content)
    
    @property
    def estimated_tokens(self) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters for English text
        return max(1, len(self.content) // 4)
    
    @property
    def quality_score(self) -> float:
        """Combined quality score for chunk selection"""
        return (
            self.semantic_coherence_score * 0.4 +
            self.structural_completeness * 0.3 +
            self.information_density * 0.3
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization"""
        return {
            'content': self.content,
            'chunk_id': self.chunk_id,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'size': self.size,
            'estimated_tokens': self.estimated_tokens,
            'chunk_type': self.chunk_type,
            'language': self.language,
            'section_title': self.section_title,
            'section_level': self.section_level,
            'document_position': self.document_position,
            'quality_score': self.quality_score,
            'semantic_coherence_score': self.semantic_coherence_score,
            'structural_completeness': self.structural_completeness,
            'information_density': self.information_density,
            'metadata': self.metadata
        }


class ContentChunker:
    """
    Main content chunker optimized for Context Assembly Engine
    
    Coordinates multiple chunking strategies and provides intelligent
    chunk creation with rich metadata for optimal RAG performance.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize content chunker
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        
        # Initialize component strategies
        self.semantic_chunker = SemanticChunker(self.config)
        self.structural_chunker = StructuralChunker(self.config)
        self.document_analyzer = DocumentAnalyzer()
        self.overlap_manager = OverlapManager(self.config)
        self.metadata_preserver = MetadataPreserver()
        
        # Performance tracking
        self._stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'strategy_usage': {}
        }
        
        logger.info(f"ContentChunker initialized with strategy: {self.config.strategy.value}")
    
    async def chunk_document(
        self,
        content: str,
        document_id: Optional[str] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None
    ) -> List[ContentChunk]:
        """
        Chunk a document using the configured strategy
        
        Args:
            content: Document content to chunk
            document_id: Unique document identifier
            source_metadata: Metadata about the source document
            file_path: Path to the source file (for type detection)
            
        Returns:
            List of content chunks with rich metadata
        """
        if not content or not content.strip():
            logger.warning("Empty content provided for chunking")
            return []
        
        import time
        start_time = time.time()
        
        try:
            # Analyze document structure
            doc_structure = await self.document_analyzer.analyze_document(
                content, file_path
            )
            
            # Select optimal chunking strategy
            strategy = self._select_strategy(doc_structure)
            logger.debug(f"Selected chunking strategy: {strategy.value} for document")
            
            # Perform initial chunking
            if strategy == ChunkingStrategy.SEMANTIC:
                initial_chunks = await self.semantic_chunker.chunk_content(
                    content, doc_structure
                )
            elif strategy == ChunkingStrategy.STRUCTURAL:
                initial_chunks = await self.structural_chunker.chunk_content(
                    content, doc_structure
                )
            elif strategy == ChunkingStrategy.HYBRID:
                initial_chunks = await self._hybrid_chunking(content, doc_structure)
            else:  # ADAPTIVE
                initial_chunks = await self._adaptive_chunking(content, doc_structure)
            
            # Apply overlap management
            chunks = await self.overlap_manager.apply_overlap(
                initial_chunks, content, doc_structure
            )
            
            # Enhance chunks with metadata
            enhanced_chunks = await self._enhance_chunks(
                chunks, content, doc_structure, document_id, source_metadata
            )
            
            # Post-process chunks
            final_chunks = await self._post_process_chunks(
                enhanced_chunks, doc_structure
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(final_chunks), processing_time, strategy)
            
            logger.info(
                f"Document chunked successfully: {len(final_chunks)} chunks "
                f"in {processing_time:.2f}s using {strategy.value} strategy"
            )
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}", exc_info=True)
            # Return simple fallback chunks
            return await self._fallback_chunking(content, document_id, source_metadata)
    
    def _select_strategy(self, doc_structure: DocumentStructure) -> ChunkingStrategy:
        """Select optimal chunking strategy based on document analysis"""
        if self.config.strategy != ChunkingStrategy.ADAPTIVE:
            return self.config.strategy
        
        # Adaptive strategy selection based on document characteristics
        if doc_structure.is_highly_structured and doc_structure.has_clear_sections:
            return ChunkingStrategy.STRUCTURAL
        elif doc_structure.content_type in ['markdown', 'restructuredtext']:
            return ChunkingStrategy.HYBRID
        elif doc_structure.has_code_blocks or doc_structure.content_type == 'code':
            return ChunkingStrategy.STRUCTURAL
        elif doc_structure.average_paragraph_length > 500:
            return ChunkingStrategy.SEMANTIC
        else:
            return ChunkingStrategy.HYBRID
    
    async def _hybrid_chunking(
        self, 
        content: str, 
        doc_structure: DocumentStructure
    ) -> List[ContentChunk]:
        """Combine semantic and structural chunking strategies"""
        
        # Start with structural chunking to respect document boundaries
        structural_chunks = await self.structural_chunker.chunk_content(
            content, doc_structure
        )
        
        # Apply semantic refinement to large structural chunks
        refined_chunks = []
        for chunk in structural_chunks:
            if chunk.size > self.config.max_chunk_size:
                # Use semantic chunking to split large structural chunks
                semantic_sub_chunks = await self.semantic_chunker.chunk_content(
                    chunk.content, doc_structure, chunk.start_char
                )
                refined_chunks.extend(semantic_sub_chunks)
            else:
                refined_chunks.append(chunk)
        
        return refined_chunks
    
    async def _adaptive_chunking(
        self,
        content: str,
        doc_structure: DocumentStructure
    ) -> List[ContentChunk]:
        """Adaptively chunk based on content characteristics"""
        
        # Analyze content sections and apply different strategies
        chunks = []
        
        for section in doc_structure.sections:
            section_content = content[section.start_pos:section.end_pos]
            
            if section.section_type == 'code':
                # Use structural chunking for code sections
                section_chunks = await self.structural_chunker.chunk_content(
                    section_content, doc_structure, section.start_pos
                )
            elif section.section_type == 'table':
                # Keep tables intact
                chunk = ContentChunk(
                    content=section_content,
                    chunk_id=f"section_{len(chunks)}",
                    chunk_index=len(chunks),
                    start_char=section.start_pos,
                    end_char=section.end_pos,
                    chunk_type='table'
                )
                section_chunks = [chunk]
            elif len(section_content) > self.config.target_chunk_size * 2:
                # Use semantic chunking for long text sections
                section_chunks = await self.semantic_chunker.chunk_content(
                    section_content, doc_structure, section.start_pos
                )
            else:
                # Use structural chunking for regular sections
                section_chunks = await self.structural_chunker.chunk_content(
                    section_content, doc_structure, section.start_pos
                )
            
            chunks.extend(section_chunks)
        
        return chunks
    
    async def _enhance_chunks(
        self,
        chunks: List[ContentChunk],
        content: str,
        doc_structure: DocumentStructure,
        document_id: Optional[str],
        source_metadata: Optional[Dict[str, Any]]
    ) -> List[ContentChunk]:
        """Enhance chunks with rich metadata and quality scores"""
        
        enhanced_chunks = []
        total_length = len(content)
        
        for i, chunk in enumerate(chunks):
            # Generate unique chunk ID
            chunk.chunk_id = f"{document_id or 'doc'}_{i:04d}"
            chunk.chunk_index = i
            
            # Calculate document position
            chunk.document_position = chunk.start_char / total_length
            
            # Determine section context
            section = doc_structure.get_section_at_position(chunk.start_char)
            if section:
                chunk.section_title = section.title
                chunk.section_level = section.level
            
            # Calculate quality scores
            chunk.semantic_coherence_score = await self._calculate_semantic_coherence(
                chunk, doc_structure
            )
            chunk.structural_completeness = self._calculate_structural_completeness(
                chunk, doc_structure
            )
            chunk.information_density = self._calculate_information_density(chunk)
            
            # Set content hash for deduplication
            import hashlib
            chunk.content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
            
            # Add source metadata
            if self.config.include_metadata:
                chunk.metadata = await self.metadata_preserver.preserve_metadata(
                    chunk, source_metadata, doc_structure
                )
            
            enhanced_chunks.append(chunk)
        
        # Establish chunk relationships
        for i, chunk in enumerate(enhanced_chunks):
            if i > 0:
                chunk.preceding_chunks = [enhanced_chunks[i-1].chunk_id]
            if i < len(enhanced_chunks) - 1:
                chunk.following_chunks = [enhanced_chunks[i+1].chunk_id]
        
        return enhanced_chunks
    
    async def _post_process_chunks(
        self,
        chunks: List[ContentChunk],
        doc_structure: DocumentStructure  
    ) -> List[ContentChunk]:
        """Post-process chunks for optimization"""
        
        if not chunks:
            return chunks
        
        processed_chunks = chunks.copy()
        
        # Merge short chunks if enabled
        if self.config.merge_short_chunks:
            processed_chunks = await self._merge_short_chunks(processed_chunks)
        
        # Remove duplicates based on content hash
        processed_chunks = self._remove_duplicate_chunks(processed_chunks)
        
        # Limit number of chunks if configured
        if len(processed_chunks) > self.config.max_chunks_per_document:
            # Keep highest quality chunks
            processed_chunks.sort(key=lambda x: x.quality_score, reverse=True)
            processed_chunks = processed_chunks[:self.config.max_chunks_per_document]
            # Re-sort by document position
            processed_chunks.sort(key=lambda x: x.start_char)
            # Update indices
            for i, chunk in enumerate(processed_chunks):
                chunk.chunk_index = i
        
        return processed_chunks
    
    async def _merge_short_chunks(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        """Merge chunks that are shorter than minimum size"""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_merge_group = []
        
        for chunk in chunks:
            if chunk.size < self.config.min_chunk_size:
                current_merge_group.append(chunk)
            else:
                # Merge any pending short chunks
                if current_merge_group:
                    merged_chunk = self._merge_chunk_group(current_merge_group)
                    merged_chunks.append(merged_chunk)
                    current_merge_group = []
                
                # Add current chunk
                merged_chunks.append(chunk)
        
        # Handle any remaining short chunks
        if current_merge_group:
            merged_chunk = self._merge_chunk_group(current_merge_group)
            merged_chunks.append(merged_chunk)
        
        return merged_chunks
    
    def _merge_chunk_group(self, chunk_group: List[ContentChunk]) -> ContentChunk:
        """Merge a group of chunks into a single chunk"""
        if not chunk_group:
            raise ValueError("Cannot merge empty chunk group")
        
        if len(chunk_group) == 1:
            return chunk_group[0]
        
        # Combine content
        combined_content = "\n\n".join(chunk.content for chunk in chunk_group)
        
        # Create merged chunk
        merged_chunk = ContentChunk(
            content=combined_content,
            chunk_id=f"merged_{chunk_group[0].chunk_id}",
            chunk_index=chunk_group[0].chunk_index,
            start_char=chunk_group[0].start_char,
            end_char=chunk_group[-1].end_char,
            chunk_type="merged",
            section_title=chunk_group[0].section_title,
            section_level=chunk_group[0].section_level,
            document_position=chunk_group[0].document_position
        )
        
        # Average quality scores
        merged_chunk.semantic_coherence_score = sum(
            c.semantic_coherence_score for c in chunk_group
        ) / len(chunk_group)
        merged_chunk.structural_completeness = sum(
            c.structural_completeness for c in chunk_group
        ) / len(chunk_group)
        merged_chunk.information_density = sum(
            c.information_density for c in chunk_group
        ) / len(chunk_group)
        
        return merged_chunk
    
    def _remove_duplicate_chunks(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        """Remove duplicate chunks based on content hash"""
        seen_hashes = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk.content_hash not in seen_hashes:
                seen_hashes.add(chunk.content_hash)
                unique_chunks.append(chunk)
            else:
                logger.debug(f"Removed duplicate chunk: {chunk.chunk_id}")
        
        return unique_chunks
    
    async def _calculate_semantic_coherence(
        self, 
        chunk: ContentChunk, 
        doc_structure: DocumentStructure
    ) -> float:
        """Calculate semantic coherence score for a chunk"""
        # Simplified implementation - in production, could use embeddings
        
        # Check sentence boundaries
        sentences = chunk.content.split('. ')
        if len(sentences) < 2:
            return 1.0  # Single sentence is perfectly coherent
        
        # Penalize chunks that end mid-sentence
        if not chunk.content.strip().endswith(('.', '!', '?', ':', ';')):
            return 0.6
        
        # Reward chunks that start and end at natural boundaries
        starts_with_capital = chunk.content.strip()[0].isupper()
        ends_with_punctuation = chunk.content.strip()[-1] in '.!?:;'
        
        base_score = 0.7
        if starts_with_capital:
            base_score += 0.15
        if ends_with_punctuation:
            base_score += 0.15
        
        return min(1.0, base_score)
    
    def _calculate_structural_completeness(
        self, 
        chunk: ContentChunk, 
        doc_structure: DocumentStructure
    ) -> float:
        """Calculate structural completeness score for a chunk"""
        
        # Check if chunk respects structural boundaries
        score = 0.5  # Base score
        
        # Reward chunks that align with document structure
        if chunk.chunk_type in ['header', 'section']:
            score += 0.3
        
        # Check for complete lists, code blocks, tables
        if '- ' in chunk.content or '* ' in chunk.content:
            # List handling
            lines = chunk.content.split('\n')
            list_lines = [l for l in lines if l.strip().startswith(('- ', '* ', '+ '))]
            if list_lines and not chunk.content.endswith('...'):
                score += 0.2
        
        if '```' in chunk.content:
            # Code block handling
            if chunk.content.count('```') % 2 == 0:  # Complete code blocks
                score += 0.3
            else:
                score -= 0.2  # Incomplete code blocks
        
        return min(1.0, max(0.0, score))
    
    def _calculate_information_density(self, chunk: ContentChunk) -> float:
        """Calculate information density score for a chunk"""
        
        content = chunk.content.strip()
        if not content:
            return 0.0
        
        # Count meaningful words
        import re
        words = re.findall(r'\b\w+\b', content.lower())
        meaningful_words = [w for w in words if len(w) > 2 and not w.isdigit()]
        
        # Count unique words
        unique_words = set(meaningful_words)
        
        # Calculate ratios
        word_density = len(meaningful_words) / len(content) if content else 0
        uniqueness_ratio = len(unique_words) / len(meaningful_words) if meaningful_words else 0
        
        # Combine metrics
        density_score = min(1.0, word_density * 1000)  # Scale up
        uniqueness_score = uniqueness_ratio
        
        return (density_score + uniqueness_score) / 2
    
    async def _fallback_chunking(
        self,
        content: str,
        document_id: Optional[str],
        source_metadata: Optional[Dict[str, Any]]
    ) -> List[ContentChunk]:
        """Fallback chunking when main strategy fails"""
        logger.warning("Using fallback chunking strategy")
        
        chunks = []
        chunk_size = self.config.target_chunk_size
        overlap = self.config.min_overlap_size
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            if chunk_content.strip():
                chunk = ContentChunk(
                    content=chunk_content,
                    chunk_id=f"{document_id or 'fallback'}_{len(chunks):04d}",
                    chunk_index=len(chunks),
                    start_char=i,
                    end_char=min(i + chunk_size, len(content)),
                    chunk_type="fallback"
                )
                chunks.append(chunk)
        
        return chunks
    
    def _update_stats(self, chunk_count: int, processing_time: float, strategy: ChunkingStrategy):
        """Update processing statistics"""
        self._stats['documents_processed'] += 1
        self._stats['chunks_created'] += chunk_count
        self._stats['total_processing_time'] += processing_time
        
        strategy_name = strategy.value
        if strategy_name not in self._stats['strategy_usage']:
            self._stats['strategy_usage'][strategy_name] = 0
        self._stats['strategy_usage'][strategy_name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        stats = self._stats.copy()
        if stats['documents_processed'] > 0:
            stats['average_chunks_per_document'] = (
                stats['chunks_created'] / stats['documents_processed']
            )
            stats['average_processing_time'] = (
                stats['total_processing_time'] / stats['documents_processed']
            )
        return stats


# Factory functions
def create_content_chunker(
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
    target_chunk_size: int = 1500,
    overlap_ratio: float = 0.1,
    preserve_structure: bool = True
) -> ContentChunker:
    """Factory function to create a configured content chunker"""
    config = ChunkingConfig(
        strategy=strategy,
        target_chunk_size=target_chunk_size,
        overlap_ratio=overlap_ratio,
        preserve_structure=preserve_structure
    )
    return ContentChunker(config)


async def chunk_document(
    content: str,
    document_id: Optional[str] = None,
    file_path: Optional[str] = None,
    config: Optional[ChunkingConfig] = None
) -> List[ContentChunk]:
    """Convenience function for chunking a single document"""
    chunker = ContentChunker(config)
    return await chunker.chunk_document(content, document_id, file_path=file_path)


if __name__ == "__main__":
    # Test chunking functionality
    import sys
    import asyncio
    
    async def test_chunking():
        if len(sys.argv) < 2:
            print("Usage: python -m src.preprocessing.chunker <test_file>")
            return
        
        file_path = sys.argv[1]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"Testing content chunker with: {file_path}")
            print(f"Content length: {len(content)} characters")
            
            # Test with different strategies
            strategies = [ChunkingStrategy.SEMANTIC, ChunkingStrategy.STRUCTURAL, ChunkingStrategy.HYBRID]
            
            for strategy in strategies:
                print(f"\n--- Testing {strategy.value.upper()} strategy ---")
                
                config = ChunkingConfig(strategy=strategy)
                chunker = ContentChunker(config)
                
                chunks = await chunker.chunk_document(
                    content, 
                    document_id=f"test_{strategy.value}",
                    file_path=file_path
                )
                
                print(f"Chunks created: {len(chunks)}")
                print(f"Avg chunk size: {sum(c.size for c in chunks) / len(chunks):.0f} chars")
                print(f"Avg quality score: {sum(c.quality_score for c in chunks) / len(chunks):.3f}")
                
                # Show sample chunks
                for i, chunk in enumerate(chunks[:2]):
                    print(f"\nChunk {i+1} ({chunk.chunk_type}):")
                    print(f"  Size: {chunk.size} chars")
                    print(f"  Quality: {chunk.quality_score:.3f}")
                    print(f"  Section: {chunk.section_title or 'None'}")
                    print(f"  Preview: {chunk.content[:100]}...")
                
                # Print stats
                stats = chunker.get_stats()
                print(f"\nStats: {stats}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(test_chunking())