"""
Semantic Chunking Strategy

This module implements semantic boundary detection for intelligent content chunking.
It analyzes text content to identify natural semantic breaks and creates chunks
based on meaning rather than just structural markers.
"""

import logging
import re
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass  
class SemanticBoundary:
    """Represents a semantic boundary in text"""
    position: int                    # Character position
    strength: float                  # Boundary strength (0-1)
    boundary_type: str              # 'topic_shift', 'paragraph', 'sentence', 'clause'
    context_before: str             # Text before boundary
    context_after: str              # Text after boundary
    confidence: float = 0.0         # Confidence in boundary detection


class SemanticChunker:
    """
    Semantic chunking strategy that identifies natural semantic boundaries
    and creates chunks based on content meaning and coherence.
    """
    
    def __init__(self, config):
        """Initialize semantic chunker
        
        Args:
            config: ChunkingConfig instance
        """
        self.config = config
        
        # Compile regex patterns for performance
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.topic_indicators = re.compile(
            r'\b(?:however|therefore|moreover|furthermore|nevertheless|'
            r'in contrast|on the other hand|meanwhile|finally|'
            r'first|second|third|next|then|additionally|also)\b',
            re.IGNORECASE
        )
        
        # Transition words that indicate semantic shifts
        self.transition_words = {
            'contrast': ['however', 'nevertheless', 'on the other hand', 'in contrast', 'but', 'yet'],
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'sequence': ['first', 'second', 'third', 'next', 'then', 'finally', 'subsequently'],
            'causation': ['therefore', 'thus', 'consequently', 'as a result', 'hence'],
            'temporal': ['meanwhile', 'simultaneously', 'previously', 'afterwards', 'later']
        }
        
        logger.debug("SemanticChunker initialized")
    
    async def chunk_content(
        self, 
        content: str, 
        doc_structure,
        start_offset: int = 0
    ) -> List:
        """
        Chunk content using semantic boundary detection
        
        Args:
            content: Text content to chunk
            doc_structure: DocumentStructure instance
            start_offset: Character offset in original document
            
        Returns:
            List of ContentChunk objects
        """
        if not content or not content.strip():
            return []
        
        logger.debug(f"Semantic chunking {len(content)} characters")
        
        # Detect semantic boundaries
        boundaries = await self._detect_semantic_boundaries(content)
        
        # Create chunks based on boundaries
        chunks = await self._create_chunks_from_boundaries(
            content, boundaries, start_offset, doc_structure
        )
        
        # Refine chunks for optimal size and coherence
        refined_chunks = await self._refine_chunks(chunks, content)
        
        logger.debug(f"Created {len(refined_chunks)} semantic chunks")
        return refined_chunks
    
    async def _detect_semantic_boundaries(self, content: str) -> List[SemanticBoundary]:
        """Detect semantic boundaries in text content"""
        boundaries = []
        
        # Detect paragraph boundaries
        paragraph_boundaries = self._detect_paragraph_boundaries(content)
        boundaries.extend(paragraph_boundaries)
        
        # Detect sentence boundaries with semantic significance
        sentence_boundaries = self._detect_sentence_boundaries(content)
        boundaries.extend(sentence_boundaries)
        
        # Detect topic shift boundaries
        topic_boundaries = await self._detect_topic_shifts(content)
        boundaries.extend(topic_boundaries)
        
        # Sort boundaries by position and remove duplicates
        boundaries.sort(key=lambda x: x.position)
        boundaries = self._remove_duplicate_boundaries(boundaries)
        
        # Score boundary strength
        boundaries = self._score_boundary_strength(boundaries, content)
        
        return boundaries
    
    def _detect_paragraph_boundaries(self, content: str) -> List[SemanticBoundary]:
        """Detect paragraph boundaries as potential semantic breaks"""
        boundaries = []
        
        for match in self.paragraph_pattern.finditer(content):
            position = match.end()
            
            # Get context around boundary
            context_start = max(0, position - 100)
            context_end = min(len(content), position + 100)
            context_before = content[context_start:position].strip()
            context_after = content[position:context_end].strip()
            
            boundary = SemanticBoundary(
                position=position,
                strength=0.7,  # Paragraphs are strong semantic boundaries
                boundary_type='paragraph',
                context_before=context_before[-50:] if context_before else '',
                context_after=context_after[:50] if context_after else ''
            )
            boundaries.append(boundary)
        
        return boundaries
    
    def _detect_sentence_boundaries(self, content: str) -> List[SemanticBoundary]:
        """Detect sentence boundaries that might indicate semantic shifts"""
        boundaries = []
        
        sentences = self.sentence_pattern.split(content)
        position = 0
        
        for i, sentence in enumerate(sentences[:-1]):  # Exclude last empty split
            position += len(sentence)
            
            # Look for transition words at sentence start
            next_sentence = sentences[i + 1].strip() if i + 1 < len(sentences) else ''
            if next_sentence:
                # Check if next sentence starts with transition word
                transition_strength = self._analyze_transition_strength(next_sentence)
                
                if transition_strength > 0.3:  # Only significant transitions
                    boundary = SemanticBoundary(
                        position=position,
                        strength=transition_strength,
                        boundary_type='sentence',
                        context_before=sentence[-50:] if sentence else '',
                        context_after=next_sentence[:50]
                    )
                    boundaries.append(boundary)
            
            # Account for the delimiter in position
            position += len(self.sentence_pattern.search(content[position:]).group(0)) if self.sentence_pattern.search(content[position:]) else 1
        
        return boundaries
    
    async def _detect_topic_shifts(self, content: str) -> List[SemanticBoundary]:
        """Detect topic shifts using lexical and semantic analysis"""
        boundaries = []
        
        # Split content into windows for analysis
        window_size = 200
        step_size = 50
        
        windows = []
        for i in range(0, len(content) - window_size, step_size):
            window_text = content[i:i + window_size]
            windows.append({
                'start': i,
                'end': i + window_size,
                'text': window_text,
                'words': self._extract_keywords(window_text)
            })
        
        # Compare adjacent windows for topic shifts
        for i in range(len(windows) - 1):
            current_window = windows[i]
            next_window = windows[i + 1]
            
            # Calculate lexical similarity
            similarity = self._calculate_lexical_similarity(
                current_window['words'], next_window['words']
            )
            
            # Low similarity indicates potential topic shift
            if similarity < self.config.semantic_similarity_threshold:
                boundary_position = next_window['start']
                
                # Find the best sentence boundary near this position
                best_position = self._find_nearest_sentence_boundary(
                    content, boundary_position
                )
                
                if best_position > 0:
                    boundary = SemanticBoundary(
                        position=best_position,
                        strength=1.0 - similarity,  # Inverse of similarity
                        boundary_type='topic_shift',
                        context_before=content[max(0, best_position-50):best_position],
                        context_after=content[best_position:best_position+50],
                        confidence=1.0 - similarity
                    )
                    boundaries.append(boundary)
        
        return boundaries
    
    def _analyze_transition_strength(self, sentence: str) -> float:
        """Analyze the strength of semantic transition at sentence start"""
        sentence_lower = sentence.lower().strip()
        
        # Check for strong transition words
        for category, words in self.transition_words.items():
            for word in words:
                if sentence_lower.startswith(word + ' ') or sentence_lower.startswith(word + ','):
                    # Different categories have different strengths
                    strengths = {
                        'contrast': 0.9,
                        'causation': 0.8,
                        'sequence': 0.7,
                        'temporal': 0.6,
                        'addition': 0.4
                    }
                    return strengths.get(category, 0.5)
        
        # Check for other indicators
        if sentence_lower.startswith(('in summary', 'to conclude', 'in conclusion')):
            return 0.9
        
        if sentence_lower.startswith(('for example', 'for instance', 'such as')):
            return 0.3
        
        return 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Simple keyword extraction - could be enhanced with NLP libraries
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words (simplified list)
        stop_words = {
            'the', 'and', 'are', 'for', 'not', 'but', 'have', 'this', 'that',
            'with', 'from', 'they', 'she', 'her', 'his', 'him', 'you', 'your',
            'can', 'will', 'was', 'were', 'been', 'said', 'each', 'which',
            'what', 'when', 'where', 'who', 'why', 'how', 'all', 'any', 'may',
            'use', 'way', 'about', 'many', 'then', 'them', 'would', 'like',
            'into', 'time', 'has', 'two', 'more', 'very', 'after', 'words',
            'first', 'came', 'work', 'must', 'through', 'back', 'years'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _calculate_lexical_similarity(self, words1: List[str], words2: List[str]) -> float:
        """Calculate lexical similarity between two word lists"""
        if not words1 or not words2:
            return 0.0
        
        set1 = set(words1)
        set2 = set(words2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return jaccard
    
    def _find_nearest_sentence_boundary(self, content: str, position: int) -> int:
        """Find the nearest sentence boundary to a given position"""
        # Look for sentence endings within a reasonable range
        search_range = 100
        search_start = max(0, position - search_range // 2)
        search_end = min(len(content), position + search_range // 2)
        search_text = content[search_start:search_end]
        
        # Find sentence boundaries in the search range
        boundaries = []
        for match in self.sentence_pattern.finditer(search_text):
            abs_position = search_start + match.end()
            distance = abs(abs_position - position)
            boundaries.append((abs_position, distance))
        
        if boundaries:
            # Return the closest boundary
            boundaries.sort(key=lambda x: x[1])
            return boundaries[0][0]
        
        return position  # Fallback to original position
    
    def _remove_duplicate_boundaries(self, boundaries: List[SemanticBoundary]) -> List[SemanticBoundary]:
        """Remove duplicate boundaries that are too close together"""
        if not boundaries:
            return boundaries
        
        min_distance = 50  # Minimum characters between boundaries
        deduplicated = [boundaries[0]]
        
        for boundary in boundaries[1:]:
            if boundary.position - deduplicated[-1].position >= min_distance:
                deduplicated.append(boundary)
            else:
                # Keep the stronger boundary
                if boundary.strength > deduplicated[-1].strength:
                    deduplicated[-1] = boundary
        
        return deduplicated
    
    def _score_boundary_strength(
        self, 
        boundaries: List[SemanticBoundary], 
        content: str
    ) -> List[SemanticBoundary]:
        """Score and adjust boundary strengths based on context"""
        
        for boundary in boundaries:
            # Base strength is already set, now adjust based on context
            
            # Boost strength for boundaries after headers
            context_before = boundary.context_before.strip()
            if context_before and context_before.endswith(':'):
                boundary.strength = min(1.0, boundary.strength + 0.2)
            
            # Boost strength for boundaries before lists
            context_after = boundary.context_after.strip()
            if context_after.startswith(('- ', '* ', '1. ', '2. ', '3. ')):
                boundary.strength = min(1.0, boundary.strength + 0.15)
            
            # Reduce strength for boundaries in the middle of quotes
            if ('"' in boundary.context_before and '"' in boundary.context_after):
                boundary.strength *= 0.5
            
            # Boost strength for paragraph boundaries with topic indicators
            if boundary.boundary_type == 'paragraph':
                if self.topic_indicators.search(boundary.context_after):
                    boundary.strength = min(1.0, boundary.strength + 0.1)
        
        return boundaries
    
    async def _create_chunks_from_boundaries(
        self,
        content: str,
        boundaries: List[SemanticBoundary],
        start_offset: int,
        doc_structure
    ) -> List:
        """Create chunks based on detected semantic boundaries"""
        from ..chunker import ContentChunk  # Import here to avoid circular imports
        
        if not boundaries:
            # No boundaries found, create a single chunk
            chunk = ContentChunk(
                content=content,
                chunk_id="semantic_0",
                chunk_index=0,
                start_char=start_offset,
                end_char=start_offset + len(content),
                chunk_type="semantic"
            )
            return [chunk]
        
        chunks = []
        current_start = 0
        
        # Filter boundaries by strength threshold
        significant_boundaries = [
            b for b in boundaries 
            if b.strength >= 0.5  # Only use strong boundaries
        ]
        
        # Add end boundary
        if significant_boundaries and significant_boundaries[-1].position < len(content):
            end_boundary = SemanticBoundary(
                position=len(content),
                strength=1.0,
                boundary_type='end',
                context_before=content[-50:],
                context_after=''
            )
            significant_boundaries.append(end_boundary)
        
        for i, boundary in enumerate(significant_boundaries):
            chunk_content = content[current_start:boundary.position].strip()
            
            if chunk_content and len(chunk_content) >= self.config.min_chunk_size // 2:
                # Create chunk
                chunk = ContentChunk(
                    content=chunk_content,
                    chunk_id=f"semantic_{len(chunks)}",
                    chunk_index=len(chunks),
                    start_char=start_offset + current_start,
                    end_char=start_offset + boundary.position,
                    chunk_type="semantic"
                )
                
                # Add boundary information to metadata
                chunk.metadata = {
                    'boundary_type': boundary.boundary_type,
                    'boundary_strength': boundary.strength,
                    'boundary_confidence': boundary.confidence
                }
                
                chunks.append(chunk)
            
            current_start = boundary.position
        
        # Handle any remaining content
        if current_start < len(content):
            remaining_content = content[current_start:].strip()
            if remaining_content:
                chunk = ContentChunk(
                    content=remaining_content,
                    chunk_id=f"semantic_{len(chunks)}",
                    chunk_index=len(chunks),
                    start_char=start_offset + current_start,
                    end_char=start_offset + len(content),
                    chunk_type="semantic"
                )
                chunks.append(chunk)
        
        return chunks
    
    async def _refine_chunks(self, chunks: List, content: str) -> List:
        """Refine chunks for optimal size and coherence"""
        if not chunks:
            return chunks
        
        refined_chunks = []
        
        for chunk in chunks:
            if chunk.size > self.config.max_chunk_size:
                # Split large chunks
                sub_chunks = await self._split_large_chunk(chunk, content)
                refined_chunks.extend(sub_chunks)
            elif chunk.size < self.config.min_chunk_size:
                # Mark small chunks for potential merging
                chunk.metadata['needs_merge'] = True
                refined_chunks.append(chunk)
            else:
                refined_chunks.append(chunk)
        
        # Merge small chunks with adjacent chunks
        refined_chunks = self._merge_small_chunks(refined_chunks)
        
        return refined_chunks
    
    async def _split_large_chunk(self, chunk, content: str) -> List:
        """Split a chunk that exceeds maximum size"""
        from ..chunker import ContentChunk  # Import here to avoid circular imports
        
        # Use simple sentence splitting for large chunks
        sentences = self.sentence_pattern.split(chunk.content)
        
        sub_chunks = []
        current_content = ""
        current_start = chunk.start_char
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_content = current_content + (" " if current_content else "") + sentence
            
            if len(potential_content) > self.config.max_chunk_size and current_content:
                # Create sub-chunk
                sub_chunk = ContentChunk(
                    content=current_content.strip(),
                    chunk_id=f"{chunk.chunk_id}_sub_{len(sub_chunks)}",
                    chunk_index=len(sub_chunks),
                    start_char=current_start,
                    end_char=current_start + len(current_content),
                    chunk_type="semantic_split"
                )
                sub_chunks.append(sub_chunk)
                
                # Start new sub-chunk
                current_content = sentence
                current_start = current_start + len(current_content) + 1
            else:
                current_content = potential_content
        
        # Add final sub-chunk
        if current_content.strip():
            sub_chunk = ContentChunk(
                content=current_content.strip(),
                chunk_id=f"{chunk.chunk_id}_sub_{len(sub_chunks)}",
                chunk_index=len(sub_chunks),
                start_char=current_start,
                end_char=chunk.end_char,
                chunk_type="semantic_split"
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _merge_small_chunks(self, chunks: List) -> List:
        """Merge chunks that are too small with adjacent chunks"""
        if not chunks:
            return chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            if (current_chunk.metadata.get('needs_merge', False) and 
                i + 1 < len(chunks)):
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                
                merged_content = current_chunk.content + "\n\n" + next_chunk.content
                merged_chunk = type(current_chunk)(
                    content=merged_content,
                    chunk_id=f"merged_{current_chunk.chunk_id}",
                    chunk_index=len(merged_chunks),
                    start_char=current_chunk.start_char,
                    end_char=next_chunk.end_char,
                    chunk_type="semantic_merged"
                )
                
                # Combine metadata
                merged_chunk.metadata = current_chunk.metadata.copy()
                merged_chunk.metadata.update(next_chunk.metadata)
                merged_chunk.metadata.pop('needs_merge', None)
                
                merged_chunks.append(merged_chunk)
                i += 2  # Skip next chunk as it's been merged
            else:
                # Keep chunk as is
                current_chunk.metadata.pop('needs_merge', None)
                merged_chunks.append(current_chunk)
                i += 1
        
        return merged_chunks


if __name__ == "__main__":
    # Test semantic chunking
    import asyncio
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        target_chunk_size: int = 1500
        max_chunk_size: int = 2000
        min_chunk_size: int = 200
        semantic_similarity_threshold: float = 0.75
    
    @dataclass
    class MockDocStructure:
        pass
    
    async def test_semantic_chunking():
        test_content = """
        # Introduction
        
        This is an introduction to the topic. It provides basic background information
        that helps set the context for what follows.
        
        However, there are several important considerations to keep in mind. First,
        the methodology used in this analysis has certain limitations.
        
        ## Methodology
        
        The approach taken in this study follows established protocols. We collected
        data from multiple sources and applied statistical analysis.
        
        Furthermore, we validated our results through cross-validation techniques.
        This ensures the reliability of our findings.
        
        ## Results
        
        The results show significant improvements over baseline methods. Therefore,
        we can conclude that the proposed approach is effective.
        
        In contrast to previous studies, our method demonstrates better performance
        across all metrics. This suggests that the new approach has broader applicability.
        """
        
        config = MockConfig()
        doc_structure = MockDocStructure()
        
        chunker = SemanticChunker(config)
        chunks = await chunker.chunk_content(test_content, doc_structure)
        
        print(f"Created {len(chunks)} semantic chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({chunk.chunk_type}):")
            print(f"  Size: {chunk.size} chars")
            print(f"  Metadata: {chunk.metadata}")
            print(f"  Content preview: {chunk.content[:100]}...")
    
    asyncio.run(test_semantic_chunking())