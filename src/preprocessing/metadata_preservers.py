"""
Metadata Preservation for Content Chunking

This module handles the preservation and enhancement of metadata during
the chunking process, ensuring that important context information is
maintained for optimal retrieval and assembly.
"""

import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import asdict

logger = logging.getLogger(__name__)


class MetadataPreserver:
    """
    Preserves and enhances metadata during content chunking to maintain
    context information crucial for retrieval and assembly operations.
    """
    
    def __init__(self):
        """Initialize metadata preserver"""
        self.preserved_fields = {
            'source_document_id',
            'source_file_path', 
            'source_url',
            'document_title',
            'author',
            'created_date',
            'modified_date',
            'document_type',
            'language',
            'project_id',
            'repository_name',
            'branch_name',
            'commit_hash',
            'file_size',
            'line_count'
        }
        
        logger.debug("MetadataPreserver initialized")
    
    async def preserve_metadata(
        self,
        chunk,
        source_metadata: Optional[Dict[str, Any]] = None,
        doc_structure = None
    ) -> Dict[str, Any]:
        """
        Preserve and enhance metadata for a content chunk
        
        Args:
            chunk: ContentChunk instance
            source_metadata: Original document metadata
            doc_structure: DocumentStructure instance
            
        Returns:
            Enhanced metadata dictionary
        """
        metadata = {}
        
        # Preserve source document metadata
        if source_metadata:
            metadata.update(self._preserve_source_metadata(source_metadata))
        
        # Add chunk-specific metadata
        metadata.update(self._generate_chunk_metadata(chunk))
        
        # Add document structure context
        if doc_structure:
            metadata.update(self._preserve_structure_metadata(chunk, doc_structure))
        
        # Add processing metadata
        metadata.update(self._generate_processing_metadata(chunk))
        
        # Add quality indicators
        metadata.update(self._calculate_quality_indicators(chunk))
        
        return metadata
    
    def _preserve_source_metadata(self, source_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve relevant source document metadata"""
        preserved = {}
        
        for field in self.preserved_fields:
            if field in source_metadata:
                preserved[f'source_{field}'] = source_metadata[field]
        
        # Handle special cases and transformations
        if 'file_path' in source_metadata:
            file_path = source_metadata['file_path']
            preserved['source_file_extension'] = self._extract_file_extension(file_path)
            preserved['source_file_name'] = self._extract_file_name(file_path)
        
        if 'created_date' in source_metadata:
            preserved['source_age_days'] = self._calculate_document_age(
                source_metadata['created_date']
            )
        
        if 'tags' in source_metadata:
            preserved['source_tags'] = source_metadata['tags']
        
        return preserved
    
    def _generate_chunk_metadata(self, chunk) -> Dict[str, Any]:
        """Generate chunk-specific metadata"""
        metadata = {
            # Basic chunk information
            'chunk_id': chunk.chunk_id,
            'chunk_index': chunk.chunk_index,
            'chunk_size_chars': chunk.size,
            'chunk_size_words': len(chunk.content.split()),
            'chunk_type': chunk.chunk_type,
            
            # Position information
            'document_position_ratio': chunk.document_position,
            'start_char': chunk.start_char,
            'end_char': chunk.end_char,
            
            # Content characteristics
            'estimated_tokens': chunk.estimated_tokens,
            'content_hash': chunk.content_hash,
            'language': chunk.language,
            
            # Quality metrics
            'semantic_coherence_score': chunk.semantic_coherence_score,
            'structural_completeness': chunk.structural_completeness,
            'information_density': chunk.information_density,
            'quality_score': chunk.quality_score,
            
            # Processing timestamp
            'processed_at': datetime.now().isoformat(),
        }
        
        # Add section context if available
        if chunk.section_title:
            metadata['section_title'] = chunk.section_title
            metadata['section_level'] = chunk.section_level
        
        # Add relationships if available
        if chunk.preceding_chunks:
            metadata['preceding_chunks'] = chunk.preceding_chunks
        if chunk.following_chunks:
            metadata['following_chunks'] = chunk.following_chunks
        if chunk.related_chunks:
            metadata['related_chunks'] = chunk.related_chunks
        
        return metadata
    
    def _preserve_structure_metadata(self, chunk, doc_structure) -> Dict[str, Any]:
        """Preserve document structure context"""
        metadata = {
            'document_content_type': doc_structure.content_type.value,
            'document_language': doc_structure.language,
            'document_is_structured': doc_structure.is_highly_structured,
            'document_has_sections': doc_structure.has_clear_sections,
            'document_max_header_level': doc_structure.max_header_level,
            'document_structure_complexity': doc_structure.structure_complexity,
            'document_content_density': doc_structure.content_density
        }
        
        # Find the section this chunk belongs to
        section = doc_structure.get_section_at_position(chunk.start_char)
        if section:
            metadata.update({
                'section_type': section.section_type,
                'section_size': section.size,
                'section_has_subsections': section.has_subsections,
                'section_metadata': section.metadata
            })
        
        # Add document-level statistics for context
        metadata.update({
            'document_word_count': doc_structure.word_count,
            'document_paragraph_count': doc_structure.paragraph_count,
            'document_avg_paragraph_length': doc_structure.average_paragraph_length,
            'document_has_code_blocks': doc_structure.has_code_blocks,
            'document_has_tables': doc_structure.has_tables,
            'document_has_lists': doc_structure.has_lists
        })
        
        # Code-specific metadata
        if doc_structure.content_type.value == 'code':
            metadata.update({
                'document_function_count': doc_structure.function_count,
                'document_class_count': doc_structure.class_count,
                'document_import_count': doc_structure.import_count
            })
        
        return metadata
    
    def _generate_processing_metadata(self, chunk) -> Dict[str, Any]:
        """Generate processing-related metadata"""
        metadata = {
            'chunking_version': '1.0',
            'chunking_strategy': getattr(chunk, 'chunking_strategy', 'unknown'),
            'processing_date': datetime.now().isoformat(),
        }
        
        # Add overlap information if present
        if hasattr(chunk, 'metadata') and chunk.metadata:
            if chunk.metadata.get('has_overlap', False):
                metadata.update({
                    'has_overlap': True,
                    'overlap_size': chunk.metadata.get('overlap_size', 0),
                    'overlap_type': chunk.metadata.get('overlap_type', 'unknown')
                })
            
            # Add boundary information if present
            if 'boundary_type' in chunk.metadata:
                metadata.update({
                    'boundary_type': chunk.metadata['boundary_type'],
                    'boundary_strength': chunk.metadata.get('boundary_strength', 0.0),
                    'boundary_confidence': chunk.metadata.get('boundary_confidence', 0.0)
                })
        
        return metadata
    
    def _calculate_quality_indicators(self, chunk) -> Dict[str, Any]:
        """Calculate quality indicators for the chunk"""
        content = chunk.content
        
        # Text quality indicators
        indicators = {
            'sentence_count': len([s for s in content.split('.') if s.strip()]),
            'avg_sentence_length': 0,
            'paragraph_count': len(content.split('\n\n')),
            'line_count': content.count('\n') + 1,
            'has_punctuation': any(p in content for p in '.!?'),
            'capitalization_ratio': sum(1 for c in content if c.isupper()) / max(len(content), 1),
            'whitespace_ratio': sum(1 for c in content if c.isspace()) / max(len(content), 1)
        }
        
        # Calculate average sentence length
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if sentences:
            indicators['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Content type specific indicators
        if chunk.chunk_type == 'code':
            indicators.update({
                'code_line_count': content.count('\n') + 1,
                'code_has_comments': '#' in content or '//' in content or '/*' in content,
                'code_indentation_levels': self._count_indentation_levels(content),
                'code_brace_balance': self._check_brace_balance(content)
            })
        elif chunk.chunk_type == 'table':
            indicators.update({
                'table_row_count': content.count('|') // content.count('\n') if '\n' in content else 0,
                'table_has_header': '---' in content
            })
        elif chunk.chunk_type == 'list':
            indicators.update({
                'list_item_count': len([l for l in content.split('\n') if l.strip().startswith(('-', '*', '+')) or l.strip().split('.')[0].isdigit()]),
                'list_nesting_levels': self._count_list_nesting(content)
            })
        
        # Readability indicators (simplified)
        words = content.split()
        if words:
            indicators.update({
                'avg_word_length': sum(len(word) for word in words) / len(words),
                'unique_word_ratio': len(set(words)) / len(words),
                'complex_word_ratio': sum(1 for word in words if len(word) > 6) / len(words)
            })
        
        return {'quality_indicators': indicators}
    
    def _extract_file_extension(self, file_path: str) -> str:
        """Extract file extension from path"""
        if '.' in file_path:
            return file_path.split('.')[-1].lower()
        return ''
    
    def _extract_file_name(self, file_path: str) -> str:
        """Extract file name from path"""
        if '/' in file_path:
            return file_path.split('/')[-1]
        elif '\\' in file_path:
            return file_path.split('\\')[-1]
        return file_path
    
    def _calculate_document_age(self, created_date) -> int:
        """Calculate document age in days"""
        try:
            if isinstance(created_date, str):
                from datetime import datetime
                created = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            else:
                created = created_date
            
            age = datetime.now() - created.replace(tzinfo=None)
            return age.days
        except:
            return 0
    
    def _count_indentation_levels(self, content: str) -> int:
        """Count indentation levels in code"""
        levels = set()
        for line in content.split('\n'):
            if line.strip():  # Skip empty lines
                leading_spaces = len(line) - len(line.lstrip())
                levels.add(leading_spaces)
        return len(levels)
    
    def _check_brace_balance(self, content: str) -> bool:
        """Check if braces are balanced in code"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in content:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    def _count_list_nesting(self, content: str) -> int:
        """Count list nesting levels"""
        max_nesting = 0
        for line in content.split('\n'):
            if line.strip().startswith(('-', '*', '+')):
                leading_spaces = len(line) - len(line.lstrip())
                nesting_level = leading_spaces // 2  # Assume 2 spaces per level
                max_nesting = max(max_nesting, nesting_level)
        return max_nesting
    
    def enhance_metadata_for_retrieval(
        self, 
        chunks: List, 
        query_context: Optional[Dict[str, Any]] = None
    ) -> List:
        """Enhance metadata specifically for retrieval optimization"""
        
        enhanced_chunks = []
        
        for chunk in chunks:
            # Add retrieval-specific metadata
            retrieval_metadata = {
                'retrieval_keywords': self._extract_keywords(chunk.content),
                'retrieval_topics': self._identify_topics(chunk.content),
                'retrieval_entities': self._extract_entities(chunk.content),
                'retrieval_importance_score': self._calculate_importance_score(chunk)
            }
            
            # Add to existing metadata
            if hasattr(chunk, 'metadata') and chunk.metadata:
                chunk.metadata.update(retrieval_metadata)
            else:
                chunk.metadata = retrieval_metadata
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content (simplified implementation)"""
        import re
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'are', 'for', 'not', 'but', 'have', 'this', 'that',
            'with', 'from', 'they', 'she', 'her', 'his', 'him', 'you', 'your',
            'can', 'will', 'was', 'were', 'been', 'said', 'each', 'which'
        }
        
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Return most frequent keywords
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]
    
    def _identify_topics(self, content: str) -> List[str]:
        """Identify topics in content (simplified implementation)"""
        topics = []
        
        # Technical topics
        if any(word in content.lower() for word in ['api', 'endpoint', 'request', 'response']):
            topics.append('api')
        
        if any(word in content.lower() for word in ['database', 'query', 'table', 'sql']):
            topics.append('database')
        
        if any(word in content.lower() for word in ['authentication', 'login', 'password', 'security']):
            topics.append('security')
        
        if any(word in content.lower() for word in ['configuration', 'config', 'settings']):
            topics.append('configuration')
        
        if any(word in content.lower() for word in ['deployment', 'docker', 'kubernetes', 'deploy']):
            topics.append('deployment')
        
        return topics
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities (simplified implementation)"""
        import re
        
        entities = []
        
        # Extract potential file names
        file_patterns = re.findall(r'\b\w+\.\w+\b', content)
        entities.extend([f for f in file_patterns if '.' in f])
        
        # Extract URLs
        url_patterns = re.findall(r'https?://[^\s]+', content)
        entities.extend(url_patterns)
        
        # Extract code identifiers (functions, classes)
        code_patterns = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\(\)', content)
        entities.extend([p.replace('()', '') for p in code_patterns])
        
        return list(set(entities))  # Remove duplicates
    
    def _calculate_importance_score(self, chunk) -> float:
        """Calculate importance score for retrieval ranking"""
        score = 0.0
        
        # Base score from quality metrics
        score += chunk.quality_score * 0.4
        
        # Position in document (earlier chunks might be more important for intros)
        if chunk.document_position < 0.1:  # First 10% of document
            score += 0.2
        elif chunk.document_position > 0.9:  # Last 10% of document (conclusions)
            score += 0.1
        
        # Section level (higher level sections are more important)
        if chunk.section_level > 0:
            score += min(chunk.section_level / 6.0, 0.2)
        
        # Content type bonuses
        if chunk.chunk_type in ['header', 'section']:
            score += 0.3
        elif chunk.chunk_type == 'code':
            score += 0.1
        
        # Size penalty for very large or very small chunks
        if 500 <= chunk.size <= 2000:  # Optimal size range
            score += 0.1
        elif chunk.size < 200 or chunk.size > 3000:
            score -= 0.1
        
        return min(score, 1.0)
    
    def create_metadata_summary(self, chunks: List) -> Dict[str, Any]:
        """Create a summary of metadata across all chunks"""
        summary = {
            'total_chunks': len(chunks),
            'chunk_types': {},
            'section_levels': {},
            'languages': set(),
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'size_distribution': {'small': 0, 'medium': 0, 'large': 0},
            'total_content_size': 0,
            'avg_quality_score': 0.0,
            'has_overlaps': 0,
            'processing_date': datetime.now().isoformat()
        }
        
        quality_scores = []
        
        for chunk in chunks:
            # Chunk types
            chunk_type = chunk.chunk_type
            summary['chunk_types'][chunk_type] = summary['chunk_types'].get(chunk_type, 0) + 1
            
            # Section levels
            level = chunk.section_level
            summary['section_levels'][level] = summary['section_levels'].get(level, 0) + 1
            
            # Languages
            if chunk.language:
                summary['languages'].add(chunk.language)
            
            # Quality distribution
            quality = chunk.quality_score
            quality_scores.append(quality)
            if quality >= 0.7:
                summary['quality_distribution']['high'] += 1
            elif quality >= 0.4:
                summary['quality_distribution']['medium'] += 1
            else:
                summary['quality_distribution']['low'] += 1
            
            # Size distribution
            size = chunk.size
            summary['total_content_size'] += size
            if size < 500:
                summary['size_distribution']['small'] += 1
            elif size < 1500:
                summary['size_distribution']['medium'] += 1
            else:
                summary['size_distribution']['large'] += 1
            
            # Overlaps
            if hasattr(chunk, 'metadata') and chunk.metadata and chunk.metadata.get('has_overlap', False):
                summary['has_overlaps'] += 1
        
        # Calculate averages
        if quality_scores:
            summary['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        summary['avg_chunk_size'] = summary['total_content_size'] / max(len(chunks), 1)
        summary['languages'] = list(summary['languages'])
        
        return summary


if __name__ == "__main__":
    # Test metadata preservation
    from dataclasses import dataclass
    from datetime import datetime
    
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
        language: str = None
        document_position: float = 0.0
        semantic_coherence_score: float = 0.8
        structural_completeness: float = 0.7
        information_density: float = 0.6
        content_hash: str = "abc123"
        preceding_chunks: list = None
        following_chunks: list = None
        related_chunks: list = None
        metadata: dict = None
        
        def __post_init__(self):
            if self.preceding_chunks is None:
                self.preceding_chunks = []
            if self.following_chunks is None:
                self.following_chunks = []
            if self.related_chunks is None:
                self.related_chunks = []
            if self.metadata is None:
                self.metadata = {}
        
        @property
        def size(self) -> int:
            return len(self.content)
        
        @property
        def estimated_tokens(self) -> int:
            return len(self.content) // 4
        
        @property
        def quality_score(self) -> float:
            return (self.semantic_coherence_score + self.structural_completeness + self.information_density) / 3
    
    @dataclass
    class MockDocStructure:
        content_type: str = "markdown"
        language: str = "markdown"
        is_highly_structured: bool = True
        has_clear_sections: bool = True
        max_header_level: int = 2
        structure_complexity: float = 0.6
        content_density: float = 0.7
        word_count: int = 500
        paragraph_count: int = 5
        average_paragraph_length: float = 100.0
        has_code_blocks: bool = True
        has_tables: bool = False
        has_lists: bool = True
        function_count: int = 0
        class_count: int = 0
        import_count: int = 0
        
        def get_section_at_position(self, position):
            return None
    
    def test_metadata_preservation():
        # Create test data
        chunk = MockChunk(
            content="This is a test chunk with some content for metadata testing. It contains multiple sentences and provides a good example.",
            chunk_id="test_chunk_1",
            chunk_index=0,
            start_char=0,
            end_char=120,
            chunk_type="text",
            section_title="Test Section",
            section_level=1,
            document_position=0.5
        )
        
        source_metadata = {
            'source_document_id': 'doc_123',
            'source_file_path': '/path/to/test.md',
            'author': 'Test Author',
            'created_date': datetime.now().isoformat(),
            'document_type': 'markdown',
            'project_id': 'proj_456',
            'tags': ['test', 'example']
        }
        
        doc_structure = MockDocStructure()
        
        # Test metadata preservation
        preserver = MetadataPreserver()
        
        # Preserve metadata
        metadata = preserver.preserve_metadata(chunk, source_metadata, doc_structure)
        
        print("Preserved Metadata:")
        for key, value in metadata.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        # Test metadata summary
        chunks = [chunk]
        summary = preserver.create_metadata_summary(chunks)
        
        print(f"\nMetadata Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    test_metadata_preservation()