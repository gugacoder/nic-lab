"""
Structural Chunking Strategy

This module implements structure-aware chunking that respects document
structure like headers, code blocks, tables, and lists. It ensures that
structural elements remain intact and meaningful.
"""

import logging
import re
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StructureType(Enum):
    """Types of document structures"""
    HEADER = "header"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    LIST = "list"
    PARAGRAPH = "paragraph"
    QUOTE = "quote"
    SECTION = "section"


@dataclass
class StructuralElement:
    """Represents a structural element in the document"""
    element_type: StructureType
    start_pos: int
    end_pos: int
    level: int = 0                  # For headers, list nesting, etc.
    language: Optional[str] = None  # For code blocks
    title: Optional[str] = None     # For headers, sections
    content: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def size(self) -> int:
        return self.end_pos - self.start_pos


class StructuralChunker:
    """
    Structural chunking strategy that respects document structure
    and keeps structural elements intact for better context preservation.
    """
    
    def __init__(self, config):
        """Initialize structural chunker
        
        Args:
            config: ChunkingConfig instance
        """
        self.config = config
        
        # Compile regex patterns for performance
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
        self.inline_code_pattern = re.compile(r'`([^`\n]+)`')
        self.list_pattern = re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(r'^\|.*\|$', re.MULTILINE)
        self.quote_pattern = re.compile(r'^>\s+(.+)$', re.MULTILINE)
        
        # Language-specific patterns for code files
        self.function_patterns = {
            'python': re.compile(r'^(\s*)def\s+(\w+)\s*\([^)]*\):'),
            'javascript': re.compile(r'^(\s*)function\s+(\w+)\s*\([^)]*\)\s*{'),
            'java': re.compile(r'^(\s*)(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*{'),
            'cpp': re.compile(r'^(\s*)\w+\s+(\w+)\s*\([^)]*\)\s*{'),
            'c': re.compile(r'^(\s*)\w+\s+(\w+)\s*\([^)]*\)\s*{')
        }
        
        self.class_patterns = {
            'python': re.compile(r'^(\s*)class\s+(\w+).*:'),
            'javascript': re.compile(r'^(\s*)class\s+(\w+).*{'),
            'java': re.compile(r'^(\s*)(public|private)?\s*class\s+(\w+).*{'),
            'cpp': re.compile(r'^(\s*)class\s+(\w+).*{')
        }
        
        logger.debug("StructuralChunker initialized")
    
    async def chunk_content(
        self, 
        content: str, 
        doc_structure,
        start_offset: int = 0
    ) -> List:
        """
        Chunk content using structural analysis
        
        Args:
            content: Text content to chunk
            doc_structure: DocumentStructure instance
            start_offset: Character offset in original document
            
        Returns:
            List of ContentChunk objects
        """
        if not content or not content.strip():
            return []
        
        logger.debug(f"Structural chunking {len(content)} characters")
        
        # Detect structural elements
        elements = await self._detect_structural_elements(content, doc_structure)
        
        # Create hierarchical structure
        structured_content = self._build_content_hierarchy(elements, content)
        
        # Create chunks respecting structure
        chunks = await self._create_structural_chunks(
            structured_content, content, start_offset
        )
        
        # Optimize chunk sizes
        optimized_chunks = await self._optimize_chunk_sizes(chunks, content)
        
        logger.debug(f"Created {len(optimized_chunks)} structural chunks")
        return optimized_chunks
    
    async def _detect_structural_elements(
        self, 
        content: str, 
        doc_structure
    ) -> List[StructuralElement]:
        """Detect all structural elements in the content"""
        elements = []
        
        # Detect headers
        elements.extend(self._detect_headers(content))
        
        # Detect code blocks
        elements.extend(self._detect_code_blocks(content))
        
        # Detect lists
        elements.extend(self._detect_lists(content))
        
        # Detect tables  
        elements.extend(self._detect_tables(content))
        
        # Detect quotes
        elements.extend(self._detect_quotes(content))
        
        # For code files, detect functions and classes
        if hasattr(doc_structure, 'content_type') and doc_structure.content_type == 'code':
            elements.extend(self._detect_code_structures(content, doc_structure.language))
        
        # Sort elements by position
        elements.sort(key=lambda x: x.start_pos)
        
        # Fill gaps with paragraph elements
        elements = self._fill_paragraph_gaps(elements, content)
        
        return elements
    
    def _detect_headers(self, content: str) -> List[StructuralElement]:
        """Detect markdown headers"""
        elements = []
        
        for match in self.header_pattern.finditer(content):
            level = len(match.group(1))  # Number of # characters
            title = match.group(2).strip()
            
            element = StructuralElement(
                element_type=StructureType.HEADER,
                start_pos=match.start(),
                end_pos=match.end(),
                level=level,
                title=title,
                content=match.group(0),
                metadata={'header_level': level, 'title': title}
            )
            elements.append(element)
        
        return elements
    
    def _detect_code_blocks(self, content: str) -> List[StructuralElement]:
        """Detect fenced code blocks"""
        elements = []
        
        for match in self.code_block_pattern.finditer(content):
            language = match.group(1) or 'unknown'
            code_content = match.group(2)
            
            element = StructuralElement(
                element_type=StructureType.CODE_BLOCK,
                start_pos=match.start(),
                end_pos=match.end(),
                language=language,
                content=match.group(0),
                metadata={
                    'language': language,
                    'code_length': len(code_content),
                    'line_count': code_content.count('\n') + 1
                }
            )
            elements.append(element)
        
        return elements
    
    def _detect_lists(self, content: str) -> List[StructuralElement]:
        """Detect markdown lists"""
        elements = []
        list_blocks = []
        current_list = None
        
        lines = content.split('\n')
        line_start = 0
        
        for line in lines:
            line_end = line_start + len(line) + 1  # +1 for newline
            
            match = self.list_pattern.match(line)
            if match:
                indent = len(match.group(1))
                marker = match.group(2)
                text = match.group(3)
                
                if current_list is None:
                    # Start new list
                    current_list = {
                        'start': line_start,
                        'end': line_end,
                        'items': [],
                        'base_indent': indent,
                        'list_type': 'ordered' if marker[0].isdigit() else 'unordered'
                    }
                
                current_list['end'] = line_end
                current_list['items'].append({
                    'indent': indent,
                    'marker': marker,
                    'text': text,
                    'start': line_start,
                    'end': line_end
                })
            else:
                # End current list if we hit a non-list line
                if current_list and line.strip():  # Non-empty line
                    list_blocks.append(current_list)
                    current_list = None
                elif current_list and not line.strip():
                    # Empty line within list - extend end position
                    current_list['end'] = line_end
            
            line_start = line_end
        
        # Handle list at end of content
        if current_list:
            list_blocks.append(current_list)
        
        # Convert to StructuralElements
        for list_block in list_blocks:
            element = StructuralElement(
                element_type=StructureType.LIST,
                start_pos=list_block['start'],
                end_pos=list_block['end'],
                content=content[list_block['start']:list_block['end']],
                metadata={
                    'list_type': list_block['list_type'],
                    'item_count': len(list_block['items']),
                    'base_indent': list_block['base_indent']
                }
            )
            elements.append(element)
        
        return elements
    
    def _detect_tables(self, content: str) -> List[StructuralElement]:
        """Detect markdown tables"""
        elements = []
        table_blocks = []
        current_table = None
        
        lines = content.split('\n')
        line_start = 0
        
        for line in lines:
            line_end = line_start + len(line) + 1
            
            if self.table_pattern.match(line.strip()):
                if current_table is None:
                    current_table = {
                        'start': line_start,
                        'end': line_end,
                        'rows': []
                    }
                
                current_table['end'] = line_end
                current_table['rows'].append(line.strip())
            else:
                if current_table:
                    table_blocks.append(current_table)
                    current_table = None
            
            line_start = line_end
        
        if current_table:
            table_blocks.append(current_table)
        
        # Convert to StructuralElements
        for table_block in table_blocks:
            element = StructuralElement(
                element_type=StructureType.TABLE,
                start_pos=table_block['start'],
                end_pos=table_block['end'],
                content=content[table_block['start']:table_block['end']],
                metadata={
                    'row_count': len(table_block['rows']),
                    'has_header': len(table_block['rows']) > 1 and '---' in table_block['rows'][1]
                }
            )
            elements.append(element)
        
        return elements
    
    def _detect_quotes(self, content: str) -> List[StructuralElement]:
        """Detect block quotes"""
        elements = []
        quote_blocks = []
        current_quote = None
        
        lines = content.split('\n')
        line_start = 0
        
        for line in lines:
            line_end = line_start + len(line) + 1
            
            if line.strip().startswith('>'):
                if current_quote is None:
                    current_quote = {
                        'start': line_start,
                        'end': line_end,
                        'lines': []
                    }
                
                current_quote['end'] = line_end
                current_quote['lines'].append(line)
            else:
                if current_quote:
                    quote_blocks.append(current_quote)
                    current_quote = None
            
            line_start = line_end
        
        if current_quote:
            quote_blocks.append(current_quote)
        
        # Convert to StructuralElements
        for quote_block in quote_blocks:
            element = StructuralElement(
                element_type=StructureType.QUOTE,
                start_pos=quote_block['start'],
                end_pos=quote_block['end'],
                content=content[quote_block['start']:quote_block['end']],
                metadata={'line_count': len(quote_block['lines'])}
            )
            elements.append(element)
        
        return elements
    
    def _detect_code_structures(self, content: str, language: str) -> List[StructuralElement]:
        """Detect functions and classes in code files"""
        elements = []
        
        if not language or language not in self.function_patterns:
            return elements
        
        # Detect functions
        function_pattern = self.function_patterns.get(language)
        if function_pattern:
            for match in function_pattern.finditer(content, re.MULTILINE):
                # Find the end of the function (simplified - could be improved)
                start_pos = match.start()
                function_name = match.groups()[-1]  # Last group is usually function name
                
                # Simple heuristic to find function end
                end_pos = self._find_code_block_end(content, start_pos, language)
                
                element = StructuralElement(
                    element_type=StructureType.CODE_BLOCK,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    language=language,
                    title=function_name,
                    content=content[start_pos:end_pos],
                    metadata={
                        'structure_type': 'function',
                        'function_name': function_name,
                        'language': language
                    }
                )
                elements.append(element)
        
        # Detect classes
        class_pattern = self.class_patterns.get(language)
        if class_pattern:
            for match in class_pattern.finditer(content, re.MULTILINE):
                start_pos = match.start()
                class_name = match.groups()[-1]  # Last group is usually class name
                
                end_pos = self._find_code_block_end(content, start_pos, language)
                
                element = StructuralElement(
                    element_type=StructureType.CODE_BLOCK,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    language=language,
                    title=class_name,
                    content=content[start_pos:end_pos],
                    metadata={
                        'structure_type': 'class',
                        'class_name': class_name,
                        'language': language
                    }
                )
                elements.append(element)
        
        return elements
    
    def _find_code_block_end(self, content: str, start_pos: int, language: str) -> int:
        """Find the end of a code block (function, class, etc.)"""
        # This is a simplified implementation
        # A full implementation would need proper parsing
        
        lines = content[start_pos:].split('\n')
        if not lines:
            return start_pos + 100  # Fallback
        
        # For Python, use indentation
        if language == 'python':
            first_line = lines[0]
            base_indent = len(first_line) - len(first_line.lstrip())
            
            for i, line in enumerate(lines[1:], 1):
                if line.strip() and not line.startswith(' ' * (base_indent + 1)):
                    # Found line with same or less indentation
                    return start_pos + sum(len(l) + 1 for l in lines[:i])
        
        # For other languages, look for matching braces
        else:
            brace_count = 0
            pos = start_pos
            
            for char in content[start_pos:]:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return pos + 1
                pos += 1
        
        # Fallback: return next 1000 characters or end of content
        return min(start_pos + 1000, len(content))
    
    def _fill_paragraph_gaps(
        self, 
        elements: List[StructuralElement], 
        content: str
    ) -> List[StructuralElement]:
        """Fill gaps between structural elements with paragraph elements"""
        if not elements:
            # No structural elements, treat entire content as one paragraph
            if content.strip():
                para_element = StructuralElement(
                    element_type=StructureType.PARAGRAPH,
                    start_pos=0,
                    end_pos=len(content),
                    content=content
                )
                return [para_element]
            return []
        
        filled_elements = []
        last_end = 0
        
        for element in elements:
            # Check for gap before this element
            if element.start_pos > last_end:
                gap_content = content[last_end:element.start_pos].strip()
                if gap_content:
                    para_element = StructuralElement(
                        element_type=StructureType.PARAGRAPH,
                        start_pos=last_end,
                        end_pos=element.start_pos,
                        content=gap_content
                    )
                    filled_elements.append(para_element)
            
            filled_elements.append(element)
            last_end = element.end_pos
        
        # Check for final gap
        if last_end < len(content):
            gap_content = content[last_end:].strip()
            if gap_content:
                para_element = StructuralElement(
                    element_type=StructureType.PARAGRAPH,
                    start_pos=last_end,
                    end_pos=len(content),
                    content=gap_content
                )
                filled_elements.append(para_element)
        
        return filled_elements
    
    def _build_content_hierarchy(
        self, 
        elements: List[StructuralElement], 
        content: str
    ) -> List[Dict[str, Any]]:
        """Build hierarchical content structure"""
        hierarchy = []
        current_section = None
        
        for element in elements:
            if element.element_type == StructureType.HEADER:
                # Start new section
                if current_section:
                    hierarchy.append(current_section)
                
                current_section = {
                    'header': element,
                    'elements': [],
                    'level': element.level,
                    'title': element.title
                }
            else:
                # Add element to current section or root
                if current_section:
                    current_section['elements'].append(element)
                else:
                    # No current section, create a root-level item
                    hierarchy.append({
                        'header': None,
                        'elements': [element],
                        'level': 0,
                        'title': None
                    })
        
        # Add final section
        if current_section:
            hierarchy.append(current_section)
        
        return hierarchy
    
    async def _create_structural_chunks(
        self, 
        structured_content: List[Dict[str, Any]], 
        content: str,
        start_offset: int
    ) -> List:
        """Create chunks respecting structural boundaries"""
        from ..chunker import ContentChunk  # Import here to avoid circular imports
        
        chunks = []
        
        for section in structured_content:
            header = section['header']
            elements = section['elements']
            
            # Calculate section bounds
            if header:
                section_start = header.start_pos
            elif elements:
                section_start = elements[0].start_pos
            else:
                continue
            
            section_end = elements[-1].end_pos if elements else header.end_pos
            section_content = content[section_start:section_end]
            
            # Check if entire section fits in one chunk
            if len(section_content) <= self.config.max_chunk_size:
                # Create single chunk for entire section
                chunk = ContentChunk(
                    content=section_content,
                    chunk_id=f"structural_{len(chunks)}",
                    chunk_index=len(chunks),
                    start_char=start_offset + section_start,
                    end_char=start_offset + section_end,
                    chunk_type="section",
                    section_title=section['title'],
                    section_level=section['level']
                )
                
                if header:
                    chunk.metadata = {
                        'has_header': True,
                        'header_level': header.level,
                        'element_count': len(elements)
                    }
                
                chunks.append(chunk)
            else:
                # Split section into multiple chunks
                section_chunks = await self._split_large_section(
                    section, content, start_offset, len(chunks)
                )
                chunks.extend(section_chunks)
        
        return chunks
    
    async def _split_large_section(
        self, 
        section: Dict[str, Any], 
        content: str,
        start_offset: int,
        chunk_id_offset: int
    ) -> List:
        """Split a large section into multiple chunks"""
        from ..chunker import ContentChunk  # Import here to avoid circular imports
        
        chunks = []
        header = section['header']
        elements = section['elements']
        
        # Always include header in first chunk if present
        current_chunk_elements = []
        current_size = 0
        
        if header:
            current_chunk_elements.append(header)
            current_size = header.size
        
        for element in elements:
            # Check if we can add this element to current chunk
            if (current_size + element.size <= self.config.max_chunk_size or
                not current_chunk_elements):  # Always include at least one element
                
                current_chunk_elements.append(element)
                current_size += element.size
            else:
                # Create chunk with current elements
                if current_chunk_elements:
                    chunk = self._create_chunk_from_elements(
                        current_chunk_elements, content, start_offset,
                        chunk_id_offset + len(chunks), section
                    )
                    chunks.append(chunk)
                
                # Start new chunk with current element
                current_chunk_elements = [element]
                current_size = element.size
        
        # Create final chunk
        if current_chunk_elements:
            chunk = self._create_chunk_from_elements(
                current_chunk_elements, content, start_offset,
                chunk_id_offset + len(chunks), section
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_elements(
        self,
        elements: List[StructuralElement],
        content: str,
        start_offset: int,
        chunk_index: int,
        section: Dict[str, Any]
    ):
        """Create a chunk from a list of structural elements"""
        from ..chunker import ContentChunk  # Import here to avoid circular imports
        
        chunk_start = elements[0].start_pos
        chunk_end = elements[-1].end_pos
        chunk_content = content[chunk_start:chunk_end]
        
        # Determine chunk type based on primary element type
        element_types = [e.element_type.value for e in elements]
        if 'code_block' in element_types:
            chunk_type = 'code'
        elif 'table' in element_types:
            chunk_type = 'table'
        elif 'list' in element_types:
            chunk_type = 'list'
        elif 'header' in element_types:
            chunk_type = 'section'
        else:
            chunk_type = 'mixed'
        
        chunk = ContentChunk(
            content=chunk_content,
            chunk_id=f"structural_{chunk_index}",
            chunk_index=chunk_index,
            start_char=start_offset + chunk_start,
            end_char=start_offset + chunk_end,
            chunk_type=chunk_type,
            section_title=section['title'],
            section_level=section['level']
        )
        
        # Add detailed metadata
        chunk.metadata = {
            'element_types': element_types,
            'element_count': len(elements),
            'has_header': any(e.element_type == StructureType.HEADER for e in elements),
            'has_code': any(e.element_type == StructureType.CODE_BLOCK for e in elements),
            'has_table': any(e.element_type == StructureType.TABLE for e in elements),
            'has_list': any(e.element_type == StructureType.LIST for e in elements)
        }
        
        # Add specific metadata for code chunks
        code_elements = [e for e in elements if e.element_type == StructureType.CODE_BLOCK]
        if code_elements:
            languages = list(set(e.language for e in code_elements if e.language))
            chunk.metadata['languages'] = languages
            chunk.language = languages[0] if languages else None
        
        return chunk
    
    async def _optimize_chunk_sizes(self, chunks: List, content: str) -> List:
        """Optimize chunk sizes while respecting structural boundaries"""
        if not chunks:
            return chunks
        
        optimized_chunks = []
        
        for chunk in chunks:
            if chunk.size > self.config.max_chunk_size:
                # Try to split large chunks at structural boundaries
                split_chunks = await self._split_oversized_chunk(chunk, content)
                optimized_chunks.extend(split_chunks)
            elif (chunk.size < self.config.min_chunk_size and 
                  chunk.chunk_type != 'code'):  # Don't merge code chunks
                # Mark for potential merging
                chunk.metadata['needs_merge'] = True
                optimized_chunks.append(chunk)
            else:
                optimized_chunks.append(chunk)
        
        # Merge small adjacent chunks
        optimized_chunks = self._merge_small_chunks(optimized_chunks)
        
        return optimized_chunks
    
    async def _split_oversized_chunk(self, chunk, content: str) -> List:
        """Split a chunk that exceeds maximum size"""
        from ..chunker import ContentChunk  # Import here to avoid circular imports
        
        # Simple splitting strategy - could be enhanced
        chunk_content = chunk.content
        target_size = self.config.target_chunk_size
        
        # Try to split at paragraph boundaries first
        paragraphs = chunk_content.split('\n\n')
        
        sub_chunks = []
        current_content = ""
        current_start = chunk.start_char
        
        for paragraph in paragraphs:
            potential_content = current_content + ('\n\n' if current_content else '') + paragraph
            
            if len(potential_content) > target_size and current_content:
                # Create sub-chunk
                sub_chunk = ContentChunk(
                    content=current_content.strip(),
                    chunk_id=f"{chunk.chunk_id}_split_{len(sub_chunks)}",
                    chunk_index=len(sub_chunks),
                    start_char=current_start,
                    end_char=current_start + len(current_content),
                    chunk_type=f"{chunk.chunk_type}_split",
                    section_title=chunk.section_title,
                    section_level=chunk.section_level
                )
                sub_chunk.metadata = chunk.metadata.copy()
                sub_chunks.append(sub_chunk)
                
                current_content = paragraph
                current_start = current_start + len(current_content) + 2  # +2 for \n\n
            else:
                current_content = potential_content
        
        # Add final sub-chunk
        if current_content.strip():
            sub_chunk = ContentChunk(
                content=current_content.strip(),
                chunk_id=f"{chunk.chunk_id}_split_{len(sub_chunks)}",
                chunk_index=len(sub_chunks),
                start_char=current_start,
                end_char=chunk.end_char,
                chunk_type=f"{chunk.chunk_type}_split",
                section_title=chunk.section_title,
                section_level=chunk.section_level
            )
            sub_chunk.metadata = chunk.metadata.copy()
            sub_chunks.append(sub_chunk)
        
        return sub_chunks if sub_chunks else [chunk]
    
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
                
                next_chunk = chunks[i + 1]
                
                # Only merge if chunks are compatible
                if self._can_merge_chunks(current_chunk, next_chunk):
                    merged_chunk = self._merge_two_chunks(current_chunk, next_chunk)
                    merged_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk
                else:
                    current_chunk.metadata.pop('needs_merge', None)
                    merged_chunks.append(current_chunk)
                    i += 1
            else:
                current_chunk.metadata.pop('needs_merge', None)
                merged_chunks.append(current_chunk)
                i += 1
        
        return merged_chunks
    
    def _can_merge_chunks(self, chunk1, chunk2) -> bool:
        """Check if two chunks can be merged"""
        # Don't merge chunks with different types
        if chunk1.chunk_type != chunk2.chunk_type:
            return False
        
        # Don't merge if combined size would be too large
        if chunk1.size + chunk2.size > self.config.max_chunk_size:
            return False
        
        # Don't merge code chunks (they should remain separate)
        if chunk1.chunk_type == 'code':
            return False
        
        # Don't merge if they're from different sections
        if chunk1.section_title != chunk2.section_title:
            return False
        
        return True
    
    def _merge_two_chunks(self, chunk1, chunk2):
        """Merge two compatible chunks"""
        merged_content = chunk1.content + '\n\n' + chunk2.content
        
        chunk1.content = merged_content
        chunk1.end_char = chunk2.end_char
        chunk1.chunk_type = 'merged'
        chunk1.metadata = chunk1.metadata.copy()
        chunk1.metadata.update(chunk2.metadata)
        chunk1.metadata.pop('needs_merge', None)
        
        return chunk1


if __name__ == "__main__":
    # Test structural chunking
    import asyncio
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        target_chunk_size: int = 1500
        max_chunk_size: int = 2000
        min_chunk_size: int = 200
        respect_headers: bool = True
        respect_code_blocks: bool = True
        respect_tables: bool = True
        respect_lists: bool = True
    
    @dataclass
    class MockDocStructure:
        content_type: str = 'markdown'
        language: Optional[str] = None
    
    async def test_structural_chunking():
        test_content = """
        # Main Title
        
        This is the introduction paragraph that explains the overall purpose
        of this document and provides context for the reader.
        
        ## Code Examples
        
        Here are some code examples:
        
        ```python
        def hello_world():
            print("Hello, World!")
            return True
        
        def another_function():
            return "Another example"
        ```
        
        ## Data Tables
        
        | Column 1 | Column 2 | Column 3 |
        |----------|----------|----------|
        | Value 1  | Value 2  | Value 3  |
        | Value 4  | Value 5  | Value 6  |
        
        ## Lists and Items
        
        Here's an unordered list:
        
        - First item with some explanation
        - Second item with more details
        - Third item with additional information
        
        And here's an ordered list:
        
        1. Step one of the process
        2. Step two with more complexity
        3. Final step to complete
        
        ## Conclusion
        
        This concludes our example document with various structural elements
        that should be preserved during chunking.
        """
        
        config = MockConfig()
        doc_structure = MockDocStructure()
        
        chunker = StructuralChunker(config)
        chunks = await chunker.chunk_content(test_content, doc_structure)
        
        print(f"Created {len(chunks)} structural chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({chunk.chunk_type}):")
            print(f"  Size: {chunk.size} chars")
            print(f"  Section: {chunk.section_title or 'None'}")
            print(f"  Level: {chunk.section_level}")
            print(f"  Metadata: {chunk.metadata}")
            print(f"  Content preview: {chunk.content[:100]}...")
    
    asyncio.run(test_structural_chunking())