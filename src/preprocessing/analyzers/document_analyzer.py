"""
Document Structure Analyzer

This module analyzes document structure, content type, and characteristics
to inform optimal chunking strategies and content processing decisions.
"""

import logging
import re
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Document content types"""
    MARKDOWN = "markdown"
    CODE = "code" 
    TEXT = "text"
    STRUCTURED_TEXT = "structured_text"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class DocumentSection:
    """Represents a section of the document"""
    title: Optional[str]
    level: int = 0                     # Header level (0 = no header)
    start_pos: int = 0
    end_pos: int = 0
    section_type: str = "text"         # text, code, table, list
    subsections: List['DocumentSection'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        return self.end_pos - self.start_pos
    
    @property 
    def has_subsections(self) -> bool:
        return len(self.subsections) > 0


@dataclass
class DocumentStructure:
    """Complete document structure analysis"""
    # Content characteristics
    content_type: ContentType
    language: Optional[str] = None
    total_length: int = 0
    
    # Structure analysis
    sections: List[DocumentSection] = field(default_factory=list)
    is_highly_structured: bool = False
    has_clear_sections: bool = False
    max_header_level: int = 0
    
    # Content statistics
    word_count: int = 0
    line_count: int = 0
    paragraph_count: int = 0
    average_paragraph_length: float = 0.0
    
    # Special elements
    has_code_blocks: bool = False
    has_tables: bool = False
    has_lists: bool = False
    has_images: bool = False
    has_links: bool = False
    
    # Code-specific analysis
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    
    # Quality indicators
    structure_complexity: float = 0.0   # 0-1 scale
    content_density: float = 0.0        # 0-1 scale
    
    # Processing recommendations
    recommended_strategy: str = "hybrid"
    optimal_chunk_size: int = 1500
    
    def get_section_at_position(self, position: int) -> Optional[DocumentSection]:
        """Get the section containing a specific character position"""
        for section in self.sections:
            if section.start_pos <= position < section.end_pos:
                # Check subsections first
                for subsection in section.subsections:
                    if subsection.start_pos <= position < subsection.end_pos:
                        return subsection
                return section
        return None


class DocumentAnalyzer:
    """
    Analyzes document structure and characteristics to inform
    optimal chunking strategies and processing decisions.
    """
    
    def __init__(self):
        """Initialize document analyzer"""
        
        # Regex patterns for structure detection
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
        self.inline_code_pattern = re.compile(r'`([^`\n]+)`')
        self.table_pattern = re.compile(r'^\|.*\|$', re.MULTILINE)
        self.list_pattern = re.compile(r'^\s*[-*+]\s+|^\s*\d+\.\s+', re.MULTILINE)
        self.image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        self.link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        
        # Programming language patterns
        self.language_patterns = {
            'python': {
                'function': re.compile(r'^\s*def\s+\w+\s*\(', re.MULTILINE),
                'class': re.compile(r'^\s*class\s+\w+', re.MULTILINE),
                'import': re.compile(r'^\s*(import|from)\s+', re.MULTILINE)
            },
            'javascript': {
                'function': re.compile(r'^\s*(function\s+\w+|const\s+\w+\s*=.*=>|\w+\s*:\s*function)', re.MULTILINE),
                'class': re.compile(r'^\s*class\s+\w+', re.MULTILINE),
                'import': re.compile(r'^\s*(import|require)', re.MULTILINE)
            },
            'java': {
                'function': re.compile(r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(', re.MULTILINE),
                'class': re.compile(r'^\s*(public|private)?\s*class\s+\w+', re.MULTILINE),
                'import': re.compile(r'^\s*import\s+', re.MULTILINE)
            }
        }
        
        # File extension to language mapping
        self.extension_language_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.sh': 'bash',
            '.ps1': 'powershell',
            '.sql': 'sql',
            '.md': 'markdown',
            '.rst': 'restructuredtext',
            '.tex': 'latex',
            '.html': 'html',
            '.css': 'css',
            '.xml': 'xml',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml'
        }
        
        logger.debug("DocumentAnalyzer initialized")
    
    async def analyze_document(
        self, 
        content: str, 
        file_path: Optional[str] = None
    ) -> DocumentStructure:
        """
        Analyze document structure and characteristics
        
        Args:
            content: Document content to analyze
            file_path: Optional file path for type detection
            
        Returns:
            DocumentStructure with complete analysis
        """
        if not content:
            return DocumentStructure(
                content_type=ContentType.UNKNOWN,
                total_length=0
            )
        
        logger.debug(f"Analyzing document structure ({len(content)} chars)")
        
        # Initialize structure
        structure = DocumentStructure(
            content_type=ContentType.UNKNOWN,
            total_length=len(content)
        )
        
        # Detect content type and language
        structure.content_type, structure.language = self._detect_content_type(
            content, file_path
        )
        
        # Basic statistics
        structure.word_count = self._count_words(content)
        structure.line_count = content.count('\n') + 1
        structure.paragraph_count = len(re.split(r'\n\s*\n', content.strip()))
        
        if structure.paragraph_count > 0:
            structure.average_paragraph_length = structure.word_count / structure.paragraph_count
        
        # Detect special elements
        structure.has_code_blocks = bool(self.code_block_pattern.search(content))
        structure.has_tables = bool(self.table_pattern.search(content))
        structure.has_lists = bool(self.list_pattern.search(content))
        structure.has_images = bool(self.image_pattern.search(content))
        structure.has_links = bool(self.link_pattern.search(content))
        
        # Analyze structure based on content type
        if structure.content_type == ContentType.MARKDOWN:
            await self._analyze_markdown_structure(content, structure)
        elif structure.content_type == ContentType.CODE:
            await self._analyze_code_structure(content, structure)
        else:
            await self._analyze_text_structure(content, structure)
        
        # Calculate quality metrics
        structure.structure_complexity = self._calculate_structure_complexity(structure)
        structure.content_density = self._calculate_content_density(content, structure)
        
        # Generate recommendations
        structure.recommended_strategy = self._recommend_chunking_strategy(structure)
        structure.optimal_chunk_size = self._recommend_chunk_size(structure)
        
        logger.debug(
            f"Document analysis complete: {structure.content_type.value}, "
            f"{len(structure.sections)} sections, complexity={structure.structure_complexity:.2f}"
        )
        
        return structure
    
    def _detect_content_type(
        self, 
        content: str, 
        file_path: Optional[str] = None
    ) -> Tuple[ContentType, Optional[str]]:
        """Detect content type and programming language"""
        
        language = None
        
        # Try file extension first
        if file_path:
            ext = Path(file_path).suffix.lower()
            language = self.extension_language_map.get(ext)
            
            if ext in ['.md', '.markdown']:
                return ContentType.MARKDOWN, 'markdown'
            elif ext in ['.rst']:
                return ContentType.STRUCTURED_TEXT, 'restructuredtext'
            elif language:
                return ContentType.CODE, language
        
        # Content-based detection
        content_lower = content.lower()
        
        # Check for markdown indicators
        markdown_score = 0
        if re.search(r'^#{1,6}\s+', content, re.MULTILINE):
            markdown_score += 3
        if '```' in content:
            markdown_score += 2
        if re.search(r'\[.*\]\(.*\)', content):
            markdown_score += 1
        if re.search(r'^\s*[-*+]\s+', content, re.MULTILINE):
            markdown_score += 1
        if re.search(r'^\|.*\|$', content, re.MULTILINE):
            markdown_score += 1
        
        if markdown_score >= 3:
            return ContentType.MARKDOWN, 'markdown'
        
        # Check for code patterns
        code_indicators = [
            'def ', 'function ', 'class ', 'import ', 'from ',
            'public class', 'private ', 'void ', 'int main',
            '#include', 'using namespace', '<?php', '<!DOCTYPE'
        ]
        
        code_score = sum(1 for indicator in code_indicators if indicator in content_lower)
        
        # Check for brackets and syntax patterns
        brace_ratio = (content.count('{') + content.count('}')) / max(len(content), 1)
        semicolon_ratio = content.count(';') / max(len(content), 1)
        
        if code_score >= 2 or brace_ratio > 0.01 or semicolon_ratio > 0.005:
            # Try to detect specific language
            if not language:
                language = self._detect_programming_language(content)
            return ContentType.CODE, language
        
        # Check for structured text
        if content.count('\n\n') > len(content) / 200:  # Many paragraphs
            return ContentType.STRUCTURED_TEXT, None
        
        # Check for mixed content
        if markdown_score > 0 and code_score > 0:
            return ContentType.MIXED, None
        
        return ContentType.TEXT, None
    
    def _detect_programming_language(self, content: str) -> Optional[str]:
        """Detect programming language from content patterns"""
        
        language_scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = 0
            for pattern_type, pattern in patterns.items():
                matches = len(pattern.findall(content))
                score += matches
            
            if score > 0:
                language_scores[lang] = score
        
        # Additional heuristics
        if 'print(' in content and 'def ' in content:
            language_scores['python'] = language_scores.get('python', 0) + 2
        
        if '{' in content and '}' in content and ';' in content:
            if 'System.out.println' in content:
                language_scores['java'] = language_scores.get('java', 0) + 3
            elif 'console.log' in content:
                language_scores['javascript'] = language_scores.get('javascript', 0) + 3
        
        if language_scores:
            return max(language_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    async def _analyze_markdown_structure(
        self, 
        content: str, 
        structure: DocumentStructure
    ):
        """Analyze markdown document structure"""
        
        # Find all headers
        headers = []
        for match in self.header_pattern.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            position = match.start()
            
            headers.append({
                'level': level,
                'title': title,
                'position': position,
                'end_position': match.end()
            })
        
        structure.max_header_level = max((h['level'] for h in headers), default=0)
        structure.has_clear_sections = len(headers) > 0
        structure.is_highly_structured = len(headers) > 2 and structure.max_header_level > 1
        
        # Build section hierarchy
        if headers:
            structure.sections = self._build_section_hierarchy(headers, content)
        else:
            # No headers, treat as single section
            section = DocumentSection(
                title=None,
                level=0,
                start_pos=0,
                end_pos=len(content),
                section_type="text"
            )
            structure.sections = [section]
        
        # Analyze content within sections
        for section in structure.sections:
            self._analyze_section_content(section, content)
    
    async def _analyze_code_structure(
        self, 
        content: str, 
        structure: DocumentStructure
    ):
        """Analyze code file structure"""
        
        if not structure.language:
            return
        
        patterns = self.language_patterns.get(structure.language, {})
        
        # Count functions and classes
        if 'function' in patterns:
            structure.function_count = len(patterns['function'].findall(content))
        if 'class' in patterns:
            structure.class_count = len(patterns['class'].findall(content))
        if 'import' in patterns:
            structure.import_count = len(patterns['import'].findall(content))
        
        structure.is_highly_structured = (
            structure.function_count > 2 or 
            structure.class_count > 0
        )
        
        # Create sections based on code structure
        structure.sections = await self._extract_code_sections(content, structure.language)
    
    async def _analyze_text_structure(
        self, 
        content: str, 
        structure: DocumentStructure
    ):
        """Analyze plain text structure"""
        
        # Simple paragraph-based sectioning
        paragraphs = re.split(r'\n\s*\n', content.strip())
        
        current_pos = 0
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Find paragraph position in original content
            para_start = content.find(paragraph, current_pos)
            para_end = para_start + len(paragraph)
            
            section = DocumentSection(
                title=f"Paragraph {i+1}",
                level=0,
                start_pos=para_start,
                end_pos=para_end,
                section_type="text"
            )
            
            structure.sections.append(section)
            current_pos = para_end
        
        structure.has_clear_sections = len(structure.sections) > 1
        structure.is_highly_structured = False
    
    def _build_section_hierarchy(
        self, 
        headers: List[Dict], 
        content: str
    ) -> List[DocumentSection]:
        """Build hierarchical section structure from headers"""
        
        sections = []
        section_stack = []
        
        for i, header in enumerate(headers):
            # Determine section content bounds
            start_pos = header['position']
            if i + 1 < len(headers):
                end_pos = headers[i + 1]['position']
            else:
                end_pos = len(content)
            
            section = DocumentSection(
                title=header['title'],
                level=header['level'],
                start_pos=start_pos,
                end_pos=end_pos,
                section_type="section"
            )
            
            # Handle hierarchy
            while (section_stack and 
                   section_stack[-1].level >= section.level):
                section_stack.pop()
            
            if section_stack:
                # Add as subsection
                section_stack[-1].subsections.append(section)
            else:
                # Add as top-level section
                sections.append(section)
            
            section_stack.append(section)
        
        return sections
    
    def _analyze_section_content(self, section: DocumentSection, content: str):
        """Analyze content within a section"""
        section_content = content[section.start_pos:section.end_pos]
        
        # Detect content type within section
        if self.code_block_pattern.search(section_content):
            section.metadata['has_code'] = True
        
        if self.table_pattern.search(section_content):
            section.metadata['has_table'] = True
            section.section_type = "table"
        
        if self.list_pattern.search(section_content):
            section.metadata['has_list'] = True
            if section.section_type == "section":
                section.section_type = "list"
        
        # Count elements
        section.metadata['word_count'] = self._count_words(section_content)
        section.metadata['line_count'] = section_content.count('\n') + 1
    
    async def _extract_code_sections(
        self, 
        content: str, 
        language: str
    ) -> List[DocumentSection]:
        """Extract sections from code based on functions and classes"""
        
        sections = []
        patterns = self.language_patterns.get(language, {})
        
        # Find all structural elements
        elements = []
        
        if 'function' in patterns:
            for match in patterns['function'].finditer(content):
                elements.append({
                    'type': 'function',
                    'start': match.start(),
                    'name': self._extract_function_name(match.group(0), language)
                })
        
        if 'class' in patterns:
            for match in patterns['class'].finditer(content):
                elements.append({
                    'type': 'class',
                    'start': match.start(),
                    'name': self._extract_class_name(match.group(0), language)
                })
        
        # Sort by position
        elements.sort(key=lambda x: x['start'])
        
        # Create sections
        for i, element in enumerate(elements):
            start_pos = element['start']
            if i + 1 < len(elements):
                end_pos = elements[i + 1]['start']
            else:
                end_pos = len(content)
            
            section = DocumentSection(
                title=f"{element['type'].title()}: {element['name']}",
                level=1 if element['type'] == 'class' else 2,
                start_pos=start_pos,
                end_pos=end_pos,
                section_type=element['type'],
                metadata={'element_type': element['type'], 'name': element['name']}
            )
            
            sections.append(section)
        
        return sections
    
    def _extract_function_name(self, function_def: str, language: str) -> str:
        """Extract function name from function definition"""
        if language == 'python':
            match = re.search(r'def\s+(\w+)', function_def)
        elif language == 'javascript':
            match = re.search(r'function\s+(\w+)|const\s+(\w+)\s*=', function_def)
        elif language == 'java':
            match = re.search(r'\w+\s+(\w+)\s*\(', function_def)
        else:
            match = re.search(r'(\w+)\s*\(', function_def)
        
        return match.group(1) if match else 'unknown'
    
    def _extract_class_name(self, class_def: str, language: str) -> str:
        """Extract class name from class definition"""
        match = re.search(r'class\s+(\w+)', class_def, re.IGNORECASE)
        return match.group(1) if match else 'unknown'
    
    def _calculate_structure_complexity(self, structure: DocumentStructure) -> float:
        """Calculate structural complexity score (0-1)"""
        complexity = 0.0
        
        # Header hierarchy contributes to complexity
        if structure.max_header_level > 0:
            complexity += min(structure.max_header_level / 6.0, 0.3)
        
        # Number of sections
        section_count = len(structure.sections)
        if section_count > 0:
            complexity += min(section_count / 20.0, 0.2)
        
        # Special elements add complexity
        if structure.has_code_blocks:
            complexity += 0.15
        if structure.has_tables:
            complexity += 0.1
        if structure.has_lists:
            complexity += 0.05
        
        # Code structure
        if structure.function_count > 0:
            complexity += min(structure.function_count / 10.0, 0.2)
        if structure.class_count > 0:
            complexity += min(structure.class_count / 5.0, 0.15)
        
        return min(complexity, 1.0)
    
    def _calculate_content_density(
        self, 
        content: str, 
        structure: DocumentStructure
    ) -> float:
        """Calculate content information density (0-1)"""
        
        if not content:
            return 0.0
        
        # Word density
        word_ratio = structure.word_count / len(content) if content else 0
        
        # Unique word ratio
        words = re.findall(r'\b\w+\b', content.lower())
        unique_ratio = len(set(words)) / len(words) if words else 0
        
        # Special content ratio
        special_chars = len(re.findall(r'[^\w\s]', content))
        special_ratio = special_chars / len(content) if content else 0
        
        # Combine metrics
        density = (word_ratio * 0.5 + unique_ratio * 0.3 + special_ratio * 0.2)
        
        return min(density * 2, 1.0)  # Scale up and cap at 1.0
    
    def _recommend_chunking_strategy(self, structure: DocumentStructure) -> str:
        """Recommend optimal chunking strategy based on analysis"""
        
        if structure.content_type == ContentType.CODE:
            return "structural"
        
        if structure.is_highly_structured and structure.has_clear_sections:
            return "structural"
        
        if structure.content_type == ContentType.MARKDOWN:
            if structure.structure_complexity > 0.5:
                return "hybrid"
            else:
                return "structural"
        
        if structure.average_paragraph_length > 300:
            return "semantic"
        
        return "hybrid"
    
    def _recommend_chunk_size(self, structure: DocumentStructure) -> int:
        """Recommend optimal chunk size based on document characteristics"""
        
        base_size = 1500
        
        # Adjust based on content type
        if structure.content_type == ContentType.CODE:
            # Smaller chunks for code to keep functions/classes together
            base_size = 1200
        elif structure.content_type == ContentType.MARKDOWN:
            if structure.is_highly_structured:
                # Larger chunks for well-structured markdown
                base_size = 1800
        
        # Adjust based on average paragraph length
        if structure.average_paragraph_length > 500:
            base_size = int(base_size * 1.2)
        elif structure.average_paragraph_length < 100:
            base_size = int(base_size * 0.8)
        
        # Adjust based on complexity
        if structure.structure_complexity > 0.7:
            base_size = int(base_size * 0.9)  # Smaller chunks for complex structure
        elif structure.structure_complexity < 0.3:
            base_size = int(base_size * 1.1)  # Larger chunks for simple structure
        
        # Ensure reasonable bounds
        return max(800, min(2500, base_size))
    
    def _count_words(self, text: str) -> int:
        """Count words in text"""
        return len(re.findall(r'\b\w+\b', text))


if __name__ == "__main__":
    # Test document analysis
    import asyncio
    
    async def test_document_analysis():
        # Test markdown content
        markdown_content = """
        # Main Title
        
        This is an introduction paragraph with some **bold** text and a [link](https://example.com).
        
        ## Section 1: Overview
        
        Here's some content with a list:
        
        - Item 1
        - Item 2
        - Item 3
        
        ### Subsection 1.1
        
        More detailed content here.
        
        ## Section 2: Code Examples
        
        Here's a code block:
        
        ```python
        def hello_world():
            print("Hello, World!")
            return True
        ```
        
        ## Section 3: Data
        
        | Column 1 | Column 2 |
        |----------|----------|
        | Value 1  | Value 2  |
        | Value 3  | Value 4  |
        """
        
        # Test code content
        python_content = """
        import os
        import sys
        from typing import List, Dict
        
        class DocumentProcessor:
            def __init__(self, config):
                self.config = config
                self.processed_count = 0
            
            def process_document(self, content: str) -> Dict:
                '''Process a document and return analysis'''
                result = {
                    'length': len(content),
                    'word_count': len(content.split())
                }
                self.processed_count += 1
                return result
            
            def get_stats(self):
                return {'processed': self.processed_count}
        
        def main():
            processor = DocumentProcessor({'debug': True})
            content = "Sample document content"
            result = processor.process_document(content)
            print(f"Processed: {result}")
        
        if __name__ == "__main__":
            main()
        """
        
        analyzer = DocumentAnalyzer()
        
        print("=== Markdown Analysis ===")
        md_structure = await analyzer.analyze_document(markdown_content, "test.md")
        print(f"Content type: {md_structure.content_type.value}")
        print(f"Language: {md_structure.language}")
        print(f"Sections: {len(md_structure.sections)}")
        print(f"Max header level: {md_structure.max_header_level}")
        print(f"Is structured: {md_structure.is_highly_structured}")
        print(f"Complexity: {md_structure.structure_complexity:.2f}")
        print(f"Density: {md_structure.content_density:.2f}")
        print(f"Recommended strategy: {md_structure.recommended_strategy}")
        print(f"Optimal chunk size: {md_structure.optimal_chunk_size}")
        
        print("\n=== Code Analysis ===")
        code_structure = await analyzer.analyze_document(python_content, "test.py")
        print(f"Content type: {code_structure.content_type.value}")
        print(f"Language: {code_structure.language}")
        print(f"Functions: {code_structure.function_count}")
        print(f"Classes: {code_structure.class_count}")
        print(f"Imports: {code_structure.import_count}")
        print(f"Sections: {len(code_structure.sections)}")
        print(f"Complexity: {code_structure.structure_complexity:.2f}")
        print(f"Recommended strategy: {code_structure.recommended_strategy}")
        
        print("\n=== Section Details ===")
        for i, section in enumerate(md_structure.sections[:3]):
            print(f"Section {i+1}: {section.title}")
            print(f"  Level: {section.level}")
            print(f"  Type: {section.section_type}")
            print(f"  Size: {section.size} chars")
            print(f"  Subsections: {len(section.subsections)}")
    
    asyncio.run(test_document_analysis())