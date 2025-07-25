"""
Text Processing Utilities for Indexing

This module provides text preprocessing functionality to prepare content
for indexing, including normalization, cleaning, and extraction of
searchable text from various formats.
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessedText:
    """Container for processed text with metadata"""
    content: str
    title: Optional[str] = None
    headings: List[str] = None
    word_count: int = 0
    language: Optional[str] = None
    code_blocks: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.headings is None:
            self.headings = []
        if self.code_blocks is None:
            self.code_blocks = []


class TextProcessor:
    """
    Process text content for indexing.
    
    Handles:
    - Text normalization and cleaning
    - Markdown parsing
    - Code block extraction
    - Language detection
    - Text statistics
    """
    
    def __init__(self):
        """Initialize text processor"""
        # Patterns for markdown processing
        self.heading_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
        self.link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        self.image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        
        # Patterns for cleaning
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
    def process(self, content: str, file_path: str = '') -> ProcessedText:
        """
        Process text content for indexing.
        
        Args:
            content: Raw text content
            file_path: Path to the file (for context)
            
        Returns:
            ProcessedText object with extracted information
        """
        if not content:
            return ProcessedText(content='', word_count=0)
        
        # Detect content type from file extension
        is_markdown = file_path.lower().endswith(('.md', '.markdown'))
        
        # Extract metadata
        title = self._extract_title(content, file_path, is_markdown)
        headings = self._extract_headings(content) if is_markdown else []
        code_blocks = self._extract_code_blocks(content) if is_markdown else []
        
        # Process content
        processed_content = self._process_content(content, is_markdown)
        
        # Calculate statistics
        word_count = self._count_words(processed_content)
        language = self._detect_language(content, file_path)
        
        return ProcessedText(
            content=processed_content,
            title=title,
            headings=headings,
            word_count=word_count,
            language=language,
            code_blocks=code_blocks
        )
    
    def _extract_title(self, content: str, file_path: str, is_markdown: bool) -> Optional[str]:
        """Extract document title"""
        if is_markdown:
            # Look for first H1 heading
            first_h1 = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if first_h1:
                return first_h1.group(1).strip()
        
        # Fall back to filename without extension
        if file_path:
            import os
            basename = os.path.basename(file_path)
            name, _ = os.path.splitext(basename)
            return name.replace('-', ' ').replace('_', ' ').title()
        
        return None
    
    def _extract_headings(self, content: str) -> List[str]:
        """Extract all headings from markdown content"""
        headings = []
        for match in self.heading_pattern.finditer(content):
            heading_text = match.group(1).strip()
            # Remove markdown formatting from heading
            heading_text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', heading_text)
            heading_text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', heading_text)
            heading_text = re.sub(r'`([^`]+)`', r'\1', heading_text)
            headings.append(heading_text)
        return headings
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks with language information"""
        code_blocks = []
        for match in self.code_block_pattern.finditer(content):
            language = match.group(1) or 'unknown'
            code = match.group(2)
            code_blocks.append({
                'language': language,
                'code': code,
                'length': len(code)
            })
        return code_blocks
    
    def _process_content(self, content: str, is_markdown: bool) -> str:
        """Process and clean content for indexing"""
        processed = content
        
        if is_markdown:
            # Remove code blocks (they're indexed separately)
            processed = self.code_block_pattern.sub('[CODE_BLOCK]', processed)
            
            # Convert markdown links to just link text
            processed = self.link_pattern.sub(r'\1', processed)
            
            # Remove images
            processed = self.image_pattern.sub('', processed)
            
            # Remove inline code formatting but keep content
            processed = self.inline_code_pattern.sub(r'\1', processed)
            
            # Remove markdown formatting
            processed = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', processed)  # Bold/italic
            processed = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', processed)  # Bold/italic
            processed = re.sub(r'^[-*+]\s+', '', processed, flags=re.MULTILINE)  # List items
            processed = re.sub(r'^\d+\.\s+', '', processed, flags=re.MULTILINE)  # Numbered lists
            processed = re.sub(r'^>\s+', '', processed, flags=re.MULTILINE)  # Blockquotes
            processed = re.sub(r'^#{1,6}\s+', '', processed, flags=re.MULTILINE)  # Headings
            
        # General cleaning
        processed = self._clean_text(processed)
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better indexing"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[=\-_]{3,}', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        return text.strip()
    
    def _count_words(self, text: str) -> int:
        """Count words in text"""
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out very short words and numbers
        words = [w for w in words if len(w) > 1 and not w.isdigit()]
        return len(words)
    
    def _detect_language(self, content: str, file_path: str) -> Optional[str]:
        """Detect programming language or document language"""
        # For code files, use file extension
        import os
        ext = os.path.splitext(file_path)[1].lower()
        
        language_map = {
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
            '.sh': 'shell',
            '.bash': 'bash',
            '.ps1': 'powershell',
            '.sql': 'sql',
            '.md': 'markdown',
            '.rst': 'restructuredtext',
            '.tex': 'latex'
        }
        
        return language_map.get(ext)
    
    def extract_chunks(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split content into overlapping chunks for indexing.
        
        Args:
            content: Text content to chunk
            chunk_size: Target size for each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not content or chunk_size <= 0:
            return []
        
        chunks = []
        sentences = self._split_sentences(content)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Calculate overlap
                overlap_size = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    overlap_size += len(sent)
                    overlap_sentences.insert(0, sent)
                    if overlap_size >= overlap:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - could be enhanced with NLTK or spaCy
        sentence_endings = re.compile(r'[.!?]+[\s\n]+')
        sentences = sentence_endings.split(text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def extract_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Extract additional metadata from content.
        
        Args:
            content: Text content
            file_path: File path
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Extract URLs
        urls = self.url_pattern.findall(content)
        if urls:
            metadata['urls'] = list(set(urls))
        
        # Extract emails (but anonymize for privacy)
        emails = self.email_pattern.findall(content)
        if emails:
            metadata['has_emails'] = True
            metadata['email_count'] = len(set(emails))
        
        # Check for common patterns
        metadata['has_todo'] = bool(re.search(r'\bTODO\b', content, re.IGNORECASE))
        metadata['has_fixme'] = bool(re.search(r'\bFIXME\b', content, re.IGNORECASE))
        metadata['has_bug'] = bool(re.search(r'\bBUG\b', content, re.IGNORECASE))
        
        # Extract version numbers
        versions = re.findall(r'\b\d+\.\d+(?:\.\d+)?\b', content)
        if versions:
            metadata['versions'] = list(set(versions))
        
        return metadata


def test_text_processor():
    """Test text processing functionality"""
    processor = TextProcessor()
    
    # Test markdown content
    test_content = """
# Main Title

This is a test document with **bold** and *italic* text.

## Section 1

Here's some content with a [link](https://example.com) and `inline code`.

```python
def hello_world():
    print("Hello, World!")
```

### Subsection

- List item 1
- List item 2

TODO: Add more content
    """
    
    result = processor.process(test_content, 'test.md')
    
    print("Processed Text:")
    print(f"  Title: {result.title}")
    print(f"  Headings: {result.headings}")
    print(f"  Word count: {result.word_count}")
    print(f"  Language: {result.language}")
    print(f"  Code blocks: {len(result.code_blocks)}")
    print(f"  Content preview: {result.content[:200]}...")
    
    # Test chunking
    chunks = processor.extract_chunks(result.content, chunk_size=100, overlap=20)
    print(f"\nChunks created: {len(chunks)}")
    
    # Test metadata extraction
    metadata = processor.extract_metadata(test_content, 'test.md')
    print(f"\nMetadata: {metadata}")


if __name__ == "__main__":
    test_text_processor()