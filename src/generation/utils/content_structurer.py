"""
Content structuring utilities for document generation.

This module provides utilities for converting raw content (chat conversations,
markdown, plain text) into structured DocumentContent that can be processed
by document generators.
"""

import re
import html
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import markdown
from markdown.extensions import toc, tables, codehilite

from ..models import (
    DocumentContent,
    DocumentSection,
    ContentElement,
    ContentType,
    DocumentMetadata,
    TextStyle,
    ImageData,
    TableData,
    TableRow,
    TableCell,
    ListData,
    ListItem,
    ListType,
    StyleWeight,
    StyleEmphasis,
    create_heading,
    create_paragraph,
    create_image,
    create_table,
    create_list
)

logger = logging.getLogger(__name__)


@dataclass
class StructuringOptions:
    """Options for content structuring."""
    # Processing options
    detect_headings: bool = True
    detect_lists: bool = True
    detect_tables: bool = True
    detect_code_blocks: bool = True
    detect_images: bool = True
    
    # Formatting options
    preserve_formatting: bool = True
    merge_short_paragraphs: bool = False
    split_long_paragraphs: bool = True
    max_paragraph_length: int = 1000
    
    # Section organization
    auto_create_sections: bool = True
    section_heading_levels: List[int] = None
    max_section_depth: int = 4
    
    # Chat-specific options
    include_timestamps: bool = False
    include_user_names: bool = True
    group_consecutive_messages: bool = True
    
    # Content cleaning
    remove_html: bool = True
    normalize_whitespace: bool = True
    remove_empty_lines: bool = True
    
    def __post_init__(self):
        if self.section_heading_levels is None:
            self.section_heading_levels = [1, 2, 3, 4]


class ContentStructurer:
    """Main content structuring utility."""
    
    def __init__(self, options: Optional[StructuringOptions] = None):
        """
        Initialize content structurer.
        
        Args:
            options: Structuring options
        """
        self.options = options or StructuringOptions()
        self._markdown_processor = self._setup_markdown_processor()
    
    async def structure_chat_conversation(
        self, 
        messages: List[Dict[str, Any]],
        conversation_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentContent:
        """
        Structure chat conversation into document content.
        
        Args:
            messages: List of chat messages with 'role', 'content', 'timestamp'
            conversation_metadata: Optional conversation metadata
            
        Returns:
            Structured DocumentContent
        """
        # Create document metadata
        metadata = DocumentMetadata(
            title=conversation_metadata.get("title", "Chat Conversation") if conversation_metadata else "Chat Conversation",
            created_date=datetime.now()
        )
        
        if conversation_metadata:
            metadata.author = conversation_metadata.get("author")
            metadata.description = conversation_metadata.get("description")
            metadata.keywords = conversation_metadata.get("keywords", [])
        
        # Group and process messages
        message_groups = self._group_messages(messages) if self.options.group_consecutive_messages else [[msg] for msg in messages]
        
        sections = []
        current_section = None
        
        for i, group in enumerate(message_groups):
            # Create section for each group or conversation topic
            if self.options.auto_create_sections and self._should_create_new_section(group, current_section):
                if current_section:
                    sections.append(current_section)
                
                section_title = self._generate_section_title(group, i)
                current_section = DocumentSection(
                    title=section_title,
                    level=1
                )
            
            if not current_section:
                current_section = DocumentSection(title="Conversation", level=1)
            
            # Process message group
            elements = await self._process_message_group(group)
            current_section.elements.extend(elements)
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        document = DocumentContent(
            metadata=metadata,
            sections=sections
        )
        document.source_type = "chat"
        document.source_id = conversation_metadata.get("id") if conversation_metadata else None
        return document
    
    async def structure_markdown_content(
        self, 
        markdown_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentContent:
        """
        Structure markdown content into document content.
        
        Args:
            markdown_text: Raw markdown text
            metadata: Optional document metadata
            
        Returns:
            Structured DocumentContent
        """
        # Clean and prepare markdown
        cleaned_markdown = self._clean_content(markdown_text)
        
        # Parse markdown to HTML first to extract structure
        html_content = self._markdown_processor.convert(cleaned_markdown)
        
        # Extract metadata from markdown (front matter)
        doc_metadata = self._extract_markdown_metadata(cleaned_markdown, metadata)
        
        # Parse HTML structure
        sections = await self._parse_html_structure(html_content)
        
        document = DocumentContent(
            metadata=doc_metadata,
            sections=sections
        )
        document.source_type = "markdown"
        return document
    
    async def structure_plain_text(
        self, 
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentContent:
        """
        Structure plain text into document content.
        
        Args:
            text: Raw plain text
            metadata: Optional document metadata
            
        Returns:
            Structured DocumentContent
        """
        # Clean text
        cleaned_text = self._clean_content(text)
        
        # Create metadata
        doc_metadata = DocumentMetadata(
            title=metadata.get("title", "Document") if metadata else "Document",
            created_date=datetime.now()
        )
        
        if metadata:
            doc_metadata.author = metadata.get("author")
            doc_metadata.description = metadata.get("description")
            doc_metadata.keywords = metadata.get("keywords", [])
        
        # Detect structure in plain text
        sections = await self._parse_plain_text_structure(cleaned_text)
        
        document = DocumentContent(
            metadata=doc_metadata,
            sections=sections
        )
        document.source_type = "text"
        return document
    
    async def structure_mixed_content(
        self, 
        content_items: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentContent:
        """
        Structure mixed content (text, images, tables) into document content.
        
        Args:
            content_items: List of content items with 'type' and 'data'
            metadata: Optional document metadata
            
        Returns:
            Structured DocumentContent
        """
        doc_metadata = DocumentMetadata(
            title=metadata.get("title", "Document") if metadata else "Document",
            created_date=datetime.now()
        )
        
        if metadata:
            doc_metadata.author = metadata.get("author")
            doc_metadata.description = metadata.get("description")
            doc_metadata.keywords = metadata.get("keywords", [])
        
        sections = []
        current_section = DocumentSection(title="Content", level=1)
        
        for item in content_items:
            element = await self._process_content_item(item)
            if element:
                current_section.elements.append(element)
        
        if current_section.elements:
            sections.append(current_section)
        
        document = DocumentContent(
            metadata=doc_metadata,
            sections=sections
        )
        document.source_type = "mixed"
        return document
    
    def _setup_markdown_processor(self) -> markdown.Markdown:
        """Setup markdown processor with extensions."""
        extensions = ['extra', 'codehilite', 'toc', 'tables']
        extension_configs = {
            'codehilite': {
                'css_class': 'highlight',
                'use_pygments': True
            },
            'toc': {
                'permalink': True
            }
        }
        
        return markdown.Markdown(
            extensions=extensions,
            extension_configs=extension_configs
        )
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        if not content:
            return ""
        
        # Remove HTML if requested
        if self.options.remove_html:
            content = html.unescape(content)
            content = re.sub(r'<[^>]+>', '', content)
        
        # Normalize whitespace
        if self.options.normalize_whitespace:
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
        
        # Remove empty lines
        if self.options.remove_empty_lines:
            lines = content.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            content = '\n'.join(lines)
        
        return content
    
    def _group_messages(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group consecutive messages from same user."""
        if not messages:
            return []
        
        groups = []
        current_group = [messages[0]]
        
        for msg in messages[1:]:
            # Group if same role and within time window
            last_msg = current_group[-1]
            if (msg.get('role') == last_msg.get('role') and 
                self._messages_should_group(msg, last_msg)):
                current_group.append(msg)
            else:
                groups.append(current_group)
                current_group = [msg]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _messages_should_group(self, msg1: Dict[str, Any], msg2: Dict[str, Any]) -> bool:
        """Check if two messages should be grouped."""
        # Simple grouping logic - can be enhanced
        return (msg1.get('role') == msg2.get('role') and
                len(msg1.get('content', '')) < 500 and
                len(msg2.get('content', '')) < 500)
    
    def _should_create_new_section(
        self, 
        message_group: List[Dict[str, Any]], 
        current_section: Optional[DocumentSection]
    ) -> bool:
        """Determine if new section should be created."""
        if not current_section:
            return True
        
        # Create new section if topic seems to change
        # This is a simplified heuristic
        if len(current_section.elements) > 10:
            return True
        
        # Check if messages contain topic indicators
        first_message = message_group[0]
        content = first_message.get('content', '').lower()
        
        topic_indicators = ['let\'s discuss', 'moving on to', 'next topic', 'new question']
        return any(indicator in content for indicator in topic_indicators)
    
    def _generate_section_title(self, message_group: List[Dict[str, Any]], index: int) -> str:
        """Generate section title from message group."""
        first_message = message_group[0]
        content = first_message.get('content', '')
        
        # Try to extract topic from first message
        lines = content.split('\n')
        first_line = lines[0].strip()
        
        # If first line looks like a question or topic, use it
        if len(first_line) < 100 and (first_line.endswith('?') or first_line.endswith(':')):
            return first_line
        
        # Extract first few words
        words = first_line.split()[:8]
        if words:
            title = ' '.join(words)
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        
        return f"Topic {index + 1}"
    
    async def _process_message_group(self, messages: List[Dict[str, Any]]) -> List[ContentElement]:
        """Process group of messages into content elements."""
        elements = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp')
            
            if not content.strip():
                continue
            
            # Create message header if including user names
            if self.options.include_user_names:
                user_display = role.title()
                if timestamp and self.options.include_timestamps:
                    if isinstance(timestamp, str):
                        user_display += f" ({timestamp})"
                    else:
                        user_display += f" ({timestamp.strftime('%H:%M')})"
                
                elements.append(create_heading(user_display, level=3))
            
            # Process message content
            content_elements = await self._process_text_content(content)
            elements.extend(content_elements)
        
        return elements
    
    async def _process_text_content(self, text: str) -> List[ContentElement]:
        """Process text content and detect structure."""
        elements = []
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Detect different content types
            if self.options.detect_code_blocks and self._is_code_block(para):
                elements.append(self._create_code_element(para))
            elif self.options.detect_lists and self._is_list(para):
                elements.append(self._create_list_element(para))
            elif self.options.detect_tables and self._is_table(para):
                elements.append(self._create_table_element(para))
            else:
                # Regular paragraph
                if self.options.split_long_paragraphs and len(para) > self.options.max_paragraph_length:
                    # Split long paragraph
                    sub_paragraphs = self._split_long_paragraph(para)
                    for sub_para in sub_paragraphs:
                        elements.append(create_paragraph(sub_para))
                else:
                    elements.append(create_paragraph(para))
        
        return elements
    
    def _is_code_block(self, text: str) -> bool:
        """Check if text is a code block."""
        # Simple heuristics for code detection
        lines = text.split('\n')
        if len(lines) < 2:
            return False
        
        # Check for common code patterns
        code_indicators = [
            r'^\s*(def|class|import|from|if|for|while|try|except)',  # Python
            r'^\s*(function|var|let|const|if|for|while)',  # JavaScript
            r'^\s*(public|private|class|interface|import)',  # Java/C#
            r'^\s*(<\?php|<html|<div|<script)',  # Web languages
        ]
        
        code_line_count = 0
        for line in lines:
            if any(re.match(pattern, line.strip(), re.IGNORECASE) for pattern in code_indicators):
                code_line_count += 1
        
        return code_line_count >= 2 or text.startswith('```')
    
    def _is_list(self, text: str) -> bool:
        """Check if text is a list."""
        lines = text.split('\n')
        if len(lines) < 2:
            return False
        
        list_line_count = 0
        for line in lines:
            line = line.strip()
            if (line.startswith('- ') or line.startswith('* ') or 
                re.match(r'^\d+\.\s', line) or
                line.startswith('• ')):
                list_line_count += 1
        
        return list_line_count >= 2
    
    def _is_table(self, text: str) -> bool:
        """Check if text is a table."""
        lines = text.split('\n')
        if len(lines) < 2:
            return False
        
        # Check for pipe-separated table format
        pipe_line_count = 0
        for line in lines:
            if '|' in line and line.count('|') >= 2:
                pipe_line_count += 1
        
        return pipe_line_count >= 2
    
    def _create_code_element(self, text: str) -> ContentElement:
        """Create code block element."""
        return ContentElement(
            type=ContentType.CODE,
            content=text,
            style=TextStyle(font_family="monospace")
        )
    
    def _create_list_element(self, text: str) -> ContentElement:
        """Create list element from text."""
        lines = text.split('\n')
        items = []
        list_type = ListType.BULLET
        
        # Detect list type
        first_line = lines[0].strip()
        if re.match(r'^\d+\.', first_line):
            list_type = ListType.NUMBERED
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove list markers
            if line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                content = line[2:].strip()
            elif re.match(r'^\d+\.\s', line):
                content = re.sub(r'^\d+\.\s', '', line)
            else:
                content = line
            
            if content:
                items.append(ListItem(content=content))
        
        list_data = ListData(items=items, list_type=list_type)
        return ContentElement(
            type=ContentType.LIST,
            list_data=list_data
        )
    
    def _create_table_element(self, text: str) -> ContentElement:
        """Create table element from text."""
        lines = text.split('\n')
        rows = []
        
        for i, line in enumerate(lines):
            if '|' not in line:
                continue
            
            # Split by pipes and clean
            cells = [cell.strip() for cell in line.split('|')]
            # Remove empty cells at start/end
            if cells and not cells[0]:
                cells = cells[1:]
            if cells and not cells[-1]:
                cells = cells[:-1]
            
            if not cells:
                continue
            
            # Skip separator lines (contain only -, |, :, spaces)
            if all(c in '-|: ' for c in line):
                continue
            
            table_cells = [TableCell(content=cell) for cell in cells]
            is_header = i == 0  # First row is header
            rows.append(TableRow(cells=table_cells, is_header=is_header))
        
        table_data = TableData(rows=rows, has_header=len(rows) > 0)
        return ContentElement(
            type=ContentType.TABLE,
            table_data=table_data
        )
    
    def _split_long_paragraph(self, text: str) -> List[str]:
        """Split long paragraph into smaller ones."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        paragraphs = []
        current_para = ""
        
        for sentence in sentences:
            if len(current_para + sentence) > self.options.max_paragraph_length:
                if current_para:
                    paragraphs.append(current_para.strip())
                    current_para = sentence
                else:
                    # Single sentence is too long, add as is
                    paragraphs.append(sentence)
            else:
                current_para += " " + sentence if current_para else sentence
        
        if current_para:
            paragraphs.append(current_para.strip())
        
        return paragraphs
    
    async def _parse_html_structure(self, html_content: str) -> List[DocumentSection]:
        """Parse HTML content into document sections."""
        # This would use an HTML parser like BeautifulSoup
        # For now, return simple structure
        return [DocumentSection(
            title="Content",
            level=1,
            elements=[create_paragraph(html_content[:500] + "...")]
        )]
    
    async def _parse_plain_text_structure(self, text: str) -> List[DocumentSection]:
        """Parse plain text into document sections."""
        # Split into paragraphs and detect headings
        lines = text.split('\n')
        sections = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect headings (simple heuristic)
            if (len(line) < 100 and 
                (line.isupper() or 
                 line.endswith(':') or
                 re.match(r'^\d+\.?\s+[A-Z]', line))):
                
                # Create new section
                if current_section:
                    sections.append(current_section)
                
                current_section = DocumentSection(
                    title=line.rstrip(':'),
                    level=1
                )
            else:
                # Add as paragraph
                if not current_section:
                    current_section = DocumentSection(title="Content", level=1)
                
                current_section.elements.append(create_paragraph(line))
        
        if current_section:
            sections.append(current_section)
        
        return sections if sections else [DocumentSection(
            title="Content",
            level=1,
            elements=[create_paragraph(text[:1000] + "...")]
        )]
    
    def _extract_markdown_metadata(
        self, 
        markdown_text: str, 
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentMetadata:
        """Extract metadata from markdown front matter."""
        metadata = DocumentMetadata(
            title="Document",
            created_date=datetime.now()
        )
        
        if additional_metadata:
            metadata.title = additional_metadata.get("title", metadata.title)
            metadata.author = additional_metadata.get("author")
            metadata.description = additional_metadata.get("description")
            metadata.keywords = additional_metadata.get("keywords", [])
        
        # Extract from front matter (simplified)
        lines = markdown_text.split('\n')
        if lines and lines[0].strip() == '---':
            # Look for front matter
            end_index = -1
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    end_index = i
                    break
            
            if end_index > 0:
                front_matter = '\n'.join(lines[1:end_index])
                # Simple key: value parsing
                for line in front_matter.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        
                        if key == 'title':
                            metadata.title = value
                        elif key == 'author':
                            metadata.author = value
                        elif key == 'description':
                            metadata.description = value
        
        return metadata
    
    async def _process_content_item(self, item: Dict[str, Any]) -> Optional[ContentElement]:
        """Process individual content item."""
        item_type = item.get('type')
        data = item.get('data', {})
        
        if item_type == 'text':
            return create_paragraph(str(data.get('content', '')))
        elif item_type == 'heading':
            return create_heading(
                str(data.get('content', '')), 
                level=data.get('level', 1)
            )
        elif item_type == 'image':
            return create_image(
                source_path=data.get('path', ''),
                caption=data.get('caption'),
                width=data.get('width'),
                height=data.get('height')
            )
        elif item_type == 'table':
            rows_data = data.get('rows', [])
            return create_table(rows_data, has_header=data.get('has_header', False))
        elif item_type == 'list':
            items = data.get('items', [])
            list_type = ListType.NUMBERED if data.get('numbered', False) else ListType.BULLET
            return create_list(items, list_type)
        
        return None


# Test function for validation
async def test():
    """Test content structuring functionality."""
    structurer = ContentStructurer()
    
    # Test chat conversation structuring
    messages = [
        {"role": "user", "content": "Hello, can you help me with Python?"},
        {"role": "assistant", "content": "Of course! What specific Python topic would you like help with?"},
        {"role": "user", "content": "I need help with lists and dictionaries."},
        {"role": "assistant", "content": "Great! Here are some examples:\n\n1. Lists: [1, 2, 3]\n2. Dictionaries: {'key': 'value'}"}
    ]
    
    try:
        document = await structurer.structure_chat_conversation(messages)
        
        if not document.sections:
            print("❌ Content structuring failed: no sections created")
            return False
        
        if not document.sections[0].elements:
            print("❌ Content structuring failed: no elements in first section")
            return False
        
        print("✅ Content structuring test passed")
        return True
        
    except Exception as e:
        print(f"❌ Content structuring test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test())