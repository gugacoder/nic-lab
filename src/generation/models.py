"""
Data models for document generation framework.

This module defines the data structures used throughout the document generation
pipeline, providing type safety and clear interfaces for content, templates,
and configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from datetime import datetime
import uuid


class ContentType(Enum):
    """Types of content elements."""
    TEXT = "text"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    QUOTE = "quote"
    DIVIDER = "divider"


class ListType(Enum):
    """Types of lists."""
    BULLET = "bullet"
    NUMBERED = "numbered"
    CHECKLIST = "checklist"


class ImageAlignment(Enum):
    """Image alignment options."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    INLINE = "inline"


class StyleWeight(Enum):
    """Text weight options."""
    NORMAL = "normal"
    BOLD = "bold"
    LIGHT = "light"


class StyleEmphasis(Enum):
    """Text emphasis options."""
    NONE = "none"
    ITALIC = "italic"
    UNDERLINE = "underline"
    STRIKETHROUGH = "strikethrough"


@dataclass
class TextStyle:
    """Text styling configuration."""
    font_family: Optional[str] = None
    font_size: Optional[int] = None
    weight: StyleWeight = StyleWeight.NORMAL
    emphasis: StyleEmphasis = StyleEmphasis.NONE
    color: Optional[str] = None
    background_color: Optional[str] = None
    line_height: Optional[float] = None
    letter_spacing: Optional[float] = None


@dataclass
class ImageData:
    """Image information and metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_type: str = "file"  # 'file', 'url', 'generated', 'base64'
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    base64_data: Optional[str] = None
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    alignment: ImageAlignment = ImageAlignment.CENTER
    maintain_aspect_ratio: bool = True
    compression_quality: Optional[int] = None  # 1-100 for JPEG
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableCell:
    """Individual table cell."""
    content: str
    style: Optional[TextStyle] = None
    colspan: int = 1
    rowspan: int = 1
    alignment: str = "left"  # 'left', 'center', 'right'
    vertical_alignment: str = "top"  # 'top', 'middle', 'bottom'
    background_color: Optional[str] = None
    border_style: Optional[str] = None


@dataclass
class TableRow:
    """Table row containing cells."""
    cells: List[TableCell]
    style: Optional[TextStyle] = None
    background_color: Optional[str] = None
    is_header: bool = False


@dataclass
class TableData:
    """Table structure and data."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rows: List[TableRow] = field(default_factory=list)
    caption: Optional[str] = None
    has_header: bool = False
    column_widths: Optional[List[Union[int, str]]] = None  # Can be pixels or percentages
    border_style: str = "solid"
    border_width: int = 1
    border_color: str = "#000000"
    cell_padding: int = 5
    cell_spacing: int = 0
    style: Optional[TextStyle] = None


@dataclass
class ListItem:
    """Individual list item."""
    content: str
    level: int = 0  # For nested lists
    style: Optional[TextStyle] = None
    checked: Optional[bool] = None  # For checklists


@dataclass
class ListData:
    """List structure and items."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    items: List[ListItem] = field(default_factory=list)
    list_type: ListType = ListType.BULLET
    start_number: int = 1  # For numbered lists
    style: Optional[TextStyle] = None


@dataclass
class ContentElement:
    """Base content element that can represent any type of content."""
    type: ContentType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    style: Optional[TextStyle] = None
    level: int = 0  # For headings, nested elements
    
    # Type-specific data
    image_data: Optional[ImageData] = None
    table_data: Optional[TableData] = None
    list_data: Optional[ListData] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSection:
    """Document section containing elements."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    level: int = 1  # Section nesting level
    elements: List[ContentElement] = field(default_factory=list)
    subsections: List['DocumentSection'] = field(default_factory=list)
    style: Optional[TextStyle] = None
    page_break_before: bool = False
    page_break_after: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentMetadata:
    """Document metadata and properties."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    language: str = "en"
    version: str = "1.0"
    category: Optional[str] = None
    status: str = "draft"
    custom_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentContent:
    """Complete document content structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    sections: List[DocumentSection] = field(default_factory=list)
    
    # Global document settings
    page_size: str = "A4"  # 'A4', 'Letter', 'Legal', etc.
    page_orientation: str = "portrait"  # 'portrait', 'landscape'
    margins: Dict[str, float] = field(default_factory=lambda: {
        "top": 2.54, "bottom": 2.54, "left": 2.54, "right": 2.54
    })  # In centimeters
    
    # Document-wide styles
    default_font_family: str = "Calibri"
    default_font_size: int = 11
    line_spacing: float = 1.15
    paragraph_spacing_before: float = 0
    paragraph_spacing_after: float = 6
    
    # Additional content
    header_content: Optional[str] = None
    footer_content: Optional[str] = None
    watermark: Optional[str] = None
    
    # Generation metadata
    source_type: str = "unknown"  # 'chat', 'markdown', 'manual', etc.
    source_id: Optional[str] = None
    created_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TemplateStyle:
    """Style definition in a template."""
    name: str
    style: TextStyle
    applies_to: List[ContentType] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateLayout:
    """Layout configuration for a template."""
    page_size: str = "A4"
    page_orientation: str = "portrait"
    margins: Dict[str, float] = field(default_factory=lambda: {
        "top": 2.54, "bottom": 2.54, "left": 2.54, "right": 2.54
    })
    header_height: float = 1.27
    footer_height: float = 1.27
    column_count: int = 1
    column_spacing: float = 1.27


@dataclass
class DocumentTemplate:
    """Document template configuration."""
    name: str
    format_type: str  # 'docx', 'pdf', 'all'
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    
    # Template content
    styles: List[TemplateStyle] = field(default_factory=list)
    layout: TemplateLayout = field(default_factory=TemplateLayout)
    
    # Template files/resources
    template_file_path: Optional[str] = None
    asset_directory: Optional[str] = None
    
    # Branding
    logo_path: Optional[str] = None
    company_name: Optional[str] = None
    brand_colors: Dict[str, str] = field(default_factory=dict)
    
    # Header/Footer templates
    header_template: Optional[str] = None
    footer_template: Optional[str] = None
    
    # Metadata
    version: str = "1.0"
    created_date: datetime = field(default_factory=datetime.now)
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    is_default: bool = False


@dataclass
class GenerationOptions:
    """Options for document generation."""
    # Quality settings
    image_quality: int = 85  # 1-100
    image_max_width: int = 800
    image_max_height: int = 600
    optimize_images: bool = True
    
    # Performance settings
    enable_async: bool = True
    max_concurrent_images: int = 5
    chunk_size: int = 1000
    
    # Output settings  
    compress_output: bool = False
    include_metadata: bool = True
    embed_fonts: bool = False
    
    # Debug settings
    debug_mode: bool = False
    preserve_temp_files: bool = False
    log_generation_steps: bool = False
    
    # Format-specific settings
    format_options: Dict[str, Any] = field(default_factory=dict)


# Helper functions for creating common content elements

def create_heading(text: str, level: int = 1, style: Optional[TextStyle] = None) -> ContentElement:
    """Create a heading element."""
    return ContentElement(
        type=ContentType.HEADING,
        content=text,
        level=level,
        style=style
    )


def create_paragraph(text: str, style: Optional[TextStyle] = None) -> ContentElement:
    """Create a paragraph element."""
    return ContentElement(
        type=ContentType.PARAGRAPH,
        content=text,
        style=style
    )


def create_image(
    source_path: str, 
    caption: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    alignment: ImageAlignment = ImageAlignment.CENTER
) -> ContentElement:
    """Create an image element."""
    image_data = ImageData(
        source_type="file",
        source_path=source_path,
        caption=caption,
        width=width,
        height=height,
        alignment=alignment
    )
    return ContentElement(
        type=ContentType.IMAGE,
        image_data=image_data
    )


def create_table(rows_data: List[List[str]], has_header: bool = False) -> ContentElement:
    """Create a table element from simple string data."""
    rows = []
    for i, row_data in enumerate(rows_data):
        is_header = has_header and i == 0
        cells = [TableCell(content=cell_content) for cell_content in row_data]
        rows.append(TableRow(cells=cells, is_header=is_header))
    
    table_data = TableData(rows=rows, has_header=has_header)
    return ContentElement(
        type=ContentType.TABLE,
        table_data=table_data
    )


def create_list(items: List[str], list_type: ListType = ListType.BULLET) -> ContentElement:
    """Create a list element."""
    list_items = [ListItem(content=item) for item in items]
    list_data = ListData(items=list_items, list_type=list_type)
    return ContentElement(
        type=ContentType.LIST,
        list_data=list_data
    )


# Validation functions

def validate_document_content(content: DocumentContent) -> List[str]:
    """Validate document content structure."""
    errors = []
    
    if not content.sections:
        errors.append("Document must contain at least one section")
    
    for i, section in enumerate(content.sections):
        if not section.elements and not section.subsections:
            errors.append(f"Section {i} is empty (no elements or subsections)")
        
        for j, element in enumerate(section.elements):
            element_errors = validate_content_element(element)
            errors.extend([f"Section {i}, Element {j}: {error}" for error in element_errors])
    
    return errors


def validate_content_element(element: ContentElement) -> List[str]:
    """Validate individual content element."""
    errors = []
    
    if element.type == ContentType.IMAGE:
        if not element.image_data:
            errors.append("Image element missing image_data")
        elif not any([
            element.image_data.source_path,
            element.image_data.source_url,
            element.image_data.base64_data
        ]):
            errors.append("Image element missing valid source")
    
    elif element.type == ContentType.TABLE:
        if not element.table_data or not element.table_data.rows:
            errors.append("Table element missing table_data or rows")
    
    elif element.type == ContentType.LIST:
        if not element.list_data or not element.list_data.items:
            errors.append("List element missing list_data or items")
    
    elif element.type in [ContentType.TEXT, ContentType.PARAGRAPH, ContentType.HEADING]:
        if not element.content:
            errors.append(f"{element.type.value} element missing content")
    
    return errors


# Test function for validation
def validate():
    """Test data models - used by validation commands."""
    
    # Test creating document content
    metadata = DocumentMetadata(
        title="Test Document",
        author="Test Author",
        keywords=["test", "validation"]
    )
    
    # Test creating elements
    heading = create_heading("Test Heading", level=1)
    paragraph = create_paragraph("This is a test paragraph.")
    test_table = create_table([["Header 1", "Header 2"], ["Cell 1", "Cell 2"]], has_header=True)
    test_list = create_list(["Item 1", "Item 2", "Item 3"])
    
    # Test creating section
    section = DocumentSection(
        title="Test Section",
        elements=[heading, paragraph, test_table, test_list]
    )
    
    # Test creating document
    document = DocumentContent(
        metadata=metadata,
        sections=[section]
    )
    
    # Test validation
    errors = validate_document_content(document)
    if errors:
        print(f"❌ Validation failed: {errors}")
        return False
    
    print("✅ Document models validation passed")
    return True


if __name__ == "__main__":
    validate()