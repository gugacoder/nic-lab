"""
Document generation framework.

This package provides a comprehensive framework for generating documents
in various formats (DOCX, PDF) from structured content.
"""

from .models import (
    DocumentContent,
    DocumentSection, 
    ContentElement,
    ContentType,
    DocumentMetadata,
    DocumentTemplate,
    TextStyle,
    ImageData,
    TableData,
    ListData,
    GenerationOptions
)

from .base import (
    BaseDocumentGenerator,
    BaseContentProcessor,
    DocumentFormat,
    GenerationStatus,
    GenerationProgress,
    GenerationError,
    ValidationError,
    TemplateError,
    FormatError
)

__version__ = "1.0.0"
__all__ = [
    # Models
    "DocumentContent",
    "DocumentSection", 
    "ContentElement",
    "ContentType",
    "DocumentMetadata", 
    "DocumentTemplate",
    "TextStyle",
    "ImageData",
    "TableData", 
    "ListData",
    "GenerationOptions",
    
    # Base classes
    "BaseDocumentGenerator",
    "BaseContentProcessor",
    "DocumentFormat",
    "GenerationStatus",
    "GenerationProgress",
    
    # Exceptions
    "GenerationError",
    "ValidationError", 
    "TemplateError",
    "FormatError"
]