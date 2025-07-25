"""
Abstract base classes for document generation framework.

This module defines the core interfaces and base classes that all document generators
must implement, providing a consistent API across different output formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from enum import Enum
from dataclasses import dataclass
import asyncio

from .models import DocumentContent, DocumentTemplate, DocumentMetadata, GenerationOptions


class DocumentFormat(Enum):
    """Supported document formats."""
    DOCX = "docx"
    PDF = "pdf"


class GenerationStatus(Enum):
    """Document generation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationProgress:
    """Progress information during document generation."""
    status: GenerationStatus
    progress_percent: float
    current_step: str
    total_steps: int
    current_step_number: int
    message: Optional[str] = None
    error: Optional[str] = None


class BaseDocumentGenerator(ABC):
    """
    Abstract base class for all document generators.
    
    This class defines the interface that all format-specific generators must implement,
    ensuring consistency across DOCX, PDF, and future format implementations.
    """
    
    def __init__(self, format_type: DocumentFormat):
        """Initialize the generator with a specific format type."""
        self.format_type = format_type
        self._generation_options: Optional[GenerationOptions] = None
    
    @property
    @abstractmethod
    def supported_features(self) -> List[str]:
        """Return list of features supported by this generator."""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension for this format."""
        pass
    
    @property
    @abstractmethod
    def mime_type(self) -> str:
        """Return the MIME type for this format."""
        pass
    
    @abstractmethod
    async def generate_document(
        self, 
        content: DocumentContent, 
        template: Optional[DocumentTemplate] = None,
        options: Optional[GenerationOptions] = None
    ) -> bytes:
        """
        Generate a document from content and template.
        
        Args:
            content: Structured document content
            template: Optional template to apply
            options: Generation options and settings
            
        Returns:
            Generated document as bytes
            
        Raises:
            GenerationError: If document generation fails
        """
        pass
    
    @abstractmethod
    async def generate_preview(
        self, 
        content: DocumentContent, 
        template: Optional[DocumentTemplate] = None,
        options: Optional[GenerationOptions] = None
    ) -> Dict[str, Any]:
        """
        Generate a preview representation of the document.
        
        Args:
            content: Structured document content
            template: Optional template to apply
            options: Generation options and settings
            
        Returns:
            Preview data (format-specific)
        """
        pass
    
    @abstractmethod
    async def validate_content(self, content: DocumentContent) -> List[str]:
        """
        Validate content compatibility with this generator.
        
        Args:
            content: Document content to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    @abstractmethod
    async def estimate_generation_time(self, content: DocumentContent) -> float:
        """
        Estimate document generation time in seconds.
        
        Args:
            content: Document content to analyze
            
        Returns:
            Estimated generation time in seconds
        """
        pass
    
    async def generate_with_progress(
        self,
        content: DocumentContent,
        template: Optional[DocumentTemplate] = None,
        options: Optional[GenerationOptions] = None
    ) -> AsyncIterator[GenerationProgress]:
        """
        Generate document with progress updates.
        
        Args:
            content: Structured document content
            template: Optional template to apply
            options: Generation options and settings
            
        Yields:
            GenerationProgress updates during generation
        """
        yield GenerationProgress(
            status=GenerationStatus.PENDING,
            progress_percent=0.0,
            current_step="Initializing",
            total_steps=5,
            current_step_number=1
        )
        
        try:
            yield GenerationProgress(
                status=GenerationStatus.PROCESSING,
                progress_percent=20.0,
                current_step="Validating content",
                total_steps=5,
                current_step_number=2
            )
            
            validation_errors = await self.validate_content(content)
            if validation_errors:
                yield GenerationProgress(
                    status=GenerationStatus.FAILED,
                    progress_percent=0.0,
                    current_step="Validation failed",
                    total_steps=5,
                    current_step_number=2,
                    error=f"Validation errors: {', '.join(validation_errors)}"
                )
                return
            
            yield GenerationProgress(
                status=GenerationStatus.PROCESSING,
                progress_percent=60.0,
                current_step="Generating document",
                total_steps=5,
                current_step_number=3
            )
            
            # Actual generation happens in subclass
            document_bytes = await self.generate_document(content, template, options)
            
            yield GenerationProgress(
                status=GenerationStatus.PROCESSING,
                progress_percent=90.0,
                current_step="Finalizing",
                total_steps=5,
                current_step_number=4
            )
            
            yield GenerationProgress(
                status=GenerationStatus.COMPLETED,
                progress_percent=100.0,
                current_step="Complete",
                total_steps=5,
                current_step_number=5
            )
            
        except Exception as e:
            yield GenerationProgress(
                status=GenerationStatus.FAILED,
                progress_percent=0.0,
                current_step="Generation failed",
                total_steps=5,
                current_step_number=0,
                error=str(e)
            )
    
    def set_generation_options(self, options: GenerationOptions) -> None:
        """Set generation options for this generator."""
        self._generation_options = options
    
    def get_generation_options(self) -> Optional[GenerationOptions]:
        """Get current generation options."""
        return self._generation_options


class BaseContentProcessor(ABC):
    """
    Abstract base class for content processors.
    
    Content processors handle the transformation of raw content (chat conversations,
    markdown, etc.) into structured DocumentContent that generators can consume.
    """
    
    @abstractmethod
    async def process_content(
        self, 
        raw_content: Union[str, Dict[str, Any]], 
        content_type: str,
        options: Optional[Dict[str, Any]] = None
    ) -> DocumentContent:
        """
        Process raw content into structured DocumentContent.
        
        Args:
            raw_content: Raw content to process
            content_type: Type of content ('chat', 'markdown', 'text', etc.)
            options: Processing options
            
        Returns:
            Structured DocumentContent
        """
        pass
    
    @abstractmethod
    async def extract_metadata(
        self, 
        raw_content: Union[str, Dict[str, Any]], 
        content_type: str
    ) -> DocumentMetadata:
        """
        Extract metadata from raw content.
        
        Args:
            raw_content: Raw content to analyze
            content_type: Type of content
            
        Returns:
            Extracted metadata
        """
        pass


class GenerationError(Exception):
    """Base exception for document generation errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ValidationError(GenerationError):
    """Exception raised when content validation fails."""
    pass


class TemplateError(GenerationError):
    """Exception raised when template processing fails."""
    pass


class FormatError(GenerationError):
    """Exception raised when format-specific operations fail."""
    pass


# Test interface for validation
async def test_interface():
    """Test the abstract interface - used by validation commands."""
    
    class TestGenerator(BaseDocumentGenerator):
        def __init__(self):
            super().__init__(DocumentFormat.DOCX)
        
        @property
        def supported_features(self) -> List[str]:
            return ["text", "images", "tables"]
        
        @property
        def file_extension(self) -> str:
            return "docx"
        
        @property
        def mime_type(self) -> str:
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        async def generate_document(self, content, template=None, options=None) -> bytes:
            return b"test document content"
        
        async def generate_preview(self, content, template=None, options=None) -> Dict[str, Any]:
            return {"preview": "test preview"}
        
        async def validate_content(self, content) -> List[str]:
            return []
        
        async def estimate_generation_time(self, content) -> float:
            return 1.0
    
    # Test instantiation and basic functionality
    generator = TestGenerator()
    assert generator.format_type == DocumentFormat.DOCX
    assert generator.file_extension == "docx"
    assert len(generator.supported_features) > 0
    
    print("✅ Abstract interface test passed")
    return True


if __name__ == "__main__":
    asyncio.run(test_interface())