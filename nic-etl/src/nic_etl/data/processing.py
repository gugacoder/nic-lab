import os
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib
from enum import Enum
import tempfile

# Optional Docling imports with fallbacks
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
    DOCLING_AVAILABLE = True
except ImportError:
    # Create mock classes for development/testing
    class DocumentConverter:
        def __init__(self, **kwargs):
            pass
        def convert(self, path):
            return MockDoclingResult()
    
    class InputFormat:
        PDF = "pdf"
        DOCX = "docx"
        TEXT = "text"
        IMAGE = "image"
    
    class PdfPipelineOptions:
        def __init__(self):
            self.do_ocr = True
            self.do_table_structure = True
            self.table_structure_options = MockTableOptions()
    
    class MockTableOptions:
        def __init__(self):
            self.do_cell_matching = True
    
    class PyPdfiumDocumentBackend:
        pass
    
    class DoclingParseDocumentBackend:
        pass
    
    class MockDoclingResult:
        def __init__(self):
            self.document = MockDocument()
    
    class MockDocument:
        def __init__(self):
            self.texts = []
            self.pages = []
            self.tables = []
            self.pictures = []
            self.meta = MockMeta()
    
    class MockMeta:
        def __init__(self):
            self.title = None
    
    DOCLING_AVAILABLE = False

class DocumentType(Enum):
    """Document type classification"""
    DIGITAL_PDF = "digital_pdf"
    SCANNED_PDF = "scanned_pdf"
    WORD_DOCUMENT = "word_document"
    TEXT_DOCUMENT = "text_document"
    IMAGE_DOCUMENT = "image_document"
    MARKDOWN = "markdown"

class ProcessingQuality(Enum):
    """Processing quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAILED = "failed"

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    ocr_engine: str = "easyocr"  # Docling OCR engine
    confidence_threshold: float = 0.8
    max_file_size_mb: int = 100
    enable_table_extraction: bool = True
    enable_figure_extraction: bool = True
    output_format: str = "json"  # json, markdown, both
    quality_gates_enabled: bool = True
    enable_fallback_processing: bool = True  # Use fallback when Docling unavailable

@dataclass
class StructuredContent:
    """Structured document content with hierarchy"""
    title: Optional[str] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)
    paragraphs: List[Dict[str, Any]] = field(default_factory=list)
    lists: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityAssessment:
    """Document processing quality metrics"""
    overall_quality: ProcessingQuality
    confidence_score: float
    text_extraction_confidence: float
    structure_detection_confidence: float
    ocr_applied: bool
    processing_method: str = "unknown"
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class ProcessedDocument:
    """Complete processed document with all extracted information"""
    file_path: str
    file_hash: str
    document_type: DocumentType
    structured_content: StructuredContent
    quality_assessment: QualityAssessment
    processing_metadata: Dict[str, Any]
    canonical_output: Dict[str, Any]
    processing_timestamp: datetime
    docling_version: str

class DoclingProcessor:
    """Production-ready document processor using Docling framework with fallback"""
    
    SUPPORTED_FORMATS = {
        '.pdf': InputFormat.PDF,
        '.docx': InputFormat.DOCX,
        '.txt': InputFormat.TEXT,
        '.md': InputFormat.TEXT,
        '.jpg': InputFormat.IMAGE,
        '.jpeg': InputFormat.IMAGE,
        '.png': InputFormat.IMAGE
    }
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.docling_available = DOCLING_AVAILABLE
        
        # Initialize Docling converter if available
        if self.docling_available:
            self._initialize_docling_converter()
        else:
            self.logger.warning("Docling not available, using fallback processing")
            self.converter = None
        
        self.logger.info(f"Docling processor initialized (Docling available: {self.docling_available})")
    
    def _initialize_docling_converter(self):
        """Initialize Docling converter with optimized pipeline"""
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = self.config.enable_table_extraction
            pipeline_options.table_structure_options.do_cell_matching = True
            
            self.converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.IMAGE],
                pdf_backend=DoclingParseDocumentBackend,
                pipeline_options=pipeline_options
            )
            
            self.logger.info("Docling converter initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Docling converter: {e}")
            self.docling_available = False
            self.converter = None
    
    def process_document(self, file_path: str, content: bytes, 
                        metadata: Optional[Dict[str, Any]] = None) -> ProcessedDocument:
        """Process document through Docling with comprehensive error handling"""
        
        try:
            # Validate input
            if len(content) > self.config.max_file_size_mb * 1024 * 1024:
                raise ValueError(f"File size exceeds limit: {len(content) / (1024*1024):.1f}MB")
            
            # Generate content hash for idempotency
            file_hash = hashlib.sha256(content).hexdigest()
            
            # Detect document type
            file_extension = Path(file_path).suffix.lower()
            doc_type = self.detect_document_type(content, file_extension)
            
            self.logger.info(f"Processing {doc_type.value} document: {file_path}")
            
            # Choose processing method
            if self.docling_available and self.converter:
                processing_method = "docling"
                docling_result = self._process_with_docling(content, file_path, doc_type)
                structured_content = self.extract_structure(docling_result)
            else:
                processing_method = "fallback"
                structured_content = self._process_with_fallback(content, file_path, doc_type)
            
            # Generate canonical output
            canonical_output = self.export_canonical_format(structured_content)
            
            # Create processed document
            processed_doc = ProcessedDocument(
                file_path=file_path,
                file_hash=file_hash,
                document_type=doc_type,
                structured_content=structured_content,
                quality_assessment=QualityAssessment(
                    overall_quality=ProcessingQuality.HIGH,
                    confidence_score=0.0,  # Will be calculated
                    text_extraction_confidence=0.0,
                    structure_detection_confidence=0.0,
                    ocr_applied=False,  # Will be determined
                    processing_method=processing_method
                ),
                processing_metadata={
                    'docling_backend': 'DoclingParseDocumentBackend' if self.docling_available else 'fallback',
                    'input_metadata': metadata or {},
                    'file_size_bytes': len(content),
                    'processing_method': processing_method
                },
                canonical_output=canonical_output,
                processing_timestamp=datetime.utcnow(),
                docling_version=self._get_docling_version()
            )
            
            # Apply quality gates
            if self.config.quality_gates_enabled:
                processed_doc.quality_assessment = self.apply_quality_gates(processed_doc)
            
            self.logger.info(f"Successfully processed document: {file_path} using {processing_method}")
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"Failed to process document {file_path}: {e}")
            
            # Return failed processing result for pipeline continuity
            return ProcessedDocument(
                file_path=file_path,
                file_hash=hashlib.sha256(content).hexdigest(),
                document_type=DocumentType.DIGITAL_PDF,  # Default
                structured_content=StructuredContent(),
                quality_assessment=QualityAssessment(
                    overall_quality=ProcessingQuality.FAILED,
                    confidence_score=0.0,
                    text_extraction_confidence=0.0,
                    structure_detection_confidence=0.0,
                    ocr_applied=False,
                    processing_method="failed",
                    issues=[str(e)]
                ),
                processing_metadata={'error': str(e)},
                canonical_output={},
                processing_timestamp=datetime.utcnow(),
                docling_version=self._get_docling_version()
            )
    
    def detect_document_type(self, content: bytes, file_extension: str) -> DocumentType:
        """Intelligent document type detection"""
        
        if file_extension in ['.jpg', '.jpeg', '.png']:
            return DocumentType.IMAGE_DOCUMENT
        elif file_extension == '.docx':
            return DocumentType.WORD_DOCUMENT
        elif file_extension == '.txt':
            return DocumentType.TEXT_DOCUMENT
        elif file_extension == '.md':
            return DocumentType.MARKDOWN
        elif file_extension == '.pdf':
            # For PDFs, we'll use simple heuristics to detect digital vs scanned
            # This is a simplified detection - Docling would handle the complexity
            if self._appears_to_be_scanned_pdf(content):
                return DocumentType.SCANNED_PDF
            else:
                return DocumentType.DIGITAL_PDF
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _appears_to_be_scanned_pdf(self, content: bytes) -> bool:
        """Simple heuristic to detect scanned PDFs"""
        # This is a basic implementation - in practice, Docling would provide better detection
        # Look for image-heavy PDFs or lack of text content
        try:
            # Check for common patterns that suggest scanned content
            content_str = content.decode('latin-1', errors='ignore')
            
            # Count image references vs text content
            image_refs = content_str.count('/Image') + content_str.count('/DCTDecode')
            text_refs = content_str.count('/Text') + content_str.count('BT') + content_str.count('ET')
            
            # Simple heuristic: if there are many images and little text, likely scanned
            if image_refs > 0 and text_refs < image_refs:
                return True
            
            return False
            
        except Exception:
            # Default to digital if we can't determine
            return False
    
    def _process_with_docling(self, content: bytes, file_path: str, doc_type: DocumentType) -> Any:
        """Core Docling processing with error handling"""
        
        try:
            # Create temporary file for Docling processing
            with tempfile.NamedTemporaryFile(suffix=Path(file_path).suffix, delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                # Process through Docling converter
                result = self.converter.convert(temp_path)
                return result
            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)
            
        except Exception as e:
            self.logger.error(f"Docling processing failed for {file_path}: {e}")
            raise
    
    def _process_with_fallback(self, content: bytes, file_path: str, doc_type: DocumentType) -> StructuredContent:
        """Fallback processing when Docling is not available"""
        
        structured_content = StructuredContent()
        
        try:
            if doc_type in [DocumentType.TEXT_DOCUMENT, DocumentType.MARKDOWN]:
                # Process text/markdown documents
                text_content = content.decode('utf-8', errors='ignore')
                structured_content = self._parse_text_content(text_content, doc_type)
            
            elif doc_type == DocumentType.DIGITAL_PDF:
                # Basic PDF text extraction (would need PyMuPDF or similar)
                structured_content = self._fallback_pdf_processing(content)
            
            elif doc_type == DocumentType.WORD_DOCUMENT:
                # Basic DOCX processing (would need python-docx)
                structured_content = self._fallback_docx_processing(content)
            
            elif doc_type in [DocumentType.IMAGE_DOCUMENT, DocumentType.SCANNED_PDF]:
                # OCR would be needed here
                structured_content = self._fallback_ocr_processing(content, doc_type)
            
            # Add fallback metadata
            structured_content.metadata.update({
                'processing_method': 'fallback',
                'docling_available': False,
                'extraction_quality': 'basic'
            })
            
        except Exception as e:
            self.logger.warning(f"Fallback processing failed: {e}")
            structured_content.metadata = {'fallback_error': str(e)}
        
        return structured_content
    
    def _parse_text_content(self, text_content: str, doc_type: DocumentType) -> StructuredContent:
        """Parse plain text or markdown content"""
        
        structured_content = StructuredContent()
        lines = text_content.split('\n')
        
        current_section = None
        current_paragraph = []
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                # Empty line - end current paragraph
                if current_paragraph:
                    structured_content.paragraphs.append({
                        'text': ' '.join(current_paragraph),
                        'line_number': line_num,
                        'confidence': 1.0
                    })
                    current_paragraph = []
                continue
            
            # Detect markdown structures
            if doc_type == DocumentType.MARKDOWN:
                if line.startswith('#'):
                    # Heading
                    level = len(line) - len(line.lstrip('#'))
                    title = line.lstrip('# ').strip()
                    
                    if current_section:
                        structured_content.sections.append(current_section)
                    
                    current_section = {
                        'title': title,
                        'level': level,
                        'content': [],
                        'line_number': line_num
                    }
                    
                    # Also add as title if it's the first heading
                    if not structured_content.title and level == 1:
                        structured_content.title = title
                    
                    continue
                
                elif line.startswith('* ') or line.startswith('- '):
                    # List item
                    structured_content.lists.append({
                        'type': 'unordered',
                        'item': line[2:].strip(),
                        'line_number': line_num
                    })
                    continue
                
                elif line.startswith('1. ') or any(line.startswith(f'{i}. ') for i in range(2, 10)):
                    # Numbered list item
                    structured_content.lists.append({
                        'type': 'ordered',
                        'item': line.split('. ', 1)[1] if '. ' in line else line,
                        'line_number': line_num
                    })
                    continue
            
            # Regular text line
            current_paragraph.append(line)
            
            # Add to current section if we have one
            if current_section:
                current_section['content'].append({
                    'text': line,
                    'type': 'paragraph'
                })
        
        # Handle remaining content
        if current_paragraph:
            structured_content.paragraphs.append({
                'text': ' '.join(current_paragraph),
                'line_number': len(lines),
                'confidence': 1.0
            })
        
        if current_section:
            structured_content.sections.append(current_section)
        
        # Extract title if not set
        if not structured_content.title and structured_content.paragraphs:
            first_para = structured_content.paragraphs[0]['text']
            if len(first_para) < 100:  # Likely a title
                structured_content.title = first_para
        
        structured_content.metadata = {
            'total_lines': len(lines),
            'paragraph_count': len(structured_content.paragraphs),
            'section_count': len(structured_content.sections),
            'list_count': len(structured_content.lists)
        }
        
        return structured_content
    
    def _fallback_pdf_processing(self, content: bytes) -> StructuredContent:
        """Basic PDF processing without Docling"""
        
        structured_content = StructuredContent()
        
        # This would require PyMuPDF or similar library
        # For now, create a placeholder structure
        structured_content.paragraphs.append({
            'text': '[PDF content - advanced processing requires Docling or PyMuPDF]',
            'page_number': 1,
            'confidence': 0.5
        })
        
        structured_content.metadata = {
            'processing_note': 'PDF processing requires Docling framework',
            'content_type': 'pdf',
            'requires_advanced_processing': True
        }
        
        return structured_content
    
    def _fallback_docx_processing(self, content: bytes) -> StructuredContent:
        """Basic DOCX processing without Docling"""
        
        structured_content = StructuredContent()
        
        # This would require python-docx library
        # For now, create a placeholder structure
        structured_content.paragraphs.append({
            'text': '[DOCX content - advanced processing requires Docling or python-docx]',
            'confidence': 0.5
        })
        
        structured_content.metadata = {
            'processing_note': 'DOCX processing requires Docling framework',
            'content_type': 'docx',
            'requires_advanced_processing': True
        }
        
        return structured_content
    
    def _fallback_ocr_processing(self, content: bytes, doc_type: DocumentType) -> StructuredContent:
        """Basic OCR processing without Docling"""
        
        structured_content = StructuredContent()
        
        # This would require OCR libraries (tesseract, easyocr, etc.)
        # For now, create a placeholder structure
        structured_content.paragraphs.append({
            'text': f'[{doc_type.value.upper()} content - OCR requires Docling framework]',
            'confidence': 0.3
        })
        
        structured_content.metadata = {
            'processing_note': 'OCR processing requires Docling framework',
            'content_type': doc_type.value,
            'ocr_required': True,
            'requires_advanced_processing': True
        }
        
        return structured_content
    
    def extract_structure(self, docling_result: Any) -> StructuredContent:
        """Extract structured content from Docling result"""
        
        structured_content = StructuredContent()
        
        try:
            # Access the document from Docling result
            document = docling_result.document
            
            # Extract title (first heading or from metadata)
            structured_content.title = self._extract_title(document)
            
            # Extract sections and hierarchy
            structured_content.sections = self._extract_sections(document)
            
            # Extract paragraphs with metadata
            structured_content.paragraphs = self._extract_paragraphs(document)
            
            # Extract lists
            structured_content.lists = self._extract_lists(document)
            
            # Extract tables if enabled
            if self.config.enable_table_extraction:
                structured_content.tables = self._extract_tables(document)
            
            # Extract figures if enabled
            if self.config.enable_figure_extraction:
                structured_content.figures = self._extract_figures(document)
            
            # Add processing metadata
            structured_content.metadata = {
                'page_count': len(document.pages) if hasattr(document, 'pages') else 0,
                'element_count': len(document.texts) if hasattr(document, 'texts') else 0,
                'has_tables': len(structured_content.tables) > 0,
                'has_figures': len(structured_content.figures) > 0,
                'processing_method': 'docling'
            }
            
        except Exception as e:
            self.logger.error(f"Structure extraction failed: {e}")
            # Return minimal structure to prevent pipeline failure
            structured_content.metadata = {'extraction_error': str(e)}
        
        return structured_content
    
    def _extract_title(self, document: Any) -> Optional[str]:
        """Extract document title from Docling document"""
        try:
            # Look for title in document metadata or first heading
            if hasattr(document, 'meta') and hasattr(document.meta, 'title') and document.meta.title:
                return document.meta.title
            
            # Find first heading element
            if hasattr(document, 'texts'):
                for element in document.texts:
                    if hasattr(element, 'label') and element.label and 'title' in element.label.lower():
                        return getattr(element, 'text', '')
            
            return None
            
        except Exception:
            return None
    
    def _extract_sections(self, document: Any) -> List[Dict[str, Any]]:
        """Extract document sections with hierarchy"""
        sections = []
        
        try:
            if not hasattr(document, 'texts'):
                return sections
            
            current_section = None
            
            for element in document.texts:
                element_label = getattr(element, 'label', '').lower()
                element_text = getattr(element, 'text', '')
                
                if 'section' in element_label or 'heading' in element_label:
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        'title': element_text,
                        'level': self._determine_heading_level(element),
                        'content': [],
                        'page_number': getattr(element, 'page', 0),
                        'position': getattr(element, 'bbox', None)
                    }
                elif current_section and element_text:
                    current_section['content'].append({
                        'text': element_text,
                        'type': getattr(element, 'label', 'paragraph')
                    })
            
            # Add final section
            if current_section:
                sections.append(current_section)
                
        except Exception as e:
            self.logger.warning(f"Section extraction warning: {e}")
        
        return sections
    
    def _extract_paragraphs(self, document: Any) -> List[Dict[str, Any]]:
        """Extract paragraphs with positioning metadata"""
        paragraphs = []
        
        try:
            if not hasattr(document, 'texts'):
                return paragraphs
            
            for element in document.texts:
                element_label = getattr(element, 'label', '').lower()
                element_text = getattr(element, 'text', '')
                
                if 'paragraph' in element_label or not element_label:  # Default to paragraph
                    paragraphs.append({
                        'text': element_text,
                        'page_number': getattr(element, 'page', 0),
                        'position': getattr(element, 'bbox', None),
                        'confidence': getattr(element, 'confidence', 1.0)
                    })
                    
        except Exception as e:
            self.logger.warning(f"Paragraph extraction warning: {e}")
        
        return paragraphs
    
    def _extract_lists(self, document: Any) -> List[Dict[str, Any]]:
        """Extract lists from Docling document"""
        lists = []
        
        try:
            if not hasattr(document, 'texts'):
                return lists
            
            for element in document.texts:
                element_label = getattr(element, 'label', '').lower()
                element_text = getattr(element, 'text', '')
                
                if 'list' in element_label:
                    lists.append({
                        'type': 'unordered' if 'bullet' in element_label else 'ordered',
                        'item': element_text,
                        'page_number': getattr(element, 'page', 0),
                        'position': getattr(element, 'bbox', None)
                    })
                    
        except Exception as e:
            self.logger.warning(f"List extraction warning: {e}")
        
        return lists
    
    def _extract_tables(self, document: Any) -> List[Dict[str, Any]]:
        """Extract tables with structure preservation"""
        tables = []
        
        try:
            if hasattr(document, 'tables'):
                for table in document.tables:
                    tables.append({
                        'data': self._serialize_table_data(table),
                        'page_number': getattr(table, 'page', 0),
                        'position': getattr(table, 'bbox', None),
                        'row_count': len(getattr(table, 'data', [])),
                        'column_count': len(getattr(table, 'data', [[]])[0]) if getattr(table, 'data', []) else 0
                    })
                    
        except Exception as e:
            self.logger.warning(f"Table extraction warning: {e}")
        
        return tables
    
    def _extract_figures(self, document: Any) -> List[Dict[str, Any]]:
        """Extract figures and images"""
        figures = []
        
        try:
            if hasattr(document, 'pictures'):
                for figure in document.pictures:
                    figures.append({
                        'caption': getattr(figure, 'caption', ''),
                        'page_number': getattr(figure, 'page', 0),
                        'position': getattr(figure, 'bbox', None),
                        'image_data': getattr(figure, 'image', None)  # Base64 or reference
                    })
                    
        except Exception as e:
            self.logger.warning(f"Figure extraction warning: {e}")
        
        return figures
    
    def apply_quality_gates(self, processed_doc: ProcessedDocument) -> QualityAssessment:
        """Apply quality assessment and confidence scoring"""
        
        assessment = QualityAssessment(
            overall_quality=ProcessingQuality.HIGH,
            confidence_score=1.0,
            text_extraction_confidence=1.0,
            structure_detection_confidence=1.0,
            ocr_applied=self._was_ocr_applied(processed_doc),
            processing_method=processed_doc.processing_metadata.get('processing_method', 'unknown')
        )
        
        # Calculate confidence scores based on extracted content
        text_length = sum(len(p['text']) for p in processed_doc.structured_content.paragraphs)
        
        # Adjust confidence based on content length
        if text_length < 50:
            assessment.warnings.append("Very low text content extracted")
            assessment.text_extraction_confidence = 0.3
        elif text_length < 200:
            assessment.warnings.append("Low text content extracted")
            assessment.text_extraction_confidence = 0.6
        else:
            assessment.text_extraction_confidence = 0.9
        
        # Check structure detection quality
        structure_elements = (
            len(processed_doc.structured_content.sections) +
            len(processed_doc.structured_content.paragraphs) +
            len(processed_doc.structured_content.lists)
        )
        
        if structure_elements < 2:
            assessment.warnings.append("Very limited document structure detected")
            assessment.structure_detection_confidence = 0.4
        elif structure_elements < 5:
            assessment.warnings.append("Limited document structure detected")
            assessment.structure_detection_confidence = 0.6
        else:
            assessment.structure_detection_confidence = 0.9
        
        # Adjust for fallback processing
        if assessment.processing_method == 'fallback':
            assessment.text_extraction_confidence *= 0.7
            assessment.structure_detection_confidence *= 0.5
            assessment.warnings.append("Using fallback processing - reduced accuracy expected")
        
        # Calculate overall confidence
        assessment.confidence_score = (
            assessment.text_extraction_confidence * 0.6 +
            assessment.structure_detection_confidence * 0.4
        )
        
        # Determine overall quality
        if assessment.confidence_score >= 0.8:
            assessment.overall_quality = ProcessingQuality.HIGH
        elif assessment.confidence_score >= 0.6:
            assessment.overall_quality = ProcessingQuality.MEDIUM
        elif assessment.confidence_score >= 0.3:
            assessment.overall_quality = ProcessingQuality.LOW
        else:
            assessment.overall_quality = ProcessingQuality.FAILED
        
        return assessment
    
    def export_canonical_format(self, structured_content: StructuredContent) -> Dict[str, Any]:
        """Export structured content to canonical JSON format"""
        
        canonical = {
            'document_structure': {
                'title': structured_content.title,
                'sections': structured_content.sections,
                'paragraphs': structured_content.paragraphs,
                'lists': structured_content.lists,
                'tables': structured_content.tables,
                'figures': structured_content.figures
            },
            'metadata': structured_content.metadata,
            'format_version': '1.0',
            'export_timestamp': datetime.utcnow().isoformat()
        }
        
        # Optional Markdown export
        if self.config.output_format in ['markdown', 'both']:
            canonical['markdown'] = self._convert_to_markdown(structured_content)
        
        return canonical
    
    def _convert_to_markdown(self, structured_content: StructuredContent) -> str:
        """Convert structured content to Markdown format"""
        md_lines = []
        
        # Add title
        if structured_content.title:
            md_lines.append(f"# {structured_content.title}")
            md_lines.append("")
        
        # Add sections
        for section in structured_content.sections:
            level = section.get('level', 1)
            md_lines.append(f"{'#' * level} {section['title']}")
            md_lines.append("")
            
            for content_item in section.get('content', []):
                md_lines.append(content_item['text'])
                md_lines.append("")
        
        # Add standalone paragraphs
        for paragraph in structured_content.paragraphs:
            md_lines.append(paragraph['text'])
            md_lines.append("")
        
        # Add lists
        for list_item in structured_content.lists:
            prefix = "- " if list_item.get('type') == 'unordered' else "1. "
            md_lines.append(f"{prefix}{list_item['item']}")
        
        if structured_content.lists:
            md_lines.append("")
        
        return "\n".join(md_lines)
    
    def _determine_heading_level(self, element: Any) -> int:
        """Determine heading level from Docling element"""
        # Implementation depends on Docling's element structure
        return getattr(element, 'level', 1)
    
    def _serialize_table_data(self, table: Any) -> List[List[str]]:
        """Serialize table data for JSON storage"""
        # Implementation depends on Docling's table structure
        if hasattr(table, 'data'):
            return getattr(table, 'data', [])
        return []
    
    def _was_ocr_applied(self, processed_doc: ProcessedDocument) -> bool:
        """Determine if OCR was applied during processing"""
        return processed_doc.document_type in [DocumentType.SCANNED_PDF, DocumentType.IMAGE_DOCUMENT]
    
    def _get_docling_version(self) -> str:
        """Get Docling framework version"""
        try:
            import docling
            return getattr(docling, '__version__', 'unknown')
        except ImportError:
            return "not_available"
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get current processing capabilities"""
        return {
            'docling_available': self.docling_available,
            'supported_formats': list(self.SUPPORTED_FORMATS.keys()),
            'ocr_enabled': self.docling_available and self.config.ocr_engine,
            'table_extraction': self.config.enable_table_extraction,
            'figure_extraction': self.config.enable_figure_extraction,
            'quality_gates': self.config.quality_gates_enabled,
            'fallback_processing': self.config.enable_fallback_processing
        }

def create_docling_processor(config_dict: Dict[str, Any]) -> DoclingProcessor:
    """Factory function for Docling processor creation"""
    config = ProcessingConfig(
        ocr_engine=config_dict.get('ocr_engine', 'easyocr'),
        confidence_threshold=config_dict.get('confidence_threshold', 0.8),
        max_file_size_mb=config_dict.get('max_file_size_mb', 100),
        enable_table_extraction=config_dict.get('enable_table_extraction', True),
        enable_figure_extraction=config_dict.get('enable_figure_extraction', True),
        output_format=config_dict.get('output_format', 'json'),
        quality_gates_enabled=config_dict.get('quality_gates_enabled', True),
        enable_fallback_processing=config_dict.get('enable_fallback_processing', True)
    )
    return DoclingProcessor(config)

# Context manager for document processing
class DoclingProcessingContext:
    """Context manager for document processing with automatic cleanup"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config_dict = config_dict
        self.processor = None
    
    def __enter__(self) -> DoclingProcessor:
        self.processor = create_docling_processor(self.config_dict)
        return self.processor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass

# Document processing error classes
class DoclingProcessingError(Exception):
    """Base exception for Docling processing errors"""
    pass

class DocumentFormatError(DoclingProcessingError):
    """Unsupported or corrupted document format"""
    pass

class OCRProcessingError(DoclingProcessingError):
    """OCR processing failures"""
    pass

class StructureExtractionError(DoclingProcessingError):
    """Document structure extraction failures"""
    pass

# Quality validation functions
def validate_processing_quality(processed_doc: ProcessedDocument) -> bool:
    """Validate processing meets quality thresholds"""
    if processed_doc.quality_assessment.overall_quality == ProcessingQuality.FAILED:
        return False
    
    if processed_doc.quality_assessment.confidence_score < 0.3:
        return False
    
    # Ensure minimum content was extracted
    total_text = sum(len(p['text']) for p in processed_doc.structured_content.paragraphs)
    if total_text < 10:  # Minimum 10 characters
        return False
    
    return True