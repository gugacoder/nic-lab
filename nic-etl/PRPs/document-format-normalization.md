# Document Format Normalization - PRP

## ROLE
**Python Developer with Document Processing expertise**

Responsible for implementing multi-format document processing pipeline that normalizes PDF, DOCX, and image formats into unified text representations. Must have experience with document parsing libraries, format detection, and content extraction techniques.

## OBJECTIVE
**Create unified document processing pipeline for multiple input formats**

Develop a robust system that:
- Automatically detects document formats (PDF, DOCX, images)
- Extracts text content from each format using appropriate libraries
- Normalizes output to consistent text representation
- Preserves document structure and metadata
- Handles corrupted or malformed files gracefully
- Provides format-specific processing statistics

Success criteria: Successfully process 95%+ of input documents across all supported formats with consistent text extraction quality.

## MOTIVATION
**Enable consistent downstream processing regardless of input format**

Different document formats require specialized processing approaches. By normalizing all formats to a unified representation, the system ensures consistent quality for embedding generation and semantic search, while maintaining the flexibility to handle diverse organizational content types.

## CONTEXT
**NIC ETL Pipeline - Format Normalization Phase**

Technology Stack:
- Python 3.8+ with jupyter notebook environment
- PyPDF2/pdfplumber for PDF processing
- python-docx for DOCX processing
- Pillow/PIL for image handling
- Input from GitLab document ingestion pipeline
- Output to OCR processing and document structuring phases

Supported Formats:
- PDF files (text-based and image-based)
- DOCX/DOC Microsoft Word documents
- Image files (PNG, JPG, JPEG) for OCR processing

## IMPLEMENTATION BLUEPRINT

### Architecture Overview
```
Input Documents → Format Detection → Format-Specific Processors → Normalized Text Output → Metadata Enrichment
```

### Code Structure
```python
# File organization
src/
├── normalization/
│   ├── __init__.py
│   ├── format_detector.py        # File format detection
│   ├── pdf_processor.py          # PDF text extraction
│   ├── docx_processor.py         # DOCX text extraction
│   ├── image_processor.py        # Image preprocessing for OCR
│   ├── normalizer.py             # Main normalization orchestrator
│   └── output_formatter.py       # Unified output formatting
├── utils/
│   ├── text_cleaner.py           # Text cleaning utilities
│   └── metadata_extractor.py     # Format-specific metadata
└── notebooks/
    └── 02_format_normalization.ipynb
```

### Format Detection Implementation
```python
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional
import magic
import logging

class FormatDetector:
    """Detect and classify document formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'image/png': 'image',
            'image/jpeg': 'image',
            'image/jpg': 'image'
        }
    
    def detect_format(self, file_path: str) -> Dict[str, Any]:
        """Detect file format using multiple methods"""
        try:
            # Method 1: File extension
            extension_format = self._detect_by_extension(file_path)
            
            # Method 2: MIME type detection
            mime_format = self._detect_by_mime(file_path)
            
            # Method 3: Magic number detection
            magic_format = self._detect_by_magic(file_path)
            
            # Consensus detection
            detected_format = self._consensus_format(extension_format, mime_format, magic_format)
            
            return {
                'format': detected_format,
                'confidence': self._calculate_confidence(extension_format, mime_format, magic_format),
                'methods': {
                    'extension': extension_format,
                    'mime': mime_format,
                    'magic': magic_format
                }
            }
        except Exception as e:
            self.logger.error(f"Format detection failed for {file_path}: {e}")
            return {'format': 'unknown', 'confidence': 0.0}
    
    def _detect_by_extension(self, file_path: str) -> str:
        """Detect format by file extension"""
        suffix = Path(file_path).suffix.lower()
        extension_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'doc',
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image'
        }
        return extension_map.get(suffix, 'unknown')
    
    def _detect_by_mime(self, file_path: str) -> str:
        """Detect format by MIME type"""
        mime_type, _ = mimetypes.guess_type(file_path)
        return self.supported_formats.get(mime_type, 'unknown')
    
    def _detect_by_magic(self, file_path: str) -> str:
        """Detect format by magic numbers"""
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(file_path)
            return self.supported_formats.get(mime_type, 'unknown')
        except:
            return 'unknown'
```

### PDF Processing Implementation
```python
import PyPDF2
import pdfplumber
from typing import Dict, List

class PDFProcessor:
    """Extract text and metadata from PDF files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF file and extract content"""
        try:
            # Try pdfplumber first (better for complex layouts)
            result = self._process_with_pdfplumber(file_path)
            
            # Fallback to PyPDF2 if pdfplumber fails
            if not result['success']:
                result = self._process_with_pypdf2(file_path)
            
            return result
        except Exception as e:
            self.logger.error(f"PDF processing failed for {file_path}: {e}")
            return {
                'success': False,
                'text': '',
                'pages': 0,
                'metadata': {},
                'is_scanned': True,  # Assume scanned if extraction fails
                'error': str(e)
            }
    
    def _process_with_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Extract text using pdfplumber"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text_pages = []
                total_chars = 0
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text)
                        total_chars += len(page_text)
                
                # Determine if document is scanned (low text density)
                is_scanned = total_chars / len(pdf.pages) < 50 if pdf.pages else True
                
                return {
                    'success': True,
                    'text': '\n\n'.join(text_pages),
                    'pages': len(pdf.pages),
                    'metadata': pdf.metadata or {},
                    'is_scanned': is_scanned,
                    'processor': 'pdfplumber'
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_with_pypdf2(self, file_path: str) -> Dict[str, Any]:
        """Extract text using PyPDF2 as fallback"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_pages = []
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text)
                
                return {
                    'success': True,
                    'text': '\n\n'.join(text_pages),
                    'pages': len(reader.pages),
                    'metadata': reader.metadata or {},
                    'is_scanned': len('\n\n'.join(text_pages)) < 100,
                    'processor': 'pypdf2'
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
```

### DOCX Processing Implementation
```python
from docx import Document
import zipfile
from typing import Dict, Any

class DOCXProcessor:
    """Extract text and metadata from DOCX files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_docx(self, file_path: str) -> Dict[str, Any]:
        """Process DOCX file and extract content"""
        try:
            doc = Document(file_path)
            
            # Extract paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text.strip())
            
            # Extract tables
            tables_text = self._extract_tables(doc)
            
            # Combine all text
            full_text = '\n\n'.join(paragraphs)
            if tables_text:
                full_text += '\n\n' + tables_text
            
            # Extract metadata
            metadata = self._extract_docx_metadata(doc)
            
            return {
                'success': True,
                'text': full_text,
                'paragraphs': len(paragraphs),
                'tables': len(doc.tables),
                'metadata': metadata,
                'processor': 'python-docx'
            }
        except Exception as e:
            self.logger.error(f"DOCX processing failed for {file_path}: {e}")
            return {
                'success': False,
                'text': '',
                'error': str(e)
            }
    
    def _extract_tables(self, doc: Document) -> str:
        """Extract text from tables in DOCX"""
        tables_text = []
        
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(' | '.join(row_data))
            
            if table_data:
                tables_text.append('\n'.join(table_data))
        
        return '\n\n'.join(tables_text)
    
    def _extract_docx_metadata(self, doc: Document) -> Dict[str, Any]:
        """Extract metadata from DOCX core properties"""
        try:
            props = doc.core_properties
            return {
                'title': props.title or '',
                'author': props.author or '',
                'subject': props.subject or '',
                'created': props.created.isoformat() if props.created else '',
                'modified': props.modified.isoformat() if props.modified else '',
                'category': props.category or '',
                'comments': props.comments or ''
            }
        except Exception:
            return {}
```

### Normalization Orchestrator
```python
from typing import Dict, Any, List
import json

class DocumentNormalizer:
    """Main orchestrator for document format normalization"""
    
    def __init__(self):
        self.format_detector = FormatDetector()
        self.pdf_processor = PDFProcessor()
        self.docx_processor = DOCXProcessor()
        self.image_processor = ImageProcessor()
        self.logger = logging.getLogger(__name__)
    
    def normalize_document(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Normalize document to unified format"""
        try:
            # Detect format
            format_info = self.format_detector.detect_format(file_path)
            
            if format_info['confidence'] < 0.7:
                self.logger.warning(f"Low confidence format detection for {file_path}")
            
            # Process based on detected format
            if format_info['format'] == 'pdf':
                result = self.pdf_processor.process_pdf(file_path)
            elif format_info['format'] == 'docx':
                result = self.docx_processor.process_docx(file_path)
            elif format_info['format'] == 'image':
                result = self.image_processor.prepare_for_ocr(file_path)
            else:
                result = {
                    'success': False,
                    'text': '',
                    'error': f"Unsupported format: {format_info['format']}"
                }
            
            # Enrich with normalized metadata
            normalized_result = self._create_normalized_output(
                result, format_info, file_path, metadata
            )
            
            return normalized_result
            
        except Exception as e:
            self.logger.error(f"Document normalization failed for {file_path}: {e}")
            return {
                'success': False,
                'normalized_text': '',
                'error': str(e)
            }
    
    def _create_normalized_output(self, process_result: Dict, format_info: Dict, 
                                 file_path: str, input_metadata: Dict = None) -> Dict[str, Any]:
        """Create unified output format"""
        return {
            'file_path': file_path,
            'success': process_result.get('success', False),
            'normalized_text': process_result.get('text', ''),
            'format_detection': format_info,
            'processing_metadata': {
                'processor': process_result.get('processor', 'unknown'),
                'pages': process_result.get('pages', 0),
                'paragraphs': process_result.get('paragraphs', 0),
                'tables': process_result.get('tables', 0),
                'is_scanned': process_result.get('is_scanned', False),
                'character_count': len(process_result.get('text', '')),
                'word_count': len(process_result.get('text', '').split())
            },
            'document_metadata': process_result.get('metadata', {}),
            'input_metadata': input_metadata or {},
            'error': process_result.get('error', None)
        }
```

## VALIDATION LOOP

### Unit Testing
```python
import pytest
from pathlib import Path
from src.normalization.normalizer import DocumentNormalizer

class TestDocumentNormalization:
    def test_pdf_processing(self):
        normalizer = DocumentNormalizer()
        result = normalizer.normalize_document('test_files/sample.pdf')
        assert result['success'] == True
        assert len(result['normalized_text']) > 0
    
    def test_docx_processing(self):
        normalizer = DocumentNormalizer()
        result = normalizer.normalize_document('test_files/sample.docx')
        assert result['success'] == True
        assert result['processing_metadata']['paragraphs'] > 0
    
    def test_format_detection_accuracy(self):
        detector = FormatDetector()
        pdf_result = detector.detect_format('sample.pdf')
        assert pdf_result['format'] == 'pdf'
        assert pdf_result['confidence'] > 0.8
```

### Integration Testing
- Test complete normalization workflow with various file types
- Validate output format consistency across processors
- Test error handling for corrupted files

### Performance Testing
- Process 100 documents in under 5 minutes
- Memory usage under 500MB for large documents
- CPU utilization optimization for concurrent processing

## ADDITIONAL NOTES

### Security Considerations
- File type validation before processing
- Size limits to prevent memory exhaustion
- Malware scanning integration for uploaded files
- Input sanitization for file paths

### Performance Optimization
- Lazy loading for large documents
- Streaming processing for memory efficiency
- Parallel processing for multiple documents
- Caching for repeated format detection

### Maintenance Requirements
- Regular updates for document format libraries
- Format support expansion based on organizational needs
- Processing quality metrics and monitoring
- Error pattern analysis and optimization