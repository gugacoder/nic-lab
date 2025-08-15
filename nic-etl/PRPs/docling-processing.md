# Docling Processing - PRP

## ROLE
**Senior Document Intelligence Engineer with Docling Expertise**

Specialist in advanced document processing, OCR technologies, and document structure analysis using IBM's Docling framework. Expert in handling mixed document formats, extracting structured content from unstructured sources, and implementing intelligent document understanding pipelines. Proficient in distinguishing between digital and scanned documents, applying appropriate OCR strategies, and maintaining high-quality text extraction standards.

## OBJECTIVE
**Implement Comprehensive Docling-Based Document Processing**

Deliver a robust Docling processing module within Jupyter Notebook cells that:
* Processes all document formats (PDF, DOCX, images) through unified Docling pipeline
* Automatically detects and handles both digital and scanned documents
* Extracts structured content including titles, sections, paragraphs, tables, and figures
* Maintains page and region mapping for extracted content
* Generates confidence scores for OCR quality assessment
* Produces standardized JSON/Markdown output with complete metadata
* Ensures deterministic extraction for reproducible results

## MOTIVATION
**Intelligent Document Understanding for Knowledge Extraction**

Docling processing represents the intelligence layer of the ETL pipeline, transforming raw documents into structured, semantically-rich content. This critical component enables accurate content understanding, preserves document structure, and ensures high-quality text extraction regardless of source format. By leveraging Docling's advanced capabilities, the system can handle complex document layouts, extract tables and figures, and maintain the logical flow of information essential for downstream embedding and search operations.

## CONTEXT
**Docling Framework within Jupyter Notebook Environment**

Technical environment specifications:
* Framework: IBM Docling for unified document processing
* Document types: Mixed digital and scanned PDFs, DOCX, images (JPG, PNG)
* OCR requirement: Conditional based on document type detection
* Output formats: Structured JSON and Markdown with metadata
* Processing constraints: CPU-based processing, Jupyter Notebook cells
* Quality requirements: Confidence scoring, deterministic results
* Volume: Hundreds of documents with varying complexity
* Integration: Must preserve metadata for NIC Schema compliance

## IMPLEMENTATION BLUEPRINT
**Complete Docling Processing Architecture**

### Architecture Overview
```
Cell 5: Docling Processing
├── DoclingProcessor class
│   ├── Document type detection
│   ├── OCR management
│   ├── Structure extraction
│   ├── Content parsing
│   └── Output generation
├── Quality assessment
├── Confidence scoring
└── Result caching
```

### Code Structure
```python
# Cell 5: Docling Processing Functions
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat, OutputFormat
from docling.datamodel.pipeline_options import PipelineOptions, TableFormerMode
from docling.datamodel.document import ConversionResult
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path
from enum import Enum
import hashlib

class DocumentType(Enum):
    DIGITAL_PDF = "digital_pdf"
    SCANNED_PDF = "scanned_pdf"
    DOCX = "docx"
    IMAGE = "image"
    TEXT = "text"

class DoclingProcessor:
    def __init__(self, cache_dir: Path, enable_ocr: bool = True):
        self.cache_dir = cache_dir
        self.enable_ocr = enable_ocr
        self.results_cache = cache_dir / "docling_results"
        self.results_cache.mkdir(exist_ok=True)
        
        # Initialize Docling converter with optimized settings
        self.pipeline_options = self._configure_pipeline()
        self.converter = DocumentConverter(
            pipeline_options=self.pipeline_options
        )
        
    def _configure_pipeline(self) -> PipelineOptions:
        """Configure Docling pipeline options"""
        options = PipelineOptions()
        
        # Table extraction settings
        options.table_structure_options.mode = TableFormerMode.ACCURATE
        options.table_structure_options.do_cell_matching = True
        
        # OCR settings
        options.ocr_options.enabled = self.enable_ocr
        options.ocr_options.force_full_page_ocr = False
        options.ocr_options.lang = ["pt", "en"]  # Portuguese and English
        
        # Figure extraction
        options.images_options.extract = True
        options.images_options.resolution_ppi = 150
        
        # Document structure
        options.generate_page_images = True
        options.generate_picture_images = True
        
        return options
    
    def process_document(self, document: Document) -> Dict[str, Any]:
        """Process document through Docling pipeline"""
        # Check cache first
        cache_key = self._generate_cache_key(document)
        cached_result = self._load_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Detect document type
        doc_type = self._detect_document_type(document)
        
        # Configure processing based on type
        processing_config = self._get_processing_config(doc_type)
        
        # Process with Docling
        result = self._process_with_docling(document, processing_config)
        
        # Extract structured content
        structured_content = self._extract_structured_content(result)
        
        # Generate output formats
        output = self._generate_output(structured_content, document, doc_type)
        
        # Cache result
        self._cache_result(cache_key, output)
        
        return output
    
    def _detect_document_type(self, document: Document) -> DocumentType:
        """Detect if document is digital or scanned"""
        if document.format == DocumentFormat.DOCX:
            return DocumentType.DOCX
        elif document.format in [DocumentFormat.JPG, DocumentFormat.PNG]:
            return DocumentType.IMAGE
        elif document.format in [DocumentFormat.TXT, DocumentFormat.MARKDOWN]:
            return DocumentType.TEXT
        elif document.format == DocumentFormat.PDF:
            # Analyze PDF to determine if scanned
            with open(document.path, 'rb') as f:
                pdf_bytes = f.read(4096)  # Read first 4KB
                
                # Simple heuristic: check for text content
                if b'/Text' in pdf_bytes or b'/Font' in pdf_bytes:
                    return DocumentType.DIGITAL_PDF
                else:
                    return DocumentType.SCANNED_PDF
        
        return DocumentType.DIGITAL_PDF  # Default
    
    def _get_processing_config(self, doc_type: DocumentType) -> Dict[str, Any]:
        """Get processing configuration based on document type"""
        config = {
            'ocr_required': False,
            'extract_tables': True,
            'extract_figures': True,
            'extract_equations': False,
            'confidence_threshold': 0.85
        }
        
        if doc_type in [DocumentType.SCANNED_PDF, DocumentType.IMAGE]:
            config['ocr_required'] = True
            config['confidence_threshold'] = 0.75  # Lower threshold for OCR
        
        return config
    
    def _process_with_docling(self, document: Document, config: Dict[str, Any]) -> ConversionResult:
        """Process document using Docling converter"""
        # Set OCR based on configuration
        self.pipeline_options.ocr_options.enabled = config['ocr_required']
        
        # Convert document
        result = self.converter.convert(
            document.path,
            output_format=OutputFormat.JSON
        )
        
        return result
    
    def _extract_structured_content(self, result: ConversionResult) -> Dict[str, Any]:
        """Extract structured content from Docling result"""
        structured = {
            'title': None,
            'sections': [],
            'paragraphs': [],
            'tables': [],
            'figures': [],
            'lists': [],
            'metadata': {},
            'page_structure': []
        }
        
        # Extract document structure
        if result.document:
            # Title extraction
            structured['title'] = result.document.title
            
            # Page-by-page extraction
            for page_num, page in enumerate(result.document.pages, 1):
                page_content = {
                    'page_number': page_num,
                    'elements': []
                }
                
                for element in page.elements:
                    element_data = {
                        'type': element.type,
                        'text': element.text,
                        'confidence': element.confidence if hasattr(element, 'confidence') else 1.0,
                        'bbox': element.bbox if hasattr(element, 'bbox') else None
                    }
                    
                    # Categorize element
                    if element.type == 'heading':
                        structured['sections'].append({
                            'level': element.level if hasattr(element, 'level') else 1,
                            'text': element.text,
                            'page': page_num
                        })
                    elif element.type == 'paragraph':
                        structured['paragraphs'].append({
                            'text': element.text,
                            'page': page_num,
                            'confidence': element_data['confidence']
                        })
                    elif element.type == 'table':
                        structured['tables'].append({
                            'content': element.content if hasattr(element, 'content') else element.text,
                            'page': page_num,
                            'rows': element.rows if hasattr(element, 'rows') else None,
                            'columns': element.columns if hasattr(element, 'columns') else None
                        })
                    elif element.type == 'figure':
                        structured['figures'].append({
                            'caption': element.caption if hasattr(element, 'caption') else None,
                            'page': page_num,
                            'image_path': element.image_path if hasattr(element, 'image_path') else None
                        })
                    elif element.type == 'list':
                        structured['lists'].append({
                            'items': element.items if hasattr(element, 'items') else [element.text],
                            'page': page_num
                        })
                    
                    page_content['elements'].append(element_data)
                
                structured['page_structure'].append(page_content)
            
            # Extract metadata
            structured['metadata'] = {
                'page_count': len(result.document.pages),
                'language': result.document.language if hasattr(result.document, 'language') else 'unknown',
                'confidence_score': result.confidence if hasattr(result, 'confidence') else None
            }
        
        return structured
    
    def _generate_output(self, structured: Dict[str, Any], document: Document, doc_type: DocumentType) -> Dict[str, Any]:
        """Generate final output with multiple formats"""
        output = {
            'document_id': document.id,
            'filename': document.filename,
            'document_type': doc_type.value,
            'processing_metadata': {
                'ocr_applied': doc_type in [DocumentType.SCANNED_PDF, DocumentType.IMAGE],
                'processor': 'docling',
                'timestamp': datetime.now().isoformat(),
                'confidence_scores': self._calculate_confidence_scores(structured)
            },
            'structured_content': structured,
            'markdown': self._generate_markdown(structured),
            'json': structured,
            'provenance': {
                'source': document.gitlab_path,
                'branch': document.branch,
                'commit_id': document.commit_id,
                'content_hash': document.content_hash
            }
        }
        
        return output
    
    def _generate_markdown(self, structured: Dict[str, Any]) -> str:
        """Convert structured content to Markdown format"""
        markdown_parts = []
        
        # Title
        if structured['title']:
            markdown_parts.append(f"# {structured['title']}\n")
        
        # Build content by page order
        for page in structured['page_structure']:
            for element in page['elements']:
                if element['type'] == 'heading':
                    level = '#' * (element.get('level', 1) + 1)
                    markdown_parts.append(f"\n{level} {element['text']}\n")
                elif element['type'] == 'paragraph':
                    markdown_parts.append(f"\n{element['text']}\n")
                elif element['type'] == 'list':
                    for item in element.get('items', [element['text']]):
                        markdown_parts.append(f"- {item}")
                    markdown_parts.append("")
                elif element['type'] == 'table':
                    markdown_parts.append("\n[TABLE]\n")
                    markdown_parts.append(element['text'])
                    markdown_parts.append("\n")
                elif element['type'] == 'figure':
                    caption = element.get('caption', 'Figure')
                    markdown_parts.append(f"\n![{caption}](figure)\n")
        
        return '\n'.join(markdown_parts)
    
    def _calculate_confidence_scores(self, structured: Dict[str, Any]) -> Dict[str, float]:
        """Calculate average confidence scores by element type"""
        scores = {}
        
        # Calculate paragraph confidence
        if structured['paragraphs']:
            para_scores = [p.get('confidence', 1.0) for p in structured['paragraphs']]
            scores['paragraphs'] = sum(para_scores) / len(para_scores)
        
        # Overall confidence
        all_scores = []
        for page in structured['page_structure']:
            for element in page['elements']:
                if 'confidence' in element:
                    all_scores.append(element['confidence'])
        
        if all_scores:
            scores['overall'] = sum(all_scores) / len(all_scores)
        else:
            scores['overall'] = 1.0
        
        return scores
    
    def _generate_cache_key(self, document: Document) -> str:
        """Generate cache key for processed document"""
        key_parts = [
            document.content_hash,
            document.format.name,
            str(self.enable_ocr)
        ]
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache processing result"""
        cache_file = self.results_cache / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _load_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached processing result"""
        cache_file = self.results_cache / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

class QualityAssessor:
    """Assess document processing quality"""
    
    @staticmethod
    def assess_extraction_quality(result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of text extraction"""
        assessment = {
            'is_acceptable': True,
            'issues': [],
            'recommendations': []
        }
        
        confidence_scores = result['processing_metadata']['confidence_scores']
        
        # Check overall confidence
        if confidence_scores.get('overall', 0) < 0.7:
            assessment['is_acceptable'] = False
            assessment['issues'].append('Low overall confidence score')
            assessment['recommendations'].append('Consider manual review or re-scanning')
        
        # Check content completeness
        structured = result['structured_content']
        if not structured['paragraphs'] and not structured['tables']:
            assessment['is_acceptable'] = False
            assessment['issues'].append('No content extracted')
            assessment['recommendations'].append('Check if document is corrupted or encrypted')
        
        # Check for OCR issues
        if result['processing_metadata']['ocr_applied']:
            if confidence_scores.get('overall', 0) < 0.85:
                assessment['recommendations'].append('OCR quality may be improved with higher resolution scan')
        
        return assessment
```

### Error Handling
```python
class DoclingProcessingError(Exception):
    """Base exception for Docling processing errors"""
    pass

class OCRError(DoclingProcessingError):
    """Raised when OCR fails"""
    pass

class StructureExtractionError(DoclingProcessingError):
    """Raised when structure extraction fails"""
    pass

def safe_process_with_retry(processor: DoclingProcessor, document: Document, max_retries: int = 2):
    """Process document with retry logic"""
    for attempt in range(max_retries):
        try:
            return processor.process_document(document)
        except Exception as e:
            if attempt == max_retries - 1:
                raise DoclingProcessingError(f"Failed after {max_retries} attempts: {str(e)}")
            
            # Try with different settings
            if attempt == 0:
                processor.pipeline_options.ocr_options.force_full_page_ocr = True
```

## VALIDATION LOOP
**Comprehensive Docling Processing Testing**

### Unit Testing
```python
def test_document_type_detection():
    """Test accurate document type detection"""
    processor = DoclingProcessor(CACHE_DIR)
    
    # Test different document types
    test_docs = [
        (create_test_doc("digital.pdf", DocumentFormat.PDF), DocumentType.DIGITAL_PDF),
        (create_test_doc("scanned.pdf", DocumentFormat.PDF), DocumentType.SCANNED_PDF),
        (create_test_doc("document.docx", DocumentFormat.DOCX), DocumentType.DOCX),
        (create_test_doc("image.jpg", DocumentFormat.JPG), DocumentType.IMAGE)
    ]
    
    for doc, expected_type in test_docs:
        detected = processor._detect_document_type(doc)
        assert detected == expected_type

def test_structure_extraction():
    """Test document structure extraction"""
    processor = DoclingProcessor(CACHE_DIR)
    
    # Process test document
    test_doc = create_test_document_with_structure()
    result = processor.process_document(test_doc)
    
    structured = result['structured_content']
    assert structured['title'] is not None
    assert len(structured['sections']) > 0
    assert len(structured['paragraphs']) > 0

def test_confidence_scoring():
    """Test confidence score calculation"""
    processor = DoclingProcessor(CACHE_DIR)
    
    structured = {
        'paragraphs': [
            {'confidence': 0.9},
            {'confidence': 0.85},
            {'confidence': 0.95}
        ],
        'page_structure': []
    }
    
    scores = processor._calculate_confidence_scores(structured)
    assert 0.85 < scores['paragraphs'] < 0.95
```

### Integration Testing
```python
def test_full_docling_pipeline():
    """Test complete Docling processing pipeline"""
    processor = DoclingProcessor(CACHE_DIR, enable_ocr=True)
    manager = DocumentManager(CACHE_DIR, STATE_FILE)
    
    # Get sample documents
    documents = manager.get_pending_documents()[:5]
    
    results = []
    for doc in documents:
        result = processor.process_document(doc)
        quality = QualityAssessor.assess_extraction_quality(result)
        
        results.append({
            'document': doc.filename,
            'type': result['document_type'],
            'quality': quality['is_acceptable'],
            'confidence': result['processing_metadata']['confidence_scores']['overall']
        })
    
    assert all(r['quality'] for r in results)

def test_ocr_vs_digital_processing():
    """Test different processing paths for OCR vs digital"""
    processor = DoclingProcessor(CACHE_DIR)
    
    # Process digital PDF
    digital_doc = get_digital_test_document()
    digital_result = processor.process_document(digital_doc)
    assert not digital_result['processing_metadata']['ocr_applied']
    
    # Process scanned PDF
    scanned_doc = get_scanned_test_document()
    scanned_result = processor.process_document(scanned_doc)
    assert scanned_result['processing_metadata']['ocr_applied']
```

### Performance Testing
* Processing speed: Target < 10 seconds per page for digital, < 30 seconds for OCR
* Memory usage: < 500MB per document
* Cache hit ratio: > 95% for repeated processing
* Confidence thresholds: > 0.85 for digital, > 0.75 for OCR

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
* **Input Validation**: Validate document format before processing
* **Resource Limits**: Set maximum file size and processing time limits
* **Sandboxing**: Process untrusted documents in isolated environment
* **Output Sanitization**: Clean extracted text from potential injection attacks
* **Secure Caching**: Encrypt cached results containing sensitive data

### Performance Optimization
* **Parallel Processing**: Process multiple pages concurrently
* **Selective OCR**: Apply OCR only to pages that need it
* **Result Caching**: Cache processed results by content hash
* **Memory Management**: Stream large documents instead of loading entirely
* **GPU Acceleration**: Use GPU for OCR when available

### Maintenance Requirements
* **Docling Updates**: Regular updates to latest Docling version
* **Model Updates**: Update OCR models for better accuracy
* **Language Support**: Add new language models as needed
* **Quality Monitoring**: Track extraction quality metrics
* **Error Analysis**: Regular review of failed extractions