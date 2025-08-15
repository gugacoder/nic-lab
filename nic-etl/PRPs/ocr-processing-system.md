# OCR Processing System - PRP

## ROLE
**Python Developer with OCR and Computer Vision expertise**

Responsible for implementing robust OCR pipeline using open-source tools (Tesseract, OCRmyPDF, Poppler) to extract text from scanned documents and images. Must have experience with image preprocessing, OCR optimization, and text extraction quality assessment.

## OBJECTIVE
**Implement high-quality OCR processing for scanned documents and images**

Develop a comprehensive OCR system that:
- Detects when documents require OCR processing (scanned vs. digital text)
- Applies appropriate preprocessing to optimize OCR accuracy
- Uses Tesseract OCR with optimized configurations
- Handles multiple languages and document layouts
- Generates searchable PDF outputs using OCRmyPDF
- Provides confidence scores and quality metrics
- Maintains original document structure and layout information

Success criteria: Achieve >90% text extraction accuracy on typical business documents with <5% false positive rate for OCR necessity detection.

## MOTIVATION
**Enable text extraction from image-based and scanned documents**

Many organizational documents exist as scanned PDFs or image files without searchable text. OCR processing converts these visual documents into machine-readable text, enabling full-text search, semantic analysis, and AI-powered insights across the complete document corpus.

## CONTEXT
**NIC ETL Pipeline - OCR Processing Phase**

Technology Stack:
- Python 3.8+ with jupyter notebook environment
- Tesseract OCR engine with Portuguese language support
- OCRmyPDF for PDF OCR processing
- Poppler for PDF to image conversion
- Pillow/OpenCV for image preprocessing
- Input from document format normalization pipeline
- Output to document structuring pipeline

OCR Requirements:
- Primary language: Portuguese (Brazil)
- Secondary languages: English, Spanish
- Document types: Business reports, technical documentation, forms
- Quality threshold: Minimum 85% confidence score

## IMPLEMENTATION BLUEPRINT

### Architecture Overview
```
Input Documents → Scanned Detection → Image Preprocessing → Tesseract OCR → Text Post-processing → Quality Assessment → Structured Output
```

### Code Structure
```python
# File organization
src/
├── ocr/
│   ├── __init__.py
│   ├── scan_detector.py          # Scanned document detection
│   ├── image_preprocessor.py     # Image enhancement for OCR
│   ├── tesseract_wrapper.py      # Tesseract OCR integration
│   ├── ocrmypdf_processor.py     # OCRmyPDF integration
│   ├── text_postprocessor.py     # OCR output cleaning
│   ├── quality_assessor.py       # OCR quality metrics
│   └── ocr_orchestrator.py       # Main OCR pipeline
├── config/
│   └── ocr_config.py             # OCR configuration settings
└── notebooks/
    └── 03_ocr_processing.ipynb
```

### Scanned Document Detection
```python
import cv2
import numpy as np
from typing import Dict, Any, Tuple
import logging
from PIL import Image
import pdf2image

class ScanDetector:
    """Detect if documents are scanned/image-based or contain searchable text"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_scanned_document(self, file_path: str, normalized_result: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if document requires OCR processing"""
        try:
            # Method 1: Text density analysis from normalization phase
            text_density_score = self._analyze_text_density(normalized_result)
            
            # Method 2: Image analysis for scanned characteristics
            image_analysis_score = self._analyze_image_characteristics(file_path)
            
            # Method 3: Text pattern analysis
            pattern_analysis_score = self._analyze_text_patterns(normalized_result.get('normalized_text', ''))
            
            # Combine scores for final decision
            combined_score = (text_density_score * 0.5 + 
                            image_analysis_score * 0.3 + 
                            pattern_analysis_score * 0.2)
            
            is_scanned = combined_score > 0.7
            
            return {
                'is_scanned': is_scanned,
                'confidence': combined_score,
                'analysis': {
                    'text_density': text_density_score,
                    'image_characteristics': image_analysis_score,
                    'text_patterns': pattern_analysis_score
                },
                'recommendation': 'ocr_required' if is_scanned else 'text_available'
            }
            
        except Exception as e:
            self.logger.error(f"Scan detection failed for {file_path}: {e}")
            return {
                'is_scanned': True,  # Default to OCR if uncertain
                'confidence': 0.5,
                'error': str(e)
            }
    
    def _analyze_text_density(self, normalized_result: Dict[str, Any]) -> float:
        """Analyze text density from normalization results"""
        if not normalized_result.get('success', False):
            return 1.0  # Assume scanned if normalization failed
        
        text = normalized_result.get('normalized_text', '')
        pages = normalized_result.get('processing_metadata', {}).get('pages', 1)
        
        if not text or pages == 0:
            return 1.0
        
        chars_per_page = len(text) / pages
        
        # Threshold: < 100 characters per page suggests scanned
        if chars_per_page < 100:
            return 1.0
        elif chars_per_page < 500:
            return 0.8
        else:
            return 0.2
    
    def _analyze_image_characteristics(self, file_path: str) -> float:
        """Analyze visual characteristics of document pages"""
        try:
            if file_path.lower().endswith('.pdf'):
                # Convert first page to image for analysis
                pages = pdf2image.convert_from_path(file_path, first_page=1, last_page=1)
                if not pages:
                    return 0.5
                image = np.array(pages[0])
            else:
                # Direct image analysis
                image = cv2.imread(file_path)
            
            if image is None:
                return 0.5
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate metrics indicating scanned documents
            noise_level = self._calculate_noise_level(gray)
            edge_density = self._calculate_edge_density(gray)
            text_regions = self._detect_text_regions(gray)
            
            # Combine metrics (higher values = more likely scanned)
            scan_likelihood = (noise_level * 0.4 + edge_density * 0.3 + text_regions * 0.3)
            
            return scan_likelihood
            
        except Exception as e:
            self.logger.warning(f"Image analysis failed: {e}")
            return 0.5
    
    def _calculate_noise_level(self, gray_image: np.ndarray) -> float:
        """Calculate noise level in image"""
        # Use Laplacian variance to detect blur/noise
        variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        # Normalize and invert (higher variance = less noise = less likely scanned)
        return max(0, 1 - (variance / 1000))
    
    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density (scanned docs have more irregular edges)"""
        edges = cv2.Canny(gray_image, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        # Higher edge density suggests scanned document
        return min(1.0, edge_ratio * 10)
```

### Image Preprocessing
```python
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Dict, Any

class ImagePreprocessor:
    """Enhance images for optimal OCR performance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_for_ocr(self, image_path: str) -> Dict[str, Any]:
        """Apply preprocessing pipeline to optimize OCR accuracy"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocessing pipeline
            processed = self._deskew_image(image)
            processed = self._enhance_contrast(processed)
            processed = self._reduce_noise(processed)
            processed = self._binarize_image(processed)
            processed = self._remove_artifacts(processed)
            
            # Save preprocessed image
            output_path = image_path.replace('.', '_preprocessed.')
            cv2.imwrite(output_path, processed)
            
            # Calculate improvement metrics
            quality_metrics = self._calculate_quality_metrics(image, processed)
            
            return {
                'success': True,
                'preprocessed_path': output_path,
                'quality_metrics': quality_metrics,
                'preprocessing_steps': [
                    'deskew', 'contrast_enhancement', 'noise_reduction', 
                    'binarization', 'artifact_removal'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed for {image_path}: {e}")
            return {
                'success': False,
                'preprocessed_path': image_path,  # Use original
                'error': str(e)
            }
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct document skew/rotation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect lines using HoughLinesP
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Calculate average angle of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                
                # Rotate image if significant skew detected (> 1 degree)
                if abs(median_angle) > 1:
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
        
        return image
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction filtering"""
        # Use bilateral filter to reduce noise while preserving edges
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary image using adaptive thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def _remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Remove small artifacts and clean up image"""
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Remove noise
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Fill gaps
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        return image
```

### Tesseract OCR Integration
```python
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import Dict, Any, List
import json

class TesseractProcessor:
    """Tesseract OCR wrapper with optimized configurations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Configure Tesseract for Portuguese + English
        self.languages = 'por+eng'
        self.base_config = '--oem 3 --psm 6'  # LSTM + uniform text block
    
    def extract_text(self, image_path: str, preprocessing_result: Dict = None) -> Dict[str, Any]:
        """Extract text using Tesseract with multiple configurations"""
        try:
            # Use preprocessed image if available
            input_path = preprocessing_result.get('preprocessed_path', image_path) if preprocessing_result else image_path
            
            # Try multiple PSM modes for best results
            psm_modes = [6, 3, 4, 8]  # Different page segmentation modes
            best_result = None
            best_confidence = 0
            
            for psm in psm_modes:
                config = f'--oem 3 --psm {psm}'
                result = self._extract_with_config(input_path, config)
                
                if result['confidence'] > best_confidence:
                    best_result = result
                    best_confidence = result['confidence']
            
            # Enhanced post-processing
            if best_result and best_result['success']:
                best_result['text'] = self._post_process_text(best_result['text'])
                best_result['preprocessing_used'] = preprocessing_result is not None
            
            return best_result or {'success': False, 'text': '', 'confidence': 0}
            
        except Exception as e:
            self.logger.error(f"Tesseract OCR failed for {image_path}: {e}")
            return {'success': False, 'text': '', 'confidence': 0, 'error': str(e)}
    
    def _extract_with_config(self, image_path: str, config: str) -> Dict[str, Any]:
        """Extract text with specific Tesseract configuration"""
        try:
            image = Image.open(image_path)
            
            # Extract text with confidence data
            data = pytesseract.image_to_data(
                image, 
                lang=self.languages,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract full text
            text = pytesseract.image_to_string(
                image,
                lang=self.languages,
                config=config
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract word-level information
            word_info = self._extract_word_details(data)
            
            return {
                'success': True,
                'text': text.strip(),
                'confidence': avg_confidence,
                'word_count': len([w for w in data['text'] if w.strip()]),
                'word_details': word_info,
                'config_used': config
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_word_details(self, tesseract_data: Dict) -> List[Dict]:
        """Extract detailed word-level OCR information"""
        words = []
        
        for i, word in enumerate(tesseract_data['text']):
            if word.strip():
                words.append({
                    'text': word,
                    'confidence': tesseract_data['conf'][i],
                    'bbox': {
                        'x': tesseract_data['left'][i],
                        'y': tesseract_data['top'][i],
                        'width': tesseract_data['width'][i],
                        'height': tesseract_data['height'][i]
                    }
                })
        
        return words
    
    def _post_process_text(self, text: str) -> str:
        """Clean and improve OCR text output"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR errors for Portuguese
        corrections = {
            'ç': 'ç',  # Ensure proper encoding
            '0': 'o',  # Common number/letter confusion
            '1': 'I',  # In specific contexts
            '5': 'S',  # In specific contexts
        }
        
        # Apply basic corrections (can be expanded with ML-based correction)
        for wrong, right in corrections.items():
            # Context-aware replacements would go here
            pass
        
        return text
```

### OCR Quality Assessment
```python
import re
from typing import Dict, Any, List
import statistics

class OCRQualityAssessor:
    """Assess OCR output quality and provide confidence metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def assess_quality(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive OCR quality assessment"""
        try:
            text = ocr_result.get('text', '')
            word_details = ocr_result.get('word_details', [])
            base_confidence = ocr_result.get('confidence', 0)
            
            # Multiple quality metrics
            language_score = self._assess_language_patterns(text)
            coherence_score = self._assess_text_coherence(text)
            confidence_distribution = self._assess_confidence_distribution(word_details)
            formatting_score = self._assess_formatting_quality(text)
            
            # Combined quality score
            overall_quality = (
                base_confidence * 0.3 +
                language_score * 0.25 +
                coherence_score * 0.25 +
                confidence_distribution * 0.2
            )
            
            quality_assessment = {
                'overall_quality': overall_quality,
                'quality_grade': self._grade_quality(overall_quality),
                'metrics': {
                    'tesseract_confidence': base_confidence,
                    'language_patterns': language_score,
                    'text_coherence': coherence_score,
                    'confidence_distribution': confidence_distribution,
                    'formatting_quality': formatting_score
                },
                'recommendations': self._generate_recommendations(overall_quality, text, word_details),
                'pass_threshold': overall_quality >= 75  # Minimum quality threshold
            }
            
            return quality_assessment
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return {
                'overall_quality': 0,
                'quality_grade': 'F',
                'pass_threshold': False,
                'error': str(e)
            }
    
    def _assess_language_patterns(self, text: str) -> float:
        """Assess if text follows expected language patterns"""
        if not text:
            return 0
        
        # Portuguese/English language indicators
        portuguese_patterns = [
            r'\b(que|para|com|por|uma|são|não|mais|como|dos|das)\b',
            r'ção\b', r'mente\b', r'ade\b'
        ]
        
        english_patterns = [
            r'\b(the|and|for|are|with|this|that|from|they)\b',
            r'tion\b', r'ing\b', r'ness\b'
        ]
        
        # Count pattern matches
        pt_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in portuguese_patterns)
        en_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in english_patterns)
        
        total_words = len(text.split())
        pattern_ratio = (pt_matches + en_matches) / total_words if total_words > 0 else 0
        
        return min(100, pattern_ratio * 100)
    
    def _assess_text_coherence(self, text: str) -> float:
        """Assess text coherence and readability"""
        if not text:
            return 0
        
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0
        
        # Metrics for coherence
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Penalize extremely short or long sentences (likely OCR errors)
        ideal_length_score = max(0, 100 - abs(avg_sentence_length - 15) * 2)
        
        # Check for excessive special characters (OCR artifacts)
        special_char_ratio = len(re.findall(r'[^a-zA-ZÀ-ÿ0-9\s.,;:!?()-]', text)) / len(text) if text else 0
        special_char_penalty = max(0, 100 - special_char_ratio * 200)
        
        return (ideal_length_score + special_char_penalty) / 2
    
    def _assess_confidence_distribution(self, word_details: List[Dict]) -> float:
        """Assess distribution of word-level confidences"""
        if not word_details:
            return 0
        
        confidences = [word['confidence'] for word in word_details if word['confidence'] > 0]
        if not confidences:
            return 0
        
        # Statistics on confidence distribution
        mean_confidence = statistics.mean(confidences)
        std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0
        
        # High standard deviation indicates inconsistent OCR quality
        consistency_score = max(0, 100 - std_confidence)
        
        return (mean_confidence + consistency_score) / 2
```

## VALIDATION LOOP

### Unit Testing
```python
import pytest
from src.ocr.ocr_orchestrator import OCROrchestrator

class TestOCRProcessing:
    def test_scan_detection_accuracy(self):
        detector = ScanDetector()
        # Test with known scanned document
        result = detector.is_scanned_document('test_scanned.pdf', {})
        assert result['is_scanned'] == True
        assert result['confidence'] > 0.7
    
    def test_tesseract_text_extraction(self):
        processor = TesseractProcessor()
        result = processor.extract_text('test_clear_text.png')
        assert result['success'] == True
        assert result['confidence'] > 80
        assert len(result['text']) > 0
    
    def test_image_preprocessing_improvement(self):
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess_for_ocr('test_noisy_image.png')
        assert result['success'] == True
        assert 'quality_metrics' in result
```

### Integration Testing
- End-to-end OCR pipeline with various document types
- Validation against ground truth text samples
- Performance testing with batch processing

### Performance Testing
- Process 50 pages per minute minimum
- Memory usage under 2GB for large images
- Accuracy >90% on business documents

## ADDITIONAL NOTES

### Security Considerations
- Validate image file formats and sizes
- Prevent resource exhaustion attacks
- Secure handling of sensitive document content
- Audit logging for OCR processing activities

### Performance Optimization
- GPU acceleration for Tesseract (if available)
- Parallel processing of multiple pages
- Image compression optimization
- Caching of preprocessing results

### Maintenance Requirements
- Regular Tesseract language pack updates
- OCR accuracy monitoring and reporting
- Quality threshold adjustment based on document types
- Integration with document feedback loop for continuous improvement