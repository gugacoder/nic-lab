#!/usr/bin/env python3
"""
Document Preview Validation Test

Quick validation test to ensure all preview components are working correctly.
Can be run standalone: python tests/validation_test.py
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all preview components can be imported"""
    print("Testing imports...")
    
    try:
        from src.components.preview.document_viewer import DocumentViewer, DocumentViewerConfig
        print("✅ DocumentViewer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import DocumentViewer: {e}")
        return False
    
    try:
        from src.components.preview.zoom_controls import ZoomControls, ZoomLevel
        print("✅ ZoomControls imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ZoomControls: {e}")
        return False
    
    try:
        from src.components.preview.page_display import PageDisplay, PageContent, PageSize
        print("✅ PageDisplay imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PageDisplay: {e}")
        return False
    
    try:
        from src.components.preview.preview_renderer import PreviewRenderer, RenderConfig
        print("✅ PreviewRenderer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PreviewRenderer: {e}")
        return False
    
    try:
        from src.utils.style_mapper import StyleMapper, get_style_mapper
        print("✅ StyleMapper imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import StyleMapper: {e}")
        return False
    
    return True

def test_css_file():
    """Test that CSS file exists and is readable"""
    print("\nTesting CSS file...")
    
    css_path = "src/styles/document_preview.css"
    
    if not os.path.exists(css_path):
        print(f"❌ CSS file not found: {css_path}")
        return False
    
    try:
        with open(css_path, 'r') as f:
            content = f.read()
        
        if len(content) < 1000:  # Basic size check
            print(f"❌ CSS file seems too small ({len(content)} chars)")
            return False
        
        # Check for key CSS classes
        required_classes = [
            '.document-preview-container',
            '.document-page',
            '.document-content',
            '.zoom-controls',
            '.doc-heading-1'
        ]
        
        for css_class in required_classes:
            if css_class not in content:
                print(f"❌ Required CSS class not found: {css_class}")
                return False
        
        print(f"✅ CSS file is valid ({len(content)} chars)")
        return True
        
    except Exception as e:
        print(f"❌ Error reading CSS file: {e}")
        return False

def test_style_mapper():
    """Test basic style mapper functionality"""
    print("\nTesting StyleMapper...")
    
    try:
        from src.utils.style_mapper import get_style_mapper, CSSStyle
        
        mapper = get_style_mapper()
        
        # Test basic style mapping
        test_style = {
            'font_family': 'Arial',
            'font_size': 12,
            'weight': 'bold',
            'color': '#000000'
        }
        
        css_style = mapper.map_text_style_to_css(test_style)
        
        if not isinstance(css_style, CSSStyle):
            print("❌ Style mapping didn't return CSSStyle object")
            return False
        
        css_string = css_style.to_css_string()
        
        if 'Arial' not in css_string or 'bold' not in css_string:
            print(f"❌ Style mapping incomplete: {css_string}")
            return False
        
        print(f"✅ StyleMapper working correctly")
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ StyleMapper test failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def test_page_content_creation():
    """Test PageContent creation and basic functionality"""
    print("\nTesting PageContent...")
    
    try:
        from src.components.preview.page_display import PageContent, PageSize
        
        # Create test page
        page = PageContent(
            page_number=1,
            content="<h1>Test Page</h1><p>Test content</p>",
            content_type="html",
            page_size=PageSize.A4
        )
        
        if page.page_number != 1:
            print("❌ Page number not set correctly")
            return False
        
        if "Test Page" not in page.content:
            print("❌ Page content not set correctly")
            return False
        
        if page.page_size != PageSize.A4:
            print("❌ Page size not set correctly")
            return False
        
        print("✅ PageContent creation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ PageContent test failed: {e}")
        return False

def test_zoom_calculations():
    """Test zoom calculation utilities"""
    print("\nTesting Zoom calculations...")
    
    try:
        from src.components.preview.zoom_controls import (
            get_zoom_levels_for_range,
            snap_to_nearest_zoom_level,
            is_zoom_level_valid
        )
        
        # Test zoom levels generation
        levels = get_zoom_levels_for_range(50, 200, 25)
        expected_levels = [50, 75, 100, 125, 150, 175, 200]
        
        if levels != expected_levels:
            print(f"❌ Zoom levels incorrect: {levels}")
            return False
        
        # Test snapping
        snapped = snap_to_nearest_zoom_level(87, [50, 75, 100, 125])
        if snapped != 75:
            print(f"❌ Zoom snapping incorrect: {snapped}")
            return False
        
        # Test validation
        if not is_zoom_level_valid(100):
            print("❌ Zoom validation failed for valid level")
            return False
        
        if is_zoom_level_valid(500):
            print("❌ Zoom validation failed for invalid level")
            return False
        
        print("✅ Zoom calculations working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Zoom calculations test failed: {e}")
        return False

def run_all_tests():
    """Run all validation tests"""
    print("🧪 Running Document Preview Validation Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_css_file,
        test_style_mapper,
        test_page_content_creation,
        test_zoom_calculations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Document preview system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)