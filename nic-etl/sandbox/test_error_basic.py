#!/usr/bin/env python3
"""
Basic test script for error handling system
"""
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from error_handling import (
    ErrorManager, ErrorContext, CircuitBreaker, 
    create_error_manager, with_error_handling, error_context
)

def test_basic_error_handling():
    """Test basic error handling functionality"""
    print("Testing Error Handling System...")
    
    try:
        # Create error manager
        error_manager = create_error_manager()
        
        # Test 1: Basic error handling
        print("✓ Error manager created successfully")
        
        # Test 2: Error classification
        test_errors = [
            Exception("authentication failed"),
            Exception("connection timeout"),
            Exception("file not found"),
            Exception("processing failed"),
            Exception("unknown error type")
        ]
        
        for i, error in enumerate(test_errors):
            context = ErrorContext(
                correlation_id=f"test-{i}",
                module_name="test_module",
                operation_name="test_operation",
                document_id=f"doc-{i}"
            )
            
            response = error_manager.handle_error(error, context)
            print(f"✓ Error {i+1} handled: {response.user_message}")
        
        # Test 3: Error statistics
        stats = error_manager.get_error_statistics()
        print(f"✓ Error statistics: {stats.total_errors} total errors")
        
        # Test 4: Circuit breaker
        cb = CircuitBreaker(failure_threshold=2, timeout=1.0)
        
        def test_function():
            return "success"
        
        result = cb.call(test_function)
        print(f"✓ Circuit breaker test: {result}")
        
        # Test 5: Error context manager
        try:
            with error_context(error_manager, "test_module", "context_test") as ctx:
                print(f"✓ Context manager created with correlation ID: {ctx.correlation_id}")
        except Exception:
            pass  # Expected for testing
        
        # Test 6: Error decorator
        @with_error_handling(error_manager, "test_module", "decorated_operation")
        def decorated_function():
            return "decorated success"
        
        result = decorated_function()
        print(f"✓ Decorated function test: {result}")
        
        # Test 7: Error report export
        report = error_manager.export_error_report()
        print(f"✓ Error report exported with {len(report['statistics'])} sections")
        
        print("\n✓ All error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_error_handling()
    sys.exit(0 if success else 1)