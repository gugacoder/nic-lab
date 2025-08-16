import pytest
import time
import sys
from pathlib import Path

# Add modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from error_handling import (
    ErrorManager, ErrorContext, ErrorSeverity, ErrorCategory, 
    CircuitBreaker, create_error_manager
)

class TestErrorManager:
    
    @pytest.fixture
    def error_manager(self):
        return create_error_manager()
    
    @pytest.fixture
    def error_context(self):
        return ErrorContext(
            correlation_id="test-123",
            module_name="test_module",
            operation_name="test_operation"
        )
    
    def test_error_classification(self, error_manager):
        """Test error classification logic"""
        
        # Test GitLab authentication error
        auth_error = Exception("authentication failed")
        error_code = error_manager._classify_error(auth_error)
        assert error_code == "GITLAB_AUTH_FAILED"
        
        # Test network error
        network_error = Exception("connection timeout")
        error_code = error_manager._classify_error(network_error)
        assert error_code == "GITLAB_NETWORK_ERROR"
        
        # Test unknown error
        unknown_error = Exception("some random error")
        error_code = error_manager._classify_error(unknown_error)
        assert error_code == "UNKNOWN_ERROR"
    
    def test_error_handling_flow(self, error_manager, error_context):
        """Test complete error handling flow"""
        
        test_error = Exception("Test error for handling")
        response = error_manager.handle_error(test_error, error_context)
        
        assert response.correlation_id == error_context.correlation_id
        assert response.user_message is not None
        assert response.technical_details is not None
        assert hasattr(response, 'should_retry')
        assert hasattr(response, 'retry_delay')
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        
        cb = CircuitBreaker(failure_threshold=2, timeout=1.0)
        
        # Test function that always fails
        def failing_function():
            raise Exception("Always fails")
        
        # First failure
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == "CLOSED"
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == "OPEN"
        
        # Third call should fail due to open circuit
        with pytest.raises(Exception, match="Circuit breaker OPEN"):
            cb.call(failing_function)
    
    def test_error_statistics(self, error_manager, error_context):
        """Test error statistics tracking"""
        
        # Generate some test errors
        for i in range(5):
            test_error = Exception(f"Test error {i}")
            error_manager.handle_error(test_error, error_context)
        
        stats = error_manager.get_error_statistics()
        assert stats.total_errors > 0
        assert isinstance(stats.errors_by_category, dict)
        assert isinstance(stats.errors_by_severity, dict)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism"""
        
        cb = CircuitBreaker(failure_threshold=1, timeout=0.1)  # Very short timeout for testing
        
        def failing_function():
            raise Exception("Fails initially")
        
        # Trigger circuit open
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == "OPEN"
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Should attempt reset
        def succeeding_function():
            return "success"
        
        result = cb.call(succeeding_function)
        assert result == "success"
        assert cb.state == "CLOSED"
    
    def test_error_export_report(self, error_manager, error_context):
        """Test error report export functionality"""
        
        # Generate test error
        test_error = Exception("Test error for report")
        error_manager.handle_error(test_error, error_context)
        
        # Export report
        report = error_manager.export_error_report()
        
        assert 'timestamp' in report
        assert 'statistics' in report
        assert 'circuit_breaker_status' in report
        assert report['statistics']['total_errors'] > 0
        
        # Test with history
        report_with_history = error_manager.export_error_report(include_history=True)
        assert 'error_history' in report_with_history
    
    def test_correlation_id_generation(self, error_manager):
        """Test automatic correlation ID generation"""
        
        context = ErrorContext(
            correlation_id="",  # Empty correlation ID
            module_name="test_module",
            operation_name="test_operation"
        )
        
        test_error = Exception("Test error")
        response = error_manager.handle_error(test_error, context)
        
        # Should have generated a correlation ID
        assert response.correlation_id != ""
        assert len(response.correlation_id) > 0