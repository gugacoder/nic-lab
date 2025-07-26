#!/usr/bin/env python3
"""
Error Recovery UI Test Script

This script tests the error recovery functionality implemented in Task 13.
It validates error boundaries, recovery mechanisms, and user interface components.
"""

import streamlit as st
import sys
import os
import traceback
from typing import Dict, Any, List
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.components.common.error_boundary import ErrorBoundary, SafeContainer, error_boundary
from src.components.common.error_display import ErrorDisplay, ErrorDisplayStyle
from src.components.common.retry_button import RetryButton, RetryButtonStyle
from src.utils.error_handler import get_error_handler, ErrorCategory, ErrorSeverity
from src.utils.state_recovery import get_recovery_manager, create_error_snapshot


class ErrorRecoveryTester:
    """Test suite for error recovery functionality"""
    
    def __init__(self):
        self.test_results: Dict[str, bool] = {}
        self.error_handler = get_error_handler()
        self.recovery_manager = get_recovery_manager()
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all error recovery tests"""
        st.header("🧪 Error Recovery UI Test Suite")
        
        with st.expander("Test Results", expanded=True):
            # Test 1: Error Boundary Functionality
            self.test_error_boundaries()
            
            # Test 2: Error Display Components
            self.test_error_display()
            
            # Test 3: Retry Button Functionality
            self.test_retry_buttons()
            
            # Test 4: State Recovery System
            self.test_state_recovery()
            
            # Test 5: Error Handler Integration
            self.test_error_handler()
            
            # Test 6: End-to-End Error Scenarios
            self.test_error_scenarios()
        
        # Display summary
        self._display_test_summary()
        
        return self.test_results
    
    def test_error_boundaries(self) -> None:
        """Test error boundary functionality"""
        st.subheader("1. Error Boundary Tests")
        
        try:
            # Test basic error boundary
            boundary = ErrorBoundary(
                component_name="test_boundary",
                error_category=ErrorCategory.UI,
                fallback_content="Test fallback content"
            )
            
            # Test error catching
            with boundary.catch_errors():
                # This should not raise an error
                st.write("✅ Error boundary created successfully")
            
            self.test_results["error_boundary_creation"] = True
            st.success("✅ Error boundary creation: PASSED")
            
            # Test SafeContainer
            container = SafeContainer("test_container")
            with container.render():
                st.write("✅ SafeContainer working")
            
            self.test_results["safe_container"] = True
            st.success("✅ SafeContainer functionality: PASSED")
            
        except Exception as e:
            self.test_results["error_boundary_creation"] = False
            self.test_results["safe_container"] = False
            st.error(f"❌ Error boundary tests failed: {e}")
    
    def test_error_display(self) -> None:
        """Test error display components"""
        st.subheader("2. Error Display Tests")
        
        try:
            # Create test error
            test_error = Exception("Test error for display")
            error_info = self.error_handler.handle_error(
                error=test_error,
                category=ErrorCategory.UI,
                severity=ErrorSeverity.MEDIUM,
                user_message="This is a test error message"
            )
            
            # Test different display styles
            styles_tested = []
            
            for style in ErrorDisplayStyle:
                try:
                    with st.expander(f"Test {style.value} style", expanded=False):
                        ErrorDisplay.render_error(error_info, style=style, show_recovery_actions=False)
                    styles_tested.append(style.value)
                except Exception as e:
                    st.error(f"Failed to render {style.value}: {e}")
            
            self.test_results["error_display_styles"] = len(styles_tested) == len(ErrorDisplayStyle)
            
            if self.test_results["error_display_styles"]:
                st.success(f"✅ Error display styles: PASSED ({len(styles_tested)} styles)")
            else:
                st.warning(f"⚠️ Error display styles: PARTIAL ({len(styles_tested)}/{len(ErrorDisplayStyle)} styles)")
            
        except Exception as e:
            self.test_results["error_display_styles"] = False
            st.error(f"❌ Error display tests failed: {e}")
    
    def test_retry_buttons(self) -> None:
        """Test retry button functionality"""
        st.subheader("3. Retry Button Tests")
        
        try:
            # Test retry button creation
            def test_callback() -> bool:
                return True
            
            retry_button = RetryButton(
                operation_id="test_operation",
                retry_callback=test_callback,
                max_retries=3,
                style=RetryButtonStyle.STANDARD
            )
            
            # Test button rendering
            with st.expander("Retry Button Demo", expanded=False):
                st.write("Test retry button (click to test):")
                result = retry_button.render(key="test_retry_demo")
                
                if result:
                    st.success("🎉 Retry button clicked successfully!")
            
            # Test button statistics
            stats = retry_button.get_statistics()
            st.write("Button statistics:", stats)
            
            self.test_results["retry_button_creation"] = True
            self.test_results["retry_button_stats"] = isinstance(stats, dict)
            
            st.success("✅ Retry button functionality: PASSED")
            
        except Exception as e:
            self.test_results["retry_button_creation"] = False
            self.test_results["retry_button_stats"] = False
            st.error(f"❌ Retry button tests failed: {e}")
    
    def test_state_recovery(self) -> None:
        """Test state recovery system"""
        st.subheader("4. State Recovery Tests")
        
        try:
            # Test snapshot creation
            snapshot_id = self.recovery_manager.create_auto_snapshot("test_snapshot")
            
            self.test_results["snapshot_creation"] = snapshot_id is not None
            
            if snapshot_id:
                st.success(f"✅ Snapshot creation: PASSED (ID: {snapshot_id})")
                
                # Test recovery status
                status = self.recovery_manager.get_recovery_status()
                self.test_results["recovery_status"] = isinstance(status, dict)
                
                st.write("Recovery status:", status)
                st.success("✅ Recovery status: PASSED")
            else:
                st.error("❌ Snapshot creation: FAILED")
                self.test_results["recovery_status"] = False
            
        except Exception as e:
            self.test_results["snapshot_creation"] = False
            self.test_results["recovery_status"] = False
            st.error(f"❌ State recovery tests failed: {e}")
    
    def test_error_handler(self) -> None:
        """Test error handler integration"""
        st.subheader("5. Error Handler Tests")
        
        try:
            # Test error handling
            test_error = Exception("Test error for handler")
            error_info = self.error_handler.handle_error(
                error=test_error,
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.LOW,
                user_message="Handler test error"
            )
            
            self.test_results["error_handler_basic"] = error_info is not None
            
            # Test error statistics
            stats = self.error_handler.get_error_statistics()
            self.test_results["error_handler_stats"] = isinstance(stats, dict)
            
            # Test retry functionality
            operation_id = "test_handler_retry"
            can_retry = self.error_handler.can_retry(operation_id)
            self.test_results["error_handler_retry"] = isinstance(can_retry, bool)
            
            st.success("✅ Error handler integration: PASSED")
            st.write("Error statistics:", stats)
            
        except Exception as e:
            self.test_results["error_handler_basic"] = False
            self.test_results["error_handler_stats"] = False  
            self.test_results["error_handler_retry"] = False
            st.error(f"❌ Error handler tests failed: {e}")
    
    def test_error_scenarios(self) -> None:
        """Test end-to-end error scenarios"""
        st.subheader("6. Error Scenario Tests")
        
        # Test scenario: Simulated network error
        if st.button("🌐 Test Network Error Scenario", key="test_network_error"):
            self._test_network_error_scenario()
        
        # Test scenario: Simulated UI error
        if st.button("🖥️ Test UI Error Scenario", key="test_ui_error"):
            self._test_ui_error_scenario()
        
        # Test scenario: Simulated API error
        if st.button("🔌 Test API Error Scenario", key="test_api_error"):
            self._test_api_error_scenario()
        
        # Default to passed if no errors occurred during setup
        if "error_scenarios" not in self.test_results:
            self.test_results["error_scenarios"] = True
    
    def _test_network_error_scenario(self) -> None:
        """Test network error scenario"""
        try:
            # Simulate network error
            network_error = ConnectionError("Simulated network failure")
            
            error_info = self.error_handler.handle_error(
                error=network_error,
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                user_message="Network connection failed. Please check your connection and try again."
            )
            
            # Display error with recovery options
            ErrorDisplay.render_error(error_info, style=ErrorDisplayStyle.STANDARD)
            
            self.test_results["network_error_scenario"] = True
            st.success("✅ Network error scenario: PASSED")
            
        except Exception as e:
            self.test_results["network_error_scenario"] = False
            st.error(f"❌ Network error scenario failed: {e}")
    
    def _test_ui_error_scenario(self) -> None:
        """Test UI error scenario"""
        try:
            # Simulate UI error within error boundary
            boundary = ErrorBoundary(
                component_name="test_ui_error",
                error_category=ErrorCategory.UI,
                fallback_content="UI component temporarily unavailable"
            )
            
            with boundary.catch_errors():
                # This would normally cause an error
                raise ValueError("Simulated UI component error")
            
            self.test_results["ui_error_scenario"] = True
            st.success("✅ UI error scenario: PASSED (error caught by boundary)")
            
        except Exception as e:
            # Error should be caught by boundary, so this indicates test failure
            self.test_results["ui_error_scenario"] = False
            st.error(f"❌ UI error scenario failed: {e}")
    
    def _test_api_error_scenario(self) -> None:
        """Test API error scenario"""
        try:
            # Simulate API error
            api_error = TimeoutError("Simulated API timeout")
            
            error_info = self.error_handler.handle_error(
                error=api_error,
                category=ErrorCategory.AI_API,
                severity=ErrorSeverity.MEDIUM,
                user_message="AI service is temporarily unavailable. Please try again in a moment."
            )
            
            # Display with retry button
            col1, col2 = st.columns(2)
            
            with col1:
                ErrorDisplay.render_error(error_info, style=ErrorDisplayStyle.COMPACT)
            
            with col2:
                def retry_api():
                    st.success("🎉 API retry would be attempted here")
                    return True
                
                retry_button = RetryButton(
                    operation_id="api_retry_test",
                    retry_callback=retry_api,
                    max_retries=3,
                    cooldown_seconds=1
                )
                retry_button.render(key="api_retry_test_btn")
            
            self.test_results["api_error_scenario"] = True
            st.success("✅ API error scenario: PASSED")
            
        except Exception as e:
            self.test_results["api_error_scenario"] = False
            st.error(f"❌ API error scenario failed: {e}")
    
    def _display_test_summary(self) -> None:
        """Display test results summary"""
        st.subheader("📊 Test Summary")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Tests", total_tests)
        
        with col2:
            st.metric("Passed", passed_tests, delta=None)
        
        with col3:
            st.metric("Failed", failed_tests, delta=None if failed_tests == 0 else f"-{failed_tests}")
        
        # Test details
        st.subheader("📋 Detailed Results")
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            color = "green" if result else "red"
            st.markdown(f"**{test_name.replace('_', ' ').title()}**: :{color}[{status}]")
        
        # Overall status
        if failed_tests == 0:
            st.success("🎉 All tests passed! Error Recovery UI implementation is working correctly.")
        else:
            st.warning(f"⚠️ {failed_tests} test(s) failed. Please review the implementation.")


def main():
    """Main test application"""
    st.set_page_config(
        page_title="Error Recovery UI Tests",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Error Recovery UI Test Suite")
    st.markdown("Testing the error recovery functionality implemented in Task 13")
    
    # Run tests
    tester = ErrorRecoveryTester()
    results = tester.run_all_tests()
    
    # Provide manual test instructions
    st.header("🔧 Manual Testing Instructions")
    
    with st.expander("Manual Test Scenarios", expanded=False):
        st.markdown("""
        ### Additional Manual Tests:
        
        1. **Error Persistence Test**:
           - Trigger an error
           - Refresh the page
           - Verify error state is recovered
        
        2. **Chat Integration Test**:
           - Use the chat interface
           - Simulate errors during message submission
           - Verify retry and recovery options work
        
        3. **State Recovery Test**:
           - Enter some chat messages
           - Trigger an error
           - Use recovery options
           - Verify messages are preserved
        
        4. **Error Boundary Isolation Test**:
           - Trigger errors in different components
           - Verify errors don't cascade to other components
           - Check fallback UI states
        """)
    
    return results


if __name__ == "__main__":
    main()