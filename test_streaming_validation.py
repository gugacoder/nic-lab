#!/usr/bin/env python3
"""
Streaming Display Validation Test

Tests the streaming message display functionality per Task 11 requirements.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any

# Test the import structure
try:
    from src.components.chat.streaming_message import (
        EnhancedStreamingMessage,
        get_streaming_manager,
        create_streaming_message
    )
    from src.utils.stream_handler import StreamHandler, StreamConfig
    from src.utils.token_buffer import TokenBuffer, BufferStrategy
    from src.ai.streaming import StreamChunk
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import failed: {e}")
    IMPORTS_OK = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingValidationTest:
    """Validation tests for streaming functionality"""
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
        self.passed = 0
        self.failed = 0
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        if passed:
            self.passed += 1
            print(f"✅ {test_name}: PASSED {details}")
        else:
            self.failed += 1
            print(f"❌ {test_name}: FAILED {details}")
    
    def test_imports(self):
        """Test that all required modules import correctly"""
        self.log_test("Module Imports", IMPORTS_OK, 
                     "All streaming components import successfully")
    
    def test_token_buffer(self):
        """Test token buffer functionality"""
        if not IMPORTS_OK:
            self.log_test("Token Buffer", False, "Imports failed")
            return
        
        try:
            # Test buffer creation
            buffer = TokenBuffer(
                strategy=BufferStrategy.ADAPTIVE,
                max_size=5,
                max_time=0.1
            )
            
            # Test adding tokens
            async def test_buffer_async():
                test_tokens = ["Hello", " ", "world", "!", " ", "How", " are", " you", "?"]
                flushes = []
                
                for token in test_tokens:
                    result = await buffer.add_token(token)
                    if result:
                        flushes.append(result)
                
                # Final flush
                final = await buffer.flush()
                if final:
                    flushes.append(final)
                
                # Verify content reconstruction
                all_content = ""
                for flush_tokens in flushes:
                    for token in flush_tokens:
                        all_content += token.content
                
                expected = "".join(test_tokens)
                return all_content == expected, len(flushes)
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                content_matches, num_flushes = loop.run_until_complete(test_buffer_async())
                self.log_test("Token Buffer", content_matches, 
                             f"Content reconstruction successful, {num_flushes} flushes")
            finally:
                loop.close()
                
        except Exception as e:
            self.log_test("Token Buffer", False, f"Exception: {str(e)}")
    
    def test_stream_chunk_creation(self):
        """Test StreamChunk creation and properties"""
        if not IMPORTS_OK:
            self.log_test("StreamChunk Creation", False, "Imports failed")
            return
        
        try:
            # Test basic chunk
            chunk = StreamChunk(content="Hello world")
            
            # Test properties
            has_content = len(chunk.content) > 0
            has_timestamp = hasattr(chunk, 'timestamp')
            not_final = not chunk.is_final
            
            # Test final chunk
            final_chunk = StreamChunk(content="", finish_reason="stop")
            is_final = final_chunk.is_final
            
            success = has_content and has_timestamp and not_final and is_final
            self.log_test("StreamChunk Creation", success, 
                         "Basic and final chunks created correctly")
            
        except Exception as e:
            self.log_test("StreamChunk Creation", False, f"Exception: {str(e)}")
    
    def test_streaming_message_component(self):
        """Test EnhancedStreamingMessage component"""
        if not IMPORTS_OK:
            self.log_test("Streaming Message Component", False, "Imports failed")
            return
        
        try:
            # Test component creation
            msg_id = f"test_msg_{int(time.time())}"
            
            # Mock Streamlit session state
            import sys
            from unittest.mock import MagicMock
            
            # Mock streamlit
            mock_st = MagicMock()
            mock_st.session_state = {}
            sys.modules['streamlit'] = mock_st
            
            # Create streaming message
            streaming_msg = EnhancedStreamingMessage(
                message_id=msg_id,
                role="assistant",
                buffer_size=5,
                show_progress=False,
                show_metrics=False
            )
            
            # Test state initialization
            has_state = hasattr(streaming_msg, 'state')
            has_id = streaming_msg.message_id == msg_id
            has_role = streaming_msg.role == "assistant"
            
            success = has_state and has_id and has_role
            self.log_test("Streaming Message Component", success,
                         "Component created with correct properties")
            
        except Exception as e:
            self.log_test("Streaming Message Component", False, f"Exception: {str(e)}")
    
    def test_stream_handler_creation(self):
        """Test StreamHandler creation"""
        if not IMPORTS_OK:
            self.log_test("Stream Handler", False, "Imports failed")
            return
        
        try:
            # Test handler creation
            config = StreamConfig(
                buffer_size=10,
                show_progress=True,
                show_metrics=False
            )
            
            handler = StreamHandler(config)
            
            # Test properties
            has_config = handler.config is not None
            has_streams = hasattr(handler, 'active_streams')
            correct_buffer_size = handler.config.buffer_size == 10
            
            success = has_config and has_streams and correct_buffer_size
            self.log_test("Stream Handler", success,
                         "Handler created with correct configuration")
            
        except Exception as e:
            self.log_test("Stream Handler", False, f"Exception: {str(e)}")
    
    def test_performance_characteristics(self):
        """Test performance characteristics of buffering"""
        if not IMPORTS_OK:
            self.log_test("Performance Test", False, "Imports failed")
            return
        
        try:
            # Test buffer performance with many tokens
            buffer = TokenBuffer(
                strategy=BufferStrategy.TIME_BASED,
                max_size=50,
                max_time=0.01  # 10ms
            )
            
            async def performance_test():
                start_time = time.time()
                
                # Generate many small tokens
                tokens = [f"token_{i} " for i in range(100)]
                
                flush_count = 0
                for token in tokens:
                    result = await buffer.add_token(token)
                    if result:
                        flush_count += 1
                
                # Final flush
                final = await buffer.flush()
                if final:
                    flush_count += 1
                
                end_time = time.time()
                duration = end_time - start_time
                
                return duration, flush_count
            
            # Run performance test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                duration, flush_count = loop.run_until_complete(performance_test())
                
                # Performance criteria
                fast_enough = duration < 1.0  # Less than 1 second for 100 tokens
                reasonable_flushes = flush_count > 0 and flush_count < 50  # Not too many flushes
                
                success = fast_enough and reasonable_flushes
                self.log_test("Performance Test", success,
                             f"Duration: {duration:.3f}s, Flushes: {flush_count}")
            finally:
                loop.close()
                
        except Exception as e:
            self.log_test("Performance Test", False, f"Exception: {str(e)}")
    
    def test_error_handling(self):
        """Test error handling in streaming components"""
        if not IMPORTS_OK:
            self.log_test("Error Handling", False, "Imports failed")
            return
        
        try:
            # Test buffer with invalid parameters
            buffer_created = True
            try:
                TokenBuffer(max_size=-1)  # Invalid size
            except:
                buffer_created = False
            
            # Test graceful handling of empty content
            buffer = TokenBuffer()
            
            async def error_test():
                # Test empty content
                result1 = await buffer.add_token("")
                
                # Test None content (should handle gracefully)
                try:
                    result2 = await buffer.add_token(None)
                    handled_none = False
                except:
                    handled_none = True
                
                return result1, handled_none
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result1, handled_none = loop.run_until_complete(error_test())
                
                # Error handling criteria
                handles_empty = result1 is None  # Empty content should not trigger flush
                
                success = not buffer_created or (handles_empty and handled_none)
                self.log_test("Error Handling", success,
                             "Components handle edge cases gracefully")
            finally:
                loop.close()
                
        except Exception as e:
            self.log_test("Error Handling", False, f"Exception: {str(e)}")
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("🚀 Starting Streaming Display Validation Tests")
        print("=" * 60)
        
        # Run tests
        self.test_imports()
        self.test_token_buffer()
        self.test_stream_chunk_creation()
        self.test_streaming_message_component()
        self.test_stream_handler_creation()
        self.test_performance_characteristics()
        self.test_error_handling()
        
        # Summary
        print("=" * 60)
        print(f"📊 Test Results: {self.passed} passed, {self.failed} failed")
        
        if self.failed == 0:
            print("🎉 All tests passed! Streaming display implementation is valid.")
            return True
        else:
            print(f"⚠️  {self.failed} tests failed. Review implementation.")
            return False
    
    def get_detailed_results(self):
        """Get detailed test results"""
        return {
            "summary": {
                "total_tests": len(self.test_results),
                "passed": self.passed,
                "failed": self.failed,
                "success_rate": self.passed / len(self.test_results) if self.test_results else 0
            },
            "details": self.test_results
        }


if __name__ == "__main__":
    print("Task 11 - Streaming Display Validation")
    print("Testing streaming message display implementation...")
    
    validator = StreamingValidationTest()
    success = validator.run_all_tests()
    
    # Print detailed results
    results = validator.get_detailed_results()
    print(f"\n📋 Detailed Results:")
    print(f"Success Rate: {results['summary']['success_rate']:.1%}")
    
    # Exit with appropriate code
    exit(0 if success else 1)