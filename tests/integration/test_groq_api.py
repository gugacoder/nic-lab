"""
Integration Tests for Groq API Client

Comprehensive tests for the Groq API integration including connection,
streaming, rate limiting, error handling, and token management.
"""

import asyncio
import os
import time
from typing import List, Dict, Any
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ai.groq_client import (
    GroqClient, GroqResponse, StreamChunk, RateLimitError,
    AuthenticationError, GroqAPIError
)
from src.ai.token_manager import TokenManager, TokenCount
from src.ai.models import get_model_info, recommend_model


class TestGroqClientIntegration:
    """Integration tests for Groq API client"""
    
    @pytest.fixture
    def token_manager(self):
        """Create token manager instance"""
        return TokenManager()
    
    @pytest.mark.asyncio
    async def test_connection_success(self):
        """Test successful API connection"""
        # Use test API key or mock if not available
        api_key = os.environ.get("GROQ_API_KEY", "test-key")
        
        async with GroqClient(api_key=api_key) as client:
            # If no real API key, mock the response
            if client.api_key == "test-key":
                with patch.object(client, 'complete') as mock_complete:
                    mock_complete.return_value = GroqResponse(
                        content="Hello! I'm working fine.",
                        model="llama-3.1-8b-instant",
                        usage={"total_tokens": 10},
                        response_time=0.5,
                        cached=False
                    )
                    
                    result = await client.test_connection()
                    assert result is True
                    mock_complete.assert_called_once()
            else:
                # Real API test
                result = await client.test_connection()
                assert result is True
    
    @pytest.mark.asyncio
    async def test_basic_completion(self, client):
        """Test basic text completion"""
        if client.api_key == "test-key":
            with patch.object(client, '_make_request') as mock_request:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{
                        "message": {"content": "Paris is the capital of France."}
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18
                    }
                }
                mock_request.return_value = mock_response
                
                response = await client.complete(
                    "What is the capital of France?",
                    max_tokens=50
                )
                
                assert isinstance(response, GroqResponse)
                assert "Paris" in response.content
                assert response.usage["total_tokens"] == 18
        else:
            # Real API test
            response = await client.complete(
                "What is the capital of France?",
                max_tokens=50
            )
            
            assert isinstance(response, GroqResponse)
            assert response.content is not None
            assert response.usage["total_tokens"] > 0
    
    @pytest.mark.asyncio
    async def test_streaming_completion(self, client):
        """Test streaming response functionality"""
        if client.api_key == "test-key":
            async def mock_stream():
                chunks = ["Hello", " there", "!", ""]
                for i, content in enumerate(chunks):
                    if content:
                        yield StreamChunk(content=content)
                    else:
                        yield StreamChunk(
                            content="",
                            finish_reason="stop",
                            usage={"total_tokens": 10}
                        )
            
            with patch.object(client, 'complete') as mock_complete:
                mock_complete.return_value = mock_stream()
                
                chunks = []
                async for chunk in await client.complete("Hello", stream=True):
                    chunks.append(chunk)
                
                assert len(chunks) == 4
                assert chunks[0].content == "Hello"
                assert chunks[-1].finish_reason == "stop"
        else:
            # Real API test
            chunks = []
            async for chunk in await client.complete("Say hello", stream=True, max_tokens=20):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            assert any(chunk.content for chunk in chunks)
            assert chunks[-1].finish_reason is not None
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test rate limit handling"""
        # Configure aggressive rate limiting for test
        client.rate_limiter.requests_per_minute = 2
        client.rate_limiter.tokens = 1  # Start with 1 token
        
        start_time = time.time()
        
        # Make first request (should succeed immediately)
        await client.rate_limiter.acquire()
        
        # Make second request (should succeed immediately)
        await client.rate_limiter.acquire()
        
        # Make third request (should be rate limited)
        await client.rate_limiter.acquire()
        
        elapsed = time.time() - start_time
        
        # Should have waited due to rate limiting
        assert elapsed > 1.0
    
    @pytest.mark.asyncio
    async def test_response_caching(self, client):
        """Test response caching functionality"""
        prompt = "What is 2+2?"
        
        if client.api_key == "test-key":
            with patch.object(client, '_make_request') as mock_request:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "4"}}],
                    "usage": {"total_tokens": 10}
                }
                mock_request.return_value = mock_response
                
                # First request (not cached)
                response1 = await client.complete(prompt, cache=True)
                assert response1.cached is False
                
                # Second request (should be cached)
                response2 = await client.complete(prompt, cache=True)
                assert response2.cached is True
                assert response2.content == response1.content
                
                # Only one actual API call
                mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_auth(self, client):
        """Test authentication error handling"""
        with patch.object(client, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_request.return_value = mock_response
            
            with pytest.raises(AuthenticationError):
                await client.complete("test")
    
    @pytest.mark.asyncio
    async def test_error_handling_rate_limit(self, client):
        """Test rate limit error handling"""
        with patch.object(client, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_request.return_value = mock_response
            
            # Should retry with exponential backoff
            mock_request.side_effect = [
                mock_response,  # First attempt - rate limited
                mock_response,  # Second attempt - rate limited
                MagicMock(status_code=200, json=lambda: {  # Third attempt - success
                    "choices": [{"message": {"content": "Success"}}],
                    "usage": {"total_tokens": 10}
                })
            ]
            
            response = await client.complete("test")
            assert response.content == "Success"
            assert mock_request.call_count == 3
    
    @pytest.mark.asyncio
    async def test_token_counting(self, token_manager):
        """Test token counting accuracy"""
        # Simple text
        text = "Hello, how are you doing today?"
        tokens = token_manager.counter.count_tokens(text, "detailed")
        assert tokens > 0
        assert tokens < 20  # Reasonable range
        
        # Code-like text
        code = "def hello_world():\n    print('Hello, World!')"
        code_tokens = token_manager.counter.count_tokens(code, "detailed")
        assert code_tokens > tokens  # Code typically uses more tokens
        
        # Message format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather like?"}
        ]
        
        token_count = token_manager.counter.count_message_tokens(messages)
        assert token_count.total_tokens > 0
        assert token_count.system_tokens > 0
        assert token_count.input_tokens > 0
    
    @pytest.mark.asyncio
    async def test_token_limit_validation(self, client, token_manager):
        """Test token limit validation"""
        messages = [
            {"role": "user", "content": "Short prompt"}
        ]
        
        can_proceed, message, estimate = await token_manager.validate_request(
            messages, max_tokens=100
        )
        
        assert can_proceed is True
        assert estimate.total_tokens > 0
        assert estimate.output_tokens <= 100
    
    @pytest.mark.asyncio
    async def test_usage_tracking(self, client):
        """Test usage statistics tracking"""
        if client.api_key == "test-key":
            # Manually record some usage
            await client.usage_tracker.record_usage(
                "llama-3.1-70b-versatile",
                {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
            )
            
            stats = await client.get_usage_stats()
            model_stats = stats["llama-3.1-70b-versatile"]
            
            assert model_stats["requests"] == 1
            assert model_stats["total_tokens"] == 150
            assert model_stats["cost_estimate"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        if client.api_key == "test-key":
            with patch.object(client, '_make_request') as mock_request:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Response"}}],
                    "usage": {"total_tokens": 10}
                }
                mock_request.return_value = mock_response
                
                # Make concurrent requests
                tasks = [
                    client.complete(f"Question {i}", cache=False)
                    for i in range(5)
                ]
                
                responses = await asyncio.gather(*tasks)
                
                assert len(responses) == 5
                assert all(isinstance(r, GroqResponse) for r in responses)
    
    @pytest.mark.asyncio
    async def test_model_selection(self):
        """Test model selection and recommendation"""
        # Test default model
        default_model = recommend_model("conversation")
        assert default_model.id == "llama-3.1-70b-versatile"
        
        # Test cost-sensitive selection
        cheap_model = recommend_model("conversation", cost_sensitive=True)
        assert cheap_model.pricing.output_cost < default_model.pricing.output_cost
        
        # Test speed priority
        fast_model = recommend_model("conversation", speed_priority=True)
        assert fast_model.size.value == "small"
        
        # Test reasoning task
        reasoning_model = recommend_model("reasoning")
        assert reasoning_model.id in ["llama-3.1-405b-reasoning", "llama-3.1-70b-versatile"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check functionality"""
        if client.api_key == "test-key":
            with patch.object(client, 'complete') as mock_complete:
                mock_complete.return_value = GroqResponse(
                    content="Test",
                    model="llama-3.1-70b-versatile",
                    usage={"total_tokens": 5},
                    response_time=0.1,
                    cached=False
                )
                
                health = await client.health_check()
                
                assert health["status"] == "healthy"
                assert health["api_accessible"] is True
                assert "response_time" in health
                assert "model" in health
    
    @pytest.mark.asyncio
    async def test_context_optimization(self, token_manager):
        """Test context optimization for token limits"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Question 3"},
        ]
        
        # Optimize for small context window
        optimized = token_manager.optimizer.optimize_context(messages, max_context_tokens=100)
        
        # Should keep system message and most recent messages
        assert any(msg["role"] == "system" for msg in optimized)
        assert optimized[-1]["content"] == "Question 3"
        assert len(optimized) < len(messages)


class TestGroqStressTests:
    """Stress tests for rate limiting and performance"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.environ.get("GROQ_API_KEY", "test-key") == "test-key",
        reason="Requires real API key for stress testing"
    )
    async def test_rate_limit_stress(self):
        """Stress test rate limiting with rapid requests"""
        client = GroqClient()
        
        async def make_request(i):
            try:
                response = await client.complete(
                    f"Count to {i}",
                    max_tokens=10,
                    cache=False
                )
                return True, response.response_time
            except RateLimitError:
                return False, 0
            finally:
                if i == 29:  # Last request
                    await client.close()
        
        # Make 30 rapid requests (rate limit is 30/min)
        start_time = time.time()
        results = await asyncio.gather(*[
            make_request(i) for i in range(30)
        ])
        
        total_time = time.time() - start_time
        successful = sum(1 for success, _ in results if success)
        
        print(f"Stress test: {successful}/30 requests succeeded in {total_time:.2f}s")
        
        # All requests should succeed with rate limiting
        assert successful == 30
        assert total_time < 10  # Should handle burst efficiently


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])