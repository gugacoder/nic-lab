"""
Groq API Client Implementation

Provides high-performance LLM inference through Groq's optimized infrastructure
with comprehensive rate limiting, streaming, caching, and error handling.
"""

import asyncio
import time
import json
import hashlib
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from src.config.settings import get_settings
    SETTINGS_AVAILABLE = True
except (ImportError, Exception):
    SETTINGS_AVAILABLE = False
    get_settings = None

logger = logging.getLogger(__name__)


@dataclass
class GroqResponse:
    """Structured response from Groq API"""
    content: str
    model: str
    usage: Dict[str, int]
    response_time: float
    cached: bool = False
    stream: bool = False


@dataclass
class StreamChunk:
    """Individual chunk from streaming response"""
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class RateLimiter:
    """Token bucket rate limiter for API requests"""
    
    def __init__(self, requests_per_minute: int, burst_capacity: int = None):
        self.requests_per_minute = requests_per_minute
        self.burst_capacity = burst_capacity or requests_per_minute * 2
        self.tokens = self.burst_capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire permission to make a request"""
        async with self._lock:
            now = time.time()
            # Refill tokens based on time passed
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
            self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            # Calculate wait time for next token
            wait_time = (1 - self.tokens) * (60.0 / self.requests_per_minute)
            logger.debug(f"Rate limit hit, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            self.tokens = 0
            return True


class ResponseCache:
    """LRU cache for API responses"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple] = {}
        self.access_times = deque()
        self._lock = asyncio.Lock()
    
    def _generate_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from request parameters"""
        cache_data = {"prompt": prompt, **kwargs}
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[GroqResponse]:
        """Get cached response if valid"""
        async with self._lock:
            if key not in self.cache:
                return None
            
            response, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                return None
            
            # Update access time
            self.access_times.append((key, time.time()))
            response.cached = True
            return response
    
    async def set(self, key: str, response: GroqResponse):
        """Cache response with LRU eviction"""
        async with self._lock:
            # Evict expired entries
            current_time = time.time()
            while self.access_times and current_time - self.access_times[0][1] > self.ttl_seconds:
                old_key, _ = self.access_times.popleft()
                self.cache.pop(old_key, None)
            
            # Evict LRU if at capacity
            while len(self.cache) >= self.max_size and self.access_times:
                old_key, _ = self.access_times.popleft()
                self.cache.pop(old_key, None)
            
            self.cache[key] = (response, current_time)
            self.access_times.append((key, current_time))


class UsageTracker:
    """Track token usage and costs"""
    
    def __init__(self):
        self.usage_stats = defaultdict(lambda: {
            'requests': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'cost_estimate': 0.0
        })
        self._lock = asyncio.Lock()
    
    async def record_usage(self, model: str, usage: Dict[str, int]):
        """Record usage statistics"""
        async with self._lock:
            stats = self.usage_stats[model]
            stats['requests'] += 1
            stats['prompt_tokens'] += usage.get('prompt_tokens', 0)
            stats['completion_tokens'] += usage.get('completion_tokens', 0)
            stats['total_tokens'] += usage.get('total_tokens', 0)
            
            # Estimate cost (approximate rates for Llama-3.1)
            prompt_cost = usage.get('prompt_tokens', 0) * 0.00000059  # $0.59 per 1M tokens
            completion_cost = usage.get('completion_tokens', 0) * 0.00000079  # $0.79 per 1M tokens
            stats['cost_estimate'] += prompt_cost + completion_cost
    
    async def get_stats(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics"""
        async with self._lock:
            if model:
                return dict(self.usage_stats[model])
            return dict(self.usage_stats)


class GroqAPIError(Exception):
    """Base exception for Groq API errors"""
    pass


class RateLimitError(GroqAPIError):
    """Raised when API rate limit is exceeded"""
    pass


class AuthenticationError(GroqAPIError):
    """Raised when API authentication fails"""
    pass


class GroqClient:
    """
    High-performance Groq API client with comprehensive features
    
    Features:
    - Async/sync streaming and non-streaming responses
    - Intelligent rate limiting with exponential backoff
    - Response caching with LRU eviction
    - Token usage tracking and cost estimation
    - Comprehensive error handling and retries
    - Connection pooling and timeout management
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        if SETTINGS_AVAILABLE and get_settings:
            try:
                self.settings = get_settings().groq
                self.api_key = api_key or self.settings.api_key
            except Exception:
                # Fallback to manual configuration
                self._init_fallback_settings(api_key, **kwargs)
        else:
            self._init_fallback_settings(api_key, **kwargs)
        
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        # Initialize components
        self.rate_limiter = RateLimiter(
            getattr(self.settings, 'requests_per_minute', 30),
            kwargs.get('burst_capacity', getattr(self.settings, 'requests_per_minute', 30) * 2)
        )
        self.cache = ResponseCache(
            kwargs.get('cache_size', 1000),
            kwargs.get('cache_ttl', 3600)
        )
        self.usage_tracker = UsageTracker()
        
        # Configure HTTP client
        self.client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(getattr(self.settings, 'timeout', 30)),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        logger.info(f"Initialized Groq client with model: {getattr(self.settings, 'model', 'llama-3.1-8b-instant')}")
    
    def _init_fallback_settings(self, api_key: Optional[str] = None, **kwargs):
        """Initialize with fallback settings when main settings unavailable"""
        from dataclasses import dataclass
        
        @dataclass
        class FallbackSettings:
            api_key: Optional[str] = None
            model: str = "llama-3.1-8b-instant"
            max_tokens: int = 4096
            temperature: float = 0.7
            timeout: int = 30
            requests_per_minute: int = 30
        
        self.settings = FallbackSettings(
            api_key=api_key,
            model=kwargs.get('model', "llama-3.1-70b-versatile"),
            max_tokens=kwargs.get('max_tokens', 4096),
            temperature=kwargs.get('temperature', 0.7),
            timeout=kwargs.get('timeout', 30),
            requests_per_minute=kwargs.get('requests_per_minute', 30)
        )
        self.api_key = api_key or self.settings.api_key
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close HTTP client and cleanup resources"""
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, RateLimitError))
    )
    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> httpx.Response:
        """Make HTTP request with retries and rate limiting"""
        await self.rate_limiter.acquire()
        
        try:
            response = await self.client.post(endpoint, json=data)
            
            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code >= 400:
                error_detail = response.text
                raise GroqAPIError(f"API error {response.status_code}: {error_detail}")
            
            return response
            
        except httpx.RequestError as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        cache: bool = True,
        **kwargs
    ) -> Union[GroqResponse, AsyncGenerator[StreamChunk, None]]:
        """
        Generate completion from Groq API
        
        Args:
            prompt: Input text prompt
            model: Model to use (defaults to configured model)
            max_tokens: Maximum tokens to generate
            temperature: Response creativity (0.0 to 2.0)
            stream: Whether to stream response
            cache: Whether to use response caching
            **kwargs: Additional API parameters
        
        Returns:
            GroqResponse for non-streaming, AsyncGenerator for streaming
        """
        start_time = time.time()
        
        # Prepare request
        data = {
            "model": model or getattr(self.settings, 'model', 'llama-3.1-8b-instant'),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or getattr(self.settings, 'max_tokens', 4096),
            "temperature": temperature or getattr(self.settings, 'temperature', 0.7),
            "stream": stream,
            **kwargs
        }
        
        # Check cache for non-streaming requests
        cache_key = None
        if not stream and cache:
            cache_key = self.cache._generate_key(prompt, **data)
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                logger.debug("Returning cached response")
                return cached_response
        
        # Make API request
        if stream:
            return self._stream_completion(data, start_time)
        else:
            return await self._complete_non_streaming(data, start_time, cache_key)
    
    async def _complete_non_streaming(
        self, 
        data: Dict[str, Any], 
        start_time: float, 
        cache_key: Optional[str]
    ) -> GroqResponse:
        """Handle non-streaming completion"""
        response = await self._make_request("/chat/completions", data)
        result = response.json()
        
        # Extract response data
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        response_time = time.time() - start_time
        
        # Create response object
        groq_response = GroqResponse(
            content=content,
            model=data["model"],
            usage=usage,
            response_time=response_time,
            stream=False
        )
        
        # Cache response
        if cache_key:
            await self.cache.set(cache_key, groq_response)
        
        # Track usage
        await self.usage_tracker.record_usage(data["model"], usage)
        
        logger.info(f"Completed request in {response_time:.2f}s, tokens: {usage.get('total_tokens', 0)}")
        return groq_response
    
    async def _stream_completion(self, data: Dict[str, Any], start_time: float) -> AsyncGenerator[StreamChunk, None]:
        """Handle streaming completion"""
        response = await self._make_request("/chat/completions", data)
        
        content_buffer = ""
        usage = None
        first_token_time = None
        
        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            
            line_data = line[6:]  # Remove "data: " prefix
            if line_data == "[DONE]":
                break
            
            try:
                chunk_data = json.loads(line_data)
                choice = chunk_data["choices"][0]
                delta = choice.get("delta", {})
                
                if "content" in delta:
                    content = delta["content"]
                    content_buffer += content
                    
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = first_token_time - start_time
                        logger.debug(f"Time to first token: {ttft:.3f}s")
                    
                    yield StreamChunk(content=content)
                
                if choice.get("finish_reason"):
                    usage = chunk_data.get("usage", {})
                    yield StreamChunk(
                        content="",
                        finish_reason=choice["finish_reason"],
                        usage=usage
                    )
            
            except json.JSONDecodeError:
                continue
        
        # Track usage
        if usage:
            await self.usage_tracker.record_usage(data["model"], usage)
        
        total_time = time.time() - start_time
        logger.info(f"Streamed {len(content_buffer)} chars in {total_time:.2f}s")
    
    async def test_connection(self) -> bool:
        """Test API connection and authentication"""
        try:
            response = await self.complete(
                "Hello, how are you?",
                max_tokens=10,
                cache=False
            )
            logger.info(f"Connection test successful: {response.content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def get_usage_stats(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics"""
        return await self.usage_tracker.get_stats(model)
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        start_time = time.time()
        
        try:
            # Test basic completion
            response = await self.complete(
                "Test prompt",
                max_tokens=5,
                cache=False
            )
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "model": response.model,
                "tokens_used": response.usage.get("total_tokens", 0),
                "cache_size": len(self.cache.cache),
                "api_accessible": True
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time": time.time() - start_time,
                "api_accessible": False
            }


# Convenience functions
_client_instance: Optional[GroqClient] = None

async def get_client() -> GroqClient:
    """Get shared Groq client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = GroqClient()
    return _client_instance

async def close_client():
    """Close shared client"""
    global _client_instance
    if _client_instance:
        await _client_instance.close()
        _client_instance = None


# CLI test functions
async def test_connection():
    """Test connection via CLI"""
    async with GroqClient() as client:
        success = await client.test_connection()
        print(f"Connection test: {'PASSED' if success else 'FAILED'}")

async def test_stream(prompt: str = "Hello, how are you?"):
    """Test streaming via CLI"""
    async with GroqClient() as client:
        print(f"Streaming response to: {prompt}\n")
        async for chunk in await client.complete(prompt, stream=True):
            if chunk.content:
                print(chunk.content, end='', flush=True)
            elif chunk.finish_reason:
                print(f"\n\nFinished: {chunk.finish_reason}")
                if chunk.usage:
                    print(f"Tokens used: {chunk.usage.get('total_tokens', 0)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test-connection":
            asyncio.run(test_connection())
        elif command == "test-stream":
            prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, how are you?"
            asyncio.run(test_stream(prompt))
        else:
            print("Available commands: test-connection, test-stream")
    else:
        print("Usage: python -m src.ai.groq_client [test-connection|test-stream] [prompt]")