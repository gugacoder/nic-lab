"""
Advanced Streaming Response Handler

Provides sophisticated streaming capabilities for real-time AI response display
with buffering, error recovery, and performance optimization.
"""

import asyncio
import time
import json
import logging
from typing import AsyncGenerator, Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Streaming state enumeration"""
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    BUFFERING = "buffering"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamMetrics:
    """Streaming performance metrics"""
    start_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    total_tokens: int = 0
    total_chars: int = 0
    chunks_received: int = 0
    bytes_received: int = 0
    average_token_time: float = 0.0
    tokens_per_second: float = 0.0
    
    def update_token_received(self, content: str):
        """Update metrics when token is received"""
        current_time = time.time()
        
        if self.first_token_time is None:
            self.first_token_time = current_time
        
        self.last_token_time = current_time
        self.total_tokens += 1
        self.total_chars += len(content)
        self.chunks_received += 1
        self.bytes_received += len(content.encode('utf-8'))
        
        # Calculate rates
        if self.first_token_time:
            elapsed = current_time - self.first_token_time
            if elapsed > 0:
                self.tokens_per_second = self.total_tokens / elapsed
                self.average_token_time = elapsed / self.total_tokens
    
    def get_time_to_first_token(self) -> Optional[float]:
        """Get time to first token in seconds"""
        if self.first_token_time:
            return self.first_token_time - self.start_time
        return None
    
    def get_total_time(self) -> float:
        """Get total streaming time"""
        end_time = self.last_token_time or time.time()
        return end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "time_to_first_token": self.get_time_to_first_token(),
            "total_time": self.get_total_time(),
            "total_tokens": self.total_tokens,
            "total_chars": self.total_chars,
            "chunks_received": self.chunks_received,
            "bytes_received": self.bytes_received,
            "tokens_per_second": self.tokens_per_second,
            "average_token_time": self.average_token_time
        }


@dataclass
class StreamChunk:
    """Enhanced stream chunk with metadata"""
    content: str
    timestamp: float = field(default_factory=time.time)
    chunk_id: Optional[int] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_final(self) -> bool:
        """Check if this is the final chunk"""
        return self.finish_reason is not None
    
    @property
    def token_count(self) -> int:
        """Estimate token count (rough approximation)"""
        return max(1, len(self.content.split()))


class StreamBuffer:
    """Intelligent buffering for smooth streaming display"""
    
    def __init__(
        self,
        buffer_size: int = 10,
        min_buffer_time: float = 0.01,
        max_buffer_time: float = 0.1
    ):
        self.buffer_size = buffer_size
        self.min_buffer_time = min_buffer_time
        self.max_buffer_time = max_buffer_time
        self.buffer: deque = deque(maxlen=buffer_size)
        self.last_flush = time.time()
        self._lock = asyncio.Lock()
    
    async def add_chunk(self, chunk: StreamChunk) -> List[StreamChunk]:
        """Add chunk to buffer and return chunks ready for display"""
        async with self._lock:
            self.buffer.append(chunk)
            
            current_time = time.time()
            time_since_flush = current_time - self.last_flush
            
            # Flush if buffer is full, enough time has passed, or final chunk
            should_flush = (
                len(self.buffer) >= self.buffer_size or
                time_since_flush >= self.max_buffer_time or
                chunk.is_final or
                (time_since_flush >= self.min_buffer_time and len(self.buffer) > 0)
            )
            
            if should_flush:
                chunks = list(self.buffer)
                self.buffer.clear()
                self.last_flush = current_time
                return chunks
            
            return []
    
    async def flush(self) -> List[StreamChunk]:
        """Force flush all buffered chunks"""
        async with self._lock:
            chunks = list(self.buffer)
            self.buffer.clear()
            self.last_flush = time.time()
            return chunks


class StreamProcessor:
    """Advanced stream processing with error recovery and optimization"""
    
    def __init__(
        self,
        chunk_callback: Optional[Callable[[StreamChunk], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
        completion_callback: Optional[Callable[[StreamMetrics], None]] = None,
        buffer_size: int = 10,
        enable_buffering: bool = True
    ):
        self.chunk_callback = chunk_callback
        self.error_callback = error_callback
        self.completion_callback = completion_callback
        self.enable_buffering = enable_buffering
        
        self.state = StreamState.IDLE
        self.metrics = StreamMetrics()
        self.buffer = StreamBuffer(buffer_size) if enable_buffering else None
        self.accumulated_content = ""
        self.chunk_counter = 0
        self._cancellation_token = False
        
        logger.debug(f"Initialized StreamProcessor with buffering={'enabled' if enable_buffering else 'disabled'}")
    
    async def process_stream(
        self, 
        stream_generator: AsyncGenerator[StreamChunk, None]
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Process streaming response with advanced features
        
        Args:
            stream_generator: Source stream of chunks
            
        Yields:
            StreamChunk: Processed chunks ready for display
        """
        self.state = StreamState.CONNECTING
        self._cancellation_token = False
        
        try:
            self.state = StreamState.STREAMING
            
            async for raw_chunk in stream_generator:
                if self._cancellation_token:
                    self.state = StreamState.CANCELLED
                    break
                
                # Process chunk
                processed_chunk = await self._process_chunk(raw_chunk)
                
                if self.enable_buffering and self.buffer:
                    # Use buffering for smooth display
                    ready_chunks = await self.buffer.add_chunk(processed_chunk)
                    for chunk in ready_chunks:
                        yield chunk
                        if self.chunk_callback:
                            self.chunk_callback(chunk)
                else:
                    # Direct streaming
                    yield processed_chunk
                    if self.chunk_callback:
                        self.chunk_callback(processed_chunk)
                
                # Check for completion
                if processed_chunk.is_final:
                    break
            
            # Flush any remaining buffered chunks
            if self.enable_buffering and self.buffer:
                remaining_chunks = await self.buffer.flush()
                for chunk in remaining_chunks:
                    yield chunk
                    if self.chunk_callback:
                        self.chunk_callback(chunk)
            
            self.state = StreamState.COMPLETED
            
            # Call completion callback
            if self.completion_callback:
                self.completion_callback(self.metrics)
            
        except Exception as e:
            self.state = StreamState.ERROR
            logger.error(f"Stream processing error: {e}")
            
            if self.error_callback:
                self.error_callback(e)
            else:
                raise
    
    async def _process_chunk(self, chunk: StreamChunk) -> StreamChunk:
        """Process individual chunk with enhancements"""
        self.chunk_counter += 1
        
        # Update metrics
        if chunk.content:
            self.metrics.update_token_received(chunk.content)
            self.accumulated_content += chunk.content
        
        # Enhance chunk with metadata
        enhanced_chunk = StreamChunk(
            content=chunk.content,
            timestamp=chunk.timestamp,
            chunk_id=self.chunk_counter,
            finish_reason=chunk.finish_reason,
            usage=chunk.usage,
            metadata={
                **chunk.metadata,
                "accumulated_length": len(self.accumulated_content),
                "chunk_number": self.chunk_counter,
                "tokens_per_second": self.metrics.tokens_per_second
            }
        )
        
        return enhanced_chunk
    
    def cancel(self):
        """Cancel streaming operation"""
        self._cancellation_token = True
        self.state = StreamState.CANCELLED
        logger.info("Stream processing cancelled")
    
    def get_metrics(self) -> StreamMetrics:
        """Get current streaming metrics"""
        return self.metrics
    
    def get_accumulated_content(self) -> str:
        """Get all accumulated content"""
        return self.accumulated_content


class StreamManager:
    """High-level stream management with multiple concurrent streams"""
    
    def __init__(self):
        self.active_streams: Dict[str, StreamProcessor] = {}
        self._stream_counter = 0
        self._lock = asyncio.Lock()
    
    async def create_stream(
        self,
        stream_id: Optional[str] = None,
        **processor_kwargs
    ) -> str:
        """Create a new stream processor"""
        async with self._lock:
            if stream_id is None:
                self._stream_counter += 1
                stream_id = f"stream_{self._stream_counter}"
            
            if stream_id in self.active_streams:
                raise ValueError(f"Stream {stream_id} already exists")
            
            processor = StreamProcessor(**processor_kwargs)
            self.active_streams[stream_id] = processor
            
            logger.info(f"Created stream {stream_id}")
            return stream_id
    
    async def get_stream(self, stream_id: str) -> Optional[StreamProcessor]:
        """Get stream processor by ID"""
        async with self._lock:
            return self.active_streams.get(stream_id)
    
    async def cancel_stream(self, stream_id: str) -> bool:
        """Cancel a specific stream"""
        async with self._lock:
            if stream_id in self.active_streams:
                self.active_streams[stream_id].cancel()
                return True
            return False
    
    async def cleanup_stream(self, stream_id: str) -> bool:
        """Remove completed stream"""
        async with self._lock:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
                logger.info(f"Cleaned up stream {stream_id}")
                return True
            return False
    
    async def cancel_all_streams(self):
        """Cancel all active streams"""
        async with self._lock:
            for processor in self.active_streams.values():
                processor.cancel()
            logger.info(f"Cancelled {len(self.active_streams)} active streams")
    
    async def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all streams"""
        async with self._lock:
            return {
                stream_id: processor.get_metrics().to_dict()
                for stream_id, processor in self.active_streams.items()
            }


# Context manager for stream processing
@asynccontextmanager
async def stream_processor_context(**kwargs):
    """Context manager for automatic stream cleanup"""
    processor = StreamProcessor(**kwargs)
    try:
        yield processor
    finally:
        processor.cancel()


# Global stream manager
_stream_manager: Optional[StreamManager] = None

async def get_stream_manager() -> StreamManager:
    """Get global stream manager instance"""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager


# Utility functions for common streaming patterns
async def simple_stream_to_console(
    stream_generator: AsyncGenerator[StreamChunk, None],
    show_metrics: bool = False
):
    """Simple utility to stream to console"""
    async with stream_processor_context() as processor:
        async for chunk in processor.process_stream(stream_generator):
            if chunk.content:
                print(chunk.content, end='', flush=True)
            elif chunk.is_final:
                print()  # New line at end
                if show_metrics and chunk.finish_reason:
                    metrics = processor.get_metrics().to_dict()
                    print(f"\nMetrics: {metrics}")


async def stream_to_callback(
    stream_generator: AsyncGenerator[StreamChunk, None],
    callback: Callable[[str], None],
    buffer_size: int = 5
) -> StreamMetrics:
    """Stream to a callback function with buffering"""
    def chunk_callback(chunk: StreamChunk):
        if chunk.content:
            callback(chunk.content)
    
    async with stream_processor_context(
        chunk_callback=chunk_callback,
        buffer_size=buffer_size
    ) as processor:
        async for _ in processor.process_stream(stream_generator):
            pass  # Callback handles the chunks
        
        return processor.get_metrics()


# Performance testing utilities
class StreamBenchmark:
    """Benchmark streaming performance"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    async def benchmark_stream(
        self, 
        stream_generator: AsyncGenerator[StreamChunk, None],
        test_name: str = "unnamed"
    ) -> Dict[str, Any]:
        """Benchmark a stream's performance"""
        start_time = time.time()
        
        async with stream_processor_context() as processor:
            chunk_count = 0
            async for chunk in processor.process_stream(stream_generator):
                chunk_count += 1
            
            metrics = processor.get_metrics().to_dict()
            
            result = {
                "test_name": test_name,
                "benchmark_time": time.time() - start_time,
                "chunk_count": chunk_count,
                **metrics
            }
            
            self.results.append(result)
            return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary"""
        if not self.results:
            return {}
        
        return {
            "total_tests": len(self.results),
            "average_tokens_per_second": sum(r["tokens_per_second"] for r in self.results) / len(self.results),
            "average_ttft": sum(r.get("time_to_first_token", 0) for r in self.results if r.get("time_to_first_token")) / len(self.results),
            "tests": self.results
        }


if __name__ == "__main__":
    # Example usage and testing
    
    async def mock_stream():
        """Mock streaming generator for testing"""
        content_chunks = [
            "Hello", " there", "!", " How", " are", " you", " doing", " today", "?",
            " I", " hope", " everything", " is", " going", " well", "."
        ]
        
        for i, content in enumerate(content_chunks):
            await asyncio.sleep(0.1)  # Simulate network delay
            yield StreamChunk(content=content, chunk_id=i)
        
        # Final chunk
        yield StreamChunk(
            content="", 
            finish_reason="stop",
            usage={"total_tokens": len(content_chunks)}
        )
    
    async def test_streaming():
        print("Testing streaming functionality...")
        
        # Test simple console streaming
        print("\n1. Simple console streaming:")
        await simple_stream_to_console(mock_stream(), show_metrics=True)
        
        # Test callback streaming
        print("\n2. Callback streaming:")
        collected_content = []
        
        def content_callback(content: str):
            collected_content.append(content)
            print(f"[{content}]", end='')
        
        metrics = await stream_to_callback(mock_stream(), content_callback)
        print(f"\nCollected: {''.join(collected_content)}")
        print(f"Metrics: {metrics.to_dict()}")
        
        # Test benchmark
        print("\n3. Performance benchmark:")
        benchmark = StreamBenchmark()
        result = await benchmark.benchmark_stream(mock_stream(), "mock_test")
        print(f"Benchmark result: {result}")
    
    # Run tests
    asyncio.run(test_streaming())