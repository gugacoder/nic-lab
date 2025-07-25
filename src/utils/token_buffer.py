"""
Token Buffer Management

Provides intelligent token buffering for smooth streaming display,
optimizing the balance between responsiveness and performance.
"""

import time
import asyncio
from typing import List, Optional, Callable, Dict, Any, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BufferStrategy(Enum):
    """Token buffer strategies"""
    TIME_BASED = "time_based"      # Buffer based on time intervals
    SIZE_BASED = "size_based"      # Buffer based on token count
    ADAPTIVE = "adaptive"          # Dynamically adjust based on stream speed
    IMMEDIATE = "immediate"        # No buffering, immediate display


@dataclass
class TokenChunk:
    """Individual token chunk with metadata"""
    content: str
    timestamp: float = field(default_factory=time.time)
    token_id: Optional[int] = None
    char_count: int = field(init=False)
    word_count: int = field(init=False)
    
    def __post_init__(self):
        self.char_count = len(self.content)
        self.word_count = len(self.content.split()) if self.content.strip() else 0
    
    @property
    def is_whitespace(self) -> bool:
        """Check if chunk is only whitespace"""
        return self.content.isspace()
    
    @property
    def is_punctuation(self) -> bool:
        """Check if chunk is only punctuation"""
        return all(c in '.,!?;:' for c in self.content.strip())


@dataclass
class BufferMetrics:
    """Metrics for buffer performance"""
    total_tokens: int = 0
    total_chars: int = 0
    buffer_flushes: int = 0
    average_buffer_size: float = 0.0
    min_flush_interval: float = float('inf')
    max_flush_interval: float = 0.0
    average_flush_interval: float = 0.0
    tokens_per_second: float = 0.0
    last_flush_time: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    
    def update_flush(self, buffer_size: int, flush_time: float):
        """Update metrics after a buffer flush"""
        self.buffer_flushes += 1
        
        # Update average buffer size
        self.average_buffer_size = (
            (self.average_buffer_size * (self.buffer_flushes - 1) + buffer_size)
            / self.buffer_flushes
        )
        
        # Update flush intervals
        if self.buffer_flushes > 1:
            interval = flush_time - self.last_flush_time
            self.min_flush_interval = min(self.min_flush_interval, interval)
            self.max_flush_interval = max(self.max_flush_interval, interval)
            
            # Update average interval
            total_intervals = self.buffer_flushes - 1
            self.average_flush_interval = (
                (self.average_flush_interval * (total_intervals - 1) + interval)
                / total_intervals
            )
        
        self.last_flush_time = flush_time
        
        # Update tokens per second
        elapsed = flush_time - self.start_time
        if elapsed > 0:
            self.tokens_per_second = self.total_tokens / elapsed
    
    def add_tokens(self, count: int, chars: int):
        """Add tokens to metrics"""
        self.total_tokens += count
        self.total_chars += chars
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_tokens": self.total_tokens,
            "total_chars": self.total_chars,
            "buffer_flushes": self.buffer_flushes,
            "average_buffer_size": round(self.average_buffer_size, 2),
            "min_flush_interval": round(self.min_flush_interval, 3),
            "max_flush_interval": round(self.max_flush_interval, 3),
            "average_flush_interval": round(self.average_flush_interval, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_time": round(time.time() - self.start_time, 2)
        }


class TokenBuffer:
    """
    Intelligent token buffer for smooth streaming display
    
    Features:
    - Multiple buffering strategies
    - Adaptive buffering based on stream speed
    - Word boundary awareness
    - Performance metrics
    - Callback system for buffer flushes
    """
    
    def __init__(
        self,
        strategy: BufferStrategy = BufferStrategy.ADAPTIVE,
        max_size: int = 20,
        max_time: float = 0.1,  # 100ms
        min_time: float = 0.01,  # 10ms
        word_boundary_aware: bool = True,
        flush_callback: Optional[Callable[[List[TokenChunk]], None]] = None
    ):
        self.strategy = strategy
        self.max_size = max_size
        self.max_time = max_time
        self.min_time = min_time
        self.word_boundary_aware = word_boundary_aware
        self.flush_callback = flush_callback
        
        # Buffer state
        self.buffer: Deque[TokenChunk] = deque()
        self.last_flush_time = time.time()
        self.token_counter = 0
        self.metrics = BufferMetrics()
        
        # Adaptive strategy state
        self._adaptive_max_time = max_time
        self._recent_intervals: Deque[float] = deque(maxlen=10)
        self._lock = asyncio.Lock()
        
        logger.debug(f"Initialized TokenBuffer with strategy={strategy.value}")
    
    async def add_token(self, content: str) -> Optional[List[TokenChunk]]:
        """
        Add a token to the buffer
        
        Args:
            content: Token content
            
        Returns:
            List of tokens if buffer was flushed, None otherwise
        """
        async with self._lock:
            # Create token chunk
            self.token_counter += 1
            chunk = TokenChunk(content=content, token_id=self.token_counter)
            
            # Add to buffer
            self.buffer.append(chunk)
            self.metrics.add_tokens(1, len(content))
            
            # Check if we should flush
            should_flush = await self._should_flush()
            
            if should_flush:
                return await self._flush_buffer()
            
            return None
    
    async def _should_flush(self) -> bool:
        """Determine if buffer should be flushed"""
        current_time = time.time()
        time_since_flush = current_time - self.last_flush_time
        buffer_size = len(self.buffer)
        
        if self.strategy == BufferStrategy.IMMEDIATE:
            return buffer_size > 0
        
        elif self.strategy == BufferStrategy.SIZE_BASED:
            return buffer_size >= self.max_size
        
        elif self.strategy == BufferStrategy.TIME_BASED:
            return time_since_flush >= self.max_time
        
        elif self.strategy == BufferStrategy.ADAPTIVE:
            return await self._adaptive_should_flush(time_since_flush, buffer_size)
        
        # Default fallback
        return buffer_size >= self.max_size or time_since_flush >= self.max_time
    
    async def _adaptive_should_flush(self, time_since_flush: float, buffer_size: int) -> bool:
        """Adaptive flushing logic"""
        # Always flush if buffer is full
        if buffer_size >= self.max_size:
            return True
        
        # Always flush if minimum time has passed and we have tokens
        if time_since_flush >= self.min_time and buffer_size > 0:
            # Check for word boundaries if enabled
            if self.word_boundary_aware:
                last_chunk = self.buffer[-1] if self.buffer else None
                if last_chunk and (last_chunk.is_whitespace or last_chunk.is_punctuation):
                    return True
        
        # Adaptive timing based on recent performance
        if self._recent_intervals:
            avg_interval = sum(self._recent_intervals) / len(self._recent_intervals)
            # Adjust max time based on recent performance
            self._adaptive_max_time = min(self.max_time, max(self.min_time, avg_interval * 0.8))
        
        # Flush if adaptive time threshold is reached
        return time_since_flush >= self._adaptive_max_time
    
    async def _flush_buffer(self) -> List[TokenChunk]:
        """Flush the buffer and return tokens"""
        if not self.buffer:
            return []
        
        current_time = time.time()
        
        # Get tokens to flush
        tokens = list(self.buffer)
        buffer_size = len(tokens)
        
        # Clear buffer
        self.buffer.clear()
        
        # Update metrics
        self.metrics.update_flush(buffer_size, current_time)
        
        # Update recent intervals for adaptive strategy
        if self.strategy == BufferStrategy.ADAPTIVE:
            interval = current_time - self.last_flush_time
            self._recent_intervals.append(interval)
        
        self.last_flush_time = current_time
        
        # Call flush callback if provided
        if self.flush_callback:
            try:
                self.flush_callback(tokens)
            except Exception as e:
                logger.error(f"Flush callback error: {e}")
        
        logger.debug(f"Flushed {buffer_size} tokens (strategy={self.strategy.value})")
        return tokens
    
    async def flush(self) -> List[TokenChunk]:
        """Force flush the buffer"""
        async with self._lock:
            return await self._flush_buffer()
    
    async def add_tokens_batch(self, contents: List[str]) -> List[List[TokenChunk]]:
        """Add multiple tokens and return all flushes"""
        flushes = []
        for content in contents:
            flush_result = await self.add_token(content)
            if flush_result:
                flushes.append(flush_result)
        
        # Force final flush if there are remaining tokens
        final_flush = await self.flush()
        if final_flush:
            flushes.append(final_flush)
        
        return flushes
    
    def get_buffer_content(self) -> str:
        """Get current buffer content as string"""
        return ''.join(chunk.content for chunk in self.buffer)
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def get_metrics(self) -> BufferMetrics:
        """Get buffer performance metrics"""
        return self.metrics
    
    def clear(self):
        """Clear the buffer without flushing"""
        self.buffer.clear()
        self.last_flush_time = time.time()
    
    def set_strategy(self, strategy: BufferStrategy):
        """Change buffering strategy"""
        self.strategy = strategy
        logger.debug(f"Changed buffer strategy to {strategy.value}")
    
    def update_config(
        self,
        max_size: Optional[int] = None,
        max_time: Optional[float] = None,
        min_time: Optional[float] = None,
        word_boundary_aware: Optional[bool] = None
    ):
        """Update buffer configuration"""
        if max_size is not None:
            self.max_size = max_size
        if max_time is not None:
            self.max_time = max_time
        if min_time is not None:
            self.min_time = min_time
        if word_boundary_aware is not None:
            self.word_boundary_aware = word_boundary_aware
        
        logger.debug(f"Updated buffer config: size={self.max_size}, time={self.max_time}")


class StreamingTokenBuffer:
    """Specialized buffer for streaming scenarios with async processing"""
    
    def __init__(
        self,
        buffer_config: Optional[Dict[str, Any]] = None,
        auto_flush: bool = True,
        flush_interval: float = 0.05  # 50ms
    ):
        config = buffer_config or {}
        self.buffer = TokenBuffer(**config)
        self.auto_flush = auto_flush
        self.flush_interval = flush_interval
        
        # Async processing
        self._flush_task: Optional[asyncio.Task] = None
        self._stop_flush_task = False
        self._content_callbacks: List[Callable[[str], None]] = []
    
    def add_content_callback(self, callback: Callable[[str], None]):
        """Add callback for when content is flushed"""
        self._content_callbacks.append(callback)
    
    def remove_content_callback(self, callback: Callable[[str], None]):
        """Remove content callback"""
        if callback in self._content_callbacks:
            self._content_callbacks.remove(callback)
    
    async def start_auto_flush(self):
        """Start automatic buffer flushing"""
        if self.auto_flush and not self._flush_task:
            self._stop_flush_task = False
            self._flush_task = asyncio.create_task(self._auto_flush_loop())
    
    async def stop_auto_flush(self):
        """Stop automatic buffer flushing"""
        self._stop_flush_task = True
        if self._flush_task:
            await self._flush_task
            self._flush_task = None
    
    async def _auto_flush_loop(self):
        """Automatic flush loop"""
        while not self._stop_flush_task:
            try:
                # Check if buffer should be flushed
                if self.buffer.get_buffer_size() > 0:
                    tokens = await self.buffer.flush()
                    if tokens:
                        content = ''.join(token.content for token in tokens)
                        
                        # Call content callbacks
                        for callback in self._content_callbacks:
                            try:
                                callback(content)
                            except Exception as e:
                                logger.error(f"Content callback error: {e}")
                
                # Wait for next check
                await asyncio.sleep(self.flush_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto flush error: {e}")
                await asyncio.sleep(self.flush_interval)
    
    async def add_content(self, content: str) -> Optional[str]:
        """
        Add content to buffer and return flushed content if any
        
        Args:
            content: Content to add
            
        Returns:
            Flushed content string if buffer was flushed
        """
        tokens = await self.buffer.add_token(content)
        if tokens:
            return ''.join(token.content for token in tokens)
        return None
    
    async def finalize(self) -> Optional[str]:
        """Finalize streaming and flush remaining content"""
        await self.stop_auto_flush()
        
        # Flush any remaining content
        tokens = await self.buffer.flush()
        if tokens:
            return ''.join(token.content for token in tokens)
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            "buffer_metrics": self.buffer.get_metrics().to_dict(),
            "current_buffer_size": self.buffer.get_buffer_size(),
            "current_content": self.buffer.get_buffer_content(),
            "auto_flush_active": self._flush_task is not None,
            "content_callbacks": len(self._content_callbacks)
        }


# Utility functions
def create_adaptive_buffer(
    target_fps: int = 20,  # Target updates per second
    max_tokens_per_update: int = 10
) -> TokenBuffer:
    """Create an adaptive buffer optimized for given FPS"""
    max_time = 1.0 / target_fps  # Convert FPS to time interval
    
    return TokenBuffer(
        strategy=BufferStrategy.ADAPTIVE,
        max_size=max_tokens_per_update,
        max_time=max_time,
        min_time=max_time / 4,  # 25% of target time
        word_boundary_aware=True
    )


def create_responsive_buffer() -> TokenBuffer:
    """Create a buffer optimized for responsive display"""
    return TokenBuffer(
        strategy=BufferStrategy.TIME_BASED,
        max_size=5,
        max_time=0.05,  # 50ms
        min_time=0.01,  # 10ms
        word_boundary_aware=True
    )


def create_performance_buffer() -> TokenBuffer:
    """Create a buffer optimized for performance"""
    return TokenBuffer(
        strategy=BufferStrategy.SIZE_BASED,
        max_size=50,
        max_time=0.2,  # 200ms
        word_boundary_aware=False
    )


if __name__ == "__main__":
    # Example usage and testing
    
    async def test_buffer():
        print("Testing TokenBuffer...")
        
        # Create adaptive buffer
        buffer = create_adaptive_buffer(target_fps=10)
        
        # Test content
        test_content = [
            "Hello", " there", "!", " How", " are", " you", " doing", " today", "?",
            " I", " hope", " everything", " is", " going", " well", " for", " you", "."
        ]
        
        print(f"Adding {len(test_content)} tokens...")
        
        all_flushes = []
        for content in test_content:
            flush_result = await buffer.add_token(content)
            if flush_result:
                flush_content = ''.join(token.content for token in flush_result)
                all_flushes.append(flush_content)
                print(f"Flushed: '{flush_content}'")
            
            # Simulate streaming delay
            await asyncio.sleep(0.02)
        
        # Final flush
        final_flush = await buffer.flush()
        if final_flush:
            final_content = ''.join(token.content for token in final_flush)
            all_flushes.append(final_content)
            print(f"Final flush: '{final_content}'")
        
        # Show metrics
        metrics = buffer.get_metrics()
        print(f"\nBuffer metrics: {metrics.to_dict()}")
        print(f"Total flushes: {len(all_flushes)}")
        print(f"Reconstructed: {''.join(all_flushes)}")
    
    # Run test
    asyncio.run(test_buffer())