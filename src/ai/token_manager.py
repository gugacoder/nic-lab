"""
Token Management and Counting System

Provides accurate token counting, usage limits, cost estimation,
and optimization for AI model interactions.
"""

import re
import time
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import asyncio
import threading

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """Token type classification"""
    INPUT = "input"         # Prompt tokens
    OUTPUT = "output"       # Generated tokens
    SYSTEM = "system"       # System prompt tokens
    CONTEXT = "context"     # Context/history tokens


@dataclass
class TokenCount:
    """Detailed token count information"""
    input_tokens: int = 0
    output_tokens: int = 0
    system_tokens: int = 0
    context_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        """Calculate total if not provided"""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens + self.system_tokens + self.context_tokens
    
    def __add__(self, other: 'TokenCount') -> 'TokenCount':
        """Add token counts together"""
        return TokenCount(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            system_tokens=self.system_tokens + other.system_tokens,
            context_tokens=self.context_tokens + other.context_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "system_tokens": self.system_tokens,
            "context_tokens": self.context_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class UsageWindow:
    """Token usage within a time window"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=1))
    token_count: TokenCount = field(default_factory=TokenCount)
    request_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if usage window has expired"""
        return datetime.now() > self.end_time
    
    def add_usage(self, tokens: TokenCount):
        """Add token usage to window"""
        self.token_count += tokens
        self.request_count += 1


class TokenCounter:
    """Advanced token counting with multiple strategies"""
    
    def __init__(self):
        # GPT-style tokenization patterns (approximation)
        self.word_pattern = re.compile(r'\b\w+\b')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Token estimation factors
        self.avg_chars_per_token = 4.0  # Approximate for most languages
        self.code_factor = 1.3  # Code tends to have more tokens
        self.json_factor = 1.2  # JSON structure adds tokens
        
        logger.debug("Initialized TokenCounter with approximation patterns")
    
    def count_tokens_simple(self, text: str) -> int:
        """
        Simple token estimation using character count
        Fast but less accurate method
        """
        if not text:
            return 0
        
        # Adjust for different content types
        factor = 1.0
        if self._is_code_like(text):
            factor = self.code_factor
        elif self._is_json_like(text):
            factor = self.json_factor
        
        # Character-based estimation
        base_tokens = len(text) / self.avg_chars_per_token
        return int(base_tokens * factor)
    
    def count_tokens_detailed(self, text: str) -> int:
        """
        More accurate token estimation using word/pattern analysis
        Slower but more accurate method
        """
        if not text:
            return 0
        
        # Count different text components
        words = len(self.word_pattern.findall(text))
        punctuation = len(self.punctuation_pattern.findall(text))
        
        # Special handling for different content types
        if self._is_code_like(text):
            # Code has more operators and special characters
            estimated_tokens = int((words + punctuation) * 1.4)
        elif self._is_json_like(text):
            # JSON has structural tokens
            estimated_tokens = int((words + punctuation) * 1.3)
        else:
            # Regular text
            estimated_tokens = words + (punctuation // 2)
        
        # Minimum 1 token for non-empty text
        return max(1, estimated_tokens)
    
    def count_tokens_tiktoken(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Accurate token counting using tiktoken library (if available)
        Most accurate but requires external dependency
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except ImportError:
            logger.warning("tiktoken not available, falling back to detailed counting")
            return self.count_tokens_detailed(text)
        except Exception as e:
            logger.warning(f"tiktoken error: {e}, falling back to detailed counting")
            return self.count_tokens_detailed(text)
    
    def count_tokens(
        self, 
        text: str, 
        method: str = "detailed",
        model: Optional[str] = None
    ) -> int:
        """
        Count tokens using specified method
        
        Args:
            text: Text to count tokens for
            method: Counting method ('simple', 'detailed', 'tiktoken')
            model: Model name for tiktoken method
        
        Returns:
            Estimated token count
        """
        if method == "simple":
            return self.count_tokens_simple(text)
        elif method == "tiktoken":
            return self.count_tokens_tiktoken(text, model or "gpt-3.5-turbo")
        else:
            return self.count_tokens_detailed(text)
    
    def count_message_tokens(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-3.5-turbo"
    ) -> TokenCount:
        """
        Count tokens for OpenAI-style message format
        
        Args:
            messages: List of message dictionaries
            model: Model name for accurate counting
        
        Returns:
            Detailed token count breakdown
        """
        input_tokens = 0
        system_tokens = 0
        context_tokens = 0
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Count content tokens
            content_tokens = self.count_tokens(content, "tiktoken", model)
            
            # Add role overhead (approximately 4 tokens per message)
            message_tokens = content_tokens + 4
            
            if role == "system":
                system_tokens += message_tokens
            elif role == "user":
                input_tokens += message_tokens
            elif role == "assistant":
                context_tokens += message_tokens
        
        return TokenCount(
            input_tokens=input_tokens,
            system_tokens=system_tokens,
            context_tokens=context_tokens,
            total_tokens=input_tokens + system_tokens + context_tokens
        )
    
    def estimate_completion_tokens(
        self, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        input_length: int = 0
    ) -> int:
        """
        Estimate completion tokens based on request parameters
        
        Args:
            max_tokens: Maximum tokens requested
            temperature: Response creativity
            input_length: Input token count
        
        Returns:
            Estimated completion tokens
        """
        if max_tokens:
            # Use requested max as upper bound
            base_estimate = max_tokens
        else:
            # Estimate based on input length and temperature
            base_estimate = min(1000, max(50, int(input_length * 0.5)))
        
        # Adjust for temperature (higher temp = potentially longer responses)
        temp_factor = 1.0 + (temperature * 0.2)
        
        return int(base_estimate * temp_factor)
    
    def _is_code_like(self, text: str) -> bool:
        """Detect if text looks like code"""
        code_indicators = [
            'def ', 'class ', 'import ', 'function', 'var ', 'const ',
            '{', '}', ';', '//', '/*', '*/', '#!/', 'SELECT ', 'FROM '
        ]
        return any(indicator in text for indicator in code_indicators)
    
    def _is_json_like(self, text: str) -> bool:
        """Detect if text looks like JSON"""
        stripped = text.strip()
        return (
            (stripped.startswith('{') and stripped.endswith('}')) or
            (stripped.startswith('[') and stripped.endswith(']'))
        )


class TokenLimitManager:
    """Manage token limits and usage quotas"""
    
    def __init__(
        self,
        requests_per_minute: int = 30,
        tokens_per_minute: int = 50000,
        max_context_tokens: int = 128000,
        history_minutes: int = 60
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.max_context_tokens = max_context_tokens
        self.history_minutes = history_minutes
        
        # Usage tracking
        self.usage_windows: deque = deque()
        self.current_window = self._create_usage_window()
        self._lock = threading.Lock()
        
        logger.info(f"Initialized TokenLimitManager: {requests_per_minute} req/min, {tokens_per_minute} tokens/min")
    
    def _create_usage_window(self) -> UsageWindow:
        """Create new usage tracking window"""
        return UsageWindow(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=1)
        )
    
    def _cleanup_expired_windows(self):
        """Remove expired usage windows"""
        cutoff_time = datetime.now() - timedelta(minutes=self.history_minutes)
        
        while self.usage_windows and self.usage_windows[0].start_time < cutoff_time:
            self.usage_windows.popleft()
    
    def can_make_request(self, estimated_tokens: TokenCount) -> Tuple[bool, str]:
        """
        Check if request can be made within limits
        
        Args:
            estimated_tokens: Estimated token usage
        
        Returns:
            Tuple of (can_proceed, reason_if_not)
        """
        with self._lock:
            self._cleanup_expired_windows()
            
            # Check if current window expired
            if self.current_window.is_expired():
                self.usage_windows.append(self.current_window)
                self.current_window = self._create_usage_window()
            
            # Check request rate limit
            if self.current_window.request_count >= self.requests_per_minute:
                return False, "Request rate limit exceeded"
            
            # Check token rate limit
            projected_tokens = self.current_window.token_count.total_tokens + estimated_tokens.total_tokens
            if projected_tokens > self.tokens_per_minute:
                return False, "Token rate limit would be exceeded"
            
            # Check context window limit
            if estimated_tokens.total_tokens > self.max_context_tokens:
                return False, f"Request exceeds maximum context tokens ({self.max_context_tokens})"
            
            return True, "OK"
    
    def record_usage(self, actual_tokens: TokenCount):
        """Record actual token usage"""
        with self._lock:
            if self.current_window.is_expired():
                self.usage_windows.append(self.current_window)
                self.current_window = self._create_usage_window()
            
            self.current_window.add_usage(actual_tokens)
            logger.debug(f"Recorded usage: {actual_tokens.total_tokens} tokens")
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        with self._lock:
            return {
                "current_window": {
                    "requests": self.current_window.request_count,
                    "tokens": self.current_window.token_count.to_dict(),
                    "time_remaining_seconds": (self.current_window.end_time - datetime.now()).total_seconds()
                },
                "limits": {
                    "requests_per_minute": self.requests_per_minute,
                    "tokens_per_minute": self.tokens_per_minute,
                    "max_context_tokens": self.max_context_tokens
                },
                "utilization": {
                    "request_utilization": self.current_window.request_count / self.requests_per_minute,
                    "token_utilization": self.current_window.token_count.total_tokens / self.tokens_per_minute
                }
            }
    
    def get_usage_history(self) -> List[Dict[str, Any]]:
        """Get historical usage data"""
        with self._lock:
            self._cleanup_expired_windows()
            
            history = []
            for window in list(self.usage_windows) + [self.current_window]:
                history.append({
                    "start_time": window.start_time.isoformat(),
                    "requests": window.request_count,
                    "tokens": window.token_count.to_dict()
                })
            
            return history


class TokenOptimizer:
    """Optimize token usage for cost and performance"""
    
    def __init__(self, counter: TokenCounter):
        self.counter = counter
        logger.debug("Initialized TokenOptimizer")
    
    def optimize_prompt(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        preserve_meaning: bool = True
    ) -> Tuple[str, int, float]:
        """
        Optimize prompt to reduce token usage
        
        Args:
            prompt: Original prompt
            max_tokens: Maximum tokens allowed
            preserve_meaning: Whether to preserve semantic meaning
        
        Returns:
            Tuple of (optimized_prompt, token_savings, compression_ratio)
        """
        original_tokens = self.counter.count_tokens(prompt)
        
        if not max_tokens or original_tokens <= max_tokens:
            return prompt, 0, 1.0
        
        # Apply optimization strategies
        optimized = prompt
        
        if preserve_meaning:
            # Conservative optimizations
            optimized = self._remove_redundancy(optimized)
            optimized = self._compress_whitespace(optimized)
            optimized = self._simplify_language(optimized)
        else:
            # Aggressive optimizations
            optimized = self._extract_keywords(optimized)
            optimized = self._abbreviate_common_phrases(optimized)
        
        new_tokens = self.counter.count_tokens(optimized)
        savings = original_tokens - new_tokens
        ratio = new_tokens / original_tokens if original_tokens > 0 else 1.0
        
        logger.debug(f"Token optimization: {original_tokens} -> {new_tokens} ({savings} saved)")
        return optimized, savings, ratio
    
    def optimize_context(
        self, 
        messages: List[Dict[str, str]], 
        max_context_tokens: int
    ) -> List[Dict[str, str]]:
        """
        Optimize message context to fit within token limits
        
        Args:
            messages: List of messages
            max_context_tokens: Maximum context tokens allowed
        
        Returns:
            Optimized message list
        """
        # Count current tokens
        current_tokens = self.counter.count_message_tokens(messages)
        
        if current_tokens.total_tokens <= max_context_tokens:
            return messages
        
        # Strategy: Remove oldest messages while preserving system and recent context
        optimized_messages = []
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        # Always keep system messages
        optimized_messages.extend(system_messages)
        
        # Add other messages from most recent until we hit limit
        remaining_tokens = max_context_tokens
        
        # Count system message tokens
        for msg in system_messages:
            tokens = self.counter.count_tokens(msg.get("content", ""))
            remaining_tokens -= tokens + 4  # Message overhead
        
        # Add messages in reverse order (most recent first)
        for msg in reversed(other_messages):
            msg_tokens = self.counter.count_tokens(msg.get("content", "")) + 4
            if remaining_tokens >= msg_tokens:
                optimized_messages.insert(-len(system_messages), msg)
                remaining_tokens -= msg_tokens
            else:
                break
        
        logger.info(f"Context optimized: {len(messages)} -> {len(optimized_messages)} messages")
        return optimized_messages
    
    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant phrases and repetition"""
        # Simple redundancy removal
        lines = text.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            line_key = line.strip().lower()
            if line_key not in seen:
                unique_lines.append(line)
                seen.add(line_key)
        
        return '\n'.join(unique_lines)
    
    def _compress_whitespace(self, text: str) -> str:
        """Compress excessive whitespace"""
        # Replace multiple spaces with single space
        compressed = re.sub(r'\s+', ' ', text)
        # Remove excessive line breaks
        compressed = re.sub(r'\n\s*\n\s*\n+', '\n\n', compressed)
        return compressed.strip()
    
    def _simplify_language(self, text: str) -> str:
        """Simplify verbose language"""
        # Common verbose -> concise replacements
        replacements = {
            r'\bin order to\b': 'to',
            r'\bdue to the fact that\b': 'because',
            r'\bin the event that\b': 'if',
            r'\bat this point in time\b': 'now',
            r'\bfor the purpose of\b': 'to',
            r'\bwith regard to\b': 'regarding',
            r'\bin accordance with\b': 'per',
            r'\bas a result of\b': 'due to'
        }
        
        simplified = text
        for verbose, concise in replacements.items():
            simplified = re.sub(verbose, concise, simplified, flags=re.IGNORECASE)
        
        return simplified
    
    def _extract_keywords(self, text: str) -> str:
        """Extract key information (aggressive optimization)"""
        # This is a simple implementation - in practice you might use NLP
        words = text.split()
        
        # Keep important words (nouns, verbs, adjectives)
        # This is a simplified heuristic
        important_words = []
        skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        for word in words:
            if word.lower() not in skip_words and len(word) > 2:
                important_words.append(word)
        
        return ' '.join(important_words)
    
    def _abbreviate_common_phrases(self, text: str) -> str:
        """Abbreviate common phrases"""
        abbreviations = {
            'for example': 'e.g.',
            'that is': 'i.e.',
            'and so on': 'etc.',
            'as soon as possible': 'ASAP',
            'frequently asked questions': 'FAQ'
        }
        
        abbreviated = text
        for phrase, abbrev in abbreviations.items():
            abbreviated = abbreviated.replace(phrase, abbrev)
        
        return abbreviated


# High-level token management class
class TokenManager:
    """Comprehensive token management system"""
    
    def __init__(self, **kwargs):
        self.counter = TokenCounter()
        self.limit_manager = TokenLimitManager(**kwargs)
        self.optimizer = TokenOptimizer(self.counter)
        
        logger.info("Initialized comprehensive TokenManager")
    
    async def validate_request(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        model: str = "llama-3.1-70b-versatile"
    ) -> Tuple[bool, str, TokenCount]:
        """
        Validate request against token limits
        
        Returns:
            Tuple of (can_proceed, message, estimated_usage)
        """
        # Count input tokens
        input_count = self.counter.count_message_tokens(messages, model)
        
        # Estimate completion tokens
        completion_estimate = self.counter.estimate_completion_tokens(
            max_tokens=max_tokens,
            input_length=input_count.total_tokens
        )
        
        total_estimate = TokenCount(
            input_tokens=input_count.total_tokens,
            output_tokens=completion_estimate,
            total_tokens=input_count.total_tokens + completion_estimate
        )
        
        # Check limits
        can_proceed, reason = self.limit_manager.can_make_request(total_estimate)
        
        return can_proceed, reason, total_estimate
    
    async def record_actual_usage(
        self, 
        input_tokens: int, 
        output_tokens: int
    ):
        """Record actual token usage after request"""
        actual_usage = TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
        
        self.limit_manager.record_usage(actual_usage)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        return self.limit_manager.get_current_usage()
    
    def optimize_for_request(
        self, 
        messages: List[Dict[str, str]], 
        max_context_tokens: int = 100000
    ) -> List[Dict[str, str]]:
        """Optimize messages for a request"""
        return self.optimizer.optimize_context(messages, max_context_tokens)


# Global token manager instance
_token_manager: Optional[TokenManager] = None

def get_token_manager(**kwargs) -> TokenManager:
    """Get global token manager instance"""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager(**kwargs)
    return _token_manager


# CLI utilities
def verify_tokens(text: str, method: str = "detailed"):
    """Verify token counting via CLI"""
    counter = TokenCounter()
    
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Length: {len(text)} characters")
    
    methods = ["simple", "detailed", "tiktoken"]
    for m in methods:
        try:
            tokens = counter.count_tokens(text, m)
            print(f"Tokens ({m}): {tokens}")
        except Exception as e:
            print(f"Tokens ({m}): Error - {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "verify":
            text = sys.argv[2] if len(sys.argv) > 2 else "Hello, how are you doing today?"
            verify_tokens(text)
        elif command == "test":
            # Run comprehensive tests
            print("Testing TokenManager functionality...")
            
            manager = TokenManager()
            
            # Test messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
            
            print("Message token count:")
            token_count = manager.counter.count_message_tokens(messages)
            print(f"  {token_count.to_dict()}")
            
            print("Usage stats:")
            stats = manager.get_usage_stats()
            print(f"  {stats}")
        else:
            print("Available commands: verify [text], test")
    else:
        print("Usage: python -m src.ai.token_manager [verify|test] [text]")