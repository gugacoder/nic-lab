"""
Search Result Caching System

This module provides intelligent caching for GitLab search results to improve
performance and reduce API calls while maintaining data freshness.
"""

import logging
import json
import hashlib
import pickle
import time
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import OrderedDict

from ..gitlab_client import GitLabSearchResult

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 3600  # 1 hour default
    tags: List[str] = None
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size of the cached value"""
        try:
            return len(pickle.dumps(self.value))
        except Exception:
            return len(str(self.value))
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def touch(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'ttl_seconds': self.ttl_seconds,
            'tags': self.tags,
            'size_bytes': self.size_bytes
        }


@dataclass
class CacheConfig:
    """Configuration for search cache"""
    max_size_mb: int = 100  # Maximum cache size in MB
    default_ttl_seconds: int = 3600  # Default TTL (1 hour)
    cleanup_interval_seconds: int = 300  # Cleanup every 5 minutes
    max_entries: int = 1000  # Maximum number of cache entries
    enable_persistence: bool = True  # Persist cache to disk
    cache_directory: str = "cache"  # Directory for cache files
    compression_enabled: bool = True  # Enable compression for cached data
    cache_hit_boost_ttl: float = 1.5  # Extend TTL for frequently accessed items


class SearchResultCache:
    """Intelligent caching system for search results"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize search cache
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0
        }
        
        # Setup cache directory
        if self.config.enable_persistence:
            self.cache_dir = Path(self.config.cache_directory)
            self.cache_dir.mkdir(exist_ok=True)
            self._load_persistent_cache()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def get(
        self,
        key: str,
        default: Any = None,
        extend_ttl: bool = True
    ) -> Any:
        """Get value from cache
        
        Args:
            key: Cache key
            default: Default value if not found
            extend_ttl: Whether to extend TTL for frequently accessed items
            
        Returns:
            Cached value or default
        """
        with self._lock:
            if key not in self._cache:
                self._cache_stats['misses'] += 1
                return default
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired:
                logger.debug(f"Cache entry expired: {key}")
                del self._cache[key]
                self._cache_stats['misses'] += 1
                return default
            
            # Update access info
            entry.touch()
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            
            # Extend TTL for frequently accessed items
            if extend_ttl and entry.access_count > 5:
                entry.ttl_seconds = int(entry.ttl_seconds * self.config.cache_hit_boost_ttl)
            
            self._cache_stats['hits'] += 1
            logger.debug(f"Cache hit: {key} (age: {entry.age_seconds:.1f}s, access: {entry.access_count})")
            
            return entry.value
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Put value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            tags: Tags for cache entry
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            try:
                ttl = ttl_seconds or self.config.default_ttl_seconds
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    ttl_seconds=ttl,
                    tags=tags or []
                )
                
                # Check size limits
                if not self._check_size_limits(entry):
                    logger.warning(f"Cache entry too large: {key} ({entry.size_bytes} bytes)")
                    return False
                
                # Add to cache
                self._cache[key] = entry
                self._cache_stats['size_bytes'] += entry.size_bytes
                
                # Trigger cleanup if needed
                self._cleanup_if_needed()
                
                logger.debug(f"Cached: {key} (TTL: {ttl}s, size: {entry.size_bytes} bytes)")
                return True
                
            except Exception as e:
                logger.error(f"Error caching {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if entry was deleted
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._cache_stats['size_bytes'] -= entry.size_bytes
                del self._cache[key]
                logger.debug(f"Deleted cache entry: {key}")
                return True
            return False
    
    def clear_by_tags(self, tags: List[str]) -> int:
        """Clear cache entries matching any of the provided tags
        
        Args:
            tags: List of tags to match
            
        Returns:
            Number of entries cleared
        """
        with self._lock:
            keys_to_delete = []
            
            for key, entry in self._cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                self.delete(key)
            
            logger.info(f"Cleared {len(keys_to_delete)} cache entries with tags: {tags}")
            return len(keys_to_delete)
    
    def invalidate_project(self, project_id: int) -> int:
        """Invalidate all cache entries for a specific project
        
        Args:
            project_id: Project ID to invalidate
            
        Returns:
            Number of entries invalidated
        """
        return self.clear_by_tags([f"project:{project_id}"])
    
    def get_search_results(
        self,
        query: str,
        project_ids: Optional[List[int]] = None,
        file_extensions: Optional[List[str]] = None,
        strategy: str = "default"
    ) -> Optional[List[GitLabSearchResult]]:
        """Get cached search results
        
        Args:
            query: Search query
            project_ids: Project IDs that were searched
            file_extensions: File extensions that were filtered
            strategy: Search strategy used
            
        Returns:
            Cached search results or None
        """
        cache_key = self._generate_search_cache_key(
            query, project_ids, file_extensions, strategy
        )
        
        cached_data = self.get(cache_key)
        if cached_data is None:
            return None
        
        try:
            # Deserialize search results
            return [GitLabSearchResult(**result_data) for result_data in cached_data]
        except Exception as e:
            logger.error(f"Error deserializing cached search results: {e}")
            self.delete(cache_key)
            return None
    
    def cache_search_results(
        self,
        query: str,
        results: List[GitLabSearchResult],
        project_ids: Optional[List[int]] = None,
        file_extensions: Optional[List[str]] = None,
        strategy: str = "default",
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache search results
        
        Args:
            query: Search query
            results: Search results to cache
            project_ids: Project IDs that were searched
            file_extensions: File extensions that were filtered
            strategy: Search strategy used
            ttl_seconds: Time-to-live for cache entry
            
        Returns:
            True if successfully cached
        """
        cache_key = self._generate_search_cache_key(
            query, project_ids, file_extensions, strategy
        )
        
        # Serialize search results
        serialized_results = []
        for result in results:
            serialized_results.append({
                'project_id': result.project_id,
                'project_name': result.project_name,
                'file_path': result.file_path,
                'ref': result.ref,
                'startline': result.startline,
                'content': result.content,
                'wiki': result.wiki
            })
        
        # Create tags for invalidation
        tags = [f"query:{hashlib.md5(query.encode()).hexdigest()[:8]}", f"strategy:{strategy}"]
        if project_ids:
            tags.extend([f"project:{pid}" for pid in project_ids])
        
        return self.put(
            key=cache_key,
            value=serialized_results,
            ttl_seconds=ttl_seconds,
            tags=tags
        )
    
    def _generate_search_cache_key(
        self,
        query: str,
        project_ids: Optional[List[int]] = None,
        file_extensions: Optional[List[str]] = None,
        strategy: str = "default"
    ) -> str:
        """Generate cache key for search parameters
        
        Args:
            query: Search query
            project_ids: Project IDs
            file_extensions: File extensions
            strategy: Search strategy
            
        Returns:
            Cache key string
        """
        # Create a deterministic key from parameters
        key_data = {
            'query': query.lower().strip(),
            'project_ids': sorted(project_ids) if project_ids else None,
            'file_extensions': sorted(file_extensions) if file_extensions else None,
            'strategy': strategy
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"search:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _check_size_limits(self, entry: CacheEntry) -> bool:
        """Check if entry fits within size limits
        
        Args:
            entry: Cache entry to check
            
        Returns:
            True if entry fits within limits
        """
        # Check individual entry size (max 10MB per entry)
        max_entry_size = 10 * 1024 * 1024  # 10MB
        if entry.size_bytes > max_entry_size:
            return False
        
        # Check total cache size
        max_total_size = self.config.max_size_mb * 1024 * 1024
        if self._cache_stats['size_bytes'] + entry.size_bytes > max_total_size:
            # Try to make room by evicting old entries
            self._evict_entries_for_space(entry.size_bytes)
        
        return True
    
    def _evict_entries_for_space(self, needed_bytes: int):
        """Evict old entries to make space
        
        Args:
            needed_bytes: Number of bytes needed
        """
        max_total_size = self.config.max_size_mb * 1024 * 1024
        target_size = max_total_size - needed_bytes
        
        # Sort by last accessed time (oldest first)
        entries_by_age = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        evicted_count = 0
        for key, entry in entries_by_age:
            if self._cache_stats['size_bytes'] <= target_size:
                break
            
            self._cache_stats['size_bytes'] -= entry.size_bytes
            del self._cache[key]
            evicted_count += 1
        
        self._cache_stats['evictions'] += evicted_count
        logger.info(f"Evicted {evicted_count} cache entries to make space")
    
    def _cleanup_if_needed(self):
        """Perform cleanup if cache exceeds limits"""
        # Check entry count limit
        if len(self._cache) > self.config.max_entries:
            excess_count = len(self._cache) - self.config.max_entries
            
            # Remove oldest entries
            for _ in range(excess_count):
                if self._cache:
                    oldest_key = next(iter(self._cache))
                    entry = self._cache[oldest_key]
                    self._cache_stats['size_bytes'] -= entry.size_bytes
                    del self._cache[oldest_key]
                    self._cache_stats['evictions'] += 1
    
    def _cleanup_worker(self):
        """Background thread for periodic cleanup"""
        while True:
            try:
                time.sleep(self.config.cleanup_interval_seconds)
                self._cleanup_expired_entries()
                
                if self.config.enable_persistence:
                    self._save_persistent_cache()
                    
            except Exception as e:
                logger.error(f"Error in cache cleanup worker: {e}")
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache"""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache[key]
                self._cache_stats['size_bytes'] -= entry.size_bytes
                del self._cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _load_persistent_cache(self):
        """Load cache from persistent storage"""
        cache_file = self.cache_dir / "search_cache.pkl"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Restore cache entries that haven't expired
                restored_count = 0
                for key, entry_data in cached_data.items():
                    try:
                        entry = CacheEntry(**entry_data['metadata'])
                        entry.value = entry_data['value']
                        
                        if not entry.is_expired:
                            self._cache[key] = entry
                            self._cache_stats['size_bytes'] += entry.size_bytes
                            restored_count += 1
                    except Exception as e:
                        logger.warning(f"Error restoring cache entry {key}: {e}")
                
                logger.info(f"Restored {restored_count} cache entries from disk")
                
        except Exception as e:
            logger.error(f"Error loading persistent cache: {e}")
    
    def _save_persistent_cache(self):
        """Save cache to persistent storage"""
        cache_file = self.cache_dir / "search_cache.pkl"
        
        try:
            with self._lock:
                # Prepare data for serialization
                cache_data = {}
                for key, entry in self._cache.items():
                    cache_data[key] = {
                        'metadata': {
                            'key': entry.key,
                            'created_at': entry.created_at,
                            'last_accessed': entry.last_accessed,
                            'access_count': entry.access_count,
                            'ttl_seconds': entry.ttl_seconds,
                            'tags': entry.tags,
                            'size_bytes': entry.size_bytes
                        },
                        'value': entry.value
                    }
            
            # Write to temporary file then move (atomic operation)
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            temp_file.replace(cache_file)
            logger.debug("Saved cache to persistent storage")
            
        except Exception as e:
            logger.error(f"Error saving persistent cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
            hit_rate = (self._cache_stats['hits'] / total_requests) if total_requests > 0 else 0.0
            
            return {
                'hit_rate': hit_rate,
                'hits': self._cache_stats['hits'],
                'misses': self._cache_stats['misses'],
                'evictions': self._cache_stats['evictions'],
                'total_entries': len(self._cache),
                'size_mb': self._cache_stats['size_bytes'] / (1024 * 1024),
                'size_bytes': self._cache_stats['size_bytes'],
                'config': {
                    'max_size_mb': self.config.max_size_mb,
                    'max_entries': self.config.max_entries,
                    'default_ttl_seconds': self.config.default_ttl_seconds
                }
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._cache_stats['size_bytes'] = 0
            logger.info("Cleared all cache entries")
    
    def analyze_cache(self) -> Dict[str, Any]:
        """Analyze cache performance and provide insights
        
        Returns:
            Cache analysis results
        """
        with self._lock:
            stats = self.get_stats()
            
            # Analyze entry distribution
            entry_ages = [(datetime.now() - entry.created_at).total_seconds() 
                         for entry in self._cache.values()]
            entry_sizes = [entry.size_bytes for entry in self._cache.values()]
            access_counts = [entry.access_count for entry in self._cache.values()]
            
            analysis = {
                'performance': {
                    'hit_rate': stats['hit_rate'],
                    'efficiency_score': min(1.0, stats['hit_rate'] * 1.2),  # Boost for good hit rates
                    'memory_utilization': stats['size_mb'] / self.config.max_size_mb
                },
                'entry_distribution': {
                    'total_entries': len(entry_ages),
                    'avg_age_seconds': sum(entry_ages) / len(entry_ages) if entry_ages else 0,
                    'avg_size_bytes': sum(entry_sizes) / len(entry_sizes) if entry_sizes else 0,
                    'avg_access_count': sum(access_counts) / len(access_counts) if access_counts else 0
                },
                'recommendations': []
            }
            
            # Generate recommendations
            if stats['hit_rate'] < 0.3:
                analysis['recommendations'].append("Low hit rate - consider increasing cache TTL")
            
            if stats['size_mb'] / self.config.max_size_mb > 0.9:
                analysis['recommendations'].append("Cache nearly full - consider increasing max size")
            
            if analysis['entry_distribution']['avg_access_count'] < 2:
                analysis['recommendations'].append("Many entries accessed only once - consider shorter TTL")
            
            return analysis


# Global cache instance
_search_cache: Optional[SearchResultCache] = None


def get_search_cache(config: Optional[CacheConfig] = None) -> SearchResultCache:
    """Get global search cache instance
    
    Args:
        config: Optional cache configuration
        
    Returns:
        Search cache instance
    """
    global _search_cache
    if _search_cache is None:
        _search_cache = SearchResultCache(config)
    return _search_cache


if __name__ == "__main__":
    # Test search cache functionality
    import sys
    
    print("Testing search result cache...")
    
    # Create cache with test configuration
    config = CacheConfig(
        max_size_mb=10,
        default_ttl_seconds=60,
        enable_persistence=False
    )
    cache = SearchResultCache(config)
    
    # Test basic operations
    test_data = [
        {'key': 'test1', 'value': 'Test data 1'},
        {'key': 'test2', 'value': 'Test data 2'},
        {'key': 'test3', 'value': {'complex': 'data', 'with': ['nested', 'values']}}
    ]
    
    print("Testing cache operations:")
    
    # Test putting and getting
    for item in test_data:
        success = cache.put(item['key'], item['value'], tags=['test'])
        print(f"  Put {item['key']}: {success}")
    
    for item in test_data:
        value = cache.get(item['key'])
        hit = value is not None
        print(f"  Get {item['key']}: {'HIT' if hit else 'MISS'}")
    
    # Test search result caching
    print("\nTesting search result caching:")
    
    # Mock search results
    mock_results = [
        GitLabSearchResult(
            project_id=1,
            project_name="test-project",
            file_path="test.md",
            ref="main",
            startline=1,
            content="Test content",
            wiki=False
        )
    ]
    
    # Cache search results
    success = cache.cache_search_results(
        query="test query",
        results=mock_results,
        project_ids=[1],
        strategy="keyword"
    )
    print(f"  Cache search results: {success}")
    
    # Retrieve cached search results
    cached_results = cache.get_search_results(
        query="test query",
        project_ids=[1],
        strategy="keyword"
    )
    print(f"  Retrieve search results: {'SUCCESS' if cached_results else 'FAILED'}")
    
    # Show cache statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Size: {stats['size_mb']:.2f} MB")
    
    # Analyze cache
    analysis = cache.analyze_cache()
    print(f"\nCache Analysis:")
    print(f"  Efficiency score: {analysis['performance']['efficiency_score']:.2f}")
    print(f"  Memory utilization: {analysis['performance']['memory_utilization']:.2%}")
    if analysis['recommendations']:
        print(f"  Recommendations: {analysis['recommendations']}")
    
    print("\nCache testing complete")