"""
Multi-Level Search Cache Implementation

This module provides a sophisticated multi-level caching system for search operations,
with support for LRU eviction, TTL expiration, and intelligent cache management.
"""

import time
import pickle
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from enum import Enum
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels with different characteristics"""
    MEMORY_HOT = "memory_hot"      # In-memory, small, very fast
    MEMORY_WARM = "memory_warm"    # In-memory, larger, fast
    DISK = "disk"                  # Persistent, largest, slower


@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 1
    ttl_seconds: int = 3600
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    cache_level: CacheLevel = CacheLevel.MEMORY_WARM
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds <= 0:  # No expiration
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def update_access(self):
        """Update access timestamp and count"""
        self.accessed_at = datetime.now()
        self.access_count += 1
    
    def get_score(self) -> float:
        """Calculate entry score for eviction decisions"""
        age_seconds = (datetime.now() - self.accessed_at).total_seconds()
        frequency_score = min(self.access_count / 10, 1.0)  # Cap at 10 accesses
        recency_score = 1.0 / (1.0 + age_seconds / 3600)  # Decay over hours
        size_penalty = min(self.size_bytes / (1024 * 1024), 1.0)  # MB scale
        
        return (frequency_score * 0.4 + recency_score * 0.4 - size_penalty * 0.2)


@dataclass
class CacheLevelConfig:
    """Configuration for a cache level"""
    max_entries: int
    max_size_mb: int
    ttl_seconds: int
    eviction_threshold: float = 0.9  # Start eviction at 90% capacity


@dataclass
class MultiLevelCacheConfig:
    """Configuration for multi-level cache system"""
    # Level configurations
    hot_config: CacheLevelConfig = field(default_factory=lambda: CacheLevelConfig(
        max_entries=100,
        max_size_mb=10,
        ttl_seconds=300  # 5 minutes
    ))
    warm_config: CacheLevelConfig = field(default_factory=lambda: CacheLevelConfig(
        max_entries=1000,
        max_size_mb=100,
        ttl_seconds=3600  # 1 hour
    ))
    disk_config: CacheLevelConfig = field(default_factory=lambda: CacheLevelConfig(
        max_entries=10000,
        max_size_mb=1000,
        ttl_seconds=86400  # 24 hours
    ))
    
    # General settings
    enable_disk_cache: bool = True
    cache_dir: str = "cache/search"
    enable_compression: bool = True
    cleanup_interval_seconds: int = 300  # 5 minutes
    promotion_threshold: int = 3  # Access count for promotion
    enable_metrics: bool = True


class MultiLevelSearchCache:
    """Advanced multi-level cache implementation for search operations
    
    This class implements a hierarchical cache with hot, warm, and optional disk levels,
    intelligent promotion/demotion, and comprehensive cache management features.
    """
    
    def __init__(self, config: Optional[MultiLevelCacheConfig] = None):
        """Initialize multi-level cache
        
        Args:
            config: Cache configuration
        """
        self.config = config or MultiLevelCacheConfig()
        
        # Initialize cache levels
        self._hot_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._warm_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Cache metadata
        self._hot_size_bytes = 0
        self._warm_size_bytes = 0
        self._disk_size_bytes = 0
        
        # Tag index for fast lookup
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'promotions': 0,
            'demotions': 0,
            'disk_hits': 0,
            'disk_writes': 0,
            'total_requests': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize disk cache if enabled
        if self.config.enable_disk_cache:
            self._init_disk_cache()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def get(
        self,
        key: str,
        default: Any = None,
        promote: bool = True
    ) -> Any:
        """Get value from cache with multi-level lookup
        
        Args:
            key: Cache key
            default: Default value if not found
            promote: Whether to promote entry on access
            
        Returns:
            Cached value or default
        """
        with self._lock:
            self._stats['total_requests'] += 1
            
            # Check hot cache first
            if key in self._hot_cache:
                entry = self._hot_cache[key]
                if not entry.is_expired():
                    entry.update_access()
                    self._hot_cache.move_to_end(key)  # LRU update
                    self._stats['hits'] += 1
                    logger.debug(f"Cache hit (hot): {key}")
                    return entry.value
                else:
                    # Remove expired entry
                    self._remove_entry(key, CacheLevel.MEMORY_HOT)
            
            # Check warm cache
            if key in self._warm_cache:
                entry = self._warm_cache[key]
                if not entry.is_expired():
                    entry.update_access()
                    self._warm_cache.move_to_end(key)  # LRU update
                    self._stats['hits'] += 1
                    logger.debug(f"Cache hit (warm): {key}")
                    
                    # Consider promotion to hot cache
                    if promote and entry.access_count >= self.config.promotion_threshold:
                        self._promote_entry(key, entry)
                    
                    return entry.value
                else:
                    # Remove expired entry
                    self._remove_entry(key, CacheLevel.MEMORY_WARM)
            
            # Check disk cache if enabled
            if self.config.enable_disk_cache:
                disk_value = self._get_from_disk(key)
                if disk_value is not None:
                    self._stats['disk_hits'] += 1
                    # Promote to warm cache
                    if promote:
                        self.put(key, disk_value, ttl_seconds=self.config.warm_config.ttl_seconds)
                    return disk_value
            
            # Cache miss
            self._stats['misses'] += 1
            logger.debug(f"Cache miss: {key}")
            return default
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        cache_level: Optional[CacheLevel] = None
    ) -> bool:
        """Put value into cache with automatic level selection
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL override
            tags: Tags for grouping/invalidation
            cache_level: Force specific cache level
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            try:
                # Calculate entry size
                size_bytes = self._estimate_size(value)
                
                # Determine cache level if not specified
                if cache_level is None:
                    cache_level = self._select_cache_level(size_bytes)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    accessed_at=datetime.now(),
                    ttl_seconds=ttl_seconds or self._get_default_ttl(cache_level),
                    size_bytes=size_bytes,
                    tags=tags or [],
                    cache_level=cache_level
                )
                
                # Add to appropriate cache level
                if cache_level == CacheLevel.MEMORY_HOT:
                    self._add_to_hot(key, entry)
                elif cache_level == CacheLevel.MEMORY_WARM:
                    self._add_to_warm(key, entry)
                elif cache_level == CacheLevel.DISK and self.config.enable_disk_cache:
                    self._write_to_disk(key, entry)
                
                # Update tag index
                for tag in entry.tags:
                    self._tag_index[tag].add(key)
                
                logger.debug(f"Cache put ({cache_level.value}): {key} (size: {size_bytes} bytes)")
                return True
                
            except Exception as e:
                logger.error(f"Error putting key {key} in cache: {e}")
                return False
    
    def put_search_results(
        self,
        query: str,
        results: List[Any],
        strategy: str = "hybrid",
        project_ids: Optional[List[int]] = None,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache search results with query-specific key generation
        
        Args:
            query: Search query
            results: Search results
            strategy: Search strategy used
            project_ids: Project IDs searched
            ttl_seconds: TTL override
            
        Returns:
            True if successfully cached
        """
        # Generate cache key
        key = self._generate_search_key(query, strategy, project_ids)
        
        # Add search-specific tags
        tags = [f"search:{strategy}", "search_results"]
        if project_ids:
            tags.extend([f"project:{pid}" for pid in project_ids[:5]])  # Limit tags
        
        return self.put(key, results, ttl_seconds=ttl_seconds, tags=tags)
    
    def get_search_results(
        self,
        query: str,
        strategy: str = "hybrid",
        project_ids: Optional[List[int]] = None
    ) -> Optional[List[Any]]:
        """Get cached search results
        
        Args:
            query: Search query
            strategy: Search strategy
            project_ids: Project IDs
            
        Returns:
            Cached results or None
        """
        key = self._generate_search_key(query, strategy, project_ids)
        return self.get(key)
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry across all levels
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was found and removed
        """
        with self._lock:
            removed = False
            
            # Remove from hot cache
            if key in self._hot_cache:
                self._remove_entry(key, CacheLevel.MEMORY_HOT)
                removed = True
            
            # Remove from warm cache
            if key in self._warm_cache:
                self._remove_entry(key, CacheLevel.MEMORY_WARM)
                removed = True
            
            # Remove from disk cache
            if self.config.enable_disk_cache:
                if self._remove_from_disk(key):
                    removed = True
            
            return removed
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if tag not in self._tag_index:
                return 0
            
            keys_to_remove = list(self._tag_index[tag])
            count = 0
            
            for key in keys_to_remove:
                if self.invalidate(key):
                    count += 1
            
            # Clean up tag index
            del self._tag_index[tag]
            
            logger.info(f"Invalidated {count} entries with tag '{tag}'")
            return count
    
    def clear(self):
        """Clear all cache levels"""
        with self._lock:
            self._hot_cache.clear()
            self._warm_cache.clear()
            self._hot_size_bytes = 0
            self._warm_size_bytes = 0
            self._tag_index.clear()
            
            if self.config.enable_disk_cache:
                self._clear_disk_cache()
            
            logger.info("Cleared all cache levels")
    
    def _add_to_hot(self, key: str, entry: CacheEntry):
        """Add entry to hot cache with eviction if needed"""
        # Remove existing entry if present
        if key in self._hot_cache:
            self._remove_entry(key, CacheLevel.MEMORY_HOT)
        
        # Check capacity and evict if needed
        while (len(self._hot_cache) >= self.config.hot_config.max_entries * self.config.hot_config.eviction_threshold or
               self._hot_size_bytes + entry.size_bytes > self.config.hot_config.max_size_mb * 1024 * 1024 * self.config.hot_config.eviction_threshold):
            self._evict_from_hot()
        
        # Add new entry
        self._hot_cache[key] = entry
        self._hot_size_bytes += entry.size_bytes
    
    def _add_to_warm(self, key: str, entry: CacheEntry):
        """Add entry to warm cache with eviction if needed"""
        # Remove existing entry if present
        if key in self._warm_cache:
            self._remove_entry(key, CacheLevel.MEMORY_WARM)
        
        # Check capacity and evict if needed
        while (len(self._warm_cache) >= self.config.warm_config.max_entries * self.config.warm_config.eviction_threshold or
               self._warm_size_bytes + entry.size_bytes > self.config.warm_config.max_size_mb * 1024 * 1024 * self.config.warm_config.eviction_threshold):
            self._evict_from_warm()
        
        # Add new entry
        self._warm_cache[key] = entry
        self._warm_size_bytes += entry.size_bytes
    
    def _evict_from_hot(self):
        """Evict least valuable entry from hot cache"""
        if not self._hot_cache:
            return
        
        # Find entry with lowest score
        min_score = float('inf')
        evict_key = None
        
        for key, entry in self._hot_cache.items():
            score = entry.get_score()
            if score < min_score:
                min_score = score
                evict_key = key
        
        if evict_key:
            entry = self._hot_cache[evict_key]
            # Demote to warm cache
            self._demote_entry(evict_key, entry)
            self._stats['evictions'] += 1
    
    def _evict_from_warm(self):
        """Evict least valuable entry from warm cache"""
        if not self._warm_cache:
            return
        
        # Use LRU eviction for warm cache
        evict_key, entry = self._warm_cache.popitem(last=False)
        self._warm_size_bytes -= entry.size_bytes
        
        # Clean up tag index
        for tag in entry.tags:
            self._tag_index[tag].discard(evict_key)
        
        # Write to disk if enabled and entry is valuable
        if self.config.enable_disk_cache and entry.access_count > 1:
            self._write_to_disk(evict_key, entry)
        
        self._stats['evictions'] += 1
    
    def _promote_entry(self, key: str, entry: CacheEntry):
        """Promote entry from warm to hot cache"""
        # Remove from warm cache
        self._remove_entry(key, CacheLevel.MEMORY_WARM)
        
        # Update cache level
        entry.cache_level = CacheLevel.MEMORY_HOT
        
        # Add to hot cache
        self._add_to_hot(key, entry)
        self._stats['promotions'] += 1
        
        logger.debug(f"Promoted entry to hot cache: {key}")
    
    def _demote_entry(self, key: str, entry: CacheEntry):
        """Demote entry from hot to warm cache"""
        # Remove from hot cache
        self._remove_entry(key, CacheLevel.MEMORY_HOT)
        
        # Update cache level
        entry.cache_level = CacheLevel.MEMORY_WARM
        
        # Add to warm cache
        self._add_to_warm(key, entry)
        self._stats['demotions'] += 1
        
        logger.debug(f"Demoted entry to warm cache: {key}")
    
    def _remove_entry(self, key: str, level: CacheLevel):
        """Remove entry from specific cache level"""
        if level == CacheLevel.MEMORY_HOT and key in self._hot_cache:
            entry = self._hot_cache.pop(key)
            self._hot_size_bytes -= entry.size_bytes
        elif level == CacheLevel.MEMORY_WARM and key in self._warm_cache:
            entry = self._warm_cache.pop(key)
            self._warm_size_bytes -= entry.size_bytes
        else:
            return
        
        # Clean up tag index
        for tag in entry.tags:
            self._tag_index[tag].discard(key)
    
    def _select_cache_level(self, size_bytes: int) -> CacheLevel:
        """Select appropriate cache level based on entry size"""
        if size_bytes < 1024 * 10:  # < 10KB -> hot
            return CacheLevel.MEMORY_HOT
        elif size_bytes < 1024 * 1024:  # < 1MB -> warm
            return CacheLevel.MEMORY_WARM
        else:  # Large entries -> disk
            return CacheLevel.DISK if self.config.enable_disk_cache else CacheLevel.MEMORY_WARM
    
    def _get_default_ttl(self, level: CacheLevel) -> int:
        """Get default TTL for cache level"""
        if level == CacheLevel.MEMORY_HOT:
            return self.config.hot_config.ttl_seconds
        elif level == CacheLevel.MEMORY_WARM:
            return self.config.warm_config.ttl_seconds
        else:
            return self.config.disk_config.ttl_seconds
    
    def _generate_search_key(
        self,
        query: str,
        strategy: str,
        project_ids: Optional[List[int]] = None
    ) -> str:
        """Generate cache key for search results"""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Create key components
        components = [
            "search",
            hashlib.md5(normalized_query.encode()).hexdigest()[:16],
            strategy
        ]
        
        # Add project context if provided
        if project_ids:
            project_hash = hashlib.md5(
                ",".join(map(str, sorted(project_ids))).encode()
            ).hexdigest()[:8]
            components.append(f"p{project_hash}")
        
        return ":".join(components)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            # Try to pickle and measure
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in value.items()
                )
            else:
                return 256  # Default size estimate
    
    def _init_disk_cache(self):
        """Initialize disk cache directory"""
        self._cache_dir = Path(self.config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load disk cache metadata
        self._disk_metadata_file = self._cache_dir / "metadata.json"
        self._disk_metadata = {}
        
        if self._disk_metadata_file.exists():
            try:
                with open(self._disk_metadata_file, 'r') as f:
                    self._disk_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load disk cache metadata: {e}")
    
    def _write_to_disk(self, key: str, entry: CacheEntry) -> bool:
        """Write entry to disk cache"""
        try:
            file_path = self._cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            
            # Serialize entry
            data = {
                'key': key,
                'value': entry.value,
                'created_at': entry.created_at.isoformat(),
                'accessed_at': entry.accessed_at.isoformat(),
                'access_count': entry.access_count,
                'ttl_seconds': entry.ttl_seconds,
                'tags': entry.tags
            }
            
            # Write to disk
            with open(file_path, 'wb') as f:
                if self.config.enable_compression:
                    import gzip
                    f.write(gzip.compress(pickle.dumps(data)))
                else:
                    pickle.dump(data, f)
            
            # Update metadata
            self._disk_metadata[key] = {
                'file': file_path.name,
                'size': file_path.stat().st_size,
                'created_at': datetime.now().isoformat()
            }
            
            self._disk_size_bytes += file_path.stat().st_size
            self._stats['disk_writes'] += 1
            
            # Save metadata periodically
            if self._stats['disk_writes'] % 10 == 0:
                self._save_disk_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")
            return False
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get entry from disk cache"""
        try:
            if key not in self._disk_metadata:
                return None
            
            file_path = self._cache_dir / self._disk_metadata[key]['file']
            if not file_path.exists():
                return None
            
            # Read from disk
            with open(file_path, 'rb') as f:
                if self.config.enable_compression:
                    import gzip
                    data = pickle.loads(gzip.decompress(f.read()))
                else:
                    data = pickle.load(f)
            
            # Check expiration
            created_at = datetime.fromisoformat(data['created_at'])
            age_seconds = (datetime.now() - created_at).total_seconds()
            if age_seconds > data['ttl_seconds']:
                # Remove expired entry
                self._remove_from_disk(key)
                return None
            
            return data['value']
            
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
            return None
    
    def _remove_from_disk(self, key: str) -> bool:
        """Remove entry from disk cache"""
        try:
            if key not in self._disk_metadata:
                return False
            
            file_path = self._cache_dir / self._disk_metadata[key]['file']
            if file_path.exists():
                self._disk_size_bytes -= file_path.stat().st_size
                file_path.unlink()
            
            del self._disk_metadata[key]
            return True
            
        except Exception as e:
            logger.error(f"Error removing from disk cache: {e}")
            return False
    
    def _clear_disk_cache(self):
        """Clear disk cache"""
        try:
            for file_path in self._cache_dir.glob("*.cache"):
                file_path.unlink()
            
            self._disk_metadata.clear()
            self._disk_size_bytes = 0
            self._save_disk_metadata()
            
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")
    
    def _save_disk_metadata(self):
        """Save disk cache metadata"""
        try:
            with open(self._disk_metadata_file, 'w') as f:
                json.dump(self._disk_metadata, f)
        except Exception as e:
            logger.error(f"Error saving disk metadata: {e}")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval_seconds)
                    self._cleanup_expired_entries()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.debug("Started cache cleanup thread")
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from all cache levels"""
        with self._lock:
            # Clean hot cache
            expired_keys = [
                key for key, entry in self._hot_cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove_entry(key, CacheLevel.MEMORY_HOT)
            
            # Clean warm cache
            expired_keys = [
                key for key, entry in self._warm_cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove_entry(key, CacheLevel.MEMORY_WARM)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats['total_requests']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'total_requests': total_requests,
                'evictions': self._stats['evictions'],
                'promotions': self._stats['promotions'],
                'demotions': self._stats['demotions'],
                'disk_hits': self._stats['disk_hits'],
                'disk_writes': self._stats['disk_writes'],
                'cache_levels': {
                    'hot': {
                        'entries': len(self._hot_cache),
                        'size_mb': self._hot_size_bytes / (1024 * 1024),
                        'capacity': f"{len(self._hot_cache)}/{self.config.hot_config.max_entries}"
                    },
                    'warm': {
                        'entries': len(self._warm_cache),
                        'size_mb': self._warm_size_bytes / (1024 * 1024),
                        'capacity': f"{len(self._warm_cache)}/{self.config.warm_config.max_entries}"
                    },
                    'disk': {
                        'entries': len(self._disk_metadata),
                        'size_mb': self._disk_size_bytes / (1024 * 1024),
                        'enabled': self.config.enable_disk_cache
                    }
                },
                'total_entries': len(self._hot_cache) + len(self._warm_cache) + len(self._disk_metadata),
                'size_mb': (self._hot_size_bytes + self._warm_size_bytes + self._disk_size_bytes) / (1024 * 1024)
            }
    
    def analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache performance and suggest optimizations"""
        stats = self.get_stats()
        
        analysis = {
            'performance_metrics': {
                'hit_rate': stats['hit_rate'],
                'promotion_rate': stats['promotions'] / max(stats['hits'], 1),
                'eviction_rate': stats['evictions'] / max(stats['total_requests'], 1)
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if stats['hit_rate'] < 0.6:
            analysis['recommendations'].append("Low hit rate - consider increasing cache size or TTL")
        
        if stats['evictions'] > stats['total_requests'] * 0.1:
            analysis['recommendations'].append("High eviction rate - increase cache capacity")
        
        if stats['promotions'] < stats['hits'] * 0.05:
            analysis['recommendations'].append("Low promotion rate - consider adjusting promotion threshold")
        
        # Check cache level utilization
        hot_usage = len(self._hot_cache) / self.config.hot_config.max_entries
        warm_usage = len(self._warm_cache) / self.config.warm_config.max_entries
        
        if hot_usage > 0.9:
            analysis['recommendations'].append("Hot cache near capacity - increase hot cache size")
        
        if warm_usage > 0.9:
            analysis['recommendations'].append("Warm cache near capacity - increase warm cache size")
        
        analysis['cache_efficiency'] = {
            'hot_utilization': hot_usage,
            'warm_utilization': warm_usage,
            'avg_entry_size_kb': stats['size_mb'] * 1024 / max(stats['total_entries'], 1)
        }
        
        return analysis


# Global cache instance
_cache_instance: Optional[MultiLevelSearchCache] = None
_cache_lock = threading.Lock()


def get_search_cache(config: Optional[MultiLevelCacheConfig] = None) -> MultiLevelSearchCache:
    """Get or create global search cache instance
    
    Args:
        config: Cache configuration (used only on first call)
        
    Returns:
        Global search cache instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = MultiLevelSearchCache(config)
                logger.info("Initialized global search cache")
    
    return _cache_instance


def create_search_cache(
    hot_max_entries: int = 100,
    warm_max_entries: int = 1000,
    enable_disk_cache: bool = True,
    cache_dir: str = "cache/search"
) -> MultiLevelSearchCache:
    """Factory function to create a configured search cache
    
    Args:
        hot_max_entries: Maximum entries in hot cache
        warm_max_entries: Maximum entries in warm cache
        enable_disk_cache: Whether to enable disk caching
        cache_dir: Directory for disk cache
        
    Returns:
        Configured search cache
    """
    config = MultiLevelCacheConfig(
        hot_config=CacheLevelConfig(
            max_entries=hot_max_entries,
            max_size_mb=10,
            ttl_seconds=300
        ),
        warm_config=CacheLevelConfig(
            max_entries=warm_max_entries,
            max_size_mb=100,
            ttl_seconds=3600
        ),
        enable_disk_cache=enable_disk_cache,
        cache_dir=cache_dir
    )
    
    return MultiLevelSearchCache(config)


if __name__ == "__main__":
    # Test multi-level cache functionality
    print("Testing multi-level search cache...")
    
    # Create cache instance
    cache = create_search_cache(
        hot_max_entries=10,
        warm_max_entries=50,
        enable_disk_cache=True
    )
    
    # Test basic operations
    print("\nTesting basic cache operations:")
    
    # Put some test data
    test_queries = [
        ("authentication setup", ["result1", "result2", "result3"]),
        ("database configuration", ["result4", "result5"]),
        ("error handling", ["result6", "result7", "result8", "result9"])
    ]
    
    for query, results in test_queries:
        success = cache.put_search_results(query, results, strategy="keyword")
        print(f"  Cached '{query}': {success}")
    
    # Test retrieval
    print("\nTesting cache retrieval:")
    for query, expected in test_queries:
        results = cache.get_search_results(query, strategy="keyword")
        hit = results is not None
        print(f"  Query '{query}': {'HIT' if hit else 'MISS'}")
        if hit:
            print(f"    Results: {len(results)} items")
    
    # Test cache promotion
    print("\nTesting cache promotion (accessing same query multiple times):")
    for _ in range(5):
        cache.get_search_results("authentication setup", strategy="keyword")
    
    # Show statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Promotions: {stats['promotions']}")
    print(f"  Evictions: {stats['evictions']}")
    
    print(f"\nCache Levels:")
    for level, info in stats['cache_levels'].items():
        print(f"  {level.capitalize()}:")
        print(f"    Entries: {info['entries']}")
        print(f"    Size: {info['size_mb']:.2f} MB")
        print(f"    Capacity: {info.get('capacity', 'N/A')}")
    
    # Test tag-based invalidation
    print("\nTesting tag-based invalidation:")
    cache.put("test_key_1", "value1", tags=["test_tag"])
    cache.put("test_key_2", "value2", tags=["test_tag"])
    cache.put("test_key_3", "value3", tags=["other_tag"])
    
    invalidated = cache.invalidate_by_tag("test_tag")
    print(f"  Invalidated {invalidated} entries with tag 'test_tag'")
    
    # Performance analysis
    analysis = cache.analyze_cache_performance()
    print(f"\nPerformance Analysis:")
    print(f"  Hit rate: {analysis['performance_metrics']['hit_rate']:.2%}")
    print(f"  Promotion rate: {analysis['performance_metrics']['promotion_rate']:.2%}")
    print(f"  Eviction rate: {analysis['performance_metrics']['eviction_rate']:.2%}")
    
    if analysis['recommendations']:
        print(f"\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
    
    print("\nMulti-level cache testing complete")