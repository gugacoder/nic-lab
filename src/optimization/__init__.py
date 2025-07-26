"""
Search Performance Optimization Package

This package provides advanced optimization features for the NIC Chat search system,
including parallel processing, cache warming, and performance monitoring.
"""

from .parallel_searcher import ParallelSearcher, create_parallel_searcher
from .cache_warmer import CacheWarmer, create_cache_warmer
from .search_metrics import SearchMetrics, get_search_metrics
from .query_optimizer import QueryOptimizer, create_query_optimizer

__all__ = [
    'ParallelSearcher',
    'create_parallel_searcher',
    'CacheWarmer', 
    'create_cache_warmer',
    'SearchMetrics',
    'get_search_metrics',
    'QueryOptimizer',
    'create_query_optimizer'
]