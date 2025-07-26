"""
Cache Warming System

This module provides proactive cache population strategies to improve search
performance by pre-loading frequently accessed content and common query results.
"""

import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import threading

from ..integrations.gitlab_client import get_gitlab_client
from ..integrations.cache.search_cache import get_search_cache
from ..ai.retrievers.gitlab_retriever import create_gitlab_retriever, SearchMode
from .parallel_searcher import create_parallel_searcher
from .query_optimizer import create_query_optimizer

logger = logging.getLogger(__name__)


@dataclass
class WarmingTask:
    """Represents a cache warming task"""
    task_id: str
    task_type: str  # 'query', 'project', 'popular_content'
    priority: int  # Higher number = higher priority
    data: Dict[str, Any]
    scheduled_time: datetime
    estimated_duration_seconds: float = 60.0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WarmingStats:
    """Statistics for cache warming operations"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_items_warmed: int = 0
    total_duration_seconds: float = 0.0
    cache_hit_improvement: float = 0.0
    last_warming_time: Optional[datetime] = None


@dataclass
class CacheWarmerConfig:
    """Configuration for cache warming operations"""
    
    # Warming schedule
    enable_automatic_warming: bool = True
    warming_interval_hours: int = 6
    warming_window_hours: int = 2  # Time window to complete warming
    
    # Query-based warming
    enable_query_warming: bool = True
    popular_query_threshold: int = 5  # Minimum usage count to be considered popular
    max_queries_to_warm: int = 50
    query_warming_timeout: int = 30
    
    # Project-based warming
    enable_project_warming: bool = True
    max_projects_to_warm: int = 20
    project_warming_sample_size: int = 10  # Sample queries per project
    
    # Content-based warming
    enable_content_warming: bool = True
    recent_content_days: int = 7
    popular_content_threshold: int = 3
    
    # Performance limits
    max_concurrent_tasks: int = 3
    warming_rate_limit_per_minute: int = 30
    cache_warming_ttl_multiplier: float = 2.0  # Extend TTL for warmed items
    
    # Storage
    stats_file_path: str = "cache/warming_stats.json"
    query_log_file_path: str = "cache/query_log.json"


class CacheWarmer:
    """Advanced cache warming system for proactive performance optimization
    
    This class implements intelligent cache warming strategies based on usage patterns,
    popular queries, and content access frequencies to improve search performance.
    """
    
    def __init__(self, config: Optional[CacheWarmerConfig] = None):
        """Initialize cache warmer
        
        Args:
            config: Cache warming configuration
        """
        self.config = config or CacheWarmerConfig()
        
        # Core components
        self.gitlab_client = get_gitlab_client()
        self.search_cache = get_search_cache()
        self.retriever = create_gitlab_retriever(enable_caching=True)
        self.parallel_searcher = create_parallel_searcher()
        self.query_optimizer = create_query_optimizer()
        
        # Warming state
        self.warming_tasks: List[WarmingTask] = []
        self.stats = WarmingStats()
        self.is_warming = False
        self.warming_thread: Optional[threading.Thread] = None
        
        # Query tracking
        self.query_log: Dict[str, Dict[str, Any]] = {}
        self.popular_queries: List[str] = []
        self.popular_projects: List[int] = []
        
        # Load existing data
        self._load_stats()
        self._load_query_log()
        
        # Start automatic warming if enabled
        if self.config.enable_automatic_warming:
            self._start_automatic_warming()
    
    async def warm_cache_comprehensive(
        self,
        strategies: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Perform comprehensive cache warming using multiple strategies
        
        Args:
            strategies: List of warming strategies to use
            force_refresh: Whether to force refresh existing cache entries
            
        Returns:
            Warming results and statistics
        """
        start_time = datetime.now()
        strategies = strategies or ['popular_queries', 'recent_projects', 'trending_content']
        
        logger.info(f"Starting comprehensive cache warming with strategies: {strategies}")
        
        try:
            self.is_warming = True
            results = {
                'strategies_executed': [],
                'items_warmed': 0,
                'cache_entries_created': 0,
                'errors': [],
                'duration_seconds': 0.0
            }
            
            # Execute warming strategies
            if 'popular_queries' in strategies:
                query_results = await self._warm_popular_queries(force_refresh)
                results['strategies_executed'].append('popular_queries')
                results['items_warmed'] += query_results['queries_warmed']
                results['cache_entries_created'] += query_results['cache_entries']
                if 'errors' in query_results:
                    results['errors'].extend(query_results['errors'])
            
            if 'recent_projects' in strategies:
                project_results = await self._warm_recent_projects(force_refresh)
                results['strategies_executed'].append('recent_projects')
                results['items_warmed'] += project_results['projects_warmed']
                results['cache_entries_created'] += project_results['cache_entries']
                if 'errors' in project_results:
                    results['errors'].extend(project_results['errors'])
            
            if 'trending_content' in strategies:
                content_results = await self._warm_trending_content(force_refresh)
                results['strategies_executed'].append('trending_content')
                results['items_warmed'] += content_results['content_warmed']
                results['cache_entries_created'] += content_results['cache_entries']
                if 'errors' in content_results:
                    results['errors'].extend(content_results['errors'])
            
            # Update statistics
            duration = (datetime.now() - start_time).total_seconds()
            results['duration_seconds'] = duration
            
            self.stats.tasks_completed += 1
            self.stats.total_items_warmed += results['items_warmed']
            self.stats.total_duration_seconds += duration
            self.stats.last_warming_time = datetime.now()
            
            # Save updated stats
            self._save_stats()
            
            logger.info(f"Cache warming completed in {duration:.1f}s, warmed {results['items_warmed']} items")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive cache warming: {e}")
            self.stats.tasks_failed += 1
            raise
        finally:
            self.is_warming = False
    
    async def warm_popular_queries(
        self,
        query_limit: Optional[int] = None,
        min_usage_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Warm cache for popular queries based on usage history
        
        Args:
            query_limit: Maximum number of queries to warm
            min_usage_count: Minimum usage count to consider query popular
            
        Returns:
            Warming results
        """
        query_limit = query_limit or self.config.max_queries_to_warm
        min_usage_count = min_usage_count or self.config.popular_query_threshold
        
        # Identify popular queries
        popular_queries = self._identify_popular_queries(min_usage_count, query_limit)
        
        if not popular_queries:
            logger.info("No popular queries found for warming")
            return {'queries_warmed': 0, 'cache_entries': 0}
        
        return await self._warm_popular_queries(force_refresh=False)
    
    async def warm_project_content(
        self,
        project_ids: List[int],
        sample_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Warm cache for specific project content
        
        Args:
            project_ids: List of project IDs to warm
            sample_queries: Sample queries to use for warming
            
        Returns:
            Warming results
        """
        start_time = time.time()
        results = {
            'projects_warmed': 0,
            'cache_entries': 0,
            'errors': []
        }
        
        # Use default sample queries if none provided
        if not sample_queries:
            sample_queries = [
                "configuration setup",
                "installation guide",
                "api documentation",
                "error handling",
                "authentication",
                "database setup",
                "deployment",
                "testing",
                "security",
                "performance"
            ]
        
        try:
            # Warm each project
            for project_id in project_ids:
                try:
                    logger.debug(f"Warming cache for project {project_id}")
                    
                    project_cache_entries = 0
                    
                    # Execute sample queries for this project
                    for query in sample_queries[:self.config.project_warming_sample_size]:
                        try:
                            # Use parallel searcher for efficiency
                            search_results = await self.parallel_searcher.search_parallel(
                                query=query,
                                project_ids=[project_id],
                                strategies=['keyword', 'semantic'],
                                max_results=10
                            )
                            
                            if search_results:
                                project_cache_entries += sum(len(results) for results in search_results.values())
                        
                        except Exception as e:
                            logger.warning(f"Error warming query '{query}' for project {project_id}: {e}")
                            results['errors'].append(f"Project {project_id}, query '{query}': {e}")
                    
                    results['cache_entries'] += project_cache_entries
                    results['projects_warmed'] += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)  # Small delay between projects
                
                except Exception as e:
                    logger.error(f"Error warming project {project_id}: {e}")
                    results['errors'].append(f"Project {project_id}: {e}")
            
            duration = time.time() - start_time
            logger.info(f"Project warming completed in {duration:.1f}s for {results['projects_warmed']} projects")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in project content warming: {e}")
            results['errors'].append(str(e))
            return results
    
    def schedule_warming_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        priority: int = 1,
        scheduled_time: Optional[datetime] = None
    ) -> str:
        """Schedule a cache warming task
        
        Args:
            task_type: Type of warming task
            data: Task data
            priority: Task priority (higher = more important)
            scheduled_time: When to execute the task
            
        Returns:
            Task ID
        """
        task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        scheduled_time = scheduled_time or datetime.now()
        
        task = WarmingTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data,
            scheduled_time=scheduled_time
        )
        
        self.warming_tasks.append(task)
        
        # Sort by priority and scheduled time
        self.warming_tasks.sort(
            key=lambda t: (-t.priority, t.scheduled_time)
        )
        
        logger.info(f"Scheduled warming task {task_id} for {scheduled_time}")
        return task_id
    
    async def _warm_popular_queries(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Warm cache for popular queries"""
        results = {
            'queries_warmed': 0,
            'cache_entries': 0,
            'errors': []
        }
        
        # Get popular queries
        popular_queries = self._identify_popular_queries(
            self.config.popular_query_threshold,
            self.config.max_queries_to_warm
        )
        
        if not popular_queries:
            return results
        
        logger.info(f"Warming {len(popular_queries)} popular queries")
        
        # Process in batches to avoid overwhelming the system
        batch_size = 5
        for i in range(0, len(popular_queries), batch_size):
            batch = popular_queries[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = []
            for query in batch:
                task = self._warm_single_query(query, force_refresh)
                batch_tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results['errors'].append(f"Query '{batch[j]}': {result}")
                else:
                    results['queries_warmed'] += 1
                    results['cache_entries'] += result.get('cache_entries', 0)
            
            # Rate limiting between batches
            if i + batch_size < len(popular_queries):
                await asyncio.sleep(1)
        
        return results
    
    async def _warm_recent_projects(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Warm cache for recently active projects"""
        results = {
            'projects_warmed': 0,
            'cache_entries': 0,
            'errors': []
        }
        
        try:
            # Get recently active projects
            recent_projects = self._get_recent_projects(self.config.max_projects_to_warm)
            
            if not recent_projects:
                return results
            
            # Warm project content
            project_results = await self.warm_project_content(recent_projects)
            
            results['projects_warmed'] = project_results['projects_warmed']
            results['cache_entries'] = project_results['cache_entries']
            results['errors'] = project_results.get('errors', [])
            
            return results
            
        except Exception as e:
            results['errors'].append(str(e))
            return results
    
    async def _warm_trending_content(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Warm cache for trending content"""
        results = {
            'content_warmed': 0,
            'cache_entries': 0,
            'errors': []
        }
        
        try:
            # For now, use a simplified approach - warm common technical terms
            trending_queries = [
                "getting started",
                "installation",
                "configuration",
                "troubleshooting",
                "best practices",
                "examples",
                "tutorial",
                "api reference",
                "deployment guide",
                "security setup"
            ]
            
            # Execute trending queries
            for query in trending_queries:
                try:
                    search_results = await self.parallel_searcher.search_parallel(
                        query=query,
                        strategies=['semantic', 'keyword'],
                        max_results=15
                    )
                    
                    if search_results:
                        results['cache_entries'] += sum(len(results_list) for results_list in search_results.values())
                        results['content_warmed'] += 1
                
                except Exception as e:
                    results['errors'].append(f"Trending query '{query}': {e}")
            
            return results
            
        except Exception as e:
            results['errors'].append(str(e))
            return results
    
    async def _warm_single_query(self, query: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Warm cache for a single query"""
        try:
            # Check if already cached (unless forcing refresh)
            if not force_refresh:
                cached_results = self.search_cache.get_search_results(
                    query=query,
                    strategy="hybrid"
                )
                if cached_results:
                    return {'cache_entries': 0, 'was_cached': True}
            
            # Execute search to populate cache
            search_results = await self.parallel_searcher.search_parallel(
                query=query,
                strategies=['keyword', 'semantic'],
                max_results=20
            )
            
            cache_entries = sum(len(results) for results in search_results.values())
            
            return {
                'cache_entries': cache_entries,
                'was_cached': False
            }
            
        except Exception as e:
            logger.error(f"Error warming query '{query}': {e}")
            raise
    
    def _identify_popular_queries(self, min_usage_count: int, limit: int) -> List[str]:
        """Identify popular queries from usage history"""
        if not self.query_log:
            return []
        
        # Count query usage
        query_counts = Counter()
        for query_hash, query_data in self.query_log.items():
            usage_count = query_data.get('usage_count', 0)
            if usage_count >= min_usage_count:
                query_counts[query_data['query']] = usage_count
        
        # Return top queries
        popular = [query for query, count in query_counts.most_common(limit)]
        logger.debug(f"Identified {len(popular)} popular queries")
        
        return popular
    
    def _get_recent_projects(self, limit: int) -> List[int]:
        """Get recently active project IDs"""
        try:
            # This is a simplified implementation
            # In a real system, this would query GitLab for recently updated projects
            projects = self.gitlab_client.get_accessible_projects()
            
            # Sort by last activity and limit
            sorted_projects = sorted(
                projects,
                key=lambda p: getattr(p, 'last_activity_at', datetime.min),
                reverse=True
            )
            
            return [p.id for p in sorted_projects[:limit]]
            
        except Exception as e:
            logger.error(f"Error getting recent projects: {e}")
            return []
    
    def log_query_usage(self, query: str, project_ids: Optional[List[int]] = None):
        """Log query usage for warming analysis
        
        Args:
            query: Query that was executed
            project_ids: Projects that were searched
        """
        query_hash = self._hash_query(query)
        
        if query_hash in self.query_log:
            self.query_log[query_hash]['usage_count'] += 1
            self.query_log[query_hash]['last_used'] = datetime.now().isoformat()
        else:
            self.query_log[query_hash] = {
                'query': query,
                'usage_count': 1,
                'first_used': datetime.now().isoformat(),
                'last_used': datetime.now().isoformat(),
                'project_ids': project_ids or []
            }
        
        # Periodically save query log
        if len(self.query_log) % 10 == 0:
            self._save_query_log()
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query tracking"""
        import hashlib
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _start_automatic_warming(self):
        """Start automatic cache warming thread"""
        def warming_worker():
            while True:
                try:
                    # Calculate next warming time
                    interval_seconds = self.config.warming_interval_hours * 3600
                    time.sleep(interval_seconds)
                    
                    # Check if it's time to warm (avoid warming during peak hours)
                    current_hour = datetime.now().hour
                    if 8 <= current_hour <= 18:  # Skip during business hours
                        continue
                    
                    # Perform automatic warming
                    logger.info("Starting automatic cache warming")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        loop.run_until_complete(
                            self.warm_cache_comprehensive(force_refresh=False)
                        )
                    finally:
                        loop.close()
                    
                except Exception as e:
                    logger.error(f"Error in automatic warming: {e}")
        
        self.warming_thread = threading.Thread(target=warming_worker, daemon=True)
        self.warming_thread.start()
        logger.info("Started automatic cache warming thread")
    
    def _load_stats(self):
        """Load warming statistics from file"""
        try:
            stats_file = Path(self.config.stats_file_path)
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                    
                self.stats = WarmingStats(
                    tasks_completed=data.get('tasks_completed', 0),
                    tasks_failed=data.get('tasks_failed', 0),
                    total_items_warmed=data.get('total_items_warmed', 0),
                    total_duration_seconds=data.get('total_duration_seconds', 0.0),
                    cache_hit_improvement=data.get('cache_hit_improvement', 0.0),
                    last_warming_time=datetime.fromisoformat(data['last_warming_time']) if data.get('last_warming_time') else None
                )
        except Exception as e:
            logger.warning(f"Could not load warming stats: {e}")
    
    def _save_stats(self):
        """Save warming statistics to file"""
        try:
            stats_file = Path(self.config.stats_file_path)
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'tasks_completed': self.stats.tasks_completed,
                'tasks_failed': self.stats.tasks_failed,
                'total_items_warmed': self.stats.total_items_warmed,
                'total_duration_seconds': self.stats.total_duration_seconds,
                'cache_hit_improvement': self.stats.cache_hit_improvement,
                'last_warming_time': self.stats.last_warming_time.isoformat() if self.stats.last_warming_time else None
            }
            
            with open(stats_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving warming stats: {e}")
    
    def _load_query_log(self):
        """Load query usage log from file"""
        try:
            log_file = Path(self.config.query_log_file_path)
            if log_file.exists():
                with open(log_file, 'r') as f:
                    self.query_log = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load query log: {e}")
    
    def _save_query_log(self):
        """Save query usage log to file"""
        try:
            log_file = Path(self.config.query_log_file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, 'w') as f:
                json.dump(self.query_log, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving query log: {e}")
    
    def get_warming_stats(self) -> Dict[str, Any]:
        """Get cache warming statistics"""
        total_tasks = self.stats.tasks_completed + self.stats.tasks_failed
        success_rate = self.stats.tasks_completed / total_tasks if total_tasks > 0 else 0.0
        
        return {
            'tasks_completed': self.stats.tasks_completed,
            'tasks_failed': self.stats.tasks_failed,
            'success_rate': success_rate,
            'total_items_warmed': self.stats.total_items_warmed,
            'avg_duration_seconds': self.stats.total_duration_seconds / max(self.stats.tasks_completed, 1),
            'cache_hit_improvement': self.stats.cache_hit_improvement,
            'last_warming_time': self.stats.last_warming_time.isoformat() if self.stats.last_warming_time else None,
            'popular_queries_count': len(self._identify_popular_queries(self.config.popular_query_threshold, 100)),
            'is_currently_warming': self.is_warming,
            'automatic_warming_enabled': self.config.enable_automatic_warming
        }
    
    def analyze_warming_effectiveness(self) -> Dict[str, Any]:
        """Analyze the effectiveness of cache warming"""
        cache_stats = self.search_cache.get_stats()
        
        analysis = {
            'current_cache_hit_rate': cache_stats['hit_rate'],
            'cache_size_mb': cache_stats['size_mb'],
            'total_cache_entries': cache_stats['total_entries'],
            'warming_contribution': 'unknown'  # Would require before/after comparison
        }
        
        # Analyze query patterns
        if self.query_log:
            total_queries = len(self.query_log)
            popular_queries = len(self._identify_popular_queries(self.config.popular_query_threshold, 1000))
            
            analysis['query_analysis'] = {
                'total_unique_queries': total_queries,
                'popular_queries': popular_queries,
                'popular_query_ratio': popular_queries / total_queries if total_queries > 0 else 0.0
            }
        
        return analysis


def create_cache_warmer(
    enable_automatic_warming: bool = True,
    warming_interval_hours: int = 6,
    max_queries_to_warm: int = 50,
    max_projects_to_warm: int = 20
) -> CacheWarmer:
    """Factory function to create a configured cache warmer
    
    Args:
        enable_automatic_warming: Whether to enable automatic warming
        warming_interval_hours: Hours between automatic warming cycles
        max_queries_to_warm: Maximum queries to warm per cycle
        max_projects_to_warm: Maximum projects to warm per cycle
        
    Returns:
        Configured cache warmer
    """
    config = CacheWarmerConfig(
        enable_automatic_warming=enable_automatic_warming,
        warming_interval_hours=warming_interval_hours,
        max_queries_to_warm=max_queries_to_warm,
        max_projects_to_warm=max_projects_to_warm
    )
    
    return CacheWarmer(config)


if __name__ == "__main__":
    # Test cache warming system
    import sys
    
    async def test_cache_warming():
        print("Testing cache warming system...")
        
        # Create cache warmer
        warmer = create_cache_warmer(
            enable_automatic_warming=False,  # Disable for testing
            max_queries_to_warm=5,
            max_projects_to_warm=3
        )
        
        # Log some test queries
        test_queries = [
            "authentication setup",
            "database configuration",
            "error handling",
            "authentication setup",  # Duplicate to increase usage count
            "deployment guide"
        ]
        
        print("Logging test queries...")
        for query in test_queries:
            warmer.log_query_usage(query, [1, 2, 3])
        
        # Test comprehensive warming
        print("\nPerforming comprehensive cache warming...")
        try:
            results = await warmer.warm_cache_comprehensive(
                strategies=['popular_queries'],
                force_refresh=True
            )
            
            print(f"Warming Results:")
            print(f"  Strategies executed: {results['strategies_executed']}")
            print(f"  Items warmed: {results['items_warmed']}")
            print(f"  Cache entries created: {results['cache_entries_created']}")
            print(f"  Duration: {results['duration_seconds']:.1f}s")
            
            if results['errors']:
                print(f"  Errors: {len(results['errors'])}")
                for error in results['errors'][:3]:  # Show first 3 errors
                    print(f"    - {error}")
        
        except Exception as e:
            print(f"  Error: {e}")
        
        # Show warming statistics
        stats = warmer.get_warming_stats()
        print(f"\nWarming Statistics:")
        print(f"  Tasks completed: {stats['tasks_completed']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Items warmed: {stats['total_items_warmed']}")
        print(f"  Popular queries: {stats['popular_queries_count']}")
        
        # Analyze effectiveness
        effectiveness = warmer.analyze_warming_effectiveness()
        print(f"\nEffectiveness Analysis:")
        print(f"  Cache hit rate: {effectiveness['current_cache_hit_rate']:.2%}")
        print(f"  Cache size: {effectiveness['cache_size_mb']:.1f} MB")
        print(f"  Cache entries: {effectiveness['total_cache_entries']}")
        
        if 'query_analysis' in effectiveness:
            qa = effectiveness['query_analysis']
            print(f"  Total unique queries: {qa['total_unique_queries']}")
            print(f"  Popular queries: {qa['popular_queries']}")
            print(f"  Popular ratio: {qa['popular_query_ratio']:.2%}")
        
        print("\nCache warming testing complete")
    
    # Run async test
    asyncio.run(test_cache_warming())