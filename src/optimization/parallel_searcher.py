"""
Parallel Search Implementation

This module provides parallel search capabilities across multiple GitLab projects
and search strategies, significantly improving search performance through concurrent
execution and optimized result aggregation.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from ..integrations.gitlab_client import GitLabSearchResult, get_gitlab_client
from ..integrations.search.keyword_search import KeywordSearchStrategy
from ..integrations.search.semantic_search import SemanticSearchStrategy
from ..integrations.search.aggregator import SearchResultAggregator
from ..integrations.cache.search_cache import get_search_cache

logger = logging.getLogger(__name__)


@dataclass
class SearchTask:
    """Represents a search task for parallel execution"""
    task_id: str
    strategy: str
    query: str
    project_ids: List[int]
    file_extensions: List[str]
    limit: int
    timeout: float = 10.0
    priority: int = 1  # Higher number = higher priority


@dataclass
class SearchResult:
    """Container for search results with metadata"""
    task_id: str
    strategy: str
    results: List[GitLabSearchResult]
    duration_ms: float
    success: bool
    error: Optional[str] = None
    cache_hit: bool = False


@dataclass
class ParallelSearchConfig:
    """Configuration for parallel search operations"""
    max_concurrent_searches: int = 5
    max_projects_per_batch: int = 10
    search_timeout: float = 10.0
    enable_result_caching: bool = True
    cache_ttl_seconds: int = 3600
    priority_boost_factor: float = 1.5
    failure_retry_attempts: int = 2
    batch_delay_ms: int = 50  # Delay between batches to prevent rate limiting


class ParallelSearcher:
    """Advanced parallel search implementation for GitLab content
    
    This class provides concurrent search capabilities across multiple projects
    and search strategies, with intelligent load balancing, caching, and error handling.
    """
    
    def __init__(self, config: Optional[ParallelSearchConfig] = None):
        """Initialize parallel searcher
        
        Args:
            config: Parallel search configuration
        """
        self.config = config or ParallelSearchConfig()
        self.gitlab_client = get_gitlab_client()
        self.search_cache = get_search_cache() if self.config.enable_result_caching else None
        
        # Initialize search strategies
        self.keyword_search = KeywordSearchStrategy(self.gitlab_client)
        self.semantic_search = SemanticSearchStrategy(self.gitlab_client)
        
        # Performance tracking
        self._stats = {
            'total_searches': 0,
            'parallel_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_duration_ms': 0.0,
            'success_rate': 0.0,
            'concurrent_peak': 0
        }
        
        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_searches,
            thread_name_prefix="parallel_search"
        )
    
    async def search_parallel(
        self,
        query: str,
        project_ids: Optional[List[int]] = None,
        strategies: Optional[List[str]] = None,
        file_extensions: Optional[List[str]] = None,
        max_results: int = 50
    ) -> Dict[str, List[GitLabSearchResult]]:
        """Execute parallel searches across multiple strategies and projects
        
        Args:
            query: Search query
            project_ids: List of project IDs to search (None for all accessible)
            strategies: Search strategies to use (['keyword', 'semantic'])
            file_extensions: File extensions to filter
            max_results: Maximum results per strategy
            
        Returns:
            Dictionary mapping strategy names to search results
        """
        start_time = time.time()
        
        try:
            # Prepare search configuration
            strategies = strategies or ['keyword', 'semantic']
            project_ids = project_ids or self._get_accessible_projects()
            file_extensions = file_extensions or []
            
            # Create search tasks
            search_tasks = self._create_search_tasks(
                query=query,
                project_ids=project_ids,
                strategies=strategies,
                file_extensions=file_extensions,
                max_results=max_results
            )
            
            logger.info(f"Executing {len(search_tasks)} parallel search tasks for query: '{query}'")
            
            # Execute searches in parallel
            search_results = await self._execute_parallel_searches(search_tasks)
            
            # Aggregate results by strategy
            aggregated_results = self._aggregate_results_by_strategy(search_results)
            
            # Update statistics
            duration_ms = (time.time() - start_time) * 1000
            self._update_stats(search_results, duration_ms)
            
            logger.info(f"Parallel search completed in {duration_ms:.1f}ms, found {sum(len(results) for results in aggregated_results.values())} total results")
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Error in parallel search: {e}")
            raise
    
    def search_projects_concurrent(
        self,
        query: str,
        project_ids: List[int],
        strategy: str = 'keyword',
        file_extensions: Optional[List[str]] = None,
        max_results_per_project: int = 10
    ) -> List[GitLabSearchResult]:
        """Search multiple projects concurrently using a single strategy
        
        Args:
            query: Search query
            project_ids: List of project IDs to search
            strategy: Search strategy to use
            file_extensions: File extensions to filter
            max_results_per_project: Maximum results per project
            
        Returns:
            Combined search results from all projects
        """
        start_time = time.time()
        
        # Split projects into batches to prevent overwhelming GitLab API
        project_batches = self._create_project_batches(project_ids)
        all_results = []
        
        for batch_idx, batch in enumerate(project_batches):
            logger.debug(f"Processing project batch {batch_idx + 1}/{len(project_batches)} with {len(batch)} projects")
            
            # Create tasks for this batch
            tasks = []
            for project_id in batch:
                task = SearchTask(
                    task_id=f"{strategy}_{project_id}_{batch_idx}",
                    strategy=strategy,
                    query=query,
                    project_ids=[project_id],
                    file_extensions=file_extensions or [],
                    limit=max_results_per_project,
                    timeout=self.config.search_timeout
                )
                tasks.append(task)
            
            # Execute batch concurrently
            batch_results = []
            with ThreadPoolExecutor(max_workers=min(len(batch), self.config.max_concurrent_searches)) as executor:
                future_to_task = {
                    executor.submit(self._execute_single_search, task): task 
                    for task in tasks
                }
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result(timeout=self.config.search_timeout)
                        if result.success:
                            batch_results.extend(result.results)
                        else:
                            logger.warning(f"Search failed for project {task.project_ids[0]}: {result.error}")
                    except Exception as e:
                        logger.error(f"Exception in search task {task.task_id}: {e}")
            
            all_results.extend(batch_results)
            
            # Add delay between batches to respect rate limits
            if batch_idx < len(project_batches) - 1:
                time.sleep(self.config.batch_delay_ms / 1000)
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Concurrent project search completed in {duration_ms:.1f}ms, found {len(all_results)} results across {len(project_ids)} projects")
        
        # Sort results by relevance (this is a simple implementation)
        all_results.sort(key=lambda r: len(r.content), reverse=True)
        
        return all_results
    
    def _create_search_tasks(
        self,
        query: str,
        project_ids: List[int],
        strategies: List[str],
        file_extensions: List[str],
        max_results: int
    ) -> List[SearchTask]:
        """Create search tasks for parallel execution
        
        Args:
            query: Search query
            project_ids: Project IDs to search
            strategies: Search strategies
            file_extensions: File extensions to filter
            max_results: Maximum results per strategy
            
        Returns:
            List of search tasks
        """
        tasks = []
        
        # Create project batches to prevent overwhelming individual projects
        project_batches = self._create_project_batches(project_ids)
        
        for strategy in strategies:
            for batch_idx, project_batch in enumerate(project_batches):
                task = SearchTask(
                    task_id=f"{strategy}_batch_{batch_idx}",
                    strategy=strategy,
                    query=query,
                    project_ids=project_batch,
                    file_extensions=file_extensions,
                    limit=max_results // len(strategies),  # Divide results among strategies
                    timeout=self.config.search_timeout,
                    priority=2 if strategy == 'semantic' else 1  # Semantic search gets higher priority
                )
                tasks.append(task)
        
        # Sort by priority (higher priority first)
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        return tasks
    
    def _create_project_batches(self, project_ids: List[int]) -> List[List[int]]:
        """Split project IDs into batches for parallel processing
        
        Args:
            project_ids: List of project IDs
            
        Returns:
            List of project ID batches
        """
        batch_size = self.config.max_projects_per_batch
        batches = []
        
        for i in range(0, len(project_ids), batch_size):
            batch = project_ids[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    async def _execute_parallel_searches(self, tasks: List[SearchTask]) -> List[SearchResult]:
        """Execute search tasks in parallel
        
        Args:
            tasks: List of search tasks to execute
            
        Returns:
            List of search results
        """
        results = []
        
        # Use asyncio to manage concurrent execution
        semaphore = asyncio.Semaphore(self.config.max_concurrent_searches)
        
        async def execute_with_semaphore(task: SearchTask) -> SearchResult:
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self._executor, self._execute_single_search, task)
        
        # Create coroutines for all tasks
        coroutines = [execute_with_semaphore(task) for task in tasks]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search task {tasks[i].task_id} failed with exception: {result}")
            else:
                valid_results.append(result)
        
        # Update concurrent peak
        self._stats['concurrent_peak'] = max(self._stats['concurrent_peak'], len(tasks))
        
        return valid_results
    
    def _execute_single_search(self, task: SearchTask) -> SearchResult:
        """Execute a single search task
        
        Args:
            task: Search task to execute
            
        Returns:
            Search result
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if self.search_cache:
                cache_key = f"{task.strategy}:{task.query}:{':'.join(map(str, task.project_ids))}"
                cached_results = self.search_cache.get(cache_key)
                
                if cached_results:
                    duration_ms = (time.time() - start_time) * 1000
                    return SearchResult(
                        task_id=task.task_id,
                        strategy=task.strategy,
                        results=cached_results,
                        duration_ms=duration_ms,
                        success=True,
                        cache_hit=True
                    )
            
            # Execute search based on strategy
            results = []
            if task.strategy == 'keyword':
                results = self._execute_keyword_search(task)
            elif task.strategy == 'semantic':
                results = self._execute_semantic_search(task)
            else:
                raise ValueError(f"Unknown search strategy: {task.strategy}")
            
            # Cache results if caching is enabled
            if self.search_cache and results:
                cache_key = f"{task.strategy}:{task.query}:{':'.join(map(str, task.project_ids))}"
                self.search_cache.put(
                    key=cache_key,
                    value=results,
                    ttl_seconds=self.config.cache_ttl_seconds,
                    tags=[f"strategy:{task.strategy}", "parallel_search"]
                )
            
            duration_ms = (time.time() - start_time) * 1000
            
            return SearchResult(
                task_id=task.task_id,
                strategy=task.strategy,
                results=results,
                duration_ms=duration_ms,
                success=True,
                cache_hit=False
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Search task {task.task_id} failed: {e}")
            
            return SearchResult(
                task_id=task.task_id,
                strategy=task.strategy,
                results=[],
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )
    
    def _execute_keyword_search(self, task: SearchTask) -> List[GitLabSearchResult]:
        """Execute keyword search for a task
        
        Args:
            task: Search task
            
        Returns:
            List of search results
        """
        return self.gitlab_client.search_files(
            query=task.query,
            project_ids=task.project_ids,
            file_extensions=task.file_extensions,
            limit=task.limit
        )
    
    def _execute_semantic_search(self, task: SearchTask) -> List[GitLabSearchResult]:
        """Execute semantic search for a task
        
        Args:
            task: Search task
            
        Returns:
            List of search results
        """
        # For now, delegate to semantic search strategy
        # In a full implementation, this would use embeddings and vector search
        return self.semantic_search.search(
            query=task.query,
            project_ids=task.project_ids,
            file_extensions=task.file_extensions,
            limit=task.limit
        )
    
    def _aggregate_results_by_strategy(self, search_results: List[SearchResult]) -> Dict[str, List[GitLabSearchResult]]:
        """Aggregate search results by strategy
        
        Args:
            search_results: List of search results
            
        Returns:
            Dictionary mapping strategy to results
        """
        aggregated = defaultdict(list)
        
        for result in search_results:
            if result.success:
                aggregated[result.strategy].extend(result.results)
        
        # Remove duplicates within each strategy
        for strategy in aggregated:
            seen_sources = set()
            deduplicated = []
            
            for result in aggregated[strategy]:
                source_key = f"{result.project_id}:{result.file_path}"
                if source_key not in seen_sources:
                    deduplicated.append(result)
                    seen_sources.add(source_key)
            
            aggregated[strategy] = deduplicated
        
        return dict(aggregated)
    
    def _get_accessible_projects(self) -> List[int]:
        """Get list of accessible project IDs
        
        Returns:
            List of project IDs that the user can access
        """
        try:
            projects = self.gitlab_client.get_accessible_projects()
            return [p.id for p in projects]
        except Exception as e:
            logger.warning(f"Could not get accessible projects: {e}")
            return []
    
    def _update_stats(self, search_results: List[SearchResult], duration_ms: float):
        """Update performance statistics
        
        Args:
            search_results: List of search results
            duration_ms: Total duration in milliseconds
        """
        self._stats['total_searches'] += 1
        self._stats['parallel_searches'] += len(search_results)
        
        # Update cache statistics
        cache_hits = sum(1 for r in search_results if r.cache_hit)
        cache_misses = len(search_results) - cache_hits
        
        self._stats['cache_hits'] += cache_hits
        self._stats['cache_misses'] += cache_misses
        
        # Update duration (moving average)
        current_avg = self._stats['avg_duration_ms']
        total_searches = self._stats['total_searches']
        self._stats['avg_duration_ms'] = ((current_avg * (total_searches - 1)) + duration_ms) / total_searches
        
        # Update success rate
        successful_searches = sum(1 for r in search_results if r.success)
        total_attempted = len(search_results)
        if total_attempted > 0:
            current_success_rate = successful_searches / total_attempted
            self._stats['success_rate'] = (
                (self._stats['success_rate'] * (total_searches - 1) + current_success_rate) / total_searches
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics
        
        Returns:
            Dictionary of performance statistics
        """
        total_cache_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        cache_hit_rate = (
            self._stats['cache_hits'] / total_cache_requests 
            if total_cache_requests > 0 else 0.0
        )
        
        return {
            'total_searches': self._stats['total_searches'],
            'parallel_searches': self._stats['parallel_searches'],
            'avg_duration_ms': self._stats['avg_duration_ms'],
            'success_rate': self._stats['success_rate'],
            'concurrent_peak': self._stats['concurrent_peak'],
            'cache_performance': {
                'hit_rate': cache_hit_rate,
                'hits': self._stats['cache_hits'],
                'misses': self._stats['cache_misses']
            },
            'config': {
                'max_concurrent_searches': self.config.max_concurrent_searches,
                'max_projects_per_batch': self.config.max_projects_per_batch,
                'search_timeout': self.config.search_timeout
            }
        }
    
    def optimize_search_parameters(self, query: str, project_ids: List[int]) -> Dict[str, Any]:
        """Analyze query and suggest optimal search parameters
        
        Args:
            query: Search query to analyze
            project_ids: Project IDs to search
            
        Returns:
            Optimization suggestions
        """
        suggestions = {
            'recommended_strategies': [],
            'optimal_batch_size': self.config.max_projects_per_batch,
            'estimated_duration_ms': 0.0,
            'cache_likelihood': 0.0
        }
        
        # Analyze query characteristics
        query_lower = query.lower()
        
        # Suggest strategies based on query type
        if any(indicator in query_lower for indicator in ['how to', 'what is', 'explain', 'concept']):
            suggestions['recommended_strategies'] = ['semantic', 'keyword']
        elif any(char in query for char in ['()', '{', '}', '=', '<', '>']):
            suggestions['recommended_strategies'] = ['keyword', 'semantic']
        else:
            suggestions['recommended_strategies'] = ['keyword', 'semantic']
        
        # Estimate optimal batch size based on project count
        project_count = len(project_ids)
        if project_count > 50:
            suggestions['optimal_batch_size'] = min(20, self.config.max_projects_per_batch)
        elif project_count > 20:
            suggestions['optimal_batch_size'] = min(15, self.config.max_projects_per_batch)
        else:
            suggestions['optimal_batch_size'] = min(10, self.config.max_projects_per_batch)
        
        # Estimate duration based on project count and query complexity
        base_duration = 1000  # 1 second base
        project_factor = project_count * 50  # 50ms per project
        complexity_factor = len(query.split()) * 10  # 10ms per word
        
        suggestions['estimated_duration_ms'] = base_duration + project_factor + complexity_factor
        
        # Estimate cache likelihood (simplified heuristic)
        if self.search_cache:
            cache_stats = self.search_cache.get_stats()
            suggestions['cache_likelihood'] = cache_stats.get('hit_rate', 0.0)
        
        return suggestions
    
    def benchmark_parallel_performance(
        self,
        test_queries: List[str],
        project_ids: List[int],
        strategies: List[str] = None
    ) -> Dict[str, Any]:
        """Benchmark parallel search performance
        
        Args:
            test_queries: List of test queries
            project_ids: Project IDs to test with
            strategies: Search strategies to benchmark
            
        Returns:
            Benchmark results
        """
        strategies = strategies or ['keyword', 'semantic']
        results = {
            'sequential_times': [],
            'parallel_times': [],
            'speedup_factors': [],
            'result_counts': [],
            'queries_tested': len(test_queries)
        }
        
        for query in test_queries:
            logger.info(f"Benchmarking query: '{query}'")
            
            # Test sequential search (single strategy at a time)
            sequential_start = time.time()
            sequential_results = {}
            for strategy in strategies:
                if strategy == 'keyword':
                    strategy_results = self.gitlab_client.search_files(
                        query=query,
                        project_ids=project_ids[:5],  # Limit for benchmarking
                        limit=10
                    )
                else:
                    strategy_results = self.semantic_search.search(
                        query=query,
                        project_ids=project_ids[:5],
                        limit=10
                    )
                sequential_results[strategy] = strategy_results
            sequential_time = (time.time() - sequential_start) * 1000
            
            # Test parallel search
            parallel_start = time.time()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                parallel_results = loop.run_until_complete(
                    self.search_parallel(
                        query=query,
                        project_ids=project_ids[:5],
                        strategies=strategies,
                        max_results=20
                    )
                )
            finally:
                loop.close()
            parallel_time = (time.time() - parallel_start) * 1000
            
            # Calculate metrics
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            total_results = sum(len(results) for results in parallel_results.values())
            
            results['sequential_times'].append(sequential_time)
            results['parallel_times'].append(parallel_time)
            results['speedup_factors'].append(speedup)
            results['result_counts'].append(total_results)
        
        # Calculate summary statistics
        if results['sequential_times']:
            results['avg_sequential_time'] = sum(results['sequential_times']) / len(results['sequential_times'])
            results['avg_parallel_time'] = sum(results['parallel_times']) / len(results['parallel_times'])
            results['avg_speedup'] = sum(results['speedup_factors']) / len(results['speedup_factors'])
            results['avg_results'] = sum(results['result_counts']) / len(results['result_counts'])
        
        return results
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)


def create_parallel_searcher(
    max_concurrent_searches: int = 5,
    max_projects_per_batch: int = 10,
    search_timeout: float = 10.0,
    enable_result_caching: bool = True,
    cache_ttl_seconds: int = 3600
) -> ParallelSearcher:
    """Factory function to create a configured parallel searcher
    
    Args:
        max_concurrent_searches: Maximum concurrent search operations
        max_projects_per_batch: Maximum projects per batch
        search_timeout: Timeout for individual searches
        enable_result_caching: Whether to enable result caching
        cache_ttl_seconds: Cache TTL in seconds
        
    Returns:
        Configured parallel searcher
    """
    config = ParallelSearchConfig(
        max_concurrent_searches=max_concurrent_searches,
        max_projects_per_batch=max_projects_per_batch,
        search_timeout=search_timeout,
        enable_result_caching=enable_result_caching,
        cache_ttl_seconds=cache_ttl_seconds
    )
    
    return ParallelSearcher(config)


if __name__ == "__main__":
    # Test parallel search functionality
    import sys
    
    async def test_parallel_search():
        print("Testing parallel search system...")
        
        # Create parallel searcher
        searcher = create_parallel_searcher(
            max_concurrent_searches=3,
            max_projects_per_batch=5
        )
        
        # Test queries
        test_queries = [
            "authentication setup",
            "database configuration", 
            "error handling"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            
            try:
                results = await searcher.search_parallel(
                    query=query,
                    strategies=['keyword'],
                    max_results=10
                )
                
                total_results = sum(len(strategy_results) for strategy_results in results.values())
                print(f"  Found {total_results} results across {len(results)} strategies")
                
                for strategy, strategy_results in results.items():
                    print(f"  {strategy}: {len(strategy_results)} results")
                    if strategy_results:
                        print(f"    Sample: {strategy_results[0].file_path}")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        # Show performance stats
        stats = searcher.get_performance_stats()
        print(f"\nPerformance Statistics:")
        print(f"  Total searches: {stats['total_searches']}")
        print(f"  Average duration: {stats['avg_duration_ms']:.1f}ms")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Cache hit rate: {stats['cache_performance']['hit_rate']:.2%}")
        
        print("\nParallel search testing complete")
    
    # Run async test
    asyncio.run(test_parallel_search())