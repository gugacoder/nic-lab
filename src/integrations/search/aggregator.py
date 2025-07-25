"""
Search Result Aggregation and Ranking

This module combines results from multiple search strategies (keyword, semantic, etc.)
and provides intelligent ranking, deduplication, and result optimization.
"""

import logging
import hashlib
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum

from ..gitlab_client import GitLabSearchResult
from .keyword_search import KeywordSearchStrategy
from .semantic_search import SemanticSearchStrategy

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Available search strategies"""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    FUZZY = "fuzzy"
    HYBRID = "hybrid"


@dataclass
class ScoredResult:
    """Search result with aggregated scoring information"""
    result: GitLabSearchResult
    scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    strategies: Set[str] = field(default_factory=set)
    rank_factors: Dict[str, float] = field(default_factory=dict)
    
    @property
    def unique_key(self) -> str:
        """Generate unique key for deduplication"""
        return f"{self.result.project_id}:{self.result.file_path}:{self.result.startline}"
    
    def add_score(self, strategy: str, score: float):
        """Add score from a search strategy"""
        self.scores[strategy] = score
        self.strategies.add(strategy)
    
    def calculate_final_score(self, weights: Dict[str, float]):
        """Calculate final weighted score"""
        total_score = 0.0
        total_weight = 0.0
        
        for strategy, score in self.scores.items():
            weight = weights.get(strategy, 1.0)
            total_score += score * weight
            total_weight += weight
        
        self.final_score = total_score / total_weight if total_weight > 0 else 0.0


@dataclass
class AggregationConfig:
    """Configuration for result aggregation"""
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'keyword': 1.0,
        'semantic': 1.2,
        'fuzzy': 0.8
    })
    deduplication_threshold: float = 0.95  # Content similarity threshold for deduplication
    max_results: int = 50
    boost_multiple_matches: float = 1.3  # Boost for results found by multiple strategies
    diversity_factor: float = 0.1  # Factor for promoting diverse results
    recency_weight: float = 0.05  # Weight for file recency
    project_distribution_weight: float = 0.1  # Promote results from different projects
    min_score_threshold: float = 0.1  # Minimum score to include result


class SearchResultAggregator:
    """Aggregates and ranks results from multiple search strategies"""
    
    def __init__(self, config: Optional[AggregationConfig] = None):
        """Initialize result aggregator
        
        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()
        self.keyword_search = None
        self.semantic_search = None
    
    def set_search_strategies(
        self,
        keyword_search: Optional[KeywordSearchStrategy] = None,
        semantic_search: Optional[SemanticSearchStrategy] = None
    ):
        """Set search strategy instances
        
        Args:
            keyword_search: Keyword search strategy
            semantic_search: Semantic search strategy
        """
        self.keyword_search = keyword_search
        self.semantic_search = semantic_search
    
    def aggregate_results(
        self,
        query: str,
        project_ids: Optional[List[int]] = None,
        file_extensions: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[GitLabSearchResult]:
        """Aggregate results from multiple search strategies
        
        Args:
            query: Search query
            project_ids: Specific projects to search
            file_extensions: File extensions to filter by
            strategies: List of strategies to use ('keyword', 'semantic')
            limit: Maximum results to return
            
        Returns:
            Aggregated and ranked search results
        """
        start_time = datetime.now()
        logger.info(f"Aggregating search results for query: '{query}'")
        
        # Default to all available strategies
        if strategies is None:
            strategies = ['keyword', 'semantic']
        
        # Use provided limit or config default
        max_results = limit or self.config.max_results
        
        # Collect results from each strategy
        all_scored_results: Dict[str, ScoredResult] = {}
        
        # Keyword search
        if 'keyword' in strategies and self.keyword_search:
            try:
                keyword_results = self.keyword_search.search_projects(
                    query=query,
                    project_ids=project_ids,
                    file_extensions=file_extensions,
                    limit=max_results * 2  # Get more for better aggregation
                )
                self._add_strategy_results(all_scored_results, keyword_results, 'keyword', query)
                logger.info(f"Keyword search contributed {len(keyword_results)} results")
            except Exception as e:
                logger.error(f"Keyword search failed: {e}")
        
        # Semantic search
        if 'semantic' in strategies and self.semantic_search:
            try:
                semantic_results = self.semantic_search.search_projects(
                    query=query,
                    project_ids=project_ids,
                    file_extensions=file_extensions,
                    limit=max_results
                )
                self._add_strategy_results(all_scored_results, semantic_results, 'semantic', query)
                logger.info(f"Semantic search contributed {len(semantic_results)} results")
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
        
        # Calculate final scores and rank
        final_results = self._rank_and_filter_results(
            list(all_scored_results.values()),
            query,
            max_results
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Aggregation completed in {duration:.2f}s, returning {len(final_results)} results")
        
        return final_results
    
    def _add_strategy_results(
        self,
        all_results: Dict[str, ScoredResult],
        strategy_results: List[GitLabSearchResult],
        strategy_name: str,
        query: str
    ):
        """Add results from a specific strategy to the aggregated results
        
        Args:
            all_results: Dictionary of all scored results (modified in place)
            strategy_results: Results from this strategy
            strategy_name: Name of the strategy
            query: Original search query
        """
        for i, result in enumerate(strategy_results):
            unique_key = f"{result.project_id}:{result.file_path}:{result.startline}"
            
            # Calculate strategy-specific score (position-based with content factors)
            base_score = max(0.1, 1.0 - (i / len(strategy_results)))  # Position-based score
            content_score = self._calculate_content_score(result, query)
            strategy_score = (base_score * 0.7) + (content_score * 0.3)
            
            if unique_key in all_results:
                # Add score to existing result
                all_results[unique_key].add_score(strategy_name, strategy_score)
            else:
                # Create new scored result
                scored_result = ScoredResult(result=result)
                scored_result.add_score(strategy_name, strategy_score)
                all_results[unique_key] = scored_result
    
    def _calculate_content_score(self, result: GitLabSearchResult, query: str) -> float:
        """Calculate content-based score for a result
        
        Args:
            result: Search result
            query: Search query
            
        Returns:
            Content-based score (0-1)
        """
        score = 0.0
        content = result.content.lower()
        filename = result.file_path.lower()
        query_terms = query.lower().split()
        
        # Term frequency in content
        for term in query_terms:
            score += content.count(term) * 0.1
        
        # Filename matches
        for term in query_terms:
            if term in filename:
                score += 0.3
        
        # File type preferences
        if filename.endswith(('.md', '.txt', '.rst')):
            score *= 1.2  # Documentation boost
        elif filename.endswith(('.py', '.js', '.ts')):
            score *= 1.1  # Code boost
        
        # Wiki content boost
        if result.wiki:
            score *= 1.15
        
        # Content length normalization
        content_length = len(result.content)
        if content_length < 50:
            score *= 0.8  # Penalize very short content
        elif content_length > 1000:
            score *= 1.1  # Slight boost for substantial content
        
        return min(1.0, score)  # Cap at 1.0
    
    def _rank_and_filter_results(
        self,
        scored_results: List[ScoredResult],
        query: str,
        max_results: int
    ) -> List[GitLabSearchResult]:
        """Rank and filter aggregated results
        
        Args:
            scored_results: List of scored results
            query: Original search query
            max_results: Maximum results to return
            
        Returns:
            Final ranked list of search results
        """
        if not scored_results:
            return []
        
        # Calculate final scores for all results
        for result in scored_results:
            result.calculate_final_score(self.config.strategy_weights)
            
            # Apply additional ranking factors
            self._apply_ranking_factors(result, query)
        
        # Filter by minimum score threshold
        filtered_results = [
            r for r in scored_results 
            if r.final_score >= self.config.min_score_threshold
        ]
        
        # Sort by final score (descending)
        filtered_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply diversity and project distribution
        final_results = self._apply_diversity_filtering(filtered_results, max_results)
        
        # Extract the GitLabSearchResult objects
        return [result.result for result in final_results]
    
    def _apply_ranking_factors(self, scored_result: ScoredResult, query: str):
        """Apply additional ranking factors to a scored result
        
        Args:
            scored_result: Scored result to enhance (modified in place)
            query: Original search query
        """
        result = scored_result.result
        factors = scored_result.rank_factors
        
        # Multiple strategy boost
        if len(scored_result.strategies) > 1:
            boost = self.config.boost_multiple_matches
            factors['multiple_strategies'] = boost
            scored_result.final_score *= boost
        
        # File path relevance
        query_terms = query.lower().split()
        path_relevance = 0.0
        for term in query_terms:
            if term in result.file_path.lower():
                path_relevance += 0.1
        
        if path_relevance > 0:
            factors['path_relevance'] = 1.0 + path_relevance
            scored_result.final_score *= (1.0 + path_relevance)
        
        # Project name relevance
        project_relevance = 0.0
        for term in query_terms:
            if term in result.project_name.lower():
                project_relevance += 0.05
        
        if project_relevance > 0:
            factors['project_relevance'] = 1.0 + project_relevance
            scored_result.final_score *= (1.0 + project_relevance)
        
        # Starting line preference (earlier lines often more relevant)
        if result.startline <= 10:
            line_boost = 1.1
            factors['early_line'] = line_boost
            scored_result.final_score *= line_boost
    
    def _apply_diversity_filtering(
        self,
        scored_results: List[ScoredResult],
        max_results: int
    ) -> List[ScoredResult]:
        """Apply diversity filtering to promote varied results
        
        Args:
            scored_results: Sorted list of scored results
            max_results: Maximum results to return
            
        Returns:
            Diversified list of results
        """
        if len(scored_results) <= max_results:
            return scored_results
        
        final_results = []
        project_counts = defaultdict(int)
        file_types = defaultdict(int)
        
        # Track diversity metrics
        total_projects = len(set(r.result.project_id for r in scored_results))
        max_per_project = max(1, max_results // total_projects) if total_projects > 0 else max_results
        
        for result in scored_results:
            if len(final_results) >= max_results:
                break
            
            project_id = result.result.project_id
            file_ext = result.result.file_path.split('.')[-1].lower() if '.' in result.result.file_path else 'none'
            
            # Apply diversity constraints
            include_result = True
            
            # Project distribution
            if project_counts[project_id] >= max_per_project and len(final_results) > max_results // 2:
                # Allow project concentration only in first half of results
                if self._should_skip_for_diversity(result, final_results):
                    include_result = False
            
            # File type diversity (less strict than project diversity)
            if file_types[file_ext] >= max_results // 3:
                if result.final_score < final_results[-1].final_score * 1.5:  # Only skip if not significantly better
                    include_result = False
            
            if include_result:
                final_results.append(result)
                project_counts[project_id] += 1
                file_types[file_ext] += 1
        
        return final_results
    
    def _should_skip_for_diversity(
        self,
        candidate: ScoredResult,
        current_results: List[ScoredResult]
    ) -> bool:
        """Determine if a result should be skipped for diversity
        
        Args:
            candidate: Candidate result to evaluate
            current_results: Current results list
            
        Returns:
            True if result should be skipped for diversity
        """
        # Check if we have very similar results already
        candidate_content = candidate.result.content.lower()
        
        for existing in current_results[-5:]:  # Check last 5 results
            if existing.result.project_id == candidate.result.project_id:
                # Calculate content similarity
                similarity = self._calculate_content_similarity(
                    candidate_content,
                    existing.result.content.lower()
                )
                
                if similarity > self.config.deduplication_threshold:
                    return True  # Skip very similar content
        
        return False
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            Similarity score (0-1)
        """
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_aggregation_stats(self, scored_results: List[ScoredResult]) -> Dict[str, Any]:
        """Get statistics about the aggregation process
        
        Args:
            scored_results: List of scored results
            
        Returns:
            Aggregation statistics
        """
        if not scored_results:
            return {'total_results': 0}
        
        strategy_counts = defaultdict(int)
        multiple_strategy_count = 0
        project_distribution = defaultdict(int)
        
        for result in scored_results:
            for strategy in result.strategies:
                strategy_counts[strategy] += 1
            
            if len(result.strategies) > 1:
                multiple_strategy_count += 1
            
            project_distribution[result.result.project_name] += 1
        
        return {
            'total_results': len(scored_results),
            'strategy_counts': dict(strategy_counts),
            'multiple_strategy_matches': multiple_strategy_count,
            'project_distribution': dict(project_distribution),
            'avg_score': sum(r.final_score for r in scored_results) / len(scored_results),
            'score_range': {
                'min': min(r.final_score for r in scored_results),
                'max': max(r.final_score for r in scored_results)
            }
        }
    
    def test_aggregation(
        self,
        query: str,
        project_id: Optional[int] = None,
        strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Test aggregation functionality
        
        Args:
            query: Test query
            project_id: Optional specific project
            strategies: Strategies to test
            
        Returns:
            Test results with performance metrics
        """
        import time
        
        start_time = time.time()
        
        try:
            project_ids = [project_id] if project_id else None
            results = self.aggregate_results(
                query=query,
                project_ids=project_ids,
                strategies=strategies or ['keyword', 'semantic'],
                limit=20
            )
            
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            return {
                'success': True,
                'query': query,
                'strategies_used': strategies or ['keyword', 'semantic'],
                'results_count': len(results),
                'duration_ms': duration,
                'sample_results': [
                    {
                        'project': r.project_name,
                        'file': r.file_path,
                        'line': r.startline,
                        'content_preview': r.content[:100] + '...'
                    }
                    for r in results[:5]
                ],
                'config': {
                    'strategy_weights': self.config.strategy_weights,
                    'max_results': self.config.max_results,
                    'deduplication_threshold': self.config.deduplication_threshold
                }
            }
            
        except Exception as e:
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'duration_ms': duration
            }


def create_result_aggregator(
    strategy_weights: Optional[Dict[str, float]] = None,
    max_results: int = 50,
    boost_multiple_matches: float = 1.3
) -> SearchResultAggregator:
    """Factory function to create result aggregator
    
    Args:
        strategy_weights: Weights for different strategies
        max_results: Maximum results to return
        boost_multiple_matches: Boost factor for multi-strategy matches
        
    Returns:
        Configured search result aggregator
    """
    config = AggregationConfig(
        strategy_weights=strategy_weights or {'keyword': 1.0, 'semantic': 1.2},
        max_results=max_results,
        boost_multiple_matches=boost_multiple_matches
    )
    
    return SearchResultAggregator(config)


if __name__ == "__main__":
    # Test aggregation functionality
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.integrations.search.aggregator <query> [project_id]")
        sys.exit(1)
    
    query = sys.argv[1]
    project_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"Testing result aggregation with query: '{query}'")
    
    # Create aggregator (would need actual search strategies for full test)
    aggregator = create_result_aggregator()
    results = aggregator.test_aggregation(query, project_id)
    
    print("Test Results:")
    print(f"  Success: {results['success']}")
    print(f"  Duration: {results.get('duration_ms', 0):.1f}ms")
    
    if results['success']:
        print(f"  Strategies: {results['strategies_used']}")
        print(f"  Results found: {results['results_count']}")
        if results['sample_results']:
            print("  Sample results:")
            for result in results['sample_results']:
                print(f"    - {result['project']}: {result['file']} (line {result['line']})")
    else:
        print(f"  Error: {results['error']}")