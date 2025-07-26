"""
Query Optimization System

This module provides intelligent query optimization for search operations,
including query normalization, index utilization, and search pattern optimization.
"""

import re
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of search queries"""
    KEYWORD = "keyword"
    PHRASE = "phrase"
    BOOLEAN = "boolean"
    FUZZY = "fuzzy"
    WILDCARD = "wildcard"
    REGEX = "regex"
    SEMANTIC = "semantic"


class OptimizationLevel(Enum):
    """Optimization levels"""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"


@dataclass
class QueryAnalysis:
    """Analysis results for a search query"""
    original_query: str
    normalized_query: str
    query_type: QueryType
    complexity_score: float
    estimated_results: int
    suggested_optimizations: List[str]
    confidence: float
    keywords: List[str]
    phrases: List[str]
    filters: Dict[str, List[str]]
    is_cacheable: bool
    cache_key: str


@dataclass
class OptimizationConfig:
    """Configuration for query optimization"""
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    enable_query_normalization: bool = True
    enable_stop_word_removal: bool = True
    enable_stemming: bool = True
    enable_synonym_expansion: bool = False
    enable_fuzzy_matching: bool = True
    enable_query_rewriting: bool = True
    max_query_length: int = 500
    min_term_length: int = 2
    max_boolean_terms: int = 20
    enable_performance_hints: bool = True
    cache_optimized_queries: bool = True


class QueryOptimizer:
    """Advanced query optimization system for search operations
    
    This class provides comprehensive query analysis and optimization capabilities,
    including normalization, rewriting, and performance optimization suggestions.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize query optimizer
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        
        # Common stop words to remove (basic set)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'or', 'but', 'not', 'this', 'these',
            'they', 'we', 'you', 'have', 'had', 'what', 'when', 'where', 'who',
            'which', 'why', 'how', 'all', 'any', 'both', 'can', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own',
            'so', 'than', 'too', 'very', 'one', 'two', 'may', 'also'
        }
        
        # Technical terms that should not be treated as stop words
        self.technical_preserve = {
            'api', 'ui', 'db', 'id', 'url', 'http', 'https', 'json', 'xml',
            'css', 'js', 'py', 'go', 'cpp', 'java', 'rust', 'sql', 'git',
            'ci', 'cd', 'dev', 'prod', 'test', 'qa', 'os', 'vm', 'k8s'
        }
        
        # Common programming/technical synonyms
        self.synonyms = {
            'function': ['method', 'procedure', 'routine'],
            'variable': ['var', 'field', 'property'],
            'database': ['db', 'storage', 'datastore'],
            'configuration': ['config', 'settings', 'setup'],
            'authentication': ['auth', 'login', 'signin'],
            'authorization': ['authz', 'permissions', 'access'],
            'documentation': ['docs', 'readme', 'guide'],
            'environment': ['env', 'runtime', 'context'],
            'repository': ['repo', 'codebase', 'source'],
            'application': ['app', 'program', 'software']
        }
        
        # Pattern cache for optimized queries
        self._optimization_cache: Dict[str, QueryAnalysis] = {}
        
        # Performance statistics
        self._stats = {
            'queries_optimized': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_optimization_time_ms': 0.0,
            'complexity_reduction': 0.0
        }
    
    def optimize_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True
    ) -> QueryAnalysis:
        """Optimize a search query for maximum performance and relevance
        
        Args:
            query: Original search query
            context: Optional context (project_ids, file_extensions, etc.)
            enable_caching: Whether to use cached optimization results
            
        Returns:
            Query analysis with optimization recommendations
        """
        start_time = datetime.now()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(query, context)
            
            # Check cache first
            if enable_caching and cache_key in self._optimization_cache:
                self._stats['cache_hits'] += 1
                return self._optimization_cache[cache_key]
            
            self._stats['cache_misses'] += 1
            
            # Perform query analysis
            analysis = self._analyze_query(query, context)
            
            # Cache result if enabled
            if enable_caching and self.config.cache_optimized_queries:
                self._optimization_cache[cache_key] = analysis
            
            # Update statistics
            self._update_optimization_stats(start_time, analysis)
            
            logger.debug(f"Query optimized: '{query}' -> '{analysis.normalized_query}' (confidence: {analysis.confidence:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error optimizing query '{query}': {e}")
            # Return basic analysis on error
            return self._create_fallback_analysis(query)
    
    def batch_optimize_queries(
        self,
        queries: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[QueryAnalysis]:
        """Optimize multiple queries efficiently
        
        Args:
            queries: List of queries to optimize
            context: Optional shared context
            
        Returns:
            List of query analyses
        """
        analyses = []
        
        for query in queries:
            analysis = self.optimize_query(query, context, enable_caching=True)
            analyses.append(analysis)
        
        logger.info(f"Batch optimized {len(queries)} queries")
        return analyses
    
    def suggest_query_improvements(self, query: str) -> List[str]:
        """Suggest improvements for a query based on common patterns
        
        Args:
            query: Query to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Analyze query characteristics
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Too short query
        if len(words) < 2:
            suggestions.append("Consider adding more specific terms to improve results")
        
        # Too long query
        if len(words) > 10:
            suggestions.append("Consider breaking down into smaller, focused queries")
        
        # All stop words
        meaningful_words = [w for w in words if w not in self.stop_words or w in self.technical_preserve]
        if len(meaningful_words) == 0:
            suggestions.append("Query contains only common words - add specific technical terms")
        
        # Potential typos (very basic detection)
        if any(len(word) > 15 for word in words):
            suggestions.append("Check for potential typos in long words")
        
        # Boolean operators without quotes
        if any(op in query_lower for op in ['and', 'or', 'not']) and '"' not in query:
            suggestions.append("Consider using quotes for phrase searches with boolean terms")
        
        # File extension without proper filtering
        if any(f'.{ext}' in query_lower for ext in ['py', 'js', 'md', 'json', 'yaml']):
            suggestions.append("Consider using file extension filters instead of including them in query")
        
        # Very generic terms
        generic_terms = ['function', 'method', 'class', 'variable', 'file', 'code']
        if any(term in words for term in generic_terms) and len(meaningful_words) <= 2:
            suggestions.append("Add more specific context to generic terms")
        
        return suggestions
    
    def _analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryAnalysis:
        """Perform comprehensive query analysis
        
        Args:
            query: Query to analyze
            context: Optional context information
            
        Returns:
            Detailed query analysis
        """
        # Basic preprocessing
        normalized_query = self._normalize_query(query)
        query_type = self._detect_query_type(query)
        
        # Extract components
        keywords = self._extract_keywords(normalized_query)
        phrases = self._extract_phrases(query)
        filters = self._extract_filters(query, context)
        
        # Calculate complexity and estimations
        complexity_score = self._calculate_complexity(query, keywords, phrases)
        estimated_results = self._estimate_result_count(keywords, phrases, context)
        
        # Generate optimizations
        suggested_optimizations = self._generate_optimizations(
            query, normalized_query, query_type, keywords, phrases
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(query, keywords, phrases)
        
        # Determine cacheability
        is_cacheable = self._is_cacheable(query, query_type, complexity_score)
        cache_key = self._generate_cache_key(query, context) if is_cacheable else ""
        
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            query_type=query_type,
            complexity_score=complexity_score,
            estimated_results=estimated_results,
            suggested_optimizations=suggested_optimizations,
            confidence=confidence,
            keywords=keywords,
            phrases=phrases,
            filters=filters,
            is_cacheable=is_cacheable,
            cache_key=cache_key
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for optimal processing
        
        Args:
            query: Original query
            
        Returns:
            Normalized query string
        """
        if not self.config.enable_query_normalization:
            return query.strip()
        
        # Basic cleaning
        normalized = query.strip()
        
        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle quoted phrases (preserve them)
        quoted_phrases = re.findall(r'"([^"]*)"', normalized)
        placeholder_map = {}
        
        for i, phrase in enumerate(quoted_phrases):
            placeholder = f"__PHRASE_{i}__"
            placeholder_map[placeholder] = f'"{phrase}"'
            normalized = normalized.replace(f'"{phrase}"', placeholder)
        
        # Convert to lowercase for processing (but preserve original case in phrases)
        if self.config.optimization_level in [OptimizationLevel.ADVANCED, OptimizationLevel.AGGRESSIVE]:
            words = normalized.lower().split()
            
            # Remove stop words but preserve technical terms
            if self.config.enable_stop_word_removal:
                words = [
                    word for word in words 
                    if word not in self.stop_words 
                    or word in self.technical_preserve
                    or word.startswith('__PHRASE_')
                ]
            
            # Filter by minimum length
            words = [word for word in words if len(word) >= self.config.min_term_length or word.startswith('__PHRASE_')]
            
            normalized = ' '.join(words)
        
        # Restore quoted phrases
        for placeholder, phrase in placeholder_map.items():
            normalized = normalized.replace(placeholder, phrase)
        
        # Limit query length
        if len(normalized) > self.config.max_query_length:
            normalized = normalized[:self.config.max_query_length].strip()
        
        return normalized
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of search query
        
        Args:
            query: Query to analyze
            
        Returns:
            Detected query type
        """
        query_lower = query.lower()
        
        # Check for regex patterns
        if re.search(r'[\[\]{}().+*?^$|\\]', query) and len(query.split()) <= 2:
            return QueryType.REGEX
        
        # Check for wildcards
        if '*' in query or '?' in query:
            return QueryType.WILDCARD
        
        # Check for quoted phrases
        if '"' in query and query.count('"') >= 2:
            return QueryType.PHRASE
        
        # Check for boolean operators
        boolean_indicators = ['AND', 'OR', 'NOT', '+', '-']
        if any(op in query.upper() for op in boolean_indicators):
            return QueryType.BOOLEAN
        
        # Check for fuzzy indicators
        if '~' in query or 'similar' in query_lower:
            return QueryType.FUZZY
        
        # Check for semantic indicators
        semantic_indicators = [
            'how to', 'what is', 'explain', 'concept', 'meaning', 'definition',
            'tutorial', 'guide', 'example', 'best practice'
        ]
        if any(indicator in query_lower for indicator in semantic_indicators):
            return QueryType.SEMANTIC
        
        # Default to keyword search
        return QueryType.KEYWORD
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query
        
        Args:
            query: Normalized query
            
        Returns:
            List of keywords
        """
        # Remove quoted phrases for keyword extraction
        query_without_phrases = re.sub(r'"[^"]*"', '', query)
        
        # Split and filter
        words = query_without_phrases.lower().split()
        
        # Remove stop words and short words
        keywords = [
            word for word in words 
            if (word not in self.stop_words or word in self.technical_preserve) 
            and len(word) >= self.config.min_term_length
        ]
        
        return keywords
    
    def _extract_phrases(self, query: str) -> List[str]:
        """Extract quoted phrases from query
        
        Args:
            query: Original query
            
        Returns:
            List of phrases
        """
        return re.findall(r'"([^"]*)"', query)
    
    def _extract_filters(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """Extract search filters from query and context
        
        Args:
            query: Query string
            context: Optional context information
            
        Returns:
            Dictionary of extracted filters
        """
        filters = defaultdict(list)
        
        # Extract file extensions from query
        file_extensions = re.findall(r'\.(\w+)', query)
        if file_extensions:
            filters['file_extensions'].extend(file_extensions)
        
        # Extract potential project names
        project_patterns = re.findall(r'project:(\w+)', query.lower())
        if project_patterns:
            filters['projects'].extend(project_patterns)
        
        # Extract context filters
        if context:
            if 'project_ids' in context:
                filters['project_ids'] = context['project_ids']
            if 'file_extensions' in context:
                filters['file_extensions'].extend(context['file_extensions'])
        
        return dict(filters)
    
    def _calculate_complexity(self, query: str, keywords: List[str], phrases: List[str]) -> float:
        """Calculate query complexity score
        
        Args:
            query: Original query
            keywords: Extracted keywords
            phrases: Extracted phrases
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        base_complexity = 0.0
        
        # Length factor
        word_count = len(query.split())
        length_factor = min(word_count / 20.0, 1.0)  # Normalize to max 20 words
        base_complexity += length_factor * 0.3
        
        # Keyword diversity
        unique_keywords = len(set(keywords))
        keyword_factor = min(unique_keywords / 10.0, 1.0)  # Normalize to max 10 keywords
        base_complexity += keyword_factor * 0.2
        
        # Phrase complexity
        phrase_factor = min(len(phrases) / 5.0, 1.0)  # Normalize to max 5 phrases
        base_complexity += phrase_factor * 0.2
        
        # Special characters/operators
        special_chars = len(re.findall(r'[(){}[\]*+?^$|\\]', query))
        special_factor = min(special_chars / 10.0, 1.0)
        base_complexity += special_factor * 0.3
        
        return min(base_complexity, 1.0)
    
    def _estimate_result_count(
        self, 
        keywords: List[str], 
        phrases: List[str], 
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """Estimate number of results for the query
        
        Args:
            keywords: Query keywords
            phrases: Query phrases
            context: Optional context
            
        Returns:
            Estimated result count
        """
        # This is a simplified estimation model
        base_estimate = 100
        
        # Reduce estimate based on specificity
        if len(keywords) > 3:
            base_estimate = int(base_estimate * 0.7)
        
        if phrases:
            base_estimate = int(base_estimate * 0.5)
        
        # Adjust based on context
        if context and 'project_ids' in context:
            project_count = len(context['project_ids'])
            if project_count < 5:
                base_estimate = int(base_estimate * 0.3)
            elif project_count < 20:
                base_estimate = int(base_estimate * 0.6)
        
        return max(base_estimate, 1)
    
    def _generate_optimizations(
        self, 
        original_query: str,
        normalized_query: str,
        query_type: QueryType,
        keywords: List[str],
        phrases: List[str]
    ) -> List[str]:
        """Generate optimization suggestions
        
        Args:
            original_query: Original query
            normalized_query: Normalized query
            query_type: Detected query type
            keywords: Extracted keywords
            phrases: Extracted phrases
            
        Returns:
            List of optimization suggestions
        """
        optimizations = []
        
        # Query rewriting suggestions
        if normalized_query != original_query.strip():
            optimizations.append(f"Normalized query to: '{normalized_query}'")
        
        # Strategy suggestions based on query type
        if query_type == QueryType.SEMANTIC:
            optimizations.append("Use semantic search strategy for conceptual queries")
        elif query_type == QueryType.PHRASE:
            optimizations.append("Use phrase matching for quoted terms")
        elif query_type == QueryType.BOOLEAN:
            optimizations.append("Optimize boolean query structure")
        
        # Keyword suggestions
        if len(keywords) == 1:
            optimizations.append("Consider expanding single keyword with synonyms")
        elif len(keywords) > 7:
            optimizations.append("Consider focusing on most important keywords")
        
        # Caching suggestions
        if len(keywords) <= 5 and not phrases:
            optimizations.append("Query is suitable for aggressive caching")
        
        # Performance hints
        if self.config.enable_performance_hints:
            if any(kw in ['error', 'bug', 'issue', 'problem'] for kw in keywords):
                optimizations.append("Consider searching in issue tracking or error logs")
            if any(kw in ['config', 'setting', 'setup'] for kw in keywords):
                optimizations.append("Focus search on configuration files")
        
        return optimizations
    
    def _calculate_confidence(self, query: str, keywords: List[str], phrases: List[str]) -> float:
        """Calculate optimization confidence score
        
        Args:
            query: Original query
            keywords: Extracted keywords
            phrases: Extracted phrases
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.5  # Base confidence
        
        # Increase confidence for well-structured queries
        if 2 <= len(keywords) <= 5:
            confidence += 0.2
        
        if phrases:
            confidence += 0.1
        
        # Decrease confidence for very short or very long queries
        word_count = len(query.split())
        if word_count < 2:
            confidence -= 0.3
        elif word_count > 15:
            confidence -= 0.2
        
        # Technical terms increase confidence
        technical_keywords = [kw for kw in keywords if kw in self.technical_preserve]
        if technical_keywords:
            confidence += min(len(technical_keywords) * 0.1, 0.3)
        
        return max(0.0, min(confidence, 1.0))
    
    def _is_cacheable(self, query: str, query_type: QueryType, complexity: float) -> bool:
        """Determine if query results should be cached
        
        Args:
            query: Original query
            query_type: Query type
            complexity: Complexity score
            
        Returns:
            True if query is suitable for caching
        """
        # Don't cache very simple or very complex queries
        if complexity < 0.1 or complexity > 0.8:
            return False
        
        # Don't cache queries with user-specific terms
        user_specific_terms = ['my', 'mine', 'personal', 'private', 'user:', 'author:']
        if any(term in query.lower() for term in user_specific_terms):
            return False
        
        # Cache semantic and keyword queries
        if query_type in [QueryType.KEYWORD, QueryType.SEMANTIC, QueryType.PHRASE]:
            return True
        
        return False
    
    def _generate_cache_key(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for query optimization
        
        Args:
            query: Query string
            context: Optional context
            
        Returns:
            Cache key string
        """
        # Create deterministic key from query and context
        key_data = {
            'query': query.lower().strip(),
            'context': context or {}
        }
        
        key_string = str(key_data)
        return f"opt:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _create_fallback_analysis(self, query: str) -> QueryAnalysis:
        """Create basic analysis when optimization fails
        
        Args:
            query: Original query
            
        Returns:
            Basic query analysis
        """
        return QueryAnalysis(
            original_query=query,
            normalized_query=query.strip(),
            query_type=QueryType.KEYWORD,
            complexity_score=0.5,
            estimated_results=50,
            suggested_optimizations=["Basic analysis due to optimization error"],
            confidence=0.3,
            keywords=query.strip().split(),
            phrases=[],
            filters={},
            is_cacheable=False,
            cache_key=""
        )
    
    def _update_optimization_stats(self, start_time: datetime, analysis: QueryAnalysis):
        """Update optimization statistics
        
        Args:
            start_time: Optimization start time
            analysis: Query analysis result
        """
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        self._stats['queries_optimized'] += 1
        
        # Update average duration
        current_avg = self._stats['avg_optimization_time_ms']
        count = self._stats['queries_optimized']
        self._stats['avg_optimization_time_ms'] = ((current_avg * (count - 1)) + duration_ms) / count
        
        # Update complexity reduction (if normalization changed the query)
        if analysis.normalized_query != analysis.original_query:
            original_complexity = len(analysis.original_query.split())
            normalized_complexity = len(analysis.normalized_query.split())
            reduction = (original_complexity - normalized_complexity) / original_complexity
            
            current_reduction = self._stats['complexity_reduction']
            self._stats['complexity_reduction'] = ((current_reduction * (count - 1)) + reduction) / count
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics
        
        Returns:
            Dictionary of optimization statistics
        """
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        cache_hit_rate = self._stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'queries_optimized': self._stats['queries_optimized'],
            'avg_optimization_time_ms': self._stats['avg_optimization_time_ms'],
            'complexity_reduction': self._stats['complexity_reduction'],
            'cache_performance': {
                'hit_rate': cache_hit_rate,
                'hits': self._stats['cache_hits'],
                'misses': self._stats['cache_misses']
            },
            'config': {
                'optimization_level': self.config.optimization_level.value,
                'normalization_enabled': self.config.enable_query_normalization,
                'stop_word_removal': self.config.enable_stop_word_removal
            }
        }
    
    def clear_optimization_cache(self):
        """Clear the optimization cache"""
        self._optimization_cache.clear()
        logger.info("Cleared query optimization cache")
    
    def analyze_query_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze patterns in a collection of queries
        
        Args:
            queries: List of queries to analyze
            
        Returns:
            Pattern analysis results
        """
        if not queries:
            return {'error': 'No queries provided'}
        
        # Optimize all queries
        analyses = self.batch_optimize_queries(queries)
        
        # Analyze patterns
        query_types = [analysis.query_type.value for analysis in analyses]
        complexities = [analysis.complexity_score for analysis in analyses]
        confidences = [analysis.confidence for analysis in analyses]
        
        # Extract all keywords
        all_keywords = []
        for analysis in analyses:
            all_keywords.extend(analysis.keywords)
        
        keyword_counts = Counter(all_keywords)
        
        return {
            'total_queries': len(queries),
            'query_type_distribution': dict(Counter(query_types)),
            'avg_complexity': sum(complexities) / len(complexities),
            'avg_confidence': sum(confidences) / len(confidences),
            'most_common_keywords': keyword_counts.most_common(10),
            'optimization_recommendations': self._generate_pattern_recommendations(analyses),
            'cacheable_queries': sum(1 for a in analyses if a.is_cacheable),
            'cache_potential': sum(1 for a in analyses if a.is_cacheable) / len(analyses)
        }
    
    def _generate_pattern_recommendations(self, analyses: List[QueryAnalysis]) -> List[str]:
        """Generate recommendations based on query patterns
        
        Args:
            analyses: List of query analyses
            
        Returns:
            List of pattern-based recommendations
        """
        recommendations = []
        
        # Check for low confidence queries
        low_confidence = [a for a in analyses if a.confidence < 0.5]
        if len(low_confidence) > len(analyses) * 0.3:
            recommendations.append("Many queries have low confidence - consider query refinement training")
        
        # Check for high complexity
        high_complexity = [a for a in analyses if a.complexity_score > 0.7]
        if len(high_complexity) > len(analyses) * 0.2:
            recommendations.append("Many complex queries detected - consider query simplification guidance")
        
        # Check cache potential
        cacheable = [a for a in analyses if a.is_cacheable]
        if len(cacheable) > len(analyses) * 0.6:
            recommendations.append("High cache potential - increase cache size and TTL")
        
        return recommendations


def create_query_optimizer(
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED,
    enable_query_normalization: bool = True,
    enable_stop_word_removal: bool = True,
    cache_optimized_queries: bool = True
) -> QueryOptimizer:
    """Factory function to create a configured query optimizer
    
    Args:
        optimization_level: Level of optimization to apply
        enable_query_normalization: Whether to normalize queries
        enable_stop_word_removal: Whether to remove stop words
        cache_optimized_queries: Whether to cache optimization results
        
    Returns:
        Configured query optimizer
    """
    config = OptimizationConfig(
        optimization_level=optimization_level,
        enable_query_normalization=enable_query_normalization,
        enable_stop_word_removal=enable_stop_word_removal,
        cache_optimized_queries=cache_optimized_queries
    )
    
    return QueryOptimizer(config)


if __name__ == "__main__":
    # Test query optimization
    print("Testing query optimization system...")
    
    # Create optimizer
    optimizer = create_query_optimizer()
    
    # Test queries
    test_queries = [
        "how to setup authentication in Django",
        "function that handles user login and password validation",
        "database configuration for postgresql",
        '"error handling" best practices',
        "api endpoint returns 500 error",
        "react component state management",
        "docker compose file example",
        "git merge conflict resolution"
    ]
    
    print(f"\nOptimizing {len(test_queries)} test queries:")
    
    for query in test_queries:
        analysis = optimizer.optimize_query(query)
        
        print(f"\nQuery: '{query}'")
        print(f"  Normalized: '{analysis.normalized_query}'")
        print(f"  Type: {analysis.query_type.value}")
        print(f"  Complexity: {analysis.complexity_score:.2f}")
        print(f"  Confidence: {analysis.confidence:.2f}")
        print(f"  Keywords: {analysis.keywords}")
        print(f"  Cacheable: {analysis.is_cacheable}")
        if analysis.suggested_optimizations:
            print(f"  Optimizations: {analysis.suggested_optimizations[0]}")
    
    # Show statistics
    stats = optimizer.get_optimization_stats()
    print(f"\nOptimization Statistics:")
    print(f"  Queries optimized: {stats['queries_optimized']}")
    print(f"  Average time: {stats['avg_optimization_time_ms']:.1f}ms")
    print(f"  Complexity reduction: {stats['complexity_reduction']:.2%}")
    
    # Pattern analysis
    pattern_analysis = optimizer.analyze_query_patterns(test_queries)
    print(f"\nPattern Analysis:")
    print(f"  Query types: {pattern_analysis['query_type_distribution']}")
    print(f"  Average complexity: {pattern_analysis['avg_complexity']:.2f}")
    print(f"  Average confidence: {pattern_analysis['avg_confidence']:.2f}")
    print(f"  Cache potential: {pattern_analysis['cache_potential']:.2%}")
    
    print("\nQuery optimization testing complete")