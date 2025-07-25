"""
Keyword Search Strategy for GitLab Content

This module implements advanced keyword-based search functionality for GitLab
repositories, including boolean operations, phrase matching, and fuzzy search.
"""

import logging
import re
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..gitlab_client import GitLabClient, GitLabSearchResult, get_gitlab_client

logger = logging.getLogger(__name__)


class SearchOperator(Enum):
    """Search operators for keyword matching"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PHRASE = "PHRASE"
    WILDCARD = "WILDCARD"


@dataclass
class SearchTerm:
    """Represents a search term with metadata"""
    term: str
    operator: SearchOperator = SearchOperator.AND
    weight: float = 1.0
    field_boost: Dict[str, float] = None  # Boost for specific fields
    
    def __post_init__(self):
        if self.field_boost is None:
            self.field_boost = {}


@dataclass
class KeywordSearchConfig:
    """Configuration for keyword search"""
    case_sensitive: bool = False
    fuzzy_threshold: float = 0.8
    min_term_length: int = 2
    max_results_per_project: int = 50
    enable_phrase_search: bool = True
    enable_fuzzy_search: bool = True
    enable_wildcard_search: bool = True
    boost_filename_matches: float = 2.0
    boost_title_matches: float = 1.5
    boost_recent_files: float = 1.2


class KeywordSearchStrategy:
    """Advanced keyword search strategy for GitLab content"""
    
    def __init__(
        self,
        gitlab_client: Optional[GitLabClient] = None,
        config: Optional[KeywordSearchConfig] = None
    ):
        """Initialize keyword search strategy
        
        Args:
            gitlab_client: GitLab client instance
            config: Search configuration
        """
        self.client = gitlab_client or get_gitlab_client()
        self.config = config or KeywordSearchConfig()
        
        # Common stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
    
    def parse_query(self, query: str) -> List[SearchTerm]:
        """Parse search query into structured search terms
        
        Args:
            query: Raw search query string
            
        Returns:
            List of parsed search terms
        """
        terms = []
        
        # Handle quoted phrases first
        phrase_pattern = r'"([^"]*)"'
        phrases = re.findall(phrase_pattern, query)
        
        # Remove phrases from query to process other terms
        query_without_phrases = re.sub(phrase_pattern, '', query)
        
        # Add phrase terms
        for phrase in phrases:
            if len(phrase.strip()) >= self.config.min_term_length:
                terms.append(SearchTerm(
                    term=phrase.strip(),
                    operator=SearchOperator.PHRASE,
                    weight=1.5  # Boost phrase matches
                ))
        
        # Process remaining terms
        words = query_without_phrases.split()
        current_operator = SearchOperator.AND
        
        for word in words:
            word = word.strip().lower()
            
            # Skip empty words and stop words
            if not word or word in self.stop_words:
                continue
            
            # Handle operators
            if word in ['and', '&']:
                current_operator = SearchOperator.AND
                continue
            elif word in ['or', '|']:
                current_operator = SearchOperator.OR
                continue
            elif word in ['not', '-']:
                current_operator = SearchOperator.NOT
                continue
            
            # Handle wildcards
            if '*' in word or '?' in word:
                if self.config.enable_wildcard_search:
                    terms.append(SearchTerm(
                        term=word,
                        operator=SearchOperator.WILDCARD,
                        weight=0.9  # Slightly lower weight for wildcards
                    ))
                continue
            
            # Regular terms
            if len(word) >= self.config.min_term_length:
                terms.append(SearchTerm(
                    term=word,
                    operator=current_operator,
                    weight=1.0
                ))
        
        return terms
    
    def search_projects(
        self,
        query: str,
        project_ids: Optional[List[int]] = None,
        file_extensions: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[GitLabSearchResult]:
        """Search across GitLab projects using keyword strategy
        
        Args:
            query: Search query string
            project_ids: Specific projects to search
            file_extensions: File extensions to filter by
            limit: Maximum results to return
            
        Returns:
            List of search results with relevance scoring
        """
        logger.info(f"Starting keyword search for: '{query}'")
        
        # Parse the query into structured terms
        search_terms = self.parse_query(query)
        
        if not search_terms:
            logger.warning("No valid search terms found in query")
            return []
        
        # Get basic search results from GitLab
        basic_results = self.client.search_files(
            query=query,
            project_ids=project_ids,
            file_extensions=file_extensions,
            limit=limit * 2  # Get more to allow for filtering
        )
        
        # Apply advanced keyword matching and scoring
        scored_results = []
        for result in basic_results:
            score = self._calculate_keyword_score(result, search_terms, query)
            if score > 0:  # Only include results with positive scores
                scored_results.append((result, score))
        
        # Sort by score (highest first) and limit results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        final_results = [result for result, score in scored_results[:limit]]
        
        logger.info(f"Keyword search returned {len(final_results)} results")
        return final_results
    
    def _calculate_keyword_score(
        self,
        result: GitLabSearchResult,
        search_terms: List[SearchTerm],
        original_query: str
    ) -> float:
        """Calculate relevance score for a search result
        
        Args:
            result: GitLab search result
            search_terms: Parsed search terms
            original_query: Original query string
            
        Returns:
            Relevance score (0 = no match, higher = better match)
        """
        if not search_terms:
            return 0.0
        
        score = 0.0
        content = result.content.lower() if not self.config.case_sensitive else result.content
        filename = result.file_path.lower() if not self.config.case_sensitive else result.file_path
        
        # Track which terms matched for AND/OR logic
        matched_terms = set()
        not_terms_present = []
        
        for search_term in search_terms:
            term_score = 0.0
            term_matched = False
            
            if search_term.operator == SearchOperator.PHRASE:
                # Exact phrase matching
                if self._phrase_match(search_term.term, content):
                    term_score += 3.0 * search_term.weight
                    term_matched = True
                if self._phrase_match(search_term.term, filename):
                    term_score += 5.0 * search_term.weight  # Higher for filename
                    term_matched = True
            
            elif search_term.operator == SearchOperator.WILDCARD:
                # Wildcard matching
                if self._wildcard_match(search_term.term, content):
                    term_score += 1.5 * search_term.weight
                    term_matched = True
                if self._wildcard_match(search_term.term, filename):
                    term_score += 3.0 * search_term.weight
                    term_matched = True
            
            elif search_term.operator == SearchOperator.NOT:
                # NOT terms - if present, penalize heavily
                if search_term.term in content or search_term.term in filename:
                    not_terms_present.append(search_term.term)
                    return 0.0  # Immediate disqualification
            
            else:  # AND or OR terms
                # Exact word matching
                term_count = self._count_word_occurrences(search_term.term, content)
                if term_count > 0:
                    term_score += term_count * search_term.weight
                    term_matched = True
                
                # Filename matching (higher weight)
                if search_term.term in filename:
                    term_score += self.config.boost_filename_matches * search_term.weight
                    term_matched = True
                
                # Fuzzy matching (if enabled and no exact match)
                if not term_matched and self.config.enable_fuzzy_search:
                    fuzzy_score = self._fuzzy_match_score(search_term.term, content)
                    if fuzzy_score > self.config.fuzzy_threshold:
                        term_score += fuzzy_score * 0.5 * search_term.weight
                        term_matched = True
            
            if term_matched:
                matched_terms.add(search_term.term)
                score += term_score
        
        # Apply AND/OR logic
        and_terms = [t for t in search_terms if t.operator == SearchOperator.AND]
        or_terms = [t for t in search_terms if t.operator == SearchOperator.OR]
        
        # For AND terms, all must be present
        if and_terms:
            and_terms_matched = all(t.term in matched_terms for t in and_terms)
            if not and_terms_matched:
                score *= 0.1  # Heavy penalty for missing AND terms
        
        # For OR terms, at least one should be present
        if or_terms:
            or_terms_matched = any(t.term in matched_terms for t in or_terms)
            if not or_terms_matched:
                score *= 0.3  # Penalty for missing OR terms
        
        # Additional scoring factors
        score = self._apply_additional_scoring(score, result, original_query)
        
        return max(0.0, score)
    
    def _phrase_match(self, phrase: str, text: str) -> bool:
        """Check if phrase exists in text"""
        phrase_lower = phrase.lower() if not self.config.case_sensitive else phrase
        text_lower = text.lower() if not self.config.case_sensitive else text
        return phrase_lower in text_lower
    
    def _wildcard_match(self, pattern: str, text: str) -> bool:
        """Check if wildcard pattern matches text"""
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        
        flags = re.IGNORECASE if not self.config.case_sensitive else 0
        return bool(re.search(regex_pattern, text, flags))
    
    def _count_word_occurrences(self, word: str, text: str) -> int:
        """Count occurrences of a word in text"""
        if not self.config.case_sensitive:
            word = word.lower()
            text = text.lower()
        
        # Use word boundaries to match whole words
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text, re.IGNORECASE if not self.config.case_sensitive else 0)
        return len(matches)
    
    def _fuzzy_match_score(self, term: str, text: str) -> float:
        """Calculate fuzzy match score using simple character-based similarity"""
        # This is a simple implementation - could be enhanced with more sophisticated algorithms
        words = text.split()
        best_score = 0.0
        
        for word in words:
            if not self.config.case_sensitive:
                term_cmp = term.lower()
                word_cmp = word.lower()
            else:
                term_cmp = term
                word_cmp = word
            
            # Simple character-based similarity
            similarity = self._character_similarity(term_cmp, word_cmp)
            best_score = max(best_score, similarity)
        
        return best_score
    
    def _character_similarity(self, s1: str, s2: str) -> float:
        """Calculate character-based similarity between two strings"""
        if not s1 or not s2:
            return 0.0
        
        # Jaccard similarity based on character sets
        set1 = set(s1)
        set2 = set(s2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_additional_scoring(
        self,
        base_score: float,
        result: GitLabSearchResult,
        query: str
    ) -> float:
        """Apply additional scoring factors"""
        score = base_score
        
        # Boost for certain file types
        filename = result.file_path.lower()
        if filename.endswith(('.md', '.txt', '.rst', '.adoc')):
            score *= 1.3  # Documentation files
        elif filename.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
            score *= 1.1  # Code files
        elif filename.endswith(('.json', '.yaml', '.yml', '.xml')):
            score *= 1.05  # Configuration files
        
        # Boost for files with query terms in path
        query_words = query.lower().split()
        for word in query_words:
            if word in filename:
                score *= 1.2
                break
        
        # Penalize very short content
        if len(result.content) < 50:
            score *= 0.7
        
        # Boost for content starting at line 1 (likely more relevant)
        if result.startline <= 5:
            score *= 1.1
        
        return score
    
    def test_search(self, query: str, project_id: Optional[int] = None) -> Dict[str, Any]:
        """Test keyword search functionality
        
        Args:
            query: Test query
            project_id: Optional specific project to search
            
        Returns:
            Test results with timing and statistics
        """
        import time
        
        start_time = time.time()
        
        try:
            project_ids = [project_id] if project_id else None
            results = self.search_projects(query, project_ids=project_ids, limit=10)
            
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            # Parse terms for analysis
            search_terms = self.parse_query(query)
            
            return {
                'success': True,
                'query': query,
                'parsed_terms': [
                    {'term': t.term, 'operator': t.operator.value, 'weight': t.weight}
                    for t in search_terms
                ],
                'results_count': len(results),
                'duration_ms': duration,
                'sample_results': [
                    {
                        'project': r.project_name,
                        'file': r.file_path,
                        'line': r.startline,
                        'content_preview': r.content[:100] + '...'
                    }
                    for r in results[:3]
                ],
                'config': {
                    'case_sensitive': self.config.case_sensitive,
                    'fuzzy_enabled': self.config.enable_fuzzy_search,
                    'phrase_enabled': self.config.enable_phrase_search,
                    'wildcard_enabled': self.config.enable_wildcard_search
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


def create_keyword_search(
    gitlab_client: Optional[GitLabClient] = None,
    case_sensitive: bool = False,
    enable_fuzzy: bool = True,
    fuzzy_threshold: float = 0.8
) -> KeywordSearchStrategy:
    """Factory function to create keyword search strategy
    
    Args:
        gitlab_client: GitLab client instance
        case_sensitive: Whether search should be case sensitive
        enable_fuzzy: Whether to enable fuzzy matching
        fuzzy_threshold: Threshold for fuzzy matching (0-1)
        
    Returns:
        Configured keyword search strategy
    """
    config = KeywordSearchConfig(
        case_sensitive=case_sensitive,
        enable_fuzzy_search=enable_fuzzy,
        fuzzy_threshold=fuzzy_threshold
    )
    
    return KeywordSearchStrategy(gitlab_client, config)


if __name__ == "__main__":
    # Test keyword search functionality
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.integrations.search.keyword_search <query> [project_id]")
        sys.exit(1)
    
    query = sys.argv[1]
    project_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"Testing keyword search with query: '{query}'")
    
    search_strategy = create_keyword_search()
    results = search_strategy.test_search(query, project_id)
    
    print("Test Results:")
    print(f"  Success: {results['success']}")
    print(f"  Duration: {results.get('duration_ms', 0):.1f}ms")
    
    if results['success']:
        print(f"  Parsed terms: {results['parsed_terms']}")
        print(f"  Results found: {results['results_count']}")
        if results['sample_results']:
            print("  Sample results:")
            for result in results['sample_results']:
                print(f"    - {result['project']}: {result['file']} (line {result['line']})")
    else:
        print(f"  Error: {results['error']}")