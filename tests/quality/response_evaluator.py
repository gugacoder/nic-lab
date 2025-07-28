"""
Response Quality Evaluator

Evaluates the quality of AI-generated responses for accuracy,
relevance, and alignment with corporate guidelines.
"""

import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class QualityMetric(Enum):
    """Quality evaluation metrics"""
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    TONE = "tone"
    SOURCE_ATTRIBUTION = "source_attribution"


@dataclass
class EvaluationResult:
    """Result of quality evaluation"""
    query: str
    response: str
    scores: Dict[QualityMetric, float]
    overall_score: float
    feedback: List[str]
    
    def is_passing(self, threshold: float = 0.7) -> bool:
        """Check if response meets quality threshold"""
        return self.overall_score >= threshold


class ResponseEvaluator:
    """Evaluate response quality"""
    
    def __init__(self):
        self.evaluation_criteria = {
            QualityMetric.RELEVANCE: 0.25,       # Weight: 25%
            QualityMetric.ACCURACY: 0.25,        # Weight: 25%
            QualityMetric.COMPLETENESS: 0.20,    # Weight: 20%
            QualityMetric.CLARITY: 0.15,         # Weight: 15%
            QualityMetric.TONE: 0.10,            # Weight: 10%
            QualityMetric.SOURCE_ATTRIBUTION: 0.05  # Weight: 5%
        }
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        expected_content: List[str] = None,
        sources: List[str] = None
    ) -> EvaluationResult:
        """Evaluate a single response"""
        scores = {}
        feedback = []
        
        # Evaluate each metric
        scores[QualityMetric.RELEVANCE] = await self._evaluate_relevance(query, response)
        scores[QualityMetric.ACCURACY] = await self._evaluate_accuracy(response, expected_content)
        scores[QualityMetric.COMPLETENESS] = await self._evaluate_completeness(query, response)
        scores[QualityMetric.CLARITY] = await self._evaluate_clarity(response)
        scores[QualityMetric.TONE] = await self._evaluate_tone(response)
        scores[QualityMetric.SOURCE_ATTRIBUTION] = await self._evaluate_sources(response, sources)
        
        # Calculate weighted overall score
        overall_score = sum(
            scores[metric] * weight 
            for metric, weight in self.evaluation_criteria.items()
        )
        
        # Generate feedback
        for metric, score in scores.items():
            if score < 0.7:
                feedback.append(f"{metric.value}: Score {score:.2f} - Needs improvement")
        
        return EvaluationResult(
            query=query,
            response=response,
            scores=scores,
            overall_score=overall_score,
            feedback=feedback
        )
    
    async def _evaluate_relevance(self, query: str, response: str) -> float:
        """Evaluate if response is relevant to query"""
        # Simple keyword-based relevance check
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of'}
        query_words -= common_words
        response_words -= common_words
        
        if not query_words:
            return 1.0
        
        # Calculate overlap
        overlap = len(query_words & response_words)
        relevance = min(1.0, overlap / len(query_words) + 0.3)  # Base score of 0.3
        
        return relevance
    
    async def _evaluate_accuracy(self, response: str, expected_content: List[str]) -> float:
        """Evaluate accuracy based on expected content"""
        if not expected_content:
            return 0.8  # Default score when no expectations provided
        
        found_count = 0
        for expected in expected_content:
            if expected.lower() in response.lower():
                found_count += 1
        
        return found_count / len(expected_content)
    
    async def _evaluate_completeness(self, query: str, response: str) -> float:
        """Evaluate if response is complete"""
        # Check response length relative to query complexity
        query_words = len(query.split())
        response_words = len(response.split())
        
        # Expect at least 5x the query length for good answers
        if response_words < query_words * 2:
            return 0.3
        elif response_words < query_words * 5:
            return 0.6
        else:
            return min(1.0, 0.8 + (response_words / (query_words * 10)) * 0.2)
    
    async def _evaluate_clarity(self, response: str) -> float:
        """Evaluate response clarity"""
        # Simple heuristics for clarity
        sentences = response.split('.')
        
        # Check for structure
        has_structure = len(sentences) > 1
        
        # Check for bullet points or numbering
        has_formatting = any(marker in response for marker in ['•', '-', '1.', '2.', '*'])
        
        # Check average sentence length
        avg_sentence_length = len(response.split()) / max(1, len(sentences))
        good_length = 10 <= avg_sentence_length <= 25
        
        score = 0.5  # Base score
        if has_structure:
            score += 0.2
        if has_formatting:
            score += 0.2
        if good_length:
            score += 0.1
        
        return min(1.0, score)
    
    async def _evaluate_tone(self, response: str) -> float:
        """Evaluate professional tone"""
        # Check for professional language
        professional_indicators = [
            'please', 'thank you', 'would', 'could', 'recommend',
            'suggest', 'consider', 'additionally', 'furthermore'
        ]
        
        casual_indicators = [
            'hey', 'yeah', 'nope', 'gonna', 'wanna', 'lol', 'btw'
        ]
        
        response_lower = response.lower()
        
        professional_count = sum(1 for word in professional_indicators if word in response_lower)
        casual_count = sum(1 for word in casual_indicators if word in response_lower)
        
        # Calculate tone score
        if casual_count > 0:
            return max(0.3, 0.7 - casual_count * 0.1)
        else:
            return min(1.0, 0.7 + professional_count * 0.05)
    
    async def _evaluate_sources(self, response: str, sources: List[str]) -> float:
        """Evaluate source attribution"""
        if not sources:
            # If no sources expected, check if response claims any
            if 'according to' in response.lower() or 'source:' in response.lower():
                return 1.0
            return 0.8
        
        # Check if sources are mentioned
        mentioned = 0
        for source in sources:
            if source in response:
                mentioned += 1
        
        return mentioned / len(sources)
    
    async def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Tuple[List[EvaluationResult], Dict[str, Any]]:
        """Evaluate multiple test cases"""
        results = []
        
        for case in test_cases:
            result = await self.evaluate_response(
                query=case['query'],
                response=case['response'],
                expected_content=case.get('expected', []),
                sources=case.get('sources', [])
            )
            results.append(result)
        
        # Calculate aggregate statistics
        stats = self._calculate_statistics(results)
        
        return results, stats
    
    def _calculate_statistics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate aggregate statistics"""
        if not results:
            return {}
        
        # Overall statistics
        overall_scores = [r.overall_score for r in results]
        passing = sum(1 for r in results if r.is_passing())
        
        # Per-metric statistics
        metric_scores = {metric: [] for metric in QualityMetric}
        for result in results:
            for metric, score in result.scores.items():
                metric_scores[metric].append(score)
        
        metric_stats = {}
        for metric, scores in metric_scores.items():
            metric_stats[metric.value] = {
                'average': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores)
            }
        
        return {
            'total_evaluated': len(results),
            'passing': passing,
            'pass_rate': passing / len(results),
            'average_score': sum(overall_scores) / len(overall_scores),
            'min_score': min(overall_scores),
            'max_score': max(overall_scores),
            'metric_breakdown': metric_stats
        }
    
    def print_evaluation_report(self, results: List[EvaluationResult], stats: Dict[str, Any]):
        """Print detailed evaluation report"""
        print("\n" + "=" * 70)
        print("📊 RESPONSE QUALITY EVALUATION REPORT")
        print("=" * 70)
        
        print(f"\nTotal Responses Evaluated: {stats['total_evaluated']}")
        print(f"Passing Responses: {stats['passing']} ({stats['pass_rate']*100:.1f}%)")
        print(f"Average Quality Score: {stats['average_score']:.2f}")
        print(f"Score Range: {stats['min_score']:.2f} - {stats['max_score']:.2f}")
        
        print("\n📈 Metric Breakdown:")
        for metric, metric_stats in stats['metric_breakdown'].items():
            print(f"\n  {metric.title()}:")
            print(f"    Average: {metric_stats['average']:.2f}")
            print(f"    Range: {metric_stats['min']:.2f} - {metric_stats['max']:.2f}")
        
        # Show failing responses
        failing = [r for r in results if not r.is_passing()]
        if failing:
            print(f"\n⚠️  Failing Responses ({len(failing)}):")
            for i, result in enumerate(failing[:5]):  # Show first 5
                print(f"\n  {i+1}. Query: {result.query[:50]}...")
                print(f"     Score: {result.overall_score:.2f}")
                print(f"     Issues: {', '.join(result.feedback)}")
        
        print("\n" + "=" * 70)


async def main():
    """Run response quality evaluation tests"""
    evaluator = ResponseEvaluator()
    
    # Test cases
    test_cases = [
        {
            'query': "How do I configure GitLab authentication?",
            'response': """To configure GitLab authentication, follow these steps:

1. Create a personal access token in GitLab:
   - Go to User Settings > Access Tokens
   - Choose 'api' scope
   - Generate and copy the token

2. Set the environment variable:
   ```
   export GITLAB_PRIVATE_TOKEN=your-token-here
   export GITLAB_URL=https://your-gitlab-instance.com
   ```

3. Verify the connection using the authentication test.

For more details, refer to the GitLab API documentation.""",
            'expected': ['personal access token', 'environment variable', 'api scope'],
            'sources': ['GitLab API documentation']
        },
        {
            'query': "What is the capital of France?",
            'response': "The capital of France is Paris.",
            'expected': ['Paris']
        },
        {
            'query': "Explain async programming in Python",
            'response': "async programming is when you use async and await",
            'expected': ['async', 'await', 'coroutine', 'event loop']
        }
    ]
    
    # Evaluate responses
    results, stats = await evaluator.batch_evaluate(test_cases)
    
    # Print report
    evaluator.print_evaluation_report(results, stats)
    
    # Show detailed results for each test
    print("\n📋 Detailed Results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Query: {result.query}")
        print(f"   Overall Score: {result.overall_score:.2f} {'✅' if result.is_passing() else '❌'}")
        print(f"   Scores: {', '.join(f'{m.value}={s:.2f}' for m, s in result.scores.items())}")


if __name__ == "__main__":
    asyncio.run(main())