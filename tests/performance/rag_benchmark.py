"""
RAG Pipeline Performance Benchmark

Benchmarks the end-to-end performance of the RAG pipeline
including retrieval, context assembly, and response generation.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    query: str
    total_time: float
    retrieval_time: float
    context_assembly_time: float
    generation_time: float
    tokens_used: int
    sources_found: int
    
    @property
    def overhead_time(self) -> float:
        """Time spent on overhead (not in main operations)"""
        return max(0, self.total_time - self.retrieval_time - self.context_assembly_time - self.generation_time)


class RAGBenchmark:
    """Benchmark RAG pipeline performance"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    async def benchmark_pipeline(
        self,
        queries: List[str],
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark the complete RAG pipeline"""
        print(f"\n🚀 Running RAG pipeline benchmark...")
        print(f"   Queries: {len(queries)}")
        print(f"   Iterations per query: {iterations}")
        print(f"   Total operations: {len(queries) * iterations}")
        
        start_time = time.time()
        
        for query in queries:
            print(f"\n   Testing: '{query[:60]}...'")
            
            for i in range(iterations):
                result = await self._benchmark_single_query(query)
                self.results.append(result)
                
                if (i + 1) % 10 == 0:
                    avg_time = statistics.mean(r.total_time for r in self.results[-10:])
                    print(f"     Progress: {i+1}/{iterations} (avg: {avg_time*1000:.1f}ms)")
        
        total_time = time.time() - start_time
        
        return self._analyze_results(total_time)
    
    async def _benchmark_single_query(self, query: str) -> BenchmarkResult:
        """Benchmark a single query through the pipeline"""
        start_time = time.time()
        
        # Simulate retrieval phase
        retrieval_start = time.time()
        sources = await self._simulate_retrieval(query)
        retrieval_time = time.time() - retrieval_start
        
        # Simulate context assembly
        assembly_start = time.time()
        context = await self._simulate_context_assembly(sources)
        assembly_time = time.time() - assembly_start
        
        # Simulate generation
        generation_start = time.time()
        response, tokens = await self._simulate_generation(query, context)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        return BenchmarkResult(
            query=query,
            total_time=total_time,
            retrieval_time=retrieval_time,
            context_assembly_time=assembly_time,
            generation_time=generation_time,
            tokens_used=tokens,
            sources_found=len(sources)
        )
    
    async def _simulate_retrieval(self, query: str) -> List[str]:
        """Simulate document retrieval"""
        # Simulate variable retrieval times based on query complexity
        complexity = len(query.split())
        base_time = 0.05  # 50ms base
        
        # Add complexity-based delay
        total_time = base_time + (complexity * 0.01)
        await asyncio.sleep(total_time)
        
        # Return simulated sources
        num_sources = min(10, complexity)
        return [f"source_{i}" for i in range(num_sources)]
    
    async def _simulate_context_assembly(self, sources: List[str]) -> str:
        """Simulate context assembly from sources"""
        # Assembly time depends on number of sources
        base_time = 0.02  # 20ms base
        per_source_time = 0.005  # 5ms per source
        
        total_time = base_time + (len(sources) * per_source_time)
        await asyncio.sleep(total_time)
        
        # Return simulated context
        return f"Context from {len(sources)} sources"
    
    async def _simulate_generation(self, query: str, context: str) -> tuple[str, int]:
        """Simulate response generation"""
        # Generation time varies with output length
        base_time = 0.1  # 100ms base
        
        # Simulate token generation
        estimated_tokens = len(query.split()) * 10
        generation_time = base_time + (estimated_tokens * 0.001)
        
        await asyncio.sleep(generation_time)
        
        return f"Response to {query}", estimated_tokens
    
    def _analyze_results(self, total_benchmark_time: float) -> Dict[str, Any]:
        """Analyze benchmark results"""
        if not self.results:
            return {}
        
        # Calculate statistics for each metric
        total_times = [r.total_time for r in self.results]
        retrieval_times = [r.retrieval_time for r in self.results]
        assembly_times = [r.context_assembly_time for r in self.results]
        generation_times = [r.generation_time for r in self.results]
        overhead_times = [r.overhead_time for r in self.results]
        tokens = [r.tokens_used for r in self.results]
        sources = [r.sources_found for r in self.results]
        
        def calculate_stats(values: List[float]) -> Dict[str, float]:
            """Calculate comprehensive statistics"""
            return {
                'mean': statistics.mean(values),
                'min': min(values),
                'max': max(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'p95': statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0],
                'p99': statistics.quantiles(values, n=100)[98] if len(values) > 2 else values[0]
            }
        
        return {
            'total_benchmark_time': total_benchmark_time,
            'total_queries': len(self.results),
            'queries_per_second': len(self.results) / total_benchmark_time,
            'total_time_stats': calculate_stats(total_times),
            'retrieval_stats': calculate_stats(retrieval_times),
            'assembly_stats': calculate_stats(assembly_times),
            'generation_stats': calculate_stats(generation_times),
            'overhead_stats': calculate_stats(overhead_times),
            'token_stats': calculate_stats(tokens),
            'source_stats': calculate_stats(sources),
            'phase_breakdown': {
                'retrieval_pct': statistics.mean(retrieval_times) / statistics.mean(total_times) * 100,
                'assembly_pct': statistics.mean(assembly_times) / statistics.mean(total_times) * 100,
                'generation_pct': statistics.mean(generation_times) / statistics.mean(total_times) * 100,
                'overhead_pct': statistics.mean(overhead_times) / statistics.mean(total_times) * 100
            }
        }
    
    def print_results(self, analysis: Dict[str, Any]):
        """Print formatted benchmark results"""
        print("\n" + "=" * 80)
        print("🚀 RAG PIPELINE PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)
        
        print(f"\n📊 Overall Performance:")
        print(f"   Total Queries: {analysis['total_queries']}")
        print(f"   Total Time: {analysis['total_benchmark_time']:.2f}s")
        print(f"   Throughput: {analysis['queries_per_second']:.2f} queries/second")
        
        print(f"\n⏱️  End-to-End Response Times (ms):")
        stats = analysis['total_time_stats']
        print(f"   Mean: {stats['mean']*1000:.1f}")
        print(f"   Median: {stats['median']*1000:.1f}")
        print(f"   Min: {stats['min']*1000:.1f}")
        print(f"   Max: {stats['max']*1000:.1f}")
        print(f"   P95: {stats['p95']*1000:.1f}")
        print(f"   P99: {stats['p99']*1000:.1f}")
        print(f"   StdDev: {stats['stdev']*1000:.1f}")
        
        print(f"\n📈 Phase Breakdown:")
        breakdown = analysis['phase_breakdown']
        
        print(f"\n   1. Retrieval Phase ({breakdown['retrieval_pct']:.1f}%):")
        rstats = analysis['retrieval_stats']
        print(f"      Mean: {rstats['mean']*1000:.1f}ms")
        print(f"      P95: {rstats['p95']*1000:.1f}ms")
        
        print(f"\n   2. Context Assembly ({breakdown['assembly_pct']:.1f}%):")
        astats = analysis['assembly_stats']
        print(f"      Mean: {astats['mean']*1000:.1f}ms")
        print(f"      P95: {astats['p95']*1000:.1f}ms")
        
        print(f"\n   3. Generation Phase ({breakdown['generation_pct']:.1f}%):")
        gstats = analysis['generation_stats']
        print(f"      Mean: {gstats['mean']*1000:.1f}ms")
        print(f"      P95: {gstats['p95']*1000:.1f}ms")
        
        print(f"\n   4. Overhead ({breakdown['overhead_pct']:.1f}%):")
        ostats = analysis['overhead_stats']
        print(f"      Mean: {ostats['mean']*1000:.1f}ms")
        
        print(f"\n🔢 Resource Usage:")
        print(f"   Avg Sources Retrieved: {analysis['source_stats']['mean']:.1f}")
        print(f"   Avg Tokens Generated: {analysis['token_stats']['mean']:.0f}")
        
        # Performance rating
        mean_time_ms = stats['mean'] * 1000
        if mean_time_ms < 500:
            rating = "⚡ Excellent"
        elif mean_time_ms < 1000:
            rating = "✅ Good"
        elif mean_time_ms < 2000:
            rating = "👍 Acceptable"
        elif mean_time_ms < 3000:
            rating = "⚠️  Needs Optimization"
        else:
            rating = "❌ Poor"
        
        print(f"\n🏆 Performance Rating: {rating}")
        print("=" * 80)
    
    def save_detailed_results(self, filename: str = "rag_benchmark_results.csv"):
        """Save detailed results to CSV"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'query', 'total_time_ms', 'retrieval_ms', 
                'assembly_ms', 'generation_ms', 'overhead_ms',
                'tokens_used', 'sources_found'
            ])
            
            for r in self.results:
                writer.writerow([
                    datetime.now().isoformat(),
                    r.query[:50],
                    f"{r.total_time*1000:.2f}",
                    f"{r.retrieval_time*1000:.2f}",
                    f"{r.context_assembly_time*1000:.2f}",
                    f"{r.generation_time*1000:.2f}",
                    f"{r.overhead_time*1000:.2f}",
                    r.tokens_used,
                    r.sources_found
                ])
        
        print(f"\n💾 Detailed results saved to {filename}")


async def main():
    """Run RAG pipeline benchmark"""
    benchmark = RAGBenchmark()
    
    # Test queries of varying complexity
    test_queries = [
        "What is authentication?",
        "How do I configure GitLab integration?",
        "Explain the differences between OAuth2 and SAML authentication methods",
        "What are the best practices for securing API tokens in production?",
        "How can I implement rate limiting for API endpoints?",
        "Describe the process of setting up a CI/CD pipeline with GitLab"
    ]
    
    # Run benchmark with fewer iterations for testing
    # In production, use more iterations (e.g., 100-1000)
    results = await benchmark.benchmark_pipeline(test_queries, iterations=20)
    
    # Print results
    benchmark.print_results(results)
    
    # Save detailed results
    benchmark.save_detailed_results()


if __name__ == "__main__":
    asyncio.run(main())