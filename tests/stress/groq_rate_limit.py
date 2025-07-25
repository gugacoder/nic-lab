"""
Groq API Rate Limit Stress Testing

Tests the rate limiting behavior under various load conditions
to ensure the system handles API limits gracefully.
"""

import asyncio
import time
import statistics
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ai.groq_client import GroqClient, RateLimitError


@dataclass
class RequestResult:
    """Result of a single request"""
    request_id: int
    start_time: float
    end_time: float
    success: bool
    error: str = ""
    tokens_used: int = 0
    response_time: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def wait_time(self) -> float:
        return self.duration - self.response_time


class RateLimitStressTester:
    """Comprehensive rate limit stress testing"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.results: List[RequestResult] = []
    
    async def run_burst_test(
        self, 
        requests: int = 50,
        prompt: str = "Say hello",
        max_tokens: int = 10
    ) -> Dict[str, Any]:
        """Test burst of requests to trigger rate limiting"""
        print(f"\n🚀 Running burst test with {requests} requests...")
        
        async with GroqClient(api_key=self.api_key) as client:
            # Set aggressive rate limit for testing
            client.rate_limiter.requests_per_minute = 30
            
            async def make_request(request_id: int) -> RequestResult:
                start_time = time.time()
                try:
                    response = await client.complete(
                        f"{prompt} #{request_id}",
                        max_tokens=max_tokens,
                        cache=False
                    )
                    end_time = time.time()
                    
                    return RequestResult(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        success=True,
                        tokens_used=response.usage.get("total_tokens", 0),
                        response_time=response.response_time
                    )
                
                except RateLimitError as e:
                    end_time = time.time()
                    return RequestResult(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        success=False,
                        error=str(e)
                    )
                
                except Exception as e:
                    end_time = time.time()
                    return RequestResult(
                        request_id=request_id,
                        start_time=start_time,
                        end_time=end_time,
                        success=False,
                        error=str(e)
                    )
            
            # Launch all requests concurrently
            test_start = time.time()
            self.results = await asyncio.gather(*[
                make_request(i) for i in range(requests)
            ])
            test_duration = time.time() - test_start
        
        return self._analyze_results(test_duration)
    
    async def run_sustained_test(
        self,
        duration_seconds: int = 120,
        requests_per_second: float = 1.0
    ) -> Dict[str, Any]:
        """Test sustained load over time"""
        print(f"\n⏱️  Running sustained test for {duration_seconds}s at {requests_per_second} req/s...")
        
        async with GroqClient(api_key=self.api_key) as client:
            self.results = []
            request_id = 0
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                # Schedule next request
                request_start = time.time()
                
                # Make request
                result = await self._make_single_request(client, request_id)
                self.results.append(result)
                request_id += 1
                
                # Wait for next request slot
                elapsed = time.time() - request_start
                sleep_time = max(0, (1.0 / requests_per_second) - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            test_duration = time.time() - start_time
        
        return self._analyze_results(test_duration)
    
    async def run_pattern_test(
        self,
        pattern: List[Tuple[int, float]]
    ) -> Dict[str, Any]:
        """Test specific request patterns
        
        Args:
            pattern: List of (num_requests, delay_seconds) tuples
        """
        print(f"\n📊 Running pattern test with {len(pattern)} phases...")
        
        async with GroqClient(api_key=self.api_key) as client:
            self.results = []
            request_id = 0
            test_start = time.time()
            
            for phase, (num_requests, delay) in enumerate(pattern):
                print(f"  Phase {phase + 1}: {num_requests} requests with {delay}s delay")
                
                # Make requests for this phase
                phase_tasks = []
                for _ in range(num_requests):
                    phase_tasks.append(self._make_single_request(client, request_id))
                    request_id += 1
                
                # Execute phase requests
                phase_results = await asyncio.gather(*phase_tasks)
                self.results.extend(phase_results)
                
                # Delay before next phase
                if phase < len(pattern) - 1:
                    await asyncio.sleep(delay)
            
            test_duration = time.time() - test_start
        
        return self._analyze_results(test_duration)
    
    async def _make_single_request(
        self, 
        client: GroqClient, 
        request_id: int
    ) -> RequestResult:
        """Make a single request and record result"""
        start_time = time.time()
        try:
            response = await client.complete(
                f"Request {request_id}: What is {request_id} + {request_id}?",
                max_tokens=20,
                cache=False
            )
            end_time = time.time()
            
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                success=True,
                tokens_used=response.usage.get("total_tokens", 0),
                response_time=response.response_time
            )
        
        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                success=False,
                error=str(e)
            )
    
    def _analyze_results(self, test_duration: float) -> Dict[str, Any]:
        """Analyze test results and compute statistics"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        # Compute statistics
        analysis = {
            "test_duration": test_duration,
            "total_requests": len(self.results),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
            "requests_per_second": len(self.results) / test_duration if test_duration > 0 else 0,
        }
        
        if successful:
            response_times = [r.response_time for r in successful]
            wait_times = [r.wait_time for r in successful]
            tokens = [r.tokens_used for r in successful]
            
            analysis.update({
                "avg_response_time": statistics.mean(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0],
                "avg_wait_time": statistics.mean(wait_times),
                "max_wait_time": max(wait_times),
                "total_tokens": sum(tokens),
                "avg_tokens_per_request": statistics.mean(tokens) if tokens else 0,
            })
        
        # Analyze rate limiting behavior
        if failed:
            rate_limit_errors = [r for r in failed if "rate limit" in r.error.lower()]
            analysis["rate_limit_errors"] = len(rate_limit_errors)
        
        # Time-based analysis
        if self.results:
            sorted_results = sorted(self.results, key=lambda r: r.start_time)
            
            # Calculate request distribution over time
            time_windows = {}
            window_size = 1.0  # 1 second windows
            
            for result in sorted_results:
                window = int(result.start_time / window_size)
                if window not in time_windows:
                    time_windows[window] = {"requests": 0, "successes": 0}
                
                time_windows[window]["requests"] += 1
                if result.success:
                    time_windows[window]["successes"] += 1
            
            # Find peak request rate
            peak_requests = max(w["requests"] for w in time_windows.values()) if time_windows else 0
            analysis["peak_requests_per_second"] = peak_requests
        
        return analysis
    
    def print_results(self, analysis: Dict[str, Any]):
        """Print formatted test results"""
        print("\n" + "=" * 60)
        print("📈 RATE LIMIT STRESS TEST RESULTS")
        print("=" * 60)
        
        print(f"\n⏱️  Test Duration: {analysis['test_duration']:.2f} seconds")
        print(f"📊 Total Requests: {analysis['total_requests']}")
        print(f"✅ Successful: {analysis['successful_requests']} ({analysis['success_rate']*100:.1f}%)")
        print(f"❌ Failed: {analysis['failed_requests']}")
        
        if analysis.get('rate_limit_errors'):
            print(f"⚠️  Rate Limit Errors: {analysis['rate_limit_errors']}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Requests/second: {analysis['requests_per_second']:.2f}")
        print(f"   Peak requests/second: {analysis.get('peak_requests_per_second', 0)}")
        
        if analysis['successful_requests'] > 0:
            print(f"\n⚡ Response Times:")
            print(f"   Average: {analysis['avg_response_time']*1000:.1f}ms")
            print(f"   Min: {analysis['min_response_time']*1000:.1f}ms")
            print(f"   Max: {analysis['max_response_time']*1000:.1f}ms")
            print(f"   P95: {analysis['p95_response_time']*1000:.1f}ms")
            
            print(f"\n⏳ Wait Times (Rate Limiting):")
            print(f"   Average: {analysis['avg_wait_time']*1000:.1f}ms")
            print(f"   Max: {analysis['max_wait_time']*1000:.1f}ms")
            
            print(f"\n🔢 Token Usage:")
            print(f"   Total: {analysis['total_tokens']}")
            print(f"   Average per request: {analysis['avg_tokens_per_request']:.1f}")
        
        print("\n" + "=" * 60)
    
    def save_detailed_results(self, filename: str = "rate_limit_test_results.csv"):
        """Save detailed results to CSV file"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_id", "start_time", "end_time", "duration",
                "success", "error", "tokens_used", "response_time", "wait_time"
            ])
            
            for r in self.results:
                writer.writerow([
                    r.request_id, r.start_time, r.end_time, r.duration,
                    r.success, r.error, r.tokens_used, r.response_time, r.wait_time
                ])
        
        print(f"\n💾 Detailed results saved to {filename}")


async def main():
    """Run comprehensive rate limit stress tests"""
    import os
    
    # Check for API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "test-key":
        print("⚠️  Warning: Running in mock mode without real API key")
        print("   Set GROQ_API_KEY environment variable for real tests")
        api_key = "test-key"
    
    tester = RateLimitStressTester(api_key)
    
    # Test 1: Burst test
    print("\n🧪 Test 1: Burst Test")
    results1 = await tester.run_burst_test(requests=40)
    tester.print_results(results1)
    
    # Test 2: Sustained load test
    print("\n🧪 Test 2: Sustained Load Test")
    results2 = await tester.run_sustained_test(duration_seconds=30, requests_per_second=1.5)
    tester.print_results(results2)
    
    # Test 3: Pattern test (simulate real-world usage)
    print("\n🧪 Test 3: Pattern Test")
    pattern = [
        (5, 2.0),   # 5 requests, then 2s pause
        (10, 1.0),  # 10 requests, then 1s pause
        (20, 0.5),  # 20 requests, then 0.5s pause
        (30, 0),    # 30 requests burst
    ]
    results3 = await tester.run_pattern_test(pattern)
    tester.print_results(results3)
    
    # Save detailed results
    tester.save_detailed_results()
    
    print("\n✅ All stress tests completed!")


if __name__ == "__main__":
    asyncio.run(main())