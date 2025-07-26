"""
Search Performance Metrics Monitoring

This module provides comprehensive metrics collection and analysis for search
operations, enabling performance monitoring and optimization insights.
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
import threading
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CACHE_HIT_RATE = "cache_hit_rate"
    RESULT_COUNT = "result_count"
    ERROR_RATE = "error_rate"
    QUERY_COMPLEXITY = "query_complexity"
    PARALLEL_EFFICIENCY = "parallel_efficiency"


@dataclass
class SearchMetric:
    """Individual search metric data point"""
    timestamp: datetime
    query: str
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'query': self.query,
            'metric_type': self.metric_type.value,
            'value': self.value,
            'metadata': self.metadata
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time period"""
    period_start: datetime
    period_end: datetime
    metric_type: MetricType
    count: int
    mean: float
    median: float
    min: float
    max: float
    p95: float
    p99: float
    std_dev: float


@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    # Collection settings
    enable_metrics: bool = True
    metrics_retention_days: int = 7
    aggregation_interval_seconds: int = 300  # 5 minutes
    
    # Performance thresholds
    latency_threshold_ms: float = 2000  # 2 seconds
    error_rate_threshold: float = 0.05  # 5%
    cache_hit_threshold: float = 0.6  # 60%
    
    # Storage settings
    metrics_file_path: str = "metrics/search_metrics.json"
    max_memory_metrics: int = 10000
    
    # Monitoring settings
    enable_alerts: bool = True
    alert_cooldown_minutes: int = 15


class SearchMetricsCollector:
    """Comprehensive search metrics collection and analysis system
    
    This class collects detailed metrics about search operations including
    latency, throughput, cache performance, and query patterns.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """Initialize metrics collector
        
        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        
        # Metrics storage
        self._metrics: Dict[MetricType, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_memory_metrics)
        )
        
        # Real-time statistics
        self._realtime_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_results': 0,
            'start_time': datetime.now()
        }
        
        # Alert tracking
        self._last_alerts: Dict[str, datetime] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing metrics
        self._load_metrics()
        
        # Start aggregation thread
        if self.config.enable_metrics:
            self._start_aggregation_thread()
    
    def record_search(
        self,
        query: str,
        latency_ms: float,
        result_count: int,
        cache_hit: bool,
        success: bool,
        strategy: str = "hybrid",
        project_ids: Optional[List[int]] = None,
        error: Optional[str] = None
    ):
        """Record metrics for a search operation
        
        Args:
            query: Search query
            latency_ms: Search latency in milliseconds
            result_count: Number of results returned
            cache_hit: Whether the search hit cache
            success: Whether the search succeeded
            strategy: Search strategy used
            project_ids: Projects searched
            error: Error message if failed
        """
        if not self.config.enable_metrics:
            return
        
        with self._lock:
            timestamp = datetime.now()
            
            # Record latency
            self._record_metric(
                SearchMetric(
                    timestamp=timestamp,
                    query=query,
                    metric_type=MetricType.LATENCY,
                    value=latency_ms,
                    metadata={
                        'strategy': strategy,
                        'cache_hit': cache_hit,
                        'project_count': len(project_ids) if project_ids else 0
                    }
                )
            )
            
            # Record result count
            self._record_metric(
                SearchMetric(
                    timestamp=timestamp,
                    query=query,
                    metric_type=MetricType.RESULT_COUNT,
                    value=result_count,
                    metadata={'strategy': strategy}
                )
            )
            
            # Update real-time stats
            self._realtime_stats['total_searches'] += 1
            
            if success:
                self._realtime_stats['successful_searches'] += 1
                self._realtime_stats['total_results'] += result_count
            else:
                self._realtime_stats['failed_searches'] += 1
            
            if cache_hit:
                self._realtime_stats['cache_hits'] += 1
            else:
                self._realtime_stats['cache_misses'] += 1
            
            # Check for alerts
            if self.config.enable_alerts:
                self._check_alerts(latency_ms, success, cache_hit)
            
            logger.debug(f"Recorded search metrics: query='{query}', latency={latency_ms:.1f}ms, results={result_count}")
    
    def record_parallel_search(
        self,
        query: str,
        total_latency_ms: float,
        parallel_tasks: int,
        sequential_estimate_ms: float,
        results_per_task: List[int]
    ):
        """Record metrics for parallel search operations
        
        Args:
            query: Search query
            total_latency_ms: Total parallel execution time
            parallel_tasks: Number of parallel tasks
            sequential_estimate_ms: Estimated sequential time
            results_per_task: Results from each parallel task
        """
        if not self.config.enable_metrics:
            return
        
        with self._lock:
            # Calculate parallel efficiency
            speedup = sequential_estimate_ms / total_latency_ms if total_latency_ms > 0 else 0
            efficiency = speedup / parallel_tasks if parallel_tasks > 0 else 0
            
            self._record_metric(
                SearchMetric(
                    timestamp=datetime.now(),
                    query=query,
                    metric_type=MetricType.PARALLEL_EFFICIENCY,
                    value=efficiency,
                    metadata={
                        'parallel_tasks': parallel_tasks,
                        'speedup': speedup,
                        'total_latency_ms': total_latency_ms,
                        'avg_results_per_task': statistics.mean(results_per_task) if results_per_task else 0
                    }
                )
            )
    
    def record_query_complexity(
        self,
        query: str,
        complexity_score: float,
        keywords: List[str],
        query_type: str
    ):
        """Record query complexity metrics
        
        Args:
            query: Search query
            complexity_score: Calculated complexity (0-1)
            keywords: Extracted keywords
            query_type: Type of query
        """
        if not self.config.enable_metrics:
            return
        
        self._record_metric(
            SearchMetric(
                timestamp=datetime.now(),
                query=query,
                metric_type=MetricType.QUERY_COMPLEXITY,
                value=complexity_score,
                metadata={
                    'keyword_count': len(keywords),
                    'query_type': query_type,
                    'query_length': len(query)
                }
            )
        )
    
    def _record_metric(self, metric: SearchMetric):
        """Internal method to record a metric"""
        self._metrics[metric.metric_type].append(metric)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current real-time statistics
        
        Returns:
            Dictionary of current statistics
        """
        with self._lock:
            stats = self._realtime_stats.copy()
            
            # Calculate rates
            uptime_seconds = (datetime.now() - stats['start_time']).total_seconds()
            if uptime_seconds > 0:
                stats['searches_per_minute'] = (stats['total_searches'] / uptime_seconds) * 60
            else:
                stats['searches_per_minute'] = 0
            
            # Calculate success rate
            if stats['total_searches'] > 0:
                stats['success_rate'] = stats['successful_searches'] / stats['total_searches']
                stats['error_rate'] = stats['failed_searches'] / stats['total_searches']
                stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_searches']
                stats['avg_results_per_search'] = stats['total_results'] / stats['successful_searches'] if stats['successful_searches'] > 0 else 0
            else:
                stats['success_rate'] = 0
                stats['error_rate'] = 0
                stats['cache_hit_rate'] = 0
                stats['avg_results_per_search'] = 0
            
            return stats
    
    def get_metrics_summary(
        self,
        metric_type: Optional[MetricType] = None,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get summary of metrics over a time window
        
        Args:
            metric_type: Specific metric type or None for all
            time_window_minutes: Time window to analyze
            
        Returns:
            Metrics summary
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            if metric_type:
                metrics_to_analyze = {metric_type: self._metrics[metric_type]}
            else:
                metrics_to_analyze = self._metrics
            
            summary = {}
            
            for m_type, metrics_deque in metrics_to_analyze.items():
                # Filter metrics within time window
                recent_metrics = [
                    m for m in metrics_deque
                    if m.timestamp >= cutoff_time
                ]
                
                if not recent_metrics:
                    continue
                
                values = [m.value for m in recent_metrics]
                
                summary[m_type.value] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'p95': self._percentile(values, 0.95),
                    'p99': self._percentile(values, 0.99)
                }
                
                # Add metric-specific insights
                if m_type == MetricType.LATENCY:
                    slow_queries = sum(1 for v in values if v > self.config.latency_threshold_ms)
                    summary[m_type.value]['slow_query_ratio'] = slow_queries / len(values) if values else 0
                
                elif m_type == MetricType.CACHE_HIT_RATE:
                    # For cache hit rate, we need to calculate from metadata
                    cache_hits = sum(1 for m in recent_metrics if m.metadata.get('cache_hit', False))
                    summary[m_type.value]['cache_hit_rate'] = cache_hits / len(recent_metrics) if recent_metrics else 0
            
            return summary
    
    def get_query_patterns(self, top_n: int = 10) -> Dict[str, Any]:
        """Analyze query patterns
        
        Args:
            top_n: Number of top patterns to return
            
        Returns:
            Query pattern analysis
        """
        with self._lock:
            # Count query frequencies
            query_counts = defaultdict(int)
            query_latencies = defaultdict(list)
            query_results = defaultdict(list)
            
            for metric in self._metrics[MetricType.LATENCY]:
                query_counts[metric.query] += 1
                query_latencies[metric.query].append(metric.value)
            
            for metric in self._metrics[MetricType.RESULT_COUNT]:
                query_results[metric.query].append(metric.value)
            
            # Sort by frequency
            top_queries = sorted(
                query_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            # Build pattern analysis
            patterns = {
                'top_queries': [],
                'total_unique_queries': len(query_counts),
                'query_distribution': {}
            }
            
            for query, count in top_queries:
                avg_latency = statistics.mean(query_latencies[query]) if query_latencies[query] else 0
                avg_results = statistics.mean(query_results[query]) if query_results[query] else 0
                
                patterns['top_queries'].append({
                    'query': query,
                    'count': count,
                    'avg_latency_ms': avg_latency,
                    'avg_results': avg_results
                })
            
            # Calculate query length distribution
            query_lengths = [len(q) for q in query_counts.keys()]
            if query_lengths:
                patterns['query_distribution'] = {
                    'avg_length': statistics.mean(query_lengths),
                    'min_length': min(query_lengths),
                    'max_length': max(query_lengths)
                }
            
            return patterns
    
    def get_performance_trends(
        self,
        metric_type: MetricType,
        interval_minutes: int = 60,
        periods: int = 24
    ) -> List[Dict[str, Any]]:
        """Get performance trends over time
        
        Args:
            metric_type: Metric to analyze
            interval_minutes: Interval for each period
            periods: Number of periods to analyze
            
        Returns:
            List of trend data points
        """
        with self._lock:
            trends = []
            now = datetime.now()
            
            for i in range(periods):
                period_end = now - timedelta(minutes=i * interval_minutes)
                period_start = period_end - timedelta(minutes=interval_minutes)
                
                # Get metrics in this period
                period_metrics = [
                    m for m in self._metrics[metric_type]
                    if period_start <= m.timestamp < period_end
                ]
                
                if period_metrics:
                    values = [m.value for m in period_metrics]
                    trend_point = {
                        'period_start': period_start.isoformat(),
                        'period_end': period_end.isoformat(),
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'max': max(values)
                    }
                else:
                    trend_point = {
                        'period_start': period_start.isoformat(),
                        'period_end': period_end.isoformat(),
                        'count': 0,
                        'mean': 0,
                        'median': 0,
                        'max': 0
                    }
                
                trends.append(trend_point)
            
            # Reverse to chronological order
            trends.reverse()
            
            return trends
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Comprehensive performance analysis
        
        Returns:
            Performance analysis with insights and recommendations
        """
        stats = self.get_current_stats()
        summary = self.get_metrics_summary(time_window_minutes=60)
        patterns = self.get_query_patterns()
        
        analysis = {
            'overall_health': 'good',  # Will be updated based on analysis
            'current_stats': stats,
            'metrics_summary': summary,
            'query_patterns': patterns,
            'insights': [],
            'recommendations': []
        }
        
        # Analyze latency
        if MetricType.LATENCY.value in summary:
            latency_stats = summary[MetricType.LATENCY.value]
            if latency_stats['p95'] > self.config.latency_threshold_ms:
                analysis['overall_health'] = 'degraded'
                analysis['insights'].append(
                    f"High p95 latency: {latency_stats['p95']:.0f}ms (threshold: {self.config.latency_threshold_ms}ms)"
                )
                analysis['recommendations'].append("Consider increasing cache size or warming cache for popular queries")
        
        # Analyze cache performance
        if stats['cache_hit_rate'] < self.config.cache_hit_threshold:
            analysis['overall_health'] = 'degraded' if analysis['overall_health'] == 'good' else 'poor'
            analysis['insights'].append(
                f"Low cache hit rate: {stats['cache_hit_rate']:.2%} (threshold: {self.config.cache_hit_threshold:.0%})"
            )
            analysis['recommendations'].append("Enable cache warming for frequently used queries")
        
        # Analyze error rate
        if stats['error_rate'] > self.config.error_rate_threshold:
            analysis['overall_health'] = 'poor'
            analysis['insights'].append(
                f"High error rate: {stats['error_rate']:.2%} (threshold: {self.config.error_rate_threshold:.0%})"
            )
            analysis['recommendations'].append("Investigate error patterns and increase timeouts if needed")
        
        # Analyze query patterns
        if patterns['total_unique_queries'] > 0:
            top_query_percentage = patterns['top_queries'][0]['count'] / stats['total_searches'] if patterns['top_queries'] else 0
            if top_query_percentage > 0.2:  # Top query is >20% of traffic
                analysis['insights'].append(
                    f"High query concentration: top query accounts for {top_query_percentage:.0%} of searches"
                )
                analysis['recommendations'].append("Optimize caching for top queries")
        
        # Analyze parallel efficiency if available
        if MetricType.PARALLEL_EFFICIENCY.value in summary:
            efficiency_stats = summary[MetricType.PARALLEL_EFFICIENCY.value]
            if efficiency_stats['mean'] < 0.7:  # Less than 70% efficiency
                analysis['insights'].append(
                    f"Low parallel search efficiency: {efficiency_stats['mean']:.2f}"
                )
                analysis['recommendations'].append("Review parallel search configuration and reduce task granularity")
        
        return analysis
    
    def _check_alerts(self, latency_ms: float, success: bool, cache_hit: bool):
        """Check if alerts should be triggered
        
        Args:
            latency_ms: Search latency
            success: Whether search succeeded
            cache_hit: Whether cache was hit
        """
        now = datetime.now()
        
        # Check latency alert
        if latency_ms > self.config.latency_threshold_ms * 2:  # 2x threshold
            alert_key = "high_latency"
            if self._should_alert(alert_key, now):
                logger.warning(f"ALERT: Very high search latency: {latency_ms:.0f}ms")
                self._last_alerts[alert_key] = now
        
        # Check error rate alert (simplified check)
        if not success:
            stats = self.get_current_stats()
            if stats['error_rate'] > self.config.error_rate_threshold * 2:  # 2x threshold
                alert_key = "high_error_rate"
                if self._should_alert(alert_key, now):
                    logger.warning(f"ALERT: High error rate: {stats['error_rate']:.2%}")
                    self._last_alerts[alert_key] = now
    
    def _should_alert(self, alert_key: str, now: datetime) -> bool:
        """Check if an alert should be sent (respecting cooldown)
        
        Args:
            alert_key: Alert identifier
            now: Current time
            
        Returns:
            True if alert should be sent
        """
        if alert_key not in self._last_alerts:
            return True
        
        cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
        return now - self._last_alerts[alert_key] > cooldown
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values
        
        Args:
            values: List of values
            percentile: Percentile to calculate (0-1)
            
        Returns:
            Percentile value
        """
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        
        if index >= len(sorted_values):
            return sorted_values[-1]
        
        return sorted_values[index]
    
    def _start_aggregation_thread(self):
        """Start background thread for metric aggregation"""
        def aggregation_worker():
            while True:
                try:
                    time.sleep(self.config.aggregation_interval_seconds)
                    self._save_metrics()
                    self._cleanup_old_metrics()
                except Exception as e:
                    logger.error(f"Error in metrics aggregation: {e}")
        
        thread = threading.Thread(target=aggregation_worker, daemon=True)
        thread.start()
        logger.debug("Started metrics aggregation thread")
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            metrics_file = Path(self.config.metrics_file_path)
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert metrics to serializable format
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'realtime_stats': self._realtime_stats,
                'metrics': {}
            }
            
            # Save recent metrics
            for metric_type, metrics_deque in self._metrics.items():
                metrics_data['metrics'][metric_type.value] = [
                    m.to_dict() for m in list(metrics_deque)[-1000:]  # Save last 1000
                ]
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _load_metrics(self):
        """Load metrics from file"""
        try:
            metrics_file = Path(self.config.metrics_file_path)
            if not metrics_file.exists():
                return
            
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            # Restore metrics (only recent ones)
            cutoff_time = datetime.now() - timedelta(days=1)  # Load last 24 hours
            
            for metric_type_str, metrics_list in data.get('metrics', {}).items():
                try:
                    metric_type = MetricType(metric_type_str)
                    for m_dict in metrics_list:
                        timestamp = datetime.fromisoformat(m_dict['timestamp'])
                        if timestamp >= cutoff_time:
                            metric = SearchMetric(
                                timestamp=timestamp,
                                query=m_dict['query'],
                                metric_type=metric_type,
                                value=m_dict['value'],
                                metadata=m_dict['metadata']
                            )
                            self._metrics[metric_type].append(metric)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error loading metric: {e}")
                    
        except Exception as e:
            logger.warning(f"Could not load metrics: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now() - timedelta(days=self.config.metrics_retention_days)
        
        for metric_type, metrics_deque in self._metrics.items():
            # Create new deque with only recent metrics
            recent_metrics = [
                m for m in metrics_deque
                if m.timestamp >= cutoff_time
            ]
            
            # Replace deque if needed
            if len(recent_metrics) < len(metrics_deque):
                new_deque = deque(recent_metrics, maxlen=self.config.max_memory_metrics)
                self._metrics[metric_type] = new_deque
    
    def export_metrics(
        self,
        output_format: str = "json",
        time_window_hours: int = 24
    ) -> str:
        """Export metrics for external analysis
        
        Args:
            output_format: Export format (json, csv)
            time_window_hours: Hours of data to export
            
        Returns:
            Exported data as string
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        if output_format == "json":
            export_data = {
                'export_time': datetime.now().isoformat(),
                'time_window_hours': time_window_hours,
                'metrics': {}
            }
            
            for metric_type, metrics_deque in self._metrics.items():
                recent_metrics = [
                    m.to_dict() for m in metrics_deque
                    if m.timestamp >= cutoff_time
                ]
                export_data['metrics'][metric_type.value] = recent_metrics
            
            return json.dumps(export_data, indent=2)
        
        elif output_format == "csv":
            # Simple CSV export
            lines = ["timestamp,query,metric_type,value"]
            
            for metric_type, metrics_deque in self._metrics.items():
                for metric in metrics_deque:
                    if metric.timestamp >= cutoff_time:
                        lines.append(
                            f"{metric.timestamp.isoformat()},{metric.query},"
                            f"{metric.metric_type.value},{metric.value}"
                        )
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {output_format}")


# Global metrics instance
_metrics_instance: Optional[SearchMetricsCollector] = None
_metrics_lock = threading.Lock()


def get_search_metrics(config: Optional[MetricsConfig] = None) -> SearchMetricsCollector:
    """Get or create global search metrics instance
    
    Args:
        config: Metrics configuration (used only on first call)
        
    Returns:
        Global search metrics instance
    """
    global _metrics_instance
    
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = SearchMetricsCollector(config)
                logger.info("Initialized global search metrics collector")
    
    return _metrics_instance


def create_search_metrics(
    enable_metrics: bool = True,
    latency_threshold_ms: float = 2000,
    cache_hit_threshold: float = 0.6,
    enable_alerts: bool = True
) -> SearchMetricsCollector:
    """Factory function to create a configured metrics collector
    
    Args:
        enable_metrics: Whether to enable metrics collection
        latency_threshold_ms: Latency threshold for alerts
        cache_hit_threshold: Cache hit rate threshold
        enable_alerts: Whether to enable performance alerts
        
    Returns:
        Configured metrics collector
    """
    config = MetricsConfig(
        enable_metrics=enable_metrics,
        latency_threshold_ms=latency_threshold_ms,
        cache_hit_threshold=cache_hit_threshold,
        enable_alerts=enable_alerts
    )
    
    return SearchMetricsCollector(config)


if __name__ == "__main__":
    # Test metrics collection
    print("Testing search metrics collection...")
    
    # Create metrics collector
    metrics = create_search_metrics(
        enable_metrics=True,
        latency_threshold_ms=1000,
        enable_alerts=True
    )
    
    # Simulate some searches
    test_queries = [
        ("authentication setup", 450, 15, True, True),
        ("database configuration", 1200, 8, False, True),
        ("error handling", 350, 22, True, True),
        ("deployment guide", 2500, 5, False, True),  # Slow query
        ("api documentation", 150, 30, True, True),
        ("security best practices", 500, 0, False, False),  # Failed search
    ]
    
    print("\nRecording test search metrics:")
    for query, latency, results, cache_hit, success in test_queries:
        metrics.record_search(
            query=query,
            latency_ms=latency,
            result_count=results,
            cache_hit=cache_hit,
            success=success,
            strategy="hybrid"
        )
        print(f"  Recorded: '{query}' - {latency}ms, {results} results, cache_hit={cache_hit}")
    
    # Record some parallel search metrics
    metrics.record_parallel_search(
        query="distributed systems",
        total_latency_ms=800,
        parallel_tasks=4,
        sequential_estimate_ms=2400,
        results_per_task=[5, 8, 3, 6]
    )
    
    # Get current stats
    stats = metrics.get_current_stats()
    print(f"\nCurrent Statistics:")
    print(f"  Total searches: {stats['total_searches']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Average results: {stats['avg_results_per_search']:.1f}")
    print(f"  Searches/minute: {stats['searches_per_minute']:.1f}")
    
    # Get metrics summary
    summary = metrics.get_metrics_summary()
    print(f"\nMetrics Summary (last 60 minutes):")
    for metric_type, metric_stats in summary.items():
        print(f"  {metric_type}:")
        print(f"    Mean: {metric_stats['mean']:.1f}")
        print(f"    P95: {metric_stats['p95']:.1f}")
        print(f"    P99: {metric_stats['p99']:.1f}")
    
    # Analyze performance
    analysis = metrics.analyze_performance()
    print(f"\nPerformance Analysis:")
    print(f"  Overall health: {analysis['overall_health']}")
    print(f"  Insights:")
    for insight in analysis['insights']:
        print(f"    - {insight}")
    print(f"  Recommendations:")
    for rec in analysis['recommendations']:
        print(f"    - {rec}")
    
    # Get query patterns
    patterns = metrics.get_query_patterns(top_n=3)
    print(f"\nQuery Patterns:")
    print(f"  Unique queries: {patterns['total_unique_queries']}")
    print(f"  Top queries:")
    for q in patterns['top_queries']:
        print(f"    - '{q['query']}': {q['count']} times, avg {q['avg_latency_ms']:.0f}ms")
    
    print("\nSearch metrics testing complete")