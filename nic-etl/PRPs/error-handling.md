# Error Handling - PRP

## ROLE
**Reliability Engineer with Error Management Expertise**

Specialist in designing robust error handling systems, implementing comprehensive logging strategies, and ensuring system resilience. Expert in fault tolerance patterns, graceful degradation, and monitoring systems that provide operational visibility.

## OBJECTIVE
**Implement Comprehensive Error Handling and Monitoring**

Create a robust error handling and monitoring system within Jupyter Notebook cells that:
* Implements comprehensive exception handling across all pipeline stages
* Provides detailed logging with appropriate severity levels
* Enables graceful degradation when components fail
* Implements retry logic with exponential backoff
* Monitors system health and performance metrics
* Provides alerting for critical failures
* Maintains audit trails for debugging and compliance

## MOTIVATION
**Operational Resilience and Debugging Capability**

Robust error handling ensures system reliability, enables quick problem resolution, and provides the visibility needed for production operations. Comprehensive monitoring prevents data loss and maintains service availability even when individual components fail.

## CONTEXT
**Production Jupyter Notebook Environment**

Operational requirements:
* Environment: Production-ready Jupyter Notebook
* Components: Multiple interdependent processing stages
* Failure modes: Network issues, processing errors, resource constraints
* Recovery: Automatic retry and graceful degradation
* Monitoring: Real-time health and performance tracking
* Constraints: Notebook-based implementation

## IMPLEMENTATION BLUEPRINT

### Code Structure
```python
# Cell 11: Error Handling and Monitoring
import logging
import time
import traceback
import functools
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    NETWORK = "network"
    PROCESSING = "processing"
    VALIDATION = "validation"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"

@dataclass
class ErrorEvent:
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    message: str
    exception_type: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    resolved: bool = False

class ErrorHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_log: List[ErrorEvent] = []
        self.retry_policies: Dict[str, Dict[str, Any]] = {}
        
        # Setup logging
        self._setup_logging()
        
        # Initialize default retry policies
        self._setup_retry_policies()
        
        # Performance metrics
        self.metrics = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_severity': {},
            'retry_success_rate': 0.0,
            'avg_resolution_time': 0.0
        }
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        log_config = self.config.get('logging', {})
        
        # Create logs directory
        log_file = Path(log_config.get('file', './logs/nic_etl.log'))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('NIC_ETL_ErrorHandler')
        self.logger.info("Error handling system initialized")
    
    def _setup_retry_policies(self):
        """Setup retry policies for different error types"""
        self.retry_policies = {
            'network': {
                'max_retries': 5,
                'base_delay': 1.0,
                'exponential_backoff': True,
                'max_delay': 60.0
            },
            'processing': {
                'max_retries': 3,
                'base_delay': 2.0,
                'exponential_backoff': True,
                'max_delay': 30.0
            },
            'external_service': {
                'max_retries': 4,
                'base_delay': 5.0,
                'exponential_backoff': True,
                'max_delay': 120.0
            },
            'default': {
                'max_retries': 2,
                'base_delay': 1.0,
                'exponential_backoff': False,
                'max_delay': 10.0
            }
        }
    
    def handle_error(self,
                    exception: Exception,
                    component: str,
                    category: ErrorCategory,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None) -> ErrorEvent:
        """Handle and log error event"""
        
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            component=component,
            message=str(exception),
            exception_type=type(exception).__name__,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        # Log error
        self._log_error(error_event)
        
        # Store error
        self.error_log.append(error_event)
        
        # Update metrics
        self._update_metrics(error_event)
        
        # Check if alert is needed
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._trigger_alert(error_event)
        
        return error_event
    
    def _log_error(self, error_event: ErrorEvent):
        """Log error with appropriate severity level"""
        log_message = f"[{error_event.category.value}] {error_event.component}: {error_event.message}"
        
        if error_event.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_event.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_event.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log context if available
        if error_event.context:
            self.logger.debug(f"Error context: {json.dumps(error_event.context)}")
    
    def _update_metrics(self, error_event: ErrorEvent):
        """Update error metrics"""
        self.metrics['total_errors'] += 1
        
        # Update category metrics
        category = error_event.category.value
        self.metrics['errors_by_category'][category] = (
            self.metrics['errors_by_category'].get(category, 0) + 1
        )
        
        # Update severity metrics
        severity = error_event.severity.value
        self.metrics['errors_by_severity'][severity] = (
            self.metrics['errors_by_severity'].get(severity, 0) + 1
        )
    
    def _trigger_alert(self, error_event: ErrorEvent):
        """Trigger alert for high-severity errors"""
        alert_message = f"ALERT: {error_event.severity.value.upper()} error in {error_event.component}"
        self.logger.critical(alert_message)
        
        # In production, this would integrate with alerting systems
        # For now, we'll write to a special alert log
        alert_file = Path('./logs/alerts.log')
        with open(alert_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {alert_message}\n")
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic and exponential backoff"""
        component = kwargs.pop('_component', func.__name__)
        category = kwargs.pop('_category', ErrorCategory.PROCESSING)
        
        # Get retry policy
        policy_name = category.value if category.value in self.retry_policies else 'default'
        policy = self.retry_policies[policy_name]
        
        last_exception = None
        
        for attempt in range(policy['max_retries'] + 1):
            try:
                result = func(*args, **kwargs)
                
                # Log successful retry
                if attempt > 0:
                    self.logger.info(f"Retry successful for {component} on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < policy['max_retries']:
                    # Calculate delay
                    if policy['exponential_backoff']:
                        delay = min(
                            policy['base_delay'] * (2 ** attempt),
                            policy['max_delay']
                        )
                    else:
                        delay = policy['base_delay']
                    
                    # Log retry attempt
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {component}, "
                        f"retrying in {delay:.1f}s: {str(e)}"
                    )
                    
                    time.sleep(delay)
                else:
                    # Final failure
                    error_event = self.handle_error(
                        e, component, category, ErrorSeverity.HIGH
                    )
                    error_event.retry_count = attempt + 1
        
        # All retries exhausted
        raise last_exception
    
    def get_error_summary(self, last_hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=last_hours)
        
        recent_errors = [
            error for error in self.error_log
            if error.timestamp >= cutoff_time
        ]
        
        summary = {
            'time_period_hours': last_hours,
            'total_errors': len(recent_errors),
            'errors_by_category': {},
            'errors_by_severity': {},
            'errors_by_component': {},
            'critical_errors': [],
            'top_error_types': {}
        }
        
        # Analyze recent errors
        for error in recent_errors:
            # By category
            category = error.category.value
            summary['errors_by_category'][category] = (
                summary['errors_by_category'].get(category, 0) + 1
            )
            
            # By severity
            severity = error.severity.value
            summary['errors_by_severity'][severity] = (
                summary['errors_by_severity'].get(severity, 0) + 1
            )
            
            # By component
            component = error.component
            summary['errors_by_component'][component] = (
                summary['errors_by_component'].get(component, 0) + 1
            )
            
            # Critical errors
            if error.severity == ErrorSeverity.CRITICAL:
                summary['critical_errors'].append({
                    'timestamp': error.timestamp.isoformat(),
                    'component': error.component,
                    'message': error.message
                })
            
            # Error types
            error_type = error.exception_type
            summary['top_error_types'][error_type] = (
                summary['top_error_types'].get(error_type, 0) + 1
            )
        
        return summary

class HealthMonitor:
    """Monitor system health and performance"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.health_checks: Dict[str, Callable] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.last_health_check = None
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'healthy',
            'checks': {},
            'failed_checks': []
        }
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                duration = time.time() - start_time
                
                results['checks'][name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'duration_ms': round(duration * 1000, 2)
                }
                
                if not is_healthy:
                    results['failed_checks'].append(name)
                    results['overall_health'] = 'degraded'
                    
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                results['failed_checks'].append(name)
                results['overall_health'] = 'degraded'
        
        self.last_health_check = results
        return results
    
    def record_performance_metric(self, name: str, value: float):
        """Record a performance metric"""
        if name not in self.performance_metrics:
            self.performance_metrics[name] = []
        
        self.performance_metrics[name].append(value)
        
        # Keep only last 1000 measurements
        if len(self.performance_metrics[name]) > 1000:
            self.performance_metrics[name] = self.performance_metrics[name][-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        summary = {}
        
        for name, values in self.performance_metrics.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1] if values else 0
                }
        
        return summary

# Decorator for automatic error handling
def handle_errors(component: str, 
                 category: ErrorCategory = ErrorCategory.PROCESSING,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 retry: bool = False):
    """Decorator for automatic error handling"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if retry:
                    return error_handler.retry_with_backoff(
                        func, *args, _component=component, _category=category, **kwargs
                    )
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, component, category, severity)
                raise
        return wrapper
    return decorator

# Initialize global error handler
error_handler = ErrorHandler(CONFIG)
health_monitor = HealthMonitor(error_handler)

# Register default health checks
def check_cache_directory():
    return CACHE_DIR.exists() and CACHE_DIR.is_dir()

def check_disk_space():
    import shutil
    free_space_gb = shutil.disk_usage(CACHE_DIR).free / (1024**3)
    return free_space_gb > 1.0  # At least 1GB free

health_monitor.register_health_check('cache_directory', check_cache_directory)
health_monitor.register_health_check('disk_space', check_disk_space)

print("Error handling and monitoring system initialized")
```

## VALIDATION LOOP

### Unit Testing
```python
def test_error_handling():
    """Test error handling functionality"""
    handler = ErrorHandler({})
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_event = handler.handle_error(
            e, "test_component", ErrorCategory.PROCESSING
        )
    
    assert error_event.exception_type == "ValueError"
    assert error_event.component == "test_component"
    assert len(handler.error_log) == 1

def test_retry_mechanism():
    """Test retry with backoff"""
    handler = ErrorHandler({})
    
    call_count = 0
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary failure")
        return "success"
    
    result = handler.retry_with_backoff(
        failing_function,
        _component="test",
        _category=ErrorCategory.NETWORK
    )
    
    assert result == "success"
    assert call_count == 3

def test_health_monitoring():
    """Test health monitoring system"""
    handler = ErrorHandler({})
    monitor = HealthMonitor(handler)
    
    monitor.register_health_check('always_pass', lambda: True)
    monitor.register_health_check('always_fail', lambda: False)
    
    results = monitor.run_health_checks()
    
    assert results['overall_health'] == 'degraded'
    assert 'always_fail' in results['failed_checks']
    assert results['checks']['always_pass']['status'] == 'healthy'
```

## ADDITIONAL NOTES

### Security Considerations
* **Log Sanitization**: Remove sensitive data from error logs
* **Access Control**: Restrict access to error logs and alerts
* **Data Protection**: Encrypt sensitive error context data
* **Audit Trail**: Maintain immutable error logs for compliance

### Performance Optimization
* **Async Logging**: Use asynchronous logging for high-volume scenarios
* **Log Rotation**: Implement log rotation to manage disk space
* **Batch Processing**: Process errors in batches for efficiency
* **Sampling**: Sample high-frequency errors to reduce overhead

### Maintenance Requirements
* **Alert Tuning**: Regular review and tuning of alert thresholds
* **Log Analysis**: Periodic analysis of error patterns
* **Performance Monitoring**: Track error handling performance impact
* **Documentation**: Maintain runbooks for common error scenarios