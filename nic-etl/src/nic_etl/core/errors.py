import logging
import logging.handlers
import traceback
import time
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import uuid
from contextlib import contextmanager
import functools

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error category classification"""
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    PROCESSING = "processing"
    STORAGE = "storage"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    EXTERNAL_SERVICE = "external_service"

class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    RETRY = "retry"
    SKIP = "skip"
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful_degradation"

@dataclass
class ErrorContext:
    """Error context information"""
    correlation_id: str
    module_name: str
    operation_name: str
    document_id: Optional[str] = None
    stage: Optional[str] = None
    user_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ErrorResponse:
    """Error handling response"""
    should_retry: bool
    retry_delay: float
    recovery_action: Optional[str] = None
    user_message: str = ""
    technical_details: str = ""
    correlation_id: str = ""

@dataclass
class ErrorStatistics:
    """Error tracking statistics"""
    total_errors: int
    errors_by_category: Dict[str, int]
    errors_by_severity: Dict[str, int]
    errors_by_module: Dict[str, int]
    recovery_success_rate: float
    most_frequent_errors: List[str]
    error_trends: Dict[str, List[int]]

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.timeout)
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class ErrorManager:
    """Production-ready centralized error handling and recovery"""
    
    # Error code mappings
    ERROR_CODES = {
        # GitLab Integration Errors
        "GITLAB_AUTH_FAILED": {
            "category": ErrorCategory.AUTHENTICATION,
            "severity": ErrorSeverity.CRITICAL,
            "recovery": RecoveryStrategy.FAIL_FAST,
            "message": "GitLab authentication failed. Check access token."
        },
        "GITLAB_NETWORK_ERROR": {
            "category": ErrorCategory.NETWORK,
            "severity": ErrorSeverity.HIGH,
            "recovery": RecoveryStrategy.RETRY,
            "message": "GitLab network connectivity issue. Retrying..."
        },
        "GITLAB_FILE_NOT_FOUND": {
            "category": ErrorCategory.VALIDATION,
            "severity": ErrorSeverity.MEDIUM,
            "recovery": RecoveryStrategy.SKIP,
            "message": "Document not found in GitLab repository."
        },
        
        # Document Processing Errors
        "DOCLING_PROCESSING_FAILED": {
            "category": ErrorCategory.PROCESSING,
            "severity": ErrorSeverity.HIGH,
            "recovery": RecoveryStrategy.SKIP,
            "message": "Document processing failed. Document may be corrupted."
        },
        "OCR_CONFIDENCE_LOW": {
            "category": ErrorCategory.PROCESSING,
            "severity": ErrorSeverity.MEDIUM,
            "recovery": RecoveryStrategy.GRACEFUL_DEGRADATION,
            "message": "OCR confidence below threshold. Quality may be impacted."
        },
        
        # Embedding Errors
        "EMBEDDING_MODEL_LOAD_FAILED": {
            "category": ErrorCategory.RESOURCE,
            "severity": ErrorSeverity.CRITICAL,
            "recovery": RecoveryStrategy.FAIL_FAST,
            "message": "Failed to load embedding model. Check model availability."
        },
        "EMBEDDING_GENERATION_FAILED": {
            "category": ErrorCategory.PROCESSING,
            "severity": ErrorSeverity.HIGH,
            "recovery": RecoveryStrategy.SKIP,
            "message": "Embedding generation failed for document chunk."
        },
        
        # Qdrant Errors
        "QDRANT_CONNECTION_FAILED": {
            "category": ErrorCategory.EXTERNAL_SERVICE,
            "severity": ErrorSeverity.CRITICAL,
            "recovery": RecoveryStrategy.RETRY,
            "message": "Qdrant connection failed. Check service availability."
        },
        "QDRANT_INSERTION_FAILED": {
            "category": ErrorCategory.STORAGE,
            "severity": ErrorSeverity.HIGH,
            "recovery": RecoveryStrategy.RETRY,
            "message": "Vector insertion failed. Retrying with smaller batch."
        },
        
        # Configuration Errors
        "CONFIG_VALIDATION_FAILED": {
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.CRITICAL,
            "recovery": RecoveryStrategy.FAIL_FAST,
            "message": "Configuration validation failed. Check configuration file."
        },
        "SECRET_MISSING": {
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.CRITICAL,
            "recovery": RecoveryStrategy.FAIL_FAST,
            "message": "Required secret not found. Check environment variables."
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_counts = {}
        self.error_history = []
        self.recovery_attempts = {}
        
        # Circuit breakers for external services
        self.circuit_breakers = {
            'gitlab': CircuitBreaker(failure_threshold=3, timeout=60.0),
            'qdrant': CircuitBreaker(failure_threshold=5, timeout=30.0),
            'embedding': CircuitBreaker(failure_threshold=2, timeout=120.0)
        }
        
        # Setup structured logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure structured logging with correlation IDs"""
        
        # Create custom formatter with correlation ID
        class CorrelatedFormatter(logging.Formatter):
            def format(self, record):
                # Add correlation ID to log record
                if not hasattr(record, 'correlation_id'):
                    record.correlation_id = 'N/A'
                return super().format(record)
        
        # Configure root logger
        formatter = CorrelatedFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # File handler with rotation
        log_file = self.config.get('log_file', 'nic_etl_pipeline.log')
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=100*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        
        # Configure logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        self.logger.info("Error handling and logging system initialized")
    
    def handle_error(self, error: Exception, context: ErrorContext) -> ErrorResponse:
        """Comprehensive error handling with classification and recovery"""
        
        # Generate correlation ID if not provided
        if not context.correlation_id:
            context.correlation_id = str(uuid.uuid4())
        
        # Classify error
        error_code = self._classify_error(error)
        error_info = self.ERROR_CODES.get(error_code, {
            "category": ErrorCategory.PROCESSING,
            "severity": ErrorSeverity.MEDIUM,
            "recovery": RecoveryStrategy.SKIP,
            "message": "An unexpected error occurred."
        })
        
        # Log error with full context
        self._log_error(error, context, error_code, error_info)
        
        # Track error statistics
        self._track_error(error_code, error_info["category"], error_info["severity"])
        
        # Determine recovery strategy
        response = self._create_error_response(error, context, error_code, error_info)
        
        # Execute recovery if possible
        self._attempt_recovery(error, context, response)
        
        return response
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error and return appropriate error code"""
        
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # GitLab errors
        if 'authentication' in error_message or 'unauthorized' in error_message:
            return "GITLAB_AUTH_FAILED"
        elif 'connection' in error_message or 'network' in error_message:
            return "GITLAB_NETWORK_ERROR"
        elif 'not found' in error_message and 'file' in error_message:
            return "GITLAB_FILE_NOT_FOUND"
        
        # Processing errors
        elif 'docling' in error_message or 'processing' in error_message:
            return "DOCLING_PROCESSING_FAILED"
        elif 'ocr' in error_message and 'confidence' in error_message:
            return "OCR_CONFIDENCE_LOW"
        
        # Embedding errors
        elif 'model' in error_message and ('load' in error_message or 'download' in error_message):
            return "EMBEDDING_MODEL_LOAD_FAILED"
        elif 'embedding' in error_message and 'generation' in error_message:
            return "EMBEDDING_GENERATION_FAILED"
        
        # Qdrant errors
        elif 'qdrant' in error_message and 'connection' in error_message:
            return "QDRANT_CONNECTION_FAILED"
        elif 'qdrant' in error_message and ('insert' in error_message or 'upsert' in error_message):
            return "QDRANT_INSERTION_FAILED"
        
        # Configuration errors
        elif 'configuration' in error_message or 'config' in error_message:
            return "CONFIG_VALIDATION_FAILED"
        elif 'secret' in error_message or 'token' in error_message or 'key' in error_message:
            return "SECRET_MISSING"
        
        # Default classification
        else:
            return "UNKNOWN_ERROR"
    
    def _log_error(self, error: Exception, context: ErrorContext, 
                   error_code: str, error_info: Dict[str, Any]):
        """Log error with full context and structured data"""
        
        error_data = {
            'error_code': error_code,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'category': error_info["category"].value,
            'severity': error_info["severity"].value,
            'correlation_id': context.correlation_id,
            'module_name': context.module_name,
            'operation_name': context.operation_name,
            'document_id': context.document_id,
            'stage': context.stage,
            'timestamp': context.timestamp.isoformat(),
            'traceback': traceback.format_exc(),
            'user_data': context.user_data
        }
        
        # Log at appropriate level based on severity
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_info["severity"], logging.ERROR)
        
        # Create log record with correlation ID
        logger = logging.getLogger(context.module_name)
        logger.log(
            log_level,
            f"Error {error_code}: {error_info['message']}",
            extra={'correlation_id': context.correlation_id}
        )
        
        # Store structured error data
        self.error_history.append(error_data)
        
        # Limit history size
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-800:]  # Keep recent 800 entries
    
    def _track_error(self, error_code: str, category: ErrorCategory, severity: ErrorSeverity):
        """Track error statistics for monitoring"""
        
        # Track by error code
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        
        # Track by category
        category_key = f"category_{category.value}"
        self.error_counts[category_key] = self.error_counts.get(category_key, 0) + 1
        
        # Track by severity
        severity_key = f"severity_{severity.value}"
        self.error_counts[severity_key] = self.error_counts.get(severity_key, 0) + 1
    
    def _create_error_response(self, error: Exception, context: ErrorContext,
                              error_code: str, error_info: Dict[str, Any]) -> ErrorResponse:
        """Create appropriate error response based on classification"""
        
        recovery_strategy = error_info["recovery"]
        
        # Determine retry behavior
        should_retry = recovery_strategy == RecoveryStrategy.RETRY
        retry_delay = 0.0
        
        if should_retry:
            # Calculate exponential backoff delay
            attempt_count = self.recovery_attempts.get(error_code, 0)
            retry_delay = min(60.0, 2.0 ** attempt_count)  # Max 60 seconds
        
        # Create response
        response = ErrorResponse(
            should_retry=should_retry,
            retry_delay=retry_delay,
            recovery_action=recovery_strategy.value,
            user_message=error_info["message"],
            technical_details=str(error),
            correlation_id=context.correlation_id
        )
        
        return response
    
    def _attempt_recovery(self, error: Exception, context: ErrorContext, response: ErrorResponse):
        """Attempt error recovery based on strategy"""
        
        if response.should_retry:
            error_code = self._classify_error(error)
            self.recovery_attempts[error_code] = self.recovery_attempts.get(error_code, 0) + 1
            
            self.logger.info(
                f"Attempting recovery for {error_code} (attempt {self.recovery_attempts[error_code]})",
                extra={'correlation_id': context.correlation_id}
            )
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get circuit breaker for external service"""
        
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        
        return self.circuit_breakers[service_name]
    
    def get_error_statistics(self) -> ErrorStatistics:
        """Get comprehensive error statistics"""
        
        # Calculate statistics
        total_errors = sum(count for key, count in self.error_counts.items() 
                          if not key.startswith(('category_', 'severity_')))
        
        errors_by_category = {
            key.replace('category_', ''): count 
            for key, count in self.error_counts.items() 
            if key.startswith('category_')
        }
        
        errors_by_severity = {
            key.replace('severity_', ''): count 
            for key, count in self.error_counts.items() 
            if key.startswith('severity_')
        }
        
        # Most frequent errors
        error_codes = {key: count for key, count in self.error_counts.items() 
                      if key in self.ERROR_CODES}
        most_frequent = sorted(error_codes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate recovery success rate
        total_attempts = sum(self.recovery_attempts.values())
        recovery_success_rate = 0.8 if total_attempts > 0 else 1.0  # Placeholder calculation
        
        return ErrorStatistics(
            total_errors=total_errors,
            errors_by_category=errors_by_category,
            errors_by_severity=errors_by_severity,
            errors_by_module={},  # Would need module-specific tracking
            recovery_success_rate=recovery_success_rate,
            most_frequent_errors=[code for code, count in most_frequent],
            error_trends={}  # Would need time-series tracking
        )
    
    def export_error_report(self, include_history: bool = False) -> Dict[str, Any]:
        """Export comprehensive error report"""
        
        stats = self.get_error_statistics()
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'statistics': {
                'total_errors': stats.total_errors,
                'errors_by_category': stats.errors_by_category,
                'errors_by_severity': stats.errors_by_severity,
                'recovery_success_rate': stats.recovery_success_rate,
                'most_frequent_errors': stats.most_frequent_errors
            },
            'circuit_breaker_status': {
                name: {
                    'state': cb.state,
                    'failure_count': cb.failure_count,
                    'last_failure_time': cb.last_failure_time
                }
                for name, cb in self.circuit_breakers.items()
            }
        }
        
        if include_history:
            report['error_history'] = self.error_history[-100:]  # Last 100 errors
        
        return report

# Error handling decorators
def with_error_handling(error_manager: ErrorManager, module_name: str, operation_name: str):
    """Decorator for automatic error handling"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            correlation_id = str(uuid.uuid4())
            context = ErrorContext(
                correlation_id=correlation_id,
                module_name=module_name,
                operation_name=operation_name
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                response = error_manager.handle_error(e, context)
                
                if response.should_retry:
                    time.sleep(response.retry_delay)
                    return func(*args, **kwargs)  # Single retry
                else:
                    raise
        
        return wrapper
    return decorator

@contextmanager
def error_context(error_manager: ErrorManager, module_name: str, operation_name: str, **context_data):
    """Context manager for error handling with automatic context creation"""
    
    correlation_id = str(uuid.uuid4())
    context = ErrorContext(
        correlation_id=correlation_id,
        module_name=module_name,
        operation_name=operation_name,
        user_data=context_data
    )
    
    try:
        yield context
    except Exception as e:
        response = error_manager.handle_error(e, context)
        
        if response.should_retry:
            raise RetryableError(response.retry_delay, response.user_message)
        else:
            raise

class RetryableError(Exception):
    """Exception that indicates the operation should be retried"""
    
    def __init__(self, retry_delay: float, message: str):
        self.retry_delay = retry_delay
        super().__init__(message)

def create_error_manager(config: Optional[Dict[str, Any]] = None) -> ErrorManager:
    """Factory function for error manager creation"""
    return ErrorManager(config)