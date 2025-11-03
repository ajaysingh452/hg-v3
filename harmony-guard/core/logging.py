"""Structured logging system with PII masking for Harmony Guard."""

import json
import logging
import re
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from contextvars import ContextVar
from pathlib import Path
import sys

# Context variables for request correlation
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
tenant_id_context: ContextVar[Optional[str]] = ContextVar('tenant_id', default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


class PIIMasker:
    """PII masking utility for log sanitization."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize PII masker with configuration."""
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.mask_char = self.config.get("mask_char", "*")
        self.preserve_length = self.config.get("preserve_length", True)
        
        # PII patterns
        self.patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            "indian_phone": re.compile(r'(\+91[-.\s]?)?[6-9]\d{9}'),
            "credit_card": re.compile(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "aadhaar": re.compile(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'),
            "pan": re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),
            "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            "url": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            "api_key": re.compile(r'\b[A-Za-z0-9]{32,}\b'),
        }
        
        # Custom patterns from config
        custom_patterns = self.config.get("custom_patterns", {})
        for name, pattern in custom_patterns.items():
            self.patterns[name] = re.compile(pattern)
    
    def mask_text(self, text: str) -> str:
        """Mask PII in text."""
        if not self.enabled or not text:
            return text
        
        masked_text = text
        
        for pattern_name, pattern in self.patterns.items():
            masked_text = self._apply_pattern_mask(masked_text, pattern, pattern_name)
        
        return masked_text
    
    def _apply_pattern_mask(self, text: str, pattern: re.Pattern, pattern_name: str) -> str:
        """Apply masking for a specific pattern."""
        def mask_match(match):
            original = match.group(0)
            
            if pattern_name in ["email", "phone", "indian_phone"]:
                # Preserve domain/area code for emails/phones
                if "@" in original:
                    local, domain = original.split("@", 1)
                    masked_local = self._mask_string(local, preserve_first=1, preserve_last=0)
                    return f"{masked_local}@{domain}"
                elif pattern_name in ["phone", "indian_phone"]:
                    # Preserve country code and mask middle digits
                    if len(original) > 6:
                        return original[:3] + self._mask_string(original[3:-2]) + original[-2:]
            
            # Default masking
            return self._mask_string(original)
        
        return pattern.sub(mask_match, text)
    
    def _mask_string(self, text: str, preserve_first: int = 0, preserve_last: int = 0) -> str:
        """Mask a string with specified preservation rules."""
        if len(text) <= preserve_first + preserve_last:
            return self.mask_char * len(text)
        
        if self.preserve_length:
            masked_length = len(text) - preserve_first - preserve_last
            masked_part = self.mask_char * masked_length
            return text[:preserve_first] + masked_part + text[-preserve_last:] if preserve_last > 0 else text[:preserve_first] + masked_part
        else:
            return self.mask_char * 4  # Fixed length mask


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, service_name: str = "harmony-guard", version: str = "1.0.0", 
                 include_pii_masking: bool = True, pii_config: Optional[Dict] = None):
        """Initialize structured formatter."""
        super().__init__()
        self.service_name = service_name
        self.version = version
        self.hostname = self._get_hostname()
        
        # PII masking
        self.pii_masker = PIIMasker(pii_config) if include_pii_masking else None
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": {
                "name": self.service_name,
                "version": self.version,
                "hostname": self.hostname
            }
        }
        
        # Add context information
        request_id = request_id_context.get()
        if request_id:
            log_entry["request_id"] = request_id
        
        tenant_id = tenant_id_context.get()
        if tenant_id:
            log_entry["tenant_id"] = tenant_id
        
        user_id = user_id_context.get()
        if user_id:
            log_entry["user_id"] = user_id
        
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 
                          'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        # Add source information for debug/error levels
        if record.levelno >= logging.WARNING:
            log_entry["source"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName
            }
        
        # Apply PII masking
        if self.pii_masker:
            log_entry = self._mask_log_entry(log_entry)
        
        return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'))
    
    def _mask_log_entry(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Apply PII masking to log entry."""
        def mask_recursive(obj):
            if isinstance(obj, str):
                return self.pii_masker.mask_text(obj)
            elif isinstance(obj, dict):
                return {k: mask_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [mask_recursive(item) for item in obj]
            else:
                return obj
        
        return mask_recursive(log_entry)
    
    def _get_hostname(self) -> str:
        """Get hostname for logging."""
        import socket
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"


class HarmonyGuardLogger:
    """Enhanced logger for Harmony Guard with structured logging and PII masking."""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """Initialize Harmony Guard logger."""
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self.name = name
        
        # Configure logger if not already configured
        if not self.logger.handlers:
            self._configure_logger()
    
    def _configure_logger(self):
        """Configure logger with structured formatting."""
        level = self.config.get("level", "INFO")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Console handler with structured formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = StructuredFormatter(
            service_name=self.config.get("service_name", "harmony-guard"),
            version=self.config.get("version", "1.0.0"),
            include_pii_masking=self.config.get("pii_masking", {}).get("enabled", True),
            pii_config=self.config.get("pii_masking", {})
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if configured
        log_file = self.config.get("file_path")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
        
        # Audit log handler for sensitive operations
        audit_file = self.config.get("audit_file_path")
        if audit_file:
            audit_handler = logging.FileHandler(audit_file)
            audit_formatter = StructuredFormatter(
                service_name=f"{self.config.get('service_name', 'harmony-guard')}-audit",
                version=self.config.get("version", "1.0.0"),
                include_pii_masking=True,  # Always mask PII in audit logs
                pii_config=self.config.get("pii_masking", {})
            )
            audit_handler.setFormatter(audit_formatter)
            
            # Create audit logger
            audit_logger = logging.getLogger(f"{self.name}.audit")
            audit_logger.addHandler(audit_handler)
            audit_logger.setLevel(logging.INFO)
    
    def set_context(self, request_id: Optional[str] = None, tenant_id: Optional[str] = None, 
                   user_id: Optional[str] = None):
        """Set logging context for request correlation."""
        if request_id:
            request_id_context.set(request_id)
        if tenant_id:
            tenant_id_context.set(tenant_id)
        if user_id:
            user_id_context.set(user_id)
    
    def clear_context(self):
        """Clear logging context."""
        request_id_context.set(None)
        tenant_id_context.set(None)
        user_id_context.set(None)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with extra fields."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with extra fields."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with extra fields."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with extra fields."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with extra fields."""
        self.logger.critical(message, extra=kwargs)
    
    def audit(self, event: str, details: Dict[str, Any], **kwargs):
        """Log audit event."""
        audit_logger = logging.getLogger(f"{self.name}.audit")
        audit_entry = {
            "event": event,
            "details": details,
            **kwargs
        }
        audit_logger.info(f"AUDIT: {event}", extra=audit_entry)
    
    def log_request(self, method: str, endpoint: str, status_code: int, 
                   duration: float, **kwargs):
        """Log HTTP request with standard fields."""
        self.info(
            f"{method} {endpoint} {status_code} {duration:.3f}s",
            http_method=method,
            http_endpoint=endpoint,
            http_status_code=status_code,
            duration_seconds=duration,
            **kwargs
        )
    
    def log_analysis_request(self, text_length: int, languages: List[str], 
                           decision: str, confidence: float, duration: float, **kwargs):
        """Log content analysis request."""
        self.info(
            f"Analysis completed: {decision} (confidence: {confidence:.3f})",
            analysis_text_length=text_length,
            analysis_languages=languages,
            analysis_decision=decision,
            analysis_confidence=confidence,
            analysis_duration_seconds=duration,
            **kwargs
        )
    
    def log_component_error(self, component: str, error_type: str, error_message: str, **kwargs):
        """Log component error."""
        self.error(
            f"Component error in {component}: {error_message}",
            component_name=component,
            error_type=error_type,
            error_message=error_message,
            **kwargs
        )
    
    def log_drift_alert(self, drift_type: str, severity: str, metric_name: str, 
                       current_value: float, baseline_value: float, **kwargs):
        """Log model drift alert."""
        self.warning(
            f"Model drift detected: {drift_type} in {metric_name}",
            drift_type=drift_type,
            drift_severity=severity,
            drift_metric=metric_name,
            drift_current_value=current_value,
            drift_baseline_value=baseline_value,
            **kwargs
        )
    
    def log_feedback(self, original_decision: str, corrected_decision: str, 
                    feedback_type: str, **kwargs):
        """Log feedback submission."""
        self.info(
            f"Feedback received: {original_decision} -> {corrected_decision}",
            feedback_original_decision=original_decision,
            feedback_corrected_decision=corrected_decision,
            feedback_type=feedback_type,
            **kwargs
        )


class LoggingMiddleware:
    """Middleware for request logging with correlation IDs."""
    
    def __init__(self, logger: HarmonyGuardLogger):
        """Initialize logging middleware."""
        self.logger = logger
    
    async def __call__(self, request, call_next):
        """Process request with logging."""
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Extract tenant information
        tenant_id = request.headers.get("X-Tenant-ID", "unknown")
        
        # Set logging context
        self.logger.set_context(request_id=request_id, tenant_id=tenant_id)
        
        # Store in request state for other components
        request.state.request_id = request_id
        request.state.tenant_id = tenant_id
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log successful request
            self.logger.log_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=duration,
                request_size=request.headers.get("content-length", 0),
                user_agent=request.headers.get("user-agent", "unknown")
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log failed request
            self.logger.log_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=500,
                duration=duration,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            raise
        finally:
            # Clear context
            self.logger.clear_context()


def configure_logging(config: Dict[str, Any]) -> HarmonyGuardLogger:
    """Configure application logging."""
    # Ensure log directory exists
    log_file = config.get("file_path")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    audit_file = config.get("audit_file_path")
    if audit_file:
        Path(audit_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create main logger
    logger = HarmonyGuardLogger("harmony-guard", config)
    
    return logger


def get_logger(name: str) -> HarmonyGuardLogger:
    """Get logger instance."""
    return HarmonyGuardLogger(name)


# Default configuration
DEFAULT_LOGGING_CONFIG = {
    "level": "INFO",
    "service_name": "harmony-guard",
    "version": "1.0.0",
    "file_path": "logs/harmony-guard.log",
    "audit_file_path": "logs/harmony-guard-audit.log",
    "pii_masking": {
        "enabled": True,
        "mask_char": "*",
        "preserve_length": True,
        "custom_patterns": {}
    }
}