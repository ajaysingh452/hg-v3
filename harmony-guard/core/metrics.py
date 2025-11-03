"""Prometheus metrics collection for Harmony Guard."""

import time
from typing import Dict, Any, Optional, List
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from prometheus_client.core import REGISTRY
import logging

logger = logging.getLogger(__name__)


class HarmonyGuardMetrics:
    """Prometheus metrics collector for Harmony Guard service."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector."""
        self.registry = registry or REGISTRY
        
        # Request metrics
        self.request_total = Counter(
            'harmony_guard_requests_total',
            'Total number of analysis requests',
            ['method', 'endpoint', 'status', 'tenant_id'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'harmony_guard_request_duration_seconds',
            'Request processing duration in seconds',
            ['endpoint', 'tenant_id'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Decision metrics
        self.decision_total = Counter(
            'harmony_guard_decisions_total',
            'Total decisions by type and language',
            ['decision', 'language', 'tenant_id'],
            registry=self.registry
        )
        
        self.confidence_scores = Histogram(
            'harmony_guard_confidence_scores',
            'Distribution of confidence scores',
            ['category', 'decision', 'language'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Component performance metrics
        self.component_duration = Histogram(
            'harmony_guard_component_duration_seconds',
            'Component processing duration',
            ['component'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.component_errors = Counter(
            'harmony_guard_component_errors_total',
            'Component error count',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        # Language detection metrics
        self.language_detection = Counter(
            'harmony_guard_language_detection_total',
            'Language detection results',
            ['detected_language', 'confidence_bucket'],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_predictions = Counter(
            'harmony_guard_model_predictions_total',
            'Model prediction counts',
            ['model_type', 'prediction_class'],
            registry=self.registry
        )
        
        self.model_confidence = Histogram(
            'harmony_guard_model_confidence',
            'Model confidence distribution',
            ['model_type', 'language'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Business KPI metrics
        self.precision_estimate = Gauge(
            'harmony_guard_precision_estimate',
            'Estimated precision based on feedback',
            ['category', 'language'],
            registry=self.registry
        )
        
        self.recall_estimate = Gauge(
            'harmony_guard_recall_estimate',
            'Estimated recall based on feedback',
            ['category', 'language'],
            registry=self.registry
        )
        
        # Feedback metrics
        self.feedback_total = Counter(
            'harmony_guard_feedback_total',
            'Total feedback submissions',
            ['feedback_type', 'original_decision', 'corrected_decision'],
            registry=self.registry
        )
        
        # System metrics
        self.active_requests = Gauge(
            'harmony_guard_active_requests',
            'Number of currently active requests',
            registry=self.registry
        )
        
        self.component_health = Gauge(
            'harmony_guard_component_health',
            'Component health status (1=healthy, 0=unhealthy)',
            ['component'],
            registry=self.registry
        )
        
        # Drift detection metrics
        self.drift_alerts = Counter(
            'harmony_guard_drift_alerts_total',
            'Drift detection alerts',
            ['drift_type', 'severity', 'metric_name'],
            registry=self.registry
        )
        
        self.drift_score = Gauge(
            'harmony_guard_drift_score',
            'Current drift score for metrics',
            ['metric_name', 'drift_type'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            'harmony_guard_rate_limit_hits_total',
            'Rate limit violations',
            ['tenant_id', 'tier'],
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'harmony_guard_service_info',
            'Service information',
            registry=self.registry
        )
        
        # Initialize service info
        self.service_info.info({
            'version': '1.0.0',
            'service': 'harmony-guard',
            'build_time': str(int(time.time()))
        })
    
    def record_request(self, method: str, endpoint: str, status: str, 
                      duration: float, tenant_id: str = "unknown"):
        """Record request metrics."""
        self.request_total.labels(
            method=method,
            endpoint=endpoint,
            status=status,
            tenant_id=tenant_id
        ).inc()
        
        self.request_duration.labels(
            endpoint=endpoint,
            tenant_id=tenant_id
        ).observe(duration)
    
    def record_decision(self, decision: str, languages: List[str], 
                       confidence: float, tenant_id: str = "unknown"):
        """Record decision metrics."""
        for language in languages:
            self.decision_total.labels(
                decision=decision,
                language=language,
                tenant_id=tenant_id
            ).inc()
            
            # Record confidence for primary language
            if languages and language == languages[0]:
                self.confidence_scores.labels(
                    category="overall",
                    decision=decision,
                    language=language
                ).observe(confidence)
    
    def record_category_confidence(self, category: str, decision: str, 
                                  language: str, confidence: float):
        """Record category-specific confidence scores."""
        self.confidence_scores.labels(
            category=category,
            decision=decision,
            language=language
        ).observe(confidence)
    
    def record_component_performance(self, component: str, duration: float):
        """Record component processing time."""
        self.component_duration.labels(component=component).observe(duration)
    
    def record_component_error(self, component: str, error_type: str):
        """Record component error."""
        self.component_errors.labels(
            component=component,
            error_type=error_type
        ).inc()
    
    def record_language_detection(self, detected_language: str, confidence: float):
        """Record language detection results."""
        # Bucket confidence scores
        if confidence >= 0.9:
            confidence_bucket = "high"
        elif confidence >= 0.7:
            confidence_bucket = "medium"
        elif confidence >= 0.5:
            confidence_bucket = "low"
        else:
            confidence_bucket = "very_low"
        
        self.language_detection.labels(
            detected_language=detected_language,
            confidence_bucket=confidence_bucket
        ).inc()
    
    def record_model_prediction(self, model_type: str, prediction_class: str, 
                               confidence: float, language: str = "unknown"):
        """Record model prediction metrics."""
        self.model_predictions.labels(
            model_type=model_type,
            prediction_class=prediction_class
        ).inc()
        
        self.model_confidence.labels(
            model_type=model_type,
            language=language
        ).observe(confidence)
    
    def update_business_kpis(self, category: str, language: str, 
                           precision: float, recall: float):
        """Update business KPI estimates."""
        self.precision_estimate.labels(
            category=category,
            language=language
        ).set(precision)
        
        self.recall_estimate.labels(
            category=category,
            language=language
        ).set(recall)
    
    def record_feedback(self, feedback_type: str, original_decision: str, 
                       corrected_decision: str):
        """Record feedback submission."""
        self.feedback_total.labels(
            feedback_type=feedback_type,
            original_decision=original_decision,
            corrected_decision=corrected_decision
        ).inc()
    
    def set_active_requests(self, count: int):
        """Set current active request count."""
        self.active_requests.set(count)
    
    def set_component_health(self, component: str, healthy: bool):
        """Set component health status."""
        self.component_health.labels(component=component).set(1 if healthy else 0)
    
    def record_drift_alert(self, drift_type: str, severity: str, metric_name: str):
        """Record drift detection alert."""
        self.drift_alerts.labels(
            drift_type=drift_type,
            severity=severity,
            metric_name=metric_name
        ).inc()
    
    def set_drift_score(self, metric_name: str, drift_type: str, score: float):
        """Set current drift score."""
        self.drift_score.labels(
            metric_name=metric_name,
            drift_type=drift_type
        ).set(score)
    
    def record_rate_limit_hit(self, tenant_id: str, tier: str):
        """Record rate limit violation."""
        self.rate_limit_hits.labels(
            tenant_id=tenant_id,
            tier=tier
        ).inc()
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary for JSON responses."""
        # This is a simplified version - in production you'd want to
        # properly collect and format all metric values
        return {
            "requests_total": self._get_counter_value(self.request_total),
            "decisions_total": self._get_counter_value(self.decision_total),
            "component_errors_total": self._get_counter_value(self.component_errors),
            "feedback_total": self._get_counter_value(self.feedback_total),
            "drift_alerts_total": self._get_counter_value(self.drift_alerts),
            "rate_limit_hits_total": self._get_counter_value(self.rate_limit_hits),
            "timestamp": time.time()
        }
    
    def _get_counter_value(self, counter) -> float:
        """Get total value from a counter metric."""
        try:
            # Sum all label combinations
            total = 0
            for sample in counter.collect()[0].samples:
                total += sample.value
            return total
        except Exception:
            return 0.0


# Global metrics instance
metrics = HarmonyGuardMetrics()


class MetricsMiddleware:
    """Middleware for automatic metrics collection."""
    
    def __init__(self, metrics_collector: HarmonyGuardMetrics):
        self.metrics = metrics_collector
        self.active_requests = 0
    
    async def __call__(self, request, call_next):
        """Process request and collect metrics."""
        start_time = time.time()
        self.active_requests += 1
        self.metrics.set_active_requests(self.active_requests)
        
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception as e:
            status = "500"
            logger.error(f"Request failed: {e}")
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            method = request.method
            endpoint = self._normalize_endpoint(request.url.path)
            tenant_id = getattr(request.state, 'tenant_id', 'unknown')
            
            self.metrics.record_request(method, endpoint, status, duration, tenant_id)
            
            self.active_requests -= 1
            self.metrics.set_active_requests(self.active_requests)
        
        return response
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics."""
        # Remove request IDs and other variable parts
        if path.startswith('/v1/analyze'):
            return '/v1/analyze'
        elif path.startswith('/v1/feedback'):
            return '/v1/feedback'
        elif path.startswith('/v1/health'):
            return '/v1/health'
        elif path.startswith('/v1/readiness'):
            return '/v1/readiness'
        elif path.startswith('/v1/metrics'):
            return '/v1/metrics'
        elif path.startswith('/v1/config'):
            return '/v1/config'
        elif path.startswith('/v1/monitoring'):
            return '/v1/monitoring'
        else:
            return path


def create_metrics_endpoint():
    """Create FastAPI endpoint for Prometheus metrics."""
    from fastapi import Response
    
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        metrics_text = metrics.get_metrics_text()
        return Response(
            content=metrics_text,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    
    return prometheus_metrics