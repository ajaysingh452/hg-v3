"""Main service orchestrator for Harmony Guard."""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.models import AnalysisRequest, AnalysisResponse, FeedbackRequest, DecisionType, SeverityLevel
from ..core.interfaces import ConfigurationManagerInterface
from ..core.preprocessing import TextPreprocessor
from ..core.metrics import metrics
from ..core.logging import get_logger
from ..core.feedback import FeedbackManager
from ..core.active_learning import ActiveLearningPipeline
from ..core.model_retraining import DriftDetectionEngine, PerformanceMonitor, ModelRetrainingManager
from ..lpe.engine import LexiconPatternEngine
from ..model.classifier import TransformerClassifier
from ..intent import IntentContextLayer
from ..model.aggregator import EnsembleAggregator
from ..model.policy import PolicyEngine

# Use structured logger
app_logger = get_logger(__name__)
logger = logging.getLogger(__name__)


class HarmonyGuardService:
    """Main service orchestrator for Harmony Guard content moderation."""
    
    def __init__(self, config_manager: ConfigurationManagerInterface):
        """
        Initialize the Harmony Guard service.
        
        Args:
            config_manager: Configuration management instance
        """
        self.config_manager = config_manager
        self.preprocessor = None
        self.lpe_engine = None
        self.classifier = None
        self.intent_layer = None
        self.aggregator = None
        self.policy_engine = None
        self.feedback_manager = None
        self.active_learning = None
        self.performance_monitor = None
        self.drift_detector = None
        self.retraining_manager = None
        
        # Service metrics
        self.metrics = {
            "requests_total": 0,
            "requests_by_decision": {"allow": 0, "review": 0, "block": 0},
            "average_latency": 0.0,
            "errors_total": 0,
            "feedback_total": 0
        }
        
        # Request cache for feedback correlation
        self.request_cache = {}
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all service components."""
        try:
            logger.info("Initializing Harmony Guard components...")
            
            # Load configurations
            ensemble_config = self.config_manager.get_ensemble_config()
            preprocessing_config = self.config_manager.get_preprocessing_config()
            
            # Initialize components
            self.preprocessor = TextPreprocessor(preprocessing_config)
            self.lpe_engine = LexiconPatternEngine(self.config_manager)
            self.classifier = TransformerClassifier(ensemble_config.get("classifier", {}))
            self.intent_layer = IntentContextLayer(ensemble_config.get("intent", {}))
            self.aggregator = EnsembleAggregator(ensemble_config)
            self.policy_engine = PolicyEngine(self.config_manager)
            self.feedback_manager = FeedbackManager()
            self.active_learning = ActiveLearningPipeline(self.feedback_manager)
            self.performance_monitor = PerformanceMonitor()
            self.drift_detector = DriftDetectionEngine(self.feedback_manager, self.performance_monitor)
            self.retraining_manager = ModelRetrainingManager(self.drift_detector)
            
            # Initialize each component
            await self.preprocessor.initialize()
            await self.lpe_engine.initialize()
            await self.classifier.initialize()
            await self.intent_layer.initialize()
            await self.aggregator.initialize()
            await self.policy_engine.initialize()
            await self.feedback_manager.initialize()
            
            # Set initial component health metrics
            metrics.set_component_health("preprocessor", True)
            metrics.set_component_health("lpe_engine", True)
            metrics.set_component_health("classifier", True)
            metrics.set_component_health("intent_layer", True)
            metrics.set_component_health("aggregator", True)
            metrics.set_component_health("policy_engine", True)
            metrics.set_component_health("feedback_manager", True)
            metrics.set_component_health("active_learning", True)
            metrics.set_component_health("performance_monitor", True)
            metrics.set_component_health("drift_detector", True)
            metrics.set_component_health("retraining_manager", True)
            
            self._initialized = True
            logger.info("Harmony Guard service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Harmony Guard service: {e}")
            raise
    
    async def analyze(self, request: AnalysisRequest, request_id: str) -> AnalysisResponse:
        """
        Analyze content for corporate appropriateness with graceful degradation.
        
        Args:
            request: Analysis request
            request_id: Unique request identifier
            
        Returns:
            Analysis response with decision and details
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        start_time = asyncio.get_event_loop().time()
        component_errors = []
        
        try:
            # Step 1: Preprocess text (critical component)
            preprocessing_start = asyncio.get_event_loop().time()
            try:
                processed_text = await self.preprocessor.process(
                    request.text, 
                    request.language_hints
                )
                preprocessing_time = asyncio.get_event_loop().time() - preprocessing_start
                metrics.record_component_performance("preprocessor", preprocessing_time)
                
                # Record language detection metrics
                for lang in processed_text.detected_languages:
                    metrics.record_language_detection(lang.code, lang.confidence)
                
                logger.debug(f"Request {request_id}: Text preprocessing completed")
            except Exception as e:
                preprocessing_time = asyncio.get_event_loop().time() - preprocessing_start
                metrics.record_component_performance("preprocessor", preprocessing_time)
                metrics.record_component_error("preprocessor", type(e).__name__)
                
                app_logger.log_component_error("preprocessor", type(e).__name__, str(e))
                # Fallback to basic preprocessing
                processed_text = self._fallback_preprocessing(request.text)
                component_errors.append("preprocessing")
            
            # Step 2: Run ensemble components in parallel with error handling
            lpe_result = None
            classifier_result = None
            
            try:
                # Run components with individual error handling
                lpe_task = self._safe_component_call(
                    self.lpe_engine.analyze, processed_text, "LPE"
                )
                classifier_task = self._safe_component_call(
                    self.classifier.predict, processed_text, "Classifier"
                )
                
                results = await asyncio.gather(
                    lpe_task, classifier_task, return_exceptions=True
                )
                
                lpe_result = results[0] if not isinstance(results[0], Exception) else None
                classifier_result = results[1] if not isinstance(results[1], Exception) else None
                
                if lpe_result is None:
                    component_errors.append("lpe")
                    lpe_result = self._fallback_lpe_result()
                
                if classifier_result is None:
                    component_errors.append("classifier")
                    classifier_result = self._fallback_classifier_result()
                
                logger.debug(f"Request {request_id}: Ensemble components completed")
                
            except Exception as e:
                logger.error(f"Request {request_id}: Ensemble components failed: {e}")
                lpe_result = self._fallback_lpe_result()
                classifier_result = self._fallback_classifier_result()
                component_errors.extend(["lpe", "classifier"])
            
            # Step 3: Context analysis with error handling
            try:
                context_result = await self.intent_layer.analyze_context(
                    processed_text, lpe_result, classifier_result
                )
                logger.debug(f"Request {request_id}: Context analysis completed")
            except Exception as e:
                logger.error(f"Request {request_id}: Context analysis failed: {e}")
                context_result = self._fallback_context_result()
                component_errors.append("intent")
            
            # Step 4: Aggregate results with error handling
            try:
                aggregated_result = await self.aggregator.aggregate(
                    lpe_result, classifier_result, context_result
                )
                logger.debug(f"Request {request_id}: Result aggregation completed")
            except Exception as e:
                logger.error(f"Request {request_id}: Aggregation failed: {e}")
                aggregated_result = self._fallback_aggregation(
                    lpe_result, classifier_result, context_result
                )
                component_errors.append("aggregator")
            
            # Step 5: Apply policy rules with error handling
            try:
                final_result = await self.policy_engine.apply_policy(
                    aggregated_result, request.tenant_id
                )
                logger.debug(f"Request {request_id}: Policy application completed")
            except Exception as e:
                logger.error(f"Request {request_id}: Policy application failed: {e}")
                final_result = self._fallback_policy_application(aggregated_result)
                component_errors.append("policy")
            
            # Step 6: Build response
            response = self._build_response(
                final_result, processed_text, request.include_details, component_errors
            )
            
            # Cache request data for potential feedback correlation
            self.request_cache[request_id] = {
                'original_text': request.text,
                'decision': response.corporate_allowed.value,
                'confidence': response.confidence,
                'tenant_id': request.tenant_id,
                'timestamp': datetime.utcnow(),
                'categories': response.categories
            }
            
            # Analyze for active learning (async, don't wait)
            if self.active_learning:
                asyncio.create_task(
                    self.active_learning.analyze_prediction(
                        response, request_id, request.text, request.tenant_id
                    )
                )
            
            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(response.corporate_allowed, processing_time, component_errors)
            
            # Log completion with component status
            logger.info(
                f"Request {request_id} completed in {processing_time:.3f}s - "
                f"Decision: {response.corporate_allowed}, Confidence: {response.confidence:.3f}, "
                f"Errors: {component_errors or 'none'}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Request {request_id}: Critical error in analysis pipeline: {e}")
            self.metrics["errors_total"] += 1
            
            # Last resort fallback
            return self._emergency_fallback_response(request.text, request.include_details)
    
    async def _safe_component_call(self, component_method, *args, component_name: str):
        """Safely call a component method with timeout and error handling."""
        start_time = asyncio.get_event_loop().time()
        try:
            result = await asyncio.wait_for(component_method(*args), timeout=3.0)
            duration = asyncio.get_event_loop().time() - start_time
            metrics.record_component_performance(component_name.lower(), duration)
            return result
        except asyncio.TimeoutError:
            duration = asyncio.get_event_loop().time() - start_time
            metrics.record_component_performance(component_name.lower(), duration)
            metrics.record_component_error(component_name.lower(), "timeout")
            logger.warning(f"{component_name} component timed out")
            raise
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            metrics.record_component_performance(component_name.lower(), duration)
            metrics.record_component_error(component_name.lower(), type(e).__name__)
            logger.error(f"{component_name} component failed: {e}")
            raise
    
    async def submit_feedback(self, feedback: FeedbackRequest) -> bool:
        """
        Submit feedback for continuous learning.
        
        Args:
            feedback: Feedback request
            
        Returns:
            Success status
        """
        try:
            # Get cached request data for correlation
            cached_data = self.request_cache.get(feedback.request_id, {})
            
            # Submit feedback through feedback manager
            feedback_id = await self.feedback_manager.submit_feedback(
                feedback=feedback,
                tenant_id=cached_data.get('tenant_id'),
                original_text=cached_data.get('original_text'),
                original_decision=cached_data.get('decision'),
                confidence_score=cached_data.get('confidence')
            )
            
            logger.info(f"Processed feedback {feedback_id} for request {feedback.request_id}")
            
            self.metrics["feedback_total"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False
    
    async def is_ready(self) -> bool:
        """Check if service is ready to handle requests."""
        return self._initialized
    
    async def get_component_status(self) -> Dict[str, str]:
        """Get status of individual service components."""
        status = {}
        
        try:
            status["preprocessor"] = "ready" if self.preprocessor else "not_loaded"
            status["lpe_engine"] = "ready" if self.lpe_engine else "not_loaded"
            status["classifier"] = "ready" if self.classifier else "not_loaded"
            status["intent_layer"] = "ready" if self.intent_layer else "not_loaded"
            status["aggregator"] = "ready" if self.aggregator else "not_loaded"
            status["policy_engine"] = "ready" if self.policy_engine else "not_loaded"
            status["feedback_manager"] = "ready" if self.feedback_manager else "not_loaded"
            status["active_learning"] = "ready" if self.active_learning else "not_loaded"
            status["performance_monitor"] = "ready" if self.performance_monitor else "not_loaded"
            status["drift_detector"] = "ready" if self.drift_detector else "not_loaded"
            status["retraining_manager"] = "ready" if self.retraining_manager else "not_loaded"
            
            # Check if components are actually functional
            if self._initialized:
                # Could add more detailed health checks here
                for component_name in status:
                    if status[component_name] == "ready":
                        # Verify component is actually functional
                        component = getattr(self, component_name, None)
                        if component and hasattr(component, 'is_healthy'):
                            try:
                                is_healthy = await component.is_healthy()
                                if not is_healthy:
                                    status[component_name] = "unhealthy"
                            except Exception:
                                status[component_name] = "error"
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting component status: {e}")
            return {"error": str(e)}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return self.metrics.copy()
    
    async def get_public_config(self, tenant_id: str = None) -> Dict[str, Any]:
        """Get non-secret configuration."""
        try:
            ensemble_config = self.config_manager.get_ensemble_config(tenant_id)
            preprocessing_config = self.config_manager.get_preprocessing_config(tenant_id)
            
            # Return only non-secret configuration
            return {
                "supported_languages": preprocessing_config["preprocessing"]["language_detection"]["supported_languages"],
                "max_sequence_length": ensemble_config.get("classifier", {}).get("max_sequence_length", 512),
                "version": "1.0.0"
            }
        except Exception as e:
            logger.error(f"Error getting public config: {e}")
            return {}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics from model monitoring."""
        if not self.classifier:
            return {"error": "Classifier not initialized"}
        
        try:
            # Get classifier performance metrics
            classifier_metrics = self.classifier.get_performance_metrics()
            
            # Combine with service metrics
            combined_metrics = {
                "service_metrics": self.metrics.copy(),
                "model_metrics": classifier_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return combined_metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    async def get_drift_alerts(self, hours: int = 24) -> Dict[str, Any]:
        """Get drift alerts from model monitoring."""
        if not self.classifier:
            return {"error": "Classifier not initialized"}
        
        try:
            alerts = self.classifier.get_drift_alerts(hours)
            
            return {
                "alerts": [
                    {
                        "timestamp": alert.timestamp,
                        "drift_type": alert.drift_type,
                        "severity": alert.severity,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "baseline_value": alert.baseline_value,
                        "drift_score": alert.drift_score,
                        "description": alert.description
                    }
                    for alert in alerts
                ],
                "total_alerts": len(alerts),
                "hours_analyzed": hours
            }
            
        except Exception as e:
            logger.error(f"Error getting drift alerts: {e}")
            return {"error": str(e)}
    
    async def get_performance_summary(self, windows: int = 10) -> Dict[str, Any]:
        """Get performance summary over recent windows."""
        if not self.classifier:
            return {"error": "Classifier not initialized"}
        
        try:
            summary = self.classifier.get_performance_summary(windows)
            
            # Add service-level summary
            summary["service_summary"] = {
                "total_requests": self.metrics["requests_total"],
                "error_rate": self.metrics["errors_total"] / max(self.metrics["requests_total"], 1),
                "average_latency": self.metrics["average_latency"],
                "decision_distribution": self.metrics["requests_by_decision"]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    async def run_drift_analysis(self) -> Dict[str, Any]:
        """Run comprehensive drift analysis."""
        if not self.classifier:
            return {"error": "Classifier not initialized"}
        
        try:
            # Run statistical tests
            drift_tests = self.classifier.run_drift_tests()
            
            # Get recent alerts
            recent_alerts = self.classifier.get_drift_alerts(hours=24)
            
            # Combine results
            analysis = {
                "statistical_tests": drift_tests,
                "recent_alerts_count": len(recent_alerts),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "recommendations": self._generate_drift_recommendations(drift_tests, recent_alerts)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error running drift analysis: {e}")
            return {"error": str(e)}
    
    def _generate_drift_recommendations(self, drift_tests: Dict[str, Any], alerts: list) -> List[str]:
        """Generate recommendations based on drift analysis."""
        recommendations = []
        
        # Check statistical test results
        if "confidence_test" in drift_tests and drift_tests["confidence_test"].get("significant", False):
            recommendations.append("Significant confidence drift detected - consider model recalibration")
        
        if "processing_time_test" in drift_tests and drift_tests["processing_time_test"].get("significant", False):
            recommendations.append("Processing time drift detected - check system resources and model performance")
        
        if "distribution_test" in drift_tests and drift_tests["distribution_test"].get("significant", False):
            recommendations.append("Prediction distribution drift detected - review input data patterns")
        
        # Check alert severity
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        high_alerts = [a for a in alerts if a.severity == "high"]
        
        if critical_alerts:
            recommendations.append(f"Critical drift alerts detected ({len(critical_alerts)}) - immediate attention required")
        elif high_alerts:
            recommendations.append(f"High severity drift alerts detected ({len(high_alerts)}) - review recommended")
        
        if not recommendations:
            recommendations.append("No significant drift detected - system operating normally")
        
        return recommendations
    
    async def get_feedback_analytics(
        self, 
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get feedback analytics for monitoring and improvement."""
        if not self.feedback_manager:
            return {"error": "Feedback manager not initialized"}
        
        try:
            analytics = await self.feedback_manager.get_feedback_analytics(tenant_id, days)
            
            return {
                "total_feedback": analytics.total_feedback_count,
                "feedback_by_decision": analytics.feedback_by_decision,
                "feedback_by_category": analytics.feedback_by_category,
                "feedback_by_language": analytics.feedback_by_language,
                "correction_rate": analytics.correction_rate,
                "average_confidence_delta": analytics.average_confidence_delta,
                "recent_trend": analytics.recent_feedback_trend,
                "analysis_period_days": days,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback analytics: {e}")
            return {"error": str(e)}
    
    async def get_recent_feedback(
        self, 
        tenant_id: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get recent feedback records."""
        if not self.feedback_manager:
            return {"error": "Feedback manager not initialized"}
        
        try:
            records = await self.feedback_manager.get_recent_feedback(tenant_id, limit)
            
            # Convert to API format
            feedback_data = []
            for record in records:
                feedback_data.append({
                    "feedback_id": record.feedback_id,
                    "request_id": record.request_id,
                    "timestamp": record.timestamp.isoformat(),
                    "final_label": record.final_label,
                    "actual_categories": record.actual_categories,
                    "comment": record.comment,
                    "language_hints": record.language_hints,
                    "tenant_id": record.tenant_id,
                    "confidence_score": record.confidence_score,
                    "original_decision": record.original_decision
                })
            
            return {
                "feedback_records": feedback_data,
                "total_count": len(feedback_data),
                "limit": limit,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return {"error": str(e)}
    
    async def get_review_queue_status(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get active learning review queue status."""
        if not self.active_learning:
            return {"error": "Active learning not initialized"}
        
        try:
            return await self.active_learning.get_review_queue_status(tenant_id)
        except Exception as e:
            logger.error(f"Error getting review queue status: {e}")
            return {"error": str(e)}
    
    async def get_pending_reviews(
        self, 
        limit: int = 50,
        priority_filter: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pending review items for human reviewers."""
        if not self.active_learning:
            return []
        
        try:
            return await self.active_learning.get_pending_reviews(
                limit=limit,
                priority_filter=priority_filter,
                tenant_id=tenant_id
            )
        except Exception as e:
            logger.error(f"Error getting pending reviews: {e}")
            return []
    
    async def submit_review_feedback(
        self, 
        review_id: str,
        corrected_decision: str,
        corrected_categories: List[str],
        reviewer_comment: Optional[str] = None
    ) -> bool:
        """Submit feedback from human review."""
        if not self.active_learning:
            return False
        
        try:
            return await self.active_learning.process_review_feedback(
                review_id=review_id,
                corrected_decision=corrected_decision,
                corrected_categories=corrected_categories,
                reviewer_comment=reviewer_comment
            )
        except Exception as e:
            logger.error(f"Error submitting review feedback: {e}")
            return False
    
    async def run_comprehensive_drift_analysis(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive drift analysis."""
        if not self.drift_detector:
            return {"error": "Drift detector not initialized"}
        
        try:
            return await self.drift_detector.run_drift_analysis(tenant_id)
        except Exception as e:
            logger.error(f"Error running drift analysis: {e}")
            return {"error": str(e)}
    
    async def get_drift_alerts_detailed(
        self, 
        hours: int = 24,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed drift alerts."""
        if not self.drift_detector:
            return {"error": "Drift detector not initialized"}
        
        try:
            alerts = await self.drift_detector.get_drift_alerts(hours, tenant_id)
            
            return {
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "drift_type": alert.drift_type.value,
                        "severity": alert.severity.value,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "baseline_value": alert.baseline_value,
                        "drift_score": alert.drift_score,
                        "description": alert.description,
                        "timestamp": alert.timestamp.isoformat(),
                        "statistical_test": alert.statistical_test,
                        "p_value": alert.p_value
                    }
                    for alert in alerts
                ],
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.severity.value == "critical"]),
                "high_alerts": len([a for a in alerts if a.severity.value == "high"]),
                "hours_analyzed": hours,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting drift alerts: {e}")
            return {"error": str(e)}
    
    async def check_retraining_triggers(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if model retraining should be triggered."""
        if not self.retraining_manager:
            return {"error": "Retraining manager not initialized"}
        
        try:
            job_id = await self.retraining_manager.check_retraining_triggers(tenant_id)
            
            if job_id:
                return {
                    "retraining_triggered": True,
                    "job_id": job_id,
                    "message": "Model retraining job created",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "retraining_triggered": False,
                    "message": "No retraining triggers detected",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error checking retraining triggers: {e}")
            return {"error": str(e)}
    
    async def get_retraining_status(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get model retraining status."""
        if not self.retraining_manager:
            return {"error": "Retraining manager not initialized"}
        
        try:
            return await self.retraining_manager.get_retraining_status(tenant_id)
        except Exception as e:
            logger.error(f"Error getting retraining status: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown service and cleanup resources."""
        logger.info("Shutting down Harmony Guard service...")
        
        # Cleanup components
        if self.classifier:
            await self.classifier.shutdown()
        
        if self.feedback_manager:
            await self.feedback_manager.shutdown()
        
        self._initialized = False
        logger.info("Harmony Guard service shutdown complete")
    
    def _build_response(
        self, 
        result, 
        processed_text, 
        include_details: bool,
        component_errors: List[str] = None
    ) -> AnalysisResponse:
        """Build API response from analysis result."""
        
        # Convert languages to response format
        languages = [
            {"code": lang.code, "pct": lang.percentage}
            for lang in processed_text.detected_languages
        ]
        
        # Build basic response
        response = AnalysisResponse(
            corporate_allowed=result.final_decision,
            confidence=result.confidence_score,
            severity=result.severity_level,
            categories=[cat for cat, score in result.category_scores.items() if score > 0.5],
            languages=languages
        )
        
        # Add details if requested
        if include_details:
            response.spans = result.consolidated_spans
            response.explanations = result.explanation_traces
            response.normalized_preview = processed_text.normalized_text
            response.policy_trace = getattr(result, 'policy_trace', [])
            
            # Add component error information if present
            if component_errors:
                if response.explanations is None:
                    response.explanations = []
                response.explanations.append(
                    f"Component errors encountered: {', '.join(component_errors)}"
                )
        
        return response
    
    def _fallback_preprocessing(self, text: str):
        """Fallback preprocessing when main preprocessor fails."""
        from ..core.models import ProcessedText, LanguageDetection
        
        return ProcessedText(
            original_text=text,
            normalized_text=text.lower().strip(),
            detected_languages=[LanguageDetection(code="en", confidence=0.5, percentage=100.0)],
            tokens=text.split(),
            transliterations={},
            obfuscation_map={},
            pii_masked=False
        )
    
    def _fallback_lpe_result(self):
        """Fallback LPE result when LPE engine fails."""
        from ..core.models import LPEResult
        
        return LPEResult(
            matched_spans=[],
            categories=[],
            confidence_scores={},
            rule_traces=["LPE engine unavailable - using fallback"]
        )
    
    def _fallback_classifier_result(self):
        """Fallback classifier result when classifier fails."""
        from ..core.models import ClassifierResult
        
        return ClassifierResult(
            category_probabilities={},
            corporate_decision_prob={"allow": 0.7, "review": 0.2, "block": 0.1},
            severity_scores={"low": 0.8, "medium": 0.2, "high": 0.0, "critical": 0.0},
            attention_spans=[]
        )
    
    def _fallback_context_result(self):
        """Fallback context result when intent layer fails."""
        from ..core.models import ContextResult, DecisionType
        
        return ContextResult(
            context_modifiers={},
            safe_context_detected={},
            recommended_action=DecisionType.REVIEW
        )
    
    def _fallback_aggregation(self, lpe_result, classifier_result, context_result):
        """Fallback aggregation when aggregator fails."""
        from ..core.models import AggregatedResult, DecisionType, SeverityLevel
        
        # Conservative fallback - default to review
        return AggregatedResult(
            final_decision=DecisionType.REVIEW,
            confidence_score=0.5,
            category_scores={},
            severity_level=SeverityLevel.MEDIUM,
            explanation_traces=["Fallback aggregation used - manual review recommended"],
            consolidated_spans=[]
        )
    
    def _fallback_policy_application(self, aggregated_result):
        """Fallback policy application when policy engine fails."""
        # Add fallback policy trace
        explanation_traces = aggregated_result.explanation_traces + [
            "Policy engine unavailable - using conservative defaults"
        ]
        
        # Conservative policy - escalate uncertain cases
        if aggregated_result.confidence_score > 0.8:
            final_decision = DecisionType.BLOCK
        else:
            final_decision = DecisionType.REVIEW
        
        from ..core.models import AggregatedResult
        
        result = AggregatedResult(
            final_decision=final_decision,
            confidence_score=aggregated_result.confidence_score,
            category_scores=aggregated_result.category_scores,
            severity_level=aggregated_result.severity_level,
            explanation_traces=explanation_traces,
            consolidated_spans=aggregated_result.consolidated_spans
        )
        
        setattr(result, 'policy_trace', ["Fallback policy applied"])
        return result
    
    def _emergency_fallback_response(self, text: str, include_details: bool):
        """Emergency fallback response when entire pipeline fails."""
        from ..core.models import AnalysisResponse, DecisionType, SeverityLevel
        
        return AnalysisResponse(
            corporate_allowed=DecisionType.REVIEW,
            confidence=0.0,
            severity=SeverityLevel.MEDIUM,
            categories=[],
            languages=[{"code": "unknown", "pct": 100.0}],
            spans=[] if include_details else None,
            explanations=["System error - manual review required"] if include_details else None,
            normalized_preview=text[:100] + "..." if len(text) > 100 else text if include_details else None,
            policy_trace=["Emergency fallback applied"] if include_details else None
        )
    
    def _update_metrics(self, decision: DecisionType, processing_time: float, component_errors: List[str] = None):
        """Update service metrics with component error tracking."""
        self.metrics["requests_total"] += 1
        self.metrics["requests_by_decision"][decision.value] += 1
        
        # Track component errors
        if component_errors:
            if "component_errors" not in self.metrics:
                self.metrics["component_errors"] = {}
            
            for component in component_errors:
                if component not in self.metrics["component_errors"]:
                    self.metrics["component_errors"][component] = 0
                self.metrics["component_errors"][component] += 1
        
        # Update average latency (simple moving average)
        current_avg = self.metrics["average_latency"]
        total_requests = self.metrics["requests_total"]
        self.metrics["average_latency"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )