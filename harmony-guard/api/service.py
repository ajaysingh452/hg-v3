"""Main service orchestrator for Harmony Guard."""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..core.models import AnalysisRequest, AnalysisResponse, FeedbackRequest, DecisionType, SeverityLevel
from ..core.interfaces import ConfigurationManagerInterface
from ..core.preprocessing import TextPreprocessor
from ..lpe.engine import LexiconPatternEngine
from ..model.classifier import TransformerClassifier
from ..model.intent import IntentContextLayer
from ..model.aggregator import EnsembleAggregator
from ..model.policy import PolicyEngine


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
        
        # Service metrics
        self.metrics = {
            "requests_total": 0,
            "requests_by_decision": {"allow": 0, "review": 0, "block": 0},
            "average_latency": 0.0,
            "errors_total": 0,
            "feedback_total": 0
        }
        
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
            
            # Initialize each component
            await self.preprocessor.initialize()
            await self.lpe_engine.initialize()
            await self.classifier.initialize()
            await self.intent_layer.initialize()
            await self.aggregator.initialize()
            await self.policy_engine.initialize()
            
            self._initialized = True
            logger.info("Harmony Guard service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Harmony Guard service: {e}")
            raise
    
    async def analyze(self, request: AnalysisRequest, request_id: str) -> AnalysisResponse:
        """
        Analyze content for corporate appropriateness.
        
        Args:
            request: Analysis request
            request_id: Unique request identifier
            
        Returns:
            Analysis response with decision and details
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Step 1: Preprocess text
            processed_text = await self.preprocessor.process(
                request.text, 
                request.language_hints
            )
            
            # Step 2: Run ensemble components in parallel
            lpe_task = self.lpe_engine.analyze(processed_text)
            classifier_task = self.classifier.predict(processed_text)
            
            lpe_result, classifier_result = await asyncio.gather(
                lpe_task, classifier_task
            )
            
            # Step 3: Context analysis
            context_result = await self.intent_layer.analyze_context(
                processed_text, lpe_result, classifier_result
            )
            
            # Step 4: Aggregate results
            aggregated_result = await self.aggregator.aggregate(
                lpe_result, classifier_result, context_result
            )
            
            # Step 5: Apply policy rules
            final_result = await self.policy_engine.apply_policy(
                aggregated_result, request.tenant_id
            )
            
            # Step 6: Build response
            response = self._build_response(
                final_result, processed_text, request.include_details
            )
            
            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(response.corporate_allowed, processing_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            self.metrics["errors_total"] += 1
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
            # Store feedback for future model training
            # In a real implementation, this would write to a database
            logger.info(f"Received feedback for request {feedback.request_id}")
            
            self.metrics["feedback_total"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False
    
    async def is_ready(self) -> bool:
        """Check if service is ready to handle requests."""
        return self._initialized
    
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
    
    async def shutdown(self):
        """Shutdown service and cleanup resources."""
        logger.info("Shutting down Harmony Guard service...")
        
        # Cleanup components
        if self.classifier:
            await self.classifier.shutdown()
        
        self._initialized = False
        logger.info("Harmony Guard service shutdown complete")
    
    def _build_response(
        self, 
        result, 
        processed_text, 
        include_details: bool
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
        
        return response
    
    def _update_metrics(self, decision: DecisionType, processing_time: float):
        """Update service metrics."""
        self.metrics["requests_total"] += 1
        self.metrics["requests_by_decision"][decision.value] += 1
        
        # Update average latency (simple moving average)
        current_avg = self.metrics["average_latency"]
        total_requests = self.metrics["requests_total"]
        self.metrics["average_latency"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )