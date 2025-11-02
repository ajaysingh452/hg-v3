"""Model performance monitoring and drift detection."""

import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for model predictions."""
    timestamp: float
    prediction_distribution: Dict[str, float]
    confidence_score: float
    processing_time: float
    language_codes: List[str]
    input_length: int


@dataclass
class PerformanceWindow:
    """Performance metrics over a time window."""
    start_time: float
    end_time: float
    total_predictions: int
    avg_confidence: float
    avg_processing_time: float
    prediction_distribution: Dict[str, float]
    language_distribution: Dict[str, int]
    drift_score: float = 0.0


@dataclass
class DriftAlert:
    """Alert for detected model drift."""
    timestamp: float
    drift_type: str
    severity: str
    metric_name: str
    current_value: float
    baseline_value: float
    drift_score: float
    description: str


class ModelPerformanceMonitor:
    """Monitors model performance and detects drift."""
    
    def __init__(self, config: Dict):
        """Initialize performance monitor."""
        self.config = config
        self.window_size = config.get("window_size", 1000)
        self.time_window = config.get("time_window", 3600)
        self.drift_threshold = config.get("drift_threshold", 0.05)
        self.confidence_threshold = config.get("confidence_threshold", 0.1)
        
        self.prediction_history = deque(maxlen=self.window_size * 2)
        self.performance_windows = deque(maxlen=100)
        self.drift_alerts = deque(maxlen=1000)
        
        self.baseline_metrics = {}
        self.baseline_established = False
        self.current_window_start = time.time()
        self.current_window_predictions = []
        
        self.category_distributions = defaultdict(list)
        self.confidence_history = deque(maxlen=self.window_size)
        self.processing_time_history = deque(maxlen=self.window_size)       
 
    def record_prediction(self, predictions: Dict[str, Any], confidence_score: float,
                         processing_time: float, language_codes: List[str], input_length: int):
        """Record a model prediction for monitoring."""
        timestamp = time.time()
        prediction_dist = self._extract_prediction_distribution(predictions)
        
        metrics = PredictionMetrics(
            timestamp=timestamp,
            prediction_distribution=prediction_dist,
            confidence_score=confidence_score,
            processing_time=processing_time,
            language_codes=language_codes,
            input_length=input_length
        )
        
        self.prediction_history.append(metrics)
        self.current_window_predictions.append(metrics)
        self.confidence_history.append(confidence_score)
        self.processing_time_history.append(processing_time)
        
        for category, prob in prediction_dist.items():
            self.category_distributions[category].append(prob)
            if len(self.category_distributions[category]) > self.window_size:
                self.category_distributions[category].pop(0)
        
        if self._should_process_window():
            self._process_current_window()
        
        if len(self.prediction_history) >= self.window_size:
            self._check_for_drift()
    
    def _extract_prediction_distribution(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """Extract prediction distribution from model output."""
        if "category_probabilities" in predictions:
            return predictions["category_probabilities"]
        elif "corporate_decision_prob" in predictions:
            return predictions["corporate_decision_prob"]
        else:
            return {str(k): float(v) for k, v in predictions.items() if isinstance(v, (int, float))}
    
    def _should_process_window(self) -> bool:
        """Check if current window should be processed."""
        current_time = time.time()
        time_elapsed = current_time - self.current_window_start
        return (len(self.current_window_predictions) >= self.window_size or 
                time_elapsed >= self.time_window)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.performance_windows:
            return {"baseline_established": self.baseline_established}
        
        latest_window = self.performance_windows[-1]
        metrics = {
            "current_window": {
                "total_predictions": latest_window.total_predictions,
                "avg_confidence": latest_window.avg_confidence,
                "avg_processing_time": latest_window.avg_processing_time,
                "prediction_distribution": latest_window.prediction_distribution,
                "language_distribution": latest_window.language_distribution
            },
            "recent_alerts": len([a for a in self.drift_alerts if time.time() - a.timestamp < 3600]),
            "baseline_established": self.baseline_established
        }
        
        if self.baseline_established:
            metrics["baseline"] = self.baseline_metrics
        
        return metrics
    
    def get_drift_alerts(self, hours: int = 24) -> List[DriftAlert]:
        """Get drift alerts from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.drift_alerts if alert.timestamp >= cutoff_time]
    
    def get_performance_summary(self, windows: int = 10) -> Dict[str, Any]:
        """Get performance summary over recent windows."""
        if not self.performance_windows:
            return {}
        
        recent_windows = list(self.performance_windows)[-windows:]
        return {
            "windows_analyzed": len(recent_windows),
            "total_predictions": sum(w.total_predictions for w in recent_windows),
            "avg_confidence": np.mean([w.avg_confidence for w in recent_windows]),
            "confidence_std": np.std([w.avg_confidence for w in recent_windows]),
            "avg_processing_time": np.mean([w.avg_processing_time for w in recent_windows]),
            "processing_time_std": np.std([w.avg_processing_time for w in recent_windows]),
            "time_range": {
                "start": recent_windows[0].start_time,
                "end": recent_windows[-1].end_time
            }
        }
    
    def run_statistical_tests(self) -> Dict[str, Any]:
        """Run statistical tests for drift detection."""
        if len(self.performance_windows) < 10:
            return {"error": "Insufficient data for statistical tests"}
        
        mid_point = len(self.performance_windows) // 2
        early_windows = list(self.performance_windows)[:mid_point]
        recent_windows = list(self.performance_windows)[mid_point:]
        
        results = {}
        
        early_confidence = [w.avg_confidence for w in early_windows]
        recent_confidence = [w.avg_confidence for w in recent_windows]
        
        confidence_stat, confidence_p = stats.ttest_ind(early_confidence, recent_confidence)
        results["confidence_test"] = {
            "statistic": confidence_stat,
            "p_value": confidence_p,
            "significant": confidence_p < 0.05,
            "early_mean": np.mean(early_confidence),
            "recent_mean": np.mean(recent_confidence)
        }
        
        return results
    
    def _process_current_window(self):
        """Process current window and create performance window."""
        if not self.current_window_predictions:
            return
        
        current_time = time.time()
        
        # Calculate window metrics
        total_predictions = len(self.current_window_predictions)
        avg_confidence = np.mean([p.confidence_score for p in self.current_window_predictions])
        avg_processing_time = np.mean([p.processing_time for p in self.current_window_predictions])
        
        # Calculate prediction distribution
        all_distributions = [p.prediction_distribution for p in self.current_window_predictions]
        prediction_distribution = {}
        
        if all_distributions:
            # Get all unique categories
            all_categories = set()
            for dist in all_distributions:
                all_categories.update(dist.keys())
            
            # Calculate average probability for each category
            for category in all_categories:
                category_probs = [dist.get(category, 0.0) for dist in all_distributions]
                prediction_distribution[category] = np.mean(category_probs)
        
        # Calculate language distribution
        language_distribution = defaultdict(int)
        for prediction in self.current_window_predictions:
            for lang_code in prediction.language_codes:
                language_distribution[lang_code] += 1
        
        # Create performance window
        window = PerformanceWindow(
            start_time=self.current_window_start,
            end_time=current_time,
            total_predictions=total_predictions,
            avg_confidence=avg_confidence,
            avg_processing_time=avg_processing_time,
            prediction_distribution=prediction_distribution,
            language_distribution=dict(language_distribution)
        )
        
        self.performance_windows.append(window)
        
        # Establish baseline if we have enough windows
        if not self.baseline_established and len(self.performance_windows) >= 5:
            self._establish_baseline()
        
        # Reset current window
        self.current_window_predictions = []
        self.current_window_start = current_time
        
        logger.debug(f"Processed window with {total_predictions} predictions")
    
    def _check_for_drift(self):
        """Check for drift in recent predictions."""
        if not self.baseline_established or len(self.performance_windows) < 2:
            return
        
        latest_window = self.performance_windows[-1]
        
        # Check for different types of drift
        alerts = []
        
        # Confidence drift
        confidence_alert = self._detect_confidence_drift(latest_window)
        if confidence_alert:
            alerts.append(confidence_alert)
        
        # Performance drift
        performance_alert = self._detect_performance_drift(latest_window)
        if performance_alert:
            alerts.append(performance_alert)
        
        # Prediction distribution drift
        prediction_alerts = self._detect_prediction_drift(latest_window)
        alerts.extend(prediction_alerts)
        
        # Add alerts to history
        for alert in alerts:
            self.drift_alerts.append(alert)
            logger.warning(f"Drift detected: {alert.drift_type} - {alert.description}")
    
    def _establish_baseline(self):
        """Establish baseline metrics from recent windows."""
        if len(self.performance_windows) < 5:
            return
        
        recent_windows = list(self.performance_windows)[-5:]
        
        # Calculate baseline metrics
        confidences = [w.avg_confidence for w in recent_windows]
        processing_times = [w.avg_processing_time for w in recent_windows]
        
        # Aggregate prediction distributions
        all_categories = set()
        for window in recent_windows:
            all_categories.update(window.prediction_distribution.keys())
        
        category_stats = {}
        for category in all_categories:
            category_probs = []
            for window in recent_windows:
                category_probs.append(window.prediction_distribution.get(category, 0.0))
            
            category_stats[category] = {
                "mean": np.mean(category_probs),
                "std": np.std(category_probs)
            }
        
        self.baseline_metrics = {
            "avg_confidence": np.mean(confidences),
            "confidence_std": np.std(confidences),
            "avg_processing_time": np.mean(processing_times),
            "processing_time_std": np.std(processing_times),
            "prediction_distribution": category_stats
        }
        
        self.baseline_established = True
        logger.info("Baseline metrics established")
    
    def _detect_confidence_drift(self, window: PerformanceWindow) -> Optional[DriftAlert]:
        """Detect confidence drift."""
        if not self.baseline_established:
            return None
        
        baseline_confidence = self.baseline_metrics["avg_confidence"]
        confidence_std = self.baseline_metrics["confidence_std"]
        
        # Calculate drift score
        confidence_diff = abs(window.avg_confidence - baseline_confidence)
        drift_score = confidence_diff / (confidence_std + 1e-8)
        
        # Check if drift exceeds threshold
        if drift_score > self.drift_threshold:
            severity = self._calculate_severity(drift_score)
            
            return DriftAlert(
                timestamp=time.time(),
                drift_type="confidence",
                severity=severity,
                metric_name="avg_confidence",
                current_value=window.avg_confidence,
                baseline_value=baseline_confidence,
                drift_score=drift_score,
                description=f"Confidence drift detected: {window.avg_confidence:.3f} vs baseline {baseline_confidence:.3f}"
            )
        
        return None
    
    def _detect_performance_drift(self, window: PerformanceWindow) -> Optional[DriftAlert]:
        """Detect performance drift."""
        if not self.baseline_established:
            return None
        
        baseline_time = self.baseline_metrics["avg_processing_time"]
        time_std = self.baseline_metrics["processing_time_std"]
        
        # Calculate drift score
        time_diff = abs(window.avg_processing_time - baseline_time)
        drift_score = time_diff / (time_std + 1e-8)
        
        # Check if drift exceeds threshold
        if drift_score > self.drift_threshold:
            severity = self._calculate_severity(drift_score)
            
            return DriftAlert(
                timestamp=time.time(),
                drift_type="performance",
                severity=severity,
                metric_name="avg_processing_time",
                current_value=window.avg_processing_time,
                baseline_value=baseline_time,
                drift_score=drift_score,
                description=f"Performance drift detected: {window.avg_processing_time:.1f}ms vs baseline {baseline_time:.1f}ms"
            )
        
        return None
    
    def _detect_prediction_drift(self, window: PerformanceWindow) -> List[DriftAlert]:
        """Detect prediction distribution drift."""
        if not self.baseline_established:
            return []
        
        alerts = []
        baseline_dist = self.baseline_metrics["prediction_distribution"]
        
        for category, current_prob in window.prediction_distribution.items():
            if category in baseline_dist:
                baseline_mean = baseline_dist[category]["mean"]
                baseline_std = baseline_dist[category]["std"]
                
                # Calculate drift score
                prob_diff = abs(current_prob - baseline_mean)
                drift_score = prob_diff / (baseline_std + 1e-8)
                
                # Check if drift exceeds threshold
                if drift_score > self.drift_threshold:
                    severity = self._calculate_severity(drift_score)
                    
                    alert = DriftAlert(
                        timestamp=time.time(),
                        drift_type="prediction",
                        severity=severity,
                        metric_name=f"prediction_{category}",
                        current_value=current_prob,
                        baseline_value=baseline_mean,
                        drift_score=drift_score,
                        description=f"Prediction drift in {category}: {current_prob:.3f} vs baseline {baseline_mean:.3f}"
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _calculate_severity(self, drift_score: float) -> str:
        """Calculate severity level based on drift score."""
        if drift_score > 3.0:
            return "critical"
        elif drift_score > 2.0:
            return "high"
        elif drift_score > 1.0:
            return "medium"
        else:
            return "low"