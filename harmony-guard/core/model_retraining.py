"""Automated model retraining and drift detection system."""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from scipy import stats
import hashlib

from .models import DecisionType, SeverityLevel
from .feedback import FeedbackManager, FeedbackRecord

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of model drift."""
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    PREDICTION_DRIFT = "prediction_drift"


class DriftSeverity(str, Enum):
    """Severity levels for drift detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetrainingTrigger(str, Enum):
    """Triggers for model retraining."""
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FEEDBACK_THRESHOLD = "feedback_threshold"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


@dataclass
class DriftAlert:
    """Drift detection alert."""
    alert_id: str
    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    current_value: float
    baseline_value: float
    drift_score: float
    description: str
    timestamp: datetime
    tenant_id: Optional[str] = None
    statistical_test: Optional[str] = None
    p_value: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    timestamp: datetime
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confidence_distribution: Dict[str, float]
    decision_distribution: Dict[str, int]
    category_performance: Dict[str, Dict[str, float]]
    language_performance: Dict[str, Dict[str, float]]
    tenant_id: Optional[str] = None


@dataclass
class RetrainingJob:
    """Model retraining job."""
    job_id: str
    trigger: RetrainingTrigger
    status: str  # pending, running, completed, failed
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    model_version: str
    training_data_size: int
    performance_metrics: Optional[PerformanceMetrics]
    error_message: Optional[str] = None
    tenant_id: Optional[str] = None


class StatisticalDriftDetector:
    """Statistical tests for drift detection."""
    
    def __init__(self, significance_level: float = 0.05, window_size: int = 1000):
        """
        Initialize drift detector.
        
        Args:
            significance_level: Statistical significance threshold
            window_size: Size of sliding window for comparison
        """
        self.significance_level = significance_level
        self.window_size = window_size
        
    def detect_distribution_drift(
        self, 
        baseline_data: List[float], 
        current_data: List[float]
    ) -> Tuple[bool, float, str]:
        """
        Detect distribution drift using Kolmogorov-Smirnov test.
        
        Args:
            baseline_data: Baseline distribution
            current_data: Current distribution
            
        Returns:
            (drift_detected, p_value, test_name)
        """
        try:
            if len(baseline_data) < 10 or len(current_data) < 10:
                return False, 1.0, "insufficient_data"
            
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(baseline_data, current_data)
            
            drift_detected = p_value < self.significance_level
            
            return drift_detected, p_value, "kolmogorov_smirnov"
            
        except Exception as e:
            logger.error(f"Error in distribution drift detection: {e}")
            return False, 1.0, "error"
    
    def detect_mean_drift(
        self, 
        baseline_data: List[float], 
        current_data: List[float]
    ) -> Tuple[bool, float, str]:
        """
        Detect mean drift using t-test.
        
        Args:
            baseline_data: Baseline values
            current_data: Current values
            
        Returns:
            (drift_detected, p_value, test_name)
        """
        try:
            if len(baseline_data) < 10 or len(current_data) < 10:
                return False, 1.0, "insufficient_data"
            
            # Two-sample t-test
            statistic, p_value = stats.ttest_ind(baseline_data, current_data)
            
            drift_detected = p_value < self.significance_level
            
            return drift_detected, p_value, "t_test"
            
        except Exception as e:
            logger.error(f"Error in mean drift detection: {e}")
            return False, 1.0, "error"
    
    def detect_variance_drift(
        self, 
        baseline_data: List[float], 
        current_data: List[float]
    ) -> Tuple[bool, float, str]:
        """
        Detect variance drift using F-test.
        
        Args:
            baseline_data: Baseline values
            current_data: Current values
            
        Returns:
            (drift_detected, p_value, test_name)
        """
        try:
            if len(baseline_data) < 10 or len(current_data) < 10:
                return False, 1.0, "insufficient_data"
            
            # F-test for equal variances
            var1 = np.var(baseline_data, ddof=1)
            var2 = np.var(current_data, ddof=1)
            
            if var1 == 0 or var2 == 0:
                return False, 1.0, "zero_variance"
            
            f_statistic = var1 / var2 if var1 > var2 else var2 / var1
            df1 = len(baseline_data) - 1
            df2 = len(current_data) - 1
            
            p_value = 2 * (1 - stats.f.cdf(f_statistic, df1, df2))
            
            drift_detected = p_value < self.significance_level
            
            return drift_detected, p_value, "f_test"
            
        except Exception as e:
            logger.error(f"Error in variance drift detection: {e}")
            return False, 1.0, "error"


class PerformanceMonitor:
    """Monitor model performance metrics."""
    
    def __init__(self, storage_path: str = "logs/performance"):
        """
        Initialize performance monitor.
        
        Args:
            storage_path: Path to store performance data
        """
        self.storage_path = Path(storage_path)
        self.db_path = self.storage_path / "performance.db"
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for performance tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        precision REAL NOT NULL,
                        recall REAL NOT NULL,
                        f1_score REAL NOT NULL,
                        accuracy REAL NOT NULL,
                        confidence_distribution TEXT NOT NULL,
                        decision_distribution TEXT NOT NULL,
                        category_performance TEXT,
                        language_performance TEXT,
                        tenant_id TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_performance_timestamp 
                    ON performance_metrics(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_performance_tenant 
                    ON performance_metrics(tenant_id)
                """)
                
                conn.commit()
                logger.info("Performance monitoring database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize performance database: {e}")
            raise
    
    async def record_performance_metrics(self, metrics: PerformanceMetrics) -> bool:
        """Record performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (timestamp, precision, recall, f1_score, accuracy,
                     confidence_distribution, decision_distribution,
                     category_performance, language_performance, tenant_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp.isoformat(),
                    metrics.precision,
                    metrics.recall,
                    metrics.f1_score,
                    metrics.accuracy,
                    json.dumps(metrics.confidence_distribution),
                    json.dumps(metrics.decision_distribution),
                    json.dumps(metrics.category_performance),
                    json.dumps(metrics.language_performance),
                    metrics.tenant_id
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record performance metrics: {e}")
            return False
    
    async def get_performance_history(
        self, 
        days: int = 30,
        tenant_id: Optional[str] = None
    ) -> List[PerformanceMetrics]:
        """Get performance metrics history."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            query = """
                SELECT * FROM performance_metrics 
                WHERE timestamp >= ?
            """
            params = [start_date.isoformat()]
            
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            
            query += " ORDER BY timestamp DESC"
            
            metrics_list = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    metrics = PerformanceMetrics(
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        precision=row['precision'],
                        recall=row['recall'],
                        f1_score=row['f1_score'],
                        accuracy=row['accuracy'],
                        confidence_distribution=json.loads(row['confidence_distribution']),
                        decision_distribution=json.loads(row['decision_distribution']),
                        category_performance=json.loads(row['category_performance']) if row['category_performance'] else {},
                        language_performance=json.loads(row['language_performance']) if row['language_performance'] else {},
                        tenant_id=row['tenant_id']
                    )
                    metrics_list.append(metrics)
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []


class DriftDetectionEngine:
    """Main drift detection engine."""
    
    def __init__(
        self, 
        feedback_manager: FeedbackManager,
        performance_monitor: PerformanceMonitor,
        storage_path: str = "logs/drift"
    ):
        """
        Initialize drift detection engine.
        
        Args:
            feedback_manager: Feedback manager instance
            performance_monitor: Performance monitor instance
            storage_path: Path for drift detection storage
        """
        self.feedback_manager = feedback_manager
        self.performance_monitor = performance_monitor
        self.storage_path = Path(storage_path)
        self.db_path = self.storage_path / "drift_alerts.db"
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.statistical_detector = StatisticalDriftDetector()
        self._init_database()
        
        # Drift detection thresholds
        self.thresholds = {
            "confidence_drift": 0.1,  # 10% change in average confidence
            "decision_drift": 0.15,   # 15% change in decision distribution
            "performance_drift": 0.05, # 5% drop in F1 score
            "feedback_rate_drift": 0.2  # 20% increase in correction rate
        }
    
    def _init_database(self):
        """Initialize drift alerts database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS drift_alerts (
                        alert_id TEXT PRIMARY KEY,
                        drift_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        baseline_value REAL NOT NULL,
                        drift_score REAL NOT NULL,
                        description TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        tenant_id TEXT,
                        statistical_test TEXT,
                        p_value REAL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_drift_timestamp 
                    ON drift_alerts(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_drift_severity 
                    ON drift_alerts(severity)
                """)
                
                conn.commit()
                logger.info("Drift detection database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize drift database: {e}")
            raise
    
    async def run_drift_analysis(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive drift analysis."""
        try:
            alerts = []
            
            # Get recent performance data
            recent_metrics = await self.performance_monitor.get_performance_history(
                days=7, tenant_id=tenant_id
            )
            baseline_metrics = await self.performance_monitor.get_performance_history(
                days=30, tenant_id=tenant_id
            )
            
            if len(recent_metrics) < 5 or len(baseline_metrics) < 10:
                return {
                    "status": "insufficient_data",
                    "message": "Not enough data for drift analysis",
                    "alerts": []
                }
            
            # Confidence drift detection
            confidence_alert = await self._detect_confidence_drift(
                recent_metrics, baseline_metrics, tenant_id
            )
            if confidence_alert:
                alerts.append(confidence_alert)
            
            # Decision distribution drift
            decision_alert = await self._detect_decision_drift(
                recent_metrics, baseline_metrics, tenant_id
            )
            if decision_alert:
                alerts.append(decision_alert)
            
            # Performance drift detection
            performance_alert = await self._detect_performance_drift(
                recent_metrics, baseline_metrics, tenant_id
            )
            if performance_alert:
                alerts.append(performance_alert)
            
            # Feedback-based drift detection
            feedback_alert = await self._detect_feedback_drift(tenant_id)
            if feedback_alert:
                alerts.append(feedback_alert)
            
            # Store alerts
            for alert in alerts:
                await self._store_drift_alert(alert)
            
            return {
                "status": "completed",
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "alerts": [asdict(alert) for alert in alerts],
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.severity == DriftSeverity.CRITICAL]),
                "recommendations": self._generate_recommendations(alerts)
            }
            
        except Exception as e:
            logger.error(f"Error in drift analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _detect_confidence_drift(
        self, 
        recent_metrics: List[PerformanceMetrics],
        baseline_metrics: List[PerformanceMetrics],
        tenant_id: Optional[str]
    ) -> Optional[DriftAlert]:
        """Detect drift in confidence scores."""
        try:
            # Extract confidence distributions
            recent_confidences = []
            baseline_confidences = []
            
            for metrics in recent_metrics:
                for conf_range, count in metrics.confidence_distribution.items():
                    # Convert confidence range to midpoint
                    if conf_range == "0.0-0.2":
                        midpoint = 0.1
                    elif conf_range == "0.2-0.4":
                        midpoint = 0.3
                    elif conf_range == "0.4-0.6":
                        midpoint = 0.5
                    elif conf_range == "0.6-0.8":
                        midpoint = 0.7
                    elif conf_range == "0.8-1.0":
                        midpoint = 0.9
                    else:
                        continue
                    
                    recent_confidences.extend([midpoint] * int(count))
            
            for metrics in baseline_metrics:
                for conf_range, count in metrics.confidence_distribution.items():
                    if conf_range == "0.0-0.2":
                        midpoint = 0.1
                    elif conf_range == "0.2-0.4":
                        midpoint = 0.3
                    elif conf_range == "0.4-0.6":
                        midpoint = 0.5
                    elif conf_range == "0.6-0.8":
                        midpoint = 0.7
                    elif conf_range == "0.8-1.0":
                        midpoint = 0.9
                    else:
                        continue
                    
                    baseline_confidences.extend([midpoint] * int(count))
            
            if len(recent_confidences) < 10 or len(baseline_confidences) < 10:
                return None
            
            # Statistical test
            drift_detected, p_value, test_name = self.statistical_detector.detect_distribution_drift(
                baseline_confidences, recent_confidences
            )
            
            if drift_detected:
                recent_mean = np.mean(recent_confidences)
                baseline_mean = np.mean(baseline_confidences)
                drift_score = abs(recent_mean - baseline_mean)
                
                # Determine severity
                if drift_score > 0.2:
                    severity = DriftSeverity.CRITICAL
                elif drift_score > 0.15:
                    severity = DriftSeverity.HIGH
                elif drift_score > 0.1:
                    severity = DriftSeverity.MEDIUM
                else:
                    severity = DriftSeverity.LOW
                
                return DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    drift_type=DriftType.PREDICTION_DRIFT,
                    severity=severity,
                    metric_name="confidence_distribution",
                    current_value=recent_mean,
                    baseline_value=baseline_mean,
                    drift_score=drift_score,
                    description=f"Confidence distribution has shifted significantly (p={p_value:.4f})",
                    timestamp=datetime.utcnow(),
                    tenant_id=tenant_id,
                    statistical_test=test_name,
                    p_value=p_value
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting confidence drift: {e}")
            return None
    
    async def _detect_decision_drift(
        self, 
        recent_metrics: List[PerformanceMetrics],
        baseline_metrics: List[PerformanceMetrics],
        tenant_id: Optional[str]
    ) -> Optional[DriftAlert]:
        """Detect drift in decision distributions."""
        try:
            # Calculate decision proportions
            recent_decisions = {"allow": 0, "review": 0, "block": 0}
            baseline_decisions = {"allow": 0, "review": 0, "block": 0}
            
            recent_total = 0
            for metrics in recent_metrics:
                for decision, count in metrics.decision_distribution.items():
                    recent_decisions[decision] += count
                    recent_total += count
            
            baseline_total = 0
            for metrics in baseline_metrics:
                for decision, count in metrics.decision_distribution.items():
                    baseline_decisions[decision] += count
                    baseline_total += count
            
            if recent_total < 100 or baseline_total < 100:
                return None
            
            # Calculate proportions
            recent_props = {k: v/recent_total for k, v in recent_decisions.items()}
            baseline_props = {k: v/baseline_total for k, v in baseline_decisions.items()}
            
            # Calculate drift score (sum of absolute differences)
            drift_score = sum(abs(recent_props[k] - baseline_props[k]) for k in recent_props)
            
            if drift_score > self.thresholds["decision_drift"]:
                # Determine severity
                if drift_score > 0.3:
                    severity = DriftSeverity.CRITICAL
                elif drift_score > 0.25:
                    severity = DriftSeverity.HIGH
                elif drift_score > 0.2:
                    severity = DriftSeverity.MEDIUM
                else:
                    severity = DriftSeverity.LOW
                
                return DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    drift_type=DriftType.PREDICTION_DRIFT,
                    severity=severity,
                    metric_name="decision_distribution",
                    current_value=drift_score,
                    baseline_value=0.0,
                    drift_score=drift_score,
                    description=f"Decision distribution has shifted by {drift_score:.3f}",
                    timestamp=datetime.utcnow(),
                    tenant_id=tenant_id
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting decision drift: {e}")
            return None
    
    async def _detect_performance_drift(
        self, 
        recent_metrics: List[PerformanceMetrics],
        baseline_metrics: List[PerformanceMetrics],
        tenant_id: Optional[str]
    ) -> Optional[DriftAlert]:
        """Detect drift in model performance."""
        try:
            recent_f1 = [m.f1_score for m in recent_metrics]
            baseline_f1 = [m.f1_score for m in baseline_metrics]
            
            if len(recent_f1) < 5 or len(baseline_f1) < 5:
                return None
            
            recent_mean = np.mean(recent_f1)
            baseline_mean = np.mean(baseline_f1)
            
            # Performance degradation
            performance_drop = baseline_mean - recent_mean
            
            if performance_drop > self.thresholds["performance_drift"]:
                # Determine severity based on performance drop
                if performance_drop > 0.15:
                    severity = DriftSeverity.CRITICAL
                elif performance_drop > 0.1:
                    severity = DriftSeverity.HIGH
                elif performance_drop > 0.075:
                    severity = DriftSeverity.MEDIUM
                else:
                    severity = DriftSeverity.LOW
                
                return DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    severity=severity,
                    metric_name="f1_score",
                    current_value=recent_mean,
                    baseline_value=baseline_mean,
                    drift_score=performance_drop,
                    description=f"F1 score has dropped by {performance_drop:.3f}",
                    timestamp=datetime.utcnow(),
                    tenant_id=tenant_id
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting performance drift: {e}")
            return None
    
    async def _detect_feedback_drift(self, tenant_id: Optional[str]) -> Optional[DriftAlert]:
        """Detect drift based on feedback patterns."""
        try:
            # Get recent feedback analytics
            recent_analytics = await self.feedback_manager.get_feedback_analytics(
                tenant_id=tenant_id, days=7
            )
            baseline_analytics = await self.feedback_manager.get_feedback_analytics(
                tenant_id=tenant_id, days=30
            )
            
            recent_rate = recent_analytics.correction_rate
            baseline_rate = baseline_analytics.correction_rate
            
            if recent_analytics.total_feedback_count < 10:
                return None
            
            rate_increase = recent_rate - baseline_rate
            
            if rate_increase > self.thresholds["feedback_rate_drift"]:
                # Determine severity
                if rate_increase > 0.4:
                    severity = DriftSeverity.CRITICAL
                elif rate_increase > 0.3:
                    severity = DriftSeverity.HIGH
                elif rate_increase > 0.25:
                    severity = DriftSeverity.MEDIUM
                else:
                    severity = DriftSeverity.LOW
                
                return DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    drift_type=DriftType.CONCEPT_DRIFT,
                    severity=severity,
                    metric_name="feedback_correction_rate",
                    current_value=recent_rate,
                    baseline_value=baseline_rate,
                    drift_score=rate_increase,
                    description=f"Feedback correction rate increased by {rate_increase:.3f}",
                    timestamp=datetime.utcnow(),
                    tenant_id=tenant_id
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting feedback drift: {e}")
            return None
    
    async def _store_drift_alert(self, alert: DriftAlert) -> bool:
        """Store drift alert in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO drift_alerts 
                    (alert_id, drift_type, severity, metric_name, current_value,
                     baseline_value, drift_score, description, timestamp,
                     tenant_id, statistical_test, p_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.drift_type.value,
                    alert.severity.value,
                    alert.metric_name,
                    alert.current_value,
                    alert.baseline_value,
                    alert.drift_score,
                    alert.description,
                    alert.timestamp.isoformat(),
                    alert.tenant_id,
                    alert.statistical_test,
                    alert.p_value
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store drift alert: {e}")
            return False
    
    async def get_drift_alerts(
        self, 
        hours: int = 24,
        tenant_id: Optional[str] = None
    ) -> List[DriftAlert]:
        """Get recent drift alerts."""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            query = "SELECT * FROM drift_alerts WHERE timestamp >= ?"
            params = [start_time.isoformat()]
            
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            
            query += " ORDER BY timestamp DESC"
            
            alerts = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    alert = DriftAlert(
                        alert_id=row['alert_id'],
                        drift_type=DriftType(row['drift_type']),
                        severity=DriftSeverity(row['severity']),
                        metric_name=row['metric_name'],
                        current_value=row['current_value'],
                        baseline_value=row['baseline_value'],
                        drift_score=row['drift_score'],
                        description=row['description'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        tenant_id=row['tenant_id'],
                        statistical_test=row['statistical_test'],
                        p_value=row['p_value']
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get drift alerts: {e}")
            return []
    
    def _generate_recommendations(self, alerts: List[DriftAlert]) -> List[str]:
        """Generate recommendations based on drift alerts."""
        recommendations = []
        
        critical_alerts = [a for a in alerts if a.severity == DriftSeverity.CRITICAL]
        high_alerts = [a for a in alerts if a.severity == DriftSeverity.HIGH]
        
        if critical_alerts:
            recommendations.append("Critical drift detected - immediate model retraining recommended")
            
            performance_alerts = [a for a in critical_alerts if a.drift_type == DriftType.PERFORMANCE_DRIFT]
            if performance_alerts:
                recommendations.append("Significant performance degradation - review training data quality")
        
        if high_alerts:
            concept_alerts = [a for a in high_alerts if a.drift_type == DriftType.CONCEPT_DRIFT]
            if concept_alerts:
                recommendations.append("Concept drift detected - update training data with recent examples")
            
            prediction_alerts = [a for a in high_alerts if a.drift_type == DriftType.PREDICTION_DRIFT]
            if prediction_alerts:
                recommendations.append("Prediction patterns have changed - consider model recalibration")
        
        if not recommendations:
            recommendations.append("No significant drift detected - continue monitoring")
        
        return recommendations


class ModelRetrainingManager:
    """Manage automated model retraining."""
    
    def __init__(
        self, 
        drift_detector: DriftDetectionEngine,
        storage_path: str = "logs/retraining"
    ):
        """
        Initialize retraining manager.
        
        Args:
            drift_detector: Drift detection engine
            storage_path: Path for retraining job storage
        """
        self.drift_detector = drift_detector
        self.storage_path = Path(storage_path)
        self.db_path = self.storage_path / "retraining_jobs.db"
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Retraining thresholds
        self.retraining_thresholds = {
            "critical_alerts": 1,  # Trigger on any critical alert
            "high_alerts": 3,      # Trigger on 3+ high alerts
            "feedback_count": 100,  # Trigger after 100 feedback items
            "performance_drop": 0.1 # Trigger on 10% performance drop
        }
    
    def _init_database(self):
        """Initialize retraining jobs database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS retraining_jobs (
                        job_id TEXT PRIMARY KEY,
                        trigger_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        model_version TEXT NOT NULL,
                        training_data_size INTEGER,
                        performance_metrics TEXT,
                        error_message TEXT,
                        tenant_id TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_retraining_status 
                    ON retraining_jobs(status)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_retraining_timestamp 
                    ON retraining_jobs(created_at)
                """)
                
                conn.commit()
                logger.info("Retraining jobs database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize retraining database: {e}")
            raise
    
    async def check_retraining_triggers(self, tenant_id: Optional[str] = None) -> Optional[str]:
        """
        Check if retraining should be triggered.
        
        Args:
            tenant_id: Tenant ID to check
            
        Returns:
            Job ID if retraining triggered, None otherwise
        """
        try:
            # Run drift analysis
            drift_analysis = await self.drift_detector.run_drift_analysis(tenant_id)
            
            if drift_analysis.get("status") != "completed":
                return None
            
            alerts = drift_analysis.get("alerts", [])
            critical_count = len([a for a in alerts if a.get("severity") == "critical"])
            high_count = len([a for a in alerts if a.get("severity") == "high"])
            
            trigger_reason = None
            
            # Check critical alerts
            if critical_count >= self.retraining_thresholds["critical_alerts"]:
                trigger_reason = RetrainingTrigger.DRIFT_DETECTED
            
            # Check high alerts
            elif high_count >= self.retraining_thresholds["high_alerts"]:
                trigger_reason = RetrainingTrigger.PERFORMANCE_DEGRADATION
            
            # Check feedback threshold
            else:
                feedback_analytics = await self.drift_detector.feedback_manager.get_feedback_analytics(
                    tenant_id=tenant_id, days=7
                )
                
                if feedback_analytics.total_feedback_count >= self.retraining_thresholds["feedback_count"]:
                    trigger_reason = RetrainingTrigger.FEEDBACK_THRESHOLD
            
            if trigger_reason:
                return await self._create_retraining_job(trigger_reason, tenant_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking retraining triggers: {e}")
            return None
    
    async def _create_retraining_job(
        self, 
        trigger: RetrainingTrigger,
        tenant_id: Optional[str] = None
    ) -> str:
        """Create a new retraining job."""
        try:
            job_id = str(uuid.uuid4())
            model_version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            job = RetrainingJob(
                job_id=job_id,
                trigger=trigger,
                status="pending",
                created_at=datetime.utcnow(),
                started_at=None,
                completed_at=None,
                model_version=model_version,
                training_data_size=0,
                performance_metrics=None,
                tenant_id=tenant_id
            )
            
            # Store job
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO retraining_jobs 
                    (job_id, trigger_type, status, created_at, model_version, tenant_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    job.job_id,
                    job.trigger.value,
                    job.status,
                    job.created_at.isoformat(),
                    job.model_version,
                    job.tenant_id
                ))
                conn.commit()
            
            logger.info(f"Created retraining job {job_id} triggered by {trigger.value}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create retraining job: {e}")
            raise
    
    async def get_retraining_status(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get retraining job status."""
        try:
            query = "SELECT * FROM retraining_jobs"
            params = []
            
            if tenant_id:
                query += " WHERE tenant_id = ?"
                params.append(tenant_id)
            
            query += " ORDER BY created_at DESC LIMIT 10"
            
            jobs = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    job_data = {
                        "job_id": row['job_id'],
                        "trigger": row['trigger_type'],
                        "status": row['status'],
                        "created_at": row['created_at'],
                        "started_at": row['started_at'],
                        "completed_at": row['completed_at'],
                        "model_version": row['model_version'],
                        "training_data_size": row['training_data_size'],
                        "error_message": row['error_message']
                    }
                    jobs.append(job_data)
            
            return {
                "recent_jobs": jobs,
                "total_jobs": len(jobs),
                "pending_jobs": len([j for j in jobs if j["status"] == "pending"]),
                "running_jobs": len([j for j in jobs if j["status"] == "running"]),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting retraining status: {e}")
            return {"error": str(e)}