"""Tests for model performance monitoring."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from harmony_guard.model.monitoring import (
    ModelPerformanceMonitor, 
    PredictionMetrics, 
    PerformanceWindow, 
    DriftAlert
)
from harmony_guard.model.classifier import TransformerClassifier
from harmony_guard.core.models import ProcessedText


class TestModelPerformanceMonitor:
    """Test model performance monitoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "window_size": 10,
            "time_window": 60,  # 1 minute
            "drift_threshold": 0.1,
            "confidence_threshold": 0.15
        }
        self.monitor = ModelPerformanceMonitor(self.config)
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.window_size == 10
        assert self.monitor.time_window == 60
        assert self.monitor.drift_threshold == 0.1
        assert self.monitor.confidence_threshold == 0.15
        assert not self.monitor.baseline_established
        assert len(self.monitor.prediction_history) == 0
    
    def test_record_prediction(self):
        """Test recording predictions."""
        predictions = {
            "category_probabilities": {"harassment": 0.3, "profanity": 0.1},
            "corporate_decision_prob": {"allow": 0.7, "review": 0.2, "block": 0.1}
        }
        
        self.monitor.record_prediction(
            predictions=predictions,
            confidence_score=0.8,
            processing_time=25.5,
            language_codes=["en", "hi"],
            input_length=100
        )
        
        assert len(self.monitor.prediction_history) == 1
        assert len(self.monitor.current_window_predictions) == 1
        assert len(self.monitor.confidence_history) == 1
        assert len(self.monitor.processing_time_history) == 1
        
        # Check recorded metrics
        recorded = self.monitor.prediction_history[0]
        assert recorded.confidence_score == 0.8
        assert recorded.processing_time == 25.5
        assert recorded.language_codes == ["en", "hi"]
        assert recorded.input_length == 100
    
    def test_extract_prediction_distribution(self):
        """Test prediction distribution extraction."""
        predictions = {
            "category_probabilities": {"harassment": 0.3, "profanity": 0.1},
            "corporate_decision_prob": {"allow": 0.7, "review": 0.2, "block": 0.1}
        }
        
        dist = self.monitor._extract_prediction_distribution(predictions)
        assert "harassment" in dist
        assert "profanity" in dist
        assert dist["harassment"] == 0.3
        assert dist["profanity"] == 0.1
    
    def test_window_processing(self):
        """Test window processing logic."""
        # Record enough predictions to trigger window processing
        for i in range(12):  # More than window_size
            predictions = {
                "category_probabilities": {"harassment": 0.1 + i * 0.05},
                "corporate_decision_prob": {"allow": 0.8, "review": 0.1, "block": 0.1}
            }
            
            self.monitor.record_prediction(
                predictions=predictions,
                confidence_score=0.7 + i * 0.01,
                processing_time=20.0 + i,
                language_codes=["en"],
                input_length=50 + i * 5
            )
        
        # Should have processed at least one window
        assert len(self.monitor.performance_windows) >= 1
        
        # Check window metrics
        window = self.monitor.performance_windows[0]
        assert window.total_predictions > 0
        assert window.avg_confidence > 0
        assert window.avg_processing_time > 0
    
    def test_baseline_establishment(self):
        """Test baseline establishment."""
        # Create 5 windows to establish baseline
        for window_idx in range(5):
            for pred_idx in range(10):
                predictions = {
                    "category_probabilities": {"harassment": 0.2},
                    "corporate_decision_prob": {"allow": 0.8, "review": 0.1, "block": 0.1}
                }
                
                self.monitor.record_prediction(
                    predictions=predictions,
                    confidence_score=0.8,
                    processing_time=25.0,
                    language_codes=["en"],
                    input_length=100
                )
            
            # Force window processing
            self.monitor._process_current_window()
        
        # Baseline should be established
        assert self.monitor.baseline_established
        assert "avg_confidence" in self.monitor.baseline_metrics
        assert "avg_processing_time" in self.monitor.baseline_metrics
        assert "prediction_distribution" in self.monitor.baseline_metrics
    
    def test_confidence_drift_detection(self):
        """Test confidence drift detection."""
        # Establish baseline first
        self._establish_test_baseline()
        
        # Create a window with significantly different confidence
        window = PerformanceWindow(
            start_time=time.time() - 60,
            end_time=time.time(),
            total_predictions=10,
            avg_confidence=0.4,  # Much lower than baseline (0.8)
            avg_processing_time=25.0,
            prediction_distribution={"harassment": 0.2},
            language_distribution={"en": 10}
        )
        
        alert = self.monitor._detect_confidence_drift(window)
        
        assert alert is not None
        assert alert.drift_type == "confidence"
        assert alert.severity in ["low", "medium", "high", "critical"]
        assert alert.current_value == 0.4
        assert alert.baseline_value == 0.8
    
    def test_performance_drift_detection(self):
        """Test performance drift detection."""
        # Establish baseline first
        self._establish_test_baseline()
        
        # Create a window with significantly different processing time
        window = PerformanceWindow(
            start_time=time.time() - 60,
            end_time=time.time(),
            total_predictions=10,
            avg_confidence=0.8,
            avg_processing_time=50.0,  # Much higher than baseline (25.0)
            prediction_distribution={"harassment": 0.2},
            language_distribution={"en": 10}
        )
        
        alert = self.monitor._detect_performance_drift(window)
        
        assert alert is not None
        assert alert.drift_type == "performance"
        assert alert.current_value == 50.0
        assert alert.baseline_value == 25.0
    
    def test_prediction_drift_detection(self):
        """Test prediction distribution drift detection."""
        # Establish baseline first
        self._establish_test_baseline()
        
        # Create a window with significantly different prediction distribution
        window = PerformanceWindow(
            start_time=time.time() - 60,
            end_time=time.time(),
            total_predictions=10,
            avg_confidence=0.8,
            avg_processing_time=25.0,
            prediction_distribution={"harassment": 0.8},  # Much higher than baseline
            language_distribution={"en": 10}
        )
        
        alerts = self.monitor._detect_prediction_drift(window)
        
        assert len(alerts) > 0
        alert = alerts[0]
        assert alert.drift_type == "prediction"
        assert "harassment" in alert.metric_name
    
    def test_statistical_tests(self):
        """Test statistical drift tests."""
        # Create enough windows for statistical testing
        for i in range(20):
            window = PerformanceWindow(
                start_time=time.time() - (20 - i) * 60,
                end_time=time.time() - (19 - i) * 60,
                total_predictions=10,
                avg_confidence=0.8 + (0.1 if i > 10 else 0),  # Shift in second half
                avg_processing_time=25.0 + (5.0 if i > 10 else 0),  # Shift in second half
                prediction_distribution={"harassment": 0.2},
                language_distribution={"en": 10}
            )
            self.monitor.performance_windows.append(window)
        
        results = self.monitor.run_statistical_tests()
        
        assert "confidence_test" in results
        assert "processing_time_test" in results
        assert "distribution_test" in results
        
        # Check test structure
        confidence_test = results["confidence_test"]
        assert "statistic" in confidence_test
        assert "p_value" in confidence_test
        assert "significant" in confidence_test
        assert "early_mean" in confidence_test
        assert "recent_mean" in confidence_test
    
    def test_get_current_metrics(self):
        """Test getting current metrics."""
        # Record some predictions
        for i in range(5):
            predictions = {
                "category_probabilities": {"harassment": 0.2},
                "corporate_decision_prob": {"allow": 0.8, "review": 0.1, "block": 0.1}
            }
            
            self.monitor.record_prediction(
                predictions=predictions,
                confidence_score=0.8,
                processing_time=25.0,
                language_codes=["en"],
                input_length=100
            )
        
        # Force window processing
        self.monitor._process_current_window()
        
        metrics = self.monitor.get_current_metrics()
        
        assert "current_window" in metrics
        assert "recent_alerts" in metrics
        assert "baseline_established" in metrics
        
        current_window = metrics["current_window"]
        assert "total_predictions" in current_window
        assert "avg_confidence" in current_window
        assert "avg_processing_time" in current_window
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Create some windows
        for i in range(5):
            window = PerformanceWindow(
                start_time=time.time() - (5 - i) * 60,
                end_time=time.time() - (4 - i) * 60,
                total_predictions=10,
                avg_confidence=0.8,
                avg_processing_time=25.0,
                prediction_distribution={"harassment": 0.2},
                language_distribution={"en": 10}
            )
            self.monitor.performance_windows.append(window)
        
        summary = self.monitor.get_performance_summary(windows=3)
        
        assert "windows_analyzed" in summary
        assert "total_predictions" in summary
        assert "avg_confidence" in summary
        assert "confidence_std" in summary
        assert "avg_processing_time" in summary
        assert "processing_time_std" in summary
        assert "time_range" in summary
        
        assert summary["windows_analyzed"] == 3
        assert summary["total_predictions"] == 30  # 3 windows * 10 predictions
    
    def _establish_test_baseline(self):
        """Helper to establish baseline for testing."""
        self.monitor.baseline_established = True
        self.monitor.baseline_metrics = {
            "avg_confidence": 0.8,
            "avg_processing_time": 25.0,
            "prediction_distribution": {
                "harassment": {"mean": 0.2, "std": 0.05}
            },
            "confidence_std": 0.05,
            "processing_time_std": 2.0
        }


class TestClassifierMonitoringIntegration:
    """Test integration of monitoring with classifier."""
    
    @pytest.fixture
    def classifier_config(self):
        """Classifier configuration with monitoring enabled."""
        return {
            "model_name": "test-model",
            "monitoring": {
                "window_size": 5,
                "time_window": 30,
                "drift_threshold": 0.1,
                "confidence_threshold": 0.15
            }
        }
    
    @pytest.mark.asyncio
    async def test_classifier_with_monitoring(self, classifier_config):
        """Test classifier with monitoring integration."""
        classifier = TransformerClassifier(classifier_config)
        await classifier.initialize()
        
        # Verify monitoring is enabled
        assert classifier.monitor is not None
        assert isinstance(classifier.monitor, ModelPerformanceMonitor)
        
        # Create test input
        processed_text = ProcessedText(
            original_text="test text",
            normalized_text="test text",
            detected_languages=["en"],
            tokens=["test", "text"],
            transliterations={},
            obfuscation_map={}
        )
        
        # Make prediction (should record metrics)
        result = await classifier.predict(processed_text)
        
        # Verify prediction was recorded
        assert len(classifier.monitor.prediction_history) == 1
        
        # Test monitoring methods
        metrics = classifier.get_performance_metrics()
        assert isinstance(metrics, dict)
        
        alerts = classifier.get_drift_alerts()
        assert isinstance(alerts, list)
        
        summary = classifier.get_performance_summary()
        assert isinstance(summary, dict)
        
        drift_tests = classifier.run_drift_tests()
        assert isinstance(drift_tests, dict)