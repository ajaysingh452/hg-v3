"""Tests for model inference and aggregation logic."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import asyncio

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.aggregator import EnsembleAggregator, TemperatureScaler, EnsembleWeightOptimizer
from model.calibration import ModelCalibrator, CalibrationResult, UncertaintyEstimate
from model.multi_head_classifier import MultiHeadClassificationSystem, MultiHeadOutput, ClassificationHead
from core.models import (
    ProcessedText, LPEResult, ClassifierResult, ContextResult, AggregatedResult,
    ProblemSpan, DecisionType, SeverityLevel, AbuseCategory, LanguageDetection
)
from core.aggregator import SpanConsolidator
from core.decision_logic import DecisionContext


class TestTemperatureScaler:
    """Test temperature scaling for probability calibration."""
    
    def test_temperature_scaler_initialization(self):
        """Test temperature scaler initialization."""
        scaler = TemperatureScaler()
        
        assert not scaler.is_fitted
        assert isinstance(scaler.temperatures, dict)
        assert len(scaler.temperatures) == 0
    
    def test_temperature_scaling_fit(self):
        """Test fitting temperature scaling parameters."""
        scaler = TemperatureScaler()
        
        # Create mock logits and labels
        logits = {
            'component1': np.array([[2.0, -1.0], [1.5, -0.5], [-1.0, 2.0]]),
            'component2': np.array([[1.0, 0.0], [0.5, 0.5], [-0.5, 1.5]])
        }
        labels = np.array([0, 0, 1])
        
        scaler.fit(logits, labels)
        
        assert scaler.is_fitted
        assert 'component1' in scaler.temperatures
        assert 'component2' in scaler.temperatures
        assert scaler.temperatures['component1'] > 0
        assert scaler.temperatures['component2'] > 0
    
    def test_temperature_scaling_transform(self):
        """Test temperature scaling transformation."""
        scaler = TemperatureScaler()
        scaler.is_fitted = True
        scaler.temperatures = {'component1': 2.0, 'component2': 1.5}
        
        logits = {
            'component1': np.array([[2.0, -1.0]]),
            'component2': np.array([[1.0, 0.0]])
        }
        
        calibrated = scaler.transform(logits)
        
        assert 'component1' in calibrated
        assert 'component2' in calibrated
        assert calibrated['component1'].shape == logits['component1'].shape
        assert calibrated['component2'].shape == logits['component2'].shape
        
        # Check that probabilities are properly scaled
        prob1 = np.exp(calibrated['component1'][0]) / np.sum(np.exp(calibrated['component1'][0]))
        assert 0.0 <= prob1[0] <= 1.0
        assert 0.0 <= prob1[1] <= 1.0
        assert abs(np.sum(prob1) - 1.0) < 1e-6


class TestEnsembleWeightOptimizer:
    """Test ensemble weight optimization."""
    
    def test_weight_optimizer_initialization(self):
        """Test weight optimizer initialization."""
        optimizer = EnsembleWeightOptimizer(cv_folds=3)
        
        assert not optimizer.is_fitted
        assert optimizer.cv_folds == 3
        assert optimizer.optimization_method == "nelder_mead"
    
    def test_weight_optimization(self):
        """Test weight optimization process."""
        optimizer = EnsembleWeightOptimizer(cv_folds=3)
        
        # Create mock component predictions
        n_samples = 100
        component_predictions = {
            'lpe': np.random.rand(n_samples) * 0.8 + 0.1,  # Biased towards lower values
            'classifier': np.random.rand(n_samples) * 0.6 + 0.2,  # Different distribution
            'intent': np.random.rand(n_samples) * 0.4 + 0.1   # More conservative
        }
        
        # Create labels correlated with predictions
        labels = (component_predictions['lpe'] * 0.4 + 
                 component_predictions['classifier'] * 0.5 + 
                 component_predictions['intent'] * 0.1 > 0.5).astype(int)
        
        weights = optimizer.optimize_weights(component_predictions, labels)
        
        assert optimizer.is_fitted
        assert len(weights) == 3
        assert 'lpe' in weights
        assert 'classifier' in weights
        assert 'intent' in weights
        
        # Weights should be non-negative and sum to 1
        assert all(w >= 0 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01
    
    def test_cross_validation_scoring(self):
        """Test cross-validation scoring function."""
        optimizer = EnsembleWeightOptimizer()
        
        # Simple test data
        predictions = np.array([0.1, 0.3, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1])
        
        score = optimizer._calculate_cv_score(predictions, labels)
        
        assert isinstance(score, float)
        assert score >= 0.0  # Should be a valid score


class TestSpanConsolidator:
    """Test span consolidation functionality."""
    
    def test_span_consolidator_initialization(self):
        """Test span consolidator initialization."""
        consolidator = SpanConsolidator(overlap_threshold=0.5, max_spans=5)
        
        assert consolidator.overlap_threshold == 0.5
        assert consolidator.max_spans == 5
        assert consolidator.ranking_method == "hybrid"
    
    def test_span_consolidation_no_overlap(self):
        """Test consolidation of non-overlapping spans."""
        consolidator = SpanConsolidator()
        
        spans1 = [
            ProblemSpan("bad", 0, 3, "profanity", 0.8, "lpe"),
            ProblemSpan("hate", 10, 14, "harassment", 0.9, "lpe")
        ]
        spans2 = [
            ProblemSpan("stupid", 20, 26, "insult", 0.7, "classifier")
        ]
        
        consolidated = consolidator.consolidate_spans([spans1, spans2])
        
        # Should keep all non-overlapping spans
        assert len(consolidated) == 3
        
        # Should be sorted by confidence (descending)
        confidences = [span.confidence for span in consolidated]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_span_consolidation_with_overlap(self):
        """Test consolidation of overlapping spans."""
        consolidator = SpanConsolidator(overlap_threshold=0.3)
        
        spans1 = [
            ProblemSpan("bad word", 0, 8, "profanity", 0.9, "lpe")
        ]
        spans2 = [
            ProblemSpan("bad", 0, 3, "profanity", 0.7, "classifier")  # Overlaps with first
        ]
        
        consolidated = consolidator.consolidate_spans([spans1, spans2])
        
        # Should keep only the span with higher confidence
        assert len(consolidated) == 1
        assert consolidated[0].confidence == 0.9
        assert consolidated[0].text == "bad word"
    
    def test_span_ranking_methods(self):
        """Test different span ranking methods."""
        consolidator = SpanConsolidator(ranking_method="confidence")
        
        spans = [
            ProblemSpan("word1", 0, 5, "category1", 0.7, "source1"),
            ProblemSpan("word2", 10, 15, "category2", 0.9, "source2"),
            ProblemSpan("word3", 20, 25, "category3", 0.8, "source3")
        ]
        
        ranked = consolidator._rank_spans(spans)
        
        # Should be sorted by confidence (descending)
        confidences = [span.confidence for span in ranked]
        assert confidences == [0.9, 0.8, 0.7]
    
    def test_overlap_calculation(self):
        """Test overlap calculation between spans."""
        consolidator = SpanConsolidator()
        
        span1 = ProblemSpan("hello", 0, 5, "test", 0.8, "source")
        span2 = ProblemSpan("world", 6, 11, "test", 0.8, "source")  # No overlap
        span3 = ProblemSpan("hello world", 0, 11, "test", 0.8, "source")  # Full overlap
        span4 = ProblemSpan("lo wo", 3, 8, "test", 0.8, "source")  # Partial overlap
        
        # No overlap
        assert consolidator._calculate_overlap(span1, span2) == 0.0
        
        # Full overlap
        overlap = consolidator._calculate_overlap(span1, span3)
        assert overlap > 0.8  # span1 is fully contained in span3
        
        # Partial overlap
        overlap = consolidator._calculate_overlap(span1, span4)
        assert 0.0 < overlap < 1.0


class TestMultiHeadClassificationSystem:
    """Test multi-head classification system."""
    
    @pytest.fixture
    def multi_head_config(self):
        """Configuration for multi-head system."""
        return {
            "thresholds": {
                "abuse_categories": 0.3,
                "corporate_decision": 0.5,
                "severity_levels": 0.4
            }
        }
    
    def test_multi_head_initialization(self, multi_head_config):
        """Test multi-head system initialization."""
        system = MultiHeadClassificationSystem(multi_head_config)
        
        assert "abuse_categories" in system.heads
        assert "corporate_decision" in system.heads
        assert "severity_levels" in system.heads
        
        # Check head configurations
        abuse_head = system.heads["abuse_categories"]
        assert isinstance(abuse_head, ClassificationHead)
        assert abuse_head.activation == "sigmoid"
        assert len(abuse_head.class_names) == len([cat.value for cat in AbuseCategory])
    
    def test_logits_processing(self, multi_head_config):
        """Test processing of model logits."""
        system = MultiHeadClassificationSystem(multi_head_config)
        
        # Create sample logits
        logits = {
            "category_logits": np.array([0.1, -0.5, 0.8, -0.2, 0.3, -0.1, 0.6, -0.3]),
            "corporate_logits": np.array([0.2, -0.1, 0.4]),
            "severity_logits": np.array([-0.2, 0.1, 0.3, 0.5])
        }
        
        output = system.process_logits(logits)
        
        assert isinstance(output, MultiHeadOutput)
        assert isinstance(output.abuse_categories, dict)
        assert isinstance(output.corporate_decision, dict)
        assert isinstance(output.severity_levels, dict)
        
        # Check probability ranges
        for prob in output.abuse_categories.values():
            assert 0.0 <= prob <= 1.0
        
        # Corporate decision probabilities should sum to 1
        corp_sum = sum(output.corporate_decision.values())
        assert abs(corp_sum - 1.0) < 1e-6
    
    def test_threshold_application(self, multi_head_config):
        """Test threshold application to predictions."""
        system = MultiHeadClassificationSystem(multi_head_config)
        
        probabilities = {
            "harassment": 0.4,  # Above threshold (0.3)
            "profanity": 0.2,   # Below threshold
            "hate": 0.8         # Above threshold
        }
        
        predictions = system._apply_thresholds("abuse_categories", probabilities)
        
        assert predictions["harassment"] == 1  # Above threshold
        assert predictions["profanity"] == 0   # Below threshold
        assert predictions["hate"] == 1        # Above threshold
    
    def test_confidence_calculation(self, multi_head_config):
        """Test confidence score calculation."""
        system = MultiHeadClassificationSystem(multi_head_config)
        
        probabilities = {"class1": 0.8, "class2": 0.2, "class3": 0.1}
        
        confidence = system._calculate_confidence(probabilities)
        
        assert 0.0 <= confidence <= 1.0
        # Confidence should be high when one class dominates
        assert confidence > 0.5


class TestModelCalibrator:
    """Test model calibration functionality."""
    
    @pytest.fixture
    def calibrator_config(self):
        """Configuration for calibrator."""
        return {
            "confidence_threshold": 0.8,
            "uncertainty_threshold": 0.3
        }
    
    def test_calibrator_initialization(self, calibrator_config):
        """Test calibrator initialization."""
        calibrator = ModelCalibrator(calibrator_config)
        
        assert calibrator.confidence_threshold == 0.8
        assert calibrator.uncertainty_threshold == 0.3
        assert isinstance(calibrator.temperatures, dict)
        assert isinstance(calibrator.language_factors, dict)
    
    def test_probability_calibration(self, calibrator_config):
        """Test probability calibration."""
        calibrator = ModelCalibrator(calibrator_config)
        
        probabilities = {
            "harassment": 0.3,
            "profanity": 0.1,
            "hate": 0.05
        }
        
        result = calibrator.calibrate_probabilities(
            probabilities, "abuse_categories", ["en"]
        )
        
        assert isinstance(result, CalibrationResult)
        assert result.temperature > 0
        assert isinstance(result.calibrated_probabilities, dict)
        assert 0.0 <= result.confidence_score <= 1.0
        assert 0.0 <= result.reliability_score <= 1.0
        assert result.calibration_error >= 0.0
    
    def test_uncertainty_estimation(self, calibrator_config):
        """Test uncertainty estimation."""
        calibrator = ModelCalibrator(calibrator_config)
        
        probabilities = {"class1": 0.7, "class2": 0.2, "class3": 0.1}
        
        uncertainty = calibrator.estimate_uncertainty(probabilities)
        
        assert isinstance(uncertainty, UncertaintyEstimate)
        assert 0.0 <= uncertainty.epistemic_uncertainty <= 1.0
        assert 0.0 <= uncertainty.aleatoric_uncertainty <= 1.0
        assert 0.0 <= uncertainty.total_uncertainty <= 1.0
        assert len(uncertainty.confidence_interval) == 2
        assert uncertainty.confidence_interval[0] <= uncertainty.confidence_interval[1]
    
    def test_temperature_scaling_application(self, calibrator_config):
        """Test temperature scaling application."""
        calibrator = ModelCalibrator(calibrator_config)
        
        probabilities = {"class1": 0.8, "class2": 0.2}
        temperature = 2.0
        
        calibrated = calibrator._apply_temperature_scaling(probabilities, temperature)
        
        assert isinstance(calibrated, dict)
        assert len(calibrated) == len(probabilities)
        
        # Check that probabilities are still valid
        for prob in calibrated.values():
            assert 0.0 <= prob <= 1.0
        
        # Sum should still be approximately 1 for normalized probabilities
        prob_sum = sum(calibrated.values())
        assert abs(prob_sum - 1.0) < 0.1  # Allow some tolerance for non-normalized inputs


class TestEnsembleAggregator:
    """Test ensemble aggregation functionality."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        config_manager = Mock()
        config_manager.get_ensemble_config.return_value = {
            'ensemble': {
                'weights': {'lpe': 0.4, 'classifier': 0.5, 'intent': 0.1},
                'thresholds': {
                    'confidence_minimum': 0.6,
                    'review_threshold': 0.7,
                    'block_threshold': 0.85
                },
                'span_consolidation': {
                    'overlap_threshold': 0.3,
                    'max_spans': 10,
                    'ranking_method': 'hybrid'
                }
            }
        }
        return config_manager
    
    @pytest.fixture
    def sample_component_results(self):
        """Create sample component results."""
        lpe_result = LPEResult(
            matched_spans=[
                ProblemSpan("bad", 0, 3, "profanity", 0.9, "lpe_rule_1")
            ],
            categories=["profanity"],
            confidence_scores={"profanity": 0.8},
            rule_traces=["Matched profanity pattern"]
        )
        
        classifier_result = ClassifierResult(
            category_probabilities={"profanity": 0.7, "harassment": 0.3},
            corporate_decision_prob={"allow": 0.2, "review": 0.3, "block": 0.5},
            severity_scores={"medium": 0.6, "high": 0.4},
            attention_spans=[
                ProblemSpan("bad", 0, 3, "profanity", 0.8, "classifier")
            ]
        )
        
        context_result = ContextResult(
            context_modifiers={"profanity": 1.0},
            safe_context_detected={"profanity": False},
            recommended_action=DecisionType.BLOCK
        )
        
        return lpe_result, classifier_result, context_result
    
    def test_aggregator_initialization(self, mock_config_manager):
        """Test aggregator initialization."""
        aggregator = EnsembleAggregator(mock_config_manager)
        
        assert aggregator.weights == {'lpe': 0.4, 'classifier': 0.5, 'intent': 0.1}
        assert aggregator.confidence_minimum == 0.6
        assert aggregator.review_threshold == 0.7
        assert aggregator.block_threshold == 0.85
    
    def test_component_score_extraction(self, mock_config_manager, sample_component_results):
        """Test extraction of scores from component results."""
        aggregator = EnsembleAggregator(mock_config_manager)
        lpe_result, classifier_result, context_result = sample_component_results
        
        component_scores = aggregator._extract_component_scores(
            lpe_result, classifier_result, context_result
        )
        
        assert 'lpe' in component_scores
        assert 'classifier' in component_scores
        assert 'intent' in component_scores
        assert component_scores['lpe']['profanity'] == 0.8
        assert component_scores['classifier']['profanity'] == 0.7
    
    def test_score_combination(self, mock_config_manager, sample_component_results):
        """Test weighted score combination."""
        aggregator = EnsembleAggregator(mock_config_manager)
        lpe_result, classifier_result, context_result = sample_component_results
        
        component_scores = aggregator._extract_component_scores(
            lpe_result, classifier_result, context_result
        )
        
        final_scores = aggregator._combine_scores(component_scores)
        
        assert 'profanity' in final_scores
        # Should be weighted combination: (0.8 * 0.4 + 0.7 * 0.5 + score * 0.1)
        expected_base = (0.8 * 0.4 + 0.7 * 0.5) / (0.4 + 0.5)  # Simplified calculation
        assert 0.6 <= final_scores['profanity'] <= 0.9  # Should be in reasonable range
    
    def test_decision_making(self, mock_config_manager):
        """Test decision making logic."""
        aggregator = EnsembleAggregator(mock_config_manager)
        
        # Test block decision
        high_scores = {"profanity": 0.9}
        decision = aggregator._make_decision(high_scores)
        assert decision == DecisionType.BLOCK
        
        # Test review decision
        medium_scores = {"profanity": 0.75}
        decision = aggregator._make_decision(medium_scores)
        assert decision == DecisionType.REVIEW
        
        # Test allow decision
        low_scores = {"profanity": 0.5}
        decision = aggregator._make_decision(low_scores)
        assert decision == DecisionType.ALLOW
    
    def test_full_aggregation_process(self, mock_config_manager, sample_component_results):
        """Test complete aggregation process."""
        aggregator = EnsembleAggregator(mock_config_manager)
        lpe_result, classifier_result, context_result = sample_component_results
        
        result = aggregator.aggregate(
            lpe_result, classifier_result, context_result,
            original_text="This is bad content",
            language_confidence=0.9,
            primary_language="en"
        )
        
        assert isinstance(result, AggregatedResult)
        assert result.final_decision in [DecisionType.ALLOW, DecisionType.REVIEW, DecisionType.BLOCK]
        assert 0 <= result.confidence_score <= 1
        assert result.severity_level in [SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        assert isinstance(result.explanation_traces, list)
        assert isinstance(result.consolidated_spans, list)
    
    def test_span_consolidation_integration(self, mock_config_manager, sample_component_results):
        """Test span consolidation in aggregation."""
        aggregator = EnsembleAggregator(mock_config_manager)
        lpe_result, classifier_result, context_result = sample_component_results
        
        result = aggregator.aggregate(
            lpe_result, classifier_result, context_result
        )
        
        # Should consolidate spans from different components
        assert len(result.consolidated_spans) >= 1
        
        # Spans should be properly formatted
        for span in result.consolidated_spans:
            assert isinstance(span, ProblemSpan)
            assert 0 <= span.confidence <= 1
            assert span.start >= 0
            assert span.end > span.start


if __name__ == "__main__":
    pytest.main([__file__])