"""Tests for ensemble aggregator functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.aggregator import (
    EnsembleAggregator, TemperatureScaler, EnsembleWeightOptimizer, SpanConsolidator
)
from core.models import (
    LPEResult, ClassifierResult, ContextResult, AggregatedResult,
    ProblemSpan, DecisionType, SeverityLevel, AbuseCategory
)
from core.decision_logic import DecisionContext
from core.explanation import ExplanationLevel


class TestTemperatureScaler:
    """Test temperature scaling functionality."""
    
    def test_temperature_scaler_initialization(self):
        """Test temperature scaler initialization."""
        scaler = TemperatureScaler()
        assert not scaler.is_fitted
        assert scaler.temperatures == {}
    
    def test_temperature_scaling_fit_and_transform(self):
        """Test temperature scaling fit and transform."""
        scaler = TemperatureScaler()
        
        # Mock data
        logits = {
            'lpe': np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]),
            'classifier': np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        }
        labels = np.array([1, 0, 1])
        
        # Fit scaler
        scaler.fit(logits, labels)
        assert scaler.is_fitted
        assert 'lpe' in scaler.temperatures
        assert 'classifier' in scaler.temperatures
        
        # Transform logits
        calibrated = scaler.transform(logits)
        assert 'lpe' in calibrated
        assert 'classifier' in calibrated
        assert calibrated['lpe'].shape == logits['lpe'].shape


class TestEnsembleWeightOptimizer:
    """Test ensemble weight optimization."""
    
    def test_weight_optimizer_initialization(self):
        """Test weight optimizer initialization."""
        optimizer = EnsembleWeightOptimizer()
        assert not optimizer.is_fitted
        assert optimizer.cv_folds == 5
    
    def test_weight_optimization(self):
        """Test weight optimization process."""
        optimizer = EnsembleWeightOptimizer()
        
        # Mock component predictions
        component_predictions = {
            'lpe': np.array([0.8, 0.2, 0.7, 0.1, 0.9]),
            'classifier': np.array([0.7, 0.3, 0.8, 0.2, 0.8]),
            'intent': np.array([0.6, 0.1, 0.5, 0.0, 0.7])
        }
        labels = np.array([1, 0, 1, 0, 1])
        
        # Optimize weights
        weights = optimizer.optimize_weights(component_predictions, labels)
        
        assert optimizer.is_fitted
        assert len(weights) == 3
        assert all(w >= 0 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to 1


class TestSpanConsolidator:
    """Test span consolidation functionality."""
    
    def test_span_consolidator_initialization(self):
        """Test span consolidator initialization."""
        consolidator = SpanConsolidator()
        assert consolidator.overlap_threshold == 0.3
        assert consolidator.max_spans == 10
    
    def test_span_consolidation(self):
        """Test span consolidation with overlapping spans."""
        consolidator = SpanConsolidator(overlap_threshold=0.5, max_spans=5)
        
        # Create test spans
        spans1 = [
            ProblemSpan("bad word", 0, 8, "profanity", 0.9, "lpe"),
            ProblemSpan("another bad", 20, 31, "insult", 0.8, "lpe")
        ]
        spans2 = [
            ProblemSpan("bad", 0, 3, "profanity", 0.7, "classifier"),
            ProblemSpan("terrible", 40, 48, "insult", 0.6, "classifier")
        ]
        
        consolidated = consolidator.consolidate_spans([spans1, spans2])
        
        assert len(consolidated) <= 5
        assert all(isinstance(span, ProblemSpan) for span in consolidated)
        # Should be sorted by confidence
        if len(consolidated) > 1:
            assert consolidated[0].confidence >= consolidated[1].confidence


class TestEnsembleAggregator:
    """Test main ensemble aggregator functionality."""
    
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
    def aggregator(self, mock_config_manager):
        """Create aggregator instance with mocked config."""
        return EnsembleAggregator(mock_config_manager)
    
    @pytest.fixture
    def sample_results(self):
        """Create sample component results."""
        lpe_result = LPEResult(
            matched_spans=[
                ProblemSpan("bad", 0, 3, "profanity", 0.9, "lpe_rule_1")
            ],
            categories=["profanity"],
            confidence_scores={"obscenity/profanity": 0.8},
            rule_traces=["Matched profanity pattern"]
        )
        
        classifier_result = ClassifierResult(
            category_probabilities={"obscenity/profanity": 0.7, "insult/harassment": 0.3},
            corporate_decision_prob={"allow": 0.2, "review": 0.3, "block": 0.5},
            severity_scores={"medium": 0.6, "high": 0.4},
            attention_spans=[
                ProblemSpan("bad", 0, 3, "profanity", 0.8, "classifier")
            ]
        )
        
        context_result = ContextResult(
            context_modifiers={"obscenity/profanity": 1.0},
            safe_context_detected={"obscenity/profanity": False},
            recommended_action=DecisionType.BLOCK
        )
        
        return lpe_result, classifier_result, context_result
    
    def test_aggregator_initialization(self, aggregator):
        """Test aggregator initialization."""
        assert aggregator.weights == {'lpe': 0.4, 'classifier': 0.5, 'intent': 0.1}
        assert aggregator.confidence_minimum == 0.6
        assert aggregator.review_threshold == 0.7
        assert aggregator.block_threshold == 0.85
    
    def test_component_score_extraction(self, aggregator, sample_results):
        """Test component score extraction."""
        lpe_result, classifier_result, context_result = sample_results
        
        component_scores = aggregator._extract_component_scores(
            lpe_result, classifier_result, context_result
        )
        
        assert 'lpe' in component_scores
        assert 'classifier' in component_scores
        assert 'intent' in component_scores
        assert component_scores['lpe']['obscenity/profanity'] == 0.8
        assert component_scores['classifier']['obscenity/profanity'] == 0.7
    
    def test_score_combination(self, aggregator, sample_results):
        """Test score combination with weights."""
        lpe_result, classifier_result, context_result = sample_results
        
        component_scores = aggregator._extract_component_scores(
            lpe_result, classifier_result, context_result
        )
        
        final_scores = aggregator._combine_scores(component_scores)
        
        assert 'obscenity/profanity' in final_scores
        # Should be weighted combination
        expected_score = (0.8 * 0.4 + 0.7 * 0.5 + 0.8 * 0.1) / (0.4 + 0.5 + 0.1)
        assert abs(final_scores['obscenity/profanity'] - expected_score) < 0.01
    
    def test_decision_making(self, aggregator):
        """Test decision making logic."""
        # Test block decision
        high_scores = {"obscenity/profanity": 0.9}
        decision = aggregator._make_decision(high_scores)
        assert decision == DecisionType.BLOCK
        
        # Test review decision
        medium_scores = {"obscenity/profanity": 0.75}
        decision = aggregator._make_decision(medium_scores)
        assert decision == DecisionType.REVIEW
        
        # Test allow decision
        low_scores = {"obscenity/profanity": 0.5}
        decision = aggregator._make_decision(low_scores)
        assert decision == DecisionType.ALLOW
    
    def test_full_aggregation(self, aggregator, sample_results):
        """Test full aggregation process."""
        lpe_result, classifier_result, context_result = sample_results
        
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
    
    def test_context_detection(self, aggregator):
        """Test context detection methods."""
        # Test code mixing detection
        assert aggregator._detect_code_mixing("Hello नमस्ते") == True
        assert aggregator._detect_code_mixing("Hello world") == False
        
        # Test obfuscation detection
        lpe_result_with_obfuscation = LPEResult(
            matched_spans=[],
            categories=[],
            confidence_scores={},
            rule_traces=["Leet speak pattern detected"]
        )
        assert aggregator._detect_obfuscation(lpe_result_with_obfuscation) == True
        
        lpe_result_clean = LPEResult(
            matched_spans=[],
            categories=[],
            confidence_scores={},
            rule_traces=["Normal pattern matched"]
        )
        assert aggregator._detect_obfuscation(lpe_result_clean) == False
    
    def test_explanation_generation(self, aggregator, sample_results):
        """Test explanation generation."""
        lpe_result, classifier_result, context_result = sample_results
        
        result = aggregator.aggregate(
            lpe_result, classifier_result, context_result,
            original_text="This is bad content"
        )
        
        # Test detailed explanation
        detailed_explanation = aggregator.generate_detailed_explanation(
            lpe_result, classifier_result, context_result, result,
            explanation_level=ExplanationLevel.DETAILED
        )
        
        assert isinstance(detailed_explanation, list)
        assert len(detailed_explanation) > 0
        assert any("Content is" in exp for exp in detailed_explanation)
    
    def test_attribution_report(self, aggregator, sample_results):
        """Test attribution report generation."""
        lpe_result, classifier_result, context_result = sample_results
        
        result = aggregator.aggregate(
            lpe_result, classifier_result, context_result
        )
        
        attribution = aggregator.get_attribution_report(
            lpe_result, classifier_result, context_result, result
        )
        
        assert isinstance(attribution, dict)
        assert len(attribution) > 0
        assert all(0 <= score <= 1 for score in attribution.values())
        assert abs(sum(attribution.values()) - 1.0) < 0.01  # Should sum to 1
    
    def test_performance_tracking(self, aggregator):
        """Test performance tracking functionality."""
        # Update performance
        aggregator.update_component_performance(
            {'lpe': True, 'classifier': False, 'intent': True},
            {'lpe': 0.8, 'classifier': 0.6, 'intent': 0.7}
        )
        
        # Get performance summary
        summary = aggregator.get_performance_summary()
        
        assert isinstance(summary, dict)
        assert 'lpe' in summary
        assert 'classifier' in summary
        assert 'intent' in summary
        
        for component_summary in summary.values():
            assert 'accuracy' in component_summary
            assert 'avg_confidence' in component_summary
            assert 'sample_count' in component_summary


if __name__ == "__main__":
    pytest.main([__file__])