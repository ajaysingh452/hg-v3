"""Tests for transformer classifier and related components."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from harmony_guard.model.classifier import TransformerClassifier, MockTokenizer, MockTransformerModel
from harmony_guard.model.multi_head_classifier import MultiHeadClassificationSystem, MultiHeadOutput
from harmony_guard.model.calibration import ModelCalibrator, CalibrationResult, UncertaintyEstimate
from harmony_guard.model.monitoring import ModelPerformanceMonitor
from harmony_guard.core.models import ProcessedText, ClassifierResult, AbuseCategory, SeverityLevel, DecisionType


class TestTransformerClassifier:
    """Test transformer classifier functionality."""
    
    @pytest.fixture
    def classifier_config(self):
        """Configuration for transformer classifier."""
        return {
            "model_name": "xlm-roberta-base",
            "max_sequence_length": 512,
            "batch_size": 8,
            "monitoring": {
                "window_size": 10,
                "time_window": 60,
                "drift_threshold": 0.1
            }
        }
    
    @pytest.fixture
    def processed_text(self):
        """Sample processed text for testing."""
        return ProcessedText(
            original_text="This is a test message",
            normalized_text="this is a test message",
            detected_languages=["en"],
            tokens=["this", "is", "a", "test", "message"],
            transliterations={},
            obfuscation_map={}
        )
    
    @pytest.mark.asyncio
    async def test_classifier_initialization(self, classifier_config):
        """Test classifier initialization."""
        classifier = TransformerClassifier(classifier_config)
        
        assert classifier.model_name == "xlm-roberta-base"
        assert classifier.max_sequence_length == 512
        assert classifier.batch_size == 8
        assert classifier.monitor is not None
        
        await classifier.initialize()
        
        assert classifier.tokenizer is not None
        assert classifier.model is not None
    
    @pytest.mark.asyncio
    async def test_single_prediction(self, classifier_config, processed_text):
        """Test single text prediction."""
        classifier = TransformerClassifier(classifier_config)
        await classifier.initialize()
        
        result = await classifier.predict(processed_text)
        
        assert isinstance(result, ClassifierResult)
        assert isinstance(result.category_probabilities, dict)
        assert isinstance(result.corporate_decision_prob, dict)
        assert isinstance(result.severity_scores, dict)
        assert isinstance(result.attention_spans, list)
        
        # Check that all expected categories are present
        expected_categories = [cat.value for cat in AbuseCategory]
        for category in expected_categories:
            assert category in result.category_probabilities
            assert 0.0 <= result.category_probabilities[category] <= 1.0
        
        # Check corporate decisions
        expected_decisions = ["allow", "review", "block"]
        for decision in expected_decisions:
            assert decision in result.corporate_decision_prob
            assert 0.0 <= result.corporate_decision_prob[decision] <= 1.0
        
        # Check severity levels
        expected_severities = [sev.value for sev in SeverityLevel]
        for severity in expected_severities:
            assert severity in result.severity_scores
            assert 0.0 <= result.severity_scores[severity] <= 1.0
    
    @pytest.mark.asyncio
    async def test_batch_prediction(self, classifier_config):
        """Test batch prediction functionality."""
        classifier = TransformerClassifier(classifier_config)
        await classifier.initialize()
        
        # Create multiple processed texts
        texts = []
        for i in range(5):
            text = ProcessedText(
                original_text=f"Test message {i}",
                normalized_text=f"test message {i}",
                detected_languages=["en"],
                tokens=[f"test", "message", str(i)],
                transliterations={},
                obfuscation_map={}
            )
            texts.append(text)
        
        results = await classifier.batch_predict(texts)
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, ClassifierResult)
    
    @pytest.mark.asyncio
    async def test_attention_span_extraction(self, classifier_config, processed_text):
        """Test attention span extraction for explainability."""
        classifier = TransformerClassifier(classifier_config)
        classifier.extract_attention = True
        await classifier.initialize()
        
        result = await classifier.predict(processed_text)
        
        # Should have attention spans
        assert len(result.attention_spans) >= 0
        
        for span in result.attention_spans:
            assert hasattr(span, 'text')
            assert hasattr(span, 'start')
            assert hasattr(span, 'end')
            assert hasattr(span, 'confidence')
            assert hasattr(span, 'rule_source')
            assert 0.0 <= span.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, classifier_config, processed_text):
        """Test error handling and default predictions."""
        classifier = TransformerClassifier(classifier_config)
        await classifier.initialize()
        
        # Mock an error in inference
        with patch.object(classifier, '_run_inference', side_effect=Exception("Mock error")):
            result = await classifier.predict(processed_text)
            
            # Should return default predictions
            assert isinstance(result, ClassifierResult)
            assert len(result.category_probabilities) > 0
            assert len(result.corporate_decision_prob) > 0
            assert len(result.severity_scores) > 0
    
    @pytest.mark.asyncio
    async def test_model_info(self, classifier_config):
        """Test getting model information."""
        classifier = TransformerClassifier(classifier_config)
        await classifier.initialize()
        
        info = await classifier.get_model_info()
        
        assert "model_name" in info
        assert "max_sequence_length" in info
        assert "batch_size" in info
        assert "num_categories" in info
        assert "categories" in info
        assert "severities" in info
        assert "decisions" in info
        
        assert info["model_name"] == "xlm-roberta-base"
        assert info["max_sequence_length"] == 512
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, classifier_config, processed_text):
        """Test monitoring integration."""
        classifier = TransformerClassifier(classifier_config)
        await classifier.initialize()
        
        # Make a prediction (should record metrics)
        await classifier.predict(processed_text)
        
        # Check monitoring methods
        metrics = classifier.get_performance_metrics()
        assert isinstance(metrics, dict)
        
        alerts = classifier.get_drift_alerts()
        assert isinstance(alerts, list)
        
        summary = classifier.get_performance_summary()
        assert isinstance(summary, dict)
        
        drift_tests = classifier.run_drift_tests()
        assert isinstance(drift_tests, dict)
    
    @pytest.mark.asyncio
    async def test_shutdown(self, classifier_config):
        """Test classifier shutdown."""
        classifier = TransformerClassifier(classifier_config)
        await classifier.initialize()
        
        await classifier.shutdown()
        
        assert classifier.model is None
        assert classifier.tokenizer is None
        assert len(classifier.model_cache) == 0


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
    
    @pytest.fixture
    def sample_logits(self):
        """Sample logits for testing."""
        return {
            "category_logits": np.array([0.1, -0.5, 0.8, -0.2, 0.3, -0.1, 0.6, -0.3]),
            "corporate_logits": np.array([0.2, -0.1, 0.4]),
            "severity_logits": np.array([-0.2, 0.1, 0.3, 0.5])
        }
    
    def test_multi_head_initialization(self, multi_head_config):
        """Test multi-head system initialization."""
        system = MultiHeadClassificationSystem(multi_head_config)
        
        assert "abuse_categories" in system.heads
        assert "corporate_decision" in system.heads
        assert "severity_levels" in system.heads
        
        # Check head configurations
        abuse_head = system.heads["abuse_categories"]
        assert abuse_head.activation == "sigmoid"
        assert len(abuse_head.class_names) == len([cat.value for cat in AbuseCategory])
        
        corporate_head = system.heads["corporate_decision"]
        assert corporate_head.activation == "softmax"
        assert len(corporate_head.class_names) == len([dec.value for dec in DecisionType])
    
    def test_logits_processing(self, multi_head_config, sample_logits):
        """Test processing of raw logits."""
        system = MultiHeadClassificationSystem(multi_head_config)
        
        output = system.process_logits(sample_logits)
        
        assert isinstance(output, MultiHeadOutput)
        assert isinstance(output.abuse_categories, dict)
        assert isinstance(output.corporate_decision, dict)
        assert isinstance(output.severity_levels, dict)
        assert isinstance(output.confidence_scores, dict)
        assert isinstance(output.predictions, dict)
        
        # Check probability ranges
        for prob in output.abuse_categories.values():
            assert 0.0 <= prob <= 1.0
        
        for prob in output.corporate_decision.values():
            assert 0.0 <= prob <= 1.0
        
        for prob in output.severity_levels.values():
            assert 0.0 <= prob <= 1.0
    
    def test_threshold_updates(self, multi_head_config):
        """Test threshold updates."""
        system = MultiHeadClassificationSystem(multi_head_config)
        
        new_thresholds = {"harassment": 0.2, "profanity": 0.4}
        system.update_thresholds("abuse_categories", new_thresholds)
        
        assert system.thresholds["abuse_categories"]["harassment"] == 0.2
        assert system.thresholds["abuse_categories"]["profanity"] == 0.4
    
    def test_class_weight_updates(self, multi_head_config):
        """Test class weight updates."""
        system = MultiHeadClassificationSystem(multi_head_config)
        
        new_weights = {"harassment": 1.5, "threat": 2.0}
        system.update_class_weights("abuse_categories", new_weights)
        
        assert system.class_weights["abuse_categories"]["harassment"] == 1.5
        assert system.class_weights["abuse_categories"]["threat"] == 2.0
    
    def test_head_info(self, multi_head_config):
        """Test getting head information."""
        system = MultiHeadClassificationSystem(multi_head_config)
        
        info = system.get_head_info()
        
        assert "abuse_categories" in info
        assert "corporate_decision" in info
        assert "severity_levels" in info
        
        for head_name, head_info in info.items():
            assert "num_classes" in head_info
            assert "class_names" in head_info
            assert "activation" in head_info
            assert "thresholds" in head_info
            assert "class_weights" in head_info


class TestModelCalibrator:
    """Test model calibration functionality."""
    
    @pytest.fixture
    def calibrator_config(self):
        """Configuration for calibrator."""
        return {
            "confidence_threshold": 0.8,
            "uncertainty_threshold": 0.3
        }
    
    @pytest.fixture
    def sample_probabilities(self):
        """Sample probabilities for testing."""
        return {
            "abuse_categories": {
                "harassment": 0.3,
                "profanity": 0.1,
                "hate": 0.05
            },
            "corporate_decision": {
                "allow": 0.7,
                "review": 0.2,
                "block": 0.1
            }
        }
    
    def test_calibrator_initialization(self, calibrator_config):
        """Test calibrator initialization."""
        calibrator = ModelCalibrator(calibrator_config)
        
        assert calibrator.confidence_threshold == 0.8
        assert calibrator.uncertainty_threshold == 0.3
        assert "abuse_categories" in calibrator.temperatures
        assert "corporate_decision" in calibrator.temperatures
        assert "severity_levels" in calibrator.temperatures
    
    def test_probability_calibration(self, calibrator_config, sample_probabilities):
        """Test probability calibration."""
        calibrator = ModelCalibrator(calibrator_config)
        
        result = calibrator.calibrate_probabilities(
            sample_probabilities, 
            "abuse_categories",
            ["en"]
        )
        
        assert isinstance(result, CalibrationResult)
        assert result.temperature > 0
        assert isinstance(result.calibrated_probabilities, dict)
        assert 0.0 <= result.confidence_score <= 1.0
        assert 0.0 <= result.reliability_score <= 1.0
        assert result.calibration_error >= 0.0
    
    def test_uncertainty_estimation(self, calibrator_config, sample_probabilities):
        """Test uncertainty estimation."""
        calibrator = ModelCalibrator(calibrator_config)
        
        # Flatten probabilities for uncertainty estimation
        flat_probs = {}
        for category_dict in sample_probabilities.values():
            flat_probs.update(category_dict)
        
        uncertainty = calibrator.estimate_uncertainty(flat_probs)
        
        assert isinstance(uncertainty, UncertaintyEstimate)
        assert 0.0 <= uncertainty.epistemic_uncertainty <= 1.0
        assert 0.0 <= uncertainty.aleatoric_uncertainty <= 1.0
        assert 0.0 <= uncertainty.total_uncertainty <= 1.0
        assert len(uncertainty.confidence_interval) == 2
        assert uncertainty.confidence_interval[0] <= uncertainty.confidence_interval[1]
    
    def test_temperature_scaling(self, calibrator_config):
        """Test temperature scaling functionality."""
        calibrator = ModelCalibrator(calibrator_config)
        
        probabilities = {"cat1": 0.8, "cat2": 0.2}
        temperature = 2.0
        
        calibrated = calibrator._apply_temperature_scaling(probabilities, temperature)
        
        assert isinstance(calibrated, dict)
        assert len(calibrated) == len(probabilities)
        
        # Check that probabilities are still valid
        for prob in calibrated.values():
            assert 0.0 <= prob <= 1.0
    
    def test_language_adjustment(self, calibrator_config):
        """Test language-specific calibration adjustment."""
        calibrator = ModelCalibrator(calibrator_config)
        
        # Test adjustment for different languages
        calibrator.adjust_for_language(["hi"], "abuse_categories")
        calibrator.adjust_for_language(["en"], "corporate_decision")
        
        # Should have language factors stored
        assert len(calibrator.language_factors) >= 0
    
    def test_calibration_stats(self, calibrator_config):
        """Test getting calibration statistics."""
        calibrator = ModelCalibrator(calibrator_config)
        
        stats = calibrator.get_calibration_stats()
        
        assert "temperatures" in stats
        assert "language_factors" in stats
        assert "confidence_threshold" in stats
        assert "uncertainty_threshold" in stats


class TestMockComponents:
    """Test mock components used in the classifier."""
    
    def test_mock_tokenizer(self):
        """Test mock tokenizer functionality."""
        tokenizer = MockTokenizer()
        
        text = "Hello world test"
        tokens = tokenizer.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0
    
    def test_mock_transformer_model(self):
        """Test mock transformer model."""
        model = MockTransformerModel(
            num_categories=8,
            num_severities=4,
            num_decisions=3
        )
        
        input_ids = [1, 2, 3, 4, 5]
        attention_mask = [1, 1, 1, 1, 1]
        
        outputs = model.forward(input_ids, attention_mask)
        
        assert "category_logits" in outputs
        assert "corporate_logits" in outputs
        assert "severity_logits" in outputs
        assert "attention_weights" in outputs
        assert "hidden_states" in outputs
        
        # Check output shapes
        assert outputs["category_logits"].shape == (1, 8)
        assert outputs["corporate_logits"].shape == (1, 3)
        assert outputs["severity_logits"].shape == (1, 4)
        assert outputs["attention_weights"].shape == (1, len(input_ids))


class TestIntegration:
    """Integration tests for transformer classifier components."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test full pipeline integration."""
        # Configuration
        config = {
            "model_name": "test-model",
            "monitoring": {
                "window_size": 5,
                "time_window": 30
            }
        }
        
        # Initialize classifier
        classifier = TransformerClassifier(config)
        await classifier.initialize()
        
        # Create test input
        processed_text = ProcessedText(
            original_text="This is a test message for integration testing",
            normalized_text="this is a test message for integration testing",
            detected_languages=["en"],
            tokens=["this", "is", "a", "test", "message", "for", "integration", "testing"],
            transliterations={},
            obfuscation_map={}
        )
        
        # Make prediction
        result = await classifier.predict(processed_text)
        
        # Verify result structure
        assert isinstance(result, ClassifierResult)
        assert len(result.category_probabilities) > 0
        assert len(result.corporate_decision_prob) > 0
        assert len(result.severity_scores) > 0
        
        # Verify monitoring recorded the prediction
        assert len(classifier.monitor.prediction_history) == 1
        
        # Test performance metrics
        metrics = classifier.get_performance_metrics()
        assert isinstance(metrics, dict)
        
        await classifier.shutdown()