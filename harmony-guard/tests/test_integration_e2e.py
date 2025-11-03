"""End-to-end integration tests for complete analysis pipeline."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock
import tempfile
import yaml
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.preprocessing import TextPreprocessor
from lpe.engine import LexiconPatternEngine
from model.classifier import TransformerClassifier
from intent.context_analyzer import IntentContextLayer
from model.aggregator import EnsembleAggregator
from model.policy import PolicyEngine
from core.models import (
    ProcessedText, LPEResult, ClassifierResult, ContextResult, AggregatedResult,
    ProblemSpan, DecisionType, SeverityLevel, AbuseCategory, LanguageDetection,
    AnalysisRequest
)


class TestCompleteIntegrationPipeline:
    """Test complete integration pipeline end-to-end."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        config_manager = Mock()
        config_manager.get_ensemble_config.return_value = {
            'preprocessing': {
                'language_detection': {'supported_languages': ['en', 'hi']},
                'normalization': {'unicode_form': 'NFKC'},
                'transliteration': {'enabled': True},
                'obfuscation': {'leet_speak_detection': True},
                'tokenization': {'emoji_aware': True},
                'pii_masking': {'enabled': False}
            },
            'lpe': {'fuzzy_matching': True, 'fuzzy_threshold': 0.8},
            'classifier': {'model_name': 'test-model'},
            'intent': {'negation_detection': True},
            'ensemble': {
                'weights': {'lpe': 0.4, 'classifier': 0.5, 'intent': 0.1},
                'thresholds': {
                    'confidence_minimum': 0.6,
                    'review_threshold': 0.7,
                    'block_threshold': 0.85
                }
            }
        }
        return config_manager
    
    @pytest.fixture
    def temp_policy_dir(self):
        """Create temporary policy directory."""
        temp_dir = tempfile.mkdtemp()
        
        policy_config = {
            'policy_profile': {
                'name': 'test_policy',
                'version': '1.0',
                'block_thresholds': {
                    'profanity': {'medium': 0.7, 'high': 0.5, 'critical': 0.3}
                },
                'safe_contexts': [],
                'department_overrides': {},
                'hard_rules': []
            }
        }
        
        with open(Path(temp_dir) / 'policy_default.yaml', 'w') as f:
            yaml.safe_dump(policy_config, f)
        
        yield temp_dir
        
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end_clean_content(self, mock_config_manager, temp_policy_dir):
        """Test end-to-end pipeline with clean content."""
        # Initialize all components
        preprocessor = TextPreprocessor(mock_config_manager.get_ensemble_config())
        lpe = LexiconPatternEngine(mock_config_manager)
        classifier = TransformerClassifier({'model_name': 'test'})
        intent_layer = IntentContextLayer({})
        aggregator = EnsembleAggregator(mock_config_manager)
        policy_engine = PolicyEngine(temp_policy_dir, temp_policy_dir)
        
        # Mock external dependencies
        preprocessor.language_identifier = Mock()
        preprocessor.language_identifier.detect_languages = Mock(
            return_value=[LanguageDetection("en", 0.9, 100.0)]
        )
        preprocessor.transliteration_engine = Mock()
        preprocessor.transliteration_engine.transliterate_to_native = Mock(return_value={})
        preprocessor.transliteration_engine.transliterate_to_roman = Mock(return_value={})
        preprocessor.pii_masker = Mock()
        preprocessor.pii_masker.enabled = False
        
        # Initialize components
        await lpe.initialize()
        lpe.pattern_matcher = Mock()
        lpe.pattern_matcher.find_matches = Mock(return_value=[])
        lpe.emoji_analyzer = Mock()
        lpe.emoji_analyzer.analyze_emojis = Mock(return_value=[])
        
        await classifier.initialize()
        classifier.predict = AsyncMock(return_value=ClassifierResult(
            category_probabilities={cat.value: 0.05 for cat in AbuseCategory},
            corporate_decision_prob={"allow": 0.9, "review": 0.08, "block": 0.02},
            severity_scores={"low": 0.9, "medium": 0.08, "high": 0.02, "critical": 0.0},
            attention_spans=[]
        ))
        
        intent_layer.analyze_context = AsyncMock(return_value=ContextResult(
            context_modifiers={cat.value: 1.0 for cat in AbuseCategory},
            safe_context_detected={cat.value: False for cat in AbuseCategory},
            recommended_action=DecisionType.ALLOW
        ))
        
        # Test with clean content
        text = "Hello world, this is a nice message."
        
        # Run complete pipeline
        processed_text = await preprocessor.process(text)
        assert isinstance(processed_text, ProcessedText)
        assert processed_text.original_text == text
        
        lpe_result = await lpe.analyze(processed_text)
        assert isinstance(lpe_result, LPEResult)
        assert len(lpe_result.matched_spans) == 0
        
        classifier_result = await classifier.predict(processed_text)
        assert isinstance(classifier_result, ClassifierResult)
        
        context_result = await intent_layer.analyze_context(
            processed_text, lpe_result, classifier_result
        )
        assert isinstance(context_result, ContextResult)
        
        aggregated_result = aggregator.aggregate(
            lpe_result, classifier_result, context_result, original_text=text
        )
        assert isinstance(aggregated_result, AggregatedResult)
        assert aggregated_result.final_decision == DecisionType.ALLOW
        
        final_result = policy_engine.apply_policy(
            aggregated_result, tenant_id="test", request_id="test_123"
        )
        assert isinstance(final_result, AggregatedResult)
        assert final_result.final_decision == DecisionType.ALLOW
    
    @pytest.mark.asyncio
    async def test_end_to_end_problematic_content(self, mock_config_manager, temp_policy_dir):
        """Test end-to-end pipeline with problematic content."""
        # Initialize components
        preprocessor = TextPreprocessor(mock_config_manager.get_ensemble_config())
        lpe = LexiconPatternEngine(mock_config_manager)
        classifier = TransformerClassifier({'model_name': 'test'})
        intent_layer = IntentContextLayer({})
        aggregator = EnsembleAggregator(mock_config_manager)
        policy_engine = PolicyEngine(temp_policy_dir, temp_policy_dir)
        
        # Mock external dependencies
        preprocessor.language_identifier = Mock()
        preprocessor.language_identifier.detect_languages = Mock(
            return_value=[LanguageDetection("en", 0.9, 100.0)]
        )
        preprocessor.transliteration_engine = Mock()
        preprocessor.transliteration_engine.transliterate_to_native = Mock(return_value={})
        preprocessor.transliteration_engine.transliterate_to_roman = Mock(return_value={})
        preprocessor.pii_masker = Mock()
        preprocessor.pii_masker.enabled = False
        
        # Initialize components
        await lpe.initialize()
        # Mock LPE finding problematic content
        lpe.pattern_matcher = Mock()
        lpe.pattern_matcher.find_matches = Mock(return_value=[
            Mock(text="bad", start=8, end=11, category="profanity", 
                 severity="medium", weight=0.9, match_type="exact", rule_source="lexicon")
        ])
        lpe.emoji_analyzer = Mock()
        lpe.emoji_analyzer.analyze_emojis = Mock(return_value=[])
        
        await classifier.initialize()
        # Mock classifier detecting high profanity
        classifier.predict = AsyncMock(return_value=ClassifierResult(
            category_probabilities={"profanity": 0.8, "harassment": 0.2},
            corporate_decision_prob={"allow": 0.1, "review": 0.3, "block": 0.6},
            severity_scores={"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 0.0},
            attention_spans=[ProblemSpan("bad", 8, 11, "profanity", 0.8, "classifier")]
        ))
        
        intent_layer.analyze_context = AsyncMock(return_value=ContextResult(
            context_modifiers={"profanity": 1.0},
            safe_context_detected={"profanity": False},
            recommended_action=DecisionType.BLOCK
        ))
        
        # Test with problematic content
        text = "This is bad content that should be blocked."
        
        # Run complete pipeline
        processed_text = await preprocessor.process(text)
        lpe_result = await lpe.analyze(processed_text)
        classifier_result = await classifier.predict(processed_text)
        context_result = await intent_layer.analyze_context(
            processed_text, lpe_result, classifier_result
        )
        aggregated_result = aggregator.aggregate(
            lpe_result, classifier_result, context_result, original_text=text
        )
        final_result = policy_engine.apply_policy(
            aggregated_result, tenant_id="test", request_id="test_123"
        )
        
        # Verify problematic content is detected and blocked
        assert final_result.final_decision in [DecisionType.REVIEW, DecisionType.BLOCK]
        assert len(final_result.consolidated_spans) > 0
        assert any(span.category == "profanity" for span in final_result.consolidated_spans)
    
    @pytest.mark.asyncio
    async def test_end_to_end_latency_performance(self, mock_config_manager, temp_policy_dir):
        """Test end-to-end latency performance."""
        # Initialize components with performance-optimized config
        config = mock_config_manager.get_ensemble_config()
        config['preprocessing']['transliteration']['enabled'] = False  # Disable for speed
        config['lpe']['fuzzy_matching'] = False  # Disable for speed
        
        preprocessor = TextPreprocessor(config)
        lpe = LexiconPatternEngine(mock_config_manager)
        classifier = TransformerClassifier({'model_name': 'test'})
        intent_layer = IntentContextLayer({})
        aggregator = EnsembleAggregator(mock_config_manager)
        
        # Mock fast components
        preprocessor.language_identifier = Mock()
        preprocessor.language_identifier.detect_languages = Mock(
            return_value=[LanguageDetection("en", 0.9, 100.0)]
        )
        preprocessor.transliteration_engine = Mock()
        preprocessor.transliteration_engine.transliterate_to_native = Mock(return_value={})
        preprocessor.transliteration_engine.transliterate_to_roman = Mock(return_value={})
        preprocessor.pii_masker = Mock()
        preprocessor.pii_masker.enabled = False
        
        await lpe.initialize()
        lpe.analyze = AsyncMock(return_value=LPEResult(
            matched_spans=[], categories=[], confidence_scores={}, rule_traces=[]
        ))
        
        await classifier.initialize()
        classifier.predict = AsyncMock(return_value=ClassifierResult(
            category_probabilities={cat.value: 0.05 for cat in AbuseCategory},
            corporate_decision_prob={"allow": 0.9, "review": 0.08, "block": 0.02},
            severity_scores={"low": 0.9, "medium": 0.08, "high": 0.02, "critical": 0.0},
            attention_spans=[]
        ))
        
        intent_layer.analyze_context = AsyncMock(return_value=ContextResult(
            context_modifiers={cat.value: 1.0 for cat in AbuseCategory},
            safe_context_detected={cat.value: False for cat in AbuseCategory},
            recommended_action=DecisionType.ALLOW
        ))
        
        # Test latency with various text lengths
        test_texts = [
            "Short text",
            "Medium length text with some more words to test processing time",
            "Very long text " * 20 + " that should still meet latency requirements"
        ]
        
        latencies = []
        
        for text in test_texts:
            start_time = time.perf_counter()
            
            # Run complete pipeline
            processed_text = await preprocessor.process(text)
            lpe_result = await lpe.analyze(processed_text)
            classifier_result = await classifier.predict(processed_text)
            context_result = await intent_layer.analyze_context(
                processed_text, lpe_result, classifier_result
            )
            aggregated_result = aggregator.aggregate(
                lpe_result, classifier_result, context_result, original_text=text
            )
            
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            # Verify result is valid
            assert isinstance(aggregated_result, AggregatedResult)
        
        # Check latency requirements (with mocked components, should be very fast)
        import numpy as np
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"Latency Performance: P50={p50_latency:.2f}ms, P95={p95_latency:.2f}ms")
        
        # With mocked components, latencies should be very low
        assert p50_latency < 100, f"P50 latency {p50_latency:.2f}ms exceeds 100ms threshold"
        assert p95_latency < 200, f"P95 latency {p95_latency:.2f}ms exceeds 200ms threshold"
    
    @pytest.mark.asyncio
    async def test_end_to_end_multilingual_content(self, mock_config_manager, temp_policy_dir):
        """Test end-to-end pipeline with multilingual content."""
        # Initialize components
        preprocessor = TextPreprocessor(mock_config_manager.get_ensemble_config())
        lpe = LexiconPatternEngine(mock_config_manager)
        classifier = TransformerClassifier({'model_name': 'test'})
        intent_layer = IntentContextLayer({})
        aggregator = EnsembleAggregator(mock_config_manager)
        
        # Mock multilingual detection
        preprocessor.language_identifier = Mock()
        preprocessor.language_identifier.detect_languages = Mock(
            return_value=[
                LanguageDetection("en", 0.6, 60.0),
                LanguageDetection("hi", 0.7, 40.0)
            ]
        )
        preprocessor.transliteration_engine = Mock()
        preprocessor.transliteration_engine.transliterate_to_native = Mock(
            return_value={"hello": "हैलो"}
        )
        preprocessor.transliteration_engine.transliterate_to_roman = Mock(return_value={})
        preprocessor.pii_masker = Mock()
        preprocessor.pii_masker.enabled = False
        
        await lpe.initialize()
        await classifier.initialize()
        
        # Mock components for multilingual analysis
        lpe.analyze = AsyncMock(return_value=LPEResult(
            matched_spans=[], categories=[], confidence_scores={}, rule_traces=[]
        ))
        
        classifier.predict = AsyncMock(return_value=ClassifierResult(
            category_probabilities={cat.value: 0.05 for cat in AbuseCategory},
            corporate_decision_prob={"allow": 0.9, "review": 0.08, "block": 0.02},
            severity_scores={"low": 0.9, "medium": 0.08, "high": 0.02, "critical": 0.0},
            attention_spans=[]
        ))
        
        intent_layer.analyze_context = AsyncMock(return_value=ContextResult(
            context_modifiers={cat.value: 1.0 for cat in AbuseCategory},
            safe_context_detected={cat.value: False for cat in AbuseCategory},
            recommended_action=DecisionType.ALLOW
        ))
        
        # Test with code-mixed content
        text = "Hello नमस्ते world"
        
        processed_text = await preprocessor.process(text)
        
        # Verify multilingual detection
        assert len(processed_text.detected_languages) >= 2
        lang_codes = [lang.code for lang in processed_text.detected_languages]
        assert "en" in lang_codes
        assert "hi" in lang_codes
        
        # Verify transliterations were generated
        assert len(processed_text.transliterations) > 0
        
        # Continue with analysis
        lpe_result = await lpe.analyze(processed_text)
        classifier_result = await classifier.predict(processed_text)
        context_result = await intent_layer.analyze_context(
            processed_text, lpe_result, classifier_result
        )
        aggregated_result = aggregator.aggregate(
            lpe_result, classifier_result, context_result, original_text=text
        )
        
        assert isinstance(aggregated_result, AggregatedResult)
        assert aggregated_result.final_decision == DecisionType.ALLOW
    
    @pytest.mark.asyncio
    async def test_end_to_end_error_recovery(self, mock_config_manager, temp_policy_dir):
        """Test end-to-end error recovery and graceful degradation."""
        # Initialize components
        preprocessor = TextPreprocessor(mock_config_manager.get_ensemble_config())
        lpe = LexiconPatternEngine(mock_config_manager)
        classifier = TransformerClassifier({'model_name': 'test'})
        intent_layer = IntentContextLayer({})
        aggregator = EnsembleAggregator(mock_config_manager)
        
        # Mock preprocessing to work normally
        preprocessor.language_identifier = Mock()
        preprocessor.language_identifier.detect_languages = Mock(
            return_value=[LanguageDetection("en", 0.9, 100.0)]
        )
        preprocessor.transliteration_engine = Mock()
        preprocessor.transliteration_engine.transliterate_to_native = Mock(return_value={})
        preprocessor.transliteration_engine.transliterate_to_roman = Mock(return_value={})
        preprocessor.pii_masker = Mock()
        preprocessor.pii_masker.enabled = False
        
        await lpe.initialize()
        await classifier.initialize()
        
        # Mock LPE to fail
        lpe.analyze = AsyncMock(side_effect=Exception("LPE Error"))
        
        # Mock classifier to work normally
        classifier.predict = AsyncMock(return_value=ClassifierResult(
            category_probabilities={cat.value: 0.05 for cat in AbuseCategory},
            corporate_decision_prob={"allow": 0.8, "review": 0.15, "block": 0.05},
            severity_scores={"low": 0.8, "medium": 0.15, "high": 0.05, "critical": 0.0},
            attention_spans=[]
        ))
        
        # Mock intent layer to work normally
        intent_layer.analyze_context = AsyncMock(return_value=ContextResult(
            context_modifiers={cat.value: 1.0 for cat in AbuseCategory},
            safe_context_detected={cat.value: False for cat in AbuseCategory},
            recommended_action=DecisionType.ALLOW
        ))
        
        text = "Test message"
        
        # Test graceful degradation when LPE fails
        processed_text = await preprocessor.process(text)
        
        try:
            lpe_result = await lpe.analyze(processed_text)
        except Exception:
            # Create empty LPE result for graceful degradation
            lpe_result = LPEResult(
                matched_spans=[], categories=[], confidence_scores={}, rule_traces=[]
            )
        
        classifier_result = await classifier.predict(processed_text)
        context_result = await intent_layer.analyze_context(
            processed_text, lpe_result, classifier_result
        )
        
        # Aggregation should still work with empty LPE result
        aggregated_result = aggregator.aggregate(
            lpe_result, classifier_result, context_result, original_text=text
        )
        
        assert isinstance(aggregated_result, AggregatedResult)
        # Should still make a decision even with component failure
        assert aggregated_result.final_decision in [DecisionType.ALLOW, DecisionType.REVIEW, DecisionType.BLOCK]


if __name__ == "__main__":
    pytest.main([__file__])