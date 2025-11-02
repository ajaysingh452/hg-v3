"""Tests for language identification module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from core.language_detection import LanguageIdentifier
from core.models import LanguageDetection


class TestLanguageIdentifier:
    """Test cases for LanguageIdentifier class."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return {
            "supported_languages": ["en", "hi", "hi-Latn", "bn", "te", "ta"],
            "confidence_threshold": 0.7,
            "segment_length": 50,
            "fallback_to_ngram": True
        }
    
    @pytest.fixture
    def identifier(self, config):
        """Create LanguageIdentifier instance."""
        return LanguageIdentifier(config)
    
    def test_english_detection(self, identifier):
        """Test English language detection."""
        text = "This is a simple English sentence for testing purposes."
        languages = identifier.detect_languages(text)
        
        assert len(languages) > 0
        assert languages[0].code == "en"
        assert languages[0].confidence > 0.5
    
    def test_hindi_devanagari_detection(self, identifier):
        """Test Hindi (Devanagari) language detection."""
        text = "यह एक हिंदी वाक्य है जो परीक्षण के लिए उपयोग किया जा रहा है।"
        languages = identifier.detect_languages(text)
        
        assert len(languages) > 0
        assert languages[0].code == "hi"
        assert languages[0].confidence > 0.5
    
    def test_hinglish_detection(self, identifier):
        """Test Hinglish (Hindi in Latin script) detection."""
        text = "Yaar, main office ja raha hai. Tum kya kar rahe ho?"
        languages = identifier.detect_languages(text)
        
        assert len(languages) > 0
        # Should detect Hinglish
        hinglish_detected = any(lang.code == "hi-Latn" for lang in languages)
        assert hinglish_detected
    
    def test_code_mixed_content(self, identifier):
        """Test detection of code-mixed content."""
        text = "I am going to the market आज शाम को. Will you come with me?"
        languages = identifier.detect_languages(text)
        
        # Should detect code-mixed content (could be Hinglish or multiple languages)
        assert len(languages) >= 1
        lang_codes = [lang.code for lang in languages]
        # Should detect either Hinglish or separate English/Hindi
        assert any(code in ["hi-Latn", "en", "hi"] for code in lang_codes)
    
    def test_empty_text(self, identifier):
        """Test handling of empty text."""
        languages = identifier.detect_languages("")
        
        assert len(languages) == 1
        assert languages[0].code == "en"
        assert languages[0].confidence == 0.5
    
    def test_short_text(self, identifier):
        """Test handling of very short text."""
        text = "Hi"
        languages = identifier.detect_languages(text)
        
        assert len(languages) > 0
        assert languages[0].confidence > 0
    
    def test_per_segment_detection(self, identifier):
        """Test per-segment language detection."""
        text = "This is English. यह हिंदी है। This is English again."
        segments = identifier.detect_per_segment(text, segment_length=3)
        
        assert len(segments) > 0
        # Each segment should have language detections
        for segment_text, segment_languages in segments:
            assert len(segment_languages) > 0
            assert isinstance(segment_languages[0], LanguageDetection)
    
    def test_language_hints(self, identifier):
        """Test language detection with hints."""
        text = "Ambiguous text that could be multiple languages"
        
        # Without hints
        languages_no_hints = identifier.detect_languages(text)
        
        # With hints
        languages_with_hints = identifier.detect_languages(text, hints=["en"])
        
        # Hinted language should have higher confidence
        en_confidence_no_hints = next(
            (lang.confidence for lang in languages_no_hints if lang.code == "en"), 0
        )
        en_confidence_with_hints = next(
            (lang.confidence for lang in languages_with_hints if lang.code == "en"), 0
        )
        
        assert en_confidence_with_hints >= en_confidence_no_hints
    
    def test_confidence_thresholding(self, identifier):
        """Test confidence thresholding functionality."""
        text = "Test text"
        languages = identifier.detect_languages(text)
        
        # All returned languages should meet minimum confidence
        for lang in languages:
            assert lang.confidence >= 0.3  # Minimum threshold for keeping languages
    
    def test_percentage_normalization(self, identifier):
        """Test that language percentages sum to 100."""
        text = "Mixed content with English and हिंदी text together."
        languages = identifier.detect_languages(text)
        
        total_percentage = sum(lang.percentage for lang in languages)
        assert abs(total_percentage - 100.0) < 0.1  # Allow small floating point errors
    
    def test_script_detection(self, identifier):
        """Test script-based detection."""
        # Test different scripts
        test_cases = [
            ("Hello world", "latin"),
            ("नमस्ते दुनिया", "devanagari"),
            ("হ্যালো বিশ্ব", "bengali"),
        ]
        
        for text, expected_script in test_cases:
            script_scores = identifier._detect_scripts(text)
            assert expected_script in script_scores
            assert script_scores[expected_script] > 0.5
    
    def test_ngram_extraction(self, identifier):
        """Test n-gram extraction functionality."""
        text = "hello"
        ngrams = identifier._extract_ngrams(text, n=3)
        
        assert len(ngrams) > 0
        # Should include boundary markers
        assert any("<" in ngram for ngram in ngrams)
        assert any(">" in ngram for ngram in ngrams)
    
    def test_hinglish_patterns(self, identifier):
        """Test Hinglish pattern detection."""
        hinglish_text = "Yaar, kya hai yeh? Main samjha nahi."
        english_text = "What is this? I don't understand."
        
        hinglish_score = identifier._detect_hinglish(hinglish_text)
        english_score = identifier._detect_hinglish(english_text)
        
        assert hinglish_score > english_score
        assert hinglish_score > 0.1