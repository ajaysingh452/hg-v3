"""Tests for text preprocessing pipeline components."""

import pytest
import asyncio
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.preprocessing import TextPreprocessor
from core.normalization import TextNormalizer
from core.tokenization import MultilingualTokenizer, TokenSpan
from core.obfuscation import ObfuscationHandler
from core.models import ProcessedText, LanguageDetection


class TestTextNormalizer:
    """Test text normalization functionality."""
    
    @pytest.fixture
    def normalizer_config(self):
        """Default normalizer configuration."""
        return {
            "unicode_form": "NFKC",
            "strip_zero_width": True,
            "fold_diacritics": True,
            "normalize_punctuation": True,
            "compress_repeated_chars": True,
            "max_char_repetition": 3
        }
    
    @pytest.fixture
    def normalizer(self, normalizer_config):
        """Create normalizer instance."""
        return TextNormalizer(normalizer_config)
    
    def test_unicode_normalization(self, normalizer):
        """Test Unicode normalization."""
        # Test NFKC normalization
        text = "ﬁle"  # fi ligature
        normalized = normalizer.normalize(text)
        assert "fi" in normalized
    
    def test_zero_width_character_removal(self, normalizer):
        """Test removal of zero-width characters."""
        text = "hello\u200Bworld\u200C"  # Zero width space and non-joiner
        normalized = normalizer.normalize(text)
        assert "\u200B" not in normalized
        assert "\u200C" not in normalized
        assert normalized == "helloworld"
    
    def test_homoglyph_normalization(self, normalizer):
        """Test homoglyph character normalization."""
        # Cyrillic 'a' to Latin 'a'
        text = "hеllo"  # Contains Cyrillic 'е'
        normalized = normalizer.normalize(text)
        assert "hello" in normalized
    
    def test_diacritic_folding(self, normalizer):
        """Test diacritic folding."""
        text = "café naïve résumé"
        normalized = normalizer.normalize(text)
        assert "cafe naive resume" in normalized
    
    def test_punctuation_normalization(self, normalizer):
        """Test punctuation normalization."""
        text = "\u201cHello\u201d \u2018world\u2019 \u2013 test"  # Smart quotes and em dash
        normalized = normalizer.normalize(text)
        assert '"Hello" \'world\' - test' in normalized
    
    def test_repeated_character_compression(self, normalizer):
        """Test compression of repeated characters."""
        text = "hellooooo worlddddd"
        normalized = normalizer.normalize(text)
        assert "hellooo worlddd" in normalized  # Max 3 repetitions
    
    def test_whitespace_cleaning(self, normalizer):
        """Test whitespace normalization."""
        text = "  hello    world  \n\t  "
        normalized = normalizer.normalize(text)
        assert normalized == "hello world"
    
    def test_empty_text_handling(self, normalizer):
        """Test handling of empty text."""
        assert normalizer.normalize("") == ""
        assert normalizer.normalize(None) is None


class TestMultilingualTokenizer:
    """Test multilingual tokenization functionality."""
    
    @pytest.fixture
    def tokenizer_config(self):
        """Default tokenizer configuration."""
        return {
            "emoji_aware": True,
            "script_aware": True,
            "preserve_spans": True
        }
    
    @pytest.fixture
    def tokenizer(self, tokenizer_config):
        """Create tokenizer instance."""
        return MultilingualTokenizer(tokenizer_config)
    
    def test_basic_tokenization(self, tokenizer):
        """Test basic English tokenization."""
        text = "Hello world, how are you?"
        tokens = tokenizer.tokenize(text)
        
        expected = ["Hello", "world", ",", "how", "are", "you", "?"]
        assert tokens == expected
    
    def test_tokenization_with_spans(self, tokenizer):
        """Test tokenization with span preservation."""
        text = "Hello world"
        token_spans = tokenizer.tokenize_with_spans(text)
        
        assert len(token_spans) == 2
        assert token_spans[0].text == "Hello"
        assert token_spans[0].start == 0
        assert token_spans[0].end == 5
        assert token_spans[1].text == "world"
        assert token_spans[1].start == 6
        assert token_spans[1].end == 11
    
    def test_emoji_tokenization(self, tokenizer):
        """Test emoji-aware tokenization."""
        text = "Hello 😊 world 🌍"
        token_spans = tokenizer.tokenize_with_spans(text)
        
        emoji_tokens = [span for span in token_spans if span.token_type == "emoji"]
        assert len(emoji_tokens) == 2
        assert emoji_tokens[0].text == "😊"
        assert emoji_tokens[1].text == "🌍"
    
    def test_multilingual_tokenization(self, tokenizer):
        """Test tokenization of multilingual text."""
        text = "Hello नमस्ते world"
        tokens = tokenizer.tokenize(text)
        
        assert "Hello" in tokens
        assert "नमस्ते" in tokens
        assert "world" in tokens
    
    def test_script_detection(self, tokenizer):
        """Test script-based token grouping."""
        text = "Hello नमस्ते world বিশ্ব"
        script_tokens = tokenizer.get_script_tokens(text)
        
        assert "latin" in script_tokens
        assert "devanagari" in script_tokens
        assert "bengali" in script_tokens
        
        assert "Hello" in script_tokens["latin"]
        assert "world" in script_tokens["latin"]
        assert "নমস্ते" in script_tokens.get("devanagari", []) or "नमस्ते" in script_tokens.get("devanagari", [])
    
    def test_number_tokenization(self, tokenizer):
        """Test number tokenization."""
        text = "Price is $123.45 or ₹1,234.56"
        token_spans = tokenizer.tokenize_with_spans(text)
        
        number_tokens = [span for span in token_spans if span.token_type == "number"]
        assert len(number_tokens) >= 2
    
    def test_punctuation_tokenization(self, tokenizer):
        """Test punctuation tokenization."""
        text = "Hello, world! How are you?"
        token_spans = tokenizer.tokenize_with_spans(text)
        
        punct_tokens = [span for span in token_spans if span.token_type == "punctuation"]
        punct_texts = [span.text for span in punct_tokens]
        
        assert "," in punct_texts
        assert "!" in punct_texts
        assert "?" in punct_texts
    
    def test_empty_text_tokenization(self, tokenizer):
        """Test tokenization of empty text."""
        assert tokenizer.tokenize("") == []
        assert tokenizer.tokenize_with_spans("") == []
    
    def test_span_mapping_creation(self, tokenizer):
        """Test creation of span mappings."""
        text = "Hello world"
        tokens = ["Hello", "world"]
        spans = tokenizer.create_span_mapping(text, tokens)
        
        assert len(spans) == 2
        assert spans[0] == (0, 5)  # "Hello"
        assert spans[1] == (6, 11)  # "world"


class TestObfuscationHandler:
    """Test obfuscation detection and normalization."""
    
    @pytest.fixture
    def obfuscation_config(self):
        """Default obfuscation handler configuration."""
        return {
            "leet_speak_detection": True,
            "phonetic_normalization": True,
            "elongation_detection": True,
            "homoglyph_normalization": True
        }
    
    @pytest.fixture
    def handler(self, obfuscation_config):
        """Create obfuscation handler instance."""
        return ObfuscationHandler(obfuscation_config)
    
    def test_leet_speak_normalization(self, handler):
        """Test leet speak normalization."""
        text = "h3ll0 w0rld @nd 5h1t"
        normalized, obfuscation_map = handler.normalize_obfuscations(text)
        
        assert "hello world and shit" in normalized.lower()
        assert len(obfuscation_map) > 0
        assert "3" in obfuscation_map or "h3ll0" in obfuscation_map
    
    def test_elongation_normalization(self, handler):
        """Test elongation normalization."""
        text = "hellooooo worlddddd"
        normalized, obfuscation_map = handler.normalize_obfuscations(text)
        
        # Should compress to max 2 characters (based on default config)
        assert "hellooo" in normalized or "helloo" in normalized
        assert "worlddd" in normalized or "worldd" in normalized
        assert len(obfuscation_map) > 0
    
    def test_character_substitution_normalization(self, handler):
        """Test character substitution normalization."""
        text = "h@ll0 w0rld"
        normalized, obfuscation_map = handler.normalize_obfuscations(text)
        
        assert "hello world" in normalized.lower()
        assert "@" in obfuscation_map or "0" in obfuscation_map
    
    def test_obfuscation_technique_detection(self, handler):
        """Test detection of obfuscation techniques."""
        # Test leet speak detection
        leet_text = "h3ll0 w0rld"
        techniques = handler.detect_obfuscation_techniques(leet_text)
        assert "leet_speak" in techniques
        
        # Test elongation detection
        elongated_text = "hellooooo"
        techniques = handler.detect_obfuscation_techniques(elongated_text)
        assert "elongation" in techniques
        
        # Test clean text
        clean_text = "hello world"
        techniques = handler.detect_obfuscation_techniques(clean_text)
        assert len(techniques) == 0 or "leet_speak" not in techniques
    
    def test_obfuscation_variant_generation(self, handler):
        """Test generation of obfuscation variants."""
        word = "hello"
        variants = handler.generate_obfuscation_variants(word)
        
        assert "hello" in variants  # Original word
        assert len(variants) > 1  # Should have variants
        
        # Check for leet variants
        leet_variants = [v for v in variants if any(c in v for c in "3@!0")]
        assert len(leet_variants) > 0
    
    def test_empty_text_handling(self, handler):
        """Test handling of empty text."""
        normalized, obfuscation_map = handler.normalize_obfuscations("")
        assert normalized == ""
        assert len(obfuscation_map) == 0


class TestTextPreprocessor:
    """Test integrated text preprocessing pipeline."""
    
    @pytest.fixture
    def preprocessor_config(self):
        """Configuration for text preprocessor."""
        return {
            "preprocessing": {
                "language_detection": {
                    "supported_languages": ["en", "hi", "hi-Latn"],
                    "confidence_threshold": 0.7
                },
                "normalization": {
                    "unicode_form": "NFKC",
                    "strip_zero_width": True,
                    "fold_diacritics": True,
                    "normalize_punctuation": True,
                    "compress_repeated_chars": True,
                    "max_char_repetition": 3
                },
                "transliteration": {
                    "enabled": True
                },
                "obfuscation": {
                    "leet_speak_detection": True,
                    "phonetic_normalization": True,
                    "elongation_detection": True
                },
                "tokenization": {
                    "emoji_aware": True,
                    "script_aware": True,
                    "preserve_spans": True
                },
                "pii_masking": {
                    "enabled": False
                }
            }
        }
    
    @pytest.fixture
    def preprocessor(self, preprocessor_config):
        """Create preprocessor instance with mocked dependencies."""
        with patch('core.preprocessing.LanguageIdentifier'), \
             patch('core.preprocessing.TransliterationEngine'), \
             patch('core.preprocessing.PIIMasker'):
            
            preprocessor = TextPreprocessor(preprocessor_config)
            
            # Mock language identifier
            preprocessor.language_identifier.detect_languages = Mock(
                return_value=[LanguageDetection("en", 0.9, 100.0)]
            )
            
            # Mock transliteration engine
            preprocessor.transliteration_engine.transliterate_to_native = Mock(
                return_value={}
            )
            preprocessor.transliteration_engine.transliterate_to_roman = Mock(
                return_value={}
            )
            
            # Mock PII masker
            preprocessor.pii_masker.enabled = False
            preprocessor.pii_masker.mask_pii = Mock(
                return_value=("text", False, {})
            )
            
            return preprocessor
    
    @pytest.mark.asyncio
    async def test_basic_preprocessing(self, preprocessor):
        """Test basic preprocessing pipeline."""
        text = "Hello world! This is a test."
        
        result = await preprocessor.process(text)
        
        assert isinstance(result, ProcessedText)
        assert result.original_text == text
        assert len(result.normalized_text) > 0
        assert len(result.detected_languages) > 0
        assert len(result.tokens) > 0
        assert isinstance(result.transliterations, dict)
        assert isinstance(result.obfuscation_map, dict)
    
    @pytest.mark.asyncio
    async def test_obfuscated_text_preprocessing(self, preprocessor):
        """Test preprocessing of obfuscated text."""
        text = "h3ll0 w0rld!!! th1s 1s @ t3st..."
        
        result = await preprocessor.process(text)
        
        assert isinstance(result, ProcessedText)
        assert result.original_text == text
        # Should have normalized obfuscations
        assert "hello world" in result.normalized_text.lower()
        # Should have obfuscation mappings
        assert len(result.obfuscation_map) > 0
    
    @pytest.mark.asyncio
    async def test_multilingual_preprocessing(self, preprocessor):
        """Test preprocessing of multilingual text."""
        text = "Hello नमस्ते world"
        
        # Mock language detection for multilingual content
        preprocessor.language_identifier.detect_languages = Mock(
            return_value=[
                LanguageDetection("en", 0.7, 60.0),
                LanguageDetection("hi", 0.8, 40.0)
            ]
        )
        
        result = await preprocessor.process(text)
        
        assert isinstance(result, ProcessedText)
        assert len(result.detected_languages) >= 1
        assert any(lang.code in ["en", "hi"] for lang in result.detected_languages)
    
    @pytest.mark.asyncio
    async def test_preprocessing_with_language_hints(self, preprocessor):
        """Test preprocessing with language hints."""
        text = "Hello world"
        language_hints = ["en", "hi"]
        
        result = await preprocessor.process(text, language_hints)
        
        # Verify language hints were passed to language identifier
        preprocessor.language_identifier.detect_languages.assert_called_with(
            text, language_hints
        )
    
    @pytest.mark.asyncio
    async def test_preprocessing_error_handling(self, preprocessor):
        """Test error handling in preprocessing."""
        # Mock an error in normalization
        preprocessor.text_normalizer.normalize = Mock(side_effect=Exception("Test error"))
        
        text = "Hello world"
        result = await preprocessor.process(text)
        
        # Should return minimal processed text on error
        assert isinstance(result, ProcessedText)
        assert result.original_text == text
        assert len(result.detected_languages) > 0  # Should have fallback language
    
    @pytest.mark.asyncio
    async def test_preprocessing_statistics(self, preprocessor):
        """Test preprocessing statistics generation."""
        text = "Hello world! This is a test with obfuscations: h3ll0."
        
        result = await preprocessor.process(text)
        stats = preprocessor.get_preprocessing_stats(result)
        
        assert isinstance(stats, dict)
        assert "original_length" in stats
        assert "normalized_length" in stats
        assert "detected_languages" in stats
        assert "token_count" in stats
        assert "obfuscations_found" in stats
        assert "transliterations_found" in stats
        assert "pii_masked" in stats
        
        assert stats["original_length"] == len(text)
        assert stats["token_count"] > 0
    
    @pytest.mark.asyncio
    async def test_empty_text_preprocessing(self, preprocessor):
        """Test preprocessing of empty text."""
        result = await preprocessor.process("")
        
        assert isinstance(result, ProcessedText)
        assert result.original_text == ""
        assert len(result.detected_languages) > 0  # Should have fallback


if __name__ == "__main__":
    pytest.main([__file__])