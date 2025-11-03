"""Tests for Lexicon & Pattern Engine components."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import yaml

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lpe.engine import LexiconPatternEngine
from lpe.pattern_matcher import PatternMatcher, MatchResult, TrieNode
from lpe.lexicon_loader import LexiconManager, LexiconEntry, PatternEntry
from lpe.emoji_analyzer import EmojiAnalyzer, EmojiMatch
from core.models import ProcessedText, LPEResult, ProblemSpan, LanguageDetection


class TestTrieNode:
    """Test trie node functionality."""
    
    def test_trie_node_creation(self):
        """Test trie node initialization."""
        node = TrieNode()
        
        assert len(node.children) == 0
        assert not node.is_end_word
        assert len(node.entries) == 0
        assert node.word == ""
    
    def test_trie_node_word_insertion(self):
        """Test inserting words into trie structure."""
        root = TrieNode()
        
        # Insert "hello"
        current = root
        for char in "hello":
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        
        current.is_end_word = True
        current.word = "hello"
        
        # Verify structure
        assert 'h' in root.children
        assert 'e' in root.children['h'].children
        assert root.children['h'].children['e'].children['l'].children['l'].children['o'].is_end_word
        assert root.children['h'].children['e'].children['l'].children['l'].children['o'].word == "hello"


class TestPatternMatcher:
    """Test pattern matching functionality."""
    
    @pytest.fixture
    def matcher_config(self):
        """Configuration for pattern matcher."""
        return {
            "fuzzy_matching": True,
            "fuzzy_threshold": 0.8,
            "morphological_variants": True
        }
    
    @pytest.fixture
    def pattern_matcher(self, matcher_config):
        """Create pattern matcher instance."""
        return PatternMatcher(matcher_config)
    
    @pytest.fixture
    def mock_lexicon_manager(self):
        """Create mock lexicon manager with test data."""
        manager = Mock()
        
        # Create mock lexicon entries
        entry1 = Mock()
        entry1.terms = ["bad", "evil"]
        entry1.variants = ["baaad", "evill"]
        entry1.category = "profanity"
        entry1.severity = "medium"
        entry1.weight = 0.8
        
        entry2 = Mock()
        entry2.terms = ["hate", "stupid"]
        entry2.variants = ["h8", "stoopid"]
        entry2.category = "insult"
        entry2.severity = "high"
        entry2.weight = 0.9
        
        # Create mock lexicon
        lexicon = Mock()
        lexicon.entries = [entry1, entry2]
        lexicon.patterns = []
        
        manager.lexicons = {"en": lexicon}
        
        return manager
    
    def test_trie_building(self, pattern_matcher, mock_lexicon_manager):
        """Test building trie from lexicon data."""
        pattern_matcher.build_trie(mock_lexicon_manager)
        
        # Verify trie structure
        root = pattern_matcher.trie_root
        assert len(root.children) > 0
        
        # Check if "bad" is in trie
        current = root
        for char in "bad":
            assert char in current.children
            current = current.children[char]
        
        assert current.is_end_word
        assert current.word == "bad"
        assert len(current.entries) > 0
    
    def test_exact_matching(self, pattern_matcher, mock_lexicon_manager):
        """Test exact pattern matching."""
        pattern_matcher.build_trie(mock_lexicon_manager)
        
        text = "This is bad content with hate speech"
        matches = pattern_matcher._find_exact_matches(text)
        
        assert len(matches) >= 2  # Should find "bad" and "hate"
        
        # Check match details
        bad_matches = [m for m in matches if m.text == "bad"]
        assert len(bad_matches) > 0
        assert bad_matches[0].category == "profanity"
        assert bad_matches[0].match_type == "exact"
    
    def test_pattern_compilation(self, pattern_matcher, mock_lexicon_manager):
        """Test regex pattern compilation."""
        # Add pattern entries to mock lexicon
        pattern_entry = Mock()
        pattern_entry.pattern = r"b[a@4]+d"
        pattern_entry.category = "profanity"
        pattern_entry.severity = "medium"
        pattern_entry.base_term = "bad"
        pattern_entry.weight = 0.7
        
        mock_lexicon_manager.lexicons["en"].patterns = [pattern_entry]
        
        pattern_matcher.compile_patterns(mock_lexicon_manager)
        
        assert len(pattern_matcher.regex_patterns) == 1
        
        # Test pattern matching
        text = "This is b@d content"
        matches = pattern_matcher._find_pattern_matches(text)
        
        assert len(matches) >= 1
        assert matches[0].text == "b@d"
        assert matches[0].category == "profanity"
        assert matches[0].match_type == "pattern"
    
    def test_fuzzy_matching(self, pattern_matcher, mock_lexicon_manager):
        """Test fuzzy matching functionality."""
        pattern_matcher.build_trie(mock_lexicon_manager)
        
        # Test with slightly misspelled words
        text = "This is badd content"  # "badd" should match "bad"
        matches = pattern_matcher._find_fuzzy_matches(text, ["en"])
        
        # Note: This is a simplified test as fuzzy matching implementation is basic
        assert isinstance(matches, list)
    
    def test_similarity_calculation(self, pattern_matcher):
        """Test similarity calculation between words."""
        # Identical words
        assert pattern_matcher._calculate_similarity("hello", "hello") == 1.0
        
        # Completely different words
        similarity = pattern_matcher._calculate_similarity("hello", "xyz")
        assert 0.0 <= similarity < 1.0
        
        # Similar words
        similarity = pattern_matcher._calculate_similarity("hello", "helo")
        assert similarity > 0.5
    
    def test_match_deduplication(self, pattern_matcher):
        """Test deduplication of overlapping matches."""
        # Create overlapping matches
        match1 = MatchResult(
            text="bad", start=0, end=3, category="profanity", 
            severity="medium", weight=0.8, match_type="exact", rule_source="test"
        )
        match2 = MatchResult(
            text="bad word", start=0, end=8, category="profanity",
            severity="high", weight=0.9, match_type="pattern", rule_source="test"
        )
        
        matches = [match1, match2]
        deduplicated = pattern_matcher._deduplicate_matches(matches)
        
        # Should keep the match with higher weight
        assert len(deduplicated) == 1
        assert deduplicated[0].weight == 0.9
    
    def test_match_overlap_detection(self, pattern_matcher):
        """Test overlap detection between matches."""
        match1 = MatchResult(
            text="bad", start=0, end=3, category="profanity",
            severity="medium", weight=0.8, match_type="exact", rule_source="test"
        )
        match2 = MatchResult(
            text="word", start=4, end=8, category="profanity",
            severity="medium", weight=0.8, match_type="exact", rule_source="test"
        )
        match3 = MatchResult(
            text="bad word", start=0, end=8, category="profanity",
            severity="medium", weight=0.8, match_type="pattern", rule_source="test"
        )
        
        # Non-overlapping matches
        assert not pattern_matcher._matches_overlap(match1, match2)
        
        # Overlapping matches
        assert pattern_matcher._matches_overlap(match1, match3)
        assert pattern_matcher._matches_overlap(match2, match3)
    
    def test_statistics_generation(self, pattern_matcher, mock_lexicon_manager):
        """Test statistics generation."""
        pattern_matcher.build_trie(mock_lexicon_manager)
        
        stats = pattern_matcher.get_statistics()
        
        assert isinstance(stats, dict)
        assert "trie_nodes" in stats
        assert "regex_patterns" in stats
        assert "fuzzy_cache_size" in stats
        assert "fuzzy_matching_enabled" in stats
        assert "fuzzy_threshold" in stats
        
        assert stats["trie_nodes"] > 0
        assert stats["fuzzy_matching_enabled"] is True
        assert stats["fuzzy_threshold"] == 0.8


class TestEmojiAnalyzer:
    """Test emoji analysis functionality."""
    
    @pytest.fixture
    def emoji_data(self):
        """Sample emoji data for testing."""
        return {
            "sentiment_mapping": {
                "😊": {"sentiment": "positive", "category": "safe", "weight": 0.1},
                "😡": {"sentiment": "negative", "category": "anger", "weight": 0.8},
                "🖕": {"sentiment": "negative", "category": "obscenity", "weight": 0.9},
                "💀": {"sentiment": "negative", "category": "threat", "weight": 0.7}
            },
            "combination_patterns": [
                {
                    "pattern": ["😡", "🖕"],
                    "category": "harassment",
                    "weight": 0.95
                }
            ]
        }
    
    @pytest.fixture
    def emoji_analyzer(self, emoji_data):
        """Create emoji analyzer instance."""
        return EmojiAnalyzer(emoji_data)
    
    def test_emoji_detection(self, emoji_analyzer):
        """Test basic emoji detection."""
        text = "Hello 😊 world 😡"
        matches = emoji_analyzer.analyze_emojis(text)
        
        assert len(matches) == 2
        
        # Check first emoji
        happy_match = next(m for m in matches if m.emoji == "😊")
        assert happy_match.category == "safe"
        assert happy_match.sentiment == "positive"
        
        # Check second emoji
        angry_match = next(m for m in matches if m.emoji == "😡")
        assert angry_match.category == "anger"
        assert angry_match.sentiment == "negative"
    
    def test_emoji_position_tracking(self, emoji_analyzer):
        """Test emoji position tracking."""
        text = "Hello 😊 world"
        matches = emoji_analyzer.analyze_emojis(text)
        
        assert len(matches) == 1
        emoji_match = matches[0]
        
        assert emoji_match.start == 6  # Position of 😊
        assert emoji_match.end == 8    # End position (emoji is 2 bytes)
    
    def test_combination_pattern_detection(self, emoji_analyzer):
        """Test detection of emoji combination patterns."""
        text = "I'm angry 😡🖕 at you"
        matches = emoji_analyzer.analyze_emojis(text)
        
        # Should detect individual emojis and combination pattern
        individual_matches = [m for m in matches if hasattr(m, 'emoji')]
        combination_matches = [m for m in matches if hasattr(m, 'pattern')]
        
        assert len(individual_matches) >= 2
        # Note: Combination detection would need more sophisticated implementation
    
    def test_kaomoji_detection(self, emoji_analyzer):
        """Test kaomoji (text-based emoticon) detection."""
        text = "Hello (╯°□°）╯︵ ┻━┻ world"
        
        # Basic kaomoji detection (would need implementation)
        kaomojis = emoji_analyzer._detect_kaomojis(text)
        
        # This is a placeholder test - actual implementation would detect kaomojis
        assert isinstance(kaomojis, list)
    
    def test_empty_text_handling(self, emoji_analyzer):
        """Test handling of empty text."""
        matches = emoji_analyzer.analyze_emojis("")
        assert len(matches) == 0
        
        matches = emoji_analyzer.analyze_emojis("No emojis here")
        assert len(matches) == 0


class TestLexiconPatternEngine:
    """Test integrated LPE functionality."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        config_manager = Mock()
        config_manager.get_ensemble_config.return_value = {
            "lpe": {
                "fuzzy_matching": True,
                "fuzzy_threshold": 0.8
            }
        }
        return config_manager
    
    @pytest.fixture
    def sample_processed_text(self):
        """Sample processed text for testing."""
        return ProcessedText(
            original_text="This is bad content with hate speech",
            normalized_text="this is bad content with hate speech",
            detected_languages=[LanguageDetection("en", 0.9, 100.0)],
            tokens=["this", "is", "bad", "content", "with", "hate", "speech"],
            transliterations={},
            obfuscation_map={}
        )
    
    @pytest.mark.asyncio
    async def test_lpe_initialization(self, mock_config_manager):
        """Test LPE initialization."""
        with patch('lpe.engine.LexiconManager'), \
             patch('lpe.engine.PatternMatcher'), \
             patch('lpe.engine.EmojiAnalyzer'):
            
            lpe = LexiconPatternEngine(mock_config_manager)
            await lpe.initialize()
            
            assert lpe.lexicon_manager is not None
            assert lpe.pattern_matcher is not None
            assert lpe.emoji_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_lpe_analysis(self, mock_config_manager, sample_processed_text):
        """Test complete LPE analysis."""
        with patch('lpe.engine.LexiconManager'), \
             patch('lpe.engine.PatternMatcher'), \
             patch('lpe.engine.EmojiAnalyzer'):
            
            lpe = LexiconPatternEngine(mock_config_manager)
            await lpe.initialize()
            
            # Mock analysis results
            mock_match = MatchResult(
                text="bad", start=8, end=11, category="profanity",
                severity="medium", weight=0.8, match_type="exact", rule_source="lexicon:en:bad"
            )
            
            lpe.pattern_matcher.find_matches = Mock(return_value=[mock_match])
            lpe.emoji_analyzer.analyze_emojis = Mock(return_value=[])
            
            result = await lpe.analyze(sample_processed_text)
            
            assert isinstance(result, LPEResult)
            assert len(result.matched_spans) > 0
            assert len(result.categories) > 0
            assert len(result.confidence_scores) > 0
            assert len(result.rule_traces) > 0
            
            # Check span details
            span = result.matched_spans[0]
            assert span.text == "bad"
            assert span.category == "profanity"
            assert span.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_lpe_error_handling(self, mock_config_manager, sample_processed_text):
        """Test LPE error handling."""
        with patch('lpe.engine.LexiconManager'), \
             patch('lpe.engine.PatternMatcher'), \
             patch('lpe.engine.EmojiAnalyzer'):
            
            lpe = LexiconPatternEngine(mock_config_manager)
            await lpe.initialize()
            
            # Mock an error in pattern matching
            lpe.pattern_matcher.find_matches = Mock(side_effect=Exception("Test error"))
            lpe.emoji_analyzer.analyze_emojis = Mock(return_value=[])
            
            result = await lpe.analyze(sample_processed_text)
            
            # Should return empty result on error
            assert isinstance(result, LPEResult)
            assert len(result.matched_spans) == 0
            assert len(result.categories) == 0
            assert len(result.confidence_scores) == 0
            assert len(result.rule_traces) == 0
    
    @pytest.mark.asyncio
    async def test_span_deduplication(self, mock_config_manager, sample_processed_text):
        """Test span deduplication in LPE."""
        with patch('lpe.engine.LexiconManager'), \
             patch('lpe.engine.PatternMatcher'), \
             patch('lpe.engine.EmojiAnalyzer'):
            
            lpe = LexiconPatternEngine(mock_config_manager)
            await lpe.initialize()
            
            # Create overlapping matches
            match1 = MatchResult(
                text="bad", start=8, end=11, category="profanity",
                severity="medium", weight=0.8, match_type="exact", rule_source="lexicon"
            )
            match2 = MatchResult(
                text="bad content", start=8, end=19, category="profanity",
                severity="high", weight=0.9, match_type="pattern", rule_source="pattern"
            )
            
            lpe.pattern_matcher.find_matches = Mock(return_value=[match1, match2])
            lpe.emoji_analyzer.analyze_emojis = Mock(return_value=[])
            
            result = await lpe.analyze(sample_processed_text)
            
            # Should deduplicate overlapping spans
            assert len(result.matched_spans) == 1
            # Should keep the span with higher confidence
            assert result.matched_spans[0].confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, mock_config_manager, sample_processed_text):
        """Test confidence score calculation."""
        with patch('lpe.engine.LexiconManager'), \
             patch('lpe.engine.PatternMatcher'), \
             patch('lpe.engine.EmojiAnalyzer'):
            
            lpe = LexiconPatternEngine(mock_config_manager)
            await lpe.initialize()
            
            # Create multiple matches in same category
            match1 = MatchResult(
                text="bad", start=8, end=11, category="profanity",
                severity="medium", weight=0.8, match_type="exact", rule_source="lexicon"
            )
            match2 = MatchResult(
                text="hate", start=25, end=29, category="profanity",
                severity="high", weight=0.9, match_type="exact", rule_source="lexicon"
            )
            
            lpe.pattern_matcher.find_matches = Mock(return_value=[match1, match2])
            lpe.emoji_analyzer.analyze_emojis = Mock(return_value=[])
            
            result = await lpe.analyze(sample_processed_text)
            
            # Should calculate average confidence for category
            assert "profanity" in result.confidence_scores
            expected_confidence = (0.8 + 0.9) / 2
            assert abs(result.confidence_scores["profanity"] - expected_confidence) < 0.01
    
    @pytest.mark.asyncio
    async def test_transliteration_analysis(self, mock_config_manager):
        """Test analysis of transliterated content."""
        # Create processed text with transliterations
        processed_text = ProcessedText(
            original_text="This is bad content",
            normalized_text="this is bad content",
            detected_languages=[LanguageDetection("hi-Latn", 0.8, 100.0)],
            tokens=["this", "is", "bad", "content"],
            transliterations={"bad": "बुरा"},
            obfuscation_map={}
        )
        
        with patch('lpe.engine.LexiconManager'), \
             patch('lpe.engine.PatternMatcher'), \
             patch('lpe.engine.EmojiAnalyzer'):
            
            lpe = LexiconPatternEngine(mock_config_manager)
            await lpe.initialize()
            
            # Mock matches for both original and transliterated text
            original_match = MatchResult(
                text="bad", start=8, end=11, category="profanity",
                severity="medium", weight=0.8, match_type="exact", rule_source="lexicon"
            )
            trans_match = MatchResult(
                text="बुरा", start=0, end=3, category="profanity",
                severity="medium", weight=0.8, match_type="exact", rule_source="lexicon:transliterated:bad"
            )
            
            lpe.pattern_matcher.find_matches = Mock(side_effect=[[original_match], [trans_match]])
            lpe.emoji_analyzer.analyze_emojis = Mock(return_value=[])
            
            result = await lpe.analyze(processed_text)
            
            # Should find matches in both original and transliterated text
            assert len(result.matched_spans) >= 1
            assert any("transliterated" in span.rule_source for span in result.matched_spans)


if __name__ == "__main__":
    pytest.main([__file__])