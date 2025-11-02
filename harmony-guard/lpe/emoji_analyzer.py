"""Emoji and symbol analysis for content moderation."""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class EmojiMatch:
    """Represents an emoji match result."""
    emoji: str
    start: int
    end: int
    category: str
    severity: str
    weight: float
    description: str
    context_required: bool = False


@dataclass
class SymbolMatch:
    """Represents a symbol/kaomoji match result."""
    symbol: str
    start: int
    end: int
    category: str
    severity: str
    weight: float
    description: str


class EmojiAnalyzer:
    """Analyzes emojis and symbols for inappropriate content."""
    
    def __init__(self, emoji_lexicon: Dict = None):
        """
        Initialize emoji analyzer.
        
        Args:
            emoji_lexicon: Emoji lexicon data from YAML
        """
        self.emoji_lexicon = emoji_lexicon or {}
        
        # Build emoji maps
        self.offensive_emojis = self._build_offensive_emoji_map()
        self.anger_emojis = self._build_anger_emoji_map()
        self.kaomoji_patterns = self._build_kaomoji_patterns()
        self.combination_patterns = self._build_combination_patterns()
        
        # Emoji detection patterns
        self.emoji_pattern = self._build_emoji_pattern()
        self.kaomoji_pattern = self._build_kaomoji_pattern()
    
    def analyze_emojis(self, text: str) -> List[EmojiMatch]:
        """
        Analyze text for inappropriate emojis and symbols.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of emoji matches found
        """
        matches = []
        
        # Find individual emojis
        emoji_matches = self._find_individual_emojis(text)
        matches.extend(emoji_matches)
        
        # Find kaomoji patterns
        kaomoji_matches = self._find_kaomoji(text)
        matches.extend(kaomoji_matches)
        
        # Find combination patterns (emoji + text)
        combination_matches = self._find_combination_patterns(text)
        matches.extend(combination_matches)
        
        # Remove duplicates and sort by position
        matches = self._deduplicate_emoji_matches(matches)
        matches.sort(key=lambda m: m.start)
        
        return matches
    
    def get_emoji_sentiment(self, emoji: str) -> Optional[Tuple[str, str, float]]:
        """
        Get sentiment information for a specific emoji.
        
        Args:
            emoji: Emoji character to analyze
            
        Returns:
            Tuple of (category, severity, weight) or None if not found
        """
        # Check offensive emojis
        if emoji in self.offensive_emojis:
            data = self.offensive_emojis[emoji]
            return data["category"], data["severity"], data["weight"]
        
        # Check anger emojis
        if emoji in self.anger_emojis:
            data = self.anger_emojis[emoji]
            return data["category"], data["severity"], data["weight"]
        
        return None
    
    def detect_emoji_context(self, text: str, emoji_position: int, window_size: int = 10) -> Dict[str, any]:
        """
        Analyze context around an emoji to determine appropriateness.
        
        Args:
            text: Full text
            emoji_position: Position of emoji in text
            window_size: Number of characters to analyze around emoji
            
        Returns:
            Context analysis results
        """
        start = max(0, emoji_position - window_size)
        end = min(len(text), emoji_position + window_size)
        context = text[start:end].lower()
        
        # Analyze context for inappropriate combinations
        context_flags = {
            "has_profanity": self._has_profanity_context(context),
            "has_violence": self._has_violence_context(context),
            "has_sexual_content": self._has_sexual_context(context),
            "has_harassment": self._has_harassment_context(context)
        }
        
        return {
            "context_text": context,
            "flags": context_flags,
            "risk_score": sum(context_flags.values()) / len(context_flags)
        }
    
    def _find_individual_emojis(self, text: str) -> List[EmojiMatch]:
        """Find individual emoji matches in text."""
        matches = []
        
        for match in self.emoji_pattern.finditer(text):
            emoji = match.group(0)
            sentiment = self.get_emoji_sentiment(emoji)
            
            if sentiment:
                category, severity, weight = sentiment
                
                emoji_match = EmojiMatch(
                    emoji=emoji,
                    start=match.start(),
                    end=match.end(),
                    category=category,
                    severity=severity,
                    weight=weight,
                    description=self._get_emoji_description(emoji),
                    context_required=self._emoji_needs_context(emoji)
                )
                matches.append(emoji_match)
        
        return matches
    
    def _find_kaomoji(self, text: str) -> List[EmojiMatch]:
        """Find kaomoji (text-based emoticons) in text."""
        matches = []
        
        for pattern_data in self.kaomoji_patterns:
            pattern = pattern_data["pattern"]
            category = pattern_data["category"]
            severity = pattern_data["severity"]
            weight = pattern_data["weight"]
            description = pattern_data["description"]
            
            for match in re.finditer(re.escape(pattern), text):
                emoji_match = EmojiMatch(
                    emoji=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    category=category,
                    severity=severity,
                    weight=weight,
                    description=description
                )
                matches.append(emoji_match)
        
        return matches
    
    def _find_combination_patterns(self, text: str) -> List[EmojiMatch]:
        """Find emoji + text combination patterns."""
        matches = []
        
        for pattern_data in self.combination_patterns:
            pattern = pattern_data["pattern"]
            category = pattern_data["category"]
            severity = pattern_data["severity"]
            weight = pattern_data["weight"]
            
            try:
                compiled_pattern = re.compile(pattern, re.IGNORECASE | re.UNICODE)
                
                for match in compiled_pattern.finditer(text):
                    emoji_match = EmojiMatch(
                        emoji=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        category=category,
                        severity=severity,
                        weight=weight,
                        description=f"Combination pattern: {pattern}"
                    )
                    matches.append(emoji_match)
                    
            except re.error as e:
                logger.warning(f"Failed to compile emoji combination pattern '{pattern}': {e}")
        
        return matches
    
    def _build_offensive_emoji_map(self) -> Dict[str, Dict]:
        """Build map of offensive emojis from lexicon."""
        offensive_map = {}
        
        if "offensive_emojis" in self.emoji_lexicon:
            for emoji_data in self.emoji_lexicon["offensive_emojis"]:
                emoji = emoji_data["emoji"]
                offensive_map[emoji] = {
                    "category": emoji_data["category"],
                    "severity": emoji_data["severity"],
                    "weight": emoji_data["weight"],
                    "description": emoji_data["description"],
                    "context_required": emoji_data.get("context_required", False)
                }
        
        return offensive_map
    
    def _build_anger_emoji_map(self) -> Dict[str, Dict]:
        """Build map of anger/violence emojis from lexicon."""
        anger_map = {}
        
        if "anger_emojis" in self.emoji_lexicon:
            for emoji_data in self.emoji_lexicon["anger_emojis"]:
                emoji = emoji_data["emoji"]
                anger_map[emoji] = {
                    "category": emoji_data["category"],
                    "severity": emoji_data["severity"],
                    "weight": emoji_data["weight"],
                    "description": emoji_data["description"],
                    "context_required": emoji_data.get("context_required", False)
                }
        
        return anger_map
    
    def _build_kaomoji_patterns(self) -> List[Dict]:
        """Build kaomoji patterns from lexicon."""
        patterns = []
        
        if "kaomoji" in self.emoji_lexicon:
            for kaomoji_data in self.emoji_lexicon["kaomoji"]:
                patterns.append({
                    "pattern": kaomoji_data["pattern"],
                    "category": kaomoji_data["category"],
                    "severity": kaomoji_data["severity"],
                    "weight": kaomoji_data["weight"],
                    "description": kaomoji_data["description"]
                })
        
        return patterns
    
    def _build_combination_patterns(self) -> List[Dict]:
        """Build emoji + text combination patterns from lexicon."""
        patterns = []
        
        if "combination_patterns" in self.emoji_lexicon:
            for pattern_data in self.emoji_lexicon["combination_patterns"]:
                patterns.append({
                    "pattern": pattern_data["pattern"],
                    "category": pattern_data["category"],
                    "severity": pattern_data["severity"],
                    "weight": pattern_data["weight"]
                })
        
        return patterns
    
    def _build_emoji_pattern(self) -> re.Pattern:
        """Build regex pattern to match emojis."""
        # Unicode ranges for emojis
        emoji_ranges = [
            r'\U0001F600-\U0001F64F',  # Emoticons
            r'\U0001F300-\U0001F5FF',  # Misc Symbols and Pictographs
            r'\U0001F680-\U0001F6FF',  # Transport and Map
            r'\U0001F1E0-\U0001F1FF',  # Regional Indicator Symbols
            r'\U00002600-\U000026FF',  # Miscellaneous Symbols
            r'\U00002700-\U000027BF',  # Dingbats
            r'\U0001F900-\U0001F9FF',  # Supplemental Symbols and Pictographs
        ]
        
        pattern = '[' + ''.join(emoji_ranges) + ']+'
        return re.compile(pattern, re.UNICODE)
    
    def _build_kaomoji_pattern(self) -> re.Pattern:
        """Build regex pattern to match kaomoji."""
        # Basic kaomoji pattern (this is simplified)
        kaomoji_chars = r'[(){}[\]<>^_=~*+\-|\\/:;\'"`.,!?@#$%&]'
        pattern = f'{kaomoji_chars}{{2,}}'
        return re.compile(pattern)
    
    def _get_emoji_description(self, emoji: str) -> str:
        """Get description for an emoji."""
        # Check offensive emojis
        if emoji in self.offensive_emojis:
            return self.offensive_emojis[emoji]["description"]
        
        # Check anger emojis
        if emoji in self.anger_emojis:
            return self.anger_emojis[emoji]["description"]
        
        return f"Emoji: {emoji}"
    
    def _emoji_needs_context(self, emoji: str) -> bool:
        """Check if emoji requires context analysis."""
        if emoji in self.offensive_emojis:
            return self.offensive_emojis[emoji].get("context_required", False)
        
        if emoji in self.anger_emojis:
            return self.anger_emojis[emoji].get("context_required", False)
        
        return False
    
    def _has_profanity_context(self, context: str) -> bool:
        """Check if context contains profanity indicators."""
        profanity_indicators = [
            "fuck", "shit", "damn", "hell", "ass", "bitch", 
            "bastard", "crap", "piss", "bloody"
        ]
        return any(word in context for word in profanity_indicators)
    
    def _has_violence_context(self, context: str) -> bool:
        """Check if context contains violence indicators."""
        violence_indicators = [
            "kill", "murder", "die", "death", "hurt", "pain",
            "beat", "fight", "attack", "destroy", "violence"
        ]
        return any(word in context for word in violence_indicators)
    
    def _has_sexual_context(self, context: str) -> bool:
        """Check if context contains sexual content indicators."""
        sexual_indicators = [
            "sex", "sexual", "porn", "nude", "naked", "breast",
            "penis", "vagina", "orgasm", "masturbate", "horny"
        ]
        return any(word in context for word in sexual_indicators)
    
    def _has_harassment_context(self, context: str) -> bool:
        """Check if context contains harassment indicators."""
        harassment_indicators = [
            "hate", "stupid", "idiot", "loser", "ugly", "fat",
            "worthless", "pathetic", "disgusting", "freak"
        ]
        return any(word in context for word in harassment_indicators)
    
    def _deduplicate_emoji_matches(self, matches: List[EmojiMatch]) -> List[EmojiMatch]:
        """Remove duplicate emoji matches."""
        if not matches:
            return matches
        
        # Sort by position and weight
        matches.sort(key=lambda m: (m.start, m.end, -m.weight))
        
        deduplicated = []
        
        for match in matches:
            # Check for exact position overlap
            overlaps = False
            
            for existing in deduplicated:
                if (match.start == existing.start and 
                    match.end == existing.end):
                    # Keep the match with higher weight
                    if match.weight > existing.weight:
                        deduplicated.remove(existing)
                        deduplicated.append(match)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(match)
        
        return deduplicated
    
    def get_emoji_statistics(self, text: str) -> Dict[str, any]:
        """Get emoji usage statistics for text."""
        emoji_matches = self.analyze_emojis(text)
        
        stats = {
            "total_emojis": len(emoji_matches),
            "offensive_emojis": 0,
            "anger_emojis": 0,
            "kaomoji": 0,
            "categories": {},
            "severity_levels": {}
        }
        
        for match in emoji_matches:
            # Count by type
            if match.emoji in self.offensive_emojis:
                stats["offensive_emojis"] += 1
            elif match.emoji in self.anger_emojis:
                stats["anger_emojis"] += 1
            elif any(match.emoji == p["pattern"] for p in self.kaomoji_patterns):
                stats["kaomoji"] += 1
            
            # Count by category
            if match.category not in stats["categories"]:
                stats["categories"][match.category] = 0
            stats["categories"][match.category] += 1
            
            # Count by severity
            if match.severity not in stats["severity_levels"]:
                stats["severity_levels"][match.severity] = 0
            stats["severity_levels"][match.severity] += 1
        
        return stats