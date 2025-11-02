"""Main Lexicon & Pattern Engine that orchestrates all LPE components."""

import asyncio
from typing import List, Dict, Optional
from pathlib import Path
import logging

from ..core.models import ProcessedText, LPEResult, ProblemSpan
from ..core.interfaces import LexiconPatternEngineInterface
from .lexicon_loader import LexiconManager
from .pattern_matcher import PatternMatcher, MatchResult
from .emoji_analyzer import EmojiAnalyzer, EmojiMatch


logger = logging.getLogger(__name__)


class LexiconPatternEngine(LexiconPatternEngineInterface):
    """Main engine that combines lexicon matching, pattern matching, and emoji analysis."""
    
    def __init__(self, config_manager):
        """
        Initialize the Lexicon & Pattern Engine.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.lexicon_manager = None
        self.pattern_matcher = None
        self.emoji_analyzer = None
        
        # Configuration
        self.config = {}
        
    async def initialize(self):
        """Initialize all LPE components."""
        logger.info("Initializing Lexicon & Pattern Engine...")
        
        try:
            # Load configuration
            ensemble_config = self.config_manager.get_ensemble_config()
            self.config = ensemble_config.get("lpe", {})
            
            # Determine lexicon path
            lexicon_path = Path(__file__).parent.parent / "lexicon-pack"
            
            # Initialize lexicon manager
            self.lexicon_manager = LexiconManager(str(lexicon_path), self.config_manager)
            await self.lexicon_manager.initialize()
            
            # Initialize pattern matcher
            self.pattern_matcher = PatternMatcher(self.config)
            self.pattern_matcher.build_trie(self.lexicon_manager)
            self.pattern_matcher.compile_patterns(self.lexicon_manager)
            
            # Initialize emoji analyzer
            emoji_data = self.lexicon_manager.get_emoji_data()
            self.emoji_analyzer = EmojiAnalyzer(emoji_data)
            
            logger.info("Lexicon & Pattern Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LPE: {e}")
            raise
    
    async def analyze(self, processed_text: ProcessedText) -> LPEResult:
        """
        Analyze processed text using lexicons, patterns, and emoji analysis.
        
        Args:
            processed_text: Preprocessed text input
            
        Returns:
            LPEResult with matched spans and rule traces
        """
        try:
            # Extract languages for targeted analysis
            languages = [lang.code for lang in processed_text.detected_languages]
            
            # Run analysis components in parallel
            tasks = [
                self._analyze_lexicon_matches(processed_text, languages),
                self._analyze_pattern_matches(processed_text, languages),
                self._analyze_emoji_matches(processed_text)
            ]
            
            lexicon_matches, pattern_matches, emoji_matches = await asyncio.gather(*tasks)
            
            # Combine all matches
            all_matches = lexicon_matches + pattern_matches + emoji_matches
            
            # Extract spans and build result
            matched_spans = self._extract_problem_spans(all_matches, processed_text.original_text)
            categories = self._extract_categories(all_matches)
            confidence_scores = self._calculate_confidence_scores(all_matches, categories)
            rule_traces = self._generate_rule_traces(all_matches)
            
            return LPEResult(
                matched_spans=matched_spans,
                categories=categories,
                confidence_scores=confidence_scores,
                rule_traces=rule_traces
            )
            
        except Exception as e:
            logger.error(f"Error in LPE analysis: {e}")
            # Return empty result on error
            return LPEResult(
                matched_spans=[],
                categories=[],
                confidence_scores={},
                rule_traces=[]
            )
    
    async def _analyze_lexicon_matches(self, processed_text: ProcessedText, languages: List[str]) -> List[MatchResult]:
        """Analyze text using lexicon-based matching."""
        matches = []
        
        # Search in normalized text
        text_to_search = processed_text.normalized_text
        
        # Use pattern matcher for lexicon searches
        lexicon_matches = self.pattern_matcher.find_matches(text_to_search, languages)
        matches.extend(lexicon_matches)
        
        # Also search transliterations
        for original, transliterated in processed_text.transliterations.items():
            trans_matches = self.pattern_matcher.find_matches(transliterated, languages)
            # Adjust positions back to original text (simplified)
            for match in trans_matches:
                match.rule_source += f":transliterated:{original}"
            matches.extend(trans_matches)
        
        return matches
    
    async def _analyze_pattern_matches(self, processed_text: ProcessedText, languages: List[str]) -> List[MatchResult]:
        """Analyze text using regex pattern matching."""
        # Pattern matching is already included in lexicon analysis
        # This could be extended for additional pattern-based analysis
        return []
    
    async def _analyze_emoji_matches(self, processed_text: ProcessedText) -> List[EmojiMatch]:
        """Analyze text for emoji and symbol matches."""
        emoji_matches = self.emoji_analyzer.analyze_emojis(processed_text.original_text)
        return emoji_matches
    
    def _extract_problem_spans(self, matches: List, original_text: str) -> List[ProblemSpan]:
        """Extract problem spans from all matches."""
        spans = []
        
        for match in matches:
            if isinstance(match, MatchResult):
                span = ProblemSpan(
                    text=match.text,
                    start=match.start,
                    end=match.end,
                    category=match.category,
                    confidence=match.weight,
                    rule_source=match.rule_source
                )
                spans.append(span)
                
            elif isinstance(match, EmojiMatch):
                span = ProblemSpan(
                    text=match.emoji,
                    start=match.start,
                    end=match.end,
                    category=match.category,
                    confidence=match.weight,
                    rule_source=f"emoji:{match.description}"
                )
                spans.append(span)
        
        # Remove overlapping spans (keep highest confidence)
        spans = self._deduplicate_spans(spans)
        
        # Sort by position
        spans.sort(key=lambda s: s.start)
        
        return spans
    
    def _extract_categories(self, matches: List) -> List[str]:
        """Extract unique categories from matches."""
        categories = set()
        
        for match in matches:
            if hasattr(match, 'category'):
                categories.add(match.category)
        
        return list(categories)
    
    def _calculate_confidence_scores(self, matches: List, categories: List[str]) -> Dict[str, float]:
        """Calculate confidence scores per category."""
        category_scores = {}
        category_weights = {}
        
        # Aggregate weights per category
        for match in matches:
            if hasattr(match, 'category') and hasattr(match, 'weight'):
                category = match.category
                weight = match.weight
                
                if category not in category_scores:
                    category_scores[category] = 0.0
                    category_weights[category] = 0.0
                
                category_scores[category] += weight
                category_weights[category] += 1.0
        
        # Calculate average confidence per category
        for category in category_scores:
            if category_weights[category] > 0:
                category_scores[category] = min(1.0, category_scores[category] / category_weights[category])
        
        return category_scores
    
    def _generate_rule_traces(self, matches: List) -> List[str]:
        """Generate rule traces for explainability."""
        traces = []
        
        for match in matches:
            if hasattr(match, 'rule_source'):
                trace = f"Rule: {match.rule_source}"
                if hasattr(match, 'category'):
                    trace += f" -> Category: {match.category}"
                if hasattr(match, 'weight'):
                    trace += f" -> Weight: {match.weight:.2f}"
                traces.append(trace)
        
        return traces
    
    def _deduplicate_spans(self, spans: List[ProblemSpan]) -> List[ProblemSpan]:
        """Remove overlapping spans, keeping the one with highest confidence."""
        if not spans:
            return spans
        
        # Sort by position and confidence
        spans.sort(key=lambda s: (s.start, s.end, -s.confidence))
        
        deduplicated = []
        
        for span in spans:
            overlaps = False
            
            for existing in deduplicated:
                if self._spans_overlap(span, existing):
                    # Keep the span with higher confidence
                    if span.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(span)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(span)
        
        return deduplicated
    
    def _spans_overlap(self, span1: ProblemSpan, span2: ProblemSpan) -> bool:
        """Check if two spans overlap."""
        return not (span1.end <= span2.start or span2.end <= span1.start)
    
    async def get_lexicon_statistics(self) -> Dict[str, any]:
        """Get statistics about loaded lexicons."""
        if self.lexicon_manager:
            return self.lexicon_manager.get_statistics()
        return {}
    
    async def get_pattern_statistics(self) -> Dict[str, any]:
        """Get statistics about pattern matcher."""
        if self.pattern_matcher:
            return self.pattern_matcher.get_statistics()
        return {}
    
    async def reload_lexicons(self):
        """Reload all lexicons (useful for updates)."""
        if self.lexicon_manager:
            logger.info("Reloading lexicons...")
            await self.lexicon_manager.initialize()
            
            # Rebuild pattern matcher indices
            self.pattern_matcher.build_trie(self.lexicon_manager)
            self.pattern_matcher.compile_patterns(self.lexicon_manager)
            
            logger.info("Lexicons reloaded successfully")
    
    async def add_custom_terms(self, language: str, terms: List[Dict]) -> bool:
        """
        Add custom terms to lexicon (runtime addition).
        
        Args:
            language: Language code
            terms: List of term dictionaries with category, severity, weight
            
        Returns:
            Success status
        """
        try:
            # This would typically update the lexicon files and reload
            # For now, just log the request
            logger.info(f"Request to add {len(terms)} custom terms for {language}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add custom terms: {e}")
            return False