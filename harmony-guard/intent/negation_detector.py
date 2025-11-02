"""Negation detection and scope analysis for contextual understanding."""

import re
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass


@dataclass
class NegationScope:
    """Represents a negation scope in text."""
    negation_word: str
    start_pos: int
    end_pos: int
    scope_start: int
    scope_end: int
    confidence: float


class NegationDetector:
    """Detects negation patterns and determines their scope in text."""
    
    def __init__(self):
        # English negation patterns
        self.english_negations = {
            'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither',
            'nor', 'none', 'without', 'lack', 'lacking', 'absent', 'missing',
            'refuse', 'deny', 'reject', 'avoid', 'prevent', 'stop', 'cease',
            'quit', 'discontinue', 'cancel', 'eliminate', 'remove'
        }
        
        # Hindi negation patterns (Devanagari and romanized)
        self.hindi_negations = {
            'नहीं', 'न', 'ना', 'मत', 'कभी नहीं', 'कुछ नहीं', 'कोई नहीं',
            'nahi', 'nahin', 'na', 'mat', 'kabhi nahi', 'kuch nahi', 'koi nahi',
            'bilkul nahi', 'बिल्कुल नहीं'
        }
        
        # Negation prefixes and suffixes
        self.negation_prefixes = {'un', 'in', 'im', 'ir', 'il', 'dis', 'mis', 'non', 'anti'}
        self.negation_suffixes = {'less', 'free'}
        
        # Compile regex patterns for efficient matching
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for negation detection."""
        # English negation pattern
        english_pattern = r'\b(?:' + '|'.join(self.english_negations) + r')\b'
        self.english_negation_regex = re.compile(english_pattern, re.IGNORECASE)
        
        # Hindi negation pattern
        hindi_pattern = r'\b(?:' + '|'.join(self.hindi_negations) + r')\b'
        self.hindi_negation_regex = re.compile(hindi_pattern, re.IGNORECASE)
        
        # Contraction patterns (don't, won't, can't, etc.)
        contraction_pattern = r"\b\w+n't\b"
        self.contraction_regex = re.compile(contraction_pattern, re.IGNORECASE)
        
        # Prefix negation pattern
        prefix_pattern = r'\b(?:' + '|'.join(self.negation_prefixes) + r')\w+'
        self.prefix_negation_regex = re.compile(prefix_pattern, re.IGNORECASE)
        
        # Suffix negation pattern  
        suffix_pattern = r'\w+(?:' + '|'.join(self.negation_suffixes) + r')\b'
        self.suffix_negation_regex = re.compile(suffix_pattern, re.IGNORECASE)
    
    def detect_negations(self, text: str, tokens: List[str] = None) -> List[NegationScope]:
        """
        Detect negation patterns and determine their scope.
        
        Args:
            text: Input text to analyze
            tokens: Optional pre-tokenized text
            
        Returns:
            List of NegationScope objects
        """
        negations = []
        
        # Find explicit negation words
        negations.extend(self._find_explicit_negations(text))
        
        # Find contraction negations
        negations.extend(self._find_contraction_negations(text))
        
        # Find morphological negations (prefixes/suffixes)
        negations.extend(self._find_morphological_negations(text))
        
        # Calculate scope for each negation
        for negation in negations:
            self._calculate_scope(negation, text, tokens)
        
        return negations
    
    def _find_explicit_negations(self, text: str) -> List[NegationScope]:
        """Find explicit negation words."""
        negations = []
        
        # English negations
        for match in self.english_negation_regex.finditer(text):
            negations.append(NegationScope(
                negation_word=match.group(),
                start_pos=match.start(),
                end_pos=match.end(),
                scope_start=match.start(),
                scope_end=match.end(),
                confidence=0.9
            ))
        
        # Hindi negations
        for match in self.hindi_negation_regex.finditer(text):
            negations.append(NegationScope(
                negation_word=match.group(),
                start_pos=match.start(),
                end_pos=match.end(),
                scope_start=match.start(),
                scope_end=match.end(),
                confidence=0.9
            ))
        
        return negations
    
    def _find_contraction_negations(self, text: str) -> List[NegationScope]:
        """Find negation contractions like don't, won't, can't."""
        negations = []
        
        for match in self.contraction_regex.finditer(text):
            negations.append(NegationScope(
                negation_word=match.group(),
                start_pos=match.start(),
                end_pos=match.end(),
                scope_start=match.start(),
                scope_end=match.end(),
                confidence=0.95
            ))
        
        return negations
    
    def _find_morphological_negations(self, text: str) -> List[NegationScope]:
        """Find morphological negations (prefixes and suffixes)."""
        negations = []
        
        # Prefix negations
        for match in self.prefix_negation_regex.finditer(text):
            negations.append(NegationScope(
                negation_word=match.group(),
                start_pos=match.start(),
                end_pos=match.end(),
                scope_start=match.start(),
                scope_end=match.end(),
                confidence=0.7  # Lower confidence for morphological
            ))
        
        # Suffix negations
        for match in self.suffix_negation_regex.finditer(text):
            negations.append(NegationScope(
                negation_word=match.group(),
                start_pos=match.start(),
                end_pos=match.end(),
                scope_start=match.start(),
                scope_end=match.end(),
                confidence=0.7
            ))
        
        return negations
    
    def _calculate_scope(self, negation: NegationScope, text: str, tokens: List[str] = None):
        """Calculate the scope of negation influence."""
        # Simple heuristic: negation affects the next 5-10 words or until punctuation
        words_after_negation = 8
        
        # Find word boundaries after negation
        remaining_text = text[negation.end_pos:]
        words = re.findall(r'\b\w+\b', remaining_text)
        
        if len(words) > 0:
            # Take up to words_after_negation words or until punctuation
            scope_words = words[:min(words_after_negation, len(words))]
            
            # Find the end position of the scope
            last_word = scope_words[-1]
            last_word_pos = remaining_text.find(last_word)
            if last_word_pos != -1:
                negation.scope_end = negation.end_pos + last_word_pos + len(last_word)
        
        # Check for punctuation that might limit scope
        punctuation_match = re.search(r'[.!?;,]', remaining_text)
        if punctuation_match and punctuation_match.start() < (negation.scope_end - negation.end_pos):
            negation.scope_end = negation.end_pos + punctuation_match.start()
    
    def is_in_negation_scope(self, span_start: int, span_end: int, negations: List[NegationScope]) -> Tuple[bool, float]:
        """
        Check if a text span is within any negation scope.
        
        Args:
            span_start: Start position of the span
            span_end: End position of the span
            negations: List of detected negations
            
        Returns:
            Tuple of (is_negated, confidence)
        """
        max_confidence = 0.0
        is_negated = False
        
        for negation in negations:
            # Check if span overlaps with negation scope
            if (span_start >= negation.scope_start and span_start <= negation.scope_end) or \
               (span_end >= negation.scope_start and span_end <= negation.scope_end) or \
               (span_start <= negation.scope_start and span_end >= negation.scope_end):
                is_negated = True
                max_confidence = max(max_confidence, negation.confidence)
        
        return is_negated, max_confidence
    
    def get_negation_context(self, text: str, problem_spans: List) -> Dict[str, any]:
        """
        Analyze negation context for problem spans.
        
        Args:
            text: Input text
            problem_spans: List of problem spans to check
            
        Returns:
            Dictionary with negation analysis results
        """
        negations = self.detect_negations(text)
        
        negated_spans = []
        negation_confidence = 0.0
        
        for span in problem_spans:
            is_negated, confidence = self.is_in_negation_scope(
                span.start, span.end, negations
            )
            
            if is_negated:
                negated_spans.append({
                    'span': span,
                    'negation_confidence': confidence
                })
                negation_confidence = max(negation_confidence, confidence)
        
        return {
            'has_negation': len(negations) > 0,
            'negation_count': len(negations),
            'negated_spans': negated_spans,
            'overall_negation_confidence': negation_confidence,
            'detected_negations': negations
        }