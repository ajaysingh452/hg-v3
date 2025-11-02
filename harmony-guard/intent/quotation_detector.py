"""Quotation and third-party reference detection for contextual analysis."""

import re
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class QuotationSpan:
    """Represents a quoted text span."""
    content: str
    start_pos: int
    end_pos: int
    quote_type: str  # 'direct', 'indirect', 'reported'
    confidence: float
    attribution: str = None  # Who is being quoted


class QuotationDetector:
    """Detects quotations and third-party references in text."""
    
    def __init__(self):
        # Direct quotation patterns
        self.direct_quote_patterns = [
            r'"([^"]*)"',  # Double quotes
            r"'([^']*)'",  # Single quotes
            r'«([^»]*)»',  # French quotes
            r'"([^"]*)"'   # Smart quotes
        ]
        
        # Indirect quotation indicators
        self.indirect_indicators = {
            'english': [
                'said', 'says', 'told', 'tells', 'mentioned', 'stated', 'claimed',
                'reported', 'according to', 'as per', 'quoted', 'cited',
                'alleged', 'supposedly', 'apparently', 'reportedly'
            ],
            'hindi': [
                'कहा', 'बोला', 'बताया', 'कहते हैं', 'के अनुसार', 'के मुताबिक',
                'kaha', 'bola', 'bataya', 'kehte hain', 'ke anusar', 'ke mutabik'
            ]
        }
        
        # Third-party reference patterns
        self.third_party_patterns = [
            r'\b(?:he|she|they|someone|somebody)\s+(?:said|told|mentioned)',
            r'\b(?:according to|as per)\s+\w+',
            r'\b(?:वह|वे|कोई)\s+(?:कहा|बोला|बताया)',
            r'\b(?:के अनुसार|के मुताबिक)\s+\w+'
        ]
        
        # Compile patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Direct quotation patterns
        self.direct_quote_regexes = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.direct_quote_patterns
        ]
        
        # Indirect quotation patterns
        all_indicators = []
        for lang_indicators in self.indirect_indicators.values():
            all_indicators.extend(lang_indicators)
        
        indirect_pattern = r'\b(?:' + '|'.join(all_indicators) + r')\b'
        self.indirect_quote_regex = re.compile(indirect_pattern, re.IGNORECASE)
        
        # Third-party reference patterns
        self.third_party_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.third_party_patterns
        ]
    
    def detect_quotations(self, text: str) -> List[QuotationSpan]:
        """
        Detect all types of quotations in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of QuotationSpan objects
        """
        quotations = []
        
        # Find direct quotations
        quotations.extend(self._find_direct_quotes(text))
        
        # Find indirect quotations
        quotations.extend(self._find_indirect_quotes(text))
        
        # Find third-party references
        quotations.extend(self._find_third_party_references(text))
        
        # Sort by position and remove overlaps
        quotations = self._remove_overlapping_quotes(quotations)
        
        return quotations
    
    def _find_direct_quotes(self, text: str) -> List[QuotationSpan]:
        """Find directly quoted text."""
        quotations = []
        
        for regex in self.direct_quote_regexes:
            for match in regex.finditer(text):
                # Extract the content inside quotes
                content = match.group(1) if match.groups() else match.group()
                
                quotations.append(QuotationSpan(
                    content=content,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    quote_type='direct',
                    confidence=0.95
                ))
        
        return quotations
    
    def _find_indirect_quotes(self, text: str) -> List[QuotationSpan]:
        """Find indirect quotations and reported speech."""
        quotations = []
        
        # Find sentences with indirect quotation indicators
        sentences = re.split(r'[.!?]+', text)
        current_pos = 0
        
        for sentence in sentences:
            if self.indirect_quote_regex.search(sentence):
                # This sentence contains indirect quotation indicators
                sentence_start = text.find(sentence, current_pos)
                sentence_end = sentence_start + len(sentence)
                
                if sentence_start != -1:
                    quotations.append(QuotationSpan(
                        content=sentence.strip(),
                        start_pos=sentence_start,
                        end_pos=sentence_end,
                        quote_type='indirect',
                        confidence=0.7
                    ))
            
            current_pos += len(sentence) + 1  # +1 for the delimiter
        
        return quotations
    
    def _find_third_party_references(self, text: str) -> List[QuotationSpan]:
        """Find third-party references and attributions."""
        quotations = []
        
        for regex in self.third_party_regexes:
            for match in regex.finditer(text):
                # Extend to capture the full sentence or clause
                extended_start, extended_end = self._extend_to_clause(
                    text, match.start(), match.end()
                )
                
                quotations.append(QuotationSpan(
                    content=text[extended_start:extended_end],
                    start_pos=extended_start,
                    end_pos=extended_end,
                    quote_type='reported',
                    confidence=0.6,
                    attribution=match.group()
                ))
        
        return quotations
    
    def _extend_to_clause(self, text: str, start: int, end: int) -> Tuple[int, int]:
        """Extend match to capture the full clause or sentence."""
        # Find sentence boundaries
        sentence_start = start
        sentence_end = end
        
        # Look backwards for sentence start
        while sentence_start > 0 and text[sentence_start - 1] not in '.!?':
            sentence_start -= 1
        
        # Look forwards for sentence end
        while sentence_end < len(text) and text[sentence_end] not in '.!?':
            sentence_end += 1
        
        # Include the punctuation
        if sentence_end < len(text):
            sentence_end += 1
        
        return sentence_start, sentence_end
    
    def _remove_overlapping_quotes(self, quotations: List[QuotationSpan]) -> List[QuotationSpan]:
        """Remove overlapping quotations, keeping the highest confidence ones."""
        if not quotations:
            return quotations
        
        # Sort by start position
        quotations.sort(key=lambda q: q.start_pos)
        
        filtered = []
        for quote in quotations:
            # Check if this quote overlaps with any already filtered quote
            overlaps = False
            for existing in filtered:
                if (quote.start_pos < existing.end_pos and 
                    quote.end_pos > existing.start_pos):
                    # There's an overlap, keep the higher confidence one
                    if quote.confidence > existing.confidence:
                        filtered.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(quote)
        
        return filtered
    
    def is_in_quotation(self, span_start: int, span_end: int, quotations: List[QuotationSpan]) -> Tuple[bool, str, float]:
        """
        Check if a text span is within any quotation.
        
        Args:
            span_start: Start position of the span
            span_end: End position of the span
            quotations: List of detected quotations
            
        Returns:
            Tuple of (is_quoted, quote_type, confidence)
        """
        for quote in quotations:
            # Check if span is within quotation
            if (span_start >= quote.start_pos and span_end <= quote.end_pos):
                return True, quote.quote_type, quote.confidence
            
            # Check for partial overlap (also considered quoted)
            if (span_start < quote.end_pos and span_end > quote.start_pos):
                return True, quote.quote_type, quote.confidence * 0.8  # Reduced confidence for partial
        
        return False, None, 0.0
    
    def get_quotation_context(self, text: str, problem_spans: List) -> Dict[str, any]:
        """
        Analyze quotation context for problem spans.
        
        Args:
            text: Input text
            problem_spans: List of problem spans to check
            
        Returns:
            Dictionary with quotation analysis results
        """
        quotations = self.detect_quotations(text)
        
        quoted_spans = []
        max_quote_confidence = 0.0
        quote_types = set()
        
        for span in problem_spans:
            is_quoted, quote_type, confidence = self.is_in_quotation(
                span.start, span.end, quotations
            )
            
            if is_quoted:
                quoted_spans.append({
                    'span': span,
                    'quote_type': quote_type,
                    'quote_confidence': confidence
                })
                max_quote_confidence = max(max_quote_confidence, confidence)
                quote_types.add(quote_type)
        
        return {
            'has_quotations': len(quotations) > 0,
            'quotation_count': len(quotations),
            'quoted_spans': quoted_spans,
            'quote_types': list(quote_types),
            'overall_quote_confidence': max_quote_confidence,
            'detected_quotations': quotations
        }