"""Advanced tokenization system for multilingual and emoji-aware processing."""

import re
from typing import List, Tuple, Dict, Optional
import unicodedata
import logging


logger = logging.getLogger(__name__)


class TokenSpan:
    """Represents a token with its position in the original text."""
    
    def __init__(self, text: str, start: int, end: int, token_type: str = "word"):
        self.text = text
        self.start = start
        self.end = end
        self.token_type = token_type  # word, emoji, punctuation, number, etc.
    
    def __repr__(self):
        return f"TokenSpan('{self.text}', {self.start}-{self.end}, {self.token_type})"


class MultilingualTokenizer:
    """Advanced tokenizer supporting multiple scripts, emojis, and span preservation."""
    
    def __init__(self, config: Dict):
        """Initialize tokenizer with configuration."""
        self.config = config
        self.emoji_aware = config.get("emoji_aware", True)
        self.script_aware = config.get("script_aware", True)
        self.preserve_spans = config.get("preserve_spans", True)
        
        # Unicode ranges for different scripts
        self.script_ranges = {
            "latin": (0x0041, 0x007A, 0x00C0, 0x00FF),  # Basic + Extended Latin
            "devanagari": (0x0900, 0x097F),  # Hindi, Marathi, etc.
            "bengali": (0x0980, 0x09FF),     # Bengali, Assamese
            "tamil": (0x0B80, 0x0BFF),       # Tamil
            "telugu": (0x0C00, 0x0C7F),      # Telugu
            "kannada": (0x0C80, 0x0CFF),     # Kannada
            "malayalam": (0x0D00, 0x0D7F),   # Malayalam
            "gujarati": (0x0A80, 0x0AFF),    # Gujarati
            "odia": (0x0B00, 0x0B7F),        # Odia
            "punjabi": (0x0A00, 0x0A7F),     # Punjabi (Gurmukhi)
        }
        
        # Emoji ranges (basic coverage)
        self.emoji_ranges = [
            (0x1F600, 0x1F64F),  # Emoticons
            (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
            (0x1F680, 0x1F6FF),  # Transport and Map
            (0x1F1E0, 0x1F1FF),  # Regional Indicator Symbols
            (0x2600, 0x26FF),    # Miscellaneous Symbols
            (0x2700, 0x27BF),    # Dingbats
            (0xFE00, 0xFE0F),    # Variation Selectors
            (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        ]
        
        # Compile regex patterns
        self._compile_patterns()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of token strings.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token strings
        """
        token_spans = self.tokenize_with_spans(text)
        return [span.text for span in token_spans]
    
    def tokenize_with_spans(self, text: str) -> List[TokenSpan]:
        """
        Tokenize text and return tokens with their original positions.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of TokenSpan objects
        """
        if not text:
            return []
        
        tokens = []
        current_pos = 0
        
        # Process text character by character, grouping by type
        i = 0
        while i < len(text):
            char = text[i]
            char_code = ord(char)
            
            # Skip whitespace but track position
            if char.isspace():
                i += 1
                current_pos = i
                continue
            
            # Determine token type and extract token
            if self._is_emoji(char_code):
                token, length = self._extract_emoji_token(text, i)
                token_type = "emoji"
            elif self._is_script_char(char_code):
                token, length = self._extract_script_token(text, i)
                token_type = "word"
            elif char.isdigit():
                token, length = self._extract_number_token(text, i)
                token_type = "number"
            elif self._is_punctuation(char):
                token, length = self._extract_punctuation_token(text, i)
                token_type = "punctuation"
            else:
                # Default: single character
                token = char
                length = 1
                token_type = "other"
            
            # Create token span
            if token:
                span = TokenSpan(
                    text=token,
                    start=i,
                    end=i + length,
                    token_type=token_type
                )
                tokens.append(span)
            
            i += length
        
        return tokens
    
    def get_script_tokens(self, text: str) -> Dict[str, List[str]]:
        """
        Group tokens by their script type.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping script names to token lists
        """
        token_spans = self.tokenize_with_spans(text)
        script_tokens = {}
        
        for span in token_spans:
            if span.token_type == "word":
                script = self._detect_token_script(span.text)
                if script not in script_tokens:
                    script_tokens[script] = []
                script_tokens[script].append(span.text)
        
        return script_tokens
    
    def _extract_emoji_token(self, text: str, start: int) -> Tuple[str, int]:
        """Extract emoji token starting at position."""
        i = start
        emoji_chars = []
        
        while i < len(text):
            char = text[i]
            char_code = ord(char)
            
            if self._is_emoji(char_code) or self._is_emoji_modifier(char_code):
                emoji_chars.append(char)
                i += 1
            else:
                break
        
        emoji_token = ''.join(emoji_chars)
        return emoji_token, len(emoji_token)
    
    def _extract_script_token(self, text: str, start: int) -> Tuple[str, int]:
        """Extract word token in a specific script."""
        i = start
        word_chars = []
        
        while i < len(text):
            char = text[i]
            char_code = ord(char)
            
            if (self._is_script_char(char_code) or 
                char.isalnum() or 
                char in "'-"):  # Include apostrophes and hyphens in words
                word_chars.append(char)
                i += 1
            else:
                break
        
        word_token = ''.join(word_chars)
        return word_token, len(word_token)
    
    def _extract_number_token(self, text: str, start: int) -> Tuple[str, int]:
        """Extract numeric token."""
        i = start
        number_chars = []
        
        while i < len(text):
            char = text[i]
            
            if char.isdigit() or char in ".,":  # Include decimal points and commas
                number_chars.append(char)
                i += 1
            else:
                break
        
        number_token = ''.join(number_chars)
        return number_token, len(number_token)
    
    def _extract_punctuation_token(self, text: str, start: int) -> Tuple[str, int]:
        """Extract punctuation token."""
        # For now, treat each punctuation mark as separate token
        return text[start], 1
    
    def _is_emoji(self, char_code: int) -> bool:
        """Check if character code represents an emoji."""
        if not self.emoji_aware:
            return False
        
        for start, end in self.emoji_ranges:
            if start <= char_code <= end:
                return True
        return False
    
    def _is_emoji_modifier(self, char_code: int) -> bool:
        """Check if character is an emoji modifier (skin tone, etc.)."""
        # Emoji modifiers and variation selectors
        modifier_ranges = [
            (0x1F3FB, 0x1F3FF),  # Skin tone modifiers
            (0xFE00, 0xFE0F),    # Variation selectors
            (0x200D, 0x200D),    # Zero width joiner
        ]
        
        for start, end in modifier_ranges:
            if start <= char_code <= end:
                return True
        return False
    
    def _is_script_char(self, char_code: int) -> bool:
        """Check if character belongs to a supported script."""
        if not self.script_aware:
            return char_code < 128  # ASCII only
        
        for script, ranges in self.script_ranges.items():
            if len(ranges) == 2:
                start, end = ranges
                if start <= char_code <= end:
                    return True
            elif len(ranges) == 4:
                start1, end1, start2, end2 = ranges
                if (start1 <= char_code <= end1) or (start2 <= char_code <= end2):
                    return True
        
        # Also include basic ASCII letters
        return (0x0041 <= char_code <= 0x005A) or (0x0061 <= char_code <= 0x007A)
    
    def _is_punctuation(self, char: str) -> bool:
        """Check if character is punctuation."""
        return unicodedata.category(char).startswith('P')
    
    def _detect_token_script(self, token: str) -> str:
        """Detect the script of a token."""
        if not token:
            return "unknown"
        
        # Count characters in each script
        script_counts = {}
        
        for char in token:
            char_code = ord(char)
            detected_script = "latin"  # default
            
            for script, ranges in self.script_ranges.items():
                if len(ranges) == 2:
                    start, end = ranges
                    if start <= char_code <= end:
                        detected_script = script
                        break
                elif len(ranges) == 4:
                    start1, end1, start2, end2 = ranges
                    if (start1 <= char_code <= end1) or (start2 <= char_code <= end2):
                        detected_script = script
                        break
            
            script_counts[detected_script] = script_counts.get(detected_script, 0) + 1
        
        # Return the script with the most characters
        if script_counts:
            return max(script_counts, key=script_counts.get)
        
        return "unknown"
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient tokenization."""
        # Word pattern that handles multiple scripts
        word_patterns = []
        
        # Latin script words
        word_patterns.append(r'[a-zA-Z]+(?:\'[a-zA-Z]+)*')
        
        # Devanagari script words
        word_patterns.append(r'[\u0900-\u097F]+')
        
        # Bengali script words
        word_patterns.append(r'[\u0980-\u09FF]+')
        
        # Tamil script words
        word_patterns.append(r'[\u0B80-\u0BFF]+')
        
        # Telugu script words
        word_patterns.append(r'[\u0C00-\u0C7F]+')
        
        # Kannada script words
        word_patterns.append(r'[\u0C80-\u0CFF]+')
        
        # Malayalam script words
        word_patterns.append(r'[\u0D00-\u0D7F]+')
        
        # Gujarati script words
        word_patterns.append(r'[\u0A80-\u0AFF]+')
        
        # Odia script words
        word_patterns.append(r'[\u0B00-\u0B7F]+')
        
        # Punjabi script words
        word_patterns.append(r'[\u0A00-\u0A7F]+')
        
        # Numbers
        word_patterns.append(r'\d+(?:[.,]\d+)*')
        
        # Emoji pattern (basic)
        if self.emoji_aware:
            emoji_pattern = (
                r'[\U0001F600-\U0001F64F]|'  # Emoticons
                r'[\U0001F300-\U0001F5FF]|'  # Misc Symbols
                r'[\U0001F680-\U0001F6FF]|'  # Transport
                r'[\U0001F1E0-\U0001F1FF]|'  # Regional Indicators
                r'[\U00002600-\U000026FF]|'  # Misc Symbols
                r'[\U00002700-\U000027BF]'   # Dingbats
            )
            word_patterns.append(emoji_pattern)
        
        # Punctuation
        word_patterns.append(r'[^\w\s]')
        
        # Combine all patterns
        self.token_pattern = re.compile('|'.join(word_patterns), re.UNICODE)
    
    def regex_tokenize(self, text: str) -> List[str]:
        """
        Alternative tokenization using compiled regex patterns.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if hasattr(self, 'token_pattern'):
            tokens = self.token_pattern.findall(text)
            return [token for token in tokens if token.strip()]
        else:
            # Fallback to simple tokenization
            return text.split()
    
    def create_span_mapping(self, original_text: str, tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Create mapping from tokens back to original text positions.
        
        Args:
            original_text: Original input text
            tokens: List of tokens
            
        Returns:
            List of (start, end) position tuples for each token
        """
        spans = []
        current_pos = 0
        
        for token in tokens:
            # Find token in original text starting from current position
            token_start = original_text.find(token, current_pos)
            
            if token_start != -1:
                token_end = token_start + len(token)
                spans.append((token_start, token_end))
                current_pos = token_end
            else:
                # Token not found, use approximate position
                spans.append((current_pos, current_pos + len(token)))
                current_pos += len(token)
        
        return spans