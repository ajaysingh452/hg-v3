"""Text normalization engine for Harmony Guard."""

import re
import unicodedata
from typing import Dict, Tuple
import logging


logger = logging.getLogger(__name__)


class TextNormalizer:
    """Comprehensive text normalization for multilingual content."""
    
    def __init__(self, config: Dict):
        """Initialize text normalizer with configuration."""
        self.config = config
        self.unicode_form = config.get("unicode_form", "NFKC")
        self.strip_zero_width = config.get("strip_zero_width", True)
        self.fold_diacritics = config.get("fold_diacritics", True)
        self.normalize_punctuation = config.get("normalize_punctuation", True)
        self.compress_repeated_chars = config.get("compress_repeated_chars", True)
        self.max_char_repetition = config.get("max_char_repetition", 3)
        
        # Build normalization maps
        self.homoglyph_map = self._build_homoglyph_map()
        self.diacritic_map = self._build_diacritic_map()
        self.punctuation_map = self._build_punctuation_map()
        self.zero_width_chars = self._get_zero_width_chars()
    
    def normalize(self, text: str) -> str:
        """
        Apply comprehensive text normalization.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        # Step 1: Unicode normalization
        normalized = self._normalize_unicode(text)
        
        # Step 2: Remove zero-width characters
        if self.strip_zero_width:
            normalized = self._remove_zero_width_chars(normalized)
        
        # Step 3: Homoglyph normalization
        normalized = self._normalize_homoglyphs(normalized)
        
        # Step 4: Diacritic folding
        if self.fold_diacritics:
            normalized = self._fold_diacritics(normalized)
        
        # Step 5: Punctuation normalization
        if self.normalize_punctuation:
            normalized = self._normalize_punctuation(normalized)
        
        # Step 6: Compress repeated characters
        if self.compress_repeated_chars:
            normalized = self._compress_repeated_chars(normalized)
        
        # Step 7: Clean whitespace
        normalized = self._clean_whitespace(normalized)
        
        return normalized
    
    def _normalize_unicode(self, text: str) -> str:
        """Apply Unicode normalization."""
        try:
            return unicodedata.normalize(self.unicode_form, text)
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    def _remove_zero_width_chars(self, text: str) -> str:
        """Remove zero-width and invisible characters."""
        for char in self.zero_width_chars:
            text = text.replace(char, '')
        return text
    
    def _normalize_homoglyphs(self, text: str) -> str:
        """Normalize homoglyph characters to their standard equivalents."""
        for homoglyph, standard in self.homoglyph_map.items():
            text = text.replace(homoglyph, standard)
        return text
    
    def _fold_diacritics(self, text: str) -> str:
        """Fold diacritical marks to base characters."""
        # First try the mapping table for common cases
        for accented, base in self.diacritic_map.items():
            text = text.replace(accented, base)
        
        # Then use Unicode decomposition for remaining cases
        normalized = unicodedata.normalize('NFD', text)
        without_diacritics = ''.join(
            char for char in normalized 
            if unicodedata.category(char) != 'Mn'  # Mn = Nonspacing_Mark
        )
        return unicodedata.normalize('NFC', without_diacritics)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks to standard forms."""
        for variant, standard in self.punctuation_map.items():
            text = text.replace(variant, standard)
        return text
    
    def _compress_repeated_chars(self, text: str) -> str:
        """Compress repeated characters to maximum allowed repetition."""
        # Pattern to match 3 or more consecutive identical characters
        pattern = r'(.)\1{' + str(self.max_char_repetition - 1) + ',}'
        
        def replace_func(match):
            char = match.group(1)
            return char * self.max_char_repetition
        
        return re.sub(pattern, replace_func, text)
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    def _build_homoglyph_map(self) -> Dict[str, str]:
        """Build comprehensive homoglyph normalization map."""
        return {
            # Cyrillic to Latin
            'а': 'a', 'А': 'A',  # Cyrillic a
            'е': 'e', 'Е': 'E',  # Cyrillic e
            'о': 'o', 'О': 'O',  # Cyrillic o
            'р': 'p', 'Р': 'P',  # Cyrillic p
            'с': 'c', 'С': 'C',  # Cyrillic c
            'х': 'x', 'Х': 'X',  # Cyrillic x
            'у': 'y', 'У': 'Y',  # Cyrillic y
            'к': 'k', 'К': 'K',  # Cyrillic k
            'н': 'h', 'Н': 'H',  # Cyrillic h
            'м': 'm', 'М': 'M',  # Cyrillic m
            'т': 't', 'Т': 'T',  # Cyrillic t
            
            # Greek to Latin
            'α': 'a', 'Α': 'A',  # Greek alpha
            'β': 'b', 'Β': 'B',  # Greek beta
            'ε': 'e', 'Ε': 'E',  # Greek epsilon
            'ο': 'o', 'Ο': 'O',  # Greek omicron
            'ρ': 'p', 'Ρ': 'P',  # Greek rho
            'τ': 't', 'Τ': 'T',  # Greek tau
            'υ': 'y', 'Υ': 'Y',  # Greek upsilon
            'χ': 'x', 'Χ': 'X',  # Greek chi
            
            # Mathematical symbols
            '𝐚': 'a', '𝐀': 'A',  # Mathematical bold
            '𝑎': 'a', '𝐴': 'A',  # Mathematical italic
            '𝒂': 'a', '𝑨': 'A',  # Mathematical script
            '𝓪': 'a', '𝓐': 'A',  # Mathematical bold script
            
            # Fullwidth characters
            'ａ': 'a', 'Ａ': 'A',
            'ｂ': 'b', 'Ｂ': 'B',
            'ｃ': 'c', 'Ｃ': 'C',
            'ｄ': 'd', 'Ｄ': 'D',
            'ｅ': 'e', 'Ｅ': 'E',
            'ｆ': 'f', 'Ｆ': 'F',
            'ｇ': 'g', 'Ｇ': 'G',
            'ｈ': 'h', 'Ｈ': 'H',
            'ｉ': 'i', 'Ｉ': 'I',
            'ｊ': 'j', 'Ｊ': 'J',
            'ｋ': 'k', 'Ｋ': 'K',
            'ｌ': 'l', 'Ｌ': 'L',
            'ｍ': 'm', 'Ｍ': 'M',
            'ｎ': 'n', 'Ｎ': 'N',
            'ｏ': 'o', 'Ｏ': 'O',
            'ｐ': 'p', 'Ｐ': 'P',
            'ｑ': 'q', 'Ｑ': 'Q',
            'ｒ': 'r', 'Ｒ': 'R',
            'ｓ': 's', 'Ｓ': 'S',
            'ｔ': 't', 'Ｔ': 'T',
            'ｕ': 'u', 'Ｕ': 'U',
            'ｖ': 'v', 'Ｖ': 'V',
            'ｗ': 'w', 'Ｗ': 'W',
            'ｘ': 'x', 'Ｘ': 'X',
            'ｙ': 'y', 'Ｙ': 'Y',
            'ｚ': 'z', 'Ｚ': 'Z',
        }
    
    def _build_diacritic_map(self) -> Dict[str, str]:
        """Build diacritic folding map for common accented characters."""
        return {
            # Latin with diacritics
            'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'æ': 'ae',
            'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A', 'Æ': 'AE',
            'ç': 'c', 'Ç': 'C',
            'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
            'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
            'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
            'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I',
            'ñ': 'n', 'Ñ': 'N',
            'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o',
            'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O', 'Ø': 'O',
            'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
            'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
            'ý': 'y', 'ÿ': 'y', 'Ý': 'Y', 'Ÿ': 'Y',
            
            # Extended Latin
            'ā': 'a', 'ă': 'a', 'ą': 'a', 'Ā': 'A', 'Ă': 'A', 'Ą': 'A',
            'ć': 'c', 'ĉ': 'c', 'ċ': 'c', 'č': 'c', 'Ć': 'C', 'Ĉ': 'C', 'Ċ': 'C', 'Č': 'C',
            'ď': 'd', 'đ': 'd', 'Ď': 'D', 'Đ': 'D',
            'ē': 'e', 'ĕ': 'e', 'ė': 'e', 'ę': 'e', 'ě': 'e',
            'Ē': 'E', 'Ĕ': 'E', 'Ė': 'E', 'Ę': 'E', 'Ě': 'E',
        }
    
    def _build_punctuation_map(self) -> Dict[str, str]:
        """Build punctuation normalization map."""
        return {
            # Quotation marks
            '"': '"', '"': '"', ''': "'", ''': "'",
            '«': '"', '»': '"', '‹': "'", '›': "'",
            
            # Dashes and hyphens
            '–': '-', '—': '-', '―': '-', '‒': '-',
            
            # Ellipsis
            '…': '...',
            
            # Apostrophes
            ''': "'", '`': "'",
            
            # Spaces
            '\u00A0': ' ',  # Non-breaking space
            '\u2000': ' ',  # En quad
            '\u2001': ' ',  # Em quad
            '\u2002': ' ',  # En space
            '\u2003': ' ',  # Em space
            '\u2004': ' ',  # Three-per-em space
            '\u2005': ' ',  # Four-per-em space
            '\u2006': ' ',  # Six-per-em space
            '\u2007': ' ',  # Figure space
            '\u2008': ' ',  # Punctuation space
            '\u2009': ' ',  # Thin space
            '\u200A': ' ',  # Hair space
            '\u202F': ' ',  # Narrow no-break space
            '\u205F': ' ',  # Medium mathematical space
            '\u3000': ' ',  # Ideographic space
        }
    
    def _get_zero_width_chars(self) -> list:
        """Get list of zero-width and invisible characters to remove."""
        return [
            '\u200B',  # Zero width space
            '\u200C',  # Zero width non-joiner
            '\u200D',  # Zero width joiner
            '\u200E',  # Left-to-right mark
            '\u200F',  # Right-to-left mark
            '\u202A',  # Left-to-right embedding
            '\u202B',  # Right-to-left embedding
            '\u202C',  # Pop directional formatting
            '\u202D',  # Left-to-right override
            '\u202E',  # Right-to-left override
            '\u2060',  # Word joiner
            '\u2061',  # Function application
            '\u2062',  # Invisible times
            '\u2063',  # Invisible separator
            '\u2064',  # Invisible plus
            '\uFEFF',  # Byte order mark / Zero width no-break space
            '\uFFF9',  # Interlinear annotation anchor
            '\uFFFA',  # Interlinear annotation separator
            '\uFFFB',  # Interlinear annotation terminator
        ]