"""Transliteration system for Indic languages and Hinglish."""

import re
from typing import Dict, List, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class TransliterationEngine:
    """Bidirectional transliteration between Romanized and native scripts."""
    
    def __init__(self, config: Dict):
        """Initialize transliteration engine."""
        self.config = config
        self.romanized_to_native = config.get("romanized_to_native", True)
        self.native_to_romanized = config.get("native_to_romanized", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.generate_variants = config.get("generate_variants", True)
        
        # Build transliteration maps
        self.hindi_roman_to_devanagari = self._build_hindi_roman_to_devanagari()
        self.hindi_devanagari_to_roman = self._build_hindi_devanagari_to_roman()
        self.bengali_roman_to_native = self._build_bengali_roman_to_native()
        self.tamil_roman_to_native = self._build_tamil_roman_to_native()
        
        # Phonetic mapping rules
        self.phonetic_rules = self._build_phonetic_rules()
        
        # Common Hinglish words and their Devanagari equivalents
        self.hinglish_dictionary = self._build_hinglish_dictionary()
    
    def transliterate_to_native(self, text: str, target_language: str) -> Dict[str, str]:
        """
        Transliterate Romanized text to native script.
        
        Args:
            text: Romanized input text
            target_language: Target language code (hi, bn, ta, etc.)
            
        Returns:
            Dictionary mapping romanized words to native script
        """
        transliterations = {}
        
        if target_language == "hi":
            transliterations.update(self._transliterate_hindi_to_devanagari(text))
        elif target_language == "bn":
            transliterations.update(self._transliterate_bengali_to_native(text))
        elif target_language == "ta":
            transliterations.update(self._transliterate_tamil_to_native(text))
        
        return transliterations
    
    def transliterate_to_roman(self, text: str, source_language: str) -> Dict[str, str]:
        """
        Transliterate native script to Romanized form.
        
        Args:
            text: Native script input text
            source_language: Source language code (hi, bn, ta, etc.)
            
        Returns:
            Dictionary mapping native words to romanized form
        """
        transliterations = {}
        
        if source_language == "hi":
            transliterations.update(self._transliterate_devanagari_to_roman(text))
        
        return transliterations
    
    def generate_variants(self, word: str, language: str) -> List[str]:
        """
        Generate transliteration variants for a word.
        
        Args:
            word: Input word
            language: Language code
            
        Returns:
            List of possible transliteration variants
        """
        variants = [word]
        
        if language == "hi" or language == "hi-Latn":
            variants.extend(self._generate_hindi_variants(word))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants
    
    def _transliterate_hindi_to_devanagari(self, text: str) -> Dict[str, str]:
        """Transliterate Hindi Romanized text to Devanagari."""
        transliterations = {}
        words = text.lower().split()
        
        for word in words:
            # Check dictionary first
            if word in self.hinglish_dictionary:
                transliterations[word] = self.hinglish_dictionary[word]
                continue
            
            # Apply phonetic rules
            devanagari = self._apply_roman_to_devanagari_rules(word)
            if devanagari and devanagari != word:
                transliterations[word] = devanagari
        
        return transliterations
    
    def _transliterate_devanagari_to_roman(self, text: str) -> Dict[str, str]:
        """Transliterate Devanagari text to Roman script."""
        transliterations = {}
        
        # Split text into Devanagari words
        devanagari_words = re.findall(r'[\u0900-\u097F]+', text)
        
        for word in devanagari_words:
            roman = self._apply_devanagari_to_roman_rules(word)
            if roman:
                transliterations[word] = roman
        
        return transliterations
    
    def _transliterate_bengali_to_native(self, text: str) -> Dict[str, str]:
        """Transliterate Bengali Romanized text to Bengali script."""
        transliterations = {}
        words = text.lower().split()
        
        for word in words:
            if word in self.bengali_roman_to_native:
                transliterations[word] = self.bengali_roman_to_native[word]
        
        return transliterations
    
    def _transliterate_tamil_to_native(self, text: str) -> Dict[str, str]:
        """Transliterate Tamil Romanized text to Tamil script."""
        transliterations = {}
        words = text.lower().split()
        
        for word in words:
            if word in self.tamil_roman_to_native:
                transliterations[word] = self.tamil_roman_to_native[word]
        
        return transliterations
    
    def _apply_roman_to_devanagari_rules(self, word: str) -> Optional[str]:
        """Apply phonetic rules to convert Roman to Devanagari."""
        if len(word) < 2:
            return None
        
        # Simple rule-based transliteration
        result = word
        
        # Apply consonant mappings
        consonant_map = {
            'kh': 'ख', 'gh': 'घ', 'ch': 'च', 'chh': 'छ', 'jh': 'झ',
            'th': 'थ', 'dh': 'ध', 'ph': 'फ', 'bh': 'भ', 'sh': 'श',
            'k': 'क', 'g': 'ग', 'c': 'च', 'j': 'ज', 't': 'त',
            'd': 'द', 'n': 'न', 'p': 'प', 'b': 'ब', 'm': 'म',
            'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 's': 'स',
            'h': 'ह'
        }
        
        # Apply vowel mappings
        vowel_map = {
            'aa': 'आ', 'ii': 'ई', 'uu': 'ऊ', 'ee': 'ए', 'oo': 'ओ',
            'a': 'अ', 'i': 'इ', 'u': 'उ', 'e': 'ए', 'o': 'ओ'
        }
        
        # This is a simplified implementation
        # In production, use a proper transliteration library like Indic-NLP
        
        return None  # Return None for now, indicating no transliteration
    
    def _apply_devanagari_to_roman_rules(self, word: str) -> Optional[str]:
        """Apply rules to convert Devanagari to Roman script."""
        # Use reverse mapping from the dictionary
        for roman, devanagari in self.hinglish_dictionary.items():
            if devanagari == word:
                return roman
        
        return None
    
    def _generate_hindi_variants(self, word: str) -> List[str]:
        """Generate Hindi transliteration variants."""
        variants = []
        
        # Common phonetic variations
        variations = [
            ('ph', 'f'), ('bh', 'b'), ('th', 't'), ('dh', 'd'),
            ('kh', 'k'), ('gh', 'g'), ('ch', 'c'), ('jh', 'j'),
            ('aa', 'a'), ('ii', 'i'), ('uu', 'u'), ('ee', 'e'), ('oo', 'o'),
            ('w', 'v'), ('z', 'j'), ('x', 'ks')
        ]
        
        for old, new in variations:
            if old in word:
                variants.append(word.replace(old, new))
        
        return variants
    
    def _build_hindi_roman_to_devanagari(self) -> Dict[str, str]:
        """Build Hindi Romanized to Devanagari mapping."""
        return {
            # Common profanity and abuse terms
            "madarchod": "मादरचोद",
            "madarchod": "मदरचोद", 
            "bhosadi": "भोसड़ी",
            "bhosdi": "भोसड़ी",
            "randi": "रंडी",
            "kamina": "कमीना",
            "kameena": "कमीना",
            "harami": "हरामी",
            "haramzada": "हरामज़ादा",
            "saala": "साला",
            "sala": "साला",
            "kutte": "कुत्ते",
            "kutta": "कुत्ता",
            "suar": "सुअर",
            "gadha": "गधा",
            "gadhe": "गधे",
            "bewakoof": "बेवकूफ",
            "bevakoof": "बेवकूफ",
            "ullu": "उल्लू",
            "pagal": "पागल",
            "paagal": "पागल",
            
            # Common words
            "aur": "और",
            "hai": "है",
            "hain": "हैं",
            "kar": "कर",
            "kya": "क्या",
            "main": "मैं",
            "tu": "तू",
            "tum": "तुम",
            "yaar": "यार",
            "bhai": "भाई",
            "dost": "दोस्त",
            "ghar": "घर",
            "paisa": "पैसा",
            "paise": "पैसे",
            "rupee": "रुपया",
            "rupaye": "रुपये",
            "achha": "अच्छा",
            "accha": "अच्छा",
            "theek": "ठीक",
            "thik": "ठीक",
            "sahi": "सही",
            "galat": "गलत",
            "bas": "बस",
            "abhi": "अभी",
            "phir": "फिर",
            "kal": "कल",
            "aaj": "आज"
        }
    
    def _build_hindi_devanagari_to_roman(self) -> Dict[str, str]:
        """Build Devanagari to Roman mapping (reverse of above)."""
        return {v: k for k, v in self.hindi_roman_to_devanagari.items()}
    
    def _build_bengali_roman_to_native(self) -> Dict[str, str]:
        """Build Bengali Romanized to Bengali script mapping."""
        return {
            # Common Bengali abuse terms
            "sala": "শালা",
            "magir": "মাগীর",
            "magi": "মাগী",
            "khanki": "খানকী",
            "harami": "হারামী",
            "gadha": "গাধা",
            "pagol": "পাগল",
            "bewakoof": "বেওকুফ",
            
            # Common words
            "ami": "আমি",
            "tumi": "তুমি",
            "apni": "আপনি",
            "kemon": "কেমন",
            "achen": "আছেন",
            "bhalo": "ভালো",
            "kharap": "খারাপ",
            "bari": "বাড়ি",
            "taka": "টাকা"
        }
    
    def _build_tamil_roman_to_native(self) -> Dict[str, str]:
        """Build Tamil Romanized to Tamil script mapping."""
        return {
            # Common Tamil terms
            "naan": "நான்",
            "nee": "நீ",
            "avan": "அவன்",
            "aval": "அவள்",
            "enna": "என்ன",
            "epdi": "எப்படி",
            "veettu": "வீட்டு",
            "panam": "பணம்",
            "nalla": "நல்ல",
            "ketta": "கெட்ட"
        }
    
    def _build_phonetic_rules(self) -> Dict[str, str]:
        """Build phonetic transformation rules."""
        return {
            # Common phonetic variations
            "ph": "f",
            "bh": "b", 
            "th": "t",
            "dh": "d",
            "kh": "k",
            "gh": "g",
            "ch": "c",
            "jh": "j",
            "w": "v",
            "z": "j",
            "x": "ks",
            "q": "k"
        }
    
    def _build_hinglish_dictionary(self) -> Dict[str, str]:
        """Build comprehensive Hinglish to Devanagari dictionary."""
        return self.hindi_roman_to_devanagari