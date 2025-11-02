"""Obfuscation detection and normalization for Harmony Guard."""

import re
import yaml
from typing import Dict, List, Tuple, Set
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class ObfuscationHandler:
    """Handles various text obfuscation techniques including leet speak and phonetic variations."""
    
    def __init__(self, config: Dict, lexicon_path: str = None):
        """Initialize obfuscation handler."""
        self.config = config
        self.leet_detection = config.get("leet_speak_detection", True)
        self.phonetic_normalization = config.get("phonetic_normalization", True)
        self.elongation_detection = config.get("elongation_detection", True)
        self.homoglyph_normalization = config.get("homoglyph_normalization", True)
        
        # Load leet speak tables
        if lexicon_path:
            self.leet_data = self._load_leet_data(lexicon_path)
        else:
            self.leet_data = self._get_default_leet_data()
        
        # Build lookup tables
        self.leet_substitutions = self.leet_data.get("substitutions", {})
        self.leet_patterns = self.leet_data.get("patterns", [])
        self.phonetic_rules = self.leet_data.get("phonetic", {})
        self.elongation_config = self.leet_data.get("elongation", {})
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def normalize_obfuscations(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Normalize all types of obfuscations in text.
        
        Args:
            text: Input text with potential obfuscations
            
        Returns:
            Tuple of (normalized_text, obfuscation_map)
        """
        normalized = text
        obfuscation_map = {}
        
        # Step 1: Leet speak normalization
        if self.leet_detection:
            normalized, leet_map = self._normalize_leet_speak(normalized)
            obfuscation_map.update(leet_map)
        
        # Step 2: Phonetic normalization
        if self.phonetic_normalization:
            normalized, phonetic_map = self._normalize_phonetic(normalized)
            obfuscation_map.update(phonetic_map)
        
        # Step 3: Elongation normalization
        if self.elongation_detection:
            normalized, elongation_map = self._normalize_elongation(normalized)
            obfuscation_map.update(elongation_map)
        
        # Step 4: Character substitution patterns
        normalized, substitution_map = self._normalize_character_substitutions(normalized)
        obfuscation_map.update(substitution_map)
        
        return normalized, obfuscation_map
    
    def detect_obfuscation_techniques(self, text: str) -> List[str]:
        """
        Detect which obfuscation techniques are present in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected obfuscation technique names
        """
        techniques = []
        
        # Check for leet speak
        if self._has_leet_speak(text):
            techniques.append("leet_speak")
        
        # Check for elongation
        if self._has_elongation(text):
            techniques.append("elongation")
        
        # Check for phonetic obfuscation
        if self._has_phonetic_obfuscation(text):
            techniques.append("phonetic")
        
        # Check for character substitutions
        if self._has_character_substitutions(text):
            techniques.append("character_substitution")
        
        return techniques
    
    def _normalize_leet_speak(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Normalize leet speak obfuscations."""
        normalized = text
        leet_map = {}
        
        # Apply pattern-based leet normalization first
        for pattern_info in self.leet_patterns:
            leet_pattern = pattern_info.get("leet", "")
            normal_form = pattern_info.get("normal", "")
            
            if leet_pattern and normal_form:
                # Case-insensitive matching
                pattern = re.compile(re.escape(leet_pattern), re.IGNORECASE)
                matches = pattern.findall(normalized)
                
                for match in matches:
                    leet_map[match] = normal_form
                    normalized = pattern.sub(normal_form, normalized, count=1)
        
        # Apply character-level substitutions
        for leet_char, normal_chars in self.leet_substitutions.items():
            if leet_char in normalized:
                # Use the first normal character as replacement
                normal_char = normal_chars[0] if isinstance(normal_chars, list) else normal_chars
                if leet_char != normal_char:
                    leet_map[leet_char] = normal_char
                    normalized = normalized.replace(leet_char, normal_char)
        
        return normalized, leet_map
    
    def _normalize_phonetic(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Normalize phonetic obfuscations."""
        normalized = text
        phonetic_map = {}
        
        for phonetic, replacement in self.phonetic_rules.items():
            if phonetic in normalized:
                phonetic_map[phonetic] = replacement
                normalized = normalized.replace(phonetic, replacement)
        
        return normalized, phonetic_map
    
    def _normalize_elongation(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Normalize elongated characters."""
        elongation_map = {}
        
        # Get configuration
        min_repetition = self.elongation_config.get("min_repetition", 3)
        max_compression = self.elongation_config.get("max_compression", 2)
        
        # Pattern to match repeated characters
        pattern = r'(.)\1{' + str(min_repetition - 1) + ',}'
        
        def replace_elongation(match):
            char = match.group(1)
            original = match.group(0)
            compressed = char * max_compression
            
            if original != compressed:
                elongation_map[original] = compressed
            
            return compressed
        
        normalized = re.sub(pattern, replace_elongation, text)
        
        return normalized, elongation_map
    
    def _normalize_character_substitutions(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Normalize various character substitutions."""
        normalized = text
        substitution_map = {}
        
        # Common character substitutions for obfuscation
        substitutions = {
            '0': 'o',
            '1': 'i', 
            '3': 'e',
            '4': 'a',
            '5': 's',
            '7': 't',
            '8': 'b',
            '!': 'i',
            '@': 'a',
            '$': 's',
            '+': 't',
            '()': 'o',
            '|-|': 'h',
            '|_|': 'u',
            '\\/': 'v',
            '\\/\\/': 'w',
            '><': 'x'
        }
        
        for obfuscated, normal in substitutions.items():
            if obfuscated in normalized:
                substitution_map[obfuscated] = normal
                normalized = normalized.replace(obfuscated, normal)
        
        return normalized, substitution_map
    
    def _has_leet_speak(self, text: str) -> bool:
        """Check if text contains leet speak patterns."""
        # Check for common leet characters
        leet_chars = set('@!$+0135478')
        return any(char in leet_chars for char in text)
    
    def _has_elongation(self, text: str) -> bool:
        """Check if text contains elongated characters."""
        # Pattern for 3+ repeated characters
        pattern = r'(.)\1{2,}'
        return bool(re.search(pattern, text))
    
    def _has_phonetic_obfuscation(self, text: str) -> bool:
        """Check if text contains phonetic obfuscations."""
        phonetic_patterns = ['ph', 'ck', 'qu', 'x']
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in phonetic_patterns)
    
    def _has_character_substitutions(self, text: str) -> bool:
        """Check if text contains character substitutions."""
        substitution_chars = set('@!$+0135478')
        return any(char in substitution_chars for char in text)
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.compiled_patterns = {}
        
        # Compile leet patterns
        for pattern_info in self.leet_patterns:
            leet_pattern = pattern_info.get("leet", "")
            if leet_pattern:
                try:
                    self.compiled_patterns[leet_pattern] = re.compile(
                        re.escape(leet_pattern), re.IGNORECASE
                    )
                except re.error as e:
                    logger.warning(f"Failed to compile leet pattern '{leet_pattern}': {e}")
    
    def _load_leet_data(self, lexicon_path: str) -> Dict:
        """Load leet speak data from YAML file."""
        try:
            leet_file = Path(lexicon_path) / "leet.yaml"
            if leet_file.exists():
                with open(leet_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load leet data from {lexicon_path}: {e}")
        
        return self._get_default_leet_data()
    
    def _get_default_leet_data(self) -> Dict:
        """Get default leet speak data if file loading fails."""
        return {
            "substitutions": {
                "0": "o",
                "1": "i",
                "3": "e", 
                "4": "a",
                "5": "s",
                "7": "t",
                "8": "b",
                "@": "a",
                "!": "i",
                "$": "s",
                "+": "t"
            },
            "patterns": [
                {"leet": "f*ck", "normal": "fuck"},
                {"leet": "sh!t", "normal": "shit"},
                {"leet": "@ss", "normal": "ass"},
                {"leet": "b!tch", "normal": "bitch"},
                {"leet": "d@mn", "normal": "damn"},
                {"leet": "!d!0t", "normal": "idiot"},
                {"leet": "m0r0n", "normal": "moron"},
                {"leet": "5tup!d", "normal": "stupid"}
            ],
            "phonetic": {
                "ph": "f",
                "ck": "k", 
                "qu": "kw",
                "x": "ks"
            },
            "elongation": {
                "min_repetition": 3,
                "max_compression": 2
            }
        }
    
    def generate_obfuscation_variants(self, word: str) -> List[str]:
        """
        Generate possible obfuscation variants of a word.
        
        Args:
            word: Clean word to obfuscate
            
        Returns:
            List of possible obfuscated variants
        """
        variants = [word]
        
        # Generate leet speak variants
        leet_variants = self._generate_leet_variants(word)
        variants.extend(leet_variants)
        
        # Generate elongation variants
        elongation_variants = self._generate_elongation_variants(word)
        variants.extend(elongation_variants)
        
        # Generate phonetic variants
        phonetic_variants = self._generate_phonetic_variants(word)
        variants.extend(phonetic_variants)
        
        # Remove duplicates
        return list(set(variants))
    
    def _generate_leet_variants(self, word: str) -> List[str]:
        """Generate leet speak variants of a word."""
        variants = []
        
        # Character substitutions
        leet_map = {
            'a': ['@', '4'],
            'e': ['3'],
            'i': ['!', '1'],
            'o': ['0'],
            's': ['$', '5'],
            't': ['+', '7'],
            'b': ['8']
        }
        
        # Generate single character substitutions
        for i, char in enumerate(word.lower()):
            if char in leet_map:
                for leet_char in leet_map[char]:
                    variant = word[:i] + leet_char + word[i+1:]
                    variants.append(variant)
        
        return variants
    
    def _generate_elongation_variants(self, word: str) -> List[str]:
        """Generate elongated variants of a word."""
        variants = []
        
        # Add elongation to vowels and some consonants
        elongatable = set('aeiouflrs')
        
        for i, char in enumerate(word.lower()):
            if char in elongatable:
                # Add 2-4 extra characters
                for extra in range(2, 5):
                    variant = word[:i+1] + char * extra + word[i+1:]
                    variants.append(variant)
        
        return variants
    
    def _generate_phonetic_variants(self, word: str) -> List[str]:
        """Generate phonetic variants of a word."""
        variants = []
        
        phonetic_substitutions = {
            'f': 'ph',
            'k': 'ck', 
            'c': 'k',
            's': 'z'
        }
        
        for original, replacement in phonetic_substitutions.items():
            if original in word.lower():
                variant = word.lower().replace(original, replacement)
                variants.append(variant)
        
        return variants