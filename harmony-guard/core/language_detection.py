"""Language identification module for multilingual content."""

import re
import math
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import logging

from .models import LanguageDetection


logger = logging.getLogger(__name__)


class LanguageIdentifier:
    """FastText-style language identification using character n-grams and script detection."""
    
    def __init__(self, config: Dict):
        """Initialize language identifier with configuration."""
        self.config = config
        self.supported_languages = config.get("supported_languages", ["en", "hi"])
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.segment_length = config.get("segment_length", 50)
        self.fallback_to_ngram = config.get("fallback_to_ngram", True)
        
        # FastText-style parameters
        self.ngram_min = 2
        self.ngram_max = 5
        self.min_text_length = 10
        
        # Character ranges for different scripts
        self.script_ranges = {
            "devanagari": (0x0900, 0x097F),  # Hindi, Marathi, etc.
            "bengali": (0x0980, 0x09FF),     # Bengali, Assamese
            "tamil": (0x0B80, 0x0BFF),       # Tamil
            "telugu": (0x0C00, 0x0C7F),      # Telugu
            "kannada": (0x0C80, 0x0CFF),     # Kannada
            "malayalam": (0x0D00, 0x0D7F),   # Malayalam
            "gujarati": (0x0A80, 0x0AFF),    # Gujarati
            "odia": (0x0B00, 0x0B7F),        # Odia
            "punjabi": (0x0A00, 0x0A7F),     # Punjabi (Gurmukhi)
            "latin": (0x0020, 0x007F),       # Basic Latin + ASCII
        }
        
        # Language-specific character n-grams with frequencies
        self.language_ngrams = self._build_language_ngrams()
        
        # Enhanced Hinglish detection patterns
        self.hinglish_patterns = [
            # Common Hindi words in Latin script
            r'\b(hai|kar|kya|aur|main|tu|yaar|bhai|dude|yeh|woh|kuch|koi)\b',
            r'\b(achha|theek|sahi|galat|bas|abhi|phir|kal|aaj|raat)\b',
            r'\b(paisa|rupee|ghar|office|school|college|job|kaam)\b',
            # Mixed patterns
            r'\b(kaise|kaisa|kaisi|kyun|kyunki|lekin|matlab|samjha)\b',
            r'\b(dekho|dekh|suno|sun|bolo|bol|chal|chalo|aao)\b',
            # English-Hindi mixing indicators
            r'\b\w+\s+(hai|hain|tha|thi|the|hoga|hogi|honge)\b',
            r'\b(very|really|totally|super)\s+\w*(achha|sahi|galat|bura)\b'
        ]
        
        # Script-to-language mapping with confidence weights
        self.script_to_languages = {
            "devanagari": [("hi", 0.8), ("mr", 0.2)],  # Hindi primary, Marathi secondary
            "bengali": [("bn", 0.9), ("as", 0.1)],     # Bengali primary, Assamese secondary
            "tamil": [("ta", 1.0)],
            "telugu": [("te", 1.0)],
            "kannada": [("kn", 1.0)],
            "malayalam": [("ml", 1.0)],
            "gujarati": [("gu", 1.0)],
            "odia": [("or", 1.0)],
            "punjabi": [("pa", 1.0)],
        }
    
    def detect_languages(self, text: str, hints: List[str] = None) -> List[LanguageDetection]:
        """
        FastText-style language detection with character n-gram fallback.
        
        Args:
            text: Input text to analyze
            hints: Optional language hints to improve detection
            
        Returns:
            List of detected languages with confidence scores
        """
        if not text.strip():
            return [LanguageDetection("en", 0.5, 100.0)]
        
        # Step 1: Script-based detection (primary method)
        script_scores = self._detect_scripts(text)
        
        # Step 2: FastText-style n-gram detection
        ngram_scores = self._fasttext_style_detection(text)
        
        # Step 3: Hinglish detection for code-mixed content
        hinglish_score = self._detect_hinglish(text)
        
        # Step 4: Language distribution calculation
        languages = self._calculate_language_distribution(
            text, script_scores, ngram_scores, hinglish_score, hints
        )
        
        # Step 5: Apply confidence thresholding
        languages = self._apply_confidence_thresholding(languages)
        
        return languages
    
    def detect_per_segment(self, text: str, segment_length: int = None) -> List[Tuple[str, List[LanguageDetection]]]:
        """
        Detect languages per text segment for code-mixed content.
        
        Args:
            text: Input text
            segment_length: Number of words per segment
            
        Returns:
            List of (segment, languages) tuples
        """
        if segment_length is None:
            segment_length = self.segment_length
        
        segments = []
        words = text.split()
        
        # Handle short texts
        if len(words) <= segment_length:
            return [(text, self.detect_languages(text))]
        
        # Create overlapping segments for better code-mix detection
        step_size = max(1, segment_length // 2)  # 50% overlap
        
        for i in range(0, len(words), step_size):
            segment_words = words[i:i + segment_length]
            if len(segment_words) < 3:  # Skip very short segments
                continue
                
            segment_text = " ".join(segment_words)
            segment_languages = self.detect_languages(segment_text)
            
            # Only include segments with confident detections
            if segment_languages and segment_languages[0].confidence > 0.3:
                segments.append((segment_text, segment_languages))
        
        # Ensure we have at least one segment
        if not segments:
            segments.append((text, self.detect_languages(text)))
        
        return segments
    
    def _detect_scripts(self, text: str) -> Dict[str, float]:
        """Detect scripts based on Unicode character ranges."""
        char_counts = Counter()
        total_meaningful_chars = 0
        
        for char in text:
            # Skip whitespace and punctuation for script detection
            if char.isspace() or not char.isalnum():
                continue
                
            char_code = ord(char)
            total_meaningful_chars += 1
            
            for script, (start, end) in self.script_ranges.items():
                if start <= char_code <= end:
                    char_counts[script] += 1
                    break
        
        if total_meaningful_chars == 0:
            return {}
        
        # Convert to percentages
        script_scores = {}
        for script, count in char_counts.items():
            script_scores[script] = count / total_meaningful_chars
        
        return script_scores
    
    def _fasttext_style_detection(self, text: str) -> Dict[str, float]:
        """
        FastText-style language detection using character n-grams.
        Uses multiple n-gram sizes and frequency-based scoring.
        """
        if len(text) < self.min_text_length:
            return {}
        
        text_lower = text.lower()
        
        # Extract n-grams of different sizes
        all_ngrams = Counter()
        for n in range(self.ngram_min, self.ngram_max + 1):
            ngrams = self._extract_ngrams(text_lower, n)
            all_ngrams.update(ngrams)
        
        if not all_ngrams:
            return {}
        
        # Calculate scores for each language
        scores = {}
        for lang in self.supported_languages:
            if lang in self.language_ngrams:
                score = self._calculate_ngram_score(all_ngrams, self.language_ngrams[lang])
                scores[lang] = score
        
        # Normalize scores using softmax
        scores = self._softmax_normalize(scores)
        
        return scores
    
    def _calculate_ngram_score(self, text_ngrams: Counter, lang_ngrams: Dict[str, float]) -> float:
        """Calculate language score based on n-gram frequencies."""
        score = 0.0
        total_count = sum(text_ngrams.values())
        
        if total_count == 0:
            return 0.0
        
        for ngram, count in text_ngrams.items():
            if ngram in lang_ngrams:
                # Use log probability for better numerical stability
                ngram_freq = count / total_count
                lang_freq = lang_ngrams[ngram]
                score += ngram_freq * math.log(lang_freq + 1e-10)
        
        return score
    
    def _softmax_normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply softmax normalization to scores."""
        if not scores:
            return {}
        
        # Shift scores to prevent overflow
        max_score = max(scores.values())
        shifted_scores = {lang: score - max_score for lang, score in scores.items()}
        
        # Calculate softmax
        exp_scores = {lang: math.exp(score) for lang, score in shifted_scores.items()}
        total_exp = sum(exp_scores.values())
        
        if total_exp == 0:
            return scores
        
        normalized = {lang: exp_score / total_exp for lang, exp_score in exp_scores.items()}
        return normalized
    
    def _detect_hinglish(self, text: str) -> float:
        """
        Enhanced Hinglish (Hindi-English code-mixed) detection.
        Uses pattern matching and script mixing analysis.
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        if len(words) == 0:
            return 0.0
        
        hinglish_indicators = 0
        
        # Pattern-based detection
        for pattern in self.hinglish_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            hinglish_indicators += len(matches)
        
        # Script mixing detection (Latin + Devanagari)
        has_latin = any(ord(char) in range(0x0041, 0x007A) for char in text)
        has_devanagari = any(ord(char) in range(0x0900, 0x097F) for char in text)
        
        if has_latin and has_devanagari:
            hinglish_indicators += len(words) * 0.3  # Boost for script mixing
        
        # Romanized Hindi word detection
        romanized_hindi_patterns = [
            r'\b\w*[aeiou]{2,}\w*\b',  # Vowel clusters common in romanized Hindi
            r'\b\w*[kg]h\w*\b',        # Aspirated consonants
            r'\b\w*[td]h\w*\b',        # More aspirated consonants
        ]
        
        for pattern in romanized_hindi_patterns:
            matches = re.findall(pattern, text_lower)
            hinglish_indicators += len(matches) * 0.1
        
        # Calculate final score
        hinglish_score = min(hinglish_indicators / len(words), 1.0)
        
        # Apply threshold for noise reduction
        return hinglish_score if hinglish_score > 0.1 else 0.0
    
    def _calculate_language_distribution(
        self, 
        text: str,
        script_scores: Dict[str, float],
        ngram_scores: Dict[str, float],
        hinglish_score: float,
        hints: List[str] = None
    ) -> List[LanguageDetection]:
        """Calculate language distribution combining all detection methods."""
        
        language_scores = defaultdict(float)
        language_percentages = defaultdict(float)
        
        # Process script-based detections
        for script, script_percentage in script_scores.items():
            if script in self.script_to_languages:
                for lang_code, weight in self.script_to_languages[script]:
                    base_confidence = script_percentage * weight
                    language_scores[lang_code] += base_confidence * 0.7  # Script weight
                    language_percentages[lang_code] += script_percentage * weight
        
        # Process Latin script with n-gram analysis
        latin_percentage = script_scores.get("latin", 0)
        if latin_percentage > 0.05:  # At least 5% Latin characters
            
            # Check for Hinglish first
            if hinglish_score > 0.15:
                hinglish_confidence = min(0.9, 0.3 + hinglish_score)
                language_scores["hi-Latn"] += hinglish_confidence * 0.8
                language_percentages["hi-Latn"] += latin_percentage
            else:
                # Use n-gram scores for Latin script languages
                english_ngram_score = ngram_scores.get("en", 0)
                hindi_ngram_score = ngram_scores.get("hi", 0)
                
                # If English n-gram score is reasonable or higher than Hindi, prefer English
                if english_ngram_score > 0.1 or english_ngram_score >= hindi_ngram_score:
                    confidence = min(0.9, 0.5 + english_ngram_score)
                    language_scores["en"] += confidence * 0.8  # Higher weight for script match
                    language_percentages["en"] += latin_percentage
                else:
                    # Still default to English for Latin script if no clear winner
                    language_scores["en"] += 0.6
                    language_percentages["en"] += latin_percentage
        
        # Apply language hints
        if hints:
            for hint in hints:
                if hint in self.supported_languages and hint in language_scores:
                    language_scores[hint] *= 1.2  # 20% boost for hints
        
        # Convert to LanguageDetection objects
        languages = []
        for lang_code, confidence in language_scores.items():
            if confidence > 0.1:  # Minimum confidence threshold
                percentage = language_percentages[lang_code] * 100
                languages.append(LanguageDetection(
                    code=lang_code,
                    confidence=min(0.95, confidence),  # Cap at 95%
                    percentage=percentage
                ))
        
        # Ensure we have at least one language
        if not languages:
            languages.append(LanguageDetection("en", 0.5, 100.0))
        
        # Normalize percentages to sum to 100
        total_percentage = sum(lang.percentage for lang in languages)
        if total_percentage > 0:
            for lang in languages:
                lang.percentage = (lang.percentage / total_percentage) * 100
        
        # Sort by confidence
        languages.sort(key=lambda x: x.confidence, reverse=True)
        
        return languages
    
    def _apply_confidence_thresholding(self, languages: List[LanguageDetection]) -> List[LanguageDetection]:
        """Apply confidence thresholding to filter low-confidence detections."""
        filtered_languages = []
        
        for lang in languages:
            if lang.confidence >= self.confidence_threshold:
                filtered_languages.append(lang)
            elif lang.confidence >= 0.3 and len(filtered_languages) == 0:
                # Keep at least one language even if below threshold
                filtered_languages.append(lang)
        
        # Ensure we always return at least one language
        if not filtered_languages and languages:
            filtered_languages.append(languages[0])
        elif not filtered_languages:
            filtered_languages.append(LanguageDetection("en", 0.5, 100.0))
        
        return filtered_languages
    
    def _extract_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Extract character n-grams from text with boundary markers."""
        if len(text) < n:
            return [text] if text.strip() else []
        
        # Add boundary markers for better n-gram quality
        bounded_text = f"<{text}>"
        
        ngrams = []
        for i in range(len(bounded_text) - n + 1):
            ngram = bounded_text[i:i + n]
            # Skip n-grams that are mostly whitespace or punctuation
            if len(ngram.strip()) >= n // 2:
                ngrams.append(ngram)
        
        return ngrams
    
    def _build_language_ngrams(self) -> Dict[str, Dict[str, float]]:
        """
        Build language-specific n-gram patterns with frequencies.
        In production, these would be learned from large corpora.
        """
        return {
            "en": {
                # Common English character trigrams with relative frequencies
                "the": 0.027, " th": 0.025, "he ": 0.022, "ing": 0.020, "and": 0.018,
                "ion": 0.016, "tio": 0.015, "ent": 0.014, "ati": 0.013, "for": 0.012,
                "ted": 0.011, "ter": 0.011, "hat": 0.010, "tha": 0.010, "ere": 0.010,
                "ate": 0.009, "his": 0.009, "con": 0.009, "res": 0.008, "ver": 0.008,
                # Bigrams
                "th": 0.031, "he": 0.028, "in": 0.023, "er": 0.020, "an": 0.020,
                "ed": 0.017, "nd": 0.017, "to": 0.016, "en": 0.015, "ti": 0.014,
                # Boundary patterns
                "<t": 0.015, "e>": 0.012, "<a": 0.011, "s>": 0.010, "<i": 0.009
            },
            "hi": {
                # Common Hindi character patterns with relative frequencies
                "का": 0.035, "की": 0.030, "के": 0.028, "में": 0.025, "से": 0.022,
                "को": 0.020, "पर": 0.018, "और": 0.016, "है": 0.015, "हैं": 0.014,
                "था": 0.013, "थी": 0.012, "थे": 0.011, "गा": 0.010, "गी": 0.009,
                "गे": 0.008, "ना": 0.008, "ने": 0.007, "नी": 0.007, "रा": 0.006,
                # Common character combinations
                "कर": 0.015, "हो": 0.012, "जा": 0.011, "दे": 0.010, "ले": 0.009,
                "मे": 0.008, "तक": 0.007, "भी": 0.007, "यह": 0.006, "वह": 0.006
            },
            "hi-Latn": {
                # Romanized Hindi patterns
                "hai": 0.025, "kar": 0.020, "kya": 0.018, "aur": 0.016, "mai": 0.015,
                "yeh": 0.014, "woh": 0.013, "bas": 0.012, "tha": 0.011, "hog": 0.010,
                "ach": 0.009, "sah": 0.008, "gal": 0.008, "thi": 0.007, "the": 0.007,
                # Common romanized endings
                "ga": 0.012, "gi": 0.011, "ge": 0.010, "na": 0.009, "ne": 0.008,
                "ni": 0.007, "ra": 0.007, "ri": 0.006, "re": 0.006, "ta": 0.006
            },
            "bn": {
                # Bengali character patterns
                "এর": 0.030, "তে": 0.025, "কে": 0.022, "হয়": 0.020, "যে": 0.018,
                "না": 0.016, "দে": 0.015, "সে": 0.014, "বে": 0.013, "লে": 0.012,
                "রে": 0.011, "নে": 0.010, "মে": 0.009, "গে": 0.008, "চে": 0.007
            },
            "te": {
                # Telugu character patterns
                "లు": 0.028, "కు": 0.025, "తో": 0.022, "లో": 0.020, "నా": 0.018,
                "దు": 0.016, "రు": 0.015, "వు": 0.014, "ము": 0.013, "గా": 0.012,
                "చు": 0.011, "టు": 0.010, "పు": 0.009, "బు": 0.008, "సు": 0.007
            },
            "ta": {
                # Tamil character patterns
                "கள்": 0.025, "ில்": 0.022, "ும்": 0.020, "ான": 0.018, "என": 0.016,
                "து": 0.015, "வது": 0.014, "ற்": 0.013, "ன்": 0.012, "ல்": 0.011,
                "ம்": 0.010, "க்": 0.009, "த்": 0.008, "ப்": 0.007, "ர்": 0.006
            }
        }