"""Text preprocessing pipeline for Harmony Guard."""

import re
import unicodedata
from typing import List, Dict, Optional
import logging

from .models import ProcessedText, LanguageDetection
from .interfaces import TextPreprocessorInterface
from .language_detection import LanguageIdentifier
from .normalization import TextNormalizer
from .transliteration import TransliterationEngine
from .obfuscation import ObfuscationHandler
from .tokenization import MultilingualTokenizer
from .pii_masking import PIIMasker


logger = logging.getLogger(__name__)


class TextPreprocessor(TextPreprocessorInterface):
    """Comprehensive text preprocessing pipeline with all normalization components."""
    
    def __init__(self, config: Dict):
        """
        Initialize text preprocessor with all components.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config["preprocessing"]
        
        # Initialize all preprocessing components
        self.language_identifier = LanguageIdentifier(self.config["language_detection"])
        self.text_normalizer = TextNormalizer(self.config["normalization"])
        self.transliteration_engine = TransliterationEngine(self.config["transliteration"])
        self.obfuscation_handler = ObfuscationHandler(self.config["obfuscation"])
        self.tokenizer = MultilingualTokenizer(self.config["tokenization"])
        self.pii_masker = PIIMasker(self.config["pii_masking"])
        
    async def initialize(self):
        """Initialize preprocessing components."""
        logger.info("Initializing comprehensive text preprocessor...")
        # All components are initialized in __init__
        logger.info("Text preprocessor initialized with all components")
    
    async def process(self, text: str, language_hints: List[str] = None) -> ProcessedText:
        """
        Process raw text through comprehensive normalization pipeline.
        
        Args:
            text: Raw input text
            language_hints: Optional language hints
            
        Returns:
            ProcessedText with normalized content and metadata
        """
        try:
            # Step 1: Language detection
            detected_languages = self.language_identifier.detect_languages(text, language_hints)
            
            # Step 2: Text normalization (Unicode, homoglyphs, etc.)
            normalized_text = self.text_normalizer.normalize(text)
            
            # Step 3: Obfuscation handling (leet speak, elongation, etc.)
            normalized_text, obfuscation_map = self.obfuscation_handler.normalize_obfuscations(normalized_text)
            
            # Step 4: Transliteration
            transliterations = {}
            for lang in detected_languages:
                if lang.code in ["hi-Latn"]:  # Hinglish
                    trans_to_native = self.transliteration_engine.transliterate_to_native(
                        normalized_text, "hi"
                    )
                    transliterations.update(trans_to_native)
                elif lang.code in ["hi", "bn", "ta"]:  # Native scripts
                    trans_to_roman = self.transliteration_engine.transliterate_to_roman(
                        normalized_text, lang.code
                    )
                    transliterations.update(trans_to_roman)
            
            # Step 5: Tokenization
            tokens = self.tokenizer.tokenize(normalized_text)
            
            # Step 6: PII masking (if enabled)
            pii_masked = False
            if self.pii_masker.enabled:
                normalized_text, pii_masked, _ = self.pii_masker.mask_pii(normalized_text)
            
            return ProcessedText(
                original_text=text,
                normalized_text=normalized_text,
                detected_languages=detected_languages,
                tokens=tokens,
                transliterations=transliterations,
                obfuscation_map=obfuscation_map,
                pii_masked=pii_masked
            )
            
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            # Return minimal processed text on error
            return ProcessedText(
                original_text=text,
                normalized_text=text.lower().strip(),
                detected_languages=[LanguageDetection("en", 0.5, 100.0)],
                tokens=text.split(),
                transliterations={},
                obfuscation_map={}
            )
    
    def get_preprocessing_stats(self, processed_text: ProcessedText) -> Dict[str, any]:
        """Get statistics about the preprocessing results."""
        return {
            "original_length": len(processed_text.original_text),
            "normalized_length": len(processed_text.normalized_text),
            "detected_languages": [lang.code for lang in processed_text.detected_languages],
            "token_count": len(processed_text.tokens),
            "obfuscations_found": len(processed_text.obfuscation_map),
            "transliterations_found": len(processed_text.transliterations),
            "pii_masked": processed_text.pii_masked
        }