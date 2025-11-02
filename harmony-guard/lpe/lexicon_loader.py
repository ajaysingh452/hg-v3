"""Lexicon loading and management system for multilingual content moderation."""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class LexiconEntry:
    """Represents a single lexicon entry with metadata."""
    terms: List[str]
    category: str
    severity: str
    weight: float
    context_required: bool = False
    script: str = "native"
    variants: List[str] = None
    
    def __post_init__(self):
        if self.variants is None:
            self.variants = []


@dataclass
class PatternEntry:
    """Represents a regex pattern entry."""
    pattern: str
    base_term: str
    category: str
    severity: str
    weight: float = 0.8


@dataclass
class LanguageLexicon:
    """Complete lexicon for a specific language."""
    language: str
    version: str
    entries: List[LexiconEntry]
    patterns: List[PatternEntry]
    transliterations: Dict[str, str]
    morphological_variants: Dict[str, List[str]]


class LexiconManager:
    """Manages loading and accessing multilingual lexicons."""
    
    def __init__(self, lexicon_path: str, config_manager=None):
        """
        Initialize lexicon manager.
        
        Args:
            lexicon_path: Path to lexicon files directory
            config_manager: Configuration manager instance
        """
        self.lexicon_path = Path(lexicon_path)
        self.config_manager = config_manager
        self.lexicons: Dict[str, LanguageLexicon] = {}
        self.term_index: Dict[str, List[Tuple[str, LexiconEntry]]] = {}
        self.pattern_index: Dict[str, List[PatternEntry]] = {}
        
        # Supported languages
        self.supported_languages = [
            "en", "hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "or", "pa", "as"
        ]
        
        # Special lexicons
        self.emoji_lexicon = None
        self.leet_lexicon = None
    
    async def initialize(self):
        """Initialize lexicon manager by loading all lexicons."""
        logger.info("Initializing lexicon manager...")
        
        try:
            # Load language-specific lexicons
            for lang in self.supported_languages:
                await self._load_language_lexicon(lang)
            
            # Load special lexicons
            await self._load_emoji_lexicon()
            await self._load_leet_lexicon()
            
            # Build search indices
            self._build_indices()
            
            logger.info(f"Loaded lexicons for {len(self.lexicons)} languages")
            
        except Exception as e:
            logger.error(f"Failed to initialize lexicon manager: {e}")
            raise
    
    def get_lexicon(self, language: str) -> Optional[LanguageLexicon]:
        """Get lexicon for a specific language."""
        return self.lexicons.get(language)
    
    def search_terms(self, term: str, languages: List[str] = None) -> List[Tuple[str, LexiconEntry]]:
        """
        Search for a term across specified languages.
        
        Args:
            term: Term to search for
            languages: List of language codes to search in (None for all)
            
        Returns:
            List of (language, entry) tuples matching the term
        """
        results = []
        term_lower = term.lower()
        
        if term_lower in self.term_index:
            for lang, entry in self.term_index[term_lower]:
                if languages is None or lang in languages:
                    results.append((lang, entry))
        
        return results
    
    def get_patterns(self, language: str) -> List[PatternEntry]:
        """Get regex patterns for a specific language."""
        return self.pattern_index.get(language, [])
    
    def get_morphological_variants(self, term: str, language: str) -> List[str]:
        """Get morphological variants of a term."""
        lexicon = self.lexicons.get(language)
        if lexicon and term in lexicon.morphological_variants:
            return lexicon.morphological_variants[term]
        return []
    
    def get_transliteration(self, term: str, language: str) -> Optional[str]:
        """Get transliteration of a term."""
        lexicon = self.lexicons.get(language)
        if lexicon and term in lexicon.transliterations:
            return lexicon.transliterations[term]
        return None
    
    def get_category_terms(self, category: str, language: str = None) -> List[str]:
        """Get all terms for a specific category."""
        terms = []
        
        lexicons_to_search = [self.lexicons[language]] if language else self.lexicons.values()
        
        for lexicon in lexicons_to_search:
            for entry in lexicon.entries:
                if entry.category == category:
                    terms.extend(entry.terms)
        
        return terms
    
    def get_severity_terms(self, severity: str, language: str = None) -> List[str]:
        """Get all terms for a specific severity level."""
        terms = []
        
        lexicons_to_search = [self.lexicons[language]] if language else self.lexicons.values()
        
        for lexicon in lexicons_to_search:
            for entry in lexicon.entries:
                if entry.severity == severity:
                    terms.extend(entry.terms)
        
        return terms
    
    async def _load_language_lexicon(self, language: str):
        """Load lexicon for a specific language."""
        lexicon_file = self.lexicon_path / f"{language}.yaml"
        
        if not lexicon_file.exists():
            logger.warning(f"Lexicon file not found for language {language}: {lexicon_file}")
            return
        
        try:
            with open(lexicon_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Parse entries
            entries = []
            for entry_data in data.get("entries", []):
                entry = LexiconEntry(
                    terms=entry_data["terms"],
                    category=entry_data["category"],
                    severity=entry_data["severity"],
                    weight=entry_data["weight"],
                    context_required=entry_data.get("context_required", False),
                    script=entry_data.get("script", "native"),
                    variants=entry_data.get("variants", [])
                )
                entries.append(entry)
            
            # Parse patterns
            patterns = []
            for pattern_data in data.get("patterns", []):
                pattern = PatternEntry(
                    pattern=pattern_data["pattern"],
                    base_term=pattern_data["base_term"],
                    category=pattern_data["category"],
                    severity=pattern_data["severity"],
                    weight=pattern_data.get("weight", 0.8)
                )
                patterns.append(pattern)
            
            # Create lexicon object
            lexicon = LanguageLexicon(
                language=data["language"],
                version=data["version"],
                entries=entries,
                patterns=patterns,
                transliterations=data.get("transliteration", {}).get("romanized_to_devanagari", {}),
                morphological_variants=data.get("variants", {})
            )
            
            self.lexicons[language] = lexicon
            logger.info(f"Loaded {len(entries)} entries and {len(patterns)} patterns for {language}")
            
        except Exception as e:
            logger.error(f"Failed to load lexicon for {language}: {e}")
    
    async def _load_emoji_lexicon(self):
        """Load emoji lexicon."""
        emoji_file = self.lexicon_path / "emoji.yaml"
        
        if not emoji_file.exists():
            logger.warning(f"Emoji lexicon file not found: {emoji_file}")
            return
        
        try:
            with open(emoji_file, 'r', encoding='utf-8') as f:
                self.emoji_lexicon = yaml.safe_load(f)
            
            logger.info("Loaded emoji lexicon")
            
        except Exception as e:
            logger.error(f"Failed to load emoji lexicon: {e}")
    
    async def _load_leet_lexicon(self):
        """Load leet speak lexicon."""
        leet_file = self.lexicon_path / "leet.yaml"
        
        if not leet_file.exists():
            logger.warning(f"Leet lexicon file not found: {leet_file}")
            return
        
        try:
            with open(leet_file, 'r', encoding='utf-8') as f:
                self.leet_lexicon = yaml.safe_load(f)
            
            logger.info("Loaded leet speak lexicon")
            
        except Exception as e:
            logger.error(f"Failed to load leet lexicon: {e}")
    
    def _build_indices(self):
        """Build search indices for fast term lookup."""
        logger.info("Building lexicon search indices...")
        
        # Build term index
        for lang, lexicon in self.lexicons.items():
            for entry in lexicon.entries:
                for term in entry.terms:
                    term_lower = term.lower()
                    if term_lower not in self.term_index:
                        self.term_index[term_lower] = []
                    self.term_index[term_lower].append((lang, entry))
                
                # Also index variants
                for variant in entry.variants:
                    variant_lower = variant.lower()
                    if variant_lower not in self.term_index:
                        self.term_index[variant_lower] = []
                    self.term_index[variant_lower].append((lang, entry))
        
        # Build pattern index
        for lang, lexicon in self.lexicons.items():
            if lexicon.patterns:
                self.pattern_index[lang] = lexicon.patterns
        
        logger.info(f"Built indices with {len(self.term_index)} terms and {len(self.pattern_index)} pattern sets")
    
    def get_emoji_data(self) -> Optional[Dict]:
        """Get emoji lexicon data."""
        return self.emoji_lexicon
    
    def get_leet_data(self) -> Optional[Dict]:
        """Get leet speak lexicon data."""
        return self.leet_lexicon
    
    def reload_lexicon(self, language: str):
        """Reload lexicon for a specific language."""
        logger.info(f"Reloading lexicon for {language}")
        
        # Remove old data
        if language in self.lexicons:
            del self.lexicons[language]
        
        # Remove from indices
        terms_to_remove = []
        for term, entries in self.term_index.items():
            self.term_index[term] = [(lang, entry) for lang, entry in entries if lang != language]
            if not self.term_index[term]:
                terms_to_remove.append(term)
        
        for term in terms_to_remove:
            del self.term_index[term]
        
        if language in self.pattern_index:
            del self.pattern_index[language]
        
        # Reload
        import asyncio
        asyncio.create_task(self._load_language_lexicon(language))
        
        # Rebuild indices for this language
        if language in self.lexicons:
            lexicon = self.lexicons[language]
            for entry in lexicon.entries:
                for term in entry.terms:
                    term_lower = term.lower()
                    if term_lower not in self.term_index:
                        self.term_index[term_lower] = []
                    self.term_index[term_lower].append((language, entry))
            
            if lexicon.patterns:
                self.pattern_index[language] = lexicon.patterns
    
    def get_statistics(self) -> Dict[str, any]:
        """Get lexicon statistics."""
        stats = {
            "languages_loaded": len(self.lexicons),
            "total_terms": len(self.term_index),
            "total_patterns": sum(len(patterns) for patterns in self.pattern_index.values()),
            "languages": {}
        }
        
        for lang, lexicon in self.lexicons.items():
            stats["languages"][lang] = {
                "entries": len(lexicon.entries),
                "patterns": len(lexicon.patterns),
                "transliterations": len(lexicon.transliterations),
                "variants": len(lexicon.morphological_variants)
            }
        
        return stats