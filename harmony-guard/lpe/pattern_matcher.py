"""Advanced pattern matching engine with trie-based lookup and regex support."""

import re
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Represents a pattern match result."""
    text: str
    start: int
    end: int
    category: str
    severity: str
    weight: float
    match_type: str  # "exact", "fuzzy", "pattern", "variant"
    rule_source: str


class TrieNode:
    """Node in a trie data structure for efficient string matching."""
    
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_word = False
        self.entries: List = []  # Store lexicon entries for this word
        self.word = ""


class PatternMatcher:
    """Advanced pattern matching engine with multiple matching strategies."""
    
    def __init__(self, config: Dict):
        """Initialize pattern matcher with configuration."""
        self.config = config
        self.fuzzy_matching = config.get("fuzzy_matching", True)
        self.fuzzy_threshold = config.get("fuzzy_threshold", 0.8)
        self.morphological_variants = config.get("morphological_variants", True)
        
        # Trie for exact matching
        self.trie_root = TrieNode()
        
        # Compiled regex patterns
        self.regex_patterns: List[Tuple[re.Pattern, str, str, str, float]] = []
        
        # Fuzzy matching cache
        self.fuzzy_cache: Dict[str, List[MatchResult]] = {}
        
        # Character similarity map for fuzzy matching
        self.char_similarity = self._build_char_similarity_map()
    
    def build_trie(self, lexicon_manager):
        """Build trie from lexicon data for efficient exact matching."""
        logger.info("Building trie for exact matching...")
        
        total_terms = 0
        
        for lang, lexicon in lexicon_manager.lexicons.items():
            for entry in lexicon.entries:
                for term in entry.terms:
                    self._insert_term(term.lower(), lang, entry)
                    total_terms += 1
                
                # Also insert variants
                for variant in entry.variants:
                    self._insert_term(variant.lower(), lang, entry)
                    total_terms += 1
        
        logger.info(f"Built trie with {total_terms} terms")
    
    def compile_patterns(self, lexicon_manager):
        """Compile regex patterns for pattern-based matching."""
        logger.info("Compiling regex patterns...")
        
        pattern_count = 0
        
        for lang, lexicon in lexicon_manager.lexicons.items():
            for pattern_entry in lexicon.patterns:
                try:
                    compiled_pattern = re.compile(
                        pattern_entry.pattern, 
                        re.IGNORECASE | re.UNICODE
                    )
                    
                    self.regex_patterns.append((
                        compiled_pattern,
                        pattern_entry.category,
                        pattern_entry.severity,
                        f"pattern:{lang}:{pattern_entry.base_term}",
                        pattern_entry.weight
                    ))
                    
                    pattern_count += 1
                    
                except re.error as e:
                    logger.warning(f"Failed to compile pattern '{pattern_entry.pattern}': {e}")
        
        logger.info(f"Compiled {pattern_count} regex patterns")
    
    def find_matches(self, text: str, languages: List[str] = None) -> List[MatchResult]:
        """
        Find all matches in text using multiple matching strategies.
        
        Args:
            text: Input text to search
            languages: List of languages to search in (None for all)
            
        Returns:
            List of match results
        """
        matches = []
        
        # Strategy 1: Exact trie-based matching
        exact_matches = self._find_exact_matches(text)
        matches.extend(exact_matches)
        
        # Strategy 2: Regex pattern matching
        pattern_matches = self._find_pattern_matches(text)
        matches.extend(pattern_matches)
        
        # Strategy 3: Fuzzy matching (if enabled)
        if self.fuzzy_matching:
            fuzzy_matches = self._find_fuzzy_matches(text, languages)
            matches.extend(fuzzy_matches)
        
        # Remove duplicates and overlaps
        matches = self._deduplicate_matches(matches)
        
        # Sort by position
        matches.sort(key=lambda m: (m.start, -m.weight))
        
        return matches
    
    def _insert_term(self, term: str, language: str, entry):
        """Insert a term into the trie."""
        node = self.trie_root
        
        for char in term:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_word = True
        node.word = term
        node.entries.append((language, entry))
    
    def _find_exact_matches(self, text: str) -> List[MatchResult]:
        """Find exact matches using trie-based search."""
        matches = []
        text_lower = text.lower()
        
        for i in range(len(text_lower)):
            node = self.trie_root
            j = i
            
            while j < len(text_lower) and text_lower[j] in node.children:
                node = node.children[text_lower[j]]
                j += 1
                
                if node.is_end_word:
                    # Found a complete word
                    matched_text = text[i:j]
                    
                    for language, entry in node.entries:
                        match = MatchResult(
                            text=matched_text,
                            start=i,
                            end=j,
                            category=entry.category,
                            severity=entry.severity,
                            weight=entry.weight,
                            match_type="exact",
                            rule_source=f"trie:{language}:{node.word}"
                        )
                        matches.append(match)
        
        return matches
    
    def _find_pattern_matches(self, text: str) -> List[MatchResult]:
        """Find matches using compiled regex patterns."""
        matches = []
        
        for pattern, category, severity, rule_source, weight in self.regex_patterns:
            for match in pattern.finditer(text):
                result = MatchResult(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    category=category,
                    severity=severity,
                    weight=weight,
                    match_type="pattern",
                    rule_source=rule_source
                )
                matches.append(result)
        
        return matches
    
    def _find_fuzzy_matches(self, text: str, languages: List[str] = None) -> List[MatchResult]:
        """Find fuzzy matches using approximate string matching."""
        matches = []
        
        # Simple fuzzy matching implementation
        # In production, use more sophisticated algorithms like Levenshtein distance
        
        words = text.lower().split()
        
        for word in words:
            if len(word) < 3:  # Skip very short words
                continue
            
            # Check cache first
            cache_key = f"{word}:{','.join(languages) if languages else 'all'}"
            if cache_key in self.fuzzy_cache:
                matches.extend(self.fuzzy_cache[cache_key])
                continue
            
            fuzzy_matches = self._fuzzy_match_word(word, languages)
            self.fuzzy_cache[cache_key] = fuzzy_matches
            matches.extend(fuzzy_matches)
        
        return matches
    
    def _fuzzy_match_word(self, word: str, languages: List[str] = None) -> List[MatchResult]:
        """Perform fuzzy matching for a single word."""
        matches = []
        
        # Simple character-based fuzzy matching
        # This is a simplified implementation
        
        for candidate_word, entries in self._get_candidate_words(word, languages):
            similarity = self._calculate_similarity(word, candidate_word)
            
            if similarity >= self.fuzzy_threshold:
                for language, entry in entries:
                    # Find word position in original text (approximate)
                    # In production, maintain proper position tracking
                    match = MatchResult(
                        text=word,
                        start=0,  # Would need proper position tracking
                        end=len(word),
                        category=entry.category,
                        severity=entry.severity,
                        weight=entry.weight * similarity,  # Reduce weight by similarity
                        match_type="fuzzy",
                        rule_source=f"fuzzy:{language}:{candidate_word}"
                    )
                    matches.append(match)
        
        return matches
    
    def _get_candidate_words(self, word: str, languages: List[str] = None) -> List[Tuple[str, List]]:
        """Get candidate words for fuzzy matching."""
        candidates = []
        
        # Traverse trie to get all words within edit distance
        # This is a simplified implementation
        
        def traverse_trie(node: TrieNode, current_word: str, max_distance: int):
            if node.is_end_word and len(node.entries) > 0:
                candidates.append((current_word, node.entries))
            
            if max_distance > 0:
                for char, child_node in node.children.items():
                    # Simple heuristic: continue if character is similar or within distance
                    traverse_trie(child_node, current_word + char, max_distance - 1)
        
        # Start traversal with limited depth
        max_edit_distance = max(1, len(word) // 4)  # Allow 25% character differences
        traverse_trie(self.trie_root, "", max_edit_distance)
        
        return candidates
    
    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words."""
        if word1 == word2:
            return 1.0
        
        # Simple character-based similarity
        # In production, use proper edit distance algorithms
        
        if len(word1) == 0 or len(word2) == 0:
            return 0.0
        
        # Character overlap similarity
        chars1 = set(word1)
        chars2 = set(word2)
        
        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)
        
        if union == 0:
            return 0.0
        
        char_similarity = intersection / union
        
        # Length similarity
        length_similarity = 1.0 - abs(len(word1) - len(word2)) / max(len(word1), len(word2))
        
        # Combined similarity
        return (char_similarity + length_similarity) / 2.0
    
    def _deduplicate_matches(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Remove duplicate and overlapping matches."""
        if not matches:
            return matches
        
        # Sort by position and weight
        matches.sort(key=lambda m: (m.start, m.end, -m.weight))
        
        deduplicated = []
        
        for match in matches:
            # Check for overlap with existing matches
            overlaps = False
            
            for existing in deduplicated:
                if self._matches_overlap(match, existing):
                    # Keep the match with higher weight
                    if match.weight > existing.weight:
                        deduplicated.remove(existing)
                        deduplicated.append(match)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(match)
        
        return deduplicated
    
    def _matches_overlap(self, match1: MatchResult, match2: MatchResult) -> bool:
        """Check if two matches overlap."""
        return not (match1.end <= match2.start or match2.end <= match1.start)
    
    def _build_char_similarity_map(self) -> Dict[str, Set[str]]:
        """Build character similarity map for fuzzy matching."""
        similarity_map = {}
        
        # Similar looking characters
        similar_groups = [
            ['a', 'à', 'á', 'â', 'ã', 'ä', 'å', '@', '4'],
            ['e', 'è', 'é', 'ê', 'ë', '3'],
            ['i', 'ì', 'í', 'î', 'ï', '!', '1', 'l'],
            ['o', 'ò', 'ó', 'ô', 'õ', 'ö', '0'],
            ['u', 'ù', 'ú', 'û', 'ü'],
            ['s', '$', '5'],
            ['t', '+', '7'],
            ['b', '8'],
            ['c', 'ç'],
            ['n', 'ñ']
        ]
        
        for group in similar_groups:
            for char in group:
                similarity_map[char] = set(group) - {char}
        
        return similarity_map
    
    def get_statistics(self) -> Dict[str, any]:
        """Get pattern matcher statistics."""
        return {
            "trie_nodes": self._count_trie_nodes(),
            "regex_patterns": len(self.regex_patterns),
            "fuzzy_cache_size": len(self.fuzzy_cache),
            "fuzzy_matching_enabled": self.fuzzy_matching,
            "fuzzy_threshold": self.fuzzy_threshold
        }
    
    def _count_trie_nodes(self) -> int:
        """Count total nodes in trie."""
        def count_nodes(node: TrieNode) -> int:
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count
        
        return count_nodes(self.trie_root)