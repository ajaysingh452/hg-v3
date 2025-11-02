"""Safe context detection for HR reporting, educational content, and legitimate use cases."""

import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass


@dataclass
class SafeContext:
    """Represents a detected safe context."""
    context_type: str
    confidence: float
    indicators: List[str]
    start_pos: int = None
    end_pos: int = None


class SafeContextDetector:
    """Detects safe contexts where potentially problematic content may be legitimate."""
    
    def __init__(self):
        # HR and workplace safety contexts
        self.hr_indicators = {
            'english': [
                'hr report', 'incident report', 'harassment complaint', 'workplace investigation',
                'employee complaint', 'disciplinary action', 'policy violation', 'misconduct report',
                'grievance', 'whistleblower', 'compliance report', 'investigation findings',
                'witness statement', 'formal complaint', 'workplace harassment', 'discrimination report',
                'safety incident', 'workplace violence', 'hostile work environment'
            ],
            'hindi': [
                'शिकायत', 'रिपोर्ट', 'जांच', 'कार्यस्थल', 'उत्पीड़न', 'भेदभाव',
                'shikayat', 'report', 'jaanch', 'karyasthal', 'utpeedan', 'bhedbhav'
            ]
        }
        
        # Educational and training contexts
        self.educational_indicators = {
            'english': [
                'training material', 'educational content', 'awareness program', 'sensitivity training',
                'example of inappropriate', 'what not to say', 'inappropriate behavior example',
                'case study', 'training scenario', 'role play', 'simulation', 'workshop material',
                'policy training', 'compliance training', 'diversity training', 'inclusion training',
                'anti-harassment training', 'prevention program'
            ],
            'hindi': [
                'प्रशिक्षण', 'शिक्षा', 'जागरूकता', 'उदाहरण', 'केस स्टडी',
                'prashikshan', 'shiksha', 'jagrukta', 'udaharan', 'case study'
            ]
        }
        
        # Legal and documentation contexts
        self.legal_indicators = {
            'english': [
                'legal document', 'court filing', 'deposition', 'testimony', 'evidence',
                'legal brief', 'case law', 'statute', 'regulation', 'policy document',
                'terms of service', 'user agreement', 'privacy policy', 'code of conduct',
                'compliance documentation', 'audit report', 'regulatory filing'
            ],
            'hindi': [
                'कानूनी', 'न्यायालय', 'साक्ष्य', 'गवाही', 'दस्तावेज',
                'kanooni', 'nyayalay', 'sakshya', 'gavahi', 'dastavez'
            ]
        }
        
        # Research and academic contexts
        self.research_indicators = {
            'english': [
                'research study', 'academic paper', 'thesis', 'dissertation', 'survey data',
                'interview transcript', 'focus group', 'ethnographic study', 'content analysis',
                'linguistic research', 'social media study', 'behavioral research',
                'psychological study', 'sociological research'
            ],
            'hindi': [
                'अनुसंधान', 'अध्ययन', 'शोध', 'सर्वेक्षण', 'साक्षात्कार',
                'anusandhan', 'adhyayan', 'shodh', 'sarvekshan', 'sakshatkaar'
            ]
        }
        
        # News and journalism contexts
        self.journalism_indicators = {
            'english': [
                'news report', 'journalism', 'press release', 'media coverage', 'breaking news',
                'investigative report', 'news article', 'editorial', 'opinion piece',
                'correspondent report', 'field report', 'exclusive story'
            ],
            'hindi': [
                'समाचार', 'पत्रकारिता', 'रिपोर्ट', 'मीडिया', 'खबर',
                'samachar', 'patrakarita', 'report', 'media', 'khabar'
            ]
        }
        
        # Compile all indicators
        self.all_indicators = {
            'hr_reporting': self.hr_indicators,
            'educational': self.educational_indicators,
            'legal_documentation': self.legal_indicators,
            'research_academic': self.research_indicators,
            'journalism': self.journalism_indicators
        }
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.context_regexes = {}
        
        for context_type, lang_indicators in self.all_indicators.items():
            all_terms = []
            for lang_terms in lang_indicators.values():
                all_terms.extend(lang_terms)
            
            # Create pattern that matches any of the terms
            pattern = r'\b(?:' + '|'.join(re.escape(term) for term in all_terms) + r')\b'
            self.context_regexes[context_type] = re.compile(pattern, re.IGNORECASE)
    
    def detect_safe_contexts(self, text: str) -> List[SafeContext]:
        """
        Detect safe contexts in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of SafeContext objects
        """
        detected_contexts = []
        
        for context_type, regex in self.context_regexes.items():
            matches = list(regex.finditer(text))
            
            if matches:
                # Calculate confidence based on number and strength of indicators
                confidence = min(0.9, 0.3 + (len(matches) * 0.2))
                
                indicators = [match.group() for match in matches]
                
                detected_contexts.append(SafeContext(
                    context_type=context_type,
                    confidence=confidence,
                    indicators=indicators,
                    start_pos=matches[0].start(),
                    end_pos=matches[-1].end()
                ))
        
        return detected_contexts
    
    def is_safe_context(self, text: str, threshold: float = 0.5) -> Tuple[bool, List[SafeContext]]:
        """
        Determine if the text is in a safe context.
        
        Args:
            text: Input text to analyze
            threshold: Minimum confidence threshold for safe context
            
        Returns:
            Tuple of (is_safe, detected_contexts)
        """
        contexts = self.detect_safe_contexts(text)
        
        # Check if any context meets the threshold
        is_safe = any(context.confidence >= threshold for context in contexts)
        
        return is_safe, contexts
    
    def get_context_modifiers(self, detected_contexts: List[SafeContext]) -> Dict[str, float]:
        """
        Get confidence modifiers based on detected safe contexts.
        
        Args:
            detected_contexts: List of detected safe contexts
            
        Returns:
            Dictionary of category modifiers
        """
        if not detected_contexts:
            return {}
        
        # Calculate overall safe context confidence
        max_confidence = max(context.confidence for context in detected_contexts)
        
        # Different contexts have different modifier strengths
        context_weights = {
            'hr_reporting': 0.8,      # Strong modifier - HR reports often contain problematic content
            'educational': 0.7,       # Strong modifier - Training materials show examples
            'legal_documentation': 0.6, # Moderate modifier - Legal docs may quote problematic content
            'research_academic': 0.5,  # Moderate modifier - Research may analyze problematic content
            'journalism': 0.4         # Lower modifier - News reports problematic events
        }
        
        # Calculate weighted modifier
        total_weight = 0
        weighted_modifier = 0
        
        for context in detected_contexts:
            weight = context_weights.get(context.context_type, 0.3)
            weighted_modifier += context.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            final_modifier = weighted_modifier / total_weight
        else:
            final_modifier = max_confidence * 0.3
        
        # Apply modifier to all abuse categories
        categories = [
            'insult/harassment', 'obscenity/profanity', 'hate/targeted group',
            'threat/violence', 'sexual content', 'bullying/taunting', 'self-harm encouragement'
        ]
        
        return {category: final_modifier for category in categories}
    
    def analyze_safe_context(self, text: str) -> Dict[str, any]:
        """
        Comprehensive safe context analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with safe context analysis results
        """
        detected_contexts = self.detect_safe_contexts(text)
        is_safe, _ = self.is_safe_context(text)
        
        context_types = [context.context_type for context in detected_contexts]
        max_confidence = max([context.confidence for context in detected_contexts], default=0.0)
        
        # Get all indicators found
        all_indicators = []
        for context in detected_contexts:
            all_indicators.extend(context.indicators)
        
        return {
            'is_safe_context': is_safe,
            'detected_contexts': detected_contexts,
            'context_types': context_types,
            'max_confidence': max_confidence,
            'context_modifiers': self.get_context_modifiers(detected_contexts),
            'indicators_found': list(set(all_indicators)),  # Remove duplicates
            'context_count': len(detected_contexts)
        }