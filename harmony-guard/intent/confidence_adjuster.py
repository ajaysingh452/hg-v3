"""Context-aware confidence adjustment for different abuse categories."""

import math
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.models import ProcessedText, LPEResult, ClassifierResult, ContextResult, DecisionType, AbuseCategory


@dataclass
class ConfidenceAdjustment:
    """Represents a confidence adjustment with reasoning."""
    original_confidence: float
    adjusted_confidence: float
    adjustment_factor: float
    reasoning: List[str]


class ContextAwareConfidenceAdjuster:
    """
    Adjusts confidence scores based on contextual analysis for different abuse categories.
    """
    
    def __init__(self):
        # Category-specific adjustment parameters
        self.category_sensitivity = {
            AbuseCategory.THREAT_VIOLENCE.value: {
                'negation_sensitivity': 0.8,    # High - threats are often negated in safe contexts
                'quotation_sensitivity': 0.7,   # High - often quoted in reports
                'safe_context_sensitivity': 0.9  # Very high - critical in HR/legal contexts
            },
            AbuseCategory.HATE_TARGETED.value: {
                'negation_sensitivity': 0.7,
                'quotation_sensitivity': 0.6,
                'safe_context_sensitivity': 0.8
            },
            AbuseCategory.INSULT_HARASSMENT.value: {
                'negation_sensitivity': 0.6,
                'quotation_sensitivity': 0.5,
                'safe_context_sensitivity': 0.7
            },
            AbuseCategory.OBSCENITY_PROFANITY.value: {
                'negation_sensitivity': 0.4,    # Lower - profanity is often problematic even when negated
                'quotation_sensitivity': 0.3,   # Lower - quoted profanity still problematic
                'safe_context_sensitivity': 0.6
            },
            AbuseCategory.SEXUAL_CONTENT.value: {
                'negation_sensitivity': 0.5,
                'quotation_sensitivity': 0.4,
                'safe_context_sensitivity': 0.8  # High - often legitimate in HR contexts
            },
            AbuseCategory.BULLYING_TAUNTING.value: {
                'negation_sensitivity': 0.6,
                'quotation_sensitivity': 0.5,
                'safe_context_sensitivity': 0.7
            },
            AbuseCategory.SELF_HARM.value: {
                'negation_sensitivity': 0.9,    # Very high - "don't harm yourself" is positive
                'quotation_sensitivity': 0.8,   # High - often quoted in support contexts
                'safe_context_sensitivity': 0.9
            }
        }
        
        # Default sensitivity for unknown categories
        self.default_sensitivity = {
            'negation_sensitivity': 0.6,
            'quotation_sensitivity': 0.5,
            'safe_context_sensitivity': 0.7
        }
        
        # Language-specific adjustment factors
        self.language_adjustments = {
            'hi': 1.0,      # Hindi baseline
            'en': 1.0,      # English baseline
            'hi-latn': 0.9, # Hinglish - slightly lower confidence due to transliteration ambiguity
            'mixed': 0.85   # Code-mixed content
        }
    
    def adjust_confidence_scores(
        self,
        processed_text: ProcessedText,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult
    ) -> Dict[str, ConfidenceAdjustment]:
        """
        Adjust confidence scores based on contextual analysis.
        
        Args:
            processed_text: Preprocessed text input
            lpe_result: Result from lexicon engine
            classifier_result: Result from ML classifier
            context_result: Result from context analysis
            
        Returns:
            Dictionary mapping categories to ConfidenceAdjustment objects
        """
        adjustments = {}
        
        # Get all categories from both LPE and classifier results
        all_categories = set(lpe_result.categories)
        all_categories.update(classifier_result.category_probabilities.keys())
        
        for category in all_categories:
            # Get original confidence from both sources
            lpe_confidence = lpe_result.confidence_scores.get(category, 0.0)
            classifier_confidence = classifier_result.category_probabilities.get(category, 0.0)
            
            # Use the higher confidence as the base
            original_confidence = max(lpe_confidence, classifier_confidence)
            
            if original_confidence > 0:
                adjustment = self._calculate_category_adjustment(
                    category, original_confidence, processed_text, context_result
                )
                adjustments[category] = adjustment
        
        return adjustments
    
    def _calculate_category_adjustment(
        self,
        category: str,
        original_confidence: float,
        processed_text: ProcessedText,
        context_result: ContextResult
    ) -> ConfidenceAdjustment:
        """Calculate confidence adjustment for a specific category."""
        
        reasoning = []
        adjustment_factor = 1.0
        
        # Get category-specific sensitivities
        sensitivities = self.category_sensitivity.get(category, self.default_sensitivity)
        
        # Apply negation adjustment
        if context_result.safe_context_detected.get('has_negation', False):
            negation_factor = 1.0 - sensitivities['negation_sensitivity']
            adjustment_factor *= negation_factor
            reasoning.append(f"Negation detected (adjustment: {negation_factor:.2f})")
        
        # Apply quotation adjustment
        if context_result.safe_context_detected.get('has_quotation', False):
            quotation_factor = 1.0 - sensitivities['quotation_sensitivity']
            adjustment_factor *= quotation_factor
            reasoning.append(f"Quotation detected (adjustment: {quotation_factor:.2f})")
        
        # Apply safe context adjustments
        safe_context_factor = self._calculate_safe_context_factor(
            context_result, sensitivities['safe_context_sensitivity']
        )
        if safe_context_factor < 1.0:
            adjustment_factor *= safe_context_factor
            reasoning.append(f"Safe context detected (adjustment: {safe_context_factor:.2f})")
        
        # Apply language-specific adjustments
        language_factor = self._calculate_language_factor(processed_text)
        if language_factor != 1.0:
            adjustment_factor *= language_factor
            reasoning.append(f"Language adjustment (factor: {language_factor:.2f})")
        
        # Apply confidence calibration based on original score
        calibration_factor = self._apply_confidence_calibration(original_confidence)
        adjustment_factor *= calibration_factor
        if calibration_factor != 1.0:
            reasoning.append(f"Confidence calibration (factor: {calibration_factor:.2f})")
        
        # Calculate final adjusted confidence
        adjusted_confidence = original_confidence * adjustment_factor
        
        # Ensure adjusted confidence stays within bounds
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        return ConfidenceAdjustment(
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence,
            adjustment_factor=adjustment_factor,
            reasoning=reasoning
        )
    
    def _calculate_safe_context_factor(
        self, 
        context_result: ContextResult, 
        sensitivity: float
    ) -> float:
        """Calculate adjustment factor based on safe contexts."""
        
        safe_context_types = [
            'hr_reporting', 'educational_content', 'legal_documentation',
            'research_academic', 'journalism'
        ]
        
        # Weight different safe contexts differently
        context_weights = {
            'hr_reporting': 0.9,        # Highest weight - HR reports often contain problematic content legitimately
            'educational_content': 0.8,  # High weight - Training materials show examples
            'legal_documentation': 0.7,  # Moderate-high weight - Legal docs quote problematic content
            'research_academic': 0.6,    # Moderate weight - Research analyzes problematic content
            'journalism': 0.5           # Lower weight - News reports problematic events but context matters
        }
        
        max_weight = 0.0
        for context_type in safe_context_types:
            if context_result.safe_context_detected.get(context_type, False):
                weight = context_weights.get(context_type, 0.3)
                max_weight = max(max_weight, weight)
        
        if max_weight > 0:
            # Apply the sensitivity factor
            return 1.0 - (sensitivity * max_weight)
        
        return 1.0
    
    def _calculate_language_factor(self, processed_text: ProcessedText) -> float:
        """Calculate adjustment factor based on detected languages."""
        
        if not processed_text.detected_languages:
            return 1.0
        
        # Get the primary language (highest confidence)
        primary_lang = max(processed_text.detected_languages, key=lambda x: x.confidence)
        
        # Check for code-mixing
        if len(processed_text.detected_languages) > 1:
            # Multiple languages detected - code-mixed content
            return self.language_adjustments.get('mixed', 0.85)
        
        # Single language
        return self.language_adjustments.get(primary_lang.code, 1.0)
    
    def _apply_confidence_calibration(self, original_confidence: float) -> float:
        """
        Apply confidence calibration to adjust for model overconfidence/underconfidence.
        
        This uses a sigmoid-like function to calibrate confidence scores.
        """
        # Sigmoid calibration parameters (these would typically be learned from validation data)
        # For now, using reasonable defaults
        
        if original_confidence < 0.1:
            # Very low confidence - slight boost
            return 1.1
        elif original_confidence > 0.9:
            # Very high confidence - slight reduction to account for overconfidence
            return 0.95
        else:
            # Middle range - apply sigmoid calibration
            # This formula slightly reduces mid-range confidences to be more conservative
            calibrated = 1.0 / (1.0 + math.exp(-5 * (original_confidence - 0.5)))
            return calibrated / original_confidence if original_confidence > 0 else 1.0
    
    def generate_recommended_actions(
        self,
        adjusted_confidences: Dict[str, ConfidenceAdjustment],
        context_result: ContextResult,
        decision_thresholds: Dict[str, Dict[str, float]] = None
    ) -> Dict[str, DecisionType]:
        """
        Generate recommended actions for each category based on adjusted confidences.
        
        Args:
            adjusted_confidences: Dictionary of adjusted confidence scores
            context_result: Context analysis result
            decision_thresholds: Optional custom thresholds for decisions
            
        Returns:
            Dictionary mapping categories to recommended actions
        """
        if decision_thresholds is None:
            # Default thresholds
            decision_thresholds = {
                'block': 0.8,   # High confidence threshold for blocking
                'review': 0.5   # Medium confidence threshold for review
            }
        
        recommendations = {}
        
        for category, adjustment in adjusted_confidences.items():
            confidence = adjustment.adjusted_confidence
            
            # Apply category-specific threshold adjustments
            category_thresholds = self._get_category_thresholds(category, decision_thresholds)
            
            if confidence >= category_thresholds['block']:
                recommendations[category] = DecisionType.BLOCK
            elif confidence >= category_thresholds['review']:
                recommendations[category] = DecisionType.REVIEW
            else:
                recommendations[category] = DecisionType.ALLOW
        
        return recommendations
    
    def _get_category_thresholds(
        self, 
        category: str, 
        base_thresholds: Dict[str, float]
    ) -> Dict[str, float]:
        """Get category-specific decision thresholds."""
        
        # Some categories require higher confidence for blocking
        category_threshold_adjustments = {
            AbuseCategory.THREAT_VIOLENCE.value: {'block': -0.1, 'review': -0.1},  # Lower thresholds - more sensitive
            AbuseCategory.SELF_HARM.value: {'block': -0.1, 'review': -0.1},       # Lower thresholds - more sensitive
            AbuseCategory.OBSCENITY_PROFANITY.value: {'block': 0.1, 'review': 0.05}, # Higher thresholds - less sensitive
        }
        
        adjustments = category_threshold_adjustments.get(category, {'block': 0, 'review': 0})
        
        return {
            'block': base_thresholds['block'] + adjustments['block'],
            'review': base_thresholds['review'] + adjustments['review']
        }
    
    def get_adjustment_summary(
        self, 
        adjusted_confidences: Dict[str, ConfidenceAdjustment]
    ) -> Dict[str, any]:
        """
        Generate a summary of confidence adjustments.
        
        Args:
            adjusted_confidences: Dictionary of adjusted confidence scores
            
        Returns:
            Summary dictionary with adjustment statistics
        """
        if not adjusted_confidences:
            return {
                'total_categories': 0,
                'categories_adjusted': 0,
                'average_adjustment_factor': 1.0,
                'max_adjustment': 0.0,
                'min_adjustment': 0.0
            }
        
        adjustment_factors = [adj.adjustment_factor for adj in adjusted_confidences.values()]
        
        return {
            'total_categories': len(adjusted_confidences),
            'categories_adjusted': sum(1 for factor in adjustment_factors if factor != 1.0),
            'average_adjustment_factor': sum(adjustment_factors) / len(adjustment_factors),
            'max_adjustment': max(adjustment_factors),
            'min_adjustment': min(adjustment_factors),
            'categories_with_adjustments': [
                category for category, adj in adjusted_confidences.items() 
                if adj.adjustment_factor != 1.0
            ]
        }