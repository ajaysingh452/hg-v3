"""Main context analyzer that combines negation, quotation, and safe context detection."""

from typing import Dict, List
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.interfaces import IntentContextLayerInterface
from core.models import ProcessedText, LPEResult, ClassifierResult, ContextResult, DecisionType
from intent.negation_detector import NegationDetector
from intent.quotation_detector import QuotationDetector
from intent.safe_context_detector import SafeContextDetector
from intent.confidence_adjuster import ContextAwareConfidenceAdjuster


class ContextAnalyzer(IntentContextLayerInterface):
    """
    Main context analyzer that implements the Intent/Context Layer interface.
    Combines negation detection, quotation analysis, and safe context recognition.
    """
    
    def __init__(self):
        self.negation_detector = NegationDetector()
        self.quotation_detector = QuotationDetector()
        self.safe_context_detector = SafeContextDetector()
        self.confidence_adjuster = ContextAwareConfidenceAdjuster()
        
        # Configuration for context modifiers
        self.negation_modifier_strength = 0.7  # How much negation reduces confidence
        self.quotation_modifier_strength = 0.5  # How much quotation reduces confidence
        self.safe_context_modifier_strength = 0.8  # How much safe context reduces confidence
    
    def analyze_context(
        self, 
        processed_text: ProcessedText,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult
    ) -> ContextResult:
        """
        Analyze contextual factors to adjust confidence and decisions.
        
        Args:
            processed_text: Preprocessed text input
            lpe_result: Result from lexicon engine
            classifier_result: Result from ML classifier
            
        Returns:
            ContextResult with context modifiers and recommendations
        """
        text = processed_text.original_text
        all_spans = lpe_result.matched_spans + classifier_result.attention_spans
        
        # Analyze different contextual factors
        negation_analysis = self.negation_detector.get_negation_context(text, all_spans)
        quotation_analysis = self.quotation_detector.get_quotation_context(text, all_spans)
        safe_context_analysis = self.safe_context_detector.analyze_safe_context(text)
        
        # Calculate context modifiers for each abuse category
        context_modifiers = self._calculate_context_modifiers(
            negation_analysis, quotation_analysis, safe_context_analysis,
            lpe_result, classifier_result
        )
        
        # Determine safe context flags
        safe_context_detected = self._determine_safe_contexts(
            negation_analysis, quotation_analysis, safe_context_analysis
        )
        
        # Generate recommended action based on context analysis
        recommended_action = self._generate_recommended_action(
            context_modifiers, safe_context_detected, lpe_result, classifier_result
        )
        
        return ContextResult(
            context_modifiers=context_modifiers,
            safe_context_detected=safe_context_detected,
            recommended_action=recommended_action
        )
    
    def _calculate_context_modifiers(
        self,
        negation_analysis: Dict,
        quotation_analysis: Dict,
        safe_context_analysis: Dict,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult
    ) -> Dict[str, float]:
        """Calculate context modifiers for each abuse category."""
        
        # Get all possible abuse categories
        all_categories = set(lpe_result.categories)
        all_categories.update(classifier_result.category_probabilities.keys())
        
        context_modifiers = {}
        
        for category in all_categories:
            modifier = 1.0  # Start with no modification
            
            # Apply negation modifier
            if negation_analysis['has_negation']:
                # Check if any spans in this category are negated
                category_negated = any(
                    span['span'].category == category 
                    for span in negation_analysis['negated_spans']
                )
                
                if category_negated:
                    negation_strength = negation_analysis['overall_negation_confidence']
                    modifier *= (1.0 - (self.negation_modifier_strength * negation_strength))
            
            # Apply quotation modifier
            if quotation_analysis['has_quotations']:
                # Check if any spans in this category are quoted
                category_quoted = any(
                    span['span'].category == category 
                    for span in quotation_analysis['quoted_spans']
                )
                
                if category_quoted:
                    quote_strength = quotation_analysis['overall_quote_confidence']
                    # Different quote types have different modifier strengths
                    quote_type_modifiers = {
                        'direct': 0.6,    # Direct quotes are more likely to be reporting
                        'indirect': 0.4,  # Indirect quotes are less clear
                        'reported': 0.5   # Reported speech is moderately likely to be reporting
                    }
                    
                    max_quote_modifier = 0.0
                    for quote_type in quotation_analysis['quote_types']:
                        max_quote_modifier = max(max_quote_modifier, quote_type_modifiers.get(quote_type, 0.3))
                    
                    modifier *= (1.0 - (max_quote_modifier * quote_strength))
            
            # Apply safe context modifier
            if safe_context_analysis['is_safe_context']:
                safe_modifiers = safe_context_analysis['context_modifiers']
                if category in safe_modifiers:
                    safe_strength = safe_modifiers[category]
                    modifier *= (1.0 - (self.safe_context_modifier_strength * safe_strength))
            
            # Ensure modifier doesn't go below a minimum threshold
            modifier = max(0.1, modifier)
            context_modifiers[category] = modifier
        
        return context_modifiers
    
    def _determine_safe_contexts(
        self,
        negation_analysis: Dict,
        quotation_analysis: Dict,
        safe_context_analysis: Dict
    ) -> Dict[str, bool]:
        """Determine which safe context flags should be set."""
        
        safe_contexts = {
            'has_negation': negation_analysis['has_negation'],
            'has_quotation': quotation_analysis['has_quotations'],
            'is_safe_context': safe_context_analysis['is_safe_context'],
            'hr_reporting': False,
            'educational_content': False,
            'legal_documentation': False,
            'research_academic': False,
            'journalism': False
        }
        
        # Set specific safe context flags based on detected contexts
        if safe_context_analysis['is_safe_context']:
            for context_type in safe_context_analysis['context_types']:
                if context_type in safe_contexts:
                    safe_contexts[context_type] = True
        
        return safe_contexts
    
    def _generate_recommended_action(
        self,
        context_modifiers: Dict[str, float],
        safe_context_detected: Dict[str, bool],
        lpe_result: LPEResult,
        classifier_result: ClassifierResult
    ) -> DecisionType:
        """Generate recommended action based on context analysis."""
        
        # Calculate overall context confidence reduction
        if context_modifiers:
            avg_modifier = sum(context_modifiers.values()) / len(context_modifiers)
        else:
            avg_modifier = 1.0
        
        # Strong context indicators suggest allowing content
        if avg_modifier < 0.3:  # Strong context modification
            return DecisionType.ALLOW
        
        # Moderate context indicators suggest review
        elif avg_modifier < 0.7:  # Moderate context modification
            return DecisionType.REVIEW
        
        # Weak or no context indicators - defer to original analysis
        else:
            # Look at original confidence scores to make recommendation
            lpe_confidence = max(lpe_result.confidence_scores.values(), default=0.0)
            classifier_confidence = max(classifier_result.category_probabilities.values(), default=0.0)
            
            max_confidence = max(lpe_confidence, classifier_confidence)
            
            if max_confidence > 0.8:
                return DecisionType.BLOCK
            elif max_confidence > 0.5:
                return DecisionType.REVIEW
            else:
                return DecisionType.ALLOW
    
    def get_context_explanation(
        self,
        processed_text: ProcessedText,
        context_result: ContextResult
    ) -> List[str]:
        """
        Generate human-readable explanations for context analysis.
        
        Args:
            processed_text: Original processed text
            context_result: Context analysis result
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # Negation explanations
        if context_result.safe_context_detected.get('has_negation', False):
            explanations.append("Negation detected - content may be discussing what NOT to do")
        
        # Quotation explanations
        if context_result.safe_context_detected.get('has_quotation', False):
            explanations.append("Quoted content detected - may be reporting or referencing others")
        
        # Safe context explanations
        safe_context_types = {
            'hr_reporting': "HR reporting context detected",
            'educational_content': "Educational/training content detected", 
            'legal_documentation': "Legal documentation context detected",
            'research_academic': "Research/academic context detected",
            'journalism': "Journalism/news reporting context detected"
        }
        
        for context_type, explanation in safe_context_types.items():
            if context_result.safe_context_detected.get(context_type, False):
                explanations.append(explanation)
        
        # Overall recommendation explanation
        if context_result.recommended_action == DecisionType.ALLOW:
            explanations.append("Context analysis suggests content is likely appropriate")
        elif context_result.recommended_action == DecisionType.REVIEW:
            explanations.append("Context analysis suggests manual review recommended")
        
        return explanations
    
    def adjust_confidences(
        self,
        processed_text: ProcessedText,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult
    ) -> Dict[str, any]:
        """
        Apply context-aware confidence adjustments.
        
        Args:
            processed_text: Preprocessed text input
            lpe_result: Result from lexicon engine
            classifier_result: Result from ML classifier
            context_result: Context analysis result
            
        Returns:
            Dictionary with adjusted confidences and recommendations
        """
        # Get confidence adjustments
        adjusted_confidences = self.confidence_adjuster.adjust_confidence_scores(
            processed_text, lpe_result, classifier_result, context_result
        )
        
        # Generate recommended actions
        recommended_actions = self.confidence_adjuster.generate_recommended_actions(
            adjusted_confidences, context_result
        )
        
        # Get adjustment summary
        adjustment_summary = self.confidence_adjuster.get_adjustment_summary(adjusted_confidences)
        
        return {
            'adjusted_confidences': adjusted_confidences,
            'recommended_actions': recommended_actions,
            'adjustment_summary': adjustment_summary
        }
    
    def get_final_decision_with_context(
        self,
        processed_text: ProcessedText,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult
    ) -> Dict[str, any]:
        """
        Get final decision incorporating all contextual factors.
        
        Args:
            processed_text: Preprocessed text input
            lpe_result: Result from lexicon engine
            classifier_result: Result from ML classifier
            context_result: Context analysis result
            
        Returns:
            Dictionary with final decision and detailed reasoning
        """
        # Get confidence adjustments
        confidence_data = self.adjust_confidences(
            processed_text, lpe_result, classifier_result, context_result
        )
        
        # Determine overall decision based on adjusted confidences
        adjusted_confidences = confidence_data['adjusted_confidences']
        recommended_actions = confidence_data['recommended_actions']
        
        # Find the highest adjusted confidence and its category
        if adjusted_confidences:
            max_category = max(adjusted_confidences.keys(), 
                             key=lambda k: adjusted_confidences[k].adjusted_confidence)
            max_confidence = adjusted_confidences[max_category].adjusted_confidence
            final_decision = recommended_actions.get(max_category, DecisionType.ALLOW)
        else:
            max_category = None
            max_confidence = 0.0
            final_decision = DecisionType.ALLOW
        
        # Generate comprehensive explanation
        explanations = self.get_context_explanation(processed_text, context_result)
        
        # Add confidence adjustment explanations
        if adjusted_confidences:
            for category, adjustment in adjusted_confidences.items():
                if adjustment.reasoning:
                    explanations.append(f"{category}: {'; '.join(adjustment.reasoning)}")
        
        return {
            'final_decision': final_decision,
            'max_confidence': max_confidence,
            'max_category': max_category,
            'adjusted_confidences': adjusted_confidences,
            'recommended_actions': recommended_actions,
            'explanations': explanations,
            'adjustment_summary': confidence_data['adjustment_summary']
        }