"""Explanation generation for ensemble decisions."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .models import (
    LPEResult, ClassifierResult, ContextResult, AggregatedResult,
    ProblemSpan, DecisionType, SeverityLevel, AbuseCategory
)
from .decision_logic import DecisionContext

logger = logging.getLogger(__name__)


class ExplanationLevel(str, Enum):
    """Levels of explanation detail."""
    BASIC = "basic"
    DETAILED = "detailed"
    TECHNICAL = "technical"


@dataclass
class ComponentContribution:
    """Contribution of a component to the final decision."""
    component_name: str
    weight: float
    confidence: float
    primary_categories: List[str]
    key_evidence: List[str]


@dataclass
class DecisionTrace:
    """Detailed trace of decision making process."""
    final_decision: DecisionType
    confidence: float
    primary_category: str
    component_contributions: List[ComponentContribution]
    context_factors: List[str]
    threshold_analysis: Dict[str, float]


class ExplanationGenerator:
    """Generates human-readable explanations for ensemble decisions."""
    
    def __init__(self):
        self.category_descriptions = {
            AbuseCategory.THREAT_VIOLENCE.value: "threats or violent language",
            AbuseCategory.HATE_TARGETED.value: "hate speech or targeted harassment",
            AbuseCategory.SEXUAL_CONTENT.value: "sexual or inappropriate content",
            AbuseCategory.OBSCENITY_PROFANITY.value: "profanity or obscene language",
            AbuseCategory.INSULT_HARASSMENT.value: "insults or harassment",
            AbuseCategory.BULLYING_TAUNTING.value: "bullying or taunting behavior",
            AbuseCategory.SELF_HARM.value: "self-harm encouragement",
            AbuseCategory.SPAM_SCAM.value: "spam or scam content"
        }
        
        self.decision_descriptions = {
            DecisionType.ALLOW: "appropriate for corporate communication",
            DecisionType.REVIEW: "requires human review before use",
            DecisionType.BLOCK: "inappropriate and should be blocked"
        }
    
    def generate_explanation(
        self,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult,
        aggregated_result: AggregatedResult,
        decision_context: DecisionContext = None,
        explanation_level: ExplanationLevel = ExplanationLevel.BASIC,
        component_weights: Dict[str, float] = None
    ) -> List[str]:
        """
        Generate comprehensive explanation for the ensemble decision.
        
        Args:
            lpe_result: Result from lexicon engine
            classifier_result: Result from ML classifier
            context_result: Result from context analysis
            aggregated_result: Final aggregated result
            decision_context: Context information for decision
            explanation_level: Level of detail for explanation
            component_weights: Weights used in ensemble
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # Generate decision trace
        decision_trace = self._create_decision_trace(
            lpe_result, classifier_result, context_result,
            aggregated_result, component_weights or {}
        )
        
        if explanation_level == ExplanationLevel.BASIC:
            explanations.extend(self._generate_basic_explanation(
                decision_trace, aggregated_result
            ))
        elif explanation_level == ExplanationLevel.DETAILED:
            explanations.extend(self._generate_detailed_explanation(
                decision_trace, aggregated_result, decision_context
            ))
        else:  # TECHNICAL
            explanations.extend(self._generate_technical_explanation(
                decision_trace, aggregated_result, decision_context,
                lpe_result, classifier_result, context_result
            ))
        
        return explanations
    
    def _create_decision_trace(
        self,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult,
        aggregated_result: AggregatedResult,
        component_weights: Dict[str, float]
    ) -> DecisionTrace:
        """Create detailed decision trace."""
        # Analyze component contributions
        component_contributions = []
        
        # LPE contribution
        lpe_categories = [cat for cat, score in lpe_result.confidence_scores.items() if score > 0.3]
        lpe_evidence = lpe_result.rule_traces[:3]  # Top 3 rule traces
        component_contributions.append(ComponentContribution(
            component_name="Lexicon & Pattern Engine",
            weight=component_weights.get('lpe', 0.0),
            confidence=max(lpe_result.confidence_scores.values()) if lpe_result.confidence_scores else 0.0,
            primary_categories=lpe_categories,
            key_evidence=lpe_evidence
        ))
        
        # Classifier contribution
        classifier_categories = [
            cat for cat, score in classifier_result.category_probabilities.items() 
            if score > 0.3
        ]
        classifier_evidence = [f"ML confidence: {max(classifier_result.category_probabilities.values()):.2f}"]
        component_contributions.append(ComponentContribution(
            component_name="ML Classifier",
            weight=component_weights.get('classifier', 0.0),
            confidence=max(classifier_result.category_probabilities.values()) if classifier_result.category_probabilities else 0.0,
            primary_categories=classifier_categories,
            key_evidence=classifier_evidence
        ))
        
        # Intent layer contribution
        intent_evidence = []
        if any(context_result.safe_context_detected.values()):
            intent_evidence.append("Safe context detected")
        if context_result.recommended_action != aggregated_result.final_decision:
            intent_evidence.append(f"Recommended: {context_result.recommended_action.value}")
        
        component_contributions.append(ComponentContribution(
            component_name="Context Analysis",
            weight=component_weights.get('intent', 0.0),
            confidence=0.8 if context_result.recommended_action == DecisionType.BLOCK else 0.5,
            primary_categories=[],
            key_evidence=intent_evidence
        ))
        
        # Context factors
        context_factors = []
        if any(context_result.context_modifiers.values()):
            context_factors.append("Context modifiers applied")
        
        # Primary category
        primary_category = max(
            aggregated_result.category_scores.items(), 
            key=lambda x: x[1]
        )[0] if aggregated_result.category_scores else "unknown"
        
        return DecisionTrace(
            final_decision=aggregated_result.final_decision,
            confidence=aggregated_result.confidence_score,
            primary_category=primary_category,
            component_contributions=component_contributions,
            context_factors=context_factors,
            threshold_analysis={}  # Could be populated with threshold analysis
        )
    
    def _generate_basic_explanation(
        self,
        decision_trace: DecisionTrace,
        aggregated_result: AggregatedResult
    ) -> List[str]:
        """Generate basic explanation suitable for end users."""
        explanations = []
        
        # Main decision explanation
        decision_desc = self.decision_descriptions.get(
            decision_trace.final_decision, "requires review"
        )
        explanations.append(f"Content is {decision_desc}")
        
        # Primary reason
        if decision_trace.primary_category in self.category_descriptions:
            category_desc = self.category_descriptions[decision_trace.primary_category]
            explanations.append(f"Primary concern: {category_desc}")
        
        # Confidence level
        confidence_level = "high" if decision_trace.confidence > 0.8 else "medium" if decision_trace.confidence > 0.6 else "low"
        explanations.append(f"Confidence level: {confidence_level}")
        
        # Severity if blocked or review
        if decision_trace.final_decision in [DecisionType.BLOCK, DecisionType.REVIEW]:
            explanations.append(f"Severity: {aggregated_result.severity_level.value}")
        
        return explanations
    
    def _generate_detailed_explanation(
        self,
        decision_trace: DecisionTrace,
        aggregated_result: AggregatedResult,
        decision_context: DecisionContext = None
    ) -> List[str]:
        """Generate detailed explanation for moderators."""
        explanations = []
        
        # Start with basic explanation
        explanations.extend(self._generate_basic_explanation(decision_trace, aggregated_result))
        
        # Component analysis
        explanations.append("\nComponent Analysis:")
        for contrib in decision_trace.component_contributions:
            if contrib.weight > 0 and contrib.confidence > 0.1:
                explanations.append(
                    f"• {contrib.component_name}: {contrib.confidence:.2f} confidence "
                    f"(weight: {contrib.weight:.2f})"
                )
                
                if contrib.primary_categories:
                    categories_str = ", ".join(contrib.primary_categories)
                    explanations.append(f"  Categories: {categories_str}")
                
                if contrib.key_evidence:
                    evidence_str = "; ".join(contrib.key_evidence[:2])
                    explanations.append(f"  Evidence: {evidence_str}")
        
        # Context information
        if decision_context:
            explanations.append(f"\nContext Information:")
            explanations.append(f"• Text length: {decision_context.text_length} characters")
            explanations.append(f"• Primary language: {decision_context.primary_language}")
            explanations.append(f"• Language confidence: {decision_context.language_confidence:.2f}")
            
            if decision_context.has_code_mixing:
                explanations.append("• Code-mixing detected")
            if decision_context.obfuscation_detected:
                explanations.append("• Obfuscation techniques detected")
        
        # Problematic spans
        if aggregated_result.consolidated_spans:
            explanations.append(f"\nProblematic Content:")
            for i, span in enumerate(aggregated_result.consolidated_spans[:3]):
                explanations.append(
                    f"• \"{span.text}\" ({span.category}, confidence: {span.confidence:.2f})"
                )
        
        return explanations
    
    def _generate_technical_explanation(
        self,
        decision_trace: DecisionTrace,
        aggregated_result: AggregatedResult,
        decision_context: DecisionContext,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult
    ) -> List[str]:
        """Generate technical explanation for developers/analysts."""
        explanations = []
        
        # Start with detailed explanation
        explanations.extend(self._generate_detailed_explanation(
            decision_trace, aggregated_result, decision_context
        ))
        
        # Technical details
        explanations.append(f"\nTechnical Details:")
        explanations.append(f"• Final confidence: {decision_trace.confidence:.4f}")
        
        # Category scores
        if aggregated_result.category_scores:
            explanations.append("• Category scores:")
            sorted_scores = sorted(
                aggregated_result.category_scores.items(),
                key=lambda x: x[1], reverse=True
            )
            for category, score in sorted_scores[:5]:
                explanations.append(f"  - {category}: {score:.4f}")
        
        # Component raw outputs
        explanations.append("\nRaw Component Outputs:")
        
        # LPE details
        if lpe_result.confidence_scores:
            explanations.append("• LPE scores:")
            for category, score in lpe_result.confidence_scores.items():
                if score > 0:
                    explanations.append(f"  - {category}: {score:.4f}")
        
        # Classifier details
        if classifier_result.category_probabilities:
            explanations.append("• Classifier probabilities:")
            for category, prob in classifier_result.category_probabilities.items():
                if prob > 0.1:
                    explanations.append(f"  - {category}: {prob:.4f}")
        
        # Context modifiers
        if context_result.context_modifiers:
            explanations.append("• Context modifiers:")
            for category, modifier in context_result.context_modifiers.items():
                if modifier != 1.0:
                    explanations.append(f"  - {category}: {modifier:.4f}")
        
        # Rule traces
        if lpe_result.rule_traces:
            explanations.append("• Rule traces:")
            for trace in lpe_result.rule_traces[:5]:
                explanations.append(f"  - {trace}")
        
        return explanations
    
    def generate_policy_trace(
        self,
        aggregated_result: AggregatedResult,
        applied_policies: List[str],
        threshold_adjustments: Dict[str, float] = None
    ) -> List[str]:
        """
        Generate policy application trace.
        
        Args:
            aggregated_result: Final result after policy application
            applied_policies: List of policy rules that were applied
            threshold_adjustments: Any threshold adjustments made
            
        Returns:
            List of policy trace strings
        """
        policy_trace = []
        
        if applied_policies:
            policy_trace.append("Applied Policies:")
            for policy in applied_policies:
                policy_trace.append(f"• {policy}")
        
        if threshold_adjustments:
            policy_trace.append("Threshold Adjustments:")
            for category, adjustment in threshold_adjustments.items():
                if adjustment != 0:
                    direction = "increased" if adjustment > 0 else "decreased"
                    policy_trace.append(f"• {category}: {direction} by {abs(adjustment):.2f}")
        
        return policy_trace
    
    def generate_attribution_report(
        self,
        decision_trace: DecisionTrace,
        aggregated_result: AggregatedResult
    ) -> Dict[str, float]:
        """
        Generate attribution report showing contribution of each component.
        
        Args:
            decision_trace: Decision trace with component contributions
            aggregated_result: Final aggregated result
            
        Returns:
            Dictionary mapping component names to attribution scores
        """
        attribution = {}
        
        total_weighted_contribution = 0
        component_contributions = {}
        
        # Calculate weighted contributions
        for contrib in decision_trace.component_contributions:
            weighted_contrib = contrib.weight * contrib.confidence
            component_contributions[contrib.component_name] = weighted_contrib
            total_weighted_contribution += weighted_contrib
        
        # Normalize to get attribution percentages
        if total_weighted_contribution > 0:
            for component_name, contrib in component_contributions.items():
                attribution[component_name] = contrib / total_weighted_contribution
        else:
            # Equal attribution if no contributions
            num_components = len(decision_trace.component_contributions)
            for contrib in decision_trace.component_contributions:
                attribution[contrib.component_name] = 1.0 / num_components
        
        return attribution
    
    def format_explanation_for_api(
        self,
        explanations: List[str],
        include_technical: bool = False
    ) -> Dict[str, any]:
        """
        Format explanations for API response.
        
        Args:
            explanations: List of explanation strings
            include_technical: Whether to include technical details
            
        Returns:
            Formatted explanation dictionary
        """
        # Separate different types of explanations
        basic_explanations = []
        technical_explanations = []
        
        current_section = "basic"
        for explanation in explanations:
            if explanation.startswith("Technical Details:") or explanation.startswith("Raw Component"):
                current_section = "technical"
                continue
            
            if current_section == "basic" or not include_technical:
                basic_explanations.append(explanation)
            else:
                technical_explanations.append(explanation)
        
        result = {
            "summary": basic_explanations[0] if basic_explanations else "No explanation available",
            "details": basic_explanations[1:] if len(basic_explanations) > 1 else []
        }
        
        if include_technical and technical_explanations:
            result["technical"] = technical_explanations
        
        return result