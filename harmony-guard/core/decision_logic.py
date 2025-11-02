"""Advanced decision logic and span ranking for ensemble aggregation."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .models import (
    ProblemSpan, DecisionType, SeverityLevel, AbuseCategory,
    LPEResult, ClassifierResult, ContextResult
)

logger = logging.getLogger(__name__)


class RankingMethod(str, Enum):
    """Span ranking methods."""
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    SEVERITY_WEIGHTED = "severity_weighted"
    HYBRID = "hybrid"
    POSITION_AWARE = "position_aware"


@dataclass
class DecisionContext:
    """Context information for decision making."""
    text_length: int
    language_confidence: float
    primary_language: str
    has_code_mixing: bool
    obfuscation_detected: bool


class AdvancedSpanRanker:
    """Advanced span ranking with multiple algorithms."""
    
    def __init__(self, ranking_method: RankingMethod = RankingMethod.HYBRID):
        self.ranking_method = ranking_method
        
        # Category severity weights
        self.category_severity_weights = {
            AbuseCategory.THREAT_VIOLENCE.value: 1.0,
            AbuseCategory.SELF_HARM.value: 0.95,
            AbuseCategory.HATE_TARGETED.value: 0.9,
            AbuseCategory.SEXUAL_CONTENT.value: 0.8,
            AbuseCategory.OBSCENITY_PROFANITY.value: 0.7,
            AbuseCategory.INSULT_HARASSMENT.value: 0.6,
            AbuseCategory.BULLYING_TAUNTING.value: 0.5,
            AbuseCategory.SPAM_SCAM.value: 0.3
        }
    
    def rank_spans(
        self, 
        spans: List[ProblemSpan],
        decision_context: DecisionContext = None
    ) -> List[ProblemSpan]:
        """
        Rank spans using the configured ranking method.
        
        Args:
            spans: List of problem spans to rank
            decision_context: Additional context for ranking
            
        Returns:
            Ranked list of spans
        """
        if not spans:
            return []
        
        if self.ranking_method == RankingMethod.CONFIDENCE_WEIGHTED:
            return self._rank_by_confidence(spans)
        elif self.ranking_method == RankingMethod.SEVERITY_WEIGHTED:
            return self._rank_by_severity(spans)
        elif self.ranking_method == RankingMethod.POSITION_AWARE:
            return self._rank_position_aware(spans, decision_context)
        else:  # HYBRID
            return self._rank_hybrid(spans, decision_context)
    
    def _rank_by_confidence(self, spans: List[ProblemSpan]) -> List[ProblemSpan]:
        """Rank spans by confidence score only."""
        return sorted(spans, key=lambda x: x.confidence, reverse=True)
    
    def _rank_by_severity(self, spans: List[ProblemSpan]) -> List[ProblemSpan]:
        """Rank spans by category severity."""
        def severity_score(span: ProblemSpan) -> float:
            base_weight = self.category_severity_weights.get(span.category, 0.5)
            return base_weight * span.confidence
        
        return sorted(spans, key=severity_score, reverse=True)
    
    def _rank_position_aware(
        self, 
        spans: List[ProblemSpan],
        decision_context: DecisionContext = None
    ) -> List[ProblemSpan]:
        """Rank spans considering position in text."""
        def position_score(span: ProblemSpan) -> float:
            # Early positions get slight boost (more prominent)
            text_length = decision_context.text_length if decision_context else 1000
            position_factor = 1.0 + (0.1 * (1.0 - span.start / text_length))
            
            return span.confidence * position_factor
        
        return sorted(spans, key=position_score, reverse=True)
    
    def _rank_hybrid(
        self, 
        spans: List[ProblemSpan],
        decision_context: DecisionContext = None
    ) -> List[ProblemSpan]:
        """Hybrid ranking combining multiple factors."""
        def hybrid_score(span: ProblemSpan) -> float:
            # Base confidence score
            score = span.confidence
            
            # Category severity weight
            severity_weight = self.category_severity_weights.get(span.category, 0.5)
            score *= (0.7 + 0.3 * severity_weight)
            
            # Position factor (slight boost for early positions)
            if decision_context:
                position_factor = 1.0 + (0.05 * (1.0 - span.start / decision_context.text_length))
                score *= position_factor
            
            # Length factor (longer spans might be more significant)
            span_length = span.end - span.start
            length_factor = min(1.2, 1.0 + (span_length - 5) * 0.01)
            score *= length_factor
            
            # Rule source factor (ML predictions get slight boost)
            if "classifier" in span.rule_source.lower():
                score *= 1.05
            
            return score
        
        return sorted(spans, key=hybrid_score, reverse=True)


class AdaptiveDecisionMaker:
    """Adaptive decision making with context awareness."""
    
    def __init__(self):
        self.language_confidence_threshold = 0.8
        self.obfuscation_penalty = 0.1
        self.code_mixing_adjustment = 0.05
    
    def make_adaptive_decision(
        self,
        base_scores: Dict[str, float],
        decision_context: DecisionContext,
        base_thresholds: Dict[str, float]
    ) -> Tuple[DecisionType, float, Dict[str, float]]:
        """
        Make decision with adaptive thresholds based on context.
        
        Args:
            base_scores: Base category scores
            decision_context: Context information
            base_thresholds: Base decision thresholds
            
        Returns:
            Tuple of (decision, confidence, adjusted_scores)
        """
        # Adjust scores based on context
        adjusted_scores = self._adjust_scores_for_context(base_scores, decision_context)
        
        # Adjust thresholds based on context
        adjusted_thresholds = self._adjust_thresholds_for_context(
            base_thresholds, decision_context
        )
        
        # Make decision with adjusted values
        max_score = max(adjusted_scores.values()) if adjusted_scores else 0.0
        
        if max_score >= adjusted_thresholds['block']:
            decision = DecisionType.BLOCK
        elif max_score >= adjusted_thresholds['review']:
            decision = DecisionType.REVIEW
        else:
            decision = DecisionType.ALLOW
        
        # Calculate confidence with context awareness
        confidence = self._calculate_adaptive_confidence(
            max_score, decision, adjusted_thresholds, decision_context
        )
        
        return decision, confidence, adjusted_scores
    
    def _adjust_scores_for_context(
        self,
        base_scores: Dict[str, float],
        decision_context: DecisionContext
    ) -> Dict[str, float]:
        """Adjust scores based on context factors."""
        adjusted_scores = base_scores.copy()
        
        # Language confidence adjustment
        if decision_context.language_confidence < self.language_confidence_threshold:
            confidence_penalty = (self.language_confidence_threshold - decision_context.language_confidence) * 0.2
            for category in adjusted_scores:
                adjusted_scores[category] *= (1.0 - confidence_penalty)
        
        # Obfuscation detection adjustment
        if decision_context.obfuscation_detected:
            for category in adjusted_scores:
                adjusted_scores[category] *= (1.0 + self.obfuscation_penalty)
        
        # Code mixing adjustment
        if decision_context.has_code_mixing:
            for category in adjusted_scores:
                adjusted_scores[category] *= (1.0 - self.code_mixing_adjustment)
        
        return adjusted_scores
    
    def _adjust_thresholds_for_context(
        self,
        base_thresholds: Dict[str, float],
        decision_context: DecisionContext
    ) -> Dict[str, float]:
        """Adjust decision thresholds based on context."""
        adjusted_thresholds = base_thresholds.copy()
        
        # Lower thresholds for low language confidence (more conservative)
        if decision_context.language_confidence < self.language_confidence_threshold:
            adjustment = (self.language_confidence_threshold - decision_context.language_confidence) * 0.1
            adjusted_thresholds['block'] -= adjustment
            adjusted_thresholds['review'] -= adjustment
        
        # Adjust for text length (shorter texts might need different thresholds)
        if decision_context.text_length < 50:
            # Slightly higher thresholds for very short texts
            adjusted_thresholds['block'] += 0.05
            adjusted_thresholds['review'] += 0.03
        elif decision_context.text_length > 500:
            # Slightly lower thresholds for long texts
            adjusted_thresholds['block'] -= 0.03
            adjusted_thresholds['review'] -= 0.02
        
        return adjusted_thresholds
    
    def _calculate_adaptive_confidence(
        self,
        max_score: float,
        decision: DecisionType,
        thresholds: Dict[str, float],
        decision_context: DecisionContext
    ) -> float:
        """Calculate confidence with context awareness."""
        base_confidence = 0.5
        
        if decision == DecisionType.BLOCK:
            # Confidence increases with distance above block threshold
            distance_above = max_score - thresholds['block']
            base_confidence = min(1.0, 0.7 + distance_above * 2)
        elif decision == DecisionType.REVIEW:
            # Confidence based on position between thresholds
            review_range = thresholds['block'] - thresholds['review']
            position_in_range = (max_score - thresholds['review']) / review_range
            base_confidence = 0.5 + position_in_range * 0.3
        else:  # ALLOW
            # Confidence increases with distance below review threshold
            distance_below = thresholds['review'] - max_score
            base_confidence = min(1.0, 0.6 + distance_below * 1.5)
        
        # Adjust confidence based on context factors
        if decision_context.language_confidence < self.language_confidence_threshold:
            base_confidence *= 0.9
        
        if decision_context.obfuscation_detected:
            base_confidence *= 0.95
        
        if decision_context.has_code_mixing:
            base_confidence *= 0.92
        
        return max(0.1, min(1.0, base_confidence))


class EnsembleWeightManager:
    """Manages dynamic ensemble weights based on component performance."""
    
    def __init__(self):
        self.component_performance_history = {
            'lpe': [],
            'classifier': [],
            'intent': []
        }
        self.performance_window = 1000  # Number of recent predictions to consider
        self.min_weight = 0.05  # Minimum weight for any component
    
    def update_performance(
        self,
        component_name: str,
        prediction_correct: bool,
        confidence: float
    ) -> None:
        """
        Update performance history for a component.
        
        Args:
            component_name: Name of the component
            prediction_correct: Whether the prediction was correct
            confidence: Confidence of the prediction
        """
        if component_name not in self.component_performance_history:
            return
        
        # Store performance record
        performance_record = {
            'correct': prediction_correct,
            'confidence': confidence,
            'weighted_score': confidence if prediction_correct else (1.0 - confidence)
        }
        
        self.component_performance_history[component_name].append(performance_record)
        
        # Keep only recent history
        if len(self.component_performance_history[component_name]) > self.performance_window:
            self.component_performance_history[component_name].pop(0)
    
    def get_adaptive_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Get adaptive weights based on recent performance.
        
        Args:
            base_weights: Base ensemble weights
            
        Returns:
            Adjusted weights based on performance
        """
        if not any(self.component_performance_history.values()):
            return base_weights
        
        # Calculate performance scores
        performance_scores = {}
        for component_name, history in self.component_performance_history.items():
            if not history:
                performance_scores[component_name] = 0.5  # Neutral score
                continue
            
            # Calculate weighted average of recent performance
            recent_history = history[-100:]  # Last 100 predictions
            total_score = sum(record['weighted_score'] for record in recent_history)
            performance_scores[component_name] = total_score / len(recent_history)
        
        # Adjust weights based on performance
        adjusted_weights = {}
        total_performance = sum(performance_scores.values())
        
        if total_performance == 0:
            return base_weights
        
        for component_name, base_weight in base_weights.items():
            performance_ratio = performance_scores.get(component_name, 0.5) / (total_performance / len(performance_scores))
            
            # Adjust weight with performance ratio (but don't change too drastically)
            adjustment_factor = 0.8 + 0.4 * performance_ratio  # Range: 0.8 to 1.2
            adjusted_weight = base_weight * adjustment_factor
            
            # Ensure minimum weight
            adjusted_weights[component_name] = max(self.min_weight, adjusted_weight)
        
        # Normalize weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for component_name in adjusted_weights:
                adjusted_weights[component_name] /= total_weight
        
        return adjusted_weights
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all components."""
        summary = {}
        
        for component_name, history in self.component_performance_history.items():
            if not history:
                summary[component_name] = {
                    'accuracy': 0.0,
                    'avg_confidence': 0.0,
                    'sample_count': 0
                }
                continue
            
            correct_predictions = sum(1 for record in history if record['correct'])
            total_predictions = len(history)
            avg_confidence = sum(record['confidence'] for record in history) / total_predictions
            
            summary[component_name] = {
                'accuracy': correct_predictions / total_predictions,
                'avg_confidence': avg_confidence,
                'sample_count': total_predictions
            }
        
        return summary