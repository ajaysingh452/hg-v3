"""Ensemble aggregator for combining component outputs."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import math

from .interfaces import EnsembleAggregatorInterface
from .models import (
    LPEResult, ClassifierResult, ContextResult, AggregatedResult,
    ProblemSpan, DecisionType, SeverityLevel, AbuseCategory
)
from .decision_logic import (
    AdvancedSpanRanker, AdaptiveDecisionMaker, EnsembleWeightManager,
    DecisionContext, RankingMethod
)
from .explanation import ExplanationGenerator, ExplanationLevel
from ..configs.manager import ConfigurationManager

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Data structure for probability calibration."""
    component_outputs: np.ndarray
    true_labels: np.ndarray
    temperatures: Dict[str, float]


class TemperatureScaler:
    """Temperature scaling for probability calibration."""
    
    def __init__(self):
        self.temperatures = {}
        self.is_fitted = False
    
    def fit(self, logits: Dict[str, np.ndarray], labels: np.ndarray) -> None:
        """
        Fit temperature scaling parameters.
        
        Args:
            logits: Dictionary of component logits {component_name: logits_array}
            labels: True binary labels
        """
        self.temperatures = {}
        
        for component_name, component_logits in logits.items():
            # Find optimal temperature using cross-validation
            best_temp = self._find_optimal_temperature(component_logits, labels)
            self.temperatures[component_name] = best_temp
            logger.info(f"Optimal temperature for {component_name}: {best_temp:.3f}")
        
        self.is_fitted = True
    
    def transform(self, logits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Dictionary of component logits
            
        Returns:
            Dictionary of calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("TemperatureScaler not fitted, using default temperatures")
            return {name: self._softmax(logit) for name, logit in logits.items()}
        
        calibrated_probs = {}
        for component_name, component_logits in logits.items():
            temp = self.temperatures.get(component_name, 1.0)
            calibrated_logits = component_logits / temp
            calibrated_probs[component_name] = self._softmax(calibrated_logits)
        
        return calibrated_probs
    
    def _find_optimal_temperature(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Find optimal temperature using grid search."""
        temperatures = np.linspace(0.1, 3.0, 30)
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in temperatures:
            calibrated_probs = self._softmax(logits / temp)
            # Handle binary classification case
            if calibrated_probs.shape[1] == 2:
                loss = self._log_loss(labels, calibrated_probs[:, 1])
            else:
                loss = self._log_loss(labels, calibrated_probs)
            
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        
        return best_temp
    
    def _log_loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Simple log loss implementation."""
        epsilon = 1e-15
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
            exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        else:
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class EnsembleWeightOptimizer:
    """Optimize ensemble weights using cross-validation."""
    
    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.optimal_weights = {}
        self.is_fitted = False
    
    def optimize_weights(
        self, 
        component_predictions: Dict[str, np.ndarray],
        true_labels: np.ndarray,
        initial_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights using cross-validation.
        
        Args:
            component_predictions: Dictionary of component predictions
            true_labels: True labels for optimization
            initial_weights: Initial weight values
            
        Returns:
            Optimized weights dictionary
        """
        if initial_weights is None:
            initial_weights = {name: 1.0/len(component_predictions) 
                             for name in component_predictions.keys()}
        
        # Prepare feature matrix
        component_names = list(component_predictions.keys())
        X = np.column_stack([component_predictions[name] for name in component_names])
        
        # Simple weight optimization using correlation with true labels
        correlations = {}
        for i, component_name in enumerate(component_names):
            correlation = np.corrcoef(X[:, i], true_labels)[0, 1]
            correlations[component_name] = max(0, correlation)  # Only positive correlations
        
        # Normalize correlations to get weights
        total_correlation = sum(correlations.values())
        if total_correlation > 0:
            normalized_weights = np.array([correlations[name] / total_correlation for name in component_names])
        else:
            # Equal weights if no positive correlations
            normalized_weights = np.ones(len(component_names)) / len(component_names)
        
        self.optimal_weights = {
            component_names[i]: float(normalized_weights[i]) 
            for i in range(len(component_names))
        }
        
        self.is_fitted = True
        logger.info(f"Optimized ensemble weights: {self.optimal_weights}")
        
        return self.optimal_weights
    
    def get_weights(self) -> Dict[str, float]:
        """Get current optimal weights."""
        if not self.is_fitted:
            logger.warning("Weights not optimized, returning equal weights")
            return {"lpe": 0.33, "classifier": 0.33, "intent": 0.34}
        return self.optimal_weights


class SpanConsolidator:
    """Consolidate and rank problem spans from multiple components."""
    
    def __init__(self, overlap_threshold: float = 0.3, max_spans: int = 10):
        self.overlap_threshold = overlap_threshold
        self.max_spans = max_spans
    
    def consolidate_spans(self, span_lists: List[List[ProblemSpan]]) -> List[ProblemSpan]:
        """
        Consolidate overlapping spans from multiple components.
        
        Args:
            span_lists: List of span lists from different components
            
        Returns:
            Consolidated and ranked list of spans
        """
        # Flatten all spans
        all_spans = []
        for spans in span_lists:
            all_spans.extend(spans)
        
        if not all_spans:
            return []
        
        # Sort by start position
        all_spans.sort(key=lambda x: x.start)
        
        # Consolidate overlapping spans
        consolidated = []
        current_span = all_spans[0]
        
        for span in all_spans[1:]:
            overlap_ratio = self._calculate_overlap(current_span, span)
            
            if overlap_ratio >= self.overlap_threshold:
                # Merge spans
                current_span = self._merge_spans(current_span, span)
            else:
                consolidated.append(current_span)
                current_span = span
        
        consolidated.append(current_span)
        
        # Rank by confidence and return top spans
        consolidated.sort(key=lambda x: x.confidence, reverse=True)
        return consolidated[:self.max_spans]
    
    def _calculate_overlap(self, span1: ProblemSpan, span2: ProblemSpan) -> float:
        """Calculate overlap ratio between two spans."""
        start_overlap = max(span1.start, span2.start)
        end_overlap = min(span1.end, span2.end)
        
        if start_overlap >= end_overlap:
            return 0.0
        
        overlap_length = end_overlap - start_overlap
        min_length = min(span1.end - span1.start, span2.end - span2.start)
        
        return overlap_length / min_length if min_length > 0 else 0.0
    
    def _merge_spans(self, span1: ProblemSpan, span2: ProblemSpan) -> ProblemSpan:
        """Merge two overlapping spans."""
        # Use the span with higher confidence as base
        base_span = span1 if span1.confidence >= span2.confidence else span2
        other_span = span2 if span1.confidence >= span2.confidence else span1
        
        # Extend boundaries
        merged_start = min(span1.start, span2.start)
        merged_end = max(span1.end, span2.end)
        
        # Combine text (use the longer text)
        merged_text = base_span.text if len(base_span.text) >= len(other_span.text) else other_span.text
        
        # Average confidence
        merged_confidence = (span1.confidence + span2.confidence) / 2
        
        # Combine rule sources
        rule_sources = [base_span.rule_source, other_span.rule_source]
        merged_rule_source = " + ".join(set(rule_sources))
        
        return ProblemSpan(
            text=merged_text,
            start=merged_start,
            end=merged_end,
            category=base_span.category,
            confidence=merged_confidence,
            rule_source=merged_rule_source
        )


class EnsembleAggregator(EnsembleAggregatorInterface):
    """Main ensemble aggregator combining all component outputs."""
    
    def __init__(self, config_manager: ConfigurationManager = None):
        """
        Initialize ensemble aggregator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.temperature_scaler = TemperatureScaler()
        self.weight_optimizer = EnsembleWeightOptimizer()
        self.span_consolidator = SpanConsolidator()
        self.span_ranker = AdvancedSpanRanker()
        self.adaptive_decision_maker = AdaptiveDecisionMaker()
        self.weight_manager = EnsembleWeightManager()
        self.explanation_generator = ExplanationGenerator()
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load ensemble configuration."""
        try:
            config = self.config_manager.get_ensemble_config()
            ensemble_config = config.get('ensemble', {})
            
            # Load weights
            self.weights = ensemble_config.get('weights', {
                'lpe': 0.4, 'classifier': 0.5, 'intent': 0.1
            })
            
            # Load thresholds
            thresholds = ensemble_config.get('thresholds', {})
            self.confidence_minimum = thresholds.get('confidence_minimum', 0.6)
            self.review_threshold = thresholds.get('review_threshold', 0.7)
            self.block_threshold = thresholds.get('block_threshold', 0.85)
            
            # Load span consolidation settings
            span_config = ensemble_config.get('span_consolidation', {})
            self.span_consolidator = SpanConsolidator(
                overlap_threshold=span_config.get('overlap_threshold', 0.3),
                max_spans=span_config.get('max_spans', 10)
            )
            
            # Initialize span ranker with configured method
            ranking_method = span_config.get('ranking_method', 'hybrid')
            self.span_ranker = AdvancedSpanRanker(RankingMethod(ranking_method))
            
            logger.info("Ensemble configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ensemble configuration: {e}")
            # Use default values
            self.weights = {'lpe': 0.4, 'classifier': 0.5, 'intent': 0.1}
            self.confidence_minimum = 0.6
            self.review_threshold = 0.7
            self.block_threshold = 0.85
    
    def aggregate(
        self,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult,
        original_text: str = "",
        language_confidence: float = 1.0,
        primary_language: str = "en"
    ) -> AggregatedResult:
        """
        Combine and calibrate outputs from all ensemble components.
        
        Args:
            lpe_result: Result from lexicon engine
            classifier_result: Result from ML classifier
            context_result: Result from context analysis
            original_text: Original input text for context
            language_confidence: Confidence in language detection
            primary_language: Primary detected language
            
        Returns:
            AggregatedResult with final decision and explanations
        """
        try:
            # Create decision context
            decision_context = DecisionContext(
                text_length=len(original_text),
                language_confidence=language_confidence,
                primary_language=primary_language,
                has_code_mixing=self._detect_code_mixing(original_text),
                obfuscation_detected=self._detect_obfuscation(lpe_result)
            )
            
            # Extract component predictions
            component_scores = self._extract_component_scores(
                lpe_result, classifier_result, context_result
            )
            
            # Apply context modifiers
            adjusted_scores = self._apply_context_modifiers(
                component_scores, context_result
            )
            
            # Get adaptive weights based on performance
            adaptive_weights = self.weight_manager.get_adaptive_weights(self.weights)
            
            # Combine scores using adaptive ensemble weights
            final_scores = self._combine_scores(adjusted_scores, adaptive_weights)
            
            # Make adaptive decision
            base_thresholds = {
                'review': self.review_threshold,
                'block': self.block_threshold
            }
            
            final_decision, confidence_score, final_scores = self.adaptive_decision_maker.make_adaptive_decision(
                final_scores, decision_context, base_thresholds
            )
            
            # Determine severity level
            severity_level = self._determine_severity(final_scores)
            
            # Consolidate and rank spans
            consolidated_spans = self._consolidate_and_rank_spans(
                lpe_result, classifier_result, context_result, decision_context
            )
            
            # Generate explanations using the explanation generator
            explanation_traces = self.explanation_generator.generate_explanation(
                lpe_result, classifier_result, context_result,
                AggregatedResult(
                    final_decision=final_decision,
                    confidence_score=confidence_score,
                    category_scores=final_scores,
                    severity_level=severity_level,
                    explanation_traces=[],
                    consolidated_spans=consolidated_spans
                ),
                decision_context=decision_context,
                explanation_level=ExplanationLevel.DETAILED,
                component_weights=adaptive_weights
            )
            
            return AggregatedResult(
                final_decision=final_decision,
                confidence_score=confidence_score,
                category_scores=final_scores,
                severity_level=severity_level,
                explanation_traces=explanation_traces,
                consolidated_spans=consolidated_spans
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble aggregation: {e}")
            # Return safe fallback result
            return self._create_fallback_result()
    
    def _extract_component_scores(
        self,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult
    ) -> Dict[str, Dict[str, float]]:
        """Extract normalized scores from each component."""
        component_scores = {}
        
        # LPE scores (normalize to 0-1 range)
        lpe_scores = {}
        for category in AbuseCategory:
            category_key = category.value
            lpe_scores[category_key] = lpe_result.confidence_scores.get(category_key, 0.0)
        component_scores['lpe'] = lpe_scores
        
        # Classifier scores (already probabilities)
        classifier_scores = {}
        for category in AbuseCategory:
            category_key = category.value
            classifier_scores[category_key] = classifier_result.category_probabilities.get(category_key, 0.0)
        component_scores['classifier'] = classifier_scores
        
        # Intent layer doesn't provide category scores directly, 
        # but we can infer from recommended action
        intent_scores = {}
        base_score = 0.0
        if context_result.recommended_action == DecisionType.BLOCK:
            base_score = 0.8
        elif context_result.recommended_action == DecisionType.REVIEW:
            base_score = 0.5
        
        for category in AbuseCategory:
            category_key = category.value
            intent_scores[category_key] = base_score
        component_scores['intent'] = intent_scores
        
        return component_scores
    
    def _apply_context_modifiers(
        self,
        component_scores: Dict[str, Dict[str, float]],
        context_result: ContextResult
    ) -> Dict[str, Dict[str, float]]:
        """Apply context modifiers to component scores."""
        adjusted_scores = {}
        
        for component_name, scores in component_scores.items():
            adjusted_component_scores = {}
            
            for category, score in scores.items():
                # Apply context modifier if available
                modifier = context_result.context_modifiers.get(category, 1.0)
                adjusted_score = score * modifier
                
                # Apply safe context detection
                if context_result.safe_context_detected.get(category, False):
                    adjusted_score *= 0.3  # Significantly reduce score for safe contexts
                
                adjusted_component_scores[category] = max(0.0, min(1.0, adjusted_score))
            
            adjusted_scores[component_name] = adjusted_component_scores
        
        return adjusted_scores
    
    def _detect_code_mixing(self, text: str) -> bool:
        """Detect if text contains code-mixing."""
        # Simple heuristic: check for mixed scripts or language patterns
        # This is a simplified implementation
        has_latin = any(ord(c) < 128 for c in text if c.isalpha())
        has_non_latin = any(ord(c) > 128 for c in text if c.isalpha())
        return has_latin and has_non_latin
    
    def _detect_obfuscation(self, lpe_result: LPEResult) -> bool:
        """Detect if obfuscation techniques were found."""
        # Check if any rule traces mention obfuscation
        obfuscation_keywords = ['leet', 'obfuscation', 'elongation', 'homoglyph']
        return any(keyword in trace.lower() for trace in lpe_result.rule_traces 
                  for keyword in obfuscation_keywords)

    def _combine_scores(self, component_scores: Dict[str, Dict[str, float]], weights: Dict[str, float] = None) -> Dict[str, float]:
        """Combine component scores using ensemble weights."""
        if weights is None:
            weights = self.weights
            
        final_scores = {}
        
        # Get all categories
        all_categories = set()
        for scores in component_scores.values():
            all_categories.update(scores.keys())
        
        # Combine scores for each category
        for category in all_categories:
            weighted_score = 0.0
            total_weight = 0.0
            
            for component_name, scores in component_scores.items():
                if category in scores:
                    weight = weights.get(component_name, 0.0)
                    weighted_score += scores[category] * weight
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                final_scores[category] = weighted_score / total_weight
            else:
                final_scores[category] = 0.0
        
        return final_scores
    
    def _make_decision(self, final_scores: Dict[str, float]) -> DecisionType:
        """Make final decision based on combined scores."""
        # Get maximum score across all categories
        max_score = max(final_scores.values()) if final_scores else 0.0
        
        if max_score >= self.block_threshold:
            return DecisionType.BLOCK
        elif max_score >= self.review_threshold:
            return DecisionType.REVIEW
        else:
            return DecisionType.ALLOW
    
    def _calculate_confidence(self, final_scores: Dict[str, float], decision: DecisionType) -> float:
        """Calculate overall confidence in the decision."""
        max_score = max(final_scores.values()) if final_scores else 0.0
        
        if decision == DecisionType.BLOCK:
            # Confidence increases with how far above block threshold
            confidence = min(1.0, max_score / self.block_threshold)
        elif decision == DecisionType.REVIEW:
            # Confidence based on distance from thresholds
            distance_from_review = abs(max_score - self.review_threshold)
            distance_from_block = abs(max_score - self.block_threshold)
            min_distance = min(distance_from_review, distance_from_block)
            confidence = max(0.5, 1.0 - min_distance * 2)
        else:  # ALLOW
            # Confidence increases with how far below review threshold
            confidence = min(1.0, (self.review_threshold - max_score) / self.review_threshold)
        
        return max(self.confidence_minimum, confidence)
    
    def _determine_severity(self, final_scores: Dict[str, float]) -> SeverityLevel:
        """Determine severity level based on scores and categories."""
        max_score = max(final_scores.values()) if final_scores else 0.0
        
        # Get the category with highest score
        max_category = max(final_scores.items(), key=lambda x: x[1])[0] if final_scores else None
        
        # Critical categories that warrant higher severity
        critical_categories = {
            AbuseCategory.THREAT_VIOLENCE.value,
            AbuseCategory.SELF_HARM.value,
            AbuseCategory.HATE_TARGETED.value
        }
        
        if max_score >= 0.9 or (max_category in critical_categories and max_score >= 0.7):
            return SeverityLevel.CRITICAL
        elif max_score >= 0.8:
            return SeverityLevel.HIGH
        elif max_score >= 0.6:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _consolidate_and_rank_spans(
        self,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult,
        decision_context: DecisionContext
    ) -> List[ProblemSpan]:
        """Consolidate and rank spans from all components."""
        span_lists = [
            lpe_result.matched_spans,
            classifier_result.attention_spans
        ]
        
        # First consolidate overlapping spans
        consolidated_spans = self.span_consolidator.consolidate_spans(span_lists)
        
        # Then rank the consolidated spans
        ranked_spans = self.span_ranker.rank_spans(consolidated_spans, decision_context)
        
        return ranked_spans
    
    def _generate_explanations(
        self,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult,
        final_scores: Dict[str, float],
        final_decision: DecisionType,
        decision_context: DecisionContext = None
    ) -> List[str]:
        """Generate human-readable explanations for the decision."""
        explanations = []
        
        # Add decision explanation
        max_score = max(final_scores.values()) if final_scores else 0.0
        max_category = max(final_scores.items(), key=lambda x: x[1])[0] if final_scores else "unknown"
        
        explanations.append(
            f"Decision: {final_decision.value} (confidence: {max_score:.2f}, "
            f"primary category: {max_category})"
        )
        
        # Add component contributions
        if lpe_result.rule_traces:
            explanations.append(f"Lexicon matches: {', '.join(lpe_result.rule_traces[:3])}")
        
        if classifier_result.category_probabilities:
            top_classifier_categories = sorted(
                classifier_result.category_probabilities.items(),
                key=lambda x: x[1], reverse=True
            )[:2]
            explanations.append(
                f"ML predictions: {', '.join([f'{cat}: {score:.2f}' for cat, score in top_classifier_categories])}"
            )
        
        # Add context information
        if any(context_result.safe_context_detected.values()):
            safe_contexts = [k for k, v in context_result.safe_context_detected.items() if v]
            explanations.append(f"Safe contexts detected: {', '.join(safe_contexts)}")
        
        if context_result.recommended_action != final_decision:
            explanations.append(
                f"Context layer recommended: {context_result.recommended_action.value}, "
                f"but ensemble decided: {final_decision.value}"
            )
        
        return explanations
    
    def _create_fallback_result(self) -> AggregatedResult:
        """Create a safe fallback result in case of errors."""
        return AggregatedResult(
            final_decision=DecisionType.REVIEW,
            confidence_score=0.5,
            category_scores={},
            severity_level=SeverityLevel.MEDIUM,
            explanation_traces=["Error in analysis - defaulting to manual review"],
            consolidated_spans=[]
        )
    
    def train_calibration(
        self,
        training_data: List[Tuple[LPEResult, ClassifierResult, ContextResult, str]]
    ) -> None:
        """
        Train probability calibration using historical data.
        
        Args:
            training_data: List of (lpe_result, classifier_result, context_result, true_label) tuples
        """
        if not training_data:
            logger.warning("No training data provided for calibration")
            return
        
        # Extract component logits and true labels
        component_logits = {'lpe': [], 'classifier': [], 'intent': []}
        true_labels = []
        
        for lpe_result, classifier_result, context_result, true_label in training_data:
            # Convert results to logits (simplified approach)
            lpe_logit = max(lpe_result.confidence_scores.values()) if lpe_result.confidence_scores else 0.0
            classifier_logit = max(classifier_result.category_probabilities.values()) if classifier_result.category_probabilities else 0.0
            intent_logit = 0.8 if context_result.recommended_action == DecisionType.BLOCK else 0.2
            
            component_logits['lpe'].append([1-lpe_logit, lpe_logit])
            component_logits['classifier'].append([1-classifier_logit, classifier_logit])
            component_logits['intent'].append([1-intent_logit, intent_logit])
            
            # Convert true label to binary
            true_labels.append(1 if true_label in ['block', 'review'] else 0)
        
        # Convert to numpy arrays
        for component in component_logits:
            component_logits[component] = np.array(component_logits[component])
        true_labels = np.array(true_labels)
        
        # Fit temperature scaling
        self.temperature_scaler.fit(component_logits, true_labels)
        
        # Optimize ensemble weights
        component_predictions = {}
        for component, logits in component_logits.items():
            component_predictions[component] = logits[:, 1]  # Use positive class probabilities
        
        self.weights = self.weight_optimizer.optimize_weights(
            component_predictions, true_labels, self.weights
        )
        
        logger.info("Calibration training completed")
    
    def update_component_performance(
        self,
        component_predictions: Dict[str, bool],
        component_confidences: Dict[str, float]
    ) -> None:
        """
        Update component performance for adaptive weight management.
        
        Args:
            component_predictions: Dictionary of {component_name: prediction_correct}
            component_confidences: Dictionary of {component_name: confidence_score}
        """
        for component_name, correct in component_predictions.items():
            confidence = component_confidences.get(component_name, 0.5)
            self.weight_manager.update_performance(component_name, correct, confidence)
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all components."""
        return self.weight_manager.get_performance_summary()
    
    def generate_detailed_explanation(
        self,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult,
        aggregated_result: AggregatedResult,
        decision_context: DecisionContext = None,
        explanation_level: ExplanationLevel = ExplanationLevel.BASIC
    ) -> List[str]:
        """
        Generate detailed explanation for a specific result.
        
        Args:
            lpe_result: Result from lexicon engine
            classifier_result: Result from ML classifier
            context_result: Result from context analysis
            aggregated_result: Final aggregated result
            decision_context: Context information
            explanation_level: Level of detail for explanation
            
        Returns:
            List of explanation strings
        """
        return self.explanation_generator.generate_explanation(
            lpe_result, classifier_result, context_result,
            aggregated_result, decision_context, explanation_level,
            self.weight_manager.get_adaptive_weights(self.weights)
        )
    
    def get_attribution_report(
        self,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult,
        aggregated_result: AggregatedResult
    ) -> Dict[str, float]:
        """
        Get attribution report showing component contributions.
        
        Args:
            lpe_result: Result from lexicon engine
            classifier_result: Result from ML classifier
            context_result: Result from context analysis
            aggregated_result: Final aggregated result
            
        Returns:
            Dictionary mapping component names to attribution scores
        """
        decision_trace = self.explanation_generator._create_decision_trace(
            lpe_result, classifier_result, context_result,
            aggregated_result, self.weight_manager.get_adaptive_weights(self.weights)
        )
        
        return self.explanation_generator.generate_attribution_report(
            decision_trace, aggregated_result
        )