"""Ensemble aggregator for combining LPE, Classifier, and Intent results."""

import asyncio
from typing import Dict, List, Any
from ..core.models import (
    LPEResult, ClassifierResult, ContextResult, AggregatedResult,
    DecisionType, SeverityLevel, ProblemSpan
)
from ..core.aggregator import EnsembleAggregator as CoreAggregator


class EnsembleAggregator:
    """Main ensemble aggregator for the Harmony Guard system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ensemble aggregator."""
        self.config = config
        self.core_aggregator = CoreAggregator(config)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the aggregator."""
        await self.core_aggregator.initialize()
        self._initialized = True
    
    async def aggregate(
        self, 
        lpe_result: LPEResult, 
        classifier_result: ClassifierResult,
        context_result: ContextResult
    ) -> AggregatedResult:
        """
        Aggregate results from all ensemble components.
        
        Args:
            lpe_result: Result from Lexicon & Pattern Engine
            classifier_result: Result from Transformer Classifier
            context_result: Result from Intent/Context Layer
            
        Returns:
            AggregatedResult with final decision and explanations
        """
        if not self._initialized:
            raise RuntimeError("EnsembleAggregator not initialized")
        
        try:
            # Use the core aggregator to combine results
            result = await self.core_aggregator.aggregate_results(
                lpe_result, classifier_result, context_result
            )
            
            return result
            
        except Exception as e:
            # Fallback aggregation in case of errors
            return self._fallback_aggregation(lpe_result, classifier_result, context_result)
    
    def _fallback_aggregation(
        self, 
        lpe_result: LPEResult, 
        classifier_result: ClassifierResult,
        context_result: ContextResult
    ) -> AggregatedResult:
        """Fallback aggregation when main aggregation fails."""
        
        # Simple fallback logic
        lpe_confidence = max(lpe_result.confidence_scores.values(), default=0.0)
        classifier_confidence = max(classifier_result.category_probabilities.values(), default=0.0)
        
        # Use the higher confidence source
        if lpe_confidence > classifier_confidence:
            final_decision = DecisionType.BLOCK if lpe_confidence > 0.7 else DecisionType.REVIEW
            confidence_score = lpe_confidence
            categories = lpe_result.categories
            spans = lpe_result.matched_spans
        else:
            final_decision = DecisionType.BLOCK if classifier_confidence > 0.7 else DecisionType.REVIEW
            confidence_score = classifier_confidence
            categories = list(classifier_result.category_probabilities.keys())
            spans = classifier_result.attention_spans
        
        # Apply context modifiers
        if context_result.context_modifiers:
            avg_modifier = sum(context_result.context_modifiers.values()) / len(context_result.context_modifiers)
            confidence_score *= avg_modifier
            
            if avg_modifier < 0.5:
                final_decision = DecisionType.ALLOW
        
        # Determine severity
        if confidence_score > 0.9:
            severity = SeverityLevel.CRITICAL
        elif confidence_score > 0.7:
            severity = SeverityLevel.HIGH
        elif confidence_score > 0.5:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW
        
        return AggregatedResult(
            final_decision=final_decision,
            confidence_score=confidence_score,
            category_scores={cat: confidence_score for cat in categories},
            severity_level=severity,
            explanation_traces=["Fallback aggregation used due to processing error"],
            consolidated_spans=spans
        )
    
    async def is_healthy(self):
        """Check if the aggregator is healthy."""
        return self._initialized and await self.core_aggregator.is_healthy()
    
    async def shutdown(self):
        """Shutdown the aggregator."""
        if self.core_aggregator:
            await self.core_aggregator.shutdown()
        self._initialized = False