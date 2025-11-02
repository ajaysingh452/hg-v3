"""Policy engine for applying corporate policies to analysis results."""

import asyncio
from typing import Dict, List, Any, Optional
from ..core.models import AggregatedResult, DecisionType, SeverityLevel
from ..policy.policy_engine import PolicyEngine as CorePolicyEngine


class PolicyEngine:
    """Main policy engine for the Harmony Guard system."""
    
    def __init__(self, config_manager):
        """Initialize the policy engine."""
        self.config_manager = config_manager
        self.core_policy_engine = CorePolicyEngine(config_manager)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the policy engine."""
        await self.core_policy_engine.initialize()
        self._initialized = True
    
    async def apply_policy(
        self, 
        aggregated_result: AggregatedResult, 
        tenant_id: Optional[str] = None
    ) -> AggregatedResult:
        """
        Apply corporate policy rules to the aggregated result.
        
        Args:
            aggregated_result: Result from ensemble aggregation
            tenant_id: Optional tenant identifier for policy customization
            
        Returns:
            AggregatedResult with policy-adjusted decision
        """
        if not self._initialized:
            raise RuntimeError("PolicyEngine not initialized")
        
        try:
            # Apply policy rules using the core policy engine
            result = await self.core_policy_engine.apply_policy_rules(
                aggregated_result, tenant_id
            )
            
            return result
            
        except Exception as e:
            # Fallback policy application
            return self._fallback_policy(aggregated_result, tenant_id)
    
    def _fallback_policy(
        self, 
        aggregated_result: AggregatedResult, 
        tenant_id: Optional[str] = None
    ) -> AggregatedResult:
        """Fallback policy when main policy engine fails."""
        
        # Conservative fallback - err on the side of caution
        if aggregated_result.confidence_score > 0.8:
            final_decision = DecisionType.BLOCK
        elif aggregated_result.confidence_score > 0.5:
            final_decision = DecisionType.REVIEW
        else:
            final_decision = DecisionType.ALLOW
        
        # Add policy trace
        policy_trace = getattr(aggregated_result, 'policy_trace', [])
        policy_trace.append("Fallback policy applied due to processing error")
        
        # Create new result with policy decision
        result = AggregatedResult(
            final_decision=final_decision,
            confidence_score=aggregated_result.confidence_score,
            category_scores=aggregated_result.category_scores,
            severity_level=aggregated_result.severity_level,
            explanation_traces=aggregated_result.explanation_traces + ["Fallback policy applied"],
            consolidated_spans=aggregated_result.consolidated_spans
        )
        
        # Add policy trace as attribute
        setattr(result, 'policy_trace', policy_trace)
        
        return result
    
    async def is_healthy(self):
        """Check if the policy engine is healthy."""
        return self._initialized and await self.core_policy_engine.is_healthy()
    
    async def shutdown(self):
        """Shutdown the policy engine."""
        if self.core_policy_engine:
            await self.core_policy_engine.shutdown()
        self._initialized = False