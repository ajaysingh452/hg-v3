"""Main policy engine that integrates configuration, rules, and audit logging."""

from typing import Dict, Optional, Any
import logging

from core.interfaces import PolicyEngineInterface
from core.models import AggregatedResult, DecisionType
from .config_loader import PolicyConfigLoader
from .rule_engine import PolicyRuleEngine, PolicyDecision
from .audit_logger import PolicyAuditLogger

logger = logging.getLogger(__name__)


class PolicyEngine(PolicyEngineInterface):
    """Main policy engine that applies corporate policies to content decisions."""
    
    def __init__(
        self, 
        config_dir: str = None,
        audit_log_dir: str = None,
        enable_audit_logging: bool = True
    ):
        """
        Initialize policy engine.
        
        Args:
            config_dir: Directory containing policy configuration files
            audit_log_dir: Directory for audit log files
            enable_audit_logging: Whether to enable audit logging
        """
        self.config_loader = PolicyConfigLoader(config_dir)
        self.rule_engine = PolicyRuleEngine(self.config_loader)
        self.audit_logger = PolicyAuditLogger(
            audit_log_dir, 
            enable_file_logging=enable_audit_logging
        )
        
    def apply_policy(
        self, 
        aggregated_result: AggregatedResult,
        tenant_id: str = None,
        request_id: str = None,
        department: str = None,
        user_id: str = None,
        user_context: Dict[str, Any] = None,
        content_text: str = None
    ) -> AggregatedResult:
        """
        Apply organization-specific policies to modify final decisions.
        
        Args:
            aggregated_result: Result from ensemble aggregation
            tenant_id: Optional tenant identifier for policy selection
            request_id: Optional request identifier for audit logging
            department: Optional department context
            user_id: Optional user identifier
            user_context: Optional additional user context
            content_text: Optional content text for audit hashing
            
        Returns:
            Modified AggregatedResult with policy rules applied
        """
        try:
            # Evaluate policy rules
            policy_result = self.rule_engine.evaluate_policy(
                aggregated_result=aggregated_result,
                tenant_id=tenant_id,
                department=department,
                user_context=user_context or {}
            )
            
            # Create modified result
            modified_result = self._apply_policy_result(aggregated_result, policy_result)
            
            # Log policy decision for audit trail
            if request_id:
                decision_trace = self.audit_logger.log_policy_decision(
                    request_id=request_id,
                    aggregated_result=aggregated_result,
                    policy_result=policy_result,
                    tenant_id=tenant_id,
                    department=department,
                    user_id=user_id,
                    content_text=content_text
                )
                
                # Add policy trace to explanation
                policy_trace = self._generate_policy_trace(policy_result, decision_trace)
                modified_result.explanation_traces.extend(policy_trace)
            
            logger.debug(
                f"Policy applied for tenant {tenant_id}: "
                f"{aggregated_result.final_decision.value} -> {modified_result.final_decision.value}"
            )
            
            return modified_result
            
        except Exception as e:
            logger.error(f"Error applying policy: {e}")
            # Return original result if policy application fails
            return aggregated_result
    
    def _apply_policy_result(
        self, 
        aggregated_result: AggregatedResult, 
        policy_result
    ) -> AggregatedResult:
        """
        Apply policy rule result to aggregated result.
        
        Args:
            aggregated_result: Original aggregated result
            policy_result: Policy rule evaluation result
            
        Returns:
            Modified AggregatedResult
        """
        # Convert policy decision to DecisionType
        decision_mapping = {
            PolicyDecision.ALLOW: DecisionType.ALLOW,
            PolicyDecision.REVIEW: DecisionType.REVIEW,
            PolicyDecision.BLOCK: DecisionType.BLOCK,
            PolicyDecision.OVERRIDE: aggregated_result.final_decision  # Keep original for overrides
        }
        
        final_decision = decision_mapping.get(policy_result.decision, aggregated_result.final_decision)
        
        # Apply override logic
        if policy_result.decision == PolicyDecision.OVERRIDE and policy_result.override_reason:
            # Override can change block to review or allow based on context
            if aggregated_result.final_decision == DecisionType.BLOCK:
                if "allows" in policy_result.override_reason.lower():
                    final_decision = DecisionType.ALLOW
                else:
                    final_decision = DecisionType.REVIEW
        
        # Adjust confidence score
        adjusted_confidence = max(0.0, min(1.0, 
            aggregated_result.confidence_score + policy_result.confidence_adjustment
        ))
        
        # Create new result with policy modifications
        return AggregatedResult(
            final_decision=final_decision,
            confidence_score=adjusted_confidence,
            category_scores=aggregated_result.category_scores.copy(),
            severity_level=aggregated_result.severity_level,
            explanation_traces=aggregated_result.explanation_traces.copy(),
            consolidated_spans=aggregated_result.consolidated_spans.copy()
        )
    
    def _generate_policy_trace(self, policy_result, decision_trace) -> list:
        """Generate policy trace for explanation."""
        trace_lines = []
        
        if policy_result.applied_rules:
            trace_lines.append("Policy rules applied:")
            for rule in policy_result.applied_rules:
                trace_lines.append(f"  - {rule}")
        
        if policy_result.confidence_adjustment != 0:
            trace_lines.append(
                f"Confidence adjusted by {policy_result.confidence_adjustment:+.3f}"
            )
        
        if policy_result.override_reason:
            trace_lines.append(f"Override applied: {policy_result.override_reason}")
        
        if decision_trace.department_context:
            trace_lines.append(f"Department context: {decision_trace.department_context}")
        
        return trace_lines
    
    def update_tenant_policy(
        self,
        tenant_id: str,
        policy_config: Dict[str, Any],
        changed_by: str,
        changes_summary: Dict[str, Any] = None
    ) -> bool:
        """
        Update tenant-specific policy configuration.
        
        Args:
            tenant_id: Tenant identifier
            policy_config: New policy configuration
            changed_by: User making the change
            changes_summary: Summary of changes made
            
        Returns:
            Success status
        """
        try:
            # Get current configuration for comparison
            old_profile = self.config_loader.load_policy_profile(tenant_id)
            old_config = {
                'block_thresholds': old_profile.block_thresholds,
                'safe_contexts': old_profile.safe_contexts,
                'department_overrides': old_profile.department_overrides,
                'hard_rules': old_profile.hard_rules
            }
            
            # Convert dict to PolicyProfile and save
            from .config_loader import PolicyProfile
            from datetime import datetime
            
            new_profile = PolicyProfile(
                name=policy_config.get('name', f'tenant_{tenant_id}'),
                version=policy_config.get('version', '1.0'),
                block_thresholds=policy_config.get('block_thresholds', {}),
                safe_contexts=policy_config.get('safe_contexts', []),
                department_overrides=policy_config.get('department_overrides', {}),
                hard_rules=policy_config.get('hard_rules', []),
                tenant_lexicon_overlay=policy_config.get('tenant_lexicon_overlay'),
                updated_at=datetime.now()
            )
            
            # Save configuration
            success = self.config_loader.save_tenant_policy(tenant_id, new_profile)
            
            if success:
                # Log configuration change
                self.audit_logger.log_policy_change(
                    tenant_id=tenant_id,
                    change_type='update',
                    changed_by=changed_by,
                    old_config=old_config,
                    new_config=policy_config,
                    changes_summary=changes_summary
                )
                
                logger.info(f"Updated policy configuration for tenant {tenant_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating tenant policy: {e}")
            return False
    
    def validate_tenant_policy(self, tenant_id: str = None) -> Dict[str, Any]:
        """
        Validate tenant policy configuration.
        
        Args:
            tenant_id: Optional tenant identifier
            
        Returns:
            Validation result with warnings and errors
        """
        try:
            warnings = self.rule_engine.validate_policy_consistency(tenant_id)
            
            return {
                'valid': len(warnings) == 0,
                'warnings': warnings,
                'tenant_id': tenant_id
            }
            
        except Exception as e:
            return {
                'valid': False,
                'warnings': [f"Validation error: {e}"],
                'tenant_id': tenant_id
            }
    
    def get_policy_summary(self, tenant_id: str = None) -> Dict[str, Any]:
        """
        Get summary of policy configuration.
        
        Args:
            tenant_id: Optional tenant identifier
            
        Returns:
            Policy configuration summary
        """
        try:
            profile = self.config_loader.load_policy_profile(tenant_id)
            
            return {
                'name': profile.name,
                'version': profile.version,
                'categories_configured': len(profile.block_thresholds),
                'safe_contexts': len(profile.safe_contexts),
                'department_overrides': len(profile.department_overrides),
                'hard_rules': len(profile.hard_rules),
                'has_tenant_lexicon': profile.tenant_lexicon_overlay is not None,
                'updated_at': profile.updated_at.isoformat() if profile.updated_at else None
            }
            
        except Exception as e:
            logger.error(f"Error getting policy summary: {e}")
            return {
                'error': str(e),
                'tenant_id': tenant_id
            }
    
    def get_decision_explanation(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed explanation for a policy decision.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Decision explanation or None if not found
        """
        try:
            trace = self.audit_logger.get_decision_trace(request_id)
            
            if not trace:
                return None
            
            return {
                'request_id': request_id,
                'tenant_id': trace.tenant_id,
                'timestamp': trace.timestamp.isoformat(),
                'original_decision': trace.original_decision.value,
                'final_decision': trace.final_decision.value,
                'confidence_change': trace.confidence_adjusted - trace.confidence_original,
                'applied_rules': trace.applied_rules,
                'department_context': trace.department_context,
                'override_reason': trace.override_reason,
                'hard_rule_triggered': trace.hard_rule_triggered,
                'policy_version': trace.policy_version
            }
            
        except Exception as e:
            logger.error(f"Error getting decision explanation: {e}")
            return None