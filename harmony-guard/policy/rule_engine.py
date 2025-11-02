"""Policy rule engine for threshold-based blocking and safe context processing."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from core.models import AggregatedResult, DecisionType, SeverityLevel
from .config_loader import PolicyConfigLoader, PolicyProfile, ThresholdConfig, DepartmentOverride, HardRule

logger = logging.getLogger(__name__)


class PolicyDecision(str, Enum):
    """Policy decision types."""
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"
    OVERRIDE = "override"


@dataclass
class PolicyRuleResult:
    """Result from policy rule evaluation."""
    decision: PolicyDecision
    confidence_adjustment: float
    applied_rules: List[str]
    override_reason: Optional[str] = None
    department_context: Optional[str] = None


@dataclass
class SafeContextMatch:
    """Safe context detection result."""
    context_type: str
    confidence: float
    matched_patterns: List[str]


class PolicyRuleEngine:
    """Applies policy rules to determine final content decisions."""
    
    def __init__(self, config_loader: PolicyConfigLoader):
        """
        Initialize policy rule engine.
        
        Args:
            config_loader: Policy configuration loader
        """
        self.config_loader = config_loader
        self._safe_context_patterns = {
            'hr_reporting': [
                r'hr\s+report', r'harassment\s+complaint', r'workplace\s+incident',
                r'employee\s+grievance', r'disciplinary\s+action', r'investigation\s+report'
            ],
            'legal_documentation': [
                r'legal\s+brief', r'court\s+filing', r'evidence\s+documentation',
                r'case\s+study', r'legal\s+precedent', r'statute\s+reference'
            ],
            'educational_content': [
                r'training\s+material', r'educational\s+example', r'awareness\s+campaign',
                r'policy\s+training', r'compliance\s+education', r'sensitivity\s+training'
            ],
            'policy_discussion': [
                r'policy\s+review', r'guideline\s+discussion', r'rule\s+clarification',
                r'procedure\s+update', r'compliance\s+meeting', r'governance\s+discussion'
            ]
        }
    
    def evaluate_policy(
        self, 
        aggregated_result: AggregatedResult,
        tenant_id: str = None,
        department: str = None,
        user_context: Dict[str, any] = None
    ) -> PolicyRuleResult:
        """
        Evaluate policy rules against aggregated result.
        
        Args:
            aggregated_result: Result from ensemble aggregation
            tenant_id: Optional tenant identifier
            department: Optional department context
            user_context: Optional additional user context
            
        Returns:
            PolicyRuleResult with final decision and applied rules
        """
        applied_rules = []
        confidence_adjustment = 0.0
        override_reason = None
        
        # Load policy profile for tenant
        policy_profile = self.config_loader.load_policy_profile(tenant_id)
        
        # Check hard rules first (cannot be overridden)
        hard_rule_result = self._apply_hard_rules(
            aggregated_result, 
            policy_profile.hard_rules
        )
        if hard_rule_result:
            applied_rules.extend(hard_rule_result.applied_rules)
            return PolicyRuleResult(
                decision=PolicyDecision.BLOCK,
                confidence_adjustment=0.0,
                applied_rules=applied_rules,
                override_reason="Hard rule violation - cannot be overridden"
            )
        
        # Check for safe contexts
        safe_context_result = self._detect_safe_contexts(
            aggregated_result,
            policy_profile.safe_contexts,
            user_context or {}
        )
        
        if safe_context_result:
            applied_rules.append(f"Safe context detected: {safe_context_result.context_type}")
            confidence_adjustment -= 0.2  # Reduce blocking confidence in safe contexts
        
        # Apply department-specific overrides
        if department:
            dept_override = self.config_loader.get_department_override(department, tenant_id)
            if dept_override:
                dept_result = self._apply_department_overrides(
                    aggregated_result,
                    dept_override,
                    safe_context_result
                )
                applied_rules.extend(dept_result.applied_rules)
                confidence_adjustment += dept_result.confidence_adjustment
                if dept_result.override_reason:
                    override_reason = dept_result.override_reason
        
        # Apply threshold-based blocking rules
        threshold_result = self._apply_threshold_rules(
            aggregated_result,
            policy_profile,
            confidence_adjustment
        )
        applied_rules.extend(threshold_result.applied_rules)
        
        return PolicyRuleResult(
            decision=threshold_result.decision,
            confidence_adjustment=confidence_adjustment,
            applied_rules=applied_rules,
            override_reason=override_reason,
            department_context=department
        )
    
    def _apply_hard_rules(
        self, 
        aggregated_result: AggregatedResult,
        hard_rules: List[Dict[str, any]]
    ) -> Optional[PolicyRuleResult]:
        """
        Apply hard rules that cannot be overridden.
        
        Args:
            aggregated_result: Aggregated analysis result
            hard_rules: List of hard rule configurations
            
        Returns:
            PolicyRuleResult if hard rule triggered, None otherwise
        """
        applied_rules = []
        
        for rule in hard_rules:
            category = rule['category']
            severity = rule['severity']
            action = rule['action']
            
            # Check if this category was detected
            if category in aggregated_result.category_scores:
                category_confidence = aggregated_result.category_scores[category]
                
                # Check if severity matches
                severity_match = False
                if severity == "critical" and aggregated_result.severity_level == SeverityLevel.CRITICAL:
                    severity_match = True
                elif severity == "high" and aggregated_result.severity_level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                    severity_match = True
                elif severity == "medium" and aggregated_result.severity_level in [SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                    severity_match = True
                
                if severity_match and category_confidence > 0.5:  # High confidence threshold for hard rules
                    applied_rules.append(f"Hard rule: {category} at {severity} severity -> {action}")
                    return PolicyRuleResult(
                        decision=PolicyDecision.BLOCK,
                        confidence_adjustment=0.0,
                        applied_rules=applied_rules
                    )
        
        return None
    
    def _detect_safe_contexts(
        self,
        aggregated_result: AggregatedResult,
        safe_contexts: List[str],
        user_context: Dict[str, any]
    ) -> Optional[SafeContextMatch]:
        """
        Detect if content is in a safe context that allows otherwise flagged content.
        
        Args:
            aggregated_result: Aggregated analysis result
            safe_contexts: List of allowed safe contexts
            user_context: Additional user context information
            
        Returns:
            SafeContextMatch if safe context detected, None otherwise
        """
        import re
        
        # Check user-provided context hints
        if 'context_type' in user_context and user_context['context_type'] in safe_contexts:
            return SafeContextMatch(
                context_type=user_context['context_type'],
                confidence=0.9,
                matched_patterns=[f"User-specified context: {user_context['context_type']}"]
            )
        
        # Check for pattern-based safe context detection
        text_to_analyze = ""
        if aggregated_result.consolidated_spans:
            # Analyze surrounding context of flagged spans
            for span in aggregated_result.consolidated_spans:
                text_to_analyze += f" {span.text} "
        
        if user_context.get('full_text'):
            text_to_analyze += user_context['full_text']
        
        if not text_to_analyze.strip():
            return None
        
        text_lower = text_to_analyze.lower()
        
        for context_type in safe_contexts:
            if context_type in self._safe_context_patterns:
                patterns = self._safe_context_patterns[context_type]
                matched_patterns = []
                
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        matched_patterns.append(pattern)
                
                if matched_patterns:
                    confidence = min(0.8, len(matched_patterns) * 0.3)  # Max 0.8 confidence
                    return SafeContextMatch(
                        context_type=context_type,
                        confidence=confidence,
                        matched_patterns=matched_patterns
                    )
        
        return None
    
    def _apply_department_overrides(
        self,
        aggregated_result: AggregatedResult,
        dept_override: DepartmentOverride,
        safe_context: Optional[SafeContextMatch]
    ) -> PolicyRuleResult:
        """
        Apply department-specific policy overrides.
        
        Args:
            aggregated_result: Aggregated analysis result
            dept_override: Department override configuration
            safe_context: Safe context detection result
            
        Returns:
            PolicyRuleResult with department-specific adjustments
        """
        applied_rules = []
        confidence_adjustment = dept_override.review_threshold_adjustment
        override_reason = None
        
        # Check for department-specific allowances
        if dept_override.allow_sensitive_discussions:
            # Allow sensitive discussions for HR department
            sensitive_categories = ['insult/harassment', 'sexual content', 'bullying/taunting']
            for category in sensitive_categories:
                if category in aggregated_result.category_scores:
                    applied_rules.append(f"Department override: {dept_override.department} allows sensitive discussions")
                    confidence_adjustment -= 0.15
                    break
        
        if dept_override.allow_policy_examples:
            # Allow policy examples for legal department
            if safe_context and safe_context.context_type in ['legal_documentation', 'policy_discussion']:
                applied_rules.append(f"Department override: {dept_override.department} allows policy examples")
                confidence_adjustment -= 0.2
                override_reason = "Legal department policy example allowance"
        
        if dept_override.allow_threat_analysis:
            # Allow threat analysis for security department
            threat_categories = ['threat/violence', 'hate/targeted group']
            for category in threat_categories:
                if category in aggregated_result.category_scores:
                    applied_rules.append(f"Department override: {dept_override.department} allows threat analysis")
                    confidence_adjustment -= 0.25
                    override_reason = "Security department threat analysis allowance"
                    break
        
        return PolicyRuleResult(
            decision=PolicyDecision.OVERRIDE,
            confidence_adjustment=confidence_adjustment,
            applied_rules=applied_rules,
            override_reason=override_reason
        )
    
    def _apply_threshold_rules(
        self,
        aggregated_result: AggregatedResult,
        policy_profile: PolicyProfile,
        confidence_adjustment: float
    ) -> PolicyRuleResult:
        """
        Apply threshold-based blocking rules.
        
        Args:
            aggregated_result: Aggregated analysis result
            policy_profile: Policy profile configuration
            confidence_adjustment: Confidence adjustment from other rules
            
        Returns:
            PolicyRuleResult with threshold-based decision
        """
        applied_rules = []
        final_decision = PolicyDecision.ALLOW
        
        # Adjust confidence scores based on policy adjustments
        adjusted_confidence = max(0.0, min(1.0, aggregated_result.confidence_score + confidence_adjustment))
        
        # Check each category against thresholds
        for category, confidence in aggregated_result.category_scores.items():
            if category not in policy_profile.block_thresholds:
                continue
            
            thresholds = policy_profile.block_thresholds[category]
            adjusted_category_confidence = max(0.0, min(1.0, confidence + confidence_adjustment))
            
            # Determine severity-based threshold
            severity_threshold = None
            if aggregated_result.severity_level == SeverityLevel.CRITICAL:
                severity_threshold = thresholds.get('critical', 0.4)
            elif aggregated_result.severity_level == SeverityLevel.HIGH:
                severity_threshold = thresholds.get('high', 0.6)
            elif aggregated_result.severity_level == SeverityLevel.MEDIUM:
                severity_threshold = thresholds.get('medium', 0.8)
            else:  # LOW severity
                severity_threshold = 0.9  # Very high threshold for low severity
            
            if adjusted_category_confidence >= severity_threshold:
                if severity_threshold <= 0.5:  # High confidence threshold
                    final_decision = PolicyDecision.BLOCK
                    applied_rules.append(
                        f"Block threshold exceeded: {category} "
                        f"({adjusted_category_confidence:.3f} >= {severity_threshold})"
                    )
                else:  # Medium confidence threshold
                    if final_decision != PolicyDecision.BLOCK:
                        final_decision = PolicyDecision.REVIEW
                    applied_rules.append(
                        f"Review threshold exceeded: {category} "
                        f"({adjusted_category_confidence:.3f} >= {severity_threshold})"
                    )
        
        # Overall confidence check
        if adjusted_confidence >= 0.9:
            final_decision = PolicyDecision.BLOCK
            applied_rules.append(f"Overall confidence threshold exceeded ({adjusted_confidence:.3f} >= 0.9)")
        elif adjusted_confidence >= 0.7 and final_decision == PolicyDecision.ALLOW:
            final_decision = PolicyDecision.REVIEW
            applied_rules.append(f"Overall confidence review threshold exceeded ({adjusted_confidence:.3f} >= 0.7)")
        
        return PolicyRuleResult(
            decision=final_decision,
            confidence_adjustment=confidence_adjustment,
            applied_rules=applied_rules
        )
    
    def get_allowlist_patterns(self, tenant_id: str = None) -> Dict[str, List[str]]:
        """
        Get safe context allowlist patterns for tenant.
        
        Args:
            tenant_id: Optional tenant identifier
            
        Returns:
            Dictionary of context types to pattern lists
        """
        policy_profile = self.config_loader.load_policy_profile(tenant_id)
        
        # Return patterns for configured safe contexts
        allowed_patterns = {}
        for context_type in policy_profile.safe_contexts:
            if context_type in self._safe_context_patterns:
                allowed_patterns[context_type] = self._safe_context_patterns[context_type]
        
        return allowed_patterns
    
    def validate_policy_consistency(self, tenant_id: str = None) -> List[str]:
        """
        Validate policy configuration for consistency issues.
        
        Args:
            tenant_id: Optional tenant identifier
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        try:
            policy_profile = self.config_loader.load_policy_profile(tenant_id)
            
            # Check threshold consistency
            for category, thresholds in policy_profile.block_thresholds.items():
                critical = thresholds.get('critical', 0.4)
                high = thresholds.get('high', 0.6)
                medium = thresholds.get('medium', 0.8)
                
                if not (critical <= high <= medium):
                    warnings.append(
                        f"Inconsistent thresholds for {category}: "
                        f"critical({critical}) <= high({high}) <= medium({medium})"
                    )
            
            # Check for conflicting department overrides
            for dept, overrides in policy_profile.department_overrides.items():
                adjustment = overrides.get('review_threshold_adjustment', 0.0)
                if adjustment > 0.3:
                    warnings.append(
                        f"Large positive threshold adjustment for {dept} department: {adjustment}"
                    )
                elif adjustment < -0.5:
                    warnings.append(
                        f"Large negative threshold adjustment for {dept} department: {adjustment}"
                    )
            
        except Exception as e:
            warnings.append(f"Error validating policy configuration: {e}")
        
        return warnings