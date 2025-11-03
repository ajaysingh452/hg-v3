"""Policy audit logging and trace generation for compliance and explainability."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import uuid

from ..core.models import AggregatedResult, DecisionType, SeverityLevel
from .rule_engine import PolicyRuleResult


@dataclass
class PolicyDecisionTrace:
    """Detailed trace of policy decision process."""
    request_id: str
    tenant_id: Optional[str]
    timestamp: datetime
    original_decision: DecisionType
    final_decision: DecisionType
    confidence_original: float
    confidence_adjusted: float
    applied_rules: List[str]
    policy_version: str
    department_context: Optional[str] = None
    override_reason: Optional[str] = None
    safe_context_detected: Optional[str] = None
    hard_rule_triggered: bool = False


@dataclass
class PolicyChangeEvent:
    """Policy configuration change event."""
    event_id: str
    tenant_id: Optional[str]
    timestamp: datetime
    change_type: str  # 'create', 'update', 'delete'
    changed_by: str
    old_config_hash: Optional[str]
    new_config_hash: str
    changes_summary: Dict[str, Any]
    approval_required: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


@dataclass
class AuditLogEntry:
    """Audit log entry for policy decisions."""
    entry_id: str
    timestamp: datetime
    event_type: str  # 'decision', 'config_change', 'override'
    tenant_id: Optional[str]
    user_id: Optional[str]
    details: Dict[str, Any]
    content_hash: Optional[str] = None  # Hash of analyzed content for correlation
    compliance_flags: List[str] = None


class PolicyAuditLogger:
    """Handles audit logging and trace generation for policy decisions."""
    
    def __init__(self, log_dir: str = None, enable_file_logging: bool = True):
        """
        Initialize policy audit logger.
        
        Args:
            log_dir: Directory for audit log files
            enable_file_logging: Whether to write audit logs to files
        """
        self.enable_file_logging = enable_file_logging
        
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs" / "audit"
        self.log_dir = Path(log_dir)
        
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up structured logging
        self.logger = logging.getLogger(f"{__name__}.audit")
        self.logger.setLevel(logging.INFO)
        
        # Create audit-specific handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_policy_decision(
        self,
        request_id: str,
        aggregated_result: AggregatedResult,
        policy_result: PolicyRuleResult,
        tenant_id: str = None,
        department: str = None,
        user_id: str = None,
        content_text: str = None
    ) -> PolicyDecisionTrace:
        """
        Log a policy decision with full trace information.
        
        Args:
            request_id: Unique request identifier
            aggregated_result: Original ensemble result
            policy_result: Policy engine result
            tenant_id: Optional tenant identifier
            department: Optional department context
            user_id: Optional user identifier
            content_text: Optional content text for hashing
            
        Returns:
            PolicyDecisionTrace object
        """
        # Create decision trace
        trace = PolicyDecisionTrace(
            request_id=request_id,
            tenant_id=tenant_id,
            timestamp=datetime.now(),
            original_decision=aggregated_result.final_decision,
            final_decision=DecisionType(policy_result.decision.value),
            confidence_original=aggregated_result.confidence_score,
            confidence_adjusted=aggregated_result.confidence_score + policy_result.confidence_adjustment,
            applied_rules=policy_result.applied_rules,
            policy_version="1.0",  # TODO: Get from policy profile
            department_context=department,
            override_reason=policy_result.override_reason,
            hard_rule_triggered=any("Hard rule" in rule for rule in policy_result.applied_rules)
        )
        
        # Generate content hash for correlation (without storing content)
        content_hash = None
        if content_text:
            content_hash = hashlib.sha256(content_text.encode('utf-8')).hexdigest()[:16]
        
        # Create audit log entry
        audit_entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=trace.timestamp,
            event_type='decision',
            tenant_id=tenant_id,
            user_id=user_id,
            details={
                'request_id': request_id,
                'original_decision': trace.original_decision.value,
                'final_decision': trace.final_decision.value,
                'confidence_change': policy_result.confidence_adjustment,
                'applied_rules': policy_result.applied_rules,
                'department': department,
                'override_applied': policy_result.override_reason is not None,
                'hard_rule_triggered': trace.hard_rule_triggered,
                'category_scores': aggregated_result.category_scores,
                'severity_level': aggregated_result.severity_level.value
            },
            content_hash=content_hash,
            compliance_flags=self._generate_compliance_flags(trace, aggregated_result)
        )
        
        # Log the decision
        self._write_audit_log(audit_entry)
        
        # Log structured trace for monitoring
        self.logger.info(
            "Policy decision applied",
            extra={
                'request_id': request_id,
                'tenant_id': tenant_id,
                'original_decision': trace.original_decision.value,
                'final_decision': trace.final_decision.value,
                'rules_applied': len(policy_result.applied_rules),
                'override_applied': policy_result.override_reason is not None,
                'department': department
            }
        )
        
        return trace
    
    def log_policy_change(
        self,
        tenant_id: str,
        change_type: str,
        changed_by: str,
        old_config: Dict[str, Any] = None,
        new_config: Dict[str, Any] = None,
        changes_summary: Dict[str, Any] = None
    ) -> PolicyChangeEvent:
        """
        Log a policy configuration change.
        
        Args:
            tenant_id: Tenant identifier
            change_type: Type of change ('create', 'update', 'delete')
            changed_by: User who made the change
            old_config: Previous configuration
            new_config: New configuration
            changes_summary: Summary of changes made
            
        Returns:
            PolicyChangeEvent object
        """
        # Generate configuration hashes
        old_config_hash = None
        if old_config:
            old_config_hash = hashlib.sha256(
                json.dumps(old_config, sort_keys=True).encode('utf-8')
            ).hexdigest()[:16]
        
        new_config_hash = hashlib.sha256(
            json.dumps(new_config or {}, sort_keys=True).encode('utf-8')
        ).hexdigest()[:16]
        
        # Create change event
        change_event = PolicyChangeEvent(
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            timestamp=datetime.now(),
            change_type=change_type,
            changed_by=changed_by,
            old_config_hash=old_config_hash,
            new_config_hash=new_config_hash,
            changes_summary=changes_summary or {},
            approval_required=self._requires_approval(changes_summary or {})
        )
        
        # Create audit log entry
        audit_entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=change_event.timestamp,
            event_type='config_change',
            tenant_id=tenant_id,
            user_id=changed_by,
            details={
                'change_event_id': change_event.event_id,
                'change_type': change_type,
                'old_config_hash': old_config_hash,
                'new_config_hash': new_config_hash,
                'changes_summary': changes_summary,
                'approval_required': change_event.approval_required
            },
            compliance_flags=self._generate_change_compliance_flags(change_event)
        )
        
        # Log the change
        self._write_audit_log(audit_entry)
        
        # Log structured event for monitoring
        self.logger.info(
            "Policy configuration changed",
            extra={
                'tenant_id': tenant_id,
                'change_type': change_type,
                'changed_by': changed_by,
                'approval_required': change_event.approval_required,
                'changes_count': len(changes_summary or {})
            }
        )
        
        return change_event
    
    def log_override_event(
        self,
        request_id: str,
        tenant_id: str,
        user_id: str,
        override_reason: str,
        original_decision: DecisionType,
        override_decision: DecisionType,
        justification: str = None
    ):
        """
        Log a manual override event.
        
        Args:
            request_id: Original request identifier
            tenant_id: Tenant identifier
            user_id: User who applied override
            override_reason: Reason for override
            original_decision: Original system decision
            override_decision: Override decision
            justification: Additional justification text
        """
        audit_entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type='override',
            tenant_id=tenant_id,
            user_id=user_id,
            details={
                'request_id': request_id,
                'override_reason': override_reason,
                'original_decision': original_decision.value,
                'override_decision': override_decision.value,
                'justification': justification
            },
            compliance_flags=['manual_override']
        )
        
        self._write_audit_log(audit_entry)
        
        self.logger.warning(
            "Manual override applied",
            extra={
                'request_id': request_id,
                'tenant_id': tenant_id,
                'user_id': user_id,
                'original_decision': original_decision.value,
                'override_decision': override_decision.value
            }
        )
    
    def get_decision_trace(self, request_id: str) -> Optional[PolicyDecisionTrace]:
        """
        Retrieve decision trace for a specific request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            PolicyDecisionTrace if found, None otherwise
        """
        if not self.enable_file_logging:
            return None
        
        # Search through recent audit logs
        today = datetime.now().strftime('%Y-%m-%d')
        audit_file = self.log_dir / f"audit_{today}.jsonl"
        
        if not audit_file.exists():
            return None
        
        try:
            with open(audit_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry_data = json.loads(line.strip())
                    if (entry_data.get('event_type') == 'decision' and 
                        entry_data.get('details', {}).get('request_id') == request_id):
                        
                        # Reconstruct PolicyDecisionTrace from audit entry
                        details = entry_data['details']
                        return PolicyDecisionTrace(
                            request_id=request_id,
                            tenant_id=entry_data.get('tenant_id'),
                            timestamp=datetime.fromisoformat(entry_data['timestamp']),
                            original_decision=DecisionType(details['original_decision']),
                            final_decision=DecisionType(details['final_decision']),
                            confidence_original=details.get('confidence_original', 0.0),
                            confidence_adjusted=details.get('confidence_adjusted', 0.0),
                            applied_rules=details.get('applied_rules', []),
                            policy_version="1.0",
                            department_context=details.get('department'),
                            override_reason=details.get('override_reason'),
                            hard_rule_triggered=details.get('hard_rule_triggered', False)
                        )
        except Exception as e:
            self.logger.error(f"Error retrieving decision trace: {e}")
        
        return None
    
    def _write_audit_log(self, audit_entry: AuditLogEntry):
        """Write audit entry to log file and structured logger."""
        # Convert to dictionary for JSON serialization
        entry_dict = asdict(audit_entry)
        entry_dict['timestamp'] = audit_entry.timestamp.isoformat()
        
        # Write to file if enabled
        if self.enable_file_logging:
            today = datetime.now().strftime('%Y-%m-%d')
            audit_file = self.log_dir / f"audit_{today}.jsonl"
            
            try:
                with open(audit_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry_dict) + '\n')
            except Exception as e:
                self.logger.error(f"Error writing audit log: {e}")
        
        # Log to structured logger
        self.logger.info(
            f"Audit event: {audit_entry.event_type}",
            extra={
                'audit_entry_id': audit_entry.entry_id,
                'event_type': audit_entry.event_type,
                'tenant_id': audit_entry.tenant_id,
                'compliance_flags': audit_entry.compliance_flags
            }
        )
    
    def _generate_compliance_flags(
        self, 
        trace: PolicyDecisionTrace, 
        aggregated_result: AggregatedResult
    ) -> List[str]:
        """Generate compliance flags for decision trace."""
        flags = []
        
        # Flag high-risk decisions
        if trace.final_decision == DecisionType.BLOCK:
            flags.append('blocked_content')
        
        if trace.hard_rule_triggered:
            flags.append('hard_rule_violation')
        
        if trace.override_reason:
            flags.append('policy_override')
        
        # Flag high-confidence decisions that were overridden
        if (trace.confidence_original > 0.8 and 
            trace.original_decision != trace.final_decision):
            flags.append('high_confidence_override')
        
        # Flag sensitive categories
        sensitive_categories = ['threat/violence', 'hate/targeted group', 'self-harm encouragement']
        for category in sensitive_categories:
            if category in aggregated_result.category_scores:
                flags.append(f'sensitive_{category.replace("/", "_").replace(" ", "_")}')
        
        return flags
    
    def _generate_change_compliance_flags(self, change_event: PolicyChangeEvent) -> List[str]:
        """Generate compliance flags for configuration changes."""
        flags = ['config_change']
        
        if change_event.approval_required:
            flags.append('requires_approval')
        
        if change_event.change_type == 'delete':
            flags.append('config_deletion')
        
        # Check for sensitive changes
        changes = change_event.changes_summary
        if 'hard_rules' in changes:
            flags.append('hard_rules_modified')
        
        if 'block_thresholds' in changes:
            flags.append('thresholds_modified')
        
        return flags
    
    def _requires_approval(self, changes_summary: Dict[str, Any]) -> bool:
        """Determine if configuration change requires approval."""
        # Changes that require approval
        approval_required_changes = [
            'hard_rules',
            'block_thresholds',
            'department_overrides'
        ]
        
        return any(key in changes_summary for key in approval_required_changes)
    
    def generate_compliance_report(
        self, 
        tenant_id: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report for audit purposes.
        
        Args:
            tenant_id: Optional tenant filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Compliance report dictionary
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'tenant_id': tenant_id,
            'period': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None
            },
            'summary': {
                'total_decisions': 0,
                'blocked_content': 0,
                'overrides_applied': 0,
                'hard_rule_violations': 0,
                'config_changes': 0
            },
            'compliance_flags': {},
            'risk_indicators': []
        }
        
        # This would typically query a database or search through log files
        # For now, return the template structure
        return report