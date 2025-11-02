"""Tests for Corporate Policy Layer components."""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from core.models import (
    AggregatedResult, DecisionType, SeverityLevel, ProblemSpan
)
from policy.config_loader import PolicyConfigLoader, PolicyProfile
from policy.rule_engine import PolicyRuleEngine, PolicyDecision
from policy.audit_logger import PolicyAuditLogger
from policy.policy_engine import PolicyEngine


class TestPolicyConfigLoader:
    """Test policy configuration loading and management."""
    
    def setup_method(self):
        """Set up test configuration directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        
        # Create test policy configuration
        test_policy = {
            'policy_profile': {
                'name': 'test_policy',
                'version': '1.0',
                'block_thresholds': {
                    'threat/violence': {'medium': 0.7, 'high': 0.5, 'critical': 0.3},
                    'hate/targeted group': {'medium': 0.8, 'high': 0.6, 'critical': 0.4}
                },
                'safe_contexts': ['hr_reporting', 'legal_documentation'],
                'department_overrides': {
                    'hr': {'allow_sensitive_discussions': True, 'review_threshold_adjustment': -0.1}
                },
                'hard_rules': [
                    {'category': 'threat/violence', 'severity': 'critical', 'action': 'block', 'override_allowed': False}
                ]
            }
        }
        
        import yaml
        with open(self.config_dir / 'policy_default.yaml', 'w') as f:
            yaml.safe_dump(test_policy, f)
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_default_policy(self):
        """Test loading default policy configuration."""
        loader = PolicyConfigLoader(str(self.config_dir))
        profile = loader.load_policy_profile()
        
        assert profile.name == 'test_policy'
        assert profile.version == '1.0'
        assert 'threat/violence' in profile.block_thresholds
        assert 'hr_reporting' in profile.safe_contexts
        assert 'hr' in profile.department_overrides
        assert len(profile.hard_rules) == 1
    
    def test_get_threshold_config(self):
        """Test getting threshold configuration for category."""
        loader = PolicyConfigLoader(str(self.config_dir))
        threshold = loader.get_threshold_config('threat/violence')
        
        assert threshold is not None
        assert threshold.category == 'threat/violence'
        assert threshold.medium == 0.7
        assert threshold.high == 0.5
        assert threshold.critical == 0.3
    
    def test_get_department_override(self):
        """Test getting department override configuration."""
        loader = PolicyConfigLoader(str(self.config_dir))
        override = loader.get_department_override('hr')
        
        assert override is not None
        assert override.department == 'hr'
        assert override.allow_sensitive_discussions is True
        assert override.review_threshold_adjustment == -0.1


class TestPolicyRuleEngine:
    """Test policy rule engine functionality."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        
        # Create test policy
        test_policy = {
            'policy_profile': {
                'name': 'test_policy',
                'version': '1.0',
                'block_thresholds': {
                    'threat/violence': {'medium': 0.7, 'high': 0.5, 'critical': 0.3},
                    'hate/targeted group': {'medium': 0.8, 'high': 0.6, 'critical': 0.4}
                },
                'safe_contexts': ['hr_reporting'],
                'department_overrides': {
                    'hr': {'allow_sensitive_discussions': True, 'review_threshold_adjustment': -0.1}
                },
                'hard_rules': [
                    {'category': 'threat/violence', 'severity': 'critical', 'action': 'block', 'override_allowed': False}
                ]
            }
        }
        
        import yaml
        with open(self.config_dir / 'policy_default.yaml', 'w') as f:
            yaml.safe_dump(test_policy, f)
        
        self.config_loader = PolicyConfigLoader(str(self.config_dir))
        self.rule_engine = PolicyRuleEngine(self.config_loader)
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_threshold_blocking(self):
        """Test threshold-based blocking rules."""
        # Create aggregated result that should be blocked
        result = AggregatedResult(
            final_decision=DecisionType.ALLOW,
            confidence_score=0.8,
            category_scores={'threat/violence': 0.6},  # Above medium threshold (0.5)
            severity_level=SeverityLevel.HIGH,
            explanation_traces=[],
            consolidated_spans=[]
        )
        
        policy_result = self.rule_engine.evaluate_policy(result)
        
        assert policy_result.decision == PolicyDecision.BLOCK
        assert any('Block threshold exceeded' in rule for rule in policy_result.applied_rules)
    
    def test_hard_rule_blocking(self):
        """Test hard rule blocking that cannot be overridden."""
        result = AggregatedResult(
            final_decision=DecisionType.ALLOW,
            confidence_score=0.6,
            category_scores={'threat/violence': 0.7},
            severity_level=SeverityLevel.CRITICAL,
            explanation_traces=[],
            consolidated_spans=[]
        )
        
        policy_result = self.rule_engine.evaluate_policy(result)
        
        assert policy_result.decision == PolicyDecision.BLOCK
        assert any('Hard rule' in rule for rule in policy_result.applied_rules)
        assert 'cannot be overridden' in policy_result.override_reason
    
    def test_department_override(self):
        """Test department-specific overrides."""
        result = AggregatedResult(
            final_decision=DecisionType.REVIEW,
            confidence_score=0.7,
            category_scores={'insult/harassment': 0.6},
            severity_level=SeverityLevel.MEDIUM,
            explanation_traces=[],
            consolidated_spans=[]
        )
        
        policy_result = self.rule_engine.evaluate_policy(result, department='hr')
        
        # HR department gets both base adjustment (-0.1) and sensitive discussion allowance (-0.15)
        assert policy_result.confidence_adjustment <= -0.1  # At least HR adjustment
        assert any('Department override' in rule for rule in policy_result.applied_rules)


class TestPolicyAuditLogger:
    """Test policy audit logging functionality."""
    
    def setup_method(self):
        """Set up test audit logger."""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_logger = PolicyAuditLogger(self.temp_dir, enable_file_logging=True)
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_log_policy_decision(self):
        """Test logging policy decisions."""
        from policy.rule_engine import PolicyRuleResult
        
        # Create test data
        aggregated_result = AggregatedResult(
            final_decision=DecisionType.ALLOW,
            confidence_score=0.6,
            category_scores={'insult/harassment': 0.5},
            severity_level=SeverityLevel.MEDIUM,
            explanation_traces=[],
            consolidated_spans=[]
        )
        
        policy_result = PolicyRuleResult(
            decision=PolicyDecision.REVIEW,
            confidence_adjustment=0.1,
            applied_rules=['Test rule applied'],
            override_reason=None
        )
        
        # Log decision
        trace = self.audit_logger.log_policy_decision(
            request_id='test_123',
            aggregated_result=aggregated_result,
            policy_result=policy_result,
            tenant_id='test_tenant'
        )
        
        assert trace.request_id == 'test_123'
        assert trace.tenant_id == 'test_tenant'
        assert trace.original_decision == DecisionType.ALLOW
        assert trace.final_decision == DecisionType.REVIEW
        assert len(trace.applied_rules) == 1
    
    def test_log_policy_change(self):
        """Test logging policy configuration changes."""
        old_config = {'block_thresholds': {'threat/violence': {'medium': 0.7}}}
        new_config = {'block_thresholds': {'threat/violence': {'medium': 0.8}}}
        
        change_event = self.audit_logger.log_policy_change(
            tenant_id='test_tenant',
            change_type='update',
            changed_by='admin_user',
            old_config=old_config,
            new_config=new_config,
            changes_summary={'block_thresholds': 'Updated threat/violence threshold'}
        )
        
        assert change_event.tenant_id == 'test_tenant'
        assert change_event.change_type == 'update'
        assert change_event.changed_by == 'admin_user'
        assert change_event.old_config_hash is not None
        assert change_event.new_config_hash is not None


class TestPolicyEngine:
    """Test integrated policy engine functionality."""
    
    def setup_method(self):
        """Set up test policy engine."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        
        # Create test policy
        test_policy = {
            'policy_profile': {
                'name': 'test_policy',
                'version': '1.0',
                'block_thresholds': {
                    'threat/violence': {'medium': 0.7, 'high': 0.5, 'critical': 0.3}
                },
                'safe_contexts': ['hr_reporting'],
                'department_overrides': {
                    'hr': {'allow_sensitive_discussions': True, 'review_threshold_adjustment': -0.1}
                },
                'hard_rules': []
            }
        }
        
        import yaml
        with open(self.config_dir / 'policy_default.yaml', 'w') as f:
            yaml.safe_dump(test_policy, f)
        
        self.policy_engine = PolicyEngine(
            config_dir=str(self.config_dir),
            audit_log_dir=str(self.temp_dir),
            enable_audit_logging=True
        )
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_apply_policy(self):
        """Test applying policy to aggregated result."""
        result = AggregatedResult(
            final_decision=DecisionType.ALLOW,
            confidence_score=0.8,  # Higher confidence
            category_scores={'threat/violence': 0.8},  # Well above medium threshold (0.7)
            severity_level=SeverityLevel.HIGH,  # Higher severity
            explanation_traces=[],
            consolidated_spans=[]
        )
        
        modified_result = self.policy_engine.apply_policy(
            aggregated_result=result,
            tenant_id='test_tenant',
            request_id='test_123'
        )
        
        # Should be blocked due to threshold
        assert modified_result.final_decision == DecisionType.BLOCK
        assert len(modified_result.explanation_traces) > len(result.explanation_traces)
    
    def test_get_policy_summary(self):
        """Test getting policy configuration summary."""
        summary = self.policy_engine.get_policy_summary()
        
        assert summary['name'] == 'test_policy'
        assert summary['version'] == '1.0'
        assert summary['categories_configured'] == 1
        assert summary['safe_contexts'] == 1
        assert summary['department_overrides'] == 1


if __name__ == '__main__':
    pytest.main([__file__])