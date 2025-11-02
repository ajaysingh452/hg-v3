"""Demonstration of Corporate Policy Layer functionality."""

import sys
from pathlib import Path

# Add harmony-guard to path
sys.path.append(str(Path(__file__).parent.parent))

from core.models import (
    AggregatedResult, DecisionType, SeverityLevel, ProblemSpan
)
from policy.policy_engine import PolicyEngine


def create_sample_result(decision, confidence, categories, severity):
    """Create a sample aggregated result for testing."""
    return AggregatedResult(
        final_decision=decision,
        confidence_score=confidence,
        category_scores=categories,
        severity_level=severity,
        explanation_traces=[
            "Ensemble aggregation completed",
            f"Final confidence: {confidence:.3f}"
        ],
        consolidated_spans=[
            ProblemSpan(
                text="problematic content",
                start=10,
                end=30,
                category=list(categories.keys())[0] if categories else "unknown",
                confidence=confidence,
                rule_source="test"
            )
        ]
    )


def demo_basic_policy_application():
    """Demonstrate basic policy application."""
    print("=== Basic Policy Application Demo ===\n")
    
    # Initialize policy engine
    policy_engine = PolicyEngine()
    
    # Test case 1: Content that should be blocked
    print("1. High-confidence threat content:")
    threat_result = create_sample_result(
        decision=DecisionType.ALLOW,
        confidence=0.85,
        categories={'threat/violence': 0.8},
        severity=SeverityLevel.HIGH
    )
    
    modified_result = policy_engine.apply_policy(
        aggregated_result=threat_result,
        request_id="demo_001"
    )
    
    print(f"   Original decision: {threat_result.final_decision.value}")
    print(f"   Policy decision: {modified_result.final_decision.value}")
    print(f"   Confidence change: {modified_result.confidence_score - threat_result.confidence_score:+.3f}")
    print(f"   Policy traces: {len(modified_result.explanation_traces) - len(threat_result.explanation_traces)} added")
    print()


def demo_department_overrides():
    """Demonstrate department-specific overrides."""
    print("=== Department Override Demo ===\n")
    
    policy_engine = PolicyEngine()
    
    # Test case: HR department discussing harassment
    print("2. HR department sensitive discussion:")
    harassment_result = create_sample_result(
        decision=DecisionType.REVIEW,
        confidence=0.7,
        categories={'insult/harassment': 0.65},
        severity=SeverityLevel.MEDIUM
    )
    
    # Without department context
    normal_result = policy_engine.apply_policy(
        aggregated_result=harassment_result,
        request_id="demo_002a"
    )
    
    # With HR department context
    hr_result = policy_engine.apply_policy(
        aggregated_result=harassment_result,
        request_id="demo_002b",
        department="hr"
    )
    
    print(f"   Normal processing: {normal_result.final_decision.value} (confidence: {normal_result.confidence_score:.3f})")
    print(f"   HR department: {hr_result.final_decision.value} (confidence: {hr_result.confidence_score:.3f})")
    print(f"   HR adjustment: {hr_result.confidence_score - normal_result.confidence_score:+.3f}")
    print()


def demo_tenant_specific_policy():
    """Demonstrate tenant-specific policy configuration."""
    print("=== Tenant-Specific Policy Demo ===\n")
    
    policy_engine = PolicyEngine()
    
    # Test case: Tech company with more lenient profanity rules
    print("3. Tech company profanity handling:")
    profanity_result = create_sample_result(
        decision=DecisionType.REVIEW,
        confidence=0.75,
        categories={'obscenity/profanity': 0.8},
        severity=SeverityLevel.MEDIUM
    )
    
    # Default policy
    default_result = policy_engine.apply_policy(
        aggregated_result=profanity_result,
        request_id="demo_003a"
    )
    
    # Tech corp policy (if config exists)
    tech_result = policy_engine.apply_policy(
        aggregated_result=profanity_result,
        request_id="demo_003b",
        tenant_id="tech_corp"
    )
    
    print(f"   Default policy: {default_result.final_decision.value}")
    print(f"   Tech corp policy: {tech_result.final_decision.value}")
    print()


def demo_policy_summary():
    """Demonstrate policy configuration summary."""
    print("=== Policy Configuration Summary ===\n")
    
    policy_engine = PolicyEngine()
    
    # Get default policy summary
    default_summary = policy_engine.get_policy_summary()
    print("4. Default policy configuration:")
    for key, value in default_summary.items():
        print(f"   {key}: {value}")
    print()
    
    # Get tech corp policy summary (if exists)
    try:
        tech_summary = policy_engine.get_policy_summary("tech_corp")
        print("5. Tech corp policy configuration:")
        for key, value in tech_summary.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"5. Tech corp policy not available: {e}")
    print()


def demo_audit_trail():
    """Demonstrate audit trail functionality."""
    print("=== Audit Trail Demo ===\n")
    
    policy_engine = PolicyEngine()
    
    # Process some content with audit logging
    test_result = create_sample_result(
        decision=DecisionType.ALLOW,
        confidence=0.6,
        categories={'spam/scam': 0.7},
        severity=SeverityLevel.MEDIUM
    )
    
    modified_result = policy_engine.apply_policy(
        aggregated_result=test_result,
        request_id="audit_demo_001",
        tenant_id="demo_tenant",
        user_id="demo_user",
        department="security"
    )
    
    print("6. Content processed with audit logging:")
    print(f"   Request ID: audit_demo_001")
    print(f"   Decision: {modified_result.final_decision.value}")
    print(f"   Audit trail created for compliance")
    
    # Try to retrieve decision explanation
    explanation = policy_engine.get_decision_explanation("audit_demo_001")
    if explanation:
        print(f"   Retrieved explanation: {len(explanation)} fields")
        print(f"   Applied rules: {len(explanation.get('applied_rules', []))}")
    else:
        print("   Explanation not immediately available (may be in log files)")
    print()


if __name__ == "__main__":
    print("Corporate Policy Layer Demonstration")
    print("=" * 50)
    print()
    
    try:
        demo_basic_policy_application()
        demo_department_overrides()
        demo_tenant_specific_policy()
        demo_policy_summary()
        demo_audit_trail()
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()