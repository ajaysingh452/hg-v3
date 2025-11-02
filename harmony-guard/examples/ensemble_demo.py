"""Demo script showing ensemble aggregator functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.aggregator import EnsembleAggregator
from core.models import *
from core.explanation import ExplanationLevel
from configs.manager import ConfigurationManager


def create_sample_results():
    """Create sample component results for demonstration."""
    
    # Sample LPE result
    lpe_result = LPEResult(
        matched_spans=[
            ProblemSpan("damn", 5, 9, "obscenity/profanity", 0.85, "profanity_lexicon"),
            ProblemSpan("stupid", 15, 21, "insult/harassment", 0.75, "insult_pattern")
        ],
        categories=["obscenity/profanity", "insult/harassment"],
        confidence_scores={
            "obscenity/profanity": 0.85,
            "insult/harassment": 0.75,
            "hate/targeted group": 0.0
        },
        rule_traces=[
            "Matched profanity pattern: damn",
            "Matched insult pattern: stupid",
            "Elongation detected: stuuupid -> stupid"
        ]
    )
    
    # Sample classifier result
    classifier_result = ClassifierResult(
        category_probabilities={
            "obscenity/profanity": 0.78,
            "insult/harassment": 0.82,
            "hate/targeted group": 0.15,
            "threat/violence": 0.05
        },
        corporate_decision_prob={
            "allow": 0.15,
            "review": 0.35,
            "block": 0.50
        },
        severity_scores={
            "low": 0.20,
            "medium": 0.60,
            "high": 0.20,
            "critical": 0.00
        },
        attention_spans=[
            ProblemSpan("damn stupid", 5, 21, "insult/harassment", 0.80, "classifier_attention")
        ]
    )
    
    # Sample context result
    context_result = ContextResult(
        context_modifiers={
            "obscenity/profanity": 1.0,  # No modification
            "insult/harassment": 0.9,    # Slight reduction
            "hate/targeted group": 1.2   # Slight increase
        },
        safe_context_detected={
            "obscenity/profanity": False,
            "insult/harassment": False,
            "hate/targeted group": False
        },
        recommended_action=DecisionType.BLOCK
    )
    
    return lpe_result, classifier_result, context_result


def main():
    """Main demo function."""
    print("=== Harmony Guard Ensemble Aggregator Demo ===\n")
    
    # Initialize aggregator
    config_manager = ConfigurationManager()
    aggregator = EnsembleAggregator(config_manager)
    
    print(f"Ensemble Configuration:")
    print(f"  Weights: {aggregator.weights}")
    print(f"  Review Threshold: {aggregator.review_threshold}")
    print(f"  Block Threshold: {aggregator.block_threshold}")
    print()
    
    # Create sample results
    lpe_result, classifier_result, context_result = create_sample_results()
    
    # Sample text
    original_text = "This damn stupid idea won't work"
    
    print(f"Analyzing text: '{original_text}'\n")
    
    # Perform aggregation
    result = aggregator.aggregate(
        lpe_result=lpe_result,
        classifier_result=classifier_result,
        context_result=context_result,
        original_text=original_text,
        language_confidence=0.95,
        primary_language="en"
    )
    
    # Display results
    print("=== Aggregation Results ===")
    print(f"Final Decision: {result.final_decision.value}")
    print(f"Confidence: {result.confidence_score:.3f}")
    print(f"Severity: {result.severity_level.value}")
    print()
    
    print("Category Scores:")
    for category, score in sorted(result.category_scores.items(), key=lambda x: x[1], reverse=True):
        if score > 0.1:
            print(f"  {category}: {score:.3f}")
    print()
    
    print("Consolidated Spans:")
    for i, span in enumerate(result.consolidated_spans, 1):
        print(f"  {i}. '{span.text}' ({span.start}-{span.end})")
        print(f"     Category: {span.category}, Confidence: {span.confidence:.3f}")
        print(f"     Source: {span.rule_source}")
    print()
    
    print("Basic Explanation:")
    for explanation in result.explanation_traces[:3]:
        print(f"  • {explanation}")
    print()
    
    # Generate detailed explanation
    detailed_explanation = aggregator.generate_detailed_explanation(
        lpe_result, classifier_result, context_result, result,
        explanation_level=ExplanationLevel.DETAILED
    )
    
    print("=== Detailed Explanation ===")
    for explanation in detailed_explanation:
        print(explanation)
    print()
    
    # Generate attribution report
    attribution = aggregator.get_attribution_report(
        lpe_result, classifier_result, context_result, result
    )
    
    print("=== Component Attribution ===")
    for component, contribution in attribution.items():
        print(f"  {component}: {contribution:.1%}")
    print()
    
    # Simulate performance tracking
    print("=== Performance Tracking Demo ===")
    aggregator.update_component_performance(
        {'lpe': True, 'classifier': True, 'intent': False},
        {'lpe': 0.85, 'classifier': 0.80, 'intent': 0.60}
    )
    
    performance = aggregator.get_performance_summary()
    print("Component Performance:")
    for component, stats in performance.items():
        print(f"  {component}:")
        print(f"    Accuracy: {stats['accuracy']:.1%}")
        print(f"    Avg Confidence: {stats['avg_confidence']:.3f}")
        print(f"    Sample Count: {stats['sample_count']}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()