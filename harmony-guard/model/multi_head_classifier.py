"""Multi-head classification system for abuse detection, corporate decisions, and severity."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ..core.models import AbuseCategory, SeverityLevel, DecisionType


logger = logging.getLogger(__name__)


@dataclass
class ClassificationHead:
    """Configuration for a classification head."""
    name: str
    num_classes: int
    class_names: List[str]
    threshold: float = 0.5
    activation: str = "sigmoid"  # or "softmax"


@dataclass
class MultiHeadOutput:
    """Output from multi-head classifier."""
    abuse_categories: Dict[str, float]
    corporate_decision: Dict[str, float]
    severity_levels: Dict[str, float]
    confidence_scores: Dict[str, float]
    predictions: Dict[str, Any]


class MultiHeadClassificationSystem:
    """Multi-head classification system with separate heads for different tasks."""
    
    def __init__(self, config: Dict):
        """
        Initialize multi-head classification system.
        
        Args:
            config: Configuration for the classification system
        """
        self.config = config
        
        # Define classification heads
        self.heads = self._setup_classification_heads()
        
        # Thresholds for each head
        self.thresholds = self._setup_thresholds()
        
        # Class weights for handling imbalanced data
        self.class_weights = self._setup_class_weights()
        
    def _setup_classification_heads(self) -> Dict[str, ClassificationHead]:
        """Setup the different classification heads."""
        heads = {}
        
        # Abuse category head (multi-label)
        abuse_categories = [category.value for category in AbuseCategory]
        heads["abuse_categories"] = ClassificationHead(
            name="abuse_categories",
            num_classes=len(abuse_categories),
            class_names=abuse_categories,
            threshold=0.3,  # Lower threshold for multi-label
            activation="sigmoid"
        )
        
        # Corporate decision head (multi-class)
        corporate_decisions = [decision.value for decision in DecisionType]
        heads["corporate_decision"] = ClassificationHead(
            name="corporate_decision",
            num_classes=len(corporate_decisions),
            class_names=corporate_decisions,
            threshold=0.5,
            activation="softmax"
        )
        
        # Severity level head (ordinal classification)
        severity_levels = [level.value for level in SeverityLevel]
        heads["severity_levels"] = ClassificationHead(
            name="severity_levels",
            num_classes=len(severity_levels),
            class_names=severity_levels,
            threshold=0.4,
            activation="softmax"
        )
        
        return heads
    
    def _setup_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup classification thresholds for each head and class."""
        thresholds = {}
        
        # Abuse category thresholds (per category)
        thresholds["abuse_categories"] = {
            "insult/harassment": 0.4,
            "obscenity/profanity": 0.3,
            "hate/targeted group": 0.2,  # Lower threshold for hate speech
            "threat/violence": 0.2,      # Lower threshold for threats
            "sexual content": 0.4,
            "bullying/taunting": 0.4,
            "self-harm encouragement": 0.2,  # Lower threshold for self-harm
            "spam/scam": 0.6
        }
        
        # Corporate decision thresholds
        thresholds["corporate_decision"] = {
            "allow": 0.6,
            "review": 0.3,
            "block": 0.4
        }
        
        # Severity level thresholds
        thresholds["severity_levels"] = {
            "low": 0.5,
            "medium": 0.4,
            "high": 0.3,
            "critical": 0.2  # Lower threshold for critical content
        }
        
        return thresholds
    
    def _setup_class_weights(self) -> Dict[str, Dict[str, float]]:
        """Setup class weights to handle imbalanced datasets."""
        weights = {}
        
        # Abuse category weights (higher weight for rare but important categories)
        weights["abuse_categories"] = {
            "insult/harassment": 1.0,
            "obscenity/profanity": 1.0,
            "hate/targeted group": 2.0,    # Higher weight
            "threat/violence": 3.0,        # Highest weight
            "sexual content": 1.2,
            "bullying/taunting": 1.0,
            "self-harm encouragement": 2.5,  # High weight
            "spam/scam": 0.8
        }
        
        # Corporate decision weights
        weights["corporate_decision"] = {
            "allow": 1.0,
            "review": 1.5,
            "block": 2.0  # Higher weight for blocking decisions
        }
        
        # Severity level weights
        weights["severity_levels"] = {
            "low": 1.0,
            "medium": 1.2,
            "high": 1.5,
            "critical": 2.0  # Highest weight for critical content
        }
        
        return weights
    
    def process_logits(self, raw_logits: Dict[str, np.ndarray]) -> MultiHeadOutput:
        """
        Process raw logits from model into final predictions.
        
        Args:
            raw_logits: Dictionary of raw logits from each head
            
        Returns:
            MultiHeadOutput with processed predictions
        """
        # Process each head
        abuse_categories = self._process_abuse_categories(raw_logits.get("category_logits"))
        corporate_decision = self._process_corporate_decision(raw_logits.get("corporate_logits"))
        severity_levels = self._process_severity_levels(raw_logits.get("severity_logits"))
        
        # Calculate overall confidence scores
        confidence_scores = self._calculate_confidence_scores(
            abuse_categories, corporate_decision, severity_levels
        )
        
        # Create final predictions
        predictions = self._create_final_predictions(
            abuse_categories, corporate_decision, severity_levels
        )
        
        return MultiHeadOutput(
            abuse_categories=abuse_categories,
            corporate_decision=corporate_decision,
            severity_levels=severity_levels,
            confidence_scores=confidence_scores,
            predictions=predictions
        )
    
    def _process_abuse_categories(self, logits: Optional[np.ndarray]) -> Dict[str, float]:
        """Process abuse category logits (multi-label classification)."""
        if logits is None:
            return {cat: 0.0 for cat in self.heads["abuse_categories"].class_names}
        
        # Apply sigmoid for multi-label classification
        probabilities = self._sigmoid(logits)
        
        # Apply thresholds and weights
        results = {}
        head = self.heads["abuse_categories"]
        
        for i, category in enumerate(head.class_names):
            prob = float(probabilities[i])
            
            # Apply class weight
            weight = self.class_weights["abuse_categories"].get(category, 1.0)
            weighted_prob = min(1.0, prob * weight)
            
            results[category] = weighted_prob
        
        return results
    
    def _process_corporate_decision(self, logits: Optional[np.ndarray]) -> Dict[str, float]:
        """Process corporate decision logits (multi-class classification)."""
        if logits is None:
            return {dec: 1.0/3 for dec in self.heads["corporate_decision"].class_names}
        
        # Apply softmax for multi-class classification
        probabilities = self._softmax(logits)
        
        # Apply weights
        results = {}
        head = self.heads["corporate_decision"]
        
        for i, decision in enumerate(head.class_names):
            prob = float(probabilities[i])
            
            # Apply class weight
            weight = self.class_weights["corporate_decision"].get(decision, 1.0)
            weighted_prob = prob * weight
            
            results[decision] = weighted_prob
        
        # Renormalize after weighting
        total = sum(results.values())
        if total > 0:
            results = {k: v/total for k, v in results.items()}
        
        return results
    
    def _process_severity_levels(self, logits: Optional[np.ndarray]) -> Dict[str, float]:
        """Process severity level logits (ordinal classification)."""
        if logits is None:
            return {sev: 0.25 for sev in self.heads["severity_levels"].class_names}
        
        # Apply softmax for ordinal classification
        probabilities = self._softmax(logits)
        
        # Apply ordinal constraints (higher severity should have lower base probability)
        ordinal_weights = [1.0, 0.8, 0.6, 0.4]  # Decreasing weights for low->critical
        
        results = {}
        head = self.heads["severity_levels"]
        
        for i, severity in enumerate(head.class_names):
            prob = float(probabilities[i])
            
            # Apply ordinal weight and class weight
            ordinal_weight = ordinal_weights[i] if i < len(ordinal_weights) else 0.2
            class_weight = self.class_weights["severity_levels"].get(severity, 1.0)
            
            weighted_prob = prob * ordinal_weight * class_weight
            results[severity] = weighted_prob
        
        # Renormalize
        total = sum(results.values())
        if total > 0:
            results = {k: v/total for k, v in results.items()}
        
        return results
    
    def _calculate_confidence_scores(
        self, 
        abuse_categories: Dict[str, float],
        corporate_decision: Dict[str, float],
        severity_levels: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate confidence scores for each head."""
        confidence_scores = {}
        
        # Abuse categories confidence (max probability)
        if abuse_categories:
            confidence_scores["abuse_categories"] = max(abuse_categories.values())
        else:
            confidence_scores["abuse_categories"] = 0.0
        
        # Corporate decision confidence (max probability)
        if corporate_decision:
            confidence_scores["corporate_decision"] = max(corporate_decision.values())
        else:
            confidence_scores["corporate_decision"] = 0.0
        
        # Severity levels confidence (max probability)
        if severity_levels:
            confidence_scores["severity_levels"] = max(severity_levels.values())
        else:
            confidence_scores["severity_levels"] = 0.0
        
        # Overall confidence (average of head confidences)
        head_confidences = list(confidence_scores.values())
        confidence_scores["overall"] = sum(head_confidences) / len(head_confidences) if head_confidences else 0.0
        
        return confidence_scores
    
    def _create_final_predictions(
        self,
        abuse_categories: Dict[str, float],
        corporate_decision: Dict[str, float],
        severity_levels: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create final binary/categorical predictions based on thresholds."""
        predictions = {}
        
        # Abuse categories (multi-label - can have multiple positive predictions)
        predicted_categories = []
        for category, prob in abuse_categories.items():
            threshold = self.thresholds["abuse_categories"].get(category, 0.5)
            if prob >= threshold:
                predicted_categories.append(category)
        
        predictions["abuse_categories"] = predicted_categories
        
        # Corporate decision (single prediction - highest probability)
        if corporate_decision:
            predicted_decision = max(corporate_decision.items(), key=lambda x: x[1])[0]
            predictions["corporate_decision"] = predicted_decision
        else:
            predictions["corporate_decision"] = "allow"
        
        # Severity level (single prediction - highest probability above threshold)
        predicted_severity = "low"  # Default
        max_severity_prob = 0.0
        
        for severity, prob in severity_levels.items():
            threshold = self.thresholds["severity_levels"].get(severity, 0.5)
            if prob >= threshold and prob > max_severity_prob:
                predicted_severity = severity
                max_severity_prob = prob
        
        predictions["severity_level"] = predicted_severity
        
        return predictions
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip for numerical stability
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax activation function."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    def update_thresholds(self, head_name: str, new_thresholds: Dict[str, float]):
        """Update thresholds for a specific head."""
        if head_name in self.thresholds:
            self.thresholds[head_name].update(new_thresholds)
            logger.info(f"Updated thresholds for {head_name}: {new_thresholds}")
        else:
            logger.warning(f"Unknown head name: {head_name}")
    
    def update_class_weights(self, head_name: str, new_weights: Dict[str, float]):
        """Update class weights for a specific head."""
        if head_name in self.class_weights:
            self.class_weights[head_name].update(new_weights)
            logger.info(f"Updated class weights for {head_name}: {new_weights}")
        else:
            logger.warning(f"Unknown head name: {head_name}")
    
    def get_head_info(self) -> Dict[str, Any]:
        """Get information about all classification heads."""
        head_info = {}
        
        for name, head in self.heads.items():
            head_info[name] = {
                "num_classes": head.num_classes,
                "class_names": head.class_names,
                "threshold": head.threshold,
                "activation": head.activation,
                "thresholds": self.thresholds.get(name, {}),
                "class_weights": self.class_weights.get(name, {})
            }
        
        return head_info
    
    def calibrate_thresholds(self, validation_data: List[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]):
        """
        Calibrate thresholds based on validation data to optimize precision/recall.
        
        Args:
            validation_data: List of (logits, ground_truth) tuples
        """
        logger.info("Calibrating classification thresholds...")
        
        # This would implement threshold optimization based on validation performance
        # For now, just log the request
        logger.info(f"Calibration requested for {len(validation_data)} validation samples")
        
        # In production, this would:
        # 1. Process validation data through the multi-head system
        # 2. Calculate precision/recall curves for different thresholds
        # 3. Select optimal thresholds based on business requirements
        # 4. Update self.thresholds with optimized values