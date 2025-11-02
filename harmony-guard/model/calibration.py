"""Model calibration and confidence estimation for improved reliability."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
import logging


logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of model calibration."""
    temperature: float
    calibrated_probabilities: Dict[str, float]
    confidence_score: float
    reliability_score: float
    calibration_error: float


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimation for model predictions."""
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float
    confidence_interval: Tuple[float, float]


class ModelCalibrator:
    """Handles model calibration and confidence estimation."""
    
    def __init__(self, config: Dict):
        """
        Initialize model calibrator.
        
        Args:
            config: Calibration configuration
        """
        self.config = config
        
        # Temperature scaling parameters (per head)
        self.temperatures = {
            "abuse_categories": 1.0,
            "corporate_decision": 1.0,
            "severity_levels": 1.0
        }
        
        # Calibration history for adaptive adjustment
        self.calibration_history = []
        
        # Confidence estimation parameters
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.3)
        
        # Language-specific calibration factors
        self.language_factors = {}
        
    def calibrate_probabilities(
        self, 
        raw_probabilities: Dict[str, Dict[str, float]], 
        head_name: str,
        language_codes: List[str] = None
    ) -> CalibrationResult:
        """
        Calibrate model probabilities using temperature scaling.
        
        Args:
            raw_probabilities: Raw probabilities from model
            head_name: Name of the classification head
            language_codes: Detected language codes for language-specific calibration
            
        Returns:
            CalibrationResult with calibrated probabilities
        """
        temperature = self.temperatures.get(head_name, 1.0)
        
        # Apply language-specific adjustment
        if language_codes:
            language_factor = self._get_language_calibration_factor(language_codes, head_name)
            temperature *= language_factor
        
        # Apply temperature scaling
        calibrated_probs = {}
        for category, prob_dict in raw_probabilities.items():
            if isinstance(prob_dict, dict):
                calibrated_probs[category] = self._apply_temperature_scaling(prob_dict, temperature)
            else:
                # Single probability value
                calibrated_probs[category] = self._apply_temperature_scaling_single(prob_dict, temperature)
        
        # Calculate confidence and reliability scores
        confidence_score = self._calculate_confidence_score(calibrated_probs)
        reliability_score = self._calculate_reliability_score(calibrated_probs, temperature)
        calibration_error = self._estimate_calibration_error(raw_probabilities, calibrated_probs)
        
        return CalibrationResult(
            temperature=temperature,
            calibrated_probabilities=calibrated_probs,
            confidence_score=confidence_score,
            reliability_score=reliability_score,
            calibration_error=calibration_error
        )
    
    def estimate_uncertainty(
        self, 
        probabilities: Dict[str, float],
        model_outputs: Dict[str, Any] = None
    ) -> UncertaintyEstimate:
        """
        Estimate prediction uncertainty.
        
        Args:
            probabilities: Model probabilities
            model_outputs: Additional model outputs (e.g., hidden states)
            
        Returns:
            UncertaintyEstimate with different types of uncertainty
        """
        # Epistemic uncertainty (model uncertainty) - based on prediction entropy
        epistemic = self._calculate_epistemic_uncertainty(probabilities)
        
        # Aleatoric uncertainty (data uncertainty) - based on prediction variance
        aleatoric = self._calculate_aleatoric_uncertainty(probabilities, model_outputs)
        
        # Total uncertainty
        total = np.sqrt(epistemic**2 + aleatoric**2)
        
        # Confidence interval (simplified)
        confidence_interval = self._calculate_confidence_interval(probabilities, total)
        
        return UncertaintyEstimate(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence_interval=confidence_interval
        )
    
    def update_temperature(self, head_name: str, validation_data: List[Tuple[Dict, Dict]]):
        """
        Update temperature parameter based on validation data.
        
        Args:
            head_name: Name of classification head
            validation_data: List of (predictions, ground_truth) tuples
        """
        if not validation_data:
            logger.warning(f"No validation data provided for {head_name}")
            return
        
        logger.info(f"Updating temperature for {head_name} with {len(validation_data)} samples")
        
        # Extract predictions and labels
        predictions = []
        labels = []
        
        for pred_dict, label_dict in validation_data:
            if head_name in pred_dict and head_name in label_dict:
                predictions.append(pred_dict[head_name])
                labels.append(label_dict[head_name])
        
        if not predictions:
            logger.warning(f"No matching data found for head {head_name}")
            return
        
        # Optimize temperature using negative log-likelihood
        optimal_temp = self._optimize_temperature(predictions, labels)
        
        # Update temperature with smoothing
        old_temp = self.temperatures.get(head_name, 1.0)
        smoothing_factor = 0.1  # Configurable
        new_temp = old_temp * (1 - smoothing_factor) + optimal_temp * smoothing_factor
        
        self.temperatures[head_name] = new_temp
        
        logger.info(f"Updated temperature for {head_name}: {old_temp:.3f} -> {new_temp:.3f}")
    
    def adjust_for_language(self, language_codes: List[str], head_name: str):
        """
        Adjust calibration based on detected languages.
        
        Args:
            language_codes: List of detected language codes
            head_name: Classification head name
        """
        # Language-specific calibration adjustments
        language_adjustments = {
            "hi": {"abuse_categories": 0.9, "corporate_decision": 1.1, "severity_levels": 1.0},
            "hi-Latn": {"abuse_categories": 0.85, "corporate_decision": 1.15, "severity_levels": 1.05},
            "en": {"abuse_categories": 1.0, "corporate_decision": 1.0, "severity_levels": 1.0},
            "bn": {"abuse_categories": 0.95, "corporate_decision": 1.05, "severity_levels": 1.0},
        }
        
        # Calculate weighted adjustment based on language distribution
        total_adjustment = 0.0
        total_weight = 0.0
        
        for lang_code in language_codes:
            if lang_code in language_adjustments:
                adjustment = language_adjustments[lang_code].get(head_name, 1.0)
                weight = 1.0  # Could be based on language confidence
                
                total_adjustment += adjustment * weight
                total_weight += weight
        
        if total_weight > 0:
            avg_adjustment = total_adjustment / total_weight
            key = f"{head_name}:{','.join(sorted(language_codes))}"
            self.language_factors[key] = avg_adjustment
    
    def _apply_temperature_scaling(self, probabilities: Dict[str, float], temperature: float) -> Dict[str, float]:
        """Apply temperature scaling to a dictionary of probabilities."""
        if temperature <= 0:
            temperature = 1.0
        
        # Convert probabilities to logits
        logits = {}
        for key, prob in probabilities.items():
            # Avoid log(0) by clipping
            prob_clipped = np.clip(prob, 1e-8, 1 - 1e-8)
            logits[key] = np.log(prob_clipped / (1 - prob_clipped))
        
        # Apply temperature scaling
        scaled_logits = {key: logit / temperature for key, logit in logits.items()}
        
        # Convert back to probabilities using softmax
        logit_values = np.array(list(scaled_logits.values()))
        softmax_probs = self._softmax(logit_values)
        
        calibrated = {}
        for i, key in enumerate(scaled_logits.keys()):
            calibrated[key] = float(softmax_probs[i])
        
        return calibrated
    
    def _apply_temperature_scaling_single(self, probability: float, temperature: float) -> float:
        """Apply temperature scaling to a single probability."""
        if temperature <= 0:
            temperature = 1.0
        
        # Convert to logit
        prob_clipped = np.clip(probability, 1e-8, 1 - 1e-8)
        logit = np.log(prob_clipped / (1 - prob_clipped))
        
        # Apply temperature scaling
        scaled_logit = logit / temperature
        
        # Convert back to probability
        return float(1.0 / (1.0 + np.exp(-scaled_logit)))
    
    def _calculate_confidence_score(self, probabilities: Dict[str, Any]) -> float:
        """Calculate overall confidence score from probabilities."""
        all_probs = []
        
        for value in probabilities.values():
            if isinstance(value, dict):
                all_probs.extend(value.values())
            else:
                all_probs.append(value)
        
        if not all_probs:
            return 0.0
        
        # Confidence based on maximum probability and entropy
        max_prob = max(all_probs)
        entropy = -sum(p * np.log(p + 1e-8) for p in all_probs if p > 0)
        normalized_entropy = entropy / np.log(len(all_probs)) if len(all_probs) > 1 else 0
        
        # Combine max probability and inverse entropy
        confidence = max_prob * (1 - normalized_entropy)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _calculate_reliability_score(self, probabilities: Dict[str, Any], temperature: float) -> float:
        """Calculate reliability score based on calibration quality."""
        # Reliability decreases as temperature deviates from 1.0
        temp_penalty = abs(temperature - 1.0)
        temp_reliability = np.exp(-temp_penalty)
        
        # Reliability based on probability distribution sharpness
        all_probs = []
        for value in probabilities.values():
            if isinstance(value, dict):
                all_probs.extend(value.values())
            else:
                all_probs.append(value)
        
        if not all_probs:
            return 0.0
        
        # Sharpness: higher variance indicates more decisive predictions
        prob_variance = np.var(all_probs)
        sharpness_reliability = min(1.0, prob_variance * 4)  # Scale factor
        
        # Combined reliability
        reliability = (temp_reliability + sharpness_reliability) / 2
        
        return float(np.clip(reliability, 0.0, 1.0))
    
    def _estimate_calibration_error(
        self, 
        raw_probabilities: Dict[str, Any], 
        calibrated_probabilities: Dict[str, Any]
    ) -> float:
        """Estimate calibration error (simplified)."""
        # Calculate difference between raw and calibrated probabilities
        total_diff = 0.0
        count = 0
        
        for key in raw_probabilities:
            if key in calibrated_probabilities:
                raw_val = raw_probabilities[key]
                cal_val = calibrated_probabilities[key]
                
                if isinstance(raw_val, dict) and isinstance(cal_val, dict):
                    for subkey in raw_val:
                        if subkey in cal_val:
                            total_diff += abs(raw_val[subkey] - cal_val[subkey])
                            count += 1
                else:
                    total_diff += abs(raw_val - cal_val)
                    count += 1
        
        return total_diff / count if count > 0 else 0.0
    
    def _calculate_epistemic_uncertainty(self, probabilities: Dict[str, float]) -> float:
        """Calculate epistemic (model) uncertainty using entropy."""
        all_probs = []
        
        for value in probabilities.values():
            if isinstance(value, dict):
                all_probs.extend(value.values())
            else:
                all_probs.append(value)
        
        if not all_probs:
            return 0.0
        
        # Normalize probabilities
        total = sum(all_probs)
        if total > 0:
            normalized_probs = [p / total for p in all_probs]
        else:
            normalized_probs = [1.0 / len(all_probs)] * len(all_probs)
        
        # Calculate entropy
        entropy = -sum(p * np.log(p + 1e-8) for p in normalized_probs)
        max_entropy = np.log(len(normalized_probs))
        
        # Normalize entropy to [0, 1]
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
    
    def _calculate_aleatoric_uncertainty(
        self, 
        probabilities: Dict[str, float], 
        model_outputs: Dict[str, Any] = None
    ) -> float:
        """Calculate aleatoric (data) uncertainty."""
        # Simplified aleatoric uncertainty based on prediction variance
        all_probs = []
        
        for value in probabilities.values():
            if isinstance(value, dict):
                all_probs.extend(value.values())
            else:
                all_probs.append(value)
        
        if len(all_probs) < 2:
            return 0.0
        
        # Use variance as a proxy for aleatoric uncertainty
        variance = np.var(all_probs)
        
        # Normalize to [0, 1]
        max_variance = 0.25  # Maximum variance for uniform distribution
        normalized_variance = min(variance / max_variance, 1.0)
        
        return float(normalized_variance)
    
    def _calculate_confidence_interval(
        self, 
        probabilities: Dict[str, float], 
        uncertainty: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for predictions."""
        # Get the maximum probability as the point estimate
        all_probs = []
        for value in probabilities.values():
            if isinstance(value, dict):
                all_probs.extend(value.values())
            else:
                all_probs.append(value)
        
        if not all_probs:
            return (0.0, 0.0)
        
        point_estimate = max(all_probs)
        
        # Calculate interval based on uncertainty
        margin = uncertainty * 0.5  # Scale factor
        
        lower_bound = max(0.0, point_estimate - margin)
        upper_bound = min(1.0, point_estimate + margin)
        
        return (lower_bound, upper_bound)
    
    def _get_language_calibration_factor(self, language_codes: List[str], head_name: str) -> float:
        """Get calibration factor for specific languages."""
        key = f"{head_name}:{','.join(sorted(language_codes))}"
        return self.language_factors.get(key, 1.0)
    
    def _optimize_temperature(self, predictions: List[Dict], labels: List[Dict]) -> float:
        """Optimize temperature parameter using validation data."""
        def negative_log_likelihood(temperature):
            total_nll = 0.0
            count = 0
            
            for pred, label in zip(predictions, labels):
                # Apply temperature scaling
                if isinstance(pred, dict):
                    for key in pred:
                        if key in label:
                            prob = pred[key]
                            true_label = label[key]
                            
                            # Apply temperature
                            calibrated_prob = self._apply_temperature_scaling_single(prob, temperature)
                            
                            # Calculate negative log-likelihood
                            if true_label == 1:
                                nll = -np.log(calibrated_prob + 1e-8)
                            else:
                                nll = -np.log(1 - calibrated_prob + 1e-8)
                            
                            total_nll += nll
                            count += 1
            
            return total_nll / count if count > 0 else float('inf')
        
        # Optimize temperature in range [0.1, 5.0]
        result = minimize_scalar(negative_log_likelihood, bounds=(0.1, 5.0), method='bounded')
        
        return result.x if result.success else 1.0
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        return {
            "temperatures": self.temperatures.copy(),
            "language_factors": self.language_factors.copy(),
            "calibration_history_length": len(self.calibration_history),
            "confidence_threshold": self.confidence_threshold,
            "uncertainty_threshold": self.uncertainty_threshold
        }