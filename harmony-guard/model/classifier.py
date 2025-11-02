"""Transformer-based classifier for multilingual content moderation."""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from core.models import ProcessedText, ClassifierResult, ProblemSpan, AbuseCategory, SeverityLevel
from core.interfaces import TransformerClassifierInterface
from model.monitoring import ModelPerformanceMonitor


logger = logging.getLogger(__name__)


class TransformerClassifier(TransformerClassifierInterface):
    """Multilingual transformer classifier for content moderation."""
    
    def __init__(self, config: Dict):
        """
        Initialize transformer classifier.
        
        Args:
            config: Classifier configuration
        """
        self.config = config
        self.model_name = config.get("model_name", "xlm-roberta-base")
        self.max_sequence_length = config.get("max_sequence_length", 512)
        self.batch_size = config.get("batch_size", 8)
        
        # Model components (will be loaded during initialization)
        self.tokenizer = None
        self.model = None
        self.device = "cpu"  # Default to CPU
        
        # Classification heads
        self.abuse_categories = [category.value for category in AbuseCategory]
        self.severity_levels = [level.value for level in SeverityLevel]
        self.corporate_decisions = ["allow", "review", "block"]
        
        # Model cache for performance
        self.model_cache = {}
        
        # Attention extraction for explainability
        self.extract_attention = True
        
        # Performance monitoring
        monitoring_config = config.get("monitoring", {})
        self.monitor = ModelPerformanceMonitor(monitoring_config) if monitoring_config else None
        
    async def initialize(self):
        """Initialize the transformer model and tokenizer."""
        logger.info(f"Initializing transformer classifier with model: {self.model_name}")
        
        try:
            # In a real implementation, this would load actual HuggingFace models
            # For now, we'll create a mock implementation
            await self._load_mock_model()
            
            logger.info("Transformer classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize transformer classifier: {e}")
            raise
    
    async def predict(self, processed_text: ProcessedText) -> ClassifierResult:
        """
        Generate ML-based predictions for content classification.
        
        Args:
            processed_text: Preprocessed text input
            
        Returns:
            ClassifierResult with probabilities and attention spans
        """
        start_time = time.time()
        
        try:
            # Prepare input text
            input_text = processed_text.normalized_text
            
            # Tokenize input
            tokens, input_ids, attention_mask = await self._tokenize_text(input_text)
            
            # Run inference
            predictions = await self._run_inference(input_ids, attention_mask)
            
            # Extract predictions for each head
            category_probabilities = self._extract_category_predictions(predictions)
            corporate_decision_prob = self._extract_corporate_predictions(predictions)
            severity_scores = self._extract_severity_predictions(predictions)
            
            # Extract attention spans for explainability
            attention_spans = []
            if self.extract_attention:
                attention_spans = await self._extract_attention_spans(
                    tokens, predictions, processed_text.original_text
                )
            
            result = ClassifierResult(
                category_probabilities=category_probabilities,
                corporate_decision_prob=corporate_decision_prob,
                severity_scores=severity_scores,
                attention_spans=attention_spans
            )
            
            # Record performance metrics
            if self.monitor:
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                confidence_score = max(corporate_decision_prob.values())
                
                # Combine all predictions for monitoring
                all_predictions = {
                    "category_probabilities": category_probabilities,
                    "corporate_decision_prob": corporate_decision_prob,
                    "severity_scores": severity_scores
                }
                
                self.monitor.record_prediction(
                    predictions=all_predictions,
                    confidence_score=confidence_score,
                    processing_time=processing_time,
                    language_codes=processed_text.detected_languages,
                    input_length=len(processed_text.original_text)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transformer prediction: {e}")
            # Record error in monitoring
            if self.monitor:
                processing_time = (time.time() - start_time) * 1000
                # Record with low confidence to indicate error
                self.monitor.record_prediction(
                    predictions={"error": 1.0},
                    confidence_score=0.0,
                    processing_time=processing_time,
                    language_codes=processed_text.detected_languages if hasattr(processed_text, 'detected_languages') else [],
                    input_length=len(processed_text.original_text) if hasattr(processed_text, 'original_text') else 0
                )
            
            # Return default predictions on error
            return self._get_default_predictions()
    
    async def _load_mock_model(self):
        """Load mock model for demonstration (replace with real HuggingFace model loading)."""
        # In production, this would be:
        # from transformers import AutoTokenizer, AutoModel
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModel.from_pretrained(self.model_name)
        
        # Mock implementation
        self.tokenizer = MockTokenizer()
        self.model = MockTransformerModel(
            num_categories=len(self.abuse_categories),
            num_severities=len(self.severity_levels),
            num_decisions=len(self.corporate_decisions)
        )
        
        logger.info("Mock transformer model loaded")
    
    async def _tokenize_text(self, text: str) -> Tuple[List[str], List[int], List[int]]:
        """Tokenize input text for model inference."""
        # Mock tokenization (replace with real tokenizer)
        tokens = text.split()[:self.max_sequence_length]
        
        # Mock input IDs and attention mask
        input_ids = list(range(len(tokens)))
        attention_mask = [1] * len(tokens)
        
        # Pad to max length
        while len(input_ids) < self.max_sequence_length:
            input_ids.append(0)  # Padding token
            attention_mask.append(0)
            tokens.append("[PAD]")
        
        return tokens, input_ids, attention_mask
    
    async def _run_inference(self, input_ids: List[int], attention_mask: List[int]) -> Dict[str, np.ndarray]:
        """Run model inference."""
        # Mock inference (replace with real model forward pass)
        batch_size = 1
        
        predictions = {
            "category_logits": np.random.randn(batch_size, len(self.abuse_categories)),
            "corporate_logits": np.random.randn(batch_size, len(self.corporate_decisions)),
            "severity_logits": np.random.randn(batch_size, len(self.severity_levels)),
            "attention_weights": np.random.rand(batch_size, len(input_ids))
        }
        
        return predictions
    
    def _extract_category_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract abuse category predictions."""
        logits = predictions["category_logits"][0]  # Remove batch dimension
        probabilities = self._softmax(logits)
        
        category_probs = {}
        for i, category in enumerate(self.abuse_categories):
            category_probs[category] = float(probabilities[i])
        
        return category_probs
    
    def _extract_corporate_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract corporate decision predictions."""
        logits = predictions["corporate_logits"][0]  # Remove batch dimension
        probabilities = self._softmax(logits)
        
        decision_probs = {}
        for i, decision in enumerate(self.corporate_decisions):
            decision_probs[decision] = float(probabilities[i])
        
        return decision_probs
    
    def _extract_severity_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract severity level predictions."""
        logits = predictions["severity_logits"][0]  # Remove batch dimension
        probabilities = self._softmax(logits)
        
        severity_probs = {}
        for i, severity in enumerate(self.severity_levels):
            severity_probs[severity] = float(probabilities[i])
        
        return severity_probs
    
    async def _extract_attention_spans(
        self, 
        tokens: List[str], 
        predictions: Dict[str, np.ndarray],
        original_text: str
    ) -> List[ProblemSpan]:
        """Extract attention-based spans for explainability."""
        attention_weights = predictions["attention_weights"][0]  # Remove batch dimension
        
        spans = []
        
        # Find tokens with high attention weights
        attention_threshold = 0.1  # Configurable threshold
        
        for i, (token, weight) in enumerate(zip(tokens, attention_weights)):
            if weight > attention_threshold and token != "[PAD]":
                # Create span (simplified position mapping)
                start = i * 5  # Mock position calculation
                end = start + len(token)
                
                span = ProblemSpan(
                    text=token,
                    start=start,
                    end=end,
                    category="attention_highlight",
                    confidence=float(weight),
                    rule_source=f"attention:weight:{weight:.3f}"
                )
                spans.append(span)
        
        return spans
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)
    
    def _get_default_predictions(self) -> ClassifierResult:
        """Get default predictions when model fails."""
        # Default to low confidence, neutral predictions
        default_prob = 1.0 / len(self.abuse_categories)
        
        category_probabilities = {cat: default_prob for cat in self.abuse_categories}
        corporate_decision_prob = {"allow": 0.7, "review": 0.2, "block": 0.1}
        severity_scores = {sev: 1.0 / len(self.severity_levels) for sev in self.severity_levels}
        
        return ClassifierResult(
            category_probabilities=category_probabilities,
            corporate_decision_prob=corporate_decision_prob,
            severity_scores=severity_scores,
            attention_spans=[]
        )
    
    async def batch_predict(self, texts: List[ProcessedText]) -> List[ClassifierResult]:
        """Batch prediction for multiple texts."""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Process batch (simplified - in production, use proper batching)
            batch_results = []
            for text in batch:
                result = await self.predict(text)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "max_sequence_length": self.max_sequence_length,
            "batch_size": self.batch_size,
            "device": self.device,
            "num_categories": len(self.abuse_categories),
            "num_severities": len(self.severity_levels),
            "num_decisions": len(self.corporate_decisions),
            "categories": self.abuse_categories,
            "severities": self.severity_levels,
            "decisions": self.corporate_decisions
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics from monitoring."""
        if not self.monitor:
            return {"error": "Monitoring not enabled"}
        
        return self.monitor.get_current_metrics()
    
    def get_drift_alerts(self, hours: int = 24) -> List[Any]:
        """Get drift alerts from the last N hours."""
        if not self.monitor:
            return []
        
        return self.monitor.get_drift_alerts(hours)
    
    def get_performance_summary(self, windows: int = 10) -> Dict[str, Any]:
        """Get performance summary over recent windows."""
        if not self.monitor:
            return {"error": "Monitoring not enabled"}
        
        return self.monitor.get_performance_summary(windows)
    
    def run_drift_tests(self) -> Dict[str, Any]:
        """Run statistical tests for drift detection."""
        if not self.monitor:
            return {"error": "Monitoring not enabled"}
        
        return self.monitor.run_statistical_tests()
    
    async def shutdown(self):
        """Cleanup model resources."""
        logger.info("Shutting down transformer classifier...")
        
        # Clear model cache
        self.model_cache.clear()
        
        # In production, this would free GPU memory, etc.
        self.model = None
        self.tokenizer = None
        
        logger.info("Transformer classifier shutdown complete")


class MockTokenizer:
    """Mock tokenizer for demonstration purposes."""
    
    def __init__(self):
        self.vocab_size = 50000
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word-based tokenization."""
        return text.split()
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        # Mock encoding
        return [hash(token) % self.vocab_size for token in tokens[:max_length]]


class MockTransformerModel:
    """Mock transformer model for demonstration purposes."""
    
    def __init__(self, num_categories: int, num_severities: int, num_decisions: int):
        self.num_categories = num_categories
        self.num_severities = num_severities
        self.num_decisions = num_decisions
        self.hidden_size = 768
    
    def forward(self, input_ids: List[int], attention_mask: List[int]) -> Dict[str, np.ndarray]:
        """Mock forward pass."""
        batch_size = 1
        seq_length = len(input_ids)
        
        # Mock outputs
        return {
            "category_logits": np.random.randn(batch_size, self.num_categories),
            "corporate_logits": np.random.randn(batch_size, self.num_decisions),
            "severity_logits": np.random.randn(batch_size, self.num_severities),
            "attention_weights": np.random.rand(batch_size, seq_length),
            "hidden_states": np.random.randn(batch_size, seq_length, self.hidden_size)
        }