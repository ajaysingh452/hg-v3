"""Core interfaces for Harmony Guard components."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from .models import (
    ProcessedText, LPEResult, ClassifierResult, 
    ContextResult, AggregatedResult
)


class TextPreprocessorInterface(ABC):
    """Interface for text preprocessing pipeline."""
    
    @abstractmethod
    def process(self, text: str, language_hints: list = None) -> ProcessedText:
        """
        Process raw text through normalization, language detection, and obfuscation handling.
        
        Args:
            text: Raw input text
            language_hints: Optional language hints for better processing
            
        Returns:
            ProcessedText with normalized content and metadata
        """
        pass


class LexiconPatternEngineInterface(ABC):
    """Interface for Lexicon & Pattern Engine."""
    
    @abstractmethod
    def analyze(self, processed_text: ProcessedText) -> LPEResult:
        """
        Analyze text using lexicons and pattern matching.
        
        Args:
            processed_text: Preprocessed text input
            
        Returns:
            LPEResult with matched spans and rule traces
        """
        pass


class TransformerClassifierInterface(ABC):
    """Interface for Transformer-based classifier."""
    
    @abstractmethod
    def predict(self, processed_text: ProcessedText) -> ClassifierResult:
        """
        Generate ML-based predictions for content classification.
        
        Args:
            processed_text: Preprocessed text input
            
        Returns:
            ClassifierResult with probabilities and attention spans
        """
        pass


class IntentContextLayerInterface(ABC):
    """Interface for Intent/Context analysis."""
    
    @abstractmethod
    def analyze_context(
        self, 
        processed_text: ProcessedText,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult
    ) -> ContextResult:
        """
        Analyze contextual factors to adjust confidence and decisions.
        
        Args:
            processed_text: Preprocessed text input
            lpe_result: Result from lexicon engine
            classifier_result: Result from ML classifier
            
        Returns:
            ContextResult with context modifiers and recommendations
        """
        pass


class EnsembleAggregatorInterface(ABC):
    """Interface for ensemble result aggregation."""
    
    @abstractmethod
    def aggregate(
        self,
        lpe_result: LPEResult,
        classifier_result: ClassifierResult,
        context_result: ContextResult
    ) -> AggregatedResult:
        """
        Combine and calibrate outputs from all ensemble components.
        
        Args:
            lpe_result: Result from lexicon engine
            classifier_result: Result from ML classifier
            context_result: Result from context analysis
            
        Returns:
            AggregatedResult with final decision and explanations
        """
        pass


class PolicyEngineInterface(ABC):
    """Interface for corporate policy application."""
    
    @abstractmethod
    def apply_policy(
        self, 
        aggregated_result: AggregatedResult,
        tenant_id: str = None
    ) -> AggregatedResult:
        """
        Apply organization-specific policies to modify final decisions.
        
        Args:
            aggregated_result: Result from ensemble aggregation
            tenant_id: Optional tenant identifier for policy selection
            
        Returns:
            Modified AggregatedResult with policy rules applied
        """
        pass


class ConfigurationManagerInterface(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def load_config(self, config_type: str, tenant_id: str = None) -> Dict[str, Any]:
        """
        Load configuration for specified type and tenant.
        
        Args:
            config_type: Type of configuration (ensemble, policy, preprocessing)
            tenant_id: Optional tenant identifier
            
        Returns:
            Configuration dictionary
        """
        pass
    
    @abstractmethod
    def update_config(
        self, 
        config_type: str, 
        config_data: Dict[str, Any],
        tenant_id: str = None
    ) -> bool:
        """
        Update configuration for specified type and tenant.
        
        Args:
            config_type: Type of configuration
            config_data: New configuration data
            tenant_id: Optional tenant identifier
            
        Returns:
            Success status
        """
        pass