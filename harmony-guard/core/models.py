"""Core data models for Harmony Guard system."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Literal
from enum import Enum


class DecisionType(str, Enum):
    """Corporate appropriateness decision types."""
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class SeverityLevel(str, Enum):
    """Severity levels for detected content."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AbuseCategory(str, Enum):
    """Abuse categories for content classification."""
    INSULT_HARASSMENT = "insult/harassment"
    OBSCENITY_PROFANITY = "obscenity/profanity"
    HATE_TARGETED = "hate/targeted group"
    THREAT_VIOLENCE = "threat/violence"
    SEXUAL_CONTENT = "sexual content"
    BULLYING_TAUNTING = "bullying/taunting"
    SELF_HARM = "self-harm encouragement"
    SPAM_SCAM = "spam/scam"


@dataclass
class LanguageDetection:
    """Language detection result."""
    code: str
    confidence: float
    percentage: float


@dataclass
class ProblemSpan:
    """Represents a problematic text span."""
    text: str
    start: int
    end: int
    category: str
    confidence: float
    rule_source: str


@dataclass
class ProcessedText:
    """Preprocessed text with normalization and language detection."""
    original_text: str
    normalized_text: str
    detected_languages: List[LanguageDetection]
    tokens: List[str]
    transliterations: Dict[str, str]
    obfuscation_map: Dict[str, str]
    pii_masked: bool = False


@dataclass
class LPEResult:
    """Result from Lexicon & Pattern Engine."""
    matched_spans: List[ProblemSpan]
    categories: List[str]
    confidence_scores: Dict[str, float]
    rule_traces: List[str]


@dataclass
class ClassifierResult:
    """Result from Transformer Classifier."""
    category_probabilities: Dict[str, float]
    corporate_decision_prob: Dict[str, float]
    severity_scores: Dict[str, float]
    attention_spans: List[ProblemSpan]


@dataclass
class ContextResult:
    """Result from Intent/Context Layer."""
    context_modifiers: Dict[str, float]
    safe_context_detected: Dict[str, bool]
    recommended_action: DecisionType


@dataclass
class AggregatedResult:
    """Final aggregated result from ensemble."""
    final_decision: DecisionType
    confidence_score: float
    category_scores: Dict[str, float]
    severity_level: SeverityLevel
    explanation_traces: List[str]
    consolidated_spans: List[ProblemSpan]


@dataclass
class AnalysisRequest:
    """API request for content analysis."""
    text: str
    tenant_id: Optional[str] = None
    include_details: bool = False
    language_hints: Optional[List[str]] = None


@dataclass
class AnalysisResponse:
    """API response for content analysis."""
    corporate_allowed: DecisionType
    confidence: float
    severity: SeverityLevel
    categories: List[str]
    languages: List[Dict[str, Union[str, float]]]
    spans: Optional[List[ProblemSpan]] = None
    explanations: Optional[List[str]] = None
    normalized_preview: Optional[str] = None
    policy_trace: Optional[List[str]] = None


@dataclass
class FeedbackRequest:
    """API request for feedback submission."""
    request_id: str
    final_label: str
    actual_categories: List[str]
    comment: Optional[str] = None
    language_hints: Optional[List[str]] = None
    corrected_spans: Optional[List[ProblemSpan]] = None