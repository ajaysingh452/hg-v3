"""Active learning pipeline for continuous model improvement."""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from .models import AnalysisRequest, AnalysisResponse, FeedbackRequest, DecisionType, SeverityLevel
from .feedback import FeedbackManager, FeedbackRecord

logger = logging.getLogger(__name__)


class ReviewPriority(str, Enum):
    """Priority levels for human review."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewStatus(str, Enum):
    """Status of review items."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class ReviewItem:
    """Item queued for human review."""
    review_id: str
    request_id: str
    original_text_hash: str  # For privacy
    predicted_decision: str
    confidence_score: float
    categories: List[str]
    languages: List[str]
    priority: ReviewPriority
    status: ReviewStatus
    created_at: datetime
    assigned_to: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_feedback: Optional[str] = None
    tenant_id: Optional[str] = None
    uncertainty_score: Optional[float] = None
    disagreement_score: Optional[float] = None


@dataclass
class LowConfidencePrediction:
    """Prediction identified as low confidence."""
    request_id: str
    text_hash: str
    decision: str
    confidence: float
    categories: List[str]
    languages: List[str]
    uncertainty_metrics: Dict[str, float]
    timestamp: datetime
    tenant_id: Optional[str] = None


class ConfidenceAnalyzer:
    """Analyze predictions to identify low-confidence cases."""
    
    def __init__(self, confidence_threshold: float = 0.7, uncertainty_threshold: float = 0.3):
        """
        Initialize confidence analyzer.
        
        Args:
            confidence_threshold: Threshold below which predictions are considered low confidence
            uncertainty_threshold: Threshold above which predictions are considered uncertain
        """
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
    def analyze_prediction(
        self, 
        response: AnalysisResponse,
        request_id: str,
        text_hash: str,
        tenant_id: Optional[str] = None
    ) -> Optional[LowConfidencePrediction]:
        """
        Analyze a prediction to determine if it needs human review.
        
        Args:
            response: Analysis response from the system
            request_id: Request identifier
            text_hash: Hash of original text
            tenant_id: Tenant identifier
            
        Returns:
            LowConfidencePrediction if review is needed, None otherwise
        """
        try:
            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_uncertainty_metrics(response)
            
            # Check if prediction meets criteria for review
            needs_review = self._needs_review(response, uncertainty_metrics)
            
            if needs_review:
                return LowConfidencePrediction(
                    request_id=request_id,
                    text_hash=text_hash,
                    decision=response.corporate_allowed.value,
                    confidence=response.confidence,
                    categories=response.categories,
                    languages=[lang["code"] for lang in response.languages],
                    uncertainty_metrics=uncertainty_metrics,
                    timestamp=datetime.utcnow(),
                    tenant_id=tenant_id
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing prediction confidence: {e}")
            return None
    
    def _calculate_uncertainty_metrics(self, response: AnalysisResponse) -> Dict[str, float]:
        """Calculate various uncertainty metrics."""
        metrics = {}
        
        # Basic confidence uncertainty
        metrics["confidence_uncertainty"] = 1.0 - response.confidence
        
        # Decision boundary uncertainty (how close to decision thresholds)
        if response.corporate_allowed == DecisionType.REVIEW:
            metrics["boundary_uncertainty"] = 0.8  # Review decisions are inherently uncertain
        elif response.confidence < 0.6:
            metrics["boundary_uncertainty"] = 0.7
        elif response.confidence > 0.9:
            metrics["boundary_uncertainty"] = 0.1
        else:
            metrics["boundary_uncertainty"] = 0.5
        
        # Category uncertainty (multiple categories or unclear categorization)
        if len(response.categories) == 0:
            metrics["category_uncertainty"] = 0.9  # No clear category
        elif len(response.categories) > 2:
            metrics["category_uncertainty"] = 0.6  # Multiple categories
        else:
            metrics["category_uncertainty"] = 0.2
        
        # Language uncertainty (code-mixed or uncertain language detection)
        if len(response.languages) > 2:
            metrics["language_uncertainty"] = 0.7
        elif any(lang["pct"] < 60 for lang in response.languages):
            metrics["language_uncertainty"] = 0.5
        else:
            metrics["language_uncertainty"] = 0.1
        
        # Overall uncertainty score
        metrics["overall_uncertainty"] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _needs_review(self, response: AnalysisResponse, uncertainty_metrics: Dict[str, float]) -> bool:
        """Determine if prediction needs human review."""
        # Low confidence predictions
        if response.confidence < self.confidence_threshold:
            return True
        
        # High uncertainty predictions
        if uncertainty_metrics.get("overall_uncertainty", 0) > self.uncertainty_threshold:
            return True
        
        # Edge cases that should be reviewed
        if (response.corporate_allowed == DecisionType.BLOCK and 
            response.confidence < 0.8):
            return True
        
        if (response.corporate_allowed == DecisionType.REVIEW and 
            len(response.categories) == 0):
            return True
        
        return False


class ReviewQueue:
    """Manage human review queue."""
    
    def __init__(self, storage_path: str = "logs/review_queue"):
        """
        Initialize review queue.
        
        Args:
            storage_path: Path to store review queue data
        """
        self.storage_path = Path(storage_path)
        self.db_path = self.storage_path / "review_queue.db"
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for review queue."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS review_items (
                        review_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        original_text_hash TEXT NOT NULL,
                        predicted_decision TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        categories TEXT NOT NULL,
                        languages TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        assigned_to TEXT,
                        reviewed_at TEXT,
                        review_feedback TEXT,
                        tenant_id TEXT,
                        uncertainty_score REAL,
                        disagreement_score REAL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_review_status 
                    ON review_items(status)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_review_priority 
                    ON review_items(priority, created_at)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_review_tenant 
                    ON review_items(tenant_id)
                """)
                
                conn.commit()
                logger.info("Review queue database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize review queue database: {e}")
            raise
    
    async def add_review_item(self, prediction: LowConfidencePrediction) -> str:
        """
        Add item to review queue.
        
        Args:
            prediction: Low confidence prediction to review
            
        Returns:
            Review ID
        """
        try:
            review_id = str(uuid.uuid4())
            
            # Determine priority based on uncertainty and decision
            priority = self._calculate_priority(prediction)
            
            review_item = ReviewItem(
                review_id=review_id,
                request_id=prediction.request_id,
                original_text_hash=prediction.text_hash,
                predicted_decision=prediction.decision,
                confidence_score=prediction.confidence,
                categories=prediction.categories,
                languages=prediction.languages,
                priority=priority,
                status=ReviewStatus.PENDING,
                created_at=prediction.timestamp,
                tenant_id=prediction.tenant_id,
                uncertainty_score=prediction.uncertainty_metrics.get("overall_uncertainty")
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO review_items 
                    (review_id, request_id, original_text_hash, predicted_decision,
                     confidence_score, categories, languages, priority, status,
                     created_at, tenant_id, uncertainty_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    review_item.review_id,
                    review_item.request_id,
                    review_item.original_text_hash,
                    review_item.predicted_decision,
                    review_item.confidence_score,
                    json.dumps(review_item.categories),
                    json.dumps(review_item.languages),
                    review_item.priority.value,
                    review_item.status.value,
                    review_item.created_at.isoformat(),
                    review_item.tenant_id,
                    review_item.uncertainty_score
                ))
                conn.commit()
            
            logger.info(f"Added review item {review_id} with priority {priority}")
            return review_id
            
        except Exception as e:
            logger.error(f"Failed to add review item: {e}")
            raise
    
    async def get_pending_reviews(
        self, 
        limit: int = 50,
        priority_filter: Optional[ReviewPriority] = None,
        tenant_id: Optional[str] = None
    ) -> List[ReviewItem]:
        """
        Get pending review items.
        
        Args:
            limit: Maximum number of items to return
            priority_filter: Filter by priority level
            tenant_id: Filter by tenant ID
            
        Returns:
            List of pending review items
        """
        try:
            query = """
                SELECT * FROM review_items 
                WHERE status = 'pending'
            """
            params = []
            
            if priority_filter:
                query += " AND priority = ?"
                params.append(priority_filter.value)
            
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            
            query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
            params.append(limit)
            
            items = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    item = ReviewItem(
                        review_id=row['review_id'],
                        request_id=row['request_id'],
                        original_text_hash=row['original_text_hash'],
                        predicted_decision=row['predicted_decision'],
                        confidence_score=row['confidence_score'],
                        categories=json.loads(row['categories']),
                        languages=json.loads(row['languages']),
                        priority=ReviewPriority(row['priority']),
                        status=ReviewStatus(row['status']),
                        created_at=datetime.fromisoformat(row['created_at']),
                        assigned_to=row['assigned_to'],
                        reviewed_at=datetime.fromisoformat(row['reviewed_at']) if row['reviewed_at'] else None,
                        review_feedback=row['review_feedback'],
                        tenant_id=row['tenant_id'],
                        uncertainty_score=row['uncertainty_score'],
                        disagreement_score=row['disagreement_score']
                    )
                    items.append(item)
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to get pending reviews: {e}")
            return []
    
    async def update_review_status(
        self, 
        review_id: str, 
        status: ReviewStatus,
        assigned_to: Optional[str] = None,
        review_feedback: Optional[str] = None
    ) -> bool:
        """
        Update review item status.
        
        Args:
            review_id: Review item ID
            status: New status
            assigned_to: Reviewer assignment
            review_feedback: Review feedback
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE review_items 
                    SET status = ?, assigned_to = ?, review_feedback = ?,
                        reviewed_at = ?
                    WHERE review_id = ?
                """, (
                    status.value,
                    assigned_to,
                    review_feedback,
                    datetime.utcnow().isoformat() if status == ReviewStatus.COMPLETED else None,
                    review_id
                ))
                conn.commit()
            
            logger.info(f"Updated review item {review_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update review status: {e}")
            return False
    
    async def get_review_statistics(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get review queue statistics."""
        try:
            query_base = "SELECT status, priority, COUNT(*) as count FROM review_items"
            params = []
            
            if tenant_id:
                query_base += " WHERE tenant_id = ?"
                params.append(tenant_id)
            
            stats = {
                "by_status": {},
                "by_priority": {},
                "total_pending": 0,
                "total_completed": 0,
                "average_review_time": 0.0
            }
            
            with sqlite3.connect(self.db_path) as conn:
                # Status distribution
                cursor = conn.execute(query_base + " GROUP BY status", params)
                for row in cursor.fetchall():
                    stats["by_status"][row[0]] = row[2]
                    if row[0] == "pending":
                        stats["total_pending"] = row[2]
                    elif row[0] == "completed":
                        stats["total_completed"] = row[2]
                
                # Priority distribution
                cursor = conn.execute(query_base + " GROUP BY priority", params)
                for row in cursor.fetchall():
                    stats["by_priority"][row[1]] = row[2]
                
                # Average review time
                time_query = """
                    SELECT AVG(
                        (julianday(reviewed_at) - julianday(created_at)) * 24 * 60
                    ) as avg_minutes
                    FROM review_items 
                    WHERE status = 'completed' AND reviewed_at IS NOT NULL
                """
                if tenant_id:
                    time_query += " AND tenant_id = ?"
                
                cursor = conn.execute(time_query, params if tenant_id else [])
                result = cursor.fetchone()
                if result and result[0]:
                    stats["average_review_time"] = float(result[0])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get review statistics: {e}")
            return {}
    
    def _calculate_priority(self, prediction: LowConfidencePrediction) -> ReviewPriority:
        """Calculate review priority based on prediction characteristics."""
        # High priority for block decisions with low confidence
        if (prediction.decision == "block" and 
            prediction.confidence < 0.6):
            return ReviewPriority.CRITICAL
        
        # High priority for high uncertainty
        overall_uncertainty = prediction.uncertainty_metrics.get("overall_uncertainty", 0)
        if overall_uncertainty > 0.7:
            return ReviewPriority.HIGH
        
        # Medium priority for moderate uncertainty or review decisions
        if (overall_uncertainty > 0.4 or 
            prediction.decision == "review"):
            return ReviewPriority.MEDIUM
        
        return ReviewPriority.LOW


class ActiveLearningPipeline:
    """Main active learning pipeline coordinator."""
    
    def __init__(
        self, 
        feedback_manager: FeedbackManager,
        confidence_threshold: float = 0.7,
        review_queue_path: str = "logs/review_queue"
    ):
        """
        Initialize active learning pipeline.
        
        Args:
            feedback_manager: Feedback manager instance
            confidence_threshold: Confidence threshold for review
            review_queue_path: Path for review queue storage
        """
        self.feedback_manager = feedback_manager
        self.confidence_analyzer = ConfidenceAnalyzer(confidence_threshold)
        self.review_queue = ReviewQueue(review_queue_path)
        
        # Pipeline metrics
        self.metrics = {
            "predictions_analyzed": 0,
            "items_queued_for_review": 0,
            "reviews_completed": 0,
            "feedback_integrated": 0
        }
        
    async def analyze_prediction(
        self, 
        response: AnalysisResponse,
        request_id: str,
        original_text: str,
        tenant_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Analyze prediction and queue for review if needed.
        
        Args:
            response: Analysis response
            request_id: Request identifier
            original_text: Original text (for hashing)
            tenant_id: Tenant identifier
            
        Returns:
            Review ID if queued, None otherwise
        """
        try:
            self.metrics["predictions_analyzed"] += 1
            
            # Hash text for privacy
            import hashlib
            text_hash = hashlib.sha256(original_text.encode()).hexdigest()
            
            # Analyze prediction confidence
            low_confidence = self.confidence_analyzer.analyze_prediction(
                response, request_id, text_hash, tenant_id
            )
            
            if low_confidence:
                # Add to review queue
                review_id = await self.review_queue.add_review_item(low_confidence)
                self.metrics["items_queued_for_review"] += 1
                
                logger.info(f"Queued prediction {request_id} for review as {review_id}")
                return review_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing prediction for active learning: {e}")
            return None
    
    async def process_review_feedback(
        self, 
        review_id: str,
        corrected_decision: str,
        corrected_categories: List[str],
        reviewer_comment: Optional[str] = None
    ) -> bool:
        """
        Process feedback from human review.
        
        Args:
            review_id: Review item ID
            corrected_decision: Corrected decision
            corrected_categories: Corrected categories
            reviewer_comment: Optional reviewer comment
            
        Returns:
            Success status
        """
        try:
            # Get review item
            review_items = await self.review_queue.get_pending_reviews(limit=1000)
            review_item = next((item for item in review_items if item.review_id == review_id), None)
            
            if not review_item:
                logger.error(f"Review item {review_id} not found")
                return False
            
            # Create feedback request
            feedback_request = FeedbackRequest(
                request_id=review_item.request_id,
                final_label=corrected_decision,
                actual_categories=corrected_categories,
                comment=reviewer_comment
            )
            
            # Submit feedback
            feedback_id = await self.feedback_manager.submit_feedback(
                feedback_request,
                tenant_id=review_item.tenant_id
            )
            
            # Update review status
            await self.review_queue.update_review_status(
                review_id,
                ReviewStatus.COMPLETED,
                review_feedback=f"Feedback submitted as {feedback_id}"
            )
            
            self.metrics["reviews_completed"] += 1
            self.metrics["feedback_integrated"] += 1
            
            logger.info(f"Processed review feedback {review_id} -> feedback {feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing review feedback: {e}")
            return False
    
    async def get_review_queue_status(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get review queue status and statistics."""
        try:
            stats = await self.review_queue.get_review_statistics(tenant_id)
            pending_items = await self.review_queue.get_pending_reviews(
                limit=10, tenant_id=tenant_id
            )
            
            return {
                "queue_statistics": stats,
                "pipeline_metrics": self.metrics.copy(),
                "pending_items_preview": [
                    {
                        "review_id": item.review_id,
                        "predicted_decision": item.predicted_decision,
                        "confidence_score": item.confidence_score,
                        "priority": item.priority.value,
                        "created_at": item.created_at.isoformat(),
                        "categories": item.categories,
                        "languages": item.languages
                    }
                    for item in pending_items[:5]  # Show top 5
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting review queue status: {e}")
            return {"error": str(e)}
    
    async def get_pending_reviews(
        self, 
        limit: int = 50,
        priority_filter: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pending review items for human reviewers."""
        try:
            priority_enum = ReviewPriority(priority_filter) if priority_filter else None
            items = await self.review_queue.get_pending_reviews(
                limit=limit,
                priority_filter=priority_enum,
                tenant_id=tenant_id
            )
            
            return [
                {
                    "review_id": item.review_id,
                    "request_id": item.request_id,
                    "predicted_decision": item.predicted_decision,
                    "confidence_score": item.confidence_score,
                    "categories": item.categories,
                    "languages": item.languages,
                    "priority": item.priority.value,
                    "created_at": item.created_at.isoformat(),
                    "uncertainty_score": item.uncertainty_score,
                    "tenant_id": item.tenant_id
                }
                for item in items
            ]
            
        except Exception as e:
            logger.error(f"Error getting pending reviews: {e}")
            return []