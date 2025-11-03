"""Feedback data collection and storage system for continuous learning."""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib

from .models import FeedbackRequest, ProblemSpan, DecisionType, SeverityLevel

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Internal feedback record with metadata."""
    feedback_id: str
    request_id: str
    timestamp: datetime
    final_label: str
    actual_categories: List[str]
    comment: Optional[str]
    language_hints: Optional[List[str]]
    corrected_spans: Optional[List[ProblemSpan]]
    tenant_id: Optional[str]
    original_text_hash: Optional[str]  # For privacy - store hash instead of text
    confidence_score: Optional[float]
    original_decision: Optional[str]
    processing_metadata: Dict[str, Any]


@dataclass
class FeedbackAnalytics:
    """Analytics data for feedback patterns."""
    total_feedback_count: int
    feedback_by_decision: Dict[str, int]
    feedback_by_category: Dict[str, int]
    feedback_by_language: Dict[str, int]
    correction_rate: float
    average_confidence_delta: float
    recent_feedback_trend: List[Dict[str, Any]]


class FeedbackStorage:
    """Secure feedback data storage with retention policies."""
    
    def __init__(self, storage_path: str = "logs/feedback", retention_days: int = 365):
        """
        Initialize feedback storage.
        
        Args:
            storage_path: Path to store feedback data
            retention_days: Number of days to retain feedback data
        """
        self.storage_path = Path(storage_path)
        self.retention_days = retention_days
        self.db_path = self.storage_path / "feedback.db"
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Feedback storage initialized at {self.storage_path}")
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback_records (
                        feedback_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        final_label TEXT NOT NULL,
                        actual_categories TEXT NOT NULL,
                        comment TEXT,
                        language_hints TEXT,
                        corrected_spans TEXT,
                        tenant_id TEXT,
                        original_text_hash TEXT,
                        confidence_score REAL,
                        original_decision TEXT,
                        processing_metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_timestamp 
                    ON feedback_records(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_tenant 
                    ON feedback_records(tenant_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_decision 
                    ON feedback_records(final_label)
                """)
                
                conn.commit()
                logger.info("Feedback database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize feedback database: {e}")
            raise
    
    async def store_feedback(self, feedback_record: FeedbackRecord) -> bool:
        """
        Store feedback record securely.
        
        Args:
            feedback_record: Feedback record to store
            
        Returns:
            Success status
        """
        try:
            # Convert to database format
            record_data = {
                'feedback_id': feedback_record.feedback_id,
                'request_id': feedback_record.request_id,
                'timestamp': feedback_record.timestamp.isoformat(),
                'final_label': feedback_record.final_label,
                'actual_categories': json.dumps(feedback_record.actual_categories),
                'comment': feedback_record.comment,
                'language_hints': json.dumps(feedback_record.language_hints) if feedback_record.language_hints else None,
                'corrected_spans': json.dumps([asdict(span) for span in feedback_record.corrected_spans]) if feedback_record.corrected_spans else None,
                'tenant_id': feedback_record.tenant_id,
                'original_text_hash': feedback_record.original_text_hash,
                'confidence_score': feedback_record.confidence_score,
                'original_decision': feedback_record.original_decision,
                'processing_metadata': json.dumps(feedback_record.processing_metadata)
            }
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO feedback_records 
                    (feedback_id, request_id, timestamp, final_label, actual_categories,
                     comment, language_hints, corrected_spans, tenant_id, original_text_hash,
                     confidence_score, original_decision, processing_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record_data['feedback_id'],
                    record_data['request_id'],
                    record_data['timestamp'],
                    record_data['final_label'],
                    record_data['actual_categories'],
                    record_data['comment'],
                    record_data['language_hints'],
                    record_data['corrected_spans'],
                    record_data['tenant_id'],
                    record_data['original_text_hash'],
                    record_data['confidence_score'],
                    record_data['original_decision'],
                    record_data['processing_metadata']
                ))
                conn.commit()
            
            logger.info(f"Stored feedback record {feedback_record.feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store feedback record: {e}")
            return False
    
    async def get_feedback_records(
        self, 
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[FeedbackRecord]:
        """
        Retrieve feedback records with filtering.
        
        Args:
            tenant_id: Filter by tenant ID
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of records to return
            
        Returns:
            List of feedback records
        """
        try:
            query = "SELECT * FROM feedback_records WHERE 1=1"
            params = []
            
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            records = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    # Parse JSON fields
                    actual_categories = json.loads(row['actual_categories'])
                    language_hints = json.loads(row['language_hints']) if row['language_hints'] else None
                    corrected_spans = None
                    if row['corrected_spans']:
                        span_data = json.loads(row['corrected_spans'])
                        corrected_spans = [ProblemSpan(**span) for span in span_data]
                    processing_metadata = json.loads(row['processing_metadata']) if row['processing_metadata'] else {}
                    
                    record = FeedbackRecord(
                        feedback_id=row['feedback_id'],
                        request_id=row['request_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        final_label=row['final_label'],
                        actual_categories=actual_categories,
                        comment=row['comment'],
                        language_hints=language_hints,
                        corrected_spans=corrected_spans,
                        tenant_id=row['tenant_id'],
                        original_text_hash=row['original_text_hash'],
                        confidence_score=row['confidence_score'],
                        original_decision=row['original_decision'],
                        processing_metadata=processing_metadata
                    )
                    records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to retrieve feedback records: {e}")
            return []
    
    async def cleanup_expired_records(self) -> int:
        """
        Clean up expired feedback records based on retention policy.
        
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM feedback_records WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired feedback records")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired records: {e}")
            return 0
    
    async def get_feedback_analytics(
        self, 
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> FeedbackAnalytics:
        """
        Generate analytics from feedback data.
        
        Args:
            tenant_id: Filter by tenant ID
            days: Number of days to analyze
            
        Returns:
            Feedback analytics
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            records = await self.get_feedback_records(
                tenant_id=tenant_id,
                start_date=start_date,
                limit=10000
            )
            
            # Calculate analytics
            total_count = len(records)
            feedback_by_decision = {}
            feedback_by_category = {}
            feedback_by_language = {}
            confidence_deltas = []
            
            for record in records:
                # Decision distribution
                decision = record.final_label
                feedback_by_decision[decision] = feedback_by_decision.get(decision, 0) + 1
                
                # Category distribution
                for category in record.actual_categories:
                    feedback_by_category[category] = feedback_by_category.get(category, 0) + 1
                
                # Language distribution
                if record.language_hints:
                    for lang in record.language_hints:
                        feedback_by_language[lang] = feedback_by_language.get(lang, 0) + 1
                
                # Confidence delta (if available)
                if record.confidence_score is not None and record.original_decision:
                    # Simple heuristic for confidence delta
                    if record.final_label != record.original_decision:
                        confidence_deltas.append(-record.confidence_score)
                    else:
                        confidence_deltas.append(record.confidence_score)
            
            # Calculate correction rate
            corrections = sum(1 for r in records if r.final_label != r.original_decision)
            correction_rate = corrections / max(total_count, 1)
            
            # Average confidence delta
            avg_confidence_delta = sum(confidence_deltas) / max(len(confidence_deltas), 1)
            
            # Recent trend (daily feedback counts)
            recent_trend = []
            for i in range(7):  # Last 7 days
                day_start = datetime.utcnow() - timedelta(days=i+1)
                day_end = datetime.utcnow() - timedelta(days=i)
                day_records = [r for r in records if day_start <= r.timestamp < day_end]
                recent_trend.append({
                    'date': day_start.date().isoformat(),
                    'count': len(day_records),
                    'corrections': sum(1 for r in day_records if r.final_label != r.original_decision)
                })
            
            return FeedbackAnalytics(
                total_feedback_count=total_count,
                feedback_by_decision=feedback_by_decision,
                feedback_by_category=feedback_by_category,
                feedback_by_language=feedback_by_language,
                correction_rate=correction_rate,
                average_confidence_delta=avg_confidence_delta,
                recent_feedback_trend=recent_trend
            )
            
        except Exception as e:
            logger.error(f"Failed to generate feedback analytics: {e}")
            return FeedbackAnalytics(
                total_feedback_count=0,
                feedback_by_decision={},
                feedback_by_category={},
                feedback_by_language={},
                correction_rate=0.0,
                average_confidence_delta=0.0,
                recent_feedback_trend=[]
            )


class FeedbackProcessor:
    """Process and validate feedback requests."""
    
    def __init__(self, storage: FeedbackStorage):
        """
        Initialize feedback processor.
        
        Args:
            storage: Feedback storage instance
        """
        self.storage = storage
        self.request_cache = {}  # Cache for request metadata
        
    async def process_feedback(
        self, 
        feedback: FeedbackRequest,
        tenant_id: Optional[str] = None,
        original_text: Optional[str] = None,
        original_decision: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> str:
        """
        Process and store feedback request.
        
        Args:
            feedback: Feedback request
            tenant_id: Tenant ID
            original_text: Original text (for hashing)
            original_decision: Original system decision
            confidence_score: Original confidence score
            
        Returns:
            Feedback ID
        """
        try:
            # Generate feedback ID
            feedback_id = str(uuid.uuid4())
            
            # Hash original text for privacy
            text_hash = None
            if original_text:
                text_hash = hashlib.sha256(original_text.encode()).hexdigest()
            
            # Create feedback record
            record = FeedbackRecord(
                feedback_id=feedback_id,
                request_id=feedback.request_id,
                timestamp=datetime.utcnow(),
                final_label=feedback.final_label,
                actual_categories=feedback.actual_categories,
                comment=feedback.comment,
                language_hints=feedback.language_hints,
                corrected_spans=feedback.corrected_spans,
                tenant_id=tenant_id,
                original_text_hash=text_hash,
                confidence_score=confidence_score,
                original_decision=original_decision,
                processing_metadata={
                    'feedback_source': 'api',
                    'processing_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Store feedback
            success = await self.storage.store_feedback(record)
            
            if success:
                logger.info(f"Processed feedback {feedback_id} for request {feedback.request_id}")
                return feedback_id
            else:
                raise Exception("Failed to store feedback")
                
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
            raise
    
    def validate_feedback(self, feedback: FeedbackRequest) -> bool:
        """
        Validate feedback request.
        
        Args:
            feedback: Feedback request to validate
            
        Returns:
            Validation status
        """
        try:
            # Check required fields
            if not feedback.request_id or not feedback.final_label:
                return False
            
            # Validate decision
            valid_decisions = {'allow', 'review', 'block'}
            if feedback.final_label not in valid_decisions:
                return False
            
            # Validate categories
            valid_categories = {
                'insult/harassment', 'obscenity/profanity', 'hate/targeted group',
                'threat/violence', 'sexual content', 'bullying/taunting',
                'self-harm encouragement', 'spam/scam'
            }
            
            for category in feedback.actual_categories:
                if category not in valid_categories:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating feedback: {e}")
            return False


class FeedbackManager:
    """Main feedback management system."""
    
    def __init__(self, storage_path: str = "logs/feedback", retention_days: int = 365):
        """
        Initialize feedback manager.
        
        Args:
            storage_path: Path for feedback storage
            retention_days: Data retention period
        """
        self.storage = FeedbackStorage(storage_path, retention_days)
        self.processor = FeedbackProcessor(self.storage)
        
        # Start cleanup task
        self._cleanup_task = None
        
    async def initialize(self):
        """Initialize feedback manager."""
        # Start periodic cleanup
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Feedback manager initialized")
    
    async def submit_feedback(
        self, 
        feedback: FeedbackRequest,
        tenant_id: Optional[str] = None,
        original_text: Optional[str] = None,
        original_decision: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> str:
        """
        Submit feedback for processing.
        
        Args:
            feedback: Feedback request
            tenant_id: Tenant ID
            original_text: Original text
            original_decision: Original decision
            confidence_score: Original confidence
            
        Returns:
            Feedback ID
        """
        # Validate feedback
        if not self.processor.validate_feedback(feedback):
            raise ValueError("Invalid feedback data")
        
        # Process feedback
        return await self.processor.process_feedback(
            feedback, tenant_id, original_text, original_decision, confidence_score
        )
    
    async def get_feedback_analytics(
        self, 
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> FeedbackAnalytics:
        """Get feedback analytics."""
        return await self.storage.get_feedback_analytics(tenant_id, days)
    
    async def get_recent_feedback(
        self, 
        tenant_id: Optional[str] = None,
        limit: int = 100
    ) -> List[FeedbackRecord]:
        """Get recent feedback records."""
        return await self.storage.get_feedback_records(
            tenant_id=tenant_id,
            limit=limit
        )
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired records."""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                await self.storage.cleanup_expired_records()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def shutdown(self):
        """Shutdown feedback manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Feedback manager shutdown complete")