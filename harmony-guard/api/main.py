"""FastAPI application for Harmony Guard content moderation service."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid
import time
import logging
from contextlib import asynccontextmanager

from ..core.models import AnalysisRequest, AnalysisResponse, FeedbackRequest
from ..configs.manager import ConfigurationManager
from .service import HarmonyGuardService


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
harmony_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global harmony_service
    
    # Startup
    logger.info("Starting Harmony Guard service...")
    config_manager = ConfigurationManager()
    harmony_service = HarmonyGuardService(config_manager)
    await harmony_service.initialize()
    logger.info("Harmony Guard service initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Harmony Guard service...")
    if harmony_service:
        await harmony_service.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Harmony Guard",
    description="Content moderation service for corporate communications",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class AnalyzeRequest(BaseModel):
    text: str
    tenant_id: Optional[str] = None
    include_details: bool = False
    language_hints: Optional[List[str]] = None


class FeedbackSubmission(BaseModel):
    request_id: str
    final_label: str
    actual_categories: List[str]
    comment: Optional[str] = None
    language_hints: Optional[List[str]] = None


def get_service() -> HarmonyGuardService:
    """Dependency to get the service instance."""
    if harmony_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return harmony_service


@app.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze_content(
    request: AnalyzeRequest,
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Analyze text content for corporate appropriateness.
    
    Returns decision (allow/review/block), confidence, severity, categories,
    and optional detailed analysis including spans and explanations.
    """
    try:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Convert to internal request model
        analysis_request = AnalysisRequest(
            text=request.text,
            tenant_id=request.tenant_id,
            include_details=request.include_details,
            language_hints=request.language_hints
        )
        
        # Process the request
        result = await service.analyze(analysis_request, request_id)
        
        # Log request metrics
        processing_time = time.time() - start_time
        logger.info(
            f"Request {request_id} processed in {processing_time:.3f}s - "
            f"Decision: {result.corporate_allowed}, Confidence: {result.confidence:.3f}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing analysis request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/feedback")
async def submit_feedback(
    feedback: FeedbackSubmission,
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Submit feedback for continuous learning and model improvement.
    """
    try:
        # Convert to internal feedback model
        feedback_request = FeedbackRequest(
            request_id=feedback.request_id,
            final_label=feedback.final_label,
            actual_categories=feedback.actual_categories,
            comment=feedback.comment,
            language_hints=feedback.language_hints
        )
        
        success = await service.submit_feedback(feedback_request)
        
        if success:
            return {"status": "success", "message": "Feedback submitted successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to submit feedback")
            
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "service": "harmony-guard"
    }


@app.get("/v1/readiness")
async def readiness_check(service: HarmonyGuardService = Depends(get_service)):
    """Readiness check endpoint."""
    is_ready = await service.is_ready()
    
    if is_ready:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/v1/metrics")
async def get_metrics(service: HarmonyGuardService = Depends(get_service)):
    """Get service metrics."""
    metrics = await service.get_metrics()
    return metrics


@app.get("/v1/config")
async def get_config(
    tenant_id: Optional[str] = None,
    service: HarmonyGuardService = Depends(get_service)
):
    """Get non-secret configuration."""
    config = await service.get_public_config(tenant_id)
    return config


@app.get("/v1/monitoring/performance")
async def get_performance_metrics(service: HarmonyGuardService = Depends(get_service)):
    """
    Get detailed performance metrics including model monitoring data.
    
    Returns current performance metrics, prediction distributions,
    confidence scores, and processing times.
    """
    try:
        metrics = await service.get_performance_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")


@app.get("/v1/monitoring/drift")
async def get_drift_alerts(
    hours: int = 24,
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Get drift alerts from model monitoring.
    
    Args:
        hours: Number of hours to look back for alerts (default: 24)
    
    Returns drift alerts with severity levels and descriptions.
    """
    try:
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        alerts = await service.get_drift_alerts(hours)
        return alerts
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting drift alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve drift alerts")


@app.get("/v1/monitoring/summary")
async def get_performance_summary(
    windows: int = 10,
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Get performance summary over recent time windows.
    
    Args:
        windows: Number of recent windows to analyze (default: 10)
    
    Returns aggregated performance metrics and trends.
    """
    try:
        if windows < 1 or windows > 100:
            raise HTTPException(status_code=400, detail="Windows must be between 1 and 100")
        
        summary = await service.get_performance_summary(windows)
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance summary")


@app.get("/v1/monitoring/drift-analysis")
async def run_drift_analysis(service: HarmonyGuardService = Depends(get_service)):
    """
    Run comprehensive drift analysis using statistical tests.
    
    Returns statistical test results, recent alerts, and recommendations
    for addressing any detected drift.
    """
    try:
        analysis = await service.run_drift_analysis()
        return analysis
    except Exception as e:
        logger.error(f"Error running drift analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to run drift analysis")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)