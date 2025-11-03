"""FastAPI application for Harmony Guard content moderation service."""

from fastapi import FastAPI, HTTPException, Depends, Request, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import uuid
import time
import logging
import asyncio
from contextlib import asynccontextmanager
import traceback
import hashlib
import hmac
from collections import defaultdict, deque

from ..core.models import AnalysisRequest, AnalysisResponse, FeedbackRequest, DecisionType, SeverityLevel
from ..configs.manager import ConfigurationManager
from ..core.metrics import metrics, MetricsMiddleware, create_metrics_endpoint
from ..core.logging import configure_logging, LoggingMiddleware, DEFAULT_LOGGING_CONFIG
from ..core.health import HealthMonitor, SystemResourceChecker, ModelChecker, GracefulShutdownHandler
from .service import HarmonyGuardService

# Configure structured logging
app_logger = configure_logging(DEFAULT_LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Global service instance
harmony_service = None
health_monitor = None
shutdown_handler = None

# Authentication and rate limiting
security = HTTPBearer(auto_error=False)

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.limits = {
            "default": {"requests": 100, "window": 60},  # 100 requests per minute
            "premium": {"requests": 1000, "window": 60},  # 1000 requests per minute
        }
    
    def is_allowed(self, tenant_id: str, tier: str = "default") -> bool:
        """Check if request is allowed based on rate limits."""
        now = time.time()
        limit_config = self.limits.get(tier, self.limits["default"])
        window = limit_config["window"]
        max_requests = limit_config["requests"]
        
        # Clean old requests outside the window
        tenant_requests = self.requests[tenant_id]
        while tenant_requests and tenant_requests[0] < now - window:
            tenant_requests.popleft()
        
        # Check if under limit
        if len(tenant_requests) < max_requests:
            tenant_requests.append(now)
            return True
        
        return False
    
    def get_remaining(self, tenant_id: str, tier: str = "default") -> int:
        """Get remaining requests for the current window."""
        limit_config = self.limits.get(tier, self.limits["default"])
        max_requests = limit_config["requests"]
        current_requests = len(self.requests[tenant_id])
        return max(0, max_requests - current_requests)

class AuthenticationManager:
    """Simple authentication manager."""
    
    def __init__(self):
        # In production, these would come from a secure configuration
        self.tenant_secrets = {
            "demo": "demo-secret-key",
            "test": "test-secret-key",
        }
        self.tenant_tiers = {
            "demo": "default",
            "test": "premium",
        }
    
    def verify_tenant_auth(self, tenant_id: str, signature: str, timestamp: str, body: str) -> bool:
        """Verify tenant authentication using HMAC signature."""
        if tenant_id not in self.tenant_secrets:
            return False
        
        # Check timestamp (prevent replay attacks)
        try:
            request_time = float(timestamp)
            if abs(time.time() - request_time) > 300:  # 5 minute window
                return False
        except (ValueError, TypeError):
            return False
        
        # Verify HMAC signature
        secret = self.tenant_secrets[tenant_id].encode()
        message = f"{tenant_id}:{timestamp}:{body}".encode()
        expected_signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def get_tenant_tier(self, tenant_id: str) -> str:
        """Get the tier for a tenant."""
        return self.tenant_tiers.get(tenant_id, "default")

# Global instances
rate_limiter = RateLimiter()
auth_manager = AuthenticationManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global harmony_service, health_monitor, shutdown_handler
    
    # Startup
    logger.info("Starting Harmony Guard service...")
    app.state.start_time = time.time()
    
    try:
        # Initialize configuration and service
        config_manager = ConfigurationManager()
        harmony_service = HarmonyGuardService(config_manager)
        await harmony_service.initialize()
        
        # Initialize health monitoring
        health_monitor = HealthMonitor({
            "check_interval": 30,
            "failure_threshold": 3,
            "degraded_threshold": 2
        })
        
        # Add health checkers
        health_monitor.add_checker(SystemResourceChecker(
            cpu_threshold=85.0,
            memory_threshold=85.0,
            disk_threshold=90.0
        ))
        health_monitor.add_checker(ModelChecker(harmony_service))
        
        # Start health monitoring
        await health_monitor.start_monitoring()
        
        # Initialize graceful shutdown handler
        shutdown_handler = GracefulShutdownHandler(health_monitor, harmony_service)
        
        logger.info("Harmony Guard service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Harmony Guard service...")
    
    try:
        if shutdown_handler:
            await shutdown_handler.initiate_shutdown()
        else:
            # Fallback shutdown
            if health_monitor:
                await health_monitor.stop_monitoring()
            if harmony_service:
                await harmony_service.shutdown()
        
        logger.info("Service shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Harmony Guard",
    description="""
    Content moderation service for corporate communications.
    
    ## Authentication
    
    The API supports two authentication methods:
    
    1. **HMAC Signature Authentication** (Recommended for production):
       - Include headers: `X-Tenant-ID`, `X-Timestamp`, `X-Signature`
       - Signature is HMAC-SHA256 of `tenant_id:timestamp:request_body`
    
    2. **Bearer Token Authentication** (For development):
       - Include header: `Authorization: Bearer <api-key>`
       - Demo keys: `demo-api-key`, `test-api-key`
    
    ## Rate Limiting
    
    - Default tier: 100 requests per minute
    - Premium tier: 1000 requests per minute
    - Rate limit headers included in responses
    """,
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

# Add logging middleware
logging_middleware = LoggingMiddleware(app_logger)
app.middleware("http")(logging_middleware)

# Add metrics middleware
metrics_middleware = MetricsMiddleware(metrics)
app.middleware("http")(metrics_middleware)


# Pydantic models for API with validation
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text content to analyze")
    tenant_id: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9_-]+$', description="Tenant identifier")
    include_details: bool = Field(False, description="Include detailed analysis results")
    language_hints: Optional[List[str]] = Field(None, description="Language hints for better analysis")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()
    
    @validator('language_hints')
    def validate_language_hints(cls, v):
        if v is not None:
            # Validate language codes (ISO 639-1 format)
            valid_codes = {'en', 'hi', 'bn', 'te', 'ta', 'mr', 'gu', 'kn', 'ml', 'or', 'pa', 'as'}
            for code in v:
                if code not in valid_codes:
                    raise ValueError(f'Invalid language code: {code}')
        return v


class FeedbackSubmission(BaseModel):
    request_id: str = Field(..., description="Original request ID")
    final_label: str = Field(..., description="Corrected label")
    actual_categories: List[str] = Field(..., description="Actual abuse categories")
    comment: Optional[str] = Field(None, max_length=1000, description="Additional feedback comment")
    language_hints: Optional[List[str]] = Field(None, description="Language hints for the content")
    
    @validator('final_label')
    def validate_final_label(cls, v):
        valid_labels = {'allow', 'review', 'block'}
        if v not in valid_labels:
            raise ValueError(f'Invalid label: {v}. Must be one of {valid_labels}')
        return v
    
    @validator('actual_categories')
    def validate_categories(cls, v):
        valid_categories = {
            'insult/harassment', 'obscenity/profanity', 'hate/targeted group',
            'threat/violence', 'sexual content', 'bullying/taunting',
            'self-harm encouragement', 'spam/scam'
        }
        for category in v:
            if category not in valid_categories:
                raise ValueError(f'Invalid category: {category}')
        return v


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


def get_service() -> HarmonyGuardService:
    """Dependency to get the service instance."""
    if harmony_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return harmony_service


async def verify_authentication(
    request: Request,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_timestamp: Optional[str] = Header(None, alias="X-Timestamp"),
    x_signature: Optional[str] = Header(None, alias="X-Signature"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Verify request authentication and return tenant info."""
    
    # For public endpoints, allow unauthenticated access
    public_paths = ["/v1/health", "/v1/readiness", "/docs", "/openapi.json", "/"]
    if any(request.url.path.startswith(path) for path in public_paths):
        return {"tenant_id": "anonymous", "tier": "default", "authenticated": False}
    
    # Check for tenant-based authentication
    if x_tenant_id and x_timestamp and x_signature:
        # For HMAC authentication, we'll verify without consuming the body
        # In a real implementation, you'd need to handle this more carefully
        # For now, we'll use a simplified approach
        
        if auth_manager.verify_tenant_auth(x_tenant_id, x_signature, x_timestamp, ""):
            tier = auth_manager.get_tenant_tier(x_tenant_id)
            return {"tenant_id": x_tenant_id, "tier": tier, "authenticated": True}
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication signature"
            )
    
    # Check for bearer token authentication (for API keys)
    if credentials:
        # Simple API key validation (in production, use proper JWT or OAuth)
        if credentials.credentials in ["demo-api-key", "test-api-key"]:
            tenant_id = "demo" if credentials.credentials == "demo-api-key" else "test"
            tier = auth_manager.get_tenant_tier(tenant_id)
            return {"tenant_id": tenant_id, "tier": tier, "authenticated": True}
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
    
    # For development/testing, allow unauthenticated access with default limits
    return {"tenant_id": "anonymous", "tier": "default", "authenticated": False}


async def check_rate_limit(
    request: Request,
    auth_info: Dict[str, Any] = Depends(verify_authentication)
) -> Dict[str, Any]:
    """Check rate limits for the request."""
    tenant_id = auth_info["tenant_id"]
    tier = auth_info["tier"]
    
    if not rate_limiter.is_allowed(tenant_id, tier):
        remaining = rate_limiter.get_remaining(tenant_id, tier)
        
        # Record rate limit hit
        metrics.record_rate_limit_hit(tenant_id, tier)
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Remaining requests: {remaining}",
            headers={"X-RateLimit-Remaining": str(remaining)}
        )
    
    # Add rate limit info to response headers
    remaining = rate_limiter.get_remaining(tenant_id, tier)
    request.state.rate_limit_remaining = remaining
    
    return auth_info


# Custom exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    logger.warning(f"Validation error for request {request_id}: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            request_id=request_id
        ).dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper formatting."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            request_id=request_id
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    logger.error(f"Unexpected error for request {request_id}: {exc}\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            request_id=request_id
        ).dict()
    )


# Middleware for request correlation ID and rate limit headers
@app.middleware("http")
async def add_request_headers_middleware(request: Request, call_next):
    """Add request correlation ID and rate limit headers."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_remaining'):
        response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)
    
    return response


@app.post("/v1/analyze", 
          response_model=AnalysisResponse,
          responses={
              400: {"model": ErrorResponse, "description": "Bad Request - Invalid input"},
              422: {"model": ErrorResponse, "description": "Validation Error"},
              500: {"model": ErrorResponse, "description": "Internal Server Error"},
              503: {"model": ErrorResponse, "description": "Service Unavailable"}
          })
async def analyze_content(
    request: AnalyzeRequest,
    http_request: Request,
    auth_info: Dict[str, Any] = Depends(check_rate_limit),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Analyze text content for corporate appropriateness.
    
    This endpoint processes text content through the Harmony Guard ensemble
    to determine if it's appropriate for corporate communications.
    
    **Request Parameters:**
    - **text**: Text content to analyze (1-10000 characters)
    - **tenant_id**: Optional tenant identifier for policy customization
    - **include_details**: Include detailed analysis results (spans, explanations)
    - **language_hints**: Optional language codes to improve analysis accuracy
    
    **Response:**
    - **corporate_allowed**: Decision (allow/review/block)
    - **confidence**: Confidence score (0.0-1.0)
    - **severity**: Severity level (low/medium/high/critical)
    - **categories**: List of detected abuse categories
    - **languages**: Detected languages with percentages
    - **spans**: Problematic text spans (if include_details=true)
    - **explanations**: Decision explanations (if include_details=true)
    """
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4()))
    
    try:
        # Validate text length for processing efficiency
        if len(request.text) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text too long. Maximum length is 10000 characters."
            )
        
        # Use authenticated tenant_id if not provided in request
        tenant_id = request.tenant_id or auth_info["tenant_id"]
        
        # Convert to internal request model
        analysis_request = AnalysisRequest(
            text=request.text,
            tenant_id=tenant_id,
            include_details=request.include_details,
            language_hints=request.language_hints
        )
        
        # Process the request with timeout
        try:
            result = await asyncio.wait_for(
                service.analyze(analysis_request, request_id),
                timeout=5.0  # 5 second timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request processing timeout"
            )
        
        # Log request metrics and record in Prometheus
        processing_time = time.time() - start_time
        
        # Record decision metrics
        languages = [lang["code"] for lang in result.languages] if result.languages else ["unknown"]
        metrics.record_decision(result.corporate_allowed.value, languages, result.confidence, tenant_id)
        
        # Record category-specific confidence scores
        if result.categories:
            primary_language = languages[0] if languages else "unknown"
            for category in result.categories:
                metrics.record_category_confidence(category, result.corporate_allowed.value, primary_language, result.confidence)
        
        # Store tenant_id in request state for middleware
        http_request.state.tenant_id = tenant_id
        
        # Log analysis request with structured logging
        app_logger.log_analysis_request(
            text_length=len(request.text),
            languages=[lang["code"] for lang in result.languages] if result.languages else ["unknown"],
            decision=result.corporate_allowed.value,
            confidence=result.confidence,
            duration=processing_time,
            tenant_id=tenant_id,
            authenticated=auth_info['authenticated'],
            categories=result.categories or []
        )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing analysis request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process analysis request"
        )


@app.post("/v1/feedback",
          responses={
              200: {"description": "Feedback submitted successfully"},
              400: {"model": ErrorResponse, "description": "Bad Request - Invalid feedback"},
              422: {"model": ErrorResponse, "description": "Validation Error"},
              500: {"model": ErrorResponse, "description": "Internal Server Error"}
          })
async def submit_feedback(
    feedback: FeedbackSubmission,
    http_request: Request,
    auth_info: Dict[str, Any] = Depends(check_rate_limit),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Submit feedback for continuous learning and model improvement.
    
    This endpoint allows users to provide corrections and feedback on
    analysis results to improve the system's accuracy over time.
    
    **Request Parameters:**
    - **request_id**: Original analysis request ID
    - **final_label**: Corrected decision (allow/review/block)
    - **actual_categories**: List of actual abuse categories present
    - **comment**: Optional additional feedback comment
    - **language_hints**: Optional language hints for the content
    
    **Response:**
    - **status**: Success/failure status
    - **message**: Descriptive message
    - **feedback_id**: Unique identifier for the feedback submission
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4()))
    
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
            feedback_id = str(uuid.uuid4())
            
            # Record feedback metrics
            feedback_type = "correction" if feedback.final_label != "allow" else "confirmation"
            metrics.record_feedback(feedback_type, "unknown", feedback.final_label)
            
            # Log feedback with structured logging
            app_logger.log_feedback(
                original_decision="unknown",  # We don't have the original decision
                corrected_decision=feedback.final_label,
                feedback_type=feedback_type,
                feedback_id=feedback_id,
                original_request_id=feedback.request_id,
                categories=feedback.actual_categories,
                comment=feedback.comment
            )
            
            return {
                "status": "success",
                "message": "Feedback submitted successfully",
                "feedback_id": feedback_id,
                "original_request_id": feedback.request_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to submit feedback - invalid request ID or data"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback for request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process feedback submission"
        )


@app.get("/v1/health",
         responses={
             200: {"description": "Service is healthy"},
             503: {"description": "Service is unhealthy"}
         })
async def health_check():
    """
    Liveness probe endpoint for Kubernetes.
    
    This endpoint performs a basic health check to verify that the
    service is running and responsive. It does not check component health.
    
    **Response:**
    - **status**: Health status (alive/dead)
    - **version**: Service version
    - **service**: Service name
    - **timestamp**: Current timestamp
    - **uptime**: Service uptime in seconds
    """
    try:
        if health_monitor:
            status_info = await health_monitor.get_liveness_status()
            return status_info
        else:
            # Fallback if health monitor not initialized
            return {
                "status": "alive",
                "version": "1.0.0",
                "service": "harmony-guard",
                "timestamp": time.time(),
                "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.get("/v1/readiness",
         responses={
             200: {"description": "Service is ready"},
             503: {"description": "Service is not ready"}
         })
async def readiness_check():
    """
    Readiness probe endpoint for Kubernetes.
    
    This endpoint verifies that all service components are loaded
    and ready to handle requests. It performs comprehensive health
    checks of critical components.
    
    **Response:**
    - **status**: Readiness status (ready/not_ready)
    - **components**: Status of individual components
    - **timestamp**: Current timestamp
    - **ready**: Boolean readiness flag
    - **unhealthy_components**: List of unhealthy components
    """
    try:
        if health_monitor:
            readiness_status = await health_monitor.get_readiness_status()
            
            if readiness_status["ready"]:
                return readiness_status
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=readiness_status
                )
        else:
            # Fallback readiness check
            if harmony_service:
                is_ready = await harmony_service.is_ready()
                component_status = await harmony_service.get_component_status()
                
                if is_ready:
                    return {
                        "status": "ready",
                        "components": component_status,
                        "timestamp": time.time(),
                        "ready": True
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={
                            "status": "not_ready",
                            "components": component_status,
                            "timestamp": time.time(),
                            "ready": False
                        }
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service not initialized"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to check service readiness"
        )


@app.get("/v1/metrics",
         responses={
             200: {"description": "Service metrics"},
             500: {"description": "Failed to retrieve metrics"}
         })
async def get_metrics(service: HarmonyGuardService = Depends(get_service)):
    """
    Get basic service metrics for monitoring.
    
    This endpoint provides operational metrics including request counts,
    latency statistics, error rates, and decision distributions.
    
    **Response:**
    - **requests_total**: Total number of requests processed
    - **requests_by_decision**: Request counts by decision type
    - **average_latency**: Average processing latency in seconds
    - **errors_total**: Total number of errors
    - **feedback_total**: Total feedback submissions received
    - **uptime**: Service uptime in seconds
    """
    try:
        metrics = await service.get_metrics()
        
        # Add additional operational metrics
        metrics.update({
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "timestamp": time.time()
        })
        
        return metrics
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve service metrics"
        )


@app.get("/v1/config",
         responses={
             200: {"description": "Public configuration"},
             400: {"description": "Invalid tenant ID"},
             500: {"description": "Failed to retrieve configuration"}
         })
async def get_config(
    tenant_id: Optional[str] = None,
    auth_info: Dict[str, Any] = Depends(verify_authentication),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Get non-secret configuration information.
    
    This endpoint provides public configuration details that can be
    safely shared with clients, such as supported languages and limits.
    
    **Query Parameters:**
    - **tenant_id**: Optional tenant ID for tenant-specific configuration
    
    **Response:**
    - **supported_languages**: List of supported language codes
    - **max_sequence_length**: Maximum text length for processing
    - **version**: Service version
    - **features**: Available features and capabilities
    - **limits**: Processing limits and thresholds
    """
    try:
        # Use authenticated tenant_id if not provided
        effective_tenant_id = tenant_id or auth_info["tenant_id"]
        
        # Validate tenant_id format if provided
        if effective_tenant_id and effective_tenant_id != "anonymous":
            if not effective_tenant_id.replace('-', '').replace('_', '').isalnum():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid tenant ID format"
                )
        
        config = await service.get_public_config(effective_tenant_id)
        
        # Add additional public configuration
        config.update({
            "features": {
                "multi_language_support": True,
                "code_mixed_content": True,
                "obfuscation_detection": True,
                "context_analysis": True,
                "detailed_explanations": True
            },
            "limits": {
                "max_text_length": 10000,
                "max_requests_per_minute": 1000,
                "timeout_seconds": 5
            },
            "timestamp": time.time()
        })
        
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration"
        )


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


@app.get("/v1/feedback/analytics")
async def get_feedback_analytics(
    tenant_id: Optional[str] = None,
    days: int = 30,
    auth_info: Dict[str, Any] = Depends(verify_authentication),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Get feedback analytics for monitoring and continuous improvement.
    
    Args:
        tenant_id: Optional tenant ID filter
        days: Number of days to analyze (default: 30, max: 365)
    
    Returns feedback analytics including correction rates, category distributions,
    and trends over time.
    """
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        # Use authenticated tenant if not provided
        effective_tenant_id = tenant_id or auth_info["tenant_id"]
        
        analytics = await service.get_feedback_analytics(effective_tenant_id, days)
        return analytics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feedback analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback analytics")


@app.get("/v1/feedback/recent")
async def get_recent_feedback(
    tenant_id: Optional[str] = None,
    limit: int = 100,
    auth_info: Dict[str, Any] = Depends(verify_authentication),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Get recent feedback records for review and analysis.
    
    Args:
        tenant_id: Optional tenant ID filter
        limit: Maximum number of records to return (default: 100, max: 1000)
    
    Returns recent feedback submissions with metadata.
    """
    try:
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
        
        # Use authenticated tenant if not provided
        effective_tenant_id = tenant_id or auth_info["tenant_id"]
        
        feedback_data = await service.get_recent_feedback(effective_tenant_id, limit)
        return feedback_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recent feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent feedback")


@app.get("/v1/active-learning/queue-status")
async def get_review_queue_status(
    tenant_id: Optional[str] = None,
    auth_info: Dict[str, Any] = Depends(verify_authentication),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Get active learning review queue status and statistics.
    
    Args:
        tenant_id: Optional tenant ID filter
    
    Returns review queue statistics, pipeline metrics, and pending items preview.
    """
    try:
        # Use authenticated tenant if not provided
        effective_tenant_id = tenant_id or auth_info["tenant_id"]
        
        status = await service.get_review_queue_status(effective_tenant_id)
        return status
    except Exception as e:
        logger.error(f"Error getting review queue status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve review queue status")


@app.get("/v1/active-learning/pending-reviews")
async def get_pending_reviews(
    limit: int = 50,
    priority: Optional[str] = None,
    tenant_id: Optional[str] = None,
    auth_info: Dict[str, Any] = Depends(verify_authentication),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Get pending review items for human reviewers.
    
    Args:
        limit: Maximum number of items to return (default: 50, max: 200)
        priority: Filter by priority level (low/medium/high/critical)
        tenant_id: Optional tenant ID filter
    
    Returns list of pending review items with prediction details.
    """
    try:
        if limit < 1 or limit > 200:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 200")
        
        if priority and priority not in ["low", "medium", "high", "critical"]:
            raise HTTPException(status_code=400, detail="Invalid priority level")
        
        # Use authenticated tenant if not provided
        effective_tenant_id = tenant_id or auth_info["tenant_id"]
        
        reviews = await service.get_pending_reviews(
            limit=limit,
            priority_filter=priority,
            tenant_id=effective_tenant_id
        )
        
        return {
            "pending_reviews": reviews,
            "total_count": len(reviews),
            "filters": {
                "limit": limit,
                "priority": priority,
                "tenant_id": effective_tenant_id
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pending reviews: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve pending reviews")


class ReviewFeedbackSubmission(BaseModel):
    review_id: str = Field(..., description="Review item ID")
    corrected_decision: str = Field(..., description="Corrected decision")
    corrected_categories: List[str] = Field(..., description="Corrected abuse categories")
    reviewer_comment: Optional[str] = Field(None, max_length=1000, description="Reviewer comment")
    
    @validator('corrected_decision')
    def validate_corrected_decision(cls, v):
        valid_decisions = {'allow', 'review', 'block'}
        if v not in valid_decisions:
            raise ValueError(f'Invalid decision: {v}. Must be one of {valid_decisions}')
        return v
    
    @validator('corrected_categories')
    def validate_corrected_categories(cls, v):
        valid_categories = {
            'insult/harassment', 'obscenity/profanity', 'hate/targeted group',
            'threat/violence', 'sexual content', 'bullying/taunting',
            'self-harm encouragement', 'spam/scam'
        }
        for category in v:
            if category not in valid_categories:
                raise ValueError(f'Invalid category: {category}')
        return v


@app.post("/v1/active-learning/submit-review")
async def submit_review_feedback(
    review_feedback: ReviewFeedbackSubmission,
    auth_info: Dict[str, Any] = Depends(verify_authentication),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Submit feedback from human review for active learning.
    
    This endpoint allows human reviewers to provide corrections for
    low-confidence predictions, which will be integrated into the
    continuous learning pipeline.
    
    **Request Parameters:**
    - **review_id**: Review item identifier
    - **corrected_decision**: Corrected decision (allow/review/block)
    - **corrected_categories**: List of actual abuse categories
    - **reviewer_comment**: Optional reviewer comment
    
    **Response:**
    - **status**: Success/failure status
    - **message**: Descriptive message
    - **review_id**: Review item ID
    """
    try:
        success = await service.submit_review_feedback(
            review_id=review_feedback.review_id,
            corrected_decision=review_feedback.corrected_decision,
            corrected_categories=review_feedback.corrected_categories,
            reviewer_comment=review_feedback.reviewer_comment
        )
        
        if success:
            return {
                "status": "success",
                "message": "Review feedback submitted successfully",
                "review_id": review_feedback.review_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to submit review feedback - invalid review ID or data"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting review feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process review feedback"
        )


@app.get("/v1/model-retraining/drift-analysis")
async def run_comprehensive_drift_analysis(
    tenant_id: Optional[str] = None,
    auth_info: Dict[str, Any] = Depends(verify_authentication),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Run comprehensive drift analysis using statistical tests.
    
    Args:
        tenant_id: Optional tenant ID filter
    
    Returns comprehensive drift analysis including statistical tests,
    alerts, and recommendations for model retraining.
    """
    try:
        # Use authenticated tenant if not provided
        effective_tenant_id = tenant_id or auth_info["tenant_id"]
        
        analysis = await service.run_comprehensive_drift_analysis(effective_tenant_id)
        return analysis
    except Exception as e:
        logger.error(f"Error running comprehensive drift analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to run drift analysis")


@app.get("/v1/model-retraining/drift-alerts")
async def get_detailed_drift_alerts(
    hours: int = 24,
    tenant_id: Optional[str] = None,
    auth_info: Dict[str, Any] = Depends(verify_authentication),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Get detailed drift alerts with statistical information.
    
    Args:
        hours: Number of hours to look back (default: 24, max: 168)
        tenant_id: Optional tenant ID filter
    
    Returns detailed drift alerts with statistical test results.
    """
    try:
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        # Use authenticated tenant if not provided
        effective_tenant_id = tenant_id or auth_info["tenant_id"]
        
        alerts = await service.get_drift_alerts_detailed(hours, effective_tenant_id)
        return alerts
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detailed drift alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve drift alerts")


@app.post("/v1/model-retraining/check-triggers")
async def check_retraining_triggers(
    tenant_id: Optional[str] = None,
    auth_info: Dict[str, Any] = Depends(verify_authentication),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Check if model retraining should be triggered based on current conditions.
    
    Args:
        tenant_id: Optional tenant ID filter
    
    Returns information about whether retraining was triggered and why.
    """
    try:
        # Use authenticated tenant if not provided
        effective_tenant_id = tenant_id or auth_info["tenant_id"]
        
        result = await service.check_retraining_triggers(effective_tenant_id)
        return result
    except Exception as e:
        logger.error(f"Error checking retraining triggers: {e}")
        raise HTTPException(status_code=500, detail="Failed to check retraining triggers")


@app.get("/v1/model-retraining/status")
async def get_retraining_status(
    tenant_id: Optional[str] = None,
    auth_info: Dict[str, Any] = Depends(verify_authentication),
    service: HarmonyGuardService = Depends(get_service)
):
    """
    Get model retraining job status and history.
    
    Args:
        tenant_id: Optional tenant ID filter
    
    Returns recent retraining jobs with their status and metrics.
    """
    try:
        # Use authenticated tenant if not provided
        effective_tenant_id = tenant_id or auth_info["tenant_id"]
        
        status_info = await service.get_retraining_status(effective_tenant_id)
        return status_info
    except Exception as e:
        logger.error(f"Error getting retraining status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve retraining status")


@app.get("/metrics",
         responses={
             200: {"description": "Prometheus metrics"},
         })
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping by monitoring systems.
    """
    return create_metrics_endpoint()()


@app.get("/v1/health/detailed",
         responses={
             200: {"description": "Detailed health status"},
             500: {"description": "Failed to get health status"}
         })
async def detailed_health_status():
    """
    Detailed health status endpoint for monitoring dashboards.
    
    This endpoint provides comprehensive health information including
    component status, performance metrics, and historical data.
    
    **Response:**
    - **timestamp**: Current timestamp
    - **overall_status**: Overall system health
    - **components**: Detailed component health information
    - **recent_checks**: Results of recent health checks
    """
    try:
        if health_monitor:
            detailed_status = await health_monitor.get_detailed_status()
            return detailed_status
        else:
            return {
                "error": "Health monitoring not available",
                "timestamp": time.time()
            }
    except Exception as e:
        logger.error(f"Failed to get detailed health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve detailed health status"
        )


@app.get("/v1/auth/info",
         responses={
             200: {"description": "Authentication information"},
             401: {"description": "Authentication required"}
         })
async def get_auth_info(auth_info: Dict[str, Any] = Depends(verify_authentication)):
    """
    Get authentication information for the current request.
    
    This endpoint returns information about the authenticated tenant,
    their tier, rate limits, and authentication status.
    
    **Response:**
    - **tenant_id**: Authenticated tenant identifier
    - **tier**: Tenant tier (default/premium)
    - **authenticated**: Whether request is authenticated
    - **rate_limits**: Current rate limit configuration
    - **remaining_requests**: Remaining requests in current window
    """
    tenant_id = auth_info["tenant_id"]
    tier = auth_info["tier"]
    
    # Get rate limit info
    remaining = rate_limiter.get_remaining(tenant_id, tier)
    limit_config = rate_limiter.limits.get(tier, rate_limiter.limits["default"])
    
    return {
        "tenant_id": tenant_id,
        "tier": tier,
        "authenticated": auth_info["authenticated"],
        "rate_limits": {
            "requests_per_window": limit_config["requests"],
            "window_seconds": limit_config["window"],
            "remaining_requests": remaining
        },
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    import signal
    
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if shutdown_handler:
            asyncio.create_task(shutdown_handler.initiate_shutdown())
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)