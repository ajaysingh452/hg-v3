"""Health check and readiness probe system for Harmony Guard."""

import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    details: Optional[Dict[str, Any]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    name: str
    status: HealthStatus
    last_check: float
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0
    average_duration_ms: float = 0.0
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """Base class for health checkers."""
    
    def __init__(self, name: str, timeout: float = 5.0):
        """Initialize health checker."""
        self.name = name
        self.timeout = timeout
    
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(self._check_health(), timeout=self.timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=result.get("message", "Health check passed"),
                duration_ms=duration_ms,
                details=result.get("details")
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error_type": type(e).__name__, "error_message": str(e)}
            )
    
    async def _check_health(self) -> Dict[str, Any]:
        """Override this method to implement specific health check logic."""
        raise NotImplementedError


class SystemResourceChecker(HealthChecker):
    """Health checker for system resources."""
    
    def __init__(self, cpu_threshold: float = 90.0, memory_threshold: float = 90.0, 
                 disk_threshold: float = 90.0):
        """Initialize system resource checker."""
        super().__init__("system_resources")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def _check_health(self) -> Dict[str, Any]:
        """Check system resource usage."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_free_gb": disk.free / (1024**3)
        }
        
        # Check thresholds
        issues = []
        if cpu_percent > self.cpu_threshold:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        if memory_percent > self.memory_threshold:
            issues.append(f"High memory usage: {memory_percent:.1f}%")
        if disk_percent > self.disk_threshold:
            issues.append(f"High disk usage: {disk_percent:.1f}%")
        
        if issues:
            return {
                "message": f"Resource issues detected: {', '.join(issues)}",
                "details": details
            }
        
        return {
            "message": "System resources are healthy",
            "details": details
        }


class DatabaseChecker(HealthChecker):
    """Health checker for database connections."""
    
    def __init__(self, connection_string: str):
        """Initialize database checker."""
        super().__init__("database")
        self.connection_string = connection_string
    
    async def _check_health(self) -> Dict[str, Any]:
        """Check database connectivity."""
        # This is a placeholder - implement actual database check
        # For now, just simulate a successful check
        await asyncio.sleep(0.1)  # Simulate database query
        
        return {
            "message": "Database connection is healthy",
            "details": {"connection_pool_size": 10, "active_connections": 3}
        }


class ModelChecker(HealthChecker):
    """Health checker for ML models."""
    
    def __init__(self, model_service):
        """Initialize model checker."""
        super().__init__("ml_models")
        self.model_service = model_service
    
    async def _check_health(self) -> Dict[str, Any]:
        """Check model availability and performance."""
        if not self.model_service:
            raise Exception("Model service not available")
        
        # Check if models are loaded
        if not hasattr(self.model_service, 'classifier') or not self.model_service.classifier:
            raise Exception("Classifier model not loaded")
        
        # Perform a quick inference test
        test_text = "Hello world"
        try:
            # This would be a real inference call
            await asyncio.sleep(0.05)  # Simulate inference
            
            return {
                "message": "ML models are healthy",
                "details": {
                    "classifier_loaded": True,
                    "test_inference_time_ms": 50,
                    "model_memory_usage_mb": 512
                }
            }
        except Exception as e:
            raise Exception(f"Model inference test failed: {str(e)}")


class ExternalServiceChecker(HealthChecker):
    """Health checker for external service dependencies."""
    
    def __init__(self, service_name: str, endpoint: str, timeout: float = 3.0):
        """Initialize external service checker."""
        super().__init__(f"external_{service_name}")
        self.service_name = service_name
        self.endpoint = endpoint
        self.timeout = timeout
    
    async def _check_health(self) -> Dict[str, Any]:
        """Check external service availability."""
        import aiohttp
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(self.endpoint) as response:
                if response.status == 200:
                    return {
                        "message": f"{self.service_name} is healthy",
                        "details": {
                            "status_code": response.status,
                            "response_time_ms": self.timeout * 1000
                        }
                    }
                else:
                    raise Exception(f"Service returned status {response.status}")


class HealthMonitor:
    """Health monitoring system for Harmony Guard."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize health monitor."""
        self.config = config or {}
        self.checkers: Dict[str, HealthChecker] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.last_full_check: Optional[float] = None
        
        # Configuration
        self.check_interval = self.config.get("check_interval", 30)  # seconds
        self.failure_threshold = self.config.get("failure_threshold", 3)
        self.degraded_threshold = self.config.get("degraded_threshold", 2)
        
        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
    
    def add_checker(self, checker: HealthChecker):
        """Add a health checker."""
        self.checkers[checker.name] = checker
        self.component_health[checker.name] = ComponentHealth(
            name=checker.name,
            status=HealthStatus.UNKNOWN,
            last_check=0
        )
    
    def remove_checker(self, name: str):
        """Remove a health checker."""
        self.checkers.pop(name, None)
        self.component_health.pop(name, None)
    
    async def start_monitoring(self):
        """Start background health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop background health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                await self.check_all_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def check_all_components(self) -> Dict[str, HealthCheckResult]:
        """Check health of all registered components."""
        results = {}
        
        # Run all health checks concurrently
        tasks = {name: checker.check() for name, checker in self.checkers.items()}
        
        if tasks:
            completed_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            for (name, _), result in zip(tasks.items(), completed_results):
                if isinstance(result, Exception):
                    result = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {str(result)}",
                        duration_ms=0
                    )
                
                results[name] = result
                self._update_component_health(name, result)
        
        self.last_full_check = time.time()
        return results
    
    def _update_component_health(self, name: str, result: HealthCheckResult):
        """Update component health tracking."""
        if name not in self.component_health:
            self.component_health[name] = ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                last_check=0
            )
        
        component = self.component_health[name]
        component.last_check = result.timestamp
        component.total_checks += 1
        
        # Update duration average
        if component.total_checks == 1:
            component.average_duration_ms = result.duration_ms
        else:
            component.average_duration_ms = (
                (component.average_duration_ms * (component.total_checks - 1) + result.duration_ms) 
                / component.total_checks
            )
        
        # Update failure tracking
        if result.status == HealthStatus.UNHEALTHY:
            component.consecutive_failures += 1
            component.total_failures += 1
        else:
            component.consecutive_failures = 0
        
        # Determine overall status
        if component.consecutive_failures >= self.failure_threshold:
            component.status = HealthStatus.UNHEALTHY
        elif component.consecutive_failures >= self.degraded_threshold:
            component.status = HealthStatus.DEGRADED
        else:
            component.status = result.status
        
        component.details = result.details
    
    async def get_liveness_status(self) -> Dict[str, Any]:
        """Get liveness probe status (basic service availability)."""
        return {
            "status": "alive",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - (self.last_full_check or time.time()),
            "service": "harmony-guard",
            "version": "1.0.0"
        }
    
    async def get_readiness_status(self) -> Dict[str, Any]:
        """Get readiness probe status (service ready to handle requests)."""
        # Check if we have recent health check results
        if not self.last_full_check or time.time() - self.last_full_check > self.check_interval * 2:
            # Perform immediate health check if data is stale
            await self.check_all_components()
        
        # Determine overall readiness
        critical_components = ["ml_models", "system_resources"]
        ready = True
        unhealthy_components = []
        
        for name, component in self.component_health.items():
            if component.status == HealthStatus.UNHEALTHY:
                if name in critical_components:
                    ready = False
                unhealthy_components.append(name)
        
        status = "ready" if ready else "not_ready"
        
        return {
            "status": status,
            "timestamp": time.time(),
            "last_check": self.last_full_check,
            "components": {
                name: {
                    "status": component.status.value,
                    "last_check": component.last_check,
                    "consecutive_failures": component.consecutive_failures,
                    "success_rate": (component.total_checks - component.total_failures) / max(component.total_checks, 1),
                    "average_duration_ms": component.average_duration_ms
                }
                for name, component in self.component_health.items()
            },
            "unhealthy_components": unhealthy_components,
            "ready": ready
        }
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed health status for monitoring dashboards."""
        # Perform fresh health check
        check_results = await self.check_all_components()
        
        return {
            "timestamp": time.time(),
            "overall_status": self._calculate_overall_status(),
            "components": {
                name: {
                    "status": component.status.value,
                    "last_check": component.last_check,
                    "total_checks": component.total_checks,
                    "total_failures": component.total_failures,
                    "consecutive_failures": component.consecutive_failures,
                    "success_rate": (component.total_checks - component.total_failures) / max(component.total_checks, 1),
                    "average_duration_ms": component.average_duration_ms,
                    "details": component.details
                }
                for name, component in self.component_health.items()
            },
            "recent_checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "details": result.details
                }
                for name, result in check_results.items()
            }
        }
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall system health status."""
        if not self.component_health:
            return HealthStatus.UNKNOWN.value
        
        unhealthy_count = sum(1 for c in self.component_health.values() 
                             if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in self.component_health.values() 
                            if c.status == HealthStatus.DEGRADED)
        
        total_components = len(self.component_health)
        
        if unhealthy_count > total_components * 0.5:
            return HealthStatus.UNHEALTHY.value
        elif unhealthy_count > 0 or degraded_count > total_components * 0.3:
            return HealthStatus.DEGRADED.value
        else:
            return HealthStatus.HEALTHY.value


class GracefulShutdownHandler:
    """Handler for graceful service shutdown."""
    
    def __init__(self, health_monitor: HealthMonitor, service_instance=None):
        """Initialize shutdown handler."""
        self.health_monitor = health_monitor
        self.service_instance = service_instance
        self.shutdown_timeout = 30  # seconds
        self._shutdown_initiated = False
    
    async def initiate_shutdown(self):
        """Initiate graceful shutdown sequence."""
        if self._shutdown_initiated:
            return
        
        self._shutdown_initiated = True
        logger.info("Initiating graceful shutdown...")
        
        try:
            # Stop accepting new requests (this would be handled by the web server)
            logger.info("Stopping health monitoring...")
            await self.health_monitor.stop_monitoring()
            
            # Wait for active requests to complete
            logger.info("Waiting for active requests to complete...")
            await self._wait_for_active_requests()
            
            # Shutdown service components
            if self.service_instance:
                logger.info("Shutting down service components...")
                await self.service_instance.shutdown()
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            raise
    
    async def _wait_for_active_requests(self):
        """Wait for active requests to complete."""
        # This is a placeholder - in a real implementation, you'd track active requests
        # and wait for them to complete or timeout
        await asyncio.sleep(1)
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated."""
        return self._shutdown_initiated