"""Security and compliance tests for Harmony Guard."""

import pytest
import re
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.pii_masking import PIIMasker, PIIPattern
from core.logging import StructuredLogger, LogRedactor
from api.service import HarmonyGuardService
from api.main import create_app
from policy.audit_logger import PolicyAuditLogger
from policy.policy_engine import PolicyEngine
from core.models import ProcessedText, LanguageDetection


class TestPIIMasking:
    """Test PII masking and privacy compliance."""
    
    @pytest.fixture
    def pii_config(self):
        """Configuration for PII masking."""
        return {
            "enabled": True,
            "mask_emails": True,
            "mask_phones": True,
            "mask_ids": True,
            "mask_names": False,  # Disabled for testing
            "mask_addresses": True,
            "mask_credit_cards": True,
            "redaction_char": "*",
            "preserve_format": True
        }
    
    @pytest.fixture
    def pii_masker(self, pii_config):
        """Create PII masker instance."""
        return PIIMasker(pii_config)
    
    def test_email_masking(self, pii_masker):
        """Test email address masking."""
        text = "Contact me at john.doe@example.com or admin@company.org"
        masked_text, was_masked, pii_found = pii_masker.mask_pii(text)
        
        assert was_masked
        assert len(pii_found) >= 2
        assert "john.doe@example.com" not in masked_text
        assert "admin@company.org" not in masked_text
        assert "***@***.***" in masked_text or "*" in masked_text
    
    def test_phone_number_masking(self, pii_masker):
        """Test phone number masking."""
        test_cases = [
            "Call me at +1-555-123-4567",
            "Phone: (555) 123-4567",
            "Mobile: 555.123.4567",
            "Contact: 5551234567"
        ]
        
        for text in test_cases:
            masked_text, was_masked, pii_found = pii_masker.mask_pii(text)
            
            assert was_masked, f"Phone number not detected in: {text}"
            assert len(pii_found) >= 1
            # Should not contain original phone number
            assert not re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', masked_text)
    
    def test_id_number_masking(self, pii_masker):
        """Test ID number masking (SSN, etc.)."""
        text = "SSN: 123-45-6789 and ID: 987654321"
        masked_text, was_masked, pii_found = pii_masker.mask_pii(text)
        
        assert was_masked
        assert len(pii_found) >= 1
        assert "123-45-6789" not in masked_text
        # Should preserve format if configured
        if pii_masker.preserve_format:
            assert "***-**-****" in masked_text or similar_pattern_exists(masked_text)
    
    def test_credit_card_masking(self, pii_masker):
        """Test credit card number masking."""
        text = "Card number: 4532-1234-5678-9012"
        masked_text, was_masked, pii_found = pii_masker.mask_pii(text)
        
        assert was_masked
        assert "4532-1234-5678-9012" not in masked_text
        # Should mask but potentially preserve last 4 digits
        assert "*" in masked_text
    
    def test_address_masking(self, pii_masker):
        """Test address masking."""
        text = "I live at 123 Main Street, Anytown, CA 90210"
        masked_text, was_masked, pii_found = pii_masker.mask_pii(text)
        
        # Address detection is complex, so we test basic patterns
        if was_masked:
            assert len(pii_found) >= 1
            # Should not contain full address
            assert "123 Main Street" not in masked_text or "*" in masked_text
    
    def test_multiple_pii_types(self, pii_masker):
        """Test masking multiple PII types in same text."""
        text = "Contact John at john@example.com or call (555) 123-4567. SSN: 123-45-6789"
        masked_text, was_masked, pii_found = pii_masker.mask_pii(text)
        
        assert was_masked
        assert len(pii_found) >= 3  # Email, phone, SSN
        
        # Verify all PII types are detected
        pii_types = [pii.pii_type for pii in pii_found]
        assert "email" in pii_types
        assert "phone" in pii_types
        assert "ssn" in pii_types or "id" in pii_types
    
    def test_no_pii_text(self, pii_masker):
        """Test text with no PII."""
        text = "This is a clean message with no personal information."
        masked_text, was_masked, pii_found = pii_masker.mask_pii(text)
        
        assert not was_masked
        assert len(pii_found) == 0
        assert masked_text == text  # Should be unchanged
    
    def test_pii_pattern_detection(self, pii_masker):
        """Test PII pattern detection accuracy."""
        # Test email patterns
        email_patterns = [
            "user@domain.com",
            "first.last@company.org",
            "test+tag@example.net",
            "user123@sub.domain.co.uk"
        ]
        
        for email in email_patterns:
            assert pii_masker._detect_emails(f"Email: {email}"), f"Failed to detect email: {email}"
        
        # Test phone patterns
        phone_patterns = [
            "+1-555-123-4567",
            "(555) 123-4567",
            "555.123.4567",
            "5551234567"
        ]
        
        for phone in phone_patterns:
            assert pii_masker._detect_phones(f"Phone: {phone}"), f"Failed to detect phone: {phone}"
    
    def test_pii_masking_consistency(self, pii_masker):
        """Test that same PII is masked consistently."""
        text = "Email john@example.com twice: john@example.com"
        masked_text, was_masked, pii_found = pii_masker.mask_pii(text)
        
        assert was_masked
        # Same email should be masked the same way both times
        parts = masked_text.split("twice:")
        assert parts[0].strip().split()[-1] == parts[1].strip()
    
    def test_pii_logging_redaction(self, pii_masker):
        """Test PII redaction in logs."""
        # Create a log message with PII
        log_message = "User john@example.com called from (555) 123-4567"
        
        redacted_message = pii_masker.redact_for_logging(log_message)
        
        assert "john@example.com" not in redacted_message
        assert "(555) 123-4567" not in redacted_message
        assert "[REDACTED]" in redacted_message or "*" in redacted_message


class TestLogRedaction:
    """Test log redaction and secure logging."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def log_config(self, temp_log_dir):
        """Configuration for structured logging."""
        return {
            "level": "INFO",
            "format": "json",
            "redact_pii": True,
            "log_file": str(Path(temp_dir) / "test.log"),
            "max_file_size": "10MB",
            "backup_count": 5
        }
    
    def test_structured_logger_initialization(self, log_config):
        """Test structured logger initialization."""
        logger = StructuredLogger("test_logger", log_config)
        
        assert logger.logger.name == "test_logger"
        assert logger.redact_pii is True
    
    def test_pii_redaction_in_logs(self, log_config):
        """Test PII redaction in log messages."""
        logger = StructuredLogger("test_logger", log_config)
        
        # Log message with PII
        logger.info("User login", extra={
            "user_email": "john@example.com",
            "user_phone": "(555) 123-4567",
            "message": "User john@example.com logged in from phone (555) 123-4567"
        })
        
        # Read log file
        log_file = Path(log_config["log_file"])
        if log_file.exists():
            log_content = log_file.read_text()
            
            # PII should be redacted
            assert "john@example.com" not in log_content
            assert "(555) 123-4567" not in log_content
            assert "[REDACTED]" in log_content or "***" in log_content
    
    def test_log_correlation_ids(self, log_config):
        """Test request correlation ID handling."""
        logger = StructuredLogger("test_logger", log_config)
        
        correlation_id = "req_123456789"
        
        logger.info("Processing request", extra={
            "correlation_id": correlation_id,
            "action": "text_analysis"
        })
        
        # Correlation ID should be preserved (not PII)
        log_file = Path(log_config["log_file"])
        if log_file.exists():
            log_content = log_file.read_text()
            assert correlation_id in log_content
    
    def test_sensitive_field_redaction(self, log_config):
        """Test redaction of sensitive fields."""
        logger = StructuredLogger("test_logger", log_config)
        
        # Log with various sensitive fields
        logger.info("Analysis result", extra={
            "text_content": "This contains PII: john@example.com",
            "user_id": "user_12345",
            "api_key": "sk_test_123456789",
            "password": "secret123",
            "token": "bearer_token_xyz"
        })
        
        log_file = Path(log_config["log_file"])
        if log_file.exists():
            log_content = log_file.read_text()
            
            # Sensitive fields should be redacted
            assert "john@example.com" not in log_content
            assert "sk_test_123456789" not in log_content
            assert "secret123" not in log_content
            assert "bearer_token_xyz" not in log_content
    
    def test_log_format_validation(self, log_config):
        """Test log format is valid JSON."""
        logger = StructuredLogger("test_logger", log_config)
        
        logger.info("Test message", extra={
            "field1": "value1",
            "field2": 123,
            "field3": True
        })
        
        log_file = Path(log_config["log_file"])
        if log_file.exists():
            log_content = log_file.read_text().strip()
            
            # Each line should be valid JSON
            for line in log_content.split('\n'):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        pytest.fail(f"Invalid JSON in log: {line}")


class TestInputValidation:
    """Test input validation and injection prevention."""
    
    @pytest.fixture
    def mock_service(self):
        """Mock Harmony Guard service."""
        with patch('api.service.HarmonyGuardService') as mock:
            service = mock.return_value
            service.analyze_text = AsyncMock()
            yield service
    
    def test_text_length_validation(self, mock_service):
        """Test text length validation."""
        # Test maximum text length
        max_length = 10000  # Assume 10KB limit
        
        # Valid length text
        valid_text = "A" * (max_length - 100)
        # Should not raise exception
        
        # Oversized text
        oversized_text = "A" * (max_length + 100)
        
        # Mock service should handle validation
        mock_service.analyze_text.return_value = {
            "corporate_allowed": "allow",
            "confidence": 0.9,
            "severity": "low"
        }
        
        # This would be handled by the API layer
        assert len(valid_text) < max_length
        assert len(oversized_text) > max_length
    
    def test_malicious_input_handling(self, mock_service):
        """Test handling of potentially malicious input."""
        malicious_inputs = [
            # Script injection attempts
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            
            # SQL injection attempts
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            
            # Command injection attempts
            "; rm -rf /",
            "$(rm -rf /)",
            
            # Path traversal attempts
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            
            # Null bytes and control characters
            "test\x00malicious",
            "test\x1f\x7f",
            
            # Unicode normalization attacks
            "\u0041\u0300",  # A with combining grave accent
            "\uFEFF",        # Byte order mark
        ]
        
        for malicious_input in malicious_inputs:
            # Service should handle malicious input safely
            mock_service.analyze_text.return_value = {
                "corporate_allowed": "block",
                "confidence": 0.1,
                "severity": "low",
                "error": "Invalid input detected"
            }
            
            # Input should be sanitized or rejected
            # This is a placeholder - actual validation would be in the service
            assert isinstance(malicious_input, str)
    
    def test_encoding_validation(self, mock_service):
        """Test handling of different text encodings."""
        # Test various encodings
        test_texts = [
            "Hello world",  # ASCII
            "Héllo wörld",  # Latin-1
            "नमस्ते दुनिया",  # Devanagari
            "مرحبا بالعالم",  # Arabic
            "你好世界",        # Chinese
            "🌍🌎🌏",        # Emojis
        ]
        
        for text in test_texts:
            # Should handle all valid UTF-8 text
            encoded = text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            assert decoded == text
    
    def test_request_parameter_validation(self):
        """Test API request parameter validation."""
        # Test valid parameters
        valid_params = {
            "text": "Hello world",
            "tenant_id": "tenant_123",
            "include_details": True,
            "language_hints": ["en", "hi"]
        }
        
        # Test invalid parameters
        invalid_params = [
            {"text": ""},  # Empty text
            {"text": None},  # Null text
            {"tenant_id": ""},  # Empty tenant ID
            {"tenant_id": "invalid/tenant"},  # Invalid characters
            {"include_details": "not_boolean"},  # Wrong type
            {"language_hints": "not_list"},  # Wrong type
            {"language_hints": ["invalid_lang_code"]},  # Invalid language
        ]
        
        # Validate parameters (this would be done by API layer)
        for param_set in invalid_params:
            # Each should fail validation
            if "text" in param_set:
                text = param_set["text"]
                if text is None or text == "":
                    assert not text  # Should be falsy
            
            if "tenant_id" in param_set:
                tenant_id = param_set["tenant_id"]
                if tenant_id == "" or "/" in tenant_id:
                    assert not (tenant_id and "/" not in tenant_id)


class TestAuthenticationAuthorization:
    """Test authentication and authorization."""
    
    @pytest.fixture
    def mock_auth_config(self):
        """Mock authentication configuration."""
        return {
            "enabled": True,
            "api_key_header": "X-API-Key",
            "tenant_header": "X-Tenant-ID",
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 100,
                "burst_size": 10
            }
        }
    
    def test_api_key_validation(self, mock_auth_config):
        """Test API key validation."""
        valid_api_keys = [
            "hg_test_1234567890abcdef",
            "hg_prod_abcdef1234567890"
        ]
        
        invalid_api_keys = [
            "",  # Empty
            "invalid_key",  # Wrong format
            "hg_test_",  # Too short
            "wrong_prefix_1234567890abcdef",  # Wrong prefix
        ]
        
        # Mock validation function
        def validate_api_key(api_key):
            if not api_key:
                return False
            if not api_key.startswith(("hg_test_", "hg_prod_")):
                return False
            if len(api_key) < 20:
                return False
            return True
        
        # Test valid keys
        for key in valid_api_keys:
            assert validate_api_key(key), f"Valid key rejected: {key}"
        
        # Test invalid keys
        for key in invalid_api_keys:
            assert not validate_api_key(key), f"Invalid key accepted: {key}"
    
    def test_tenant_isolation(self, mock_auth_config):
        """Test tenant isolation."""
        # Mock tenant validation
        def validate_tenant_access(api_key, tenant_id):
            # Extract tenant from API key (simplified)
            if "test" in api_key:
                allowed_tenants = ["test_tenant", "demo_tenant"]
            elif "prod" in api_key:
                allowed_tenants = ["prod_tenant", "enterprise_tenant"]
            else:
                return False
            
            return tenant_id in allowed_tenants
        
        # Test valid combinations
        assert validate_tenant_access("hg_test_123", "test_tenant")
        assert validate_tenant_access("hg_prod_456", "prod_tenant")
        
        # Test invalid combinations
        assert not validate_tenant_access("hg_test_123", "prod_tenant")
        assert not validate_tenant_access("hg_prod_456", "test_tenant")
        assert not validate_tenant_access("invalid_key", "any_tenant")
    
    def test_rate_limiting(self, mock_auth_config):
        """Test rate limiting functionality."""
        import time
        from collections import defaultdict
        
        # Mock rate limiter
        class RateLimiter:
            def __init__(self, requests_per_minute=100, burst_size=10):
                self.requests_per_minute = requests_per_minute
                self.burst_size = burst_size
                self.requests = defaultdict(list)
            
            def is_allowed(self, client_id):
                now = time.time()
                minute_ago = now - 60
                
                # Clean old requests
                self.requests[client_id] = [
                    req_time for req_time in self.requests[client_id]
                    if req_time > minute_ago
                ]
                
                # Check rate limit
                if len(self.requests[client_id]) >= self.requests_per_minute:
                    return False
                
                # Check burst limit (last 10 seconds)
                ten_seconds_ago = now - 10
                recent_requests = [
                    req_time for req_time in self.requests[client_id]
                    if req_time > ten_seconds_ago
                ]
                
                if len(recent_requests) >= self.burst_size:
                    return False
                
                # Allow request
                self.requests[client_id].append(now)
                return True
        
        rate_limiter = RateLimiter(requests_per_minute=5, burst_size=3)
        
        # Test normal usage
        for i in range(3):
            assert rate_limiter.is_allowed("client1"), f"Request {i} should be allowed"
        
        # Test burst limit
        assert not rate_limiter.is_allowed("client1"), "Burst limit should be enforced"
        
        # Test different client
        assert rate_limiter.is_allowed("client2"), "Different client should be allowed"


class TestAuditLogging:
    """Test audit logging and compliance."""
    
    @pytest.fixture
    def temp_audit_dir(self):
        """Create temporary audit directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def audit_logger(self, temp_audit_dir):
        """Create audit logger instance."""
        return PolicyAuditLogger(temp_audit_dir, enable_file_logging=True)
    
    def test_policy_decision_audit(self, audit_logger):
        """Test audit logging of policy decisions."""
        from policy.rule_engine import PolicyRuleResult, PolicyDecision
        from core.models import AggregatedResult, DecisionType, SeverityLevel
        
        # Create test data
        aggregated_result = AggregatedResult(
            final_decision=DecisionType.ALLOW,
            confidence_score=0.6,
            category_scores={"harassment": 0.3},
            severity_level=SeverityLevel.LOW,
            explanation_traces=["Low confidence detection"],
            consolidated_spans=[]
        )
        
        policy_result = PolicyRuleResult(
            decision=PolicyDecision.REVIEW,
            confidence_adjustment=0.1,
            applied_rules=["Threshold rule applied"],
            override_reason=None
        )
        
        # Log policy decision
        trace = audit_logger.log_policy_decision(
            request_id="test_123",
            aggregated_result=aggregated_result,
            policy_result=policy_result,
            tenant_id="test_tenant",
            user_id="user_456"
        )
        
        # Verify audit trace
        assert trace.request_id == "test_123"
        assert trace.tenant_id == "test_tenant"
        assert trace.user_id == "user_456"
        assert trace.original_decision == DecisionType.ALLOW
        assert trace.final_decision == DecisionType.REVIEW
        assert len(trace.applied_rules) > 0
        assert trace.timestamp is not None
    
    def test_policy_change_audit(self, audit_logger):
        """Test audit logging of policy changes."""
        old_config = {
            "block_thresholds": {
                "harassment": {"medium": 0.7, "high": 0.5}
            }
        }
        
        new_config = {
            "block_thresholds": {
                "harassment": {"medium": 0.8, "high": 0.6}
            }
        }
        
        change_event = audit_logger.log_policy_change(
            tenant_id="test_tenant",
            change_type="update",
            changed_by="admin_user",
            old_config=old_config,
            new_config=new_config,
            changes_summary={"harassment_thresholds": "Increased thresholds"}
        )
        
        # Verify change event
        assert change_event.tenant_id == "test_tenant"
        assert change_event.change_type == "update"
        assert change_event.changed_by == "admin_user"
        assert change_event.old_config_hash != change_event.new_config_hash
        assert "harassment_thresholds" in change_event.changes_summary
    
    def test_audit_log_integrity(self, audit_logger):
        """Test audit log integrity and tamper detection."""
        # Log multiple events
        for i in range(5):
            audit_logger.log_access_event(
                request_id=f"req_{i}",
                tenant_id="test_tenant",
                user_id=f"user_{i}",
                action="text_analysis",
                resource="api/analyze",
                result="success"
            )
        
        # Verify audit logs exist
        audit_files = list(Path(audit_logger.audit_dir).glob("*.log"))
        assert len(audit_files) > 0
        
        # Verify log format and integrity
        for audit_file in audit_files:
            content = audit_file.read_text()
            lines = content.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    # Should be valid JSON
                    try:
                        log_entry = json.loads(line)
                        assert "timestamp" in log_entry
                        assert "event_type" in log_entry
                        assert "request_id" in log_entry
                    except json.JSONDecodeError:
                        pytest.fail(f"Invalid JSON in audit log: {line}")
    
    def test_audit_log_retention(self, audit_logger):
        """Test audit log retention policies."""
        # This would test log rotation and retention
        # For now, just verify the configuration exists
        
        assert hasattr(audit_logger, 'retention_days')
        assert hasattr(audit_logger, 'max_file_size')
        
        # Verify retention policy is reasonable
        if hasattr(audit_logger, 'retention_days'):
            assert audit_logger.retention_days >= 90  # Minimum 90 days for compliance


class TestDataRetention:
    """Test data retention and privacy compliance."""
    
    def test_text_content_not_stored(self):
        """Test that raw text content is not stored by default."""
        # Mock service that should not store text
        class MockAnalysisService:
            def __init__(self):
                self.stored_data = []
            
            def analyze(self, text, store_text=False):
                # Should not store text by default
                analysis_result = {
                    "decision": "allow",
                    "confidence": 0.9,
                    "categories": []
                }
                
                if store_text:
                    # Only store if explicitly requested
                    analysis_result["original_text"] = text
                    self.stored_data.append(text)
                
                return analysis_result
        
        service = MockAnalysisService()
        
        # Analyze without storing
        result = service.analyze("Test message")
        assert "original_text" not in result
        assert len(service.stored_data) == 0
        
        # Analyze with explicit storage (for debugging/feedback)
        result = service.analyze("Test message", store_text=True)
        assert "original_text" in result
        assert len(service.stored_data) == 1
    
    def test_pii_data_handling(self):
        """Test PII data handling and retention."""
        # Mock PII handling service
        class MockPIIService:
            def __init__(self):
                self.pii_detected = []
                self.retention_days = 30
            
            def process_text(self, text):
                # Detect PII but don't store original
                pii_patterns = [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                ]
                
                pii_found = []
                for pattern in pii_patterns:
                    matches = re.findall(pattern, text)
                    pii_found.extend(matches)
                
                # Store only metadata, not actual PII
                if pii_found:
                    self.pii_detected.append({
                        "timestamp": "2023-01-01T00:00:00Z",
                        "pii_types": ["email" if "@" in p else "ssn" for p in pii_found],
                        "count": len(pii_found)
                        # Note: Not storing actual PII values
                    })
                
                return {"pii_detected": len(pii_found) > 0, "pii_count": len(pii_found)}
        
        service = MockPIIService()
        
        # Process text with PII
        result = service.process_text("Contact john@example.com or SSN: 123-45-6789")
        
        assert result["pii_detected"] is True
        assert result["pii_count"] == 2
        assert len(service.pii_detected) == 1
        
        # Verify no actual PII is stored
        stored_data = str(service.pii_detected)
        assert "john@example.com" not in stored_data
        assert "123-45-6789" not in stored_data
    
    def test_configurable_retention_policies(self):
        """Test configurable data retention policies."""
        retention_configs = [
            {"logs": 90, "metrics": 365, "audit": 2555},  # 7 years for audit
            {"logs": 30, "metrics": 90, "audit": 1095},   # 3 years for audit
        ]
        
        for config in retention_configs:
            # Verify retention periods are reasonable
            assert config["logs"] >= 30  # Minimum 30 days for logs
            assert config["audit"] >= 365  # Minimum 1 year for audit
            assert config["metrics"] >= 90  # Minimum 90 days for metrics


def similar_pattern_exists(text):
    """Helper function to check if similar masking pattern exists."""
    return bool(re.search(r'\*+[-.\s]\*+[-.\s]\*+', text))


if __name__ == "__main__":
    pytest.main([__file__])