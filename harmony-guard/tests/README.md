# Harmony Guard Test Suite

This directory contains the comprehensive test suite for Harmony Guard, covering unit tests, integration tests, performance tests, and security/compliance tests.

## Test Structure

### Test Categories

1. **Unit Tests** (`test_preprocessing.py`, `test_lpe_components.py`, `test_model_inference.py`)
   - Test individual components in isolation
   - Mock external dependencies
   - Focus on core functional logic
   - Fast execution (< 1 second per test)

2. **Integration Tests** (`test_integration.py`)
   - Test complete analysis pipeline
   - Test component interactions
   - End-to-end workflow validation
   - Golden dataset validation

3. **Performance Tests** (`test_performance.py`)
   - Latency requirements (P50 ≤ 25ms, P95 ≤ 80ms)
   - Throughput requirements (≥200 RPS per pod)
   - Memory usage and resource efficiency
   - Concurrent processing validation

4. **Security & Compliance Tests** (`test_security_compliance.py`)
   - PII masking and privacy compliance
   - Log redaction and secure logging
   - Input validation and injection prevention
   - Authentication and authorization
   - Audit logging and data retention

### Test Files

- `conftest.py` - Shared fixtures and pytest configuration
- `test_preprocessing.py` - Text preprocessing pipeline tests
- `test_lpe_components.py` - Lexicon & Pattern Engine tests
- `test_model_inference.py` - Model inference and aggregation tests
- `test_integration.py` - Integration and end-to-end tests
- `test_performance.py` - Performance and latency tests
- `test_security_compliance.py` - Security and compliance tests

## Running Tests

### Prerequisites

```bash
pip install pytest pytest-asyncio pytest-cov
```

### Basic Usage

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_preprocessing.py

# Run specific test class
python -m pytest tests/test_preprocessing.py::TestTextNormalizer

# Run specific test method
python -m pytest tests/test_preprocessing.py::TestTextNormalizer::test_unicode_normalization
```

### Test Categories

```bash
# Run only unit tests
python -m pytest -m unit

# Run only integration tests
python -m pytest -m integration

# Run only performance tests
python -m pytest -m performance

# Run only security tests
python -m pytest -m security

# Exclude slow tests
python -m pytest -m "not slow"
```

### Coverage Reports

```bash
# Run with coverage
python -m pytest --cov=. --cov-report=html --cov-report=term

# View HTML coverage report
open htmlcov/index.html
```

### Using Test Runner Script

```bash
# Run all tests with summary
python run_tests.py --all

# Run specific categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --performance
python run_tests.py --security

# Run with coverage
python run_tests.py --all --coverage

# Verbose output
python run_tests.py --all --verbose
```

## Test Design Principles

### Unit Tests

- **Isolation**: Each test is independent and can run in any order
- **Mocking**: External dependencies are mocked to focus on unit logic
- **Fast**: Tests complete quickly (< 1 second each)
- **Focused**: Each test validates a single piece of functionality
- **Deterministic**: Tests produce consistent results

### Integration Tests

- **Realistic**: Use realistic data and scenarios
- **End-to-End**: Test complete workflows from input to output
- **Error Handling**: Validate graceful degradation and error recovery
- **Multilingual**: Test with various languages and scripts
- **Golden Dataset**: Validate against known good/bad examples

### Performance Tests

- **Latency**: Measure and validate response times
- **Throughput**: Test sustained request rates
- **Concurrency**: Validate behavior under concurrent load
- **Resource Usage**: Monitor memory and CPU consumption
- **Scalability**: Test performance characteristics at scale

### Security Tests

- **PII Protection**: Validate PII masking and redaction
- **Input Validation**: Test against malicious inputs
- **Authentication**: Validate API key and tenant isolation
- **Audit Logging**: Ensure compliance with audit requirements
- **Data Retention**: Validate privacy-compliant data handling

## Test Data

### Sample Texts

The test suite uses various categories of sample text:

- **Clean Text**: Appropriate corporate communications
- **Problematic Text**: Content that should be flagged/blocked
- **Multilingual Text**: Hindi, English, and code-mixed content
- **Obfuscated Text**: Leet speak and other obfuscation techniques
- **Edge Cases**: Empty text, very long text, special characters

### Mock Data

- **Language Detection**: Mocked to return consistent results
- **Model Predictions**: Mocked to simulate various confidence levels
- **Configuration**: Mocked to test different policy settings
- **External Services**: All external dependencies are mocked

## Continuous Integration

### GitHub Actions

The test suite is designed to run in CI/CD pipelines:

```yaml
- name: Run Tests
  run: |
    python -m pytest tests/ -v --cov=. --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

### Test Markers

Tests are marked for selective execution:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.security` - Security tests
- `@pytest.mark.slow` - Slow-running tests

### Parallel Execution

For faster CI runs:

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
python -m pytest -n auto
```

## Debugging Tests

### Failed Test Investigation

```bash
# Run with detailed output
python -m pytest --tb=long -v

# Run single failing test
python -m pytest tests/test_file.py::test_method -s

# Drop into debugger on failure
python -m pytest --pdb
```

### Test Coverage Analysis

```bash
# Generate coverage report
python -m pytest --cov=. --cov-report=html

# View missing coverage
python -m pytest --cov=. --cov-report=term-missing
```

## Adding New Tests

### Test File Structure

```python
"""Test description."""

import pytest
from unittest.mock import Mock, patch

# Test class
class TestComponentName:
    """Test component functionality."""
    
    @pytest.fixture
    def component_config(self):
        """Configuration for component."""
        return {"setting": "value"}
    
    @pytest.fixture
    def component(self, component_config):
        """Create component instance."""
        return ComponentClass(component_config)
    
    def test_basic_functionality(self, component):
        """Test basic functionality."""
        result = component.method("input")
        assert result == "expected"
    
    @pytest.mark.asyncio
    async def test_async_functionality(self, component):
        """Test async functionality."""
        result = await component.async_method("input")
        assert result == "expected"
```

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Fixtures: Descriptive names without `test_` prefix

### Assertions

Use descriptive assertions:

```python
# Good
assert result.confidence > 0.8, f"Confidence {result.confidence} too low"

# Better
assert result.decision == DecisionType.BLOCK, \
    f"Expected BLOCK but got {result.decision} for text: {text}"
```

## Performance Benchmarks

### Target Metrics

- **P50 Latency**: ≤ 25ms
- **P95 Latency**: ≤ 80ms  
- **Throughput**: ≥ 200 RPS per pod
- **Memory Usage**: ≤ 4GB per pod
- **CPU Usage**: ≤ 80% at target RPS

### Measurement

Performance tests measure:
- End-to-end request latency
- Component-level processing time
- Memory consumption over time
- Concurrent request handling
- Resource utilization under load

## Compliance Validation

### Privacy Requirements

- PII is masked in logs and responses
- Raw text is not stored by default
- Audit logs maintain data integrity
- Retention policies are enforced

### Security Requirements

- Input validation prevents injection attacks
- Authentication and authorization work correctly
- Rate limiting prevents abuse
- Sensitive data is properly protected

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes project root
2. **Async Warnings**: Install `pytest-asyncio`
3. **Mock Issues**: Verify mock paths match actual imports
4. **Fixture Scope**: Use appropriate fixture scopes for performance

### Getting Help

- Check test output for detailed error messages
- Use `pytest --tb=long` for full tracebacks
- Run individual tests to isolate issues
- Review mock configurations for external dependencies