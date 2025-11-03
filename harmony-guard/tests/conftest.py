"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager for tests."""
    config_manager = Mock()
    config_manager.get_ensemble_config.return_value = {
        'preprocessing': {
            'language_detection': {'supported_languages': ['en', 'hi']},
            'normalization': {'unicode_form': 'NFKC'},
            'transliteration': {'enabled': True},
            'obfuscation': {'leet_speak_detection': True},
            'tokenization': {'emoji_aware': True},
            'pii_masking': {'enabled': False}
        },
        'lpe': {'fuzzy_matching': True, 'fuzzy_threshold': 0.8},
        'classifier': {'model_name': 'test-model'},
        'intent': {'negation_detection': True},
        'ensemble': {
            'weights': {'lpe': 0.4, 'classifier': 0.5, 'intent': 0.1},
            'thresholds': {
                'confidence_minimum': 0.6,
                'review_threshold': 0.7,
                'block_threshold': 0.85
            }
        }
    }
    return config_manager


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return {
        'clean': [
            "Hello world, nice to meet you!",
            "Great job on the presentation.",
            "Looking forward to our meeting tomorrow."
        ],
        'problematic': [
            "This is fucking terrible shit",
            "You are such an idiot and stupid person",
            "I hate all people from that country"
        ],
        'multilingual': [
            "Hello नमस्ते world",
            "Yaar, main office ja raha hai",
            "Good morning सुप्रभात"
        ],
        'obfuscated': [
            "h3ll0 w0rld",
            "th1s 1s b@d",
            "fuuuuuck th1s sh1t"
        ]
    }


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Skip tests that require external dependencies in CI
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add markers based on test file names
        if "test_preprocessing" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_lpe_components" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_model_inference" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "test_security_compliance" in str(item.fspath):
            item.add_marker(pytest.mark.security)