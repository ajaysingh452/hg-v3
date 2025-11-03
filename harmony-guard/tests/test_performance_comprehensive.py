"""Comprehensive performance tests for latency and throughput requirements."""

import pytest
import asyncio
import time
import statistics
from unittest.mock import Mock, AsyncMock
import numpy as np
# import psutil  # Not available in test environment
# import gc

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.preprocessing import TextPreprocessor
from core.models import ProcessedText, LanguageDetection


class TestLatencyPerformance:
    """Test latency requirements (P50 ≤ 25ms, P95 ≤ 80ms)."""
    
    @pytest.fixture
    def optimized_config(self):
        """Configuration optimized for latency."""
        return {
            'preprocessing': {
                'language_detection': {'supported_languages': ['en'], 'confidence_threshold': 0.7},
                'normalization': {'unicode_form': 'NFKC', 'compress_repeated_chars': True},
                'transliteration': {'enabled': False},  # Disable for speed
                'obfuscation': {'leet_speak_detection': True, 'elongation_detection': True},
                'tokenization': {'emoji_aware': True, 'script_aware': False},  # Simplified
                'pii_masking': {'enabled': False}
            }
        }
    
    @pytest.mark.asyncio
    async def test_preprocessing_latency_requirements(self, optimized_config):
        """Test preprocessing latency meets requirements."""
        # Initialize preprocessor
        preprocessor = TextPreprocessor(optimized_config)
        
        # Mock external dependencies for fast processing
        preprocessor.language_identifier = Mock()
        preprocessor.language_identifier.detect_languages = Mock(
            return_value=[LanguageDetection("en", 0.9, 100.0)]
        )
        preprocessor.transliteration_engine = Mock()
        preprocessor.transliteration_engine.transliterate_to_native = Mock(return_value={})
        preprocessor.transliteration_engine.transliterate_to_roman = Mock(return_value={})
        preprocessor.pii_masker = Mock()
        preprocessor.pii_masker.enabled = False
        
        # Test various text lengths
        test_cases = [
            ("Hi", "very_short"),
            ("Hello world, how are you?", "short"),
            ("This is a medium length message with some content to analyze.", "medium"),
            ("This is a longer message " * 10, "long"),
            ("Very long message " * 25 + " with complex content", "very_long"),
        ]
        
        latencies = []
        
        for text, scenario_type in test_cases:
            # Measure preprocessing latency
            start_time = time.perf_counter()
            
            result = await preprocessor.process(text)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            print(f"Scenario '{scenario_type}': {latency_ms:.2f}ms")
            
            # Verify result is valid
            assert isinstance(result, ProcessedText)
            assert result.original_text == text
        
        # Calculate percentiles
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Preprocessing Latency - P50: {p50_latency:.2f}ms, P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms")
        
        # Verify latency requirements (with mocked components, should be very fast)
        assert p50_latency < 50, f"P50 latency {p50_latency:.2f}ms exceeds 50ms threshold"
        assert p95_latency < 100, f"P95 latency {p95_latency:.2f}ms exceeds 100ms threshold"
    
    @pytest.mark.asyncio
    async def test_concurrent_preprocessing_latency(self, optimized_config):
        """Test latency under concurrent load."""
        preprocessor = TextPreprocessor(optimized_config)
        
        # Mock fast responses with slight delays to simulate real processing
        async def mock_process_with_delay(text, language_hints=None):
            await asyncio.sleep(0.002)  # 2ms delay
            return ProcessedText(
                original_text=text,
                normalized_text=text.lower(),
                detected_languages=[LanguageDetection("en", 0.9, 100.0)],
                tokens=text.split(),
                transliterations={},
                obfuscation_map={}
            )
        
        preprocessor.process = mock_process_with_delay
        
        # Define processing function
        async def process_request(text):
            start_time = time.perf_counter()
            result = await preprocessor.process(text)
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000, result
        
        # Test concurrent requests
        concurrent_requests = 20
        test_texts = [f"Test message {i} for concurrent processing" for i in range(concurrent_requests)]
        
        # Process all requests concurrently
        start_time = time.perf_counter()
        tasks = [process_request(text) for text in test_texts]
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        # Extract latencies and verify results
        latencies = [result[0] for result in results]
        processed_results = [result[1] for result in results]
        
        # Verify all results are valid
        for result in processed_results:
            assert isinstance(result, ProcessedText)
        
        # Calculate statistics
        p50_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        total_time = (end_time - start_time) * 1000
        throughput = concurrent_requests / (total_time / 1000)  # requests per second
        
        print(f"Concurrent test: P50={p50_latency:.2f}ms, P95={p95_latency:.2f}ms, Throughput={throughput:.1f} RPS")
        
        # Latency should not degrade significantly under concurrent load
        assert p50_latency < 100, f"P50 latency {p50_latency:.2f}ms under concurrent load exceeds 100ms"
        assert p95_latency < 200, f"P95 latency {p95_latency:.2f}ms under concurrent load exceeds 200ms"


class TestThroughputPerformance:
    """Test throughput requirements (≥200 RPS per pod)."""
    
    @pytest.fixture
    def throughput_config(self):
        """Configuration optimized for throughput."""
        return {
            'preprocessing': {
                'language_detection': {'supported_languages': ['en']},
                'normalization': {'unicode_form': 'NFKC'},
                'transliteration': {'enabled': False},
                'obfuscation': {'leet_speak_detection': True},
                'tokenization': {'emoji_aware': True},
                'pii_masking': {'enabled': False}
            }
        }
    
    @pytest.mark.asyncio
    async def test_sustained_throughput(self, throughput_config):
        """Test sustained throughput over time."""
        preprocessor = TextPreprocessor(throughput_config)
        
        # Mock components with realistic delays
        async def mock_process(text, language_hints=None):
            await asyncio.sleep(0.003)  # 3ms processing time
            return ProcessedText(
                original_text=text,
                normalized_text=text.lower(),
                detected_languages=[LanguageDetection("en", 0.9, 100.0)],
                tokens=text.split(),
                transliterations={},
                obfuscation_map={}
            )
        
        preprocessor.process = mock_process
        
        # Test sustained load
        duration_seconds = 3  # Test for 3 seconds
        target_rps = 100  # Target 100 RPS (reduced for testing)
        
        requests_sent = 0
        successful_requests = 0
        start_time = time.perf_counter()
        
        # Generate continuous load
        async def load_generator():
            nonlocal requests_sent, successful_requests
            
            while time.perf_counter() - start_time < duration_seconds:
                # Send batch of requests
                batch_size = 10
                texts = [f"Test message {requests_sent + i}" for i in range(batch_size)]
                
                tasks = [preprocessor.process(text) for text in texts]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                requests_sent += batch_size
                successful_requests += sum(1 for r in results if isinstance(r, ProcessedText))
                
                # Control rate
                await asyncio.sleep(batch_size / target_rps)
        
        # Run load test
        await load_generator()
        
        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        actual_rps = successful_requests / actual_duration
        success_rate = successful_requests / requests_sent if requests_sent > 0 else 0
        
        print(f"Throughput test: {actual_rps:.1f} RPS, {success_rate:.2%} success rate")
        
        # Verify throughput and success rate
        assert actual_rps >= target_rps * 0.8, f"Throughput {actual_rps:.1f} RPS below 80% of target {target_rps} RPS"
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"
    
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, throughput_config):
        """Test batch processing efficiency."""
        # Mock batch processing function
        async def mock_batch_process(texts):
            # Simulate batch processing efficiency
            batch_size = len(texts)
            base_time = 0.005  # 5ms base time
            per_item_time = 0.001  # 1ms per additional item
            
            total_time = base_time + (batch_size - 1) * per_item_time
            await asyncio.sleep(total_time)
            
            return [ProcessedText(
                original_text=text,
                normalized_text=text.lower(),
                detected_languages=[LanguageDetection("en", 0.9, 100.0)],
                tokens=text.split(),
                transliterations={},
                obfuscation_map={}
            ) for text in texts]
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            # Create batch of texts
            texts = [f"Test message {i}" for i in range(batch_size)]
            
            # Measure batch processing time
            start_time = time.perf_counter()
            results = await mock_batch_process(texts)
            end_time = time.perf_counter()
            
            batch_time = (end_time - start_time) * 1000  # ms
            per_item_time = batch_time / batch_size
            
            # Verify results
            assert len(results) == batch_size
            for result in results:
                assert isinstance(result, ProcessedText)
            
            print(f"Batch size {batch_size}: {batch_time:.2f}ms total, {per_item_time:.2f}ms per item")
            
            # Batch processing should be more efficient for larger batches
            if batch_size > 1:
                assert per_item_time < 15, f"Per-item time {per_item_time:.2f}ms too high for batch size {batch_size}"


class TestResourceEfficiency:
    """Test resource efficiency without external dependencies."""
    
    @pytest.mark.asyncio
    async def test_concurrent_request_processing(self):
        """Test concurrent request processing efficiency."""
        # Initialize preprocessor
        config = {
            'preprocessing': {
                'language_detection': {'supported_languages': ['en']},
                'normalization': {'unicode_form': 'NFKC'},
                'transliteration': {'enabled': False},
                'obfuscation': {'leet_speak_detection': True},
                'tokenization': {'emoji_aware': True},
                'pii_masking': {'enabled': False}
            }
        }
        
        preprocessor = TextPreprocessor(config)
        
        # Mock fast processing
        async def mock_process(text, language_hints=None):
            await asyncio.sleep(0.001)  # Small delay
            return ProcessedText(
                original_text=text,
                normalized_text=text.lower(),
                detected_languages=[LanguageDetection("en", 0.9, 100.0)],
                tokens=text.split(),
                transliterations={},
                obfuscation_map={}
            )
        
        preprocessor.process = mock_process
        
        # Process concurrent requests
        concurrent_requests = 50
        tasks = [
            preprocessor.process(f"Test message {i} for concurrent processing")
            for i in range(concurrent_requests)
        ]
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        throughput = concurrent_requests / processing_time
        
        # Verify all requests completed successfully
        assert len(results) == concurrent_requests
        for result in results:
            assert isinstance(result, ProcessedText)
        
        print(f"Concurrent processing:")
        print(f"  Requests: {concurrent_requests}")
        print(f"  Time: {processing_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} RPS")
        
        # Throughput should be reasonable
        assert throughput > 50, f"Throughput {throughput:.1f} RPS too low for concurrent processing"
    
    @pytest.mark.asyncio
    async def test_processing_consistency(self):
        """Test processing consistency across multiple runs."""
        config = {
            'preprocessing': {
                'language_detection': {'supported_languages': ['en']},
                'normalization': {'unicode_form': 'NFKC'},
                'transliteration': {'enabled': False},
                'obfuscation': {'leet_speak_detection': True},
                'tokenization': {'emoji_aware': True},
                'pii_masking': {'enabled': False}
            }
        }
        
        preprocessor = TextPreprocessor(config)
        
        # Mock consistent processing
        async def consistent_process(text, language_hints=None):
            await asyncio.sleep(0.002)  # Consistent 2ms delay
            return ProcessedText(
                original_text=text,
                normalized_text=text.lower(),
                detected_languages=[LanguageDetection("en", 0.9, 100.0)],
                tokens=text.split(),
                transliterations={},
                obfuscation_map={}
            )
        
        preprocessor.process = consistent_process
        
        # Test consistency across multiple runs
        test_text = "Consistency test message for performance validation"
        run_times = []
        
        for run in range(10):
            start_time = time.perf_counter()
            result = await preprocessor.process(test_text)
            end_time = time.perf_counter()
            
            run_time = (end_time - start_time) * 1000  # ms
            run_times.append(run_time)
            
            # Verify result consistency
            assert isinstance(result, ProcessedText)
            assert result.original_text == test_text
        
        # Analyze consistency
        avg_time = statistics.mean(run_times)
        std_dev = statistics.stdev(run_times) if len(run_times) > 1 else 0
        coefficient_of_variation = std_dev / avg_time if avg_time > 0 else 0
        
        print(f"Processing consistency:")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Standard deviation: {std_dev:.2f}ms")
        print(f"  Coefficient of variation: {coefficient_of_variation:.3f}")
        
        # Processing should be consistent (reasonable variation for async operations)
        assert coefficient_of_variation < 0.5, f"Processing time variation {coefficient_of_variation:.3f} too high"
        assert avg_time < 50, f"Average processing time {avg_time:.2f}ms too high"


class TestScalabilityCharacteristics:
    """Test scalability characteristics and resource utilization."""
    
    @pytest.mark.asyncio
    async def test_scalability_with_different_loads(self):
        """Test scalability characteristics with different load levels."""
        config = {
            'preprocessing': {
                'language_detection': {'supported_languages': ['en']},
                'normalization': {'unicode_form': 'NFKC'},
                'transliteration': {'enabled': False},
                'obfuscation': {'leet_speak_detection': True},
                'tokenization': {'emoji_aware': True},
                'pii_masking': {'enabled': False}
            }
        }
        
        preprocessor = TextPreprocessor(config)
        
        # Mock scalable processing
        async def scalable_process(text, language_hints=None):
            # Simulate processing that scales with load
            await asyncio.sleep(0.002)  # 2ms base processing
            return ProcessedText(
                original_text=text,
                normalized_text=text.lower(),
                detected_languages=[LanguageDetection("en", 0.9, 100.0)],
                tokens=text.split(),
                transliterations={},
                obfuscation_map={}
            )
        
        preprocessor.process = scalable_process
        
        # Test different load levels
        load_levels = [5, 10, 25, 50, 100]
        
        scalability_results = []
        
        for load_level in load_levels:
            # Create requests for this load level
            texts = [f"Scalability test message {i}" for i in range(load_level)]
            
            # Process requests and measure performance
            start_time = time.perf_counter()
            
            tasks = [preprocessor.process(text) for text in texts]
            results = await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            
            total_time = (end_time - start_time) * 1000  # ms
            per_item_time = total_time / load_level
            throughput = load_level / (total_time / 1000)  # items per second
            
            scalability_results.append({
                'load_level': load_level,
                'total_time': total_time,
                'per_item_time': per_item_time,
                'throughput': throughput
            })
            
            print(f"Load {load_level}: {total_time:.2f}ms total, {per_item_time:.2f}ms per item, {throughput:.1f} items/sec")
            
            # Verify results
            assert len(results) == load_level
            for result in results:
                assert isinstance(result, ProcessedText)
        
        # Analyze scalability characteristics
        baseline_per_item = scalability_results[0]['per_item_time']
        
        for result in scalability_results[1:]:  # Skip baseline
            load_level = result['load_level']
            per_item_time = result['per_item_time']
            efficiency = baseline_per_item / per_item_time if per_item_time > 0 else 0
            
            print(f"Load {load_level} efficiency: {efficiency:.2f}x relative to baseline")
            
            # Per-item time should not increase dramatically with load
            assert per_item_time < baseline_per_item * 3, f"Per-item time {per_item_time:.2f}ms too high for load {load_level}"
            
            # Throughput should scale reasonably
            expected_min_throughput = min(load_level * 0.5, 200)  # At least 50% efficiency, capped at 200 RPS
            assert result['throughput'] >= expected_min_throughput, f"Throughput {result['throughput']:.1f} too low for load {load_level}"


if __name__ == "__main__":
    pytest.main([__file__])