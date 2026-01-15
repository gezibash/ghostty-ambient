"""Tests for lib/factors/base.py - Factor ABC and FactorRegistry."""

from __future__ import annotations

import pytest

from ghostty_ambient.factors.base import Factor, FactorRegistry


class MockFactor(Factor):
    """Mock factor for testing."""

    def __init__(self, name: str = "mock", weight: float = 1.0):
        self._name = name
        self._weight = weight

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight_multiplier(self) -> float:
        return self._weight

    def get_bucket(self, context: dict) -> str:
        value = context.get("value")
        if value is None:
            return "unknown"
        return "high" if value > 50 else "low"


class TestFactor:
    """Tests for Factor ABC."""

    def test_abstract_methods_required(self):
        """Factor subclass must implement abstract methods."""
        with pytest.raises(TypeError):
            # Can't instantiate abstract class
            Factor()  # type: ignore

    def test_default_weight_multiplier(self):
        """Default weight_multiplier should be 1.0."""
        factor = MockFactor()
        assert factor.weight_multiplier == 1.0

    def test_custom_weight_multiplier(self):
        """Custom weight_multiplier should be respected."""
        factor = MockFactor(weight=2.0)
        assert factor.weight_multiplier == 2.0

    def test_default_required_context_keys(self):
        """Default required_context_keys should be empty set."""
        factor = MockFactor()
        assert factor.required_context_keys == set()

    def test_get_bucket_with_value(self):
        """get_bucket should return appropriate bucket."""
        factor = MockFactor()
        assert factor.get_bucket({"value": 100}) == "high"
        assert factor.get_bucket({"value": 25}) == "low"

    def test_get_bucket_without_value(self):
        """get_bucket should return 'unknown' for missing value."""
        factor = MockFactor()
        assert factor.get_bucket({}) == "unknown"


class TestFactorRegistry:
    """Tests for FactorRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        # Save current state
        self._saved_factors = FactorRegistry._factors.copy()
        FactorRegistry.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        FactorRegistry._factors = self._saved_factors

    def test_register_factor(self):
        """register() should add factor to registry."""
        factor = MockFactor("test_factor")
        FactorRegistry.register(factor)
        assert FactorRegistry.get("test_factor") is factor

    def test_register_returns_factor(self):
        """register() should return the factor for decorator usage."""
        factor = MockFactor("test_factor")
        result = FactorRegistry.register(factor)
        assert result is factor

    def test_get_nonexistent_factor(self):
        """get() should return None for nonexistent factor."""
        assert FactorRegistry.get("nonexistent") is None

    def test_all_returns_list(self):
        """all() should return list of registered factors."""
        f1 = MockFactor("factor1")
        f2 = MockFactor("factor2")
        FactorRegistry.register(f1)
        FactorRegistry.register(f2)

        factors = FactorRegistry.all()
        assert len(factors) == 2
        assert f1 in factors
        assert f2 in factors

    def test_get_all_buckets(self):
        """get_all_buckets() should return buckets for all factors."""
        FactorRegistry.register(MockFactor("factor1"))
        FactorRegistry.register(MockFactor("factor2"))

        buckets = FactorRegistry.get_all_buckets({"value": 75})
        assert buckets == {"factor1": "high", "factor2": "high"}

    def test_clear_empties_registry(self):
        """clear() should remove all factors."""
        FactorRegistry.register(MockFactor("test"))
        assert len(FactorRegistry.all()) > 0

        FactorRegistry.clear()
        assert len(FactorRegistry.all()) == 0
