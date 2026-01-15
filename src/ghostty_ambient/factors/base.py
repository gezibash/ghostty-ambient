"""Base classes for pluggable context factors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Factor(ABC):
    """
    Base class for context factors used in Bayesian preference learning.

    Factors represent independent dimensions of context that influence
    theme preferences (e.g., time of day, ambient light, system appearance).

    To create a custom factor:

        class MyFactor(Factor):
            @property
            def name(self) -> str:
                return "my_factor"

            def get_bucket(self, context: dict[str, Any]) -> str:
                value = context.get("my_key")
                if value is None:
                    return "unknown"
                return "bucket_a" if value > 50 else "bucket_b"

        FactorRegistry.register(MyFactor())
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique factor identifier (e.g., 'time', 'lux', 'system')."""

    @abstractmethod
    def get_bucket(self, context: dict[str, Any]) -> str:
        """
        Map context to a bucket string.

        Args:
            context: Dict with keys like 'hour', 'lux', 'system_appearance', etc.

        Returns:
            Bucket name (e.g., 'morning', 'dim', 'dark').
            Return 'unknown' if data unavailable.
        """

    @property
    def weight_multiplier(self) -> float:
        """
        Weight multiplier for Bayesian scoring.

        Factors with higher multipliers have more influence on theme selection.
        Default is 1.0. For example, system appearance uses 2.0 since it's
        a deliberate user choice.
        """
        return 1.0

    @property
    def required_context_keys(self) -> set[str]:
        """Context keys needed by this factor (for validation/documentation)."""
        return set()


class FactorRegistry:
    """
    Registry for Factor implementations.

    Factors are registered at module load time and can be retrieved
    for Bayesian scoring calculations.

    Example:
        # Register a factor
        FactorRegistry.register(TimeFactor())

        # Get all buckets for current context
        context = {"hour": 14, "lux": 300, "system_appearance": "dark"}
        buckets = FactorRegistry.get_all_buckets(context)
        # {"time": "afternoon", "lux": "office", "system": "dark", ...}
    """

    _factors: dict[str, Factor] = {}

    @classmethod
    def register(cls, factor: Factor) -> Factor:
        """
        Register a factor instance.

        Args:
            factor: Factor instance to register

        Returns:
            The same factor instance (for decorator-style usage)
        """
        cls._factors[factor.name] = factor
        return factor

    @classmethod
    def get(cls, name: str) -> Factor | None:
        """Get factor by name."""
        return cls._factors.get(name)

    @classmethod
    def all(cls) -> list[Factor]:
        """Get all registered factors."""
        return list(cls._factors.values())

    @classmethod
    def get_all_buckets(cls, context: dict[str, Any]) -> dict[str, str]:
        """
        Get buckets from all registered factors.

        Args:
            context: Context dict with keys like 'hour', 'lux', etc.

        Returns:
            Dict mapping factor name to bucket string
        """
        return {f.name: f.get_bucket(context) for f in cls._factors.values()}

    @classmethod
    def clear(cls) -> None:
        """Clear registry (for testing)."""
        cls._factors.clear()
