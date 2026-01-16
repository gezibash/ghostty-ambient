"""
Pluggable context factors for Bayesian preference learning.

This module provides a registry-based system for context factors that
influence theme selection. Built-in factors include time, lux, weather,
system appearance, day of week, power source, and font.

To add a custom factor:

    from ghostty_ambient.factors import Factor, FactorRegistry

    class MyFactor(Factor):
        @property
        def name(self) -> str:
            return "my_factor"

        def get_bucket(self, context: dict) -> str:
            # Return bucket based on context
            return "bucket_a"

    FactorRegistry.register(MyFactor())

Or via entry points in pyproject.toml:

    [project.entry-points."ghostty_ambient.factors"]
    my_factor = "my_package.factors:MyFactor"
"""

# Import builtin to trigger auto-registration
from . import builtin  # noqa: F401
from .base import Factor, FactorRegistry

__all__ = ["Factor", "FactorRegistry"]
