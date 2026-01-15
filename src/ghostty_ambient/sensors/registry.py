"""Sensor backend registry with platform detection."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import SensorBackend


class SensorRegistry:
    """
    Registry for sensor backends with platform detection.

    Supports:
    - Automatic platform detection
    - Manual backend selection
    - Fallback chain
    """

    _backends: dict[str, type[SensorBackend]] = {}
    _instances: dict[str, SensorBackend] = {}

    @classmethod
    def register(cls, backend_class: type[SensorBackend]) -> type[SensorBackend]:
        """Register a backend class (decorator-friendly)."""
        name = backend_class.__name__
        cls._backends[name] = backend_class
        return backend_class

    @classmethod
    def get_for_platform(cls, platform: str | None = None) -> SensorBackend | None:
        """Get the best available backend for the current/specified platform."""
        platform = platform or sys.platform

        # Find all backends for this platform
        candidates = []
        for name in cls._backends:
            try:
                instance = cls._get_instance(name)
                if instance and instance.platform == platform and instance.is_available():
                    candidates.append(instance)
            except Exception:
                pass

        if not candidates:
            return None

        # Return the one with most capabilities
        return max(candidates, key=lambda b: len(b.capabilities))

    @classmethod
    def get_by_name(cls, name: str) -> SensorBackend | None:
        """Get a specific backend by name."""
        return cls._get_instance(name)

    @classmethod
    def _get_instance(cls, name: str) -> SensorBackend | None:
        """Get or create backend instance."""
        if name not in cls._instances and name in cls._backends:
            try:
                cls._instances[name] = cls._backends[name]()
            except Exception:
                return None
        return cls._instances.get(name)

    @classmethod
    def list_backends(cls) -> list[dict]:
        """List all registered backends with status."""
        result = []
        for name in cls._backends:
            try:
                instance = cls._get_instance(name)
                if instance:
                    result.append(
                        {
                            "name": name,
                            "display_name": instance.name,
                            "platform": instance.platform,
                            "available": instance.is_available(),
                            "capabilities": [c.name for c in instance.capabilities],
                        }
                    )
            except Exception:
                pass
        return result

    @classmethod
    def clear(cls) -> None:
        """Clear all registered backends (for testing)."""
        cls._backends.clear()
        cls._instances.clear()
