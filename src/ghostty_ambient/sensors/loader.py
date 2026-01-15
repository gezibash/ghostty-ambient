"""Plugin loading for sensor backends."""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

from .registry import SensorRegistry

if TYPE_CHECKING:
    from .protocol import SensorBackend

ENTRY_POINT_GROUP = "ghostty_ambient.sensors"


def load_plugins() -> None:
    """
    Load sensor plugins from entry points.

    Plugins can register via pyproject.toml:

    [project.entry-points."ghostty_ambient.sensors"]
    my_sensor = "my_package.sensors:MySensorBackend"
    """
    try:
        eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        for ep in eps:
            try:
                backend_class = ep.load()
                # Check if it looks like a SensorBackend
                if (
                    isinstance(backend_class, type)
                    and hasattr(backend_class, "name")
                    and hasattr(backend_class, "is_available")
                    and hasattr(backend_class, "read")
                ):
                    SensorRegistry.register(backend_class)
            except Exception:
                pass  # Skip broken plugins
    except Exception:
        pass  # No plugins or metadata API issues


def load_builtin_backends() -> None:
    """Load the built-in sensor backends."""
    # Import backends to trigger registration
    from . import backends  # noqa: F401


def discover_backends() -> list[dict]:
    """Discover and list all available backends."""
    # Load built-in backends
    load_builtin_backends()

    # Load plugins
    load_plugins()

    return SensorRegistry.list_backends()


def get_best_backend() -> SensorBackend | None:
    """Get the best available backend for the current platform."""
    # Ensure backends are loaded
    load_builtin_backends()
    load_plugins()

    return SensorRegistry.get_for_platform()
