"""Sensor abstraction layer with plugin support."""

from __future__ import annotations

from .loader import discover_backends, get_best_backend, load_plugins
from .protocol import SensorBackend, SensorCapability, SensorReading
from .registry import SensorRegistry

__all__ = [
    "SensorBackend",
    "SensorCapability",
    "SensorReading",
    "SensorRegistry",
    "discover_backends",
    "get_best_backend",
    "load_plugins",
]
