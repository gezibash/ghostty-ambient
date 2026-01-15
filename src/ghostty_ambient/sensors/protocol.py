"""Sensor backend protocol and data types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable


class SensorCapability(Enum):
    """Capabilities a sensor backend may provide."""

    LUX = auto()  # Ambient light in lux
    COLOR_TEMP = auto()  # Color temperature in Kelvin
    RGB = auto()  # Raw RGB values
    DISPLAY_BRIGHTNESS = auto()  # Current display brightness


@dataclass
class SensorReading:
    """A reading from a sensor backend."""

    lux: float | None = None
    color_temp: int | None = None  # Kelvin
    confidence: float = 1.0
    raw_value: float | None = None
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """Check if the reading is valid (no error and has lux)."""
        return self.error is None and self.lux is not None


@runtime_checkable
class SensorBackend(Protocol):
    """Protocol for ambient light sensor backends."""

    @property
    def name(self) -> str:
        """Human-readable name for this backend."""
        ...

    @property
    def platform(self) -> str:
        """Platform this backend runs on (darwin, linux, win32)."""
        ...

    @property
    def capabilities(self) -> set[SensorCapability]:
        """Set of capabilities this backend provides."""
        ...

    def is_available(self) -> bool:
        """Check if this backend can be used on the current system."""
        ...

    def read(self) -> SensorReading:
        """Take a reading from the sensor. Returns SensorReading with error on failure."""
        ...

    def calibrate(self, known_lux: float) -> bool:
        """Calibrate the sensor against a known lux value. Returns success."""
        ...
