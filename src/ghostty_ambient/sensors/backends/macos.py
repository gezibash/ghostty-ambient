"""macOS Ambient Light Sensor backend."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..protocol import SensorBackend, SensorCapability, SensorReading
from ..registry import SensorRegistry


@SensorRegistry.register
class MacOSALSBackend(SensorBackend):
    """
    macOS Ambient Light Sensor backend.

    Uses the compiled 'als' binary that reads from IOKit.
    Supports built-in sensors and external displays (Studio Display, etc.).
    """

    def __init__(self, als_path: Path | None = None):
        # Default to als binary in project root
        if als_path is None:
            self._als_path = Path(__file__).parent.parent.parent.parent / "als"
        else:
            self._als_path = als_path
        self._calibration_factor = 1.0

    @property
    def name(self) -> str:
        return "macOS ALS"

    @property
    def platform(self) -> str:
        return "darwin"

    @property
    def capabilities(self) -> set[SensorCapability]:
        return {SensorCapability.LUX}

    def is_available(self) -> bool:
        return self._als_path.exists() and self._als_path.is_file()

    def read(self) -> SensorReading:
        if not self.is_available():
            return SensorReading(error="ALS binary not found")

        try:
            result = subprocess.run(
                [str(self._als_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                raw = float(result.stdout.strip())
                return SensorReading(
                    lux=raw * self._calibration_factor,
                    raw_value=raw,
                    confidence=0.95,
                )
            return SensorReading(
                error=f"als returned {result.returncode}: {result.stderr.strip()}"
            )
        except subprocess.TimeoutExpired:
            return SensorReading(error="Timeout reading ALS")
        except ValueError as e:
            return SensorReading(error=f"Invalid ALS output: {e}")
        except FileNotFoundError:
            return SensorReading(error="ALS binary not found")
        except Exception as e:
            return SensorReading(error=str(e))

    def calibrate(self, known_lux: float) -> bool:
        """Calibrate against a known lux value."""
        reading = self.read()
        if reading.raw_value and reading.raw_value > 0:
            self._calibration_factor = known_lux / reading.raw_value
            return True
        return False
