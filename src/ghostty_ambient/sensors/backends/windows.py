"""Windows sensor API backend."""

from __future__ import annotations

from ..protocol import SensorBackend, SensorCapability, SensorReading
from ..registry import SensorRegistry


@SensorRegistry.register
class WindowsSensorBackend(SensorBackend):
    """
    Windows sensor API backend.

    Uses Windows.Devices.Sensors via winsdk package.
    Requires: pip install winsdk
    """

    def __init__(self):
        self._sensor = None
        self._initialized = False

    def _init_sensor(self) -> None:
        """Lazy initialization of the sensor."""
        if self._initialized:
            return
        self._initialized = True

        try:
            # Try winsdk approach (Windows 10+)
            from winsdk.windows.devices.sensors import LightSensor

            self._sensor = LightSensor.get_default()
        except ImportError:
            pass
        except Exception:
            pass

    @property
    def name(self) -> str:
        return "Windows Sensors"

    @property
    def platform(self) -> str:
        return "win32"

    @property
    def capabilities(self) -> set[SensorCapability]:
        return {SensorCapability.LUX}

    def is_available(self) -> bool:
        self._init_sensor()
        return self._sensor is not None

    def read(self) -> SensorReading:
        self._init_sensor()

        if not self._sensor:
            return SensorReading(error="Light sensor not available")

        try:
            reading = self._sensor.get_current_reading()
            if reading:
                return SensorReading(
                    lux=reading.illuminance_in_lux,
                    confidence=0.9,
                )
            return SensorReading(error="No reading available")
        except Exception as e:
            return SensorReading(error=str(e))

    def calibrate(self, known_lux: float) -> bool:
        """Windows API doesn't support calibration."""
        return False
