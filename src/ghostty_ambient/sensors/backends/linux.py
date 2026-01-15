"""Linux sysfs-based ambient light sensor backend."""

from __future__ import annotations

import glob
from pathlib import Path

from ..protocol import SensorBackend, SensorCapability, SensorReading
from ..registry import SensorRegistry


@SensorRegistry.register
class LinuxSysfsBackend(SensorBackend):
    """
    Linux sysfs-based ambient light sensor backend.

    Reads from /sys/bus/iio/devices/*/in_illuminance_raw or similar paths.
    """

    SYSFS_PATHS = [
        "/sys/bus/iio/devices/iio:device*/in_illuminance_raw",
        "/sys/bus/iio/devices/iio:device*/in_illuminance_input",
        "/sys/bus/acpi/devices/ACPI0008:00/iio:device*/in_illuminance_raw",
    ]

    def __init__(self):
        self._device_path: Path | None = None
        self._scale = 1.0
        self._find_device()

    def _find_device(self) -> None:
        """Find an available ALS device in sysfs."""
        for pattern in self.SYSFS_PATHS:
            for path in glob.glob(pattern):
                if Path(path).exists():
                    self._device_path = Path(path)
                    # Try to read scale factor
                    scale_path = self._device_path.parent / "in_illuminance_scale"
                    if scale_path.exists():
                        try:
                            self._scale = float(scale_path.read_text().strip())
                        except (ValueError, OSError):
                            pass
                    return

    @property
    def name(self) -> str:
        return "Linux sysfs"

    @property
    def platform(self) -> str:
        return "linux"

    @property
    def capabilities(self) -> set[SensorCapability]:
        return {SensorCapability.LUX}

    def is_available(self) -> bool:
        return self._device_path is not None and self._device_path.exists()

    def read(self) -> SensorReading:
        if not self._device_path:
            return SensorReading(error="No ALS device found")

        try:
            raw = float(self._device_path.read_text().strip())
            return SensorReading(
                lux=raw * self._scale,
                raw_value=raw,
                confidence=0.85,
            )
        except (ValueError, OSError) as e:
            return SensorReading(error=str(e))

    def calibrate(self, known_lux: float) -> bool:
        """Calibrate against a known lux value."""
        reading = self.read()
        if reading.raw_value and reading.raw_value > 0:
            self._scale = known_lux / reading.raw_value
            return True
        return False
