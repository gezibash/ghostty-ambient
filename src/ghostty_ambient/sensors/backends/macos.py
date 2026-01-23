"""macOS Ambient Light Sensor backend."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from importlib import resources
from pathlib import Path

from ..protocol import SensorBackend, SensorCapability, SensorReading
from ..registry import SensorRegistry

# Cache directory for compiled binary
CACHE_DIR = Path.home() / ".cache" / "ghostty-ambient"


def _get_source_hash() -> str:
    """Get hash of the als.m source for cache invalidation."""
    try:
        # Use importlib.resources to read the bundled source
        source_file = resources.files("ghostty_ambient.sensors").joinpath("als.m")
        content = source_file.read_bytes()
        return hashlib.sha256(content).hexdigest()[:12]
    except Exception:
        return "unknown"


def _get_als_binary_path() -> Path:
    """Get path to the compiled als binary, compiling if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Include source hash in binary name for cache invalidation
    source_hash = _get_source_hash()
    binary_path = CACHE_DIR / f"als-{source_hash}"

    # Check if we already have a compiled binary with matching hash
    if binary_path.exists() and binary_path.is_file():
        # Verify it's executable
        try:
            binary_path.chmod(0o755)
            return binary_path
        except OSError:
            pass

    # Need to compile - get the source from package
    try:
        source_file = resources.files("ghostty_ambient.sensors").joinpath("als.m")
        source_content = source_file.read_text()
    except Exception:
        # Fallback: try project root (development mode)
        dev_path = Path(__file__).parent.parent.parent.parent.parent / "als"
        if dev_path.exists():
            return dev_path
        raise FileNotFoundError("als.m source not found in package") from None

    # Write source to temp location for compilation
    temp_source = CACHE_DIR / "als.m"
    temp_source.write_text(source_content)

    # Compile with clang (always available on macOS)
    compile_cmd = [
        "clang",
        "-framework",
        "IOKit",
        "-framework",
        "Foundation",
        "-framework",
        "CoreFoundation",
        "-F",
        "/System/Library/PrivateFrameworks",
        "-framework",
        "BezelServices",
        "-o",
        str(binary_path),
        str(temp_source),
    ]

    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed: {result.stderr}")

        # Make executable
        binary_path.chmod(0o755)

        # Clean up old cached binaries (keep only current)
        for old_binary in CACHE_DIR.glob("als-*"):
            if old_binary != binary_path:
                try:
                    old_binary.unlink()
                except OSError:
                    pass

        return binary_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Compilation timed out") from None
    except FileNotFoundError:
        raise RuntimeError("clang not found - Xcode Command Line Tools required") from None


@SensorRegistry.register
class MacOSALSBackend(SensorBackend):
    """
    macOS Ambient Light Sensor backend.

    Compiles and caches the ALS binary from bundled source on first use.
    Supports built-in sensors and external displays (Studio Display, etc.).
    """

    def __init__(self, als_path: Path | None = None):
        self._als_path = als_path
        self._calibration_factor = 1.0
        self._compile_error: str | None = None

    def _ensure_binary(self) -> Path | None:
        """Ensure the als binary is available, compiling if needed."""
        if self._als_path is not None:
            return self._als_path if self._als_path.exists() else None

        # Already tried and failed
        if self._compile_error is not None:
            return None

        try:
            self._als_path = _get_als_binary_path()
            return self._als_path
        except Exception as e:
            self._compile_error = str(e)
            return None

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
        if sys.platform != "darwin":
            return False
        binary = self._ensure_binary()
        return binary is not None and binary.exists()

    def read(self) -> SensorReading:
        binary = self._ensure_binary()
        if binary is None:
            error = self._compile_error or "ALS binary not available"
            return SensorReading(error=error)

        try:
            result = subprocess.run(
                [str(binary)],
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
            return SensorReading(error=f"als returned {result.returncode}: {result.stderr.strip()}")
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
