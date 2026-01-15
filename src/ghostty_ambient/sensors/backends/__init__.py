"""Built-in sensor backends."""

from __future__ import annotations

import sys

# Only import platform-specific backends
if sys.platform == "darwin":
    from .macos import MacOSALSBackend

    __all__ = ["MacOSALSBackend"]
elif sys.platform == "linux":
    from .linux import LinuxSysfsBackend

    __all__ = ["LinuxSysfsBackend"]
elif sys.platform == "win32":
    from .windows import WindowsSensorBackend

    __all__ = ["WindowsSensorBackend"]
else:
    __all__ = []
