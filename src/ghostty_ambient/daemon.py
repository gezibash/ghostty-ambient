"""Background learning daemon for ghostty-ambient.

Only learns when Ghostty is the frontmost application to avoid
incorrect preferences from idle terminal windows.
"""

import json
import subprocess
import sys
import time
from datetime import datetime

from .history import History
from .sensor import get_lux, get_power_source, get_system_appearance, get_weather
from .sensors import get_best_backend
from .themes import get_current_font, get_current_theme

DEFAULT_INTERVAL = 300  # 5 minutes

# Auto-detect if running in interactive terminal
IS_TTY = sys.stdout.isatty()


def is_ghostty_active() -> bool:
    """
    Check if Ghostty is the frontmost application.

    Returns True only when user is actively using Ghostty.
    """
    import platform

    system = platform.system()

    if system == "Darwin":
        try:
            # Use AppleScript to get frontmost app name
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of first process whose frontmost is true',
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                frontmost = result.stdout.strip().lower()
                return "ghostty" in frontmost
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    elif system == "Linux":
        try:
            # Use xdotool to get active window
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                window_name = result.stdout.strip().lower()
                return "ghostty" in window_name
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback: check if Ghostty process exists and has focus
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowpid"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                pid = result.stdout.strip()
                # Check if this PID is Ghostty
                proc_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "comm="],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if "ghostty" in proc_result.stdout.strip().lower():
                    return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return False


def take_snapshot(history: History) -> dict | None:
    """
    Take a snapshot of current theme + context.

    Returns snapshot dict or None if:
    - Ghostty is not active (user doing something else)
    - Unable to determine current theme
    """
    # Only learn when Ghostty is the active app
    if not is_ghostty_active():
        return None

    theme = get_current_theme()
    if not theme:
        return None

    # Get theme's full properties for learning
    from .themes import get_theme_properties, load_all_themes
    all_themes = load_all_themes()
    theme_dict = next((t for t in all_themes if t["name"] == theme), None)

    background_hex = None
    foreground_hex = None
    palette_chromas = None

    if theme_dict:
        background_hex = theme_dict.get("background")
        foreground_hex = theme_dict.get("foreground")
        # Extract palette chromas for chroma learning
        props = get_theme_properties(theme_dict)
        palette_chromas = props.get("palette_chromas")

    # Get context
    backend = get_best_backend()
    if backend and backend.is_available():
        reading = backend.read()
        lux = reading.lux
    else:
        lux = get_lux()

    weather = get_weather()
    system_appearance = get_system_appearance()
    power_source = get_power_source()
    font = get_current_font()
    now = datetime.now()

    # Build factor buckets using registry (includes all 11 factors)
    from .factors import FactorRegistry

    context = {
        "hour": now.hour,
        "lux": lux,
        "weather_code": weather.weather_code,
        "system_appearance": system_appearance,
        "power_source": power_source,
        "font": font,
        "sunrise": weather.sunrise,
        "sunset": weather.sunset,
        "pressure": weather.pressure,
        "cloud_cover": weather.cloud_cover,
        "uv_index": weather.uv_index,
        "datetime": now,
    }
    factors = FactorRegistry.get_all_buckets(context)

    # Record as implicit observation (source="daemon")
    history.record_snapshot(
        theme_name=theme,
        factors=factors,
        background_hex=background_hex,
        foreground_hex=foreground_hex,
        palette_chromas=palette_chromas,
    )

    return {
        "timestamp": now.isoformat(),
        "theme": theme,
        "factors": factors,
        "raw": {
            "lux": lux,
            "weather_code": weather.weather_code,
        },
    }


def _log_json(level: str, msg: str, **kwargs) -> None:
    """Output a JSON log line (Loki-style)."""
    entry = {"ts": datetime.now().isoformat(), "level": level, "msg": msg, **kwargs}
    print(json.dumps(entry, default=str), flush=True)


def _log_pretty(level: str, msg: str, **kwargs) -> None:
    """Output a human-readable log line with rich formatting."""
    from rich.console import Console
    console = Console()

    ts = datetime.now().strftime("%H:%M:%S")
    level_colors = {"info": "green", "error": "red", "warn": "yellow"}
    color = level_colors.get(level, "white")

    if msg == "daemon_started":
        console.print(f"[dim]{ts}[/] [bold {color}]daemon started[/] interval={kwargs.get('interval_str', '?')}")
    elif msg == "snapshot":
        theme = kwargs.get("theme", "?")
        lux = kwargs.get("lux")
        factors = kwargs.get("factors", {})
        lux_str = f"{lux:.0f}" if lux else "?"
        # Show only non-unknown factors
        factor_parts = [f"[cyan]{v}[/]" for k, v in factors.items() if v != "unknown"]
        factor_str = " ".join(factor_parts)
        console.print(f"[dim]{ts}[/] [bold]{theme}[/] [dim]lux={lux_str}[/] {factor_str}")
    elif msg == "skipped":
        console.print(f"[dim]{ts}[/] [yellow]skipped[/] ghostty inactive ({kwargs.get('skipped_count', 0)})")
    elif msg == "snapshot_failed":
        console.print(f"[dim]{ts}[/] [red]error[/] {kwargs.get('error', '?')}")
    else:
        console.print(f"[dim]{ts}[/] [{color}]{msg}[/] {kwargs}")


def _log(level: str, msg: str, **kwargs) -> None:
    """Log a message - pretty for TTY, JSON for pipes."""
    if IS_TTY:
        _log_pretty(level, msg, **kwargs)
    else:
        _log_json(level, msg, **kwargs)


def run_daemon(interval: int = DEFAULT_INTERVAL):
    """Run the background learning daemon.

    Auto-detects output format:
    - TTY: Pretty human-readable output with colors
    - Piped/redirected: Loki-style JSON lines

    Args:
        interval: Seconds between snapshots (default 300 = 5 min)
    """
    history = History()
    skipped = 0

    # Format interval for display
    if interval >= 3600:
        interval_str = f"{interval // 3600}h"
    elif interval >= 60:
        interval_str = f"{interval // 60}m"
    else:
        interval_str = f"{interval}s"

    _log("info", "daemon_started", interval=interval, interval_str=interval_str)

    # Calculate skip message frequency (roughly every hour)
    skip_log_every = max(1, 3600 // interval)

    while True:
        try:
            snapshot = take_snapshot(history)
            if snapshot:
                _log(
                    "info",
                    "snapshot",
                    theme=snapshot["theme"],
                    lux=snapshot["raw"]["lux"],
                    factors=snapshot["factors"],
                )
                skipped = 0
            else:
                skipped += 1
                if skipped % skip_log_every == 1:
                    _log("info", "skipped", reason="ghostty_inactive", skipped_count=skipped)
        except Exception as e:
            _log("error", "snapshot_failed", error=str(e))

        time.sleep(interval)
