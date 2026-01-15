"""Background learning daemon for ghostty-ambient.

Only learns when Ghostty is the frontmost application to avoid
incorrect preferences from idle terminal windows.
"""

import subprocess
import time
from datetime import datetime

from .history import History
from .sensor import get_lux, get_power_source, get_system_appearance, get_weather
from .sensors import get_best_backend
from .themes import get_current_font, get_current_theme

DEFAULT_INTERVAL = 300  # 5 minutes


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

    # Record as implicit choice (source="snapshot")
    history.record_snapshot(
        theme_name=theme,
        lux=lux,
        hour=now.hour,
        weather_code=weather.weather_code,
        system_appearance=system_appearance,
        power_source=power_source,
        font=font,
        background_hex=background_hex,
        foreground_hex=foreground_hex,
        palette_chromas=palette_chromas,
    )

    # Get factor buckets for display
    from .history import (
        get_day_bucket,
        get_font_bucket,
        get_lux_bucket,
        get_power_bucket,
        get_system_bucket,
        get_time_bucket,
        get_weather_bucket,
    )

    factors = {
        "time": get_time_bucket(now.hour),
        "lux": get_lux_bucket(lux),
        "weather": get_weather_bucket(weather.weather_code),
        "system": get_system_bucket(system_appearance),
        "day": get_day_bucket(),
        "power": get_power_bucket(power_source),
        "font": get_font_bucket(font),
    }

    return {
        "timestamp": now.isoformat(),
        "theme": theme,
        "factors": factors,
        "raw": {
            "lux": lux,
            "weather_code": weather.weather_code,
        },
    }


def run_daemon(verbose: bool = False, interval: int = DEFAULT_INTERVAL):
    """Run the background learning daemon.

    Args:
        verbose: Print status messages
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

    if verbose:
        print("ghostty-ambient daemon started", flush=True)
        print(f"Snapshot interval: {interval_str}", flush=True)
        print("Only recording when Ghostty is frontmost...", flush=True)
        print(flush=True)

    # Calculate skip message frequency (roughly every hour)
    skip_log_every = max(1, 3600 // interval)

    while True:
        try:
            snapshot = take_snapshot(history)
            if snapshot:
                if verbose:
                    factors = snapshot["factors"]
                    factor_str = " | ".join(
                        f"{v}" for k, v in factors.items() if v != "unknown"
                    )
                    lux = snapshot["raw"]["lux"]
                    lux_str = f"{lux:.0f}lux" if lux else "?"
                    print(f"[{snapshot['timestamp'][11:19]}] {snapshot['theme']}")
                    print(f"         {factor_str} ({lux_str})", flush=True)
                skipped = 0
            else:
                skipped += 1
                if verbose and skipped % skip_log_every == 1:
                    print(f"[{datetime.now().isoformat()[:19]}] Ghostty not active, skipping...")
        except Exception as e:
            if verbose:
                print(f"Error: {e}")

        time.sleep(interval)
