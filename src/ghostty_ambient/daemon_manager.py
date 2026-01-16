"""
Daemon management for ghostty-ambient.

Provides CLI commands to control the background learning daemon:
- status: Check if daemon is running
- start: Start the daemon
- stop: Stop the daemon
- restart: Restart the daemon
- logs: Tail daemon logs
"""

import platform
import shutil
import subprocess
import sys
from importlib.metadata import version as get_version
from pathlib import Path

# Service file locations
MACOS_PLIST = Path.home() / "Library/LaunchAgents/com.ghostty-ambient.daemon.plist"
LINUX_SERVICE = Path.home() / ".config/systemd/user/ghostty-ambient.service"
LOG_FILE = Path.home() / ".local/share/ghostty-ambient/daemon.log"
HISTORY_FILE = Path.home() / ".config/ghostty-ambient/history.json"
LABEL = "com.ghostty-ambient.daemon"


def get_platform() -> str:
    """Return 'Darwin' for macOS, 'Linux' for Linux."""
    return platform.system()


def _format_bytes(size: int) -> str:
    """Format bytes as human-readable string."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    elif size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} bytes"


def _get_process_memory(pid: str) -> str | None:
    """Get memory usage for a process."""
    try:
        if get_platform() == "Darwin":
            # macOS: use ps to get RSS (resident set size)
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", pid],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                rss_kb = int(result.stdout.strip())
                return _format_bytes(rss_kb * 1024)
        else:
            # Linux: read from /proc
            statm = Path(f"/proc/{pid}/statm").read_text()
            pages = int(statm.split()[1])  # RSS in pages
            page_size = 4096  # typical page size
            return _format_bytes(pages * page_size)
    except (subprocess.SubprocessError, ValueError, FileNotFoundError, PermissionError):
        return None
    return None


def _find_binary() -> str | None:
    """Find the ghostty-ambient binary path."""
    # First check if it's in PATH
    binary = shutil.which("ghostty-ambient")
    if binary:
        return binary

    # Check common locations
    candidates = [
        Path.home() / ".local/bin/ghostty-ambient",
        Path("/usr/local/bin/ghostty-ambient"),
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return str(path)

    return None


def _generate_macos_plist(binary_path: str, interval: str) -> str:
    """Generate macOS launchd plist content."""
    log_path = str(LOG_FILE)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{binary_path}</string>
        <string>--daemon</string>
        <string>--interval</string>
        <string>{interval}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
</dict>
</plist>
"""


def _generate_linux_service(binary_path: str, interval: str) -> str:
    """Generate Linux systemd service content."""
    return f"""[Unit]
Description=Ghostty Ambient Theme Daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart={binary_path} --daemon --interval {interval}
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""


def daemon_status() -> None:
    """Show daemon status."""
    if get_platform() == "Darwin":
        _macos_status()
    else:
        _linux_status()


def _macos_status() -> None:
    """Show macOS daemon status."""
    print(f"Version: {get_version('ghostty-ambient')}")
    result = subprocess.run(
        ["launchctl", "list"],
        capture_output=True,
        text=True,
    )

    for line in result.stdout.splitlines():
        if LABEL in line:
            parts = line.split()
            if len(parts) >= 3:
                pid = parts[0]
                status_code = parts[1]
                if pid == "-":
                    print(f"Daemon: stopped (exit code: {status_code})")
                else:
                    print(f"Daemon: running (PID: {pid})")
                    # Show memory usage
                    mem = _get_process_memory(pid)
                    if mem:
                        print(f"Memory: {mem}")

                # Show log file size
                if LOG_FILE.exists():
                    print(f"Logs: {_format_bytes(LOG_FILE.stat().st_size)}")

                # Show history file size (learning data)
                if HISTORY_FILE.exists():
                    print(f"Data: {_format_bytes(HISTORY_FILE.stat().st_size)}")
                return

    # Not found in launchctl list
    if not MACOS_PLIST.exists():
        print("Daemon: not installed")
        print("Run: ghostty-ambient --start")
    else:
        print("Daemon: not loaded")
        print("Run: ghostty-ambient --start")


def _linux_status() -> None:
    """Show Linux daemon status."""
    print(f"Version: {get_version('ghostty-ambient')}")
    if not LINUX_SERVICE.exists():
        print("Daemon: not installed")
        print("Run: ghostty-ambient --start")
        return

    result = subprocess.run(
        ["systemctl", "--user", "is-active", "ghostty-ambient"],
        capture_output=True,
        text=True,
    )

    status = result.stdout.strip()
    if status == "active":
        # Get PID
        pid_result = subprocess.run(
            ["systemctl", "--user", "show", "ghostty-ambient", "--property=MainPID"],
            capture_output=True,
            text=True,
        )
        pid = pid_result.stdout.strip().replace("MainPID=", "")
        print(f"Daemon: running (PID: {pid})")
        # Show memory usage
        mem = _get_process_memory(pid)
        if mem:
            print(f"Memory: {mem}")
    else:
        print(f"Daemon: {status}")

    # Show history file size (learning data)
    if HISTORY_FILE.exists():
        print(f"Data: {_format_bytes(HISTORY_FILE.stat().st_size)}")


def daemon_start(freq: str = "5m") -> None:
    """Start the daemon with given interval."""
    binary = _find_binary()
    if not binary:
        print("Error: ghostty-ambient binary not found", file=sys.stderr)
        print("Make sure ghostty-ambient is installed and in your PATH", file=sys.stderr)
        sys.exit(1)

    if get_platform() == "Darwin":
        _macos_start(binary, freq)
    else:
        _linux_start(binary, freq)


def _macos_start(binary: str, freq: str) -> None:
    """Start macOS daemon."""
    # Create directories
    MACOS_PLIST.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Generate and write plist
    plist_content = _generate_macos_plist(binary, freq)
    MACOS_PLIST.write_text(plist_content)

    # Unload if already loaded (ignore errors)
    subprocess.run(
        ["launchctl", "unload", str(MACOS_PLIST)],
        capture_output=True,
    )

    # Load the daemon
    result = subprocess.run(
        ["launchctl", "load", str(MACOS_PLIST)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error starting daemon: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Daemon started (interval: {freq})")
    print(f"Logs: {LOG_FILE}")


def _linux_start(binary: str, freq: str) -> None:
    """Start Linux daemon."""
    # Create directories
    LINUX_SERVICE.parent.mkdir(parents=True, exist_ok=True)

    # Generate and write service file
    service_content = _generate_linux_service(binary, freq)
    LINUX_SERVICE.write_text(service_content)

    # Reload systemd
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)

    # Enable and start
    subprocess.run(["systemctl", "--user", "enable", "ghostty-ambient"], check=True)
    result = subprocess.run(
        ["systemctl", "--user", "start", "ghostty-ambient"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error starting daemon: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Daemon started (interval: {freq})")
    print("Logs: journalctl --user -u ghostty-ambient -f")


def daemon_stop() -> None:
    """Stop the daemon."""
    if get_platform() == "Darwin":
        _macos_stop()
    else:
        _linux_stop()


def _macos_stop() -> None:
    """Stop macOS daemon."""
    if not MACOS_PLIST.exists():
        print("Daemon not installed")
        return

    result = subprocess.run(
        ["launchctl", "unload", str(MACOS_PLIST)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 and "Could not find" not in result.stderr:
        print(f"Error stopping daemon: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print("Daemon stopped")


def _linux_stop() -> None:
    """Stop Linux daemon."""
    result = subprocess.run(
        ["systemctl", "--user", "stop", "ghostty-ambient"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if "not loaded" in result.stderr.lower():
            print("Daemon not running")
            return
        print(f"Error stopping daemon: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print("Daemon stopped")


def daemon_restart() -> None:
    """Restart the daemon."""
    if get_platform() == "Darwin":
        _macos_restart()
    else:
        _linux_restart()


def _macos_restart() -> None:
    """Restart macOS daemon."""
    if not MACOS_PLIST.exists():
        print("Daemon not installed. Use --start to install and start.", file=sys.stderr)
        sys.exit(1)

    # Unload
    subprocess.run(
        ["launchctl", "unload", str(MACOS_PLIST)],
        capture_output=True,
    )

    # Load
    result = subprocess.run(
        ["launchctl", "load", str(MACOS_PLIST)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error restarting daemon: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print("Daemon restarted")


def _linux_restart() -> None:
    """Restart Linux daemon."""
    result = subprocess.run(
        ["systemctl", "--user", "restart", "ghostty-ambient"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error restarting daemon: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print("Daemon restarted")


def daemon_logs() -> None:
    """Tail daemon logs (Ctrl+C to exit)."""
    if get_platform() == "Darwin":
        _macos_logs()
    else:
        _linux_logs()


def _macos_logs() -> None:
    """Tail macOS daemon logs."""
    if not LOG_FILE.exists():
        print(f"No log file found at {LOG_FILE}")
        print("The daemon may not have run yet.")
        return

    print(f"Tailing {LOG_FILE} (Ctrl+C to exit)...")
    print()

    try:
        # Use tail -f to follow the log file
        subprocess.run(["tail", "-f", str(LOG_FILE)])
    except KeyboardInterrupt:
        print()  # Clean exit


def _linux_logs() -> None:
    """Tail Linux daemon logs via journalctl."""
    print("Tailing logs (Ctrl+C to exit)...")
    print()

    try:
        subprocess.run(["journalctl", "--user", "-u", "ghostty-ambient", "-f"])
    except KeyboardInterrupt:
        print()  # Clean exit
