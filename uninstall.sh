#!/bin/bash
set -e

# ghostty-ambient uninstaller
# Usage: curl -fsSL https://raw.githubusercontent.com/gezibash/ghostty-ambient/main/uninstall.sh | bash

echo "Uninstalling ghostty-ambient..."
echo

# Detect platform
OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)      PLATFORM="unknown" ;;
esac

# Stop and remove daemon
if [ "$PLATFORM" = "macos" ]; then
    PLIST_FILE="$HOME/Library/LaunchAgents/com.ghostty-ambient.daemon.plist"
    if [ -f "$PLIST_FILE" ]; then
        echo "Stopping macOS daemon..."
        launchctl unload "$PLIST_FILE" 2>/dev/null || true
        rm -f "$PLIST_FILE"
        echo "Daemon removed."
    fi
elif [ "$PLATFORM" = "linux" ]; then
    SERVICE_FILE="$HOME/.config/systemd/user/ghostty-ambient.service"
    if [ -f "$SERVICE_FILE" ]; then
        echo "Stopping Linux daemon..."
        systemctl --user stop ghostty-ambient 2>/dev/null || true
        systemctl --user disable ghostty-ambient 2>/dev/null || true
        rm -f "$SERVICE_FILE"
        systemctl --user daemon-reload
        echo "Daemon removed."
    fi
fi

# Uninstall the package
if command -v uv &> /dev/null; then
    echo "Uninstalling via uv..."
    uv tool uninstall ghostty-ambient 2>/dev/null || true
elif command -v pipx &> /dev/null; then
    echo "Uninstalling via pipx..."
    pipx uninstall ghostty-ambient 2>/dev/null || true
elif command -v pip &> /dev/null; then
    echo "Uninstalling via pip..."
    pip uninstall -y ghostty-ambient 2>/dev/null || true
fi

echo
echo "Uninstall complete."
echo
echo "Note: Your learned preferences are preserved in:"
echo "  ~/.config/ghostty-ambient/history.json"
echo
echo "To remove all data: rm -rf ~/.config/ghostty-ambient"
