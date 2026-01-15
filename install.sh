#!/bin/bash
set -e

# ghostty-ambient installer
# Usage: curl -fsSL https://raw.githubusercontent.com/zim/ghostty-ambient/main/install.sh | bash

echo "Installing ghostty-ambient..."
echo

# Detect platform
OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)      echo "Unsupported platform: $OS"; exit 1 ;;
esac

# Install the package
BIN_DIR="$HOME/.local/bin"

if command -v uv &> /dev/null; then
    echo "Installing via uv..."
    uv tool install ghostty-ambient
    BIN_DIR="$HOME/.local/bin"  # uv tools go here
elif command -v pipx &> /dev/null; then
    echo "Installing via pipx..."
    pipx install ghostty-ambient
    BIN_DIR="$HOME/.local/bin"
elif command -v pip &> /dev/null; then
    echo "Installing via pip..."
    pip install --user ghostty-ambient
    BIN_DIR="$HOME/.local/bin"
else
    echo "Error: No Python package manager found (uv, pipx, or pip)"
    echo "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Package installed."

# Ensure bin directory is in PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo
    echo "Adding $BIN_DIR to PATH..."

    # Detect shell and add to appropriate rc file
    SHELL_NAME="$(basename "$SHELL")"
    case "$SHELL_NAME" in
        bash)
            RC_FILE="$HOME/.bashrc"
            [ -f "$HOME/.bash_profile" ] && RC_FILE="$HOME/.bash_profile"
            ;;
        zsh)
            RC_FILE="$HOME/.zshrc"
            ;;
        fish)
            RC_FILE="$HOME/.config/fish/config.fish"
            ;;
        *)
            RC_FILE="$HOME/.profile"
            ;;
    esac

    if [ -f "$RC_FILE" ]; then
        if ! grep -q "$BIN_DIR" "$RC_FILE" 2>/dev/null; then
            echo "" >> "$RC_FILE"
            echo "# Added by ghostty-ambient installer" >> "$RC_FILE"
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$RC_FILE"
            echo "Added to $RC_FILE"
        fi
    fi

    # Also export for current session
    export PATH="$BIN_DIR:$PATH"
fi

# Verify installation
echo
if command -v ghostty-ambient &> /dev/null; then
    echo "Verified: ghostty-ambient is in PATH"
    BINARY_PATH="$(which ghostty-ambient)"
else
    # Try to find it directly
    if [ -x "$BIN_DIR/ghostty-ambient" ]; then
        BINARY_PATH="$BIN_DIR/ghostty-ambient"
        echo "Installed to: $BINARY_PATH"
        echo "Note: Restart your shell or run: export PATH=\"$BIN_DIR:\$PATH\""
    else
        echo "Warning: ghostty-ambient binary not found"
        echo "You may need to restart your shell"
        BINARY_PATH=""
    fi
fi
echo

# Set up daemon
setup_macos_daemon() {
    PLIST_DIR="$HOME/Library/LaunchAgents"
    PLIST_FILE="$PLIST_DIR/com.ghostty-ambient.daemon.plist"

    mkdir -p "$PLIST_DIR"

    # Use the binary path we already found, or search again
    if [ -z "$BINARY_PATH" ]; then
        BINARY_PATH="$(which ghostty-ambient 2>/dev/null || echo "$BIN_DIR/ghostty-ambient")"
    fi

    if [ ! -x "$BINARY_PATH" ]; then
        echo "Warning: Could not find ghostty-ambient binary"
        echo "You may need to restart your shell and run the daemon manually:"
        echo "  ghostty-ambient --daemon"
        return
    fi

    cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ghostty-ambient.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>$BINARY_PATH</string>
        <string>--daemon</string>
        <string>--interval</string>
        <string>5m</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$HOME/.local/share/ghostty-ambient/daemon.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/.local/share/ghostty-ambient/daemon.log</string>
</dict>
</plist>
EOF

    # Create log directory
    mkdir -p "$HOME/.local/share/ghostty-ambient"

    # Load the daemon
    launchctl unload "$PLIST_FILE" 2>/dev/null || true
    launchctl load "$PLIST_FILE"

    echo "macOS daemon installed and started."
    echo "  Config: $PLIST_FILE"
    echo "  Logs:   $HOME/.local/share/ghostty-ambient/daemon.log"
}

setup_linux_daemon() {
    SYSTEMD_DIR="$HOME/.config/systemd/user"
    SERVICE_FILE="$SYSTEMD_DIR/ghostty-ambient.service"

    mkdir -p "$SYSTEMD_DIR"

    # Use the binary path we already found, or search again
    if [ -z "$BINARY_PATH" ]; then
        BINARY_PATH="$(which ghostty-ambient 2>/dev/null || echo "$BIN_DIR/ghostty-ambient")"
    fi

    if [ ! -x "$BINARY_PATH" ]; then
        echo "Warning: Could not find ghostty-ambient binary"
        echo "You may need to restart your shell and run the daemon manually:"
        echo "  ghostty-ambient --daemon"
        return
    fi

    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Ghostty Ambient Theme Daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=$BINARY_PATH --daemon --interval 5m
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF

    # Reload and start
    systemctl --user daemon-reload
    systemctl --user enable ghostty-ambient
    systemctl --user restart ghostty-ambient

    echo "Linux systemd daemon installed and started."
    echo "  Config: $SERVICE_FILE"
    echo "  Status: systemctl --user status ghostty-ambient"
    echo "  Logs:   journalctl --user -u ghostty-ambient -f"
}

echo "Setting up daemon..."
if [ "$PLATFORM" = "macos" ]; then
    setup_macos_daemon
elif [ "$PLATFORM" = "linux" ]; then
    setup_linux_daemon
fi

echo
echo "Installation complete!"
echo
echo "Quick start:"
echo "  ghostty-ambient              # Interactive theme picker"
echo "  ghostty-ambient --ideal      # Generate optimal theme"
echo "  ghostty-ambient --stats      # View learned preferences"
echo
echo "The daemon is now learning your theme preferences in the background."
