#!/bin/bash
set -e

# ghostty-ambient installer
# Usage: curl -fsSL https://raw.githubusercontent.com/gezibash/ghostty-ambient/main/install.sh | bash

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

# Set up daemon using CLI
echo "Setting up daemon..."
if [ -n "$BINARY_PATH" ] && [ -x "$BINARY_PATH" ]; then
    "$BINARY_PATH" --start
else
    echo "Warning: Could not start daemon automatically."
    echo "After restarting your shell, run: ghostty-ambient --start"
fi

echo
echo "Installation complete!"
echo
echo "Quick start:"
echo "  ghostty-ambient              # Interactive theme picker"
echo "  ghostty-ambient --ideal      # Generate optimal theme"
echo "  ghostty-ambient --stats      # View learned preferences"
echo
echo "Daemon management:"
echo "  ghostty-ambient --status     # Check daemon status"
echo "  ghostty-ambient --logs       # View daemon logs"
echo "  ghostty-ambient --restart    # Restart daemon"
echo
echo "The daemon is now learning your theme preferences in the background."
