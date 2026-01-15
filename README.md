# Ghostty Ambient

Ambient light-aware theme selector for [Ghostty](https://ghostty.org/) with Bayesian preference learning.

<img src="https://github.com/gezibash/ghostty-ambient/releases/download/v0.1.0/demo.gif" alt="ghostty-ambient demo" />

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/gezibash/ghostty-ambient/main/install.sh | bash
```

This installs the package and sets up the learning daemon to run automatically.

## Features

- **Ambient Light Sensing**: Reads your Mac's ambient light sensor to suggest themes appropriate for current lighting
- **Bayesian Learning**: Learns your color preferences over time—which themes you prefer in different contexts
- **Context-Aware**: Considers time of day, weather, system dark/light mode, and power source
- **Theme Generation**: Generates custom themes from your learned preferences (background, contrast, saturation)
- **Portable Profiles**: Export/import your learned preferences across devices

## Installation

```bash
# Using uv (recommended)
uv add ghostty-ambient

# Using pip
pip install ghostty-ambient
```

## Quick Start

### Interactive Theme Picker

```bash
ghostty-ambient
```

Shows a picker with theme recommendations based on current conditions. Select a theme to apply it.

### Run the Learning Daemon

```bash
ghostty-ambient --daemon
```

Runs in the background, learning your preferences by observing which themes you use in different contexts. Only records when Ghostty is the frontmost application.

### Generate Your Ideal Theme

```bash
ghostty-ambient --ideal
```

Generates and applies a custom theme optimized for your current context, based on learned preferences:
- Background color (light vs dark)
- Contrast (how much difference between background and foreground)
- Chroma (color saturation)

## CLI Reference

```
ghostty-ambient              # Interactive theme picker
ghostty-ambient --ideal      # Generate and apply optimal theme
ghostty-ambient --set NAME   # Set theme by name
ghostty-ambient --stats      # Show learned preferences
ghostty-ambient --export-profile FILE  # Export preferences
ghostty-ambient --import-profile FILE  # Import preferences
ghostty-ambient --favorite   # Mark current theme as favorite
ghostty-ambient --dislike    # Mark current theme as disliked
ghostty-ambient --sensors    # Show available light sensors
ghostty-ambient --setup      # Configure location (for weather)
```

### Daemon Management

```bash
ghostty-ambient --status              # Check daemon status
ghostty-ambient --start               # Start daemon (5m interval)
ghostty-ambient --start --freq 1m     # Start with custom interval
ghostty-ambient --stop                # Stop daemon
ghostty-ambient --restart             # Restart daemon
ghostty-ambient --logs                # Tail daemon logs
```

### Run Daemon Inline

```bash
ghostty-ambient --daemon              # Run daemon in foreground
ghostty-ambient --daemon --interval 30s  # Custom interval
```

## How It Works

### Bayesian Preference Learning

The system learns three key properties of your theme preferences:

| Property | What It Learns |
|----------|----------------|
| **Background L** | Light vs dark theme preference (LAB lightness) |
| **Contrast** | Preferred difference between background and foreground |
| **Chroma** | Preferred color saturation in the palette |

These are learned per-context:
- Time of day (morning, afternoon, evening, night)
- Ambient light (dim, office, bright, daylight)
- System appearance (light mode, dark mode)
- Power source (AC, battery)

### View Your Learned Preferences

```bash
ghostty-ambient --stats
```

```
By Time of Day
╭───────────┬───────┬────┬──────────┬────────┬─────╮
│           │ Theme │  L │ Contrast │ Chroma │   n │
├───────────┼───────┼────┼──────────┼────────┼─────┤
│ Morning   │ light │ 97 │       84 │     49 │ 124 │
│ Afternoon │ light │ 97 │       86 │     50 │ 121 │
╰───────────┴───────┴────┴──────────┴────────┴─────╯
```

### Portable Profiles

Export your learned preferences:

```bash
ghostty-ambient --export-profile ~/my-prefs.json
```

Import on another device:

```bash
ghostty-ambient --import-profile ~/my-prefs.json
```

## Platform Support

| Platform | Ambient Light Sensor | Theme Detection |
|----------|---------------------|-----------------|
| macOS    | Native (via `als` helper) | Full support |
| Linux    | Via `iio-sensor-proxy` | Full support |
| Windows  | Via Windows SDK | Partial support |

### macOS Ambient Light Sensor

The bundled `als` binary reads the ambient light sensor. To compile from source:

```bash
clang -framework IOKit -framework CoreFoundation als.m -o als
```

## Daemon Management

The install script sets up a background daemon that learns your preferences. Use these commands to manage it:

```bash
ghostty-ambient --status              # Check if daemon is running
ghostty-ambient --start               # Start the daemon (default 5m interval)
ghostty-ambient --start --freq 30s    # Start with custom interval
ghostty-ambient --stop                # Stop the daemon
ghostty-ambient --restart             # Restart the daemon
ghostty-ambient --logs                # Tail daemon logs (Ctrl+C to exit)
```

## Uninstall

```bash
curl -fsSL https://raw.githubusercontent.com/gezibash/ghostty-ambient/main/uninstall.sh | bash
```

Your learned preferences are preserved in `~/.config/ghostty-ambient/`.

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/gezibash/ghostty-ambient
cd ghostty-ambient
uv sync --all-extras

# Run tests
uv run pytest

# Run with verbose output
uv run ghostty-ambient --daemon --interval 30s
```

## License

MIT License - see [LICENSE](LICENSE) for details.
