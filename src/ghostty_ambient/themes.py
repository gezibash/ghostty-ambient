"""Theme parsing and analysis utilities."""

import platform
import re
from pathlib import Path


def _find_ghostty_themes_dir() -> Path | None:
    """Find the Ghostty bundled themes directory based on platform."""
    system = platform.system()

    if system == "Darwin":
        # macOS: themes in app bundle
        path = Path("/Applications/Ghostty.app/Contents/Resources/ghostty/themes")
        if path.exists():
            return path

    elif system == "Linux":
        # Linux: check common installation paths
        candidates = [
            Path("/usr/share/ghostty/themes"),
            Path("/usr/local/share/ghostty/themes"),
            Path.home() / ".local/share/ghostty/themes",
        ]
        for path in candidates:
            if path.exists():
                return path

    elif system == "Windows":
        # Windows: check AppData and Program Files
        appdata = Path.home() / "AppData/Local/ghostty/themes"
        if appdata.exists():
            return appdata
        progfiles = Path("C:/Program Files/Ghostty/themes")
        if progfiles.exists():
            return progfiles

    return None


# Theme directories
GHOSTTY_THEMES = _find_ghostty_themes_dir()
USER_THEMES = Path.home() / ".config/ghostty/themes"
GHOSTTY_CONFIG = Path.home() / ".config/ghostty/config"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def calculate_brightness(hex_color: str) -> int:
    """Calculate perceived brightness (0-255) using ITU-R BT.601 luminance formula."""
    r, g, b = hex_to_rgb(hex_color)
    return int((r * 299 + g * 587 + b * 114) / 1000)


def calculate_warmth(hex_color: str) -> float:
    """Calculate color warmth (-1 to 1, negative=cool, positive=warm)."""
    r, g, b = hex_to_rgb(hex_color)
    if r + b == 0:
        return 0
    return (r - b) / (r + b + 1)


def parse_theme(theme_path: Path) -> dict | None:
    """Parse a Ghostty theme file and extract colors."""
    try:
        content = theme_path.read_text()
        theme = {"name": theme_path.name, "path": str(theme_path)}

        # Extract background color
        bg_match = re.search(r"^background\s*=\s*(#[0-9a-fA-F]{6})", content, re.M)
        if bg_match:
            theme["background"] = bg_match.group(1)
            theme["brightness"] = calculate_brightness(theme["background"])
            theme["warmth"] = calculate_warmth(theme["background"])
        else:
            return None

        # Extract foreground color
        fg_match = re.search(r"^foreground\s*=\s*(#[0-9a-fA-F]{6})", content, re.M)
        if fg_match:
            theme["foreground"] = fg_match.group(1)

        # Extract palette colors for preview
        palette = {}
        for match in re.finditer(r"^palette\s*=\s*(\d+)=(#[0-9a-fA-F]{6})", content, re.M):
            palette[int(match.group(1))] = match.group(2)
        if palette:
            theme["palette"] = palette

        return theme
    except Exception:
        return None


def load_all_themes() -> list[dict]:
    """Load and parse all available Ghostty themes."""
    themes = []
    seen_names = set()

    for theme_dir in [USER_THEMES, GHOSTTY_THEMES]:  # User themes take priority
        if theme_dir and theme_dir.exists():
            for theme_file in theme_dir.iterdir():
                if theme_file.is_file() and not theme_file.name.startswith("."):
                    if theme_file.name not in seen_names:
                        theme = parse_theme(theme_file)
                        if theme:
                            themes.append(theme)
                            seen_names.add(theme_file.name)

    return sorted(themes, key=lambda t: t["name"].lower())


def get_current_theme() -> str | None:
    """Get the currently configured theme from Ghostty config."""
    if not GHOSTTY_CONFIG.exists():
        return None

    try:
        content = GHOSTTY_CONFIG.read_text()
        match = re.search(r"^theme\s*=\s*(.+)$", content, re.M)
        if match:
            return match.group(1).strip()
    except Exception:
        pass

    return None


def get_current_font() -> str | None:
    """Get the primary font-family from Ghostty config."""
    if not GHOSTTY_CONFIG.exists():
        return None

    try:
        content = GHOSTTY_CONFIG.read_text()
        match = re.search(r"^font-family\s*=\s*(.+)$", content, re.M)
        if match:
            value = match.group(1).strip()
            # Remove surrounding quotes
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return value
    except Exception:
        pass

    return None


def apply_theme(theme_name: str) -> bool:
    """Apply a theme by updating Ghostty config. Returns success."""
    import subprocess

    if not GHOSTTY_CONFIG.exists():
        return False

    config_content = GHOSTTY_CONFIG.read_text()

    if re.search(r"^theme\s*=", config_content, re.M):
        new_content = re.sub(r"^theme\s*=.*$", f"theme = {theme_name}", config_content, flags=re.M)
    else:
        new_content = config_content.rstrip() + f"\ntheme = {theme_name}\n"

    GHOSTTY_CONFIG.write_text(new_content)

    # Reload Ghostty config
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to keystroke "," using {command down, shift down}'],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass

    return True


def get_theme_properties(theme: dict) -> dict:
    """
    Extract full theme properties for learning.

    Returns dict with:
        bg_lab: Background LAB color
        fg_lab: Foreground LAB color (if available)
        palette_chromas: List of chromas for palette colors 1-6
        contrast: Delta E between bg and fg
        avg_chroma: Average palette chroma
    """
    import math

    from .color import hex_to_lab

    result = {}

    # Background LAB
    if "background" in theme:
        result["bg_lab"] = hex_to_lab(theme["background"])

    # Foreground LAB
    if "foreground" in theme:
        result["fg_lab"] = hex_to_lab(theme["foreground"])

    # Calculate contrast if we have both
    if "bg_lab" in result and "fg_lab" in result:
        bg = result["bg_lab"]
        fg = result["fg_lab"]
        result["contrast"] = math.sqrt((bg[0] - fg[0]) ** 2 + (bg[1] - fg[1]) ** 2 + (bg[2] - fg[2]) ** 2)

    # Palette chromas (colors 1-6: red, green, yellow, blue, magenta, cyan)
    if "palette" in theme:
        chromas = []
        for i in range(1, 7):
            if i in theme["palette"]:
                lab = hex_to_lab(theme["palette"][i])
                # Chroma = sqrt(a² + b²)
                chroma = math.sqrt(lab[1] ** 2 + lab[2] ** 2)
                chromas.append(chroma)
        if chromas:
            result["palette_chromas"] = chromas
            result["avg_chroma"] = sum(chromas) / len(chromas)

    return result
