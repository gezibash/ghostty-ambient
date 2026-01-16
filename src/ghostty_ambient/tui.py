"""Rich-based theme picker with probability bars."""

import readchar
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text


def prob_bar(pct: float, width: int = 20) -> str:
    """Create a slider-style probability bar."""
    # Clamp percentage to 0-100 range
    pct = max(0, min(100, pct))
    pos = int(pct / 100 * (width - 1))  # Position of the marker (0 to width-1)
    # Slider style: ━━━━━━━━━●─────────────────────
    return "━" * pos + "●" + "─" * (width - 1 - pos)


def format_theme_line(
    theme: dict,
    selected: bool,
    current_theme: str | None = None,
) -> Text:
    """Format a single theme line with probability bar."""
    text = Text()

    # Cursor
    cursor = ">" if selected else " "
    text.append(cursor, style="bold cyan" if selected else "")
    text.append(" ")

    # Probability bar - use simple characters, no styling on bar itself
    score = theme.get("_score", 50)
    bar = prob_bar(score)
    if selected:
        text.append(bar, style="cyan")
    else:
        text.append(bar, style="dim")
    text.append(" ")

    # Percentage
    pct_str = f"{score:5.1f}%"
    text.append(pct_str, style="bold cyan" if selected else "dim")
    text.append(" ")

    # Theme name
    name = theme["name"]
    text.append(name, style="bold white" if selected else "")

    # Favorite marker
    if theme.get("_is_favorite"):
        text.append(" ★", style="yellow")

    # Current marker
    if name == current_theme:
        text.append(" (current)", style="dim italic")

    return text


def pick_theme(
    recommendations: list[dict],
    explore: list[dict] | None = None,
    current_theme: str | None = None,
    header: str | None = None,
) -> dict | None:
    """
    Show interactive theme picker with probability bars.

    Args:
        recommendations: List of recommended theme dicts with _score field
        explore: List of explore theme dicts (different from usual preferences)
        current_theme: Name of currently active theme
        header: Status header to display above the menu

    Returns:
        Selected theme dict, or None if cancelled.
    """
    console = Console()

    sections = ["recommendations", "explore"] if explore else ["recommendations"]
    current_section = 0
    cursor_idx = 0

    def get_current_list():
        if sections[current_section] == "recommendations":
            return recommendations
        return explore or []

    # Global padding (top, right, bottom, left)
    PADDING = (2, 4, 1, 4)

    def render():
        """Render the full UI."""
        console.clear()

        # Top padding
        console.print("\n" * PADDING[0], end="")

        # Header (context box) - already contains current theme info
        if header:
            console.print(Padding(header, (0, 0, 0, PADDING[3])))
            console.print()

        # Section title and description
        themes = get_current_list()
        if sections[current_section] == "recommendations":
            if explore:
                title = "Recommended  [dim][TAB → explore][/]"
            else:
                title = "Recommended"
            description = "Themes that match your preferences and current context"
        else:
            title = "Explore  [dim][TAB → back][/]"
            description = "Themes that score well but you haven't selected much"

        # Build content
        lines = []
        for i, theme in enumerate(themes):
            line = format_theme_line(theme, i == cursor_idx, current_theme)
            lines.append(line)

        # Create panel content with description
        content = Text()
        content.append(description, style="dim italic")
        content.append("\n\n")
        for i, line in enumerate(lines):
            content.append_text(line)
            if i < len(lines) - 1:
                content.append("\n")

        # Render panel with symmetric padding
        # Calculate available width for panel (terminal width minus left and right padding)
        available_width = console.width - PADDING[1] - PADDING[3]
        panel = Panel(
            content,
            title=title,
            title_align="left",
            border_style="cyan",
            padding=(1, 2),
            width=available_width,
        )
        console.print(Padding(panel, (0, 0, 0, PADDING[3])))

        # Help text with padding
        console.print()
        console.print(Padding("[dim]↑↓ navigate  ENTER select  TAB switch  q quit[/]", (0, 0, 0, PADDING[3])))

    def handle_input() -> str | None:
        """Handle keyboard input. Returns 'quit', 'select', or None."""
        nonlocal cursor_idx, current_section

        themes = get_current_list()
        char = readchar.readkey()

        if char == readchar.key.UP or char == "k":
            cursor_idx = max(0, cursor_idx - 1)
        elif char == readchar.key.DOWN or char == "j":
            cursor_idx = min(len(themes) - 1, cursor_idx + 1)
        elif char == readchar.key.ENTER or char == "\r" or char == "\n":
            return "select"
        elif char == "\t":
            if len(sections) > 1:
                current_section = (current_section + 1) % len(sections)
                cursor_idx = 0
        elif char == "q" or char == readchar.key.ESC or char == readchar.key.CTRL_C:
            return "quit"

        return None

    # Main loop
    try:
        while True:
            render()
            action = handle_input()

            if action == "quit":
                console.clear()
                return None
            elif action == "select":
                themes = get_current_list()
                if themes and 0 <= cursor_idx < len(themes):
                    console.clear()
                    return themes[cursor_idx]
    except KeyboardInterrupt:
        console.clear()
        return None
