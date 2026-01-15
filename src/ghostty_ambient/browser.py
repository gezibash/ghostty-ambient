"""History display utilities."""

from rich.console import Console
from rich.table import Table
from rich import box

from .history import History


def _get_learned_values(theme_posteriors: dict, context: str) -> dict | None:
    """Extract learned values for a context."""
    if context not in theme_posteriors:
        return None

    ctx_data = theme_posteriors[context]
    color_obs = ctx_data.get("color", {}).get("observations", [])
    contrast_obs = ctx_data.get("contrast", {}).get("observations", [])
    chroma_obs = ctx_data.get("chroma", {}).get("observations", [])

    if not color_obs:
        return None

    avg_L = sum(o[0] for o in color_obs) / len(color_obs)
    return {
        "L": avg_L,
        "type": "light" if avg_L > 50 else "dark",
        "contrast": sum(contrast_obs) / len(contrast_obs) if contrast_obs else None,
        "chroma": sum(chroma_obs) / len(chroma_obs) if chroma_obs else None,
        "n": len(color_obs),
    }


def _make_preference_table(
    console: Console,
    theme_posteriors: dict,
    title: str,
    contexts: list[tuple[str, str]],  # [(context_key, display_name), ...]
) -> bool:
    """Create a preference table for a dimension. Returns True if any data shown."""
    rows = []
    for ctx_key, display_name in contexts:
        vals = _get_learned_values(theme_posteriors, ctx_key)
        if vals:
            rows.append((display_name, vals))

    if not rows:
        return False

    console.print(f"[bold]{title}[/bold]")
    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("", style="cyan")
    table.add_column("Theme", justify="center")
    table.add_column("L", justify="right")
    table.add_column("Contrast", justify="right")
    table.add_column("Chroma", justify="right")
    table.add_column("n", justify="right", style="dim")

    for name, vals in rows:
        table.add_row(
            name,
            vals["type"],
            f"{vals['L']:.0f}",
            f"{vals['contrast']:.0f}" if vals["contrast"] else "-",
            f"{vals['chroma']:.0f}" if vals["chroma"] else "-",
            str(vals["n"]),
        )

    console.print(table)
    console.print()
    return True


def show_stats(history: History):
    """Display learning statistics with nice tables."""
    console = Console()
    data = history.data
    theme_posteriors = data.get("theme_posteriors", {})

    if not theme_posteriors:
        console.print("No learning data yet. Run: ghostty-ambient --daemon")
        return

    console.print()

    # By Time of Day
    _make_preference_table(
        console,
        theme_posteriors,
        "By Time of Day",
        [
            ("time:morning", "Morning"),
            ("time:afternoon", "Afternoon"),
            ("time:evening", "Evening"),
            ("time:night", "Night"),
        ],
    )

    # By Ambient Light
    _make_preference_table(
        console,
        theme_posteriors,
        "By Ambient Light",
        [
            ("lux:dim", "Dim"),
            ("lux:office", "Office"),
            ("lux:bright", "Bright"),
            ("lux:daylight", "Daylight"),
        ],
    )

    # By System Appearance
    _make_preference_table(
        console,
        theme_posteriors,
        "By System Mode",
        [
            ("system:light", "Light mode"),
            ("system:dark", "Dark mode"),
        ],
    )

    # By Power Source
    _make_preference_table(
        console,
        theme_posteriors,
        "By Power",
        [
            ("power:ac", "AC"),
            ("power:battery_high", "Battery"),
            ("power:battery_low", "Low battery"),
        ],
    )

    # Summary line
    snapshots = data.get("recent_snapshots", [])
    total_obs = sum(
        len(ctx.get("color", {}).get("observations", []))
        for ctx in theme_posteriors.values()
    )
    console.print(f"[dim]{len(theme_posteriors)} contexts | {total_obs} total observations | {len(snapshots)} recent snapshots[/dim]")
