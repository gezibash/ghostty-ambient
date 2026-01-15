"""Minimal theme picker using simple-term-menu."""

from simple_term_menu import TerminalMenu


def pick_theme(
    recommendations: list[dict],
    explore: list[dict] | None = None,
    current_theme: str | None = None,
    header: str | None = None,
) -> dict | None:
    """
    Show interactive theme picker with TAB to switch sections.

    Args:
        recommendations: List of recommended theme dicts (closest to learned ideal)
        explore: List of explore theme dicts (different from usual preferences)
        current_theme: Name of currently active theme
        header: Status header to display above the menu

    Returns:
        Selected theme dict, or None if cancelled.
    """
    # Build recommendations entries
    rec_entries = []
    rec_map = {}
    for theme in recommendations:
        name = theme["name"]
        fav = " ★" if theme.get("_is_favorite") else ""
        current = " (current)" if name == current_theme else ""
        entry = f"{name}{fav}{current}"
        rec_entries.append(entry)
        rec_map[entry] = theme

    # Build explore entries
    exp_entries = []
    exp_map = {}
    if explore:
        for theme in explore:
            entry = theme["name"]
            if theme["name"] == current_theme:
                entry += " (current)"
            exp_entries.append(entry)
            exp_map[entry] = theme

    # Section switching with TAB
    sections = ["recommendations", "explore"] if exp_entries else ["recommendations"]
    current_section = 0

    # Build header prefix
    header_prefix = f"{header}\n\n" if header else ""

    while True:
        if sections[current_section] == "recommendations":
            section_title = "Recommended:  [TAB → explore]" if exp_entries else "Recommended:"
            entries = rec_entries
            theme_map = rec_map
        else:
            section_title = "Explore (different from usual):  [TAB → back]"
            entries = exp_entries
            theme_map = exp_map

        title = f"{header_prefix}{section_title}"

        menu = TerminalMenu(
            entries,
            title=title,
            cursor_index=0,
            accept_keys=("enter", "tab"),
            clear_screen=True,
        )

        idx = menu.show()

        if idx is None:
            return None

        # Check if TAB was pressed to switch sections
        if hasattr(menu, "chosen_accept_key") and menu.chosen_accept_key == "tab":
            current_section = (current_section + 1) % len(sections)
            continue

        selected = entries[idx]
        return theme_map.get(selected)
