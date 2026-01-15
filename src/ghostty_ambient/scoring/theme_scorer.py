"""Theme scoring using Bayesian color preference learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ghostty_ambient.history import get_context_mismatch_penalty, get_time_bucket

if TYPE_CHECKING:
    from ghostty_ambient.history import History


def score_themes(
    themes: list[dict],
    history: History | None = None,
    include_disliked: bool = False,
    lux: float | None = None,
    hour: int | None = None,
    weather_code: int | None = None,
    system_appearance: str | None = None,
    power_source: str | None = None,
    font: str | None = None,
) -> list[tuple[float, dict]]:
    """
    Score all themes using Bayesian color preference learning.

    Scoring formula:
        final_score = color_score + familiarity_boost - context_penalty - dislike_penalty

    Where:
        - color_score: 0-100 from color posterior (distance to predicted ideal)
        - familiarity_boost: 0-10 for previously chosen themes (soft preference)
        - context_penalty: 0-50 for brightness/context mismatch
        - dislike_penalty: 1000 for explicitly disliked themes

    Cold start (no history): Uses neutral color score (50) for all themes.
    Context penalty still applies, providing reasonable defaults.

    Args:
        themes: List of theme dicts with 'background' color
        history: Optional History for color preference learning
        include_disliked: Whether to include disliked themes
        lux: Current lux reading
        hour: Current hour (0-23)
        weather_code: WMO weather code
        system_appearance: System dark/light mode
        power_source: Power source - ac/battery_high/battery_low
        font: Current font family

    Returns:
        List of (score, theme) tuples sorted by score descending
    """
    from datetime import datetime

    now = datetime.now()
    hour = hour if hour is not None else now.hour

    scored = []

    for theme in themes:
        # Skip disliked themes unless explicitly requested
        if history and not include_disliked and history.is_disliked(theme["name"]):
            continue

        # Calculate color-based preference score
        # Uses Bayesian posterior over LAB color space
        if history and "background" in theme:
            color_score = history.get_color_score(
                theme["background"],
                hour,
                lux,
                weather_code,
                system_appearance,
                power_source,
                font,
            )
        else:
            # No history or no background - neutral score
            color_score = 50.0

        # Familiarity boost: small bonus for themes user has chosen before
        # This preserves some preference for specific themes beyond just color
        familiarity_boost = 0.0
        if history:
            choice_count = history.get_choice_count(theme["name"])
            # Boost scales from 0 to 10, saturating at 5 choices
            familiarity_boost = min(10.0, choice_count * 2.0)
            # Extra boost for favorites
            if history.is_favorite(theme["name"]):
                familiarity_boost += 15.0

        # Context mismatch penalty: penalize themes that don't fit the context
        # e.g., light themes in dark mode + night, dark themes in light mode + bright
        context_penalty = get_context_mismatch_penalty(
            theme.get("brightness", 127),
            system_appearance,
            get_time_bucket(hour),
            lux,
        )

        # Dislike penalty (explicitly blocked themes)
        dislike_penalty = 0.0
        if history and history.is_disliked(theme["name"]):
            dislike_penalty = 1000.0

        # Final score
        final_score = color_score + familiarity_boost - context_penalty - dislike_penalty

        # Add metadata for display
        theme_with_meta = theme.copy()
        theme_with_meta["_color_score"] = color_score
        theme_with_meta["_familiarity_boost"] = familiarity_boost
        theme_with_meta["_context_penalty"] = context_penalty
        theme_with_meta["_final_score"] = final_score

        if history:
            theme_with_meta["_choice_count"] = history.get_choice_count(theme["name"])
            theme_with_meta["_is_favorite"] = history.is_favorite(theme["name"])
            theme_with_meta["_is_disliked"] = history.is_disliked(theme["name"])

        scored.append((final_score, theme_with_meta))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return scored
