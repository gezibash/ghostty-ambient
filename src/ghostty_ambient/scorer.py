"""
Theme scoring using Bayesian preference learning.

Scoring combines:
- Bayesian preference: Beta distributions per (theme, context)
- Usage distribution: Themes you use more get boosted
- Dislike penalty: Explicitly blocked themes are penalized
- Context penalty: Brightness/context mismatch penalty

Context includes time bucket, lux bucket, weather bucket, system appearance,
day of week, power source, and font.

Cold start (no history): Uses uniform priors Beta(1,1) giving equal
scores to all themes. Context penalty still applies for reasonable defaults.
"""

from datetime import datetime

from .history import History
from .scoring import score_themes as _score_themes


def score_themes(
    themes: list[dict],
    lux: float | None = None,
    weather_code: int | None = None,
    history: History | None = None,
    include_disliked: bool = False,
    system_appearance: str | None = None,
    power_source: str | None = None,
    font: str | None = None,
) -> list[tuple[float, dict]]:
    """
    Score themes using Bayesian preference + usage distribution.

    Uses factorized Beta distributions learned from historical choices
    to score themes based on current context (time, lux, weather, system, day, power, font).

    Cold start (no history): Uses uniform priors Beta(1,1). Context penalty
    still applies, providing reasonable defaults until learning kicks in.

    Args:
        themes: List of theme dicts with 'brightness' and 'warmth'
        lux: Ambient light reading in lux (optional)
        weather_code: WMO weather code (optional)
        history: History instance for preference learning (optional)
        include_disliked: Whether to include disliked themes
        system_appearance: System dark/light mode (optional)
        power_source: Power source - ac/battery_high/battery_low (optional)
        font: Current font family (optional)

    Returns:
        List of (score, theme) tuples sorted by score descending
    """
    return _score_themes(
        themes,
        history,
        include_disliked,
        lux=lux,
        hour=datetime.now().hour,
        weather_code=weather_code,
        system_appearance=system_appearance,
        power_source=power_source,
        font=font,
    )
