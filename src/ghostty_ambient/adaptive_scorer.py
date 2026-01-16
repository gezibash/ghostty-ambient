"""
Theme scoring using the adaptive preference model.

This replaces the old Bayesian scorer with embedding-based scoring
and phase-aware recommendations.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from .adaptive_model import AdaptivePreferenceModel
from .factors import FactorRegistry

if TYPE_CHECKING:
    from .history import History


def build_context(
    lux: float | None = None,
    hour: int | None = None,
    weather_code: int | None = None,
    system_appearance: str | None = None,
    power_source: str | None = None,
    font: str | None = None,
) -> dict[str, str]:
    """Build context dict from raw values."""
    hour = hour if hour is not None else datetime.now().hour

    raw_context = {
        "hour": hour,
        "lux": lux,
        "weather_code": weather_code,
        "system_appearance": system_appearance,
        "power_source": power_source,
        "font": font,
    }

    # Use factor registry to bucket values
    return FactorRegistry.get_all_buckets(raw_context)


def score_themes_adaptive(
    themes: list[dict],
    model: AdaptivePreferenceModel,
    lux: float | None = None,
    weather_code: int | None = None,
    system_appearance: str | None = None,
    power_source: str | None = None,
    font: str | None = None,
    include_disliked: bool = False,
) -> list[tuple[float, dict]]:
    """
    Score themes using the adaptive preference model.

    Args:
        themes: List of theme dicts
        model: AdaptivePreferenceModel instance
        lux: Ambient light reading
        weather_code: WMO weather code
        system_appearance: System dark/light mode
        power_source: Power source
        font: Current font
        include_disliked: Whether to include disliked themes

    Returns:
        List of (score, theme) tuples sorted by score descending
    """
    # Build context
    context = build_context(
        lux=lux,
        hour=datetime.now().hour,
        weather_code=weather_code,
        system_appearance=system_appearance,
        power_source=power_source,
        font=font,
    )

    # Get recommendations from model
    recommendations = model.recommend(context, themes, k=len(themes))

    # Filter disliked if needed
    if not include_disliked:
        recommendations = [t for t in recommendations if t["name"] not in model.disliked]

    # Convert to (score, theme) tuples
    return [(t.get("_score", 0), t) for t in recommendations]


def record_choice_adaptive(
    model: AdaptivePreferenceModel,
    theme: dict,
    lux: float | None = None,
    weather_code: int | None = None,
    system_appearance: str | None = None,
    power_source: str | None = None,
    font: str | None = None,
    source: str = "picker",
) -> None:
    """
    Record a theme choice in the adaptive model.

    Args:
        model: AdaptivePreferenceModel instance
        theme: Theme dict that was chosen
        lux: Ambient light reading
        weather_code: WMO weather code
        system_appearance: System dark/light mode
        power_source: Power source
        font: Current font
        source: How the theme was chosen
    """
    context = build_context(
        lux=lux,
        hour=datetime.now().hour,
        weather_code=weather_code,
        system_appearance=system_appearance,
        power_source=power_source,
        font=font,
    )

    model.record(theme, context, source=source)
