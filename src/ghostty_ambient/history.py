"""
Bayesian preference learning for theme selection.

Uses factorized Beta distributions to model theme preferences:
- Each theme gets a Beta(α, β) distribution per factor (not joint contexts)
- α = 1 + times_chosen (prior + successes)
- β = 1 + times_other_chosen_in_context (prior + failures)
- Preference strength = α / (α + β) = beta mean

Factors are independent dimensions:
- Time: morning (6-12), afternoon (12-17), evening (17-21), night (21-6)
- Lux: moonlight, dim, ambient, office, bright, daylight, sunlight
- Weather: clear, cloudy, rain, snow, other
- System: dark, light (OS appearance)
- Day: weekend, weekday
- Power: ac, battery_high, battery_low
- Font: normalized font family name
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ghostty_ambient.color import hex_to_lab
from ghostty_ambient.color_posterior import ThemePreferenceModel
from ghostty_ambient.factors import FactorRegistry

HISTORY_FILE = Path.home() / ".config/ghostty-ambient/history.json"


def _build_context(
    hour: int,
    lux: float | None = None,
    weather_code: int | None = None,
    system_appearance: str | None = None,
    power_source: str | None = None,
    font: str | None = None,
) -> dict[str, Any]:
    """Build context dict for factor bucket calculation."""
    return {
        "hour": hour,
        "lux": lux,
        "weather_code": weather_code,
        "system_appearance": system_appearance,
        "power_source": power_source,
        "font": font,
    }


def get_time_bucket(hour: int) -> str:
    """Get time bucket for preference tracking."""
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def get_lux_bucket(lux: float | None) -> str:
    """
    Get lux bucket for preference tracking.

    Based on real-world lighting conditions:
        0-10:       moonlight, night, screen-only
        10-50:      dim, candlelit, nightlight
        50-200:     ambient, evening home, cozy
        200-500:    office, typical workspace
        500-2000:   bright, near window, well-lit
        2000-10000: daylight, overcast outdoor
        10000+:     sunlight, direct outdoor
    """
    if lux is None:
        return "unknown"
    elif lux < 10:
        return "moonlight"
    elif lux < 50:
        return "dim"
    elif lux < 200:
        return "ambient"
    elif lux < 500:
        return "office"
    elif lux < 2000:
        return "bright"
    elif lux < 10000:
        return "daylight"
    else:
        return "sunlight"


def get_weather_bucket(weather_code: int | None) -> str:
    """Map WMO weather code to bucket for preference tracking."""
    if weather_code is None:
        return "unknown"
    elif weather_code <= 1:  # Clear
        return "clear"
    elif weather_code <= 3:  # Cloudy/overcast
        return "cloudy"
    elif weather_code in range(51, 68) or weather_code in range(80, 83):  # Rain/drizzle
        return "rain"
    elif weather_code in range(71, 78) or weather_code in range(85, 87):  # Snow
        return "snow"
    else:
        return "other"


def get_system_bucket(appearance: str | None) -> str:
    """Get system appearance bucket for preference tracking."""
    if appearance in ("dark", "light"):
        return appearance
    return "unknown"


def get_day_bucket() -> str:
    """Get day of week bucket for preference tracking."""
    return "weekend" if datetime.now().weekday() >= 5 else "weekday"


def get_power_bucket(power: str | None) -> str:
    """Get power source bucket for preference tracking."""
    if power in ("ac", "battery_high", "battery_low"):
        return power
    return "unknown"


def get_font_bucket(font: str | None) -> str:
    """
    Normalize font name to a bucket.

    E.g., "Rec Mono Semicasual" → "rec_mono_semicasual"
    """
    if not font:
        return "unknown"

    # Normalize: lowercase, replace spaces with underscores
    return font.lower().replace(" ", "_")


def get_context_mismatch_penalty(
    theme_brightness: int,
    system_appearance: str | None,
    time_bucket: str,
    lux: float | None,
) -> float:
    """
    Calculate penalty for theme brightness vs context mismatch.

    Penalizes:
    - Light themes (brightness > 150) in dark contexts
    - Dark themes (brightness < 100) in light contexts

    Requires 2+ agreeing signals before applying penalty.

    Args:
        theme_brightness: Theme background brightness (0-255)
        system_appearance: System dark/light mode
        time_bucket: Time of day bucket (morning/afternoon/evening/night)
        lux: Ambient light reading

    Returns:
        Penalty value 0-50 (subtracted from final score)
    """
    # Count dark/light context signals
    dark_signals = 0
    light_signals = 0

    # System appearance (strongest signal, weight 2)
    if system_appearance == "dark":
        dark_signals += 2
    elif system_appearance == "light":
        light_signals += 2

    # Time of day
    if time_bucket == "night":
        dark_signals += 1
    elif time_bucket in ("morning", "afternoon"):
        light_signals += 1

    # Ambient light
    if lux is not None:
        if lux < 200:  # dim/ambient
            dark_signals += 1
        elif lux > 500:  # bright/daylight
            light_signals += 1

    # Apply penalty based on mismatch
    penalty = 0.0

    # Light theme in dark context
    if dark_signals >= 2 and theme_brightness > 150:
        # Penalty increases with brightness and signal strength
        brightness_excess = theme_brightness - 150  # 0-105
        penalty = (brightness_excess / 100) * dark_signals * 15

    # Dark theme in light context
    elif light_signals >= 2 and theme_brightness < 100:
        brightness_deficit = 100 - theme_brightness  # 0-100
        penalty = (brightness_deficit / 100) * light_signals * 15

    return min(50.0, penalty)  # Cap at 50 points


class History:
    """Manages theme selection history and Bayesian preference learning."""

    def __init__(self):
        self.data = self._load()
        # Initialize theme preference model from stored data
        posteriors_data = self.data.get("theme_posteriors")
        self.theme_model = ThemePreferenceModel(posteriors_data)

    def _load(self) -> dict:
        """Load history from disk."""
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE) as f:
                    return json.load(f)
            except Exception:
                pass

        # Initialize empty history
        return {
            "events": [],
            "global_beta": {},  # {theme: {"alpha": 1, "beta": 1}}
            "favorites": [],
            "disliked": [],
            "theme_posteriors": {},  # Theme preference posteriors (color, contrast, chroma)
        }

    def _save(self):
        """Save history to disk."""
        # Update theme posteriors in data before saving
        self.data["theme_posteriors"] = self.theme_model.to_dict()
        # Remove legacy key if present
        self.data.pop("color_posteriors", None)
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.data, f, indent=2)

    def record_choice(
        self,
        theme_name: str,
        lux: float | None,
        hour: int,
        available_themes: list[str],
        source: str = "recommended",
        weather_code: int | None = None,
        system_appearance: str | None = None,
        power_source: str | None = None,
        font: str | None = None,
        background_hex: str | None = None,
        foreground_hex: str | None = None,
        palette_chromas: list[float] | None = None,
    ):
        """
        Record a theme selection and update Beta distributions.

        Args:
            theme_name: The chosen theme
            lux: Current lux reading (or None)
            hour: Current hour (0-23)
            available_themes: List of themes that were available to choose from
            source: How theme was chosen ("recommended", "browsed", "favorite")
            weather_code: WMO weather code (optional)
            system_appearance: System dark/light mode (optional)
            power_source: Power source - ac/battery_high/battery_low (optional)
            font: Current font family (optional)
            background_hex: Theme background color hex (optional, for color learning)
            foreground_hex: Theme foreground color hex (optional, for contrast learning)
            palette_chromas: List of palette color chromas (optional, for chroma learning)
        """
        # Build context and get factor buckets via registry
        context = _build_context(hour, lux, weather_code, system_appearance, power_source, font)
        factors = FactorRegistry.get_all_buckets(context)

        # Log the event with all context and bucket values
        event = {
            "timestamp": datetime.now().isoformat(),
            "theme": theme_name,
            "lux": lux,
            "hour": hour,
            "weather_code": weather_code,
            "system_appearance": system_appearance,
            "power_source": power_source,
            "font": font,
            "source": source,
        }
        # Add bucket values to event for debugging/analysis
        for factor_name, bucket in factors.items():
            event[f"{factor_name}_bucket"] = bucket
        self.data["events"].append(event)

        # Keep only last 1000 events
        if len(self.data["events"]) > 1000:
            self.data["events"] = self.data["events"][-1000:]

        # Update factorized Beta distributions (each factor independently)
        self._update_factor_beta(theme_name, factors, available_themes)

        # Update global Beta (context-independent preference)
        if "global_beta" not in self.data:
            self.data["global_beta"] = {}
        if theme_name not in self.data["global_beta"]:
            self.data["global_beta"][theme_name] = {"alpha": 1, "beta": 1}
        self.data["global_beta"][theme_name]["alpha"] += 1

        # Soft penalty for non-chosen themes globally
        for theme in available_themes:
            if theme != theme_name:
                if theme not in self.data["global_beta"]:
                    self.data["global_beta"][theme] = {"alpha": 1, "beta": 1}
                self.data["global_beta"][theme]["beta"] += 0.1

        # Update theme preference posteriors if colors provided
        if background_hex:
            bg_lab = hex_to_lab(background_hex)
            fg_lab = hex_to_lab(foreground_hex) if foreground_hex else None
            self.theme_model.record_theme(
                bg_lab=bg_lab,
                fg_lab=fg_lab,
                palette_chromas=palette_chromas,
                factors=factors,
            )

        self._save()

    def _update_factor_beta(
        self, chosen: str, factors: dict[str, str], available: list[str]
    ):
        """
        Update factorized Beta parameters.

        Instead of storing joint contexts like "afternoon_low_lux_cloudy_office",
        we store each factor independently: "time:afternoon", "lux:low_lux", etc.
        This allows learning from sparse data and generalizing across combinations.
        """
        if "factor_beta" not in self.data:
            self.data["factor_beta"] = {}

        for factor_type, factor_value in factors.items():
            key = f"{factor_type}:{factor_value}"

            # Increment α for chosen theme
            if chosen not in self.data["factor_beta"]:
                self.data["factor_beta"][chosen] = {}
            if key not in self.data["factor_beta"][chosen]:
                self.data["factor_beta"][chosen][key] = {"alpha": 1, "beta": 1}
            self.data["factor_beta"][chosen][key]["alpha"] += 1

            # Increment β for non-chosen themes (soft penalty)
            for theme in available:
                if theme != chosen:
                    if theme not in self.data["factor_beta"]:
                        self.data["factor_beta"][theme] = {}
                    if key not in self.data["factor_beta"][theme]:
                        self.data["factor_beta"][theme][key] = {"alpha": 1, "beta": 1}
                    self.data["factor_beta"][theme][key]["beta"] += 0.2

    def get_bayesian_score(
        self,
        theme_name: str,
        hour: int,
        lux: float | None,
        weather_code: int | None = None,
        system_appearance: str | None = None,
        power_source: str | None = None,
        font: str | None = None,
        sample: bool = False,
    ) -> float:
        """
        Get factorized Bayesian preference score for a theme.

        Uses independent factors (time, lux, weather, system, day, power, font)
        instead of joint contexts. This allows learning from sparse data and
        generalizing across unseen combinations.

        Each factor is weighted by confidence (how much data we have for it).
        Factors with more observations contribute more to the final score.

        Args:
            theme_name: Theme to score
            hour: Current hour (0-23)
            lux: Current lux reading
            weather_code: WMO weather code
            system_appearance: System dark/light mode
            power_source: Power source - ac/battery_high/battery_low
            font: Current font family
            sample: If True, sample from Beta (Thompson sampling for exploration)
                    If False, use mean (exploitation)

        Returns:
            Score in [0, 1] representing preference strength
        """
        import random

        def get_value(params: dict, use_sample: bool) -> float:
            alpha, beta = params["alpha"], params["beta"]
            if use_sample:
                return random.betavariate(alpha, beta)
            return alpha / (alpha + beta)

        # Build context and get factor buckets via registry
        context = _build_context(hour, lux, weather_code, system_appearance, power_source, font)
        factors = FactorRegistry.get_all_buckets(context)

        total_score = 0.0
        total_weight = 0.0

        # Score each factor independently using registry
        for factor in FactorRegistry.all():
            factor_value = factors.get(factor.name, "unknown")
            key = f"{factor.name}:{factor_value}"
            params = (
                self.data.get("factor_beta", {})
                .get(theme_name, {})
                .get(key, {"alpha": 1, "beta": 1})
            )

            # Confidence weighting: factors with more data get more weight
            # n = observations for this factor (subtract 2 for prior)
            n = params["alpha"] + params["beta"] - 2
            # Weight ranges from 0.15 (no data) to 0.25 (5+ observations)
            base_weight = 0.15 + 0.1 * min(1.0, n / 5)
            # Apply factor's weight multiplier (e.g., 2.0 for system appearance)
            weight = base_weight * factor.weight_multiplier

            value = get_value(params, sample)
            total_score += weight * value
            total_weight += weight

        # Add global factor (always included with fixed weight)
        global_params = self.data.get("global_beta", {}).get(
            theme_name, {"alpha": 1, "beta": 1}
        )
        global_value = get_value(global_params, sample)
        total_score += 0.2 * global_value
        total_weight += 0.2

        # Normalize to [0, 1]
        return total_score / total_weight if total_weight > 0 else 0.5

    def get_choice_count(self, theme_name: str) -> int:
        """Get total number of times a theme was chosen."""
        params = self.data.get("global_beta", {}).get(
            theme_name, {"alpha": 1, "beta": 1}
        )
        return int(params["alpha"] - 1)  # Subtract prior

    def is_favorite(self, theme_name: str) -> bool:
        """Check if theme is marked as favorite."""
        return theme_name in self.data.get("favorites", [])

    def is_disliked(self, theme_name: str) -> bool:
        """Check if theme is marked as disliked."""
        return theme_name in self.data.get("disliked", [])

    def add_favorite(self, theme_name: str):
        """Mark a theme as favorite."""
        if "favorites" not in self.data:
            self.data["favorites"] = []
        if theme_name not in self.data["favorites"]:
            self.data["favorites"].append(theme_name)
        # Remove from disliked if present
        if theme_name in self.data.get("disliked", []):
            self.data["disliked"].remove(theme_name)
        self._save()

    def add_dislike(self, theme_name: str):
        """Mark a theme as disliked."""
        if "disliked" not in self.data:
            self.data["disliked"] = []
        if theme_name not in self.data["disliked"]:
            self.data["disliked"].append(theme_name)
        # Remove from favorites if present
        if theme_name in self.data.get("favorites", []):
            self.data["favorites"].remove(theme_name)
        self._save()

    def remove_favorite(self, theme_name: str):
        """Remove a theme from favorites."""
        if theme_name in self.data.get("favorites", []):
            self.data["favorites"].remove(theme_name)
            self._save()

    def remove_dislike(self, theme_name: str):
        """Remove a theme from disliked."""
        if theme_name in self.data.get("disliked", []):
            self.data["disliked"].remove(theme_name)
            self._save()

    def get_recent_events(self, count: int = 10) -> list[dict]:
        """Get most recent theme change events."""
        return self.data.get("events", [])[-count:][::-1]

    def get_stats(self) -> dict:
        """Get summary statistics."""
        events = self.data.get("events", [])
        themes_used = set(e["theme"] for e in events)

        return {
            "total_events": len(events),
            "unique_themes": len(themes_used),
            "favorites_count": len(self.data.get("favorites", [])),
            "disliked_count": len(self.data.get("disliked", [])),
        }

    def reset_learning(self, keep_favorites: bool = True) -> dict:
        """
        Clear learned preferences and start fresh.

        Resets:
        - Theme posteriors (color, contrast, chroma preferences)
        - Factor beta distributions (per-theme preferences)
        - Global beta distributions
        - Recent snapshots
        - Event history

        Keeps (optionally):
        - Favorites list
        - Disliked list

        Args:
            keep_favorites: Whether to preserve favorites/disliked lists

        Returns:
            Dict with counts of what was cleared
        """
        stats = {
            "color_posteriors_cleared": len(self.theme_model.color_posteriors),
            "contrast_posteriors_cleared": len(self.theme_model.contrast_posteriors),
            "chroma_posteriors_cleared": len(self.theme_model.chroma_posteriors),
            "factor_betas_cleared": len(self.data.get("factor_beta", {})),
            "events_cleared": len(self.data.get("events", [])),
            "snapshots_cleared": len(self.data.get("recent_snapshots", [])),
        }

        # Reset theme model
        self.theme_model = ThemePreferenceModel()

        # Clear learning data
        self.data["theme_posteriors"] = {}
        self.data.pop("color_posteriors", None)  # Remove legacy
        self.data["factor_beta"] = {}
        self.data["global_beta"] = {}
        self.data["events"] = []
        self.data["recent_snapshots"] = []

        if not keep_favorites:
            stats["favorites_cleared"] = len(self.data.get("favorites", []))
            stats["disliked_cleared"] = len(self.data.get("disliked", []))
            self.data["favorites"] = []
            self.data["disliked"] = []

        self._save()
        return stats

    def get_color_score(
        self,
        background_hex: str,
        hour: int,
        lux: float | None = None,
        weather_code: int | None = None,
        system_appearance: str | None = None,
        power_source: str | None = None,
        font: str | None = None,
    ) -> float:
        """
        Get color-based preference score for a theme background.

        Uses Bayesian color posterior to score how close the theme's
        background color is to the predicted ideal for current context.

        Args:
            background_hex: Theme background color hex
            hour: Current hour (0-23)
            lux: Current lux reading
            weather_code: WMO weather code
            system_appearance: System dark/light mode
            power_source: Power source
            font: Current font family

        Returns:
            Score 0-100 where higher = closer to predicted ideal color
        """
        context = _build_context(
            hour, lux, weather_code, system_appearance, power_source, font
        )
        factors = FactorRegistry.get_all_buckets(context)
        lab = hex_to_lab(background_hex)
        return self.theme_model.score_theme(lab, factors)

    def get_ideal_color(
        self,
        hour: int,
        lux: float | None = None,
        weather_code: int | None = None,
        system_appearance: str | None = None,
        power_source: str | None = None,
        font: str | None = None,
    ) -> tuple[tuple[float, float, float], float] | None:
        """
        Get predicted ideal background color for current context.

        Returns None if not enough data for a confident prediction.
        Use this for generating new optimal themes.

        Args:
            hour: Current hour (0-23)
            lux: Current lux reading
            weather_code: WMO weather code
            system_appearance: System dark/light mode
            power_source: Power source
            font: Current font family

        Returns:
            ((L, a, b), confidence) or None if insufficient data
        """
        context = _build_context(
            hour, lux, weather_code, system_appearance, power_source, font
        )
        factors = FactorRegistry.get_all_buckets(context)
        return self.theme_model.get_ideal_color(factors)

    def get_theme_model_stats(self) -> dict:
        """Get statistics about theme preference learning."""
        return self.theme_model.get_stats()

    def record_snapshot(
        self,
        theme_name: str,
        lux: float | None,
        hour: int,
        weather_code: int | None = None,
        system_appearance: str | None = None,
        power_source: str | None = None,
        font: str | None = None,
        background_hex: str | None = None,
        foreground_hex: str | None = None,
        palette_chromas: list[float] | None = None,
    ):
        """
        Record a snapshot observation (passive learning).

        Unlike record_choice(), this doesn't penalize other themes.
        It only increments alpha for the observed theme in each factor.
        Used by the background daemon when Ghostty is active.

        Args:
            theme_name: Currently active theme
            lux: Current lux reading
            hour: Current hour (0-23)
            weather_code: WMO weather code
            system_appearance: System dark/light mode
            power_source: Power source
            font: Current font family
            background_hex: Theme background color hex (optional, for color learning)
            foreground_hex: Theme foreground color hex (optional, for contrast learning)
            palette_chromas: List of palette color chromas (optional, for chroma learning)
        """
        # Build context and get factor buckets via registry
        context = _build_context(hour, lux, weather_code, system_appearance, power_source, font)
        factors = FactorRegistry.get_all_buckets(context)

        # Only update alpha for observed theme (no beta updates)
        if "factor_beta" not in self.data:
            self.data["factor_beta"] = {}

        for factor_type, factor_value in factors.items():
            key = f"{factor_type}:{factor_value}"
            if theme_name not in self.data["factor_beta"]:
                self.data["factor_beta"][theme_name] = {}
            if key not in self.data["factor_beta"][theme_name]:
                self.data["factor_beta"][theme_name][key] = {"alpha": 1, "beta": 1}
            # Smaller increment for snapshots (passive observation)
            self.data["factor_beta"][theme_name][key]["alpha"] += 0.2

        # Update global too
        if "global_beta" not in self.data:
            self.data["global_beta"] = {}
        if theme_name not in self.data["global_beta"]:
            self.data["global_beta"][theme_name] = {"alpha": 1, "beta": 1}
        self.data["global_beta"][theme_name]["alpha"] += 0.2

        # Track recent snapshots for CLI display
        if "recent_snapshots" not in self.data:
            self.data["recent_snapshots"] = []
        self.data["recent_snapshots"].append({
            "timestamp": datetime.now().isoformat(),
            "theme": theme_name,
            "factors": factors,
        })
        # Keep last 100 snapshots
        self.data["recent_snapshots"] = self.data["recent_snapshots"][-100:]

        # Update theme preference posteriors if colors provided
        if background_hex:
            bg_lab = hex_to_lab(background_hex)
            fg_lab = hex_to_lab(foreground_hex) if foreground_hex else None
            self.theme_model.record_theme(
                bg_lab=bg_lab,
                fg_lab=fg_lab,
                palette_chromas=palette_chromas,
                factors=factors,
            )

        self._save()
