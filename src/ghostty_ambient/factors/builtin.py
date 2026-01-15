"""Built-in context factors for Bayesian preference learning."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .base import Factor, FactorRegistry


class TimeFactor(Factor):
    """
    Time of day factor.

    Buckets:
        - morning: 6:00-12:00
        - afternoon: 12:00-17:00
        - evening: 17:00-21:00
        - night: 21:00-6:00
    """

    @property
    def name(self) -> str:
        return "time"

    @property
    def required_context_keys(self) -> set[str]:
        return {"hour"}

    def get_bucket(self, context: dict[str, Any]) -> str:
        hour = context.get("hour")
        if hour is None:
            return "unknown"
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        return "night"


class LuxFactor(Factor):
    """
    Ambient light level factor.

    Buckets based on real-world lighting conditions:
        - moonlight: 0-10 lux (night, screen-only)
        - dim: 10-50 lux (candlelit, nightlight)
        - ambient: 50-200 lux (evening home, cozy)
        - office: 200-500 lux (typical workspace)
        - bright: 500-2000 lux (near window, well-lit)
        - daylight: 2000-10000 lux (overcast outdoor)
        - sunlight: 10000+ lux (direct outdoor)
    """

    @property
    def name(self) -> str:
        return "lux"

    @property
    def required_context_keys(self) -> set[str]:
        return {"lux"}

    def get_bucket(self, context: dict[str, Any]) -> str:
        lux = context.get("lux")
        if lux is None:
            return "unknown"
        if lux < 10:
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
        return "sunlight"


class WeatherFactor(Factor):
    """
    Weather condition factor based on WMO weather codes.

    Buckets:
        - clear: codes 0-1 (clear sky)
        - cloudy: codes 2-3 (partly cloudy to overcast)
        - rain: codes 51-67, 80-82 (drizzle, rain, showers)
        - snow: codes 71-77, 85-86 (snow, sleet)
        - other: fog, thunderstorm, etc.
    """

    @property
    def name(self) -> str:
        return "weather"

    @property
    def required_context_keys(self) -> set[str]:
        return {"weather_code"}

    def get_bucket(self, context: dict[str, Any]) -> str:
        weather_code = context.get("weather_code")
        if weather_code is None:
            return "unknown"
        if weather_code <= 1:
            return "clear"
        elif weather_code <= 3:
            return "cloudy"
        elif weather_code in range(51, 68) or weather_code in range(80, 83):
            return "rain"
        elif weather_code in range(71, 78) or weather_code in range(85, 87):
            return "snow"
        return "other"


class SystemFactor(Factor):
    """
    System appearance (dark/light mode) factor.

    This factor has a 2x weight multiplier since it represents
    a deliberate user choice about their preferred appearance.

    Buckets:
        - dark: OS dark mode enabled
        - light: OS light mode enabled
        - unknown: unable to detect
    """

    @property
    def name(self) -> str:
        return "system"

    @property
    def required_context_keys(self) -> set[str]:
        return {"system_appearance"}

    @property
    def weight_multiplier(self) -> float:
        return 2.0  # Deliberate user choice gets higher weight

    def get_bucket(self, context: dict[str, Any]) -> str:
        appearance = context.get("system_appearance")
        if appearance in ("dark", "light"):
            return appearance
        return "unknown"


class DayFactor(Factor):
    """
    Day of week factor.

    Buckets:
        - weekday: Monday-Friday
        - weekend: Saturday-Sunday
    """

    @property
    def name(self) -> str:
        return "day"

    def get_bucket(self, context: dict[str, Any]) -> str:
        # Day factor doesn't need context - uses current time
        return "weekend" if datetime.now().weekday() >= 5 else "weekday"


class PowerFactor(Factor):
    """
    Power source factor.

    Buckets:
        - ac: Connected to power
        - battery_high: On battery, charge > 20%
        - battery_low: On battery, charge <= 20%
        - unknown: Unable to detect
    """

    @property
    def name(self) -> str:
        return "power"

    @property
    def required_context_keys(self) -> set[str]:
        return {"power_source"}

    def get_bucket(self, context: dict[str, Any]) -> str:
        power = context.get("power_source")
        if power in ("ac", "battery_high", "battery_low"):
            return power
        return "unknown"


class FontFactor(Factor):
    """
    Current font family factor.

    Normalizes font names to lowercase with underscores.
    E.g., "Rec Mono Semicasual" → "rec_mono_semicasual"
    """

    @property
    def name(self) -> str:
        return "font"

    @property
    def required_context_keys(self) -> set[str]:
        return {"font"}

    def get_bucket(self, context: dict[str, Any]) -> str:
        font = context.get("font")
        if not font:
            return "unknown"
        return font.lower().replace(" ", "_")


def register_builtin_factors() -> None:
    """Register all built-in factors with the registry."""
    FactorRegistry.register(TimeFactor())
    FactorRegistry.register(LuxFactor())
    FactorRegistry.register(WeatherFactor())
    FactorRegistry.register(SystemFactor())
    FactorRegistry.register(DayFactor())
    FactorRegistry.register(PowerFactor())
    FactorRegistry.register(FontFactor())


# Auto-register on import
register_builtin_factors()
