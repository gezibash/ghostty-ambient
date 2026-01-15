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


class CircadianFactor(Factor):
    """
    Circadian rhythm factor based on proximity to sunrise/sunset.

    Human hormone levels (melatonin, cortisol) shift based on solar position,
    not clock time. This factor captures the biological relevance of light.

    Buckets:
        - night: More than 1hr after sunset or before sunrise
        - pre_dawn: Within 1hr before sunrise (melatonin still high)
        - golden_hour_morning: Within 30min of sunrise (warm light transition)
        - solar_day: Full daylight (sunrise+30min to sunset-30min)
        - golden_hour_evening: Within 30min of sunset (melatonin rising)
        - twilight: Within 1hr after sunset (blue light sensitivity increases)
    """

    @property
    def name(self) -> str:
        return "circadian"

    @property
    def required_context_keys(self) -> set[str]:
        return {"sunrise", "sunset"}

    def get_bucket(self, context: dict[str, Any]) -> str:
        now = context.get("datetime") or datetime.now()
        sunrise = context.get("sunrise")
        sunset = context.get("sunset")

        if not sunrise or not sunset:
            return "unknown"

        # Calculate minutes from sunrise/sunset
        mins_since_sunrise = (now - sunrise).total_seconds() / 60
        mins_until_sunset = (sunset - now).total_seconds() / 60

        if mins_since_sunrise < -60:  # More than 1hr before sunrise
            return "night"
        elif mins_since_sunrise < 0:  # Within 1hr before sunrise
            return "pre_dawn"
        elif mins_since_sunrise < 30:  # First 30min after sunrise
            return "golden_hour_morning"
        elif mins_until_sunset > 30:  # Daytime
            return "solar_day"
        elif mins_until_sunset > -30:  # 30min before/after sunset
            return "golden_hour_evening"
        elif mins_until_sunset > -60:  # Up to 1hr after sunset
            return "twilight"
        else:
            return "night"


class PressureFactor(Factor):
    """
    Atmospheric pressure factor.

    Low pressure systems correlate with overcast/stormy conditions
    and can affect mood. High pressure correlates with clear skies.

    Buckets:
        - low: < 1000 hPa (storm systems, overcast likely)
        - normal: 1000-1020 hPa (standard conditions)
        - high: > 1020 hPa (clear skies, bright conditions)
    """

    @property
    def name(self) -> str:
        return "pressure"

    @property
    def required_context_keys(self) -> set[str]:
        return {"pressure"}

    def get_bucket(self, context: dict[str, Any]) -> str:
        pressure = context.get("pressure")
        if pressure is None:
            return "unknown"
        if pressure < 1000:
            return "low"
        elif pressure <= 1020:
            return "normal"
        return "high"


class CloudCoverFactor(Factor):
    """
    Cloud cover factor for granular sky conditions.

    More granular than weather codes; directly affects ambient light levels.

    Buckets (aviation-style):
        - clear: 0-10% cloud cover
        - few: 10-25% (mostly clear)
        - scattered: 25-50% (partial clouds)
        - broken: 50-85% (mostly cloudy)
        - overcast: 85-100% (full cloud cover)
    """

    @property
    def name(self) -> str:
        return "clouds"

    @property
    def required_context_keys(self) -> set[str]:
        return {"cloud_cover"}

    def get_bucket(self, context: dict[str, Any]) -> str:
        cloud_cover = context.get("cloud_cover")
        if cloud_cover is None:
            return "unknown"
        if cloud_cover < 10:
            return "clear"
        elif cloud_cover < 25:
            return "few"
        elif cloud_cover < 50:
            return "scattered"
        elif cloud_cover < 85:
            return "broken"
        return "overcast"


class UVIndexFactor(Factor):
    """
    UV index factor for harsh lighting conditions.

    High UV correlates with intense, harsh lighting that may affect
    preferred screen brightness and contrast.

    Buckets:
        - none: UV index 0 (night/heavy overcast)
        - low: 1-2 (low exposure)
        - moderate: 3-5 (moderate)
        - high: 6-7 (high)
        - extreme: 8+ (very high)
    """

    @property
    def name(self) -> str:
        return "uv"

    @property
    def required_context_keys(self) -> set[str]:
        return {"uv_index"}

    def get_bucket(self, context: dict[str, Any]) -> str:
        uv_index = context.get("uv_index")
        if uv_index is None:
            return "unknown"
        if uv_index < 1:
            return "none"
        elif uv_index < 3:
            return "low"
        elif uv_index < 6:
            return "moderate"
        elif uv_index < 8:
            return "high"
        return "extreme"


def register_builtin_factors() -> None:
    """Register all built-in factors with the registry."""
    FactorRegistry.register(TimeFactor())
    FactorRegistry.register(LuxFactor())
    FactorRegistry.register(WeatherFactor())
    FactorRegistry.register(SystemFactor())
    FactorRegistry.register(DayFactor())
    FactorRegistry.register(PowerFactor())
    FactorRegistry.register(FontFactor())
    # Circadian & environmental factors
    FactorRegistry.register(CircadianFactor())
    FactorRegistry.register(PressureFactor())
    FactorRegistry.register(CloudCoverFactor())
    FactorRegistry.register(UVIndexFactor())


# Auto-register on import
register_builtin_factors()
