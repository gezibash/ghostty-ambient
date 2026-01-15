"""Tests for lib/factors/builtin.py - Built-in factors."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from ghostty_ambient.factors.builtin import (
    CircadianFactor,
    CloudCoverFactor,
    DayFactor,
    FontFactor,
    LuxFactor,
    PowerFactor,
    PressureFactor,
    SystemFactor,
    TimeFactor,
    UVIndexFactor,
    WeatherFactor,
)


class TestTimeFactor:
    """Tests for TimeFactor."""

    def test_name(self):
        assert TimeFactor().name == "time"

    def test_required_context_keys(self):
        assert TimeFactor().required_context_keys == {"hour"}

    @pytest.mark.parametrize(
        "hour,expected",
        [
            (6, "morning"),
            (9, "morning"),
            (11, "morning"),
            (12, "afternoon"),
            (14, "afternoon"),
            (16, "afternoon"),
            (17, "evening"),
            (19, "evening"),
            (20, "evening"),
            (21, "night"),
            (23, "night"),
            (0, "night"),
            (5, "night"),
        ],
    )
    def test_bucket_mapping(self, hour: int, expected: str):
        factor = TimeFactor()
        assert factor.get_bucket({"hour": hour}) == expected

    def test_missing_hour_returns_unknown(self):
        assert TimeFactor().get_bucket({}) == "unknown"


class TestLuxFactor:
    """Tests for LuxFactor."""

    def test_name(self):
        assert LuxFactor().name == "lux"

    @pytest.mark.parametrize(
        "lux,expected",
        [
            (0, "moonlight"),
            (9, "moonlight"),
            (10, "dim"),
            (49, "dim"),
            (50, "ambient"),
            (199, "ambient"),
            (200, "office"),
            (499, "office"),
            (500, "bright"),
            (1999, "bright"),
            (2000, "daylight"),
            (9999, "daylight"),
            (10000, "sunlight"),
            (50000, "sunlight"),
        ],
    )
    def test_bucket_mapping(self, lux: float, expected: str):
        factor = LuxFactor()
        assert factor.get_bucket({"lux": lux}) == expected

    def test_missing_lux_returns_unknown(self):
        assert LuxFactor().get_bucket({}) == "unknown"
        assert LuxFactor().get_bucket({"lux": None}) == "unknown"


class TestWeatherFactor:
    """Tests for WeatherFactor."""

    def test_name(self):
        assert WeatherFactor().name == "weather"

    @pytest.mark.parametrize(
        "code,expected",
        [
            (0, "clear"),
            (1, "clear"),
            (2, "cloudy"),
            (3, "cloudy"),
            (51, "rain"),
            (61, "rain"),
            (80, "rain"),
            (71, "snow"),
            (75, "snow"),
            (85, "snow"),
            (45, "other"),  # fog
            (95, "other"),  # thunderstorm
        ],
    )
    def test_bucket_mapping(self, code: int, expected: str):
        factor = WeatherFactor()
        assert factor.get_bucket({"weather_code": code}) == expected

    def test_missing_weather_returns_unknown(self):
        assert WeatherFactor().get_bucket({}) == "unknown"


class TestSystemFactor:
    """Tests for SystemFactor."""

    def test_name(self):
        assert SystemFactor().name == "system"

    def test_weight_multiplier_is_2(self):
        """System factor should have 2x weight."""
        assert SystemFactor().weight_multiplier == 2.0

    def test_dark_mode(self):
        assert SystemFactor().get_bucket({"system_appearance": "dark"}) == "dark"

    def test_light_mode(self):
        assert SystemFactor().get_bucket({"system_appearance": "light"}) == "light"

    def test_unknown_mode(self):
        assert SystemFactor().get_bucket({"system_appearance": "auto"}) == "unknown"
        assert SystemFactor().get_bucket({}) == "unknown"


class TestDayFactor:
    """Tests for DayFactor."""

    def test_name(self):
        assert DayFactor().name == "day"

    def test_weekday(self):
        with patch("ghostty_ambient.factors.builtin.datetime") as mock_dt:
            mock_dt.now.return_value.weekday.return_value = 0  # Monday
            assert DayFactor().get_bucket({}) == "weekday"

            mock_dt.now.return_value.weekday.return_value = 4  # Friday
            assert DayFactor().get_bucket({}) == "weekday"

    def test_weekend(self):
        with patch("ghostty_ambient.factors.builtin.datetime") as mock_dt:
            mock_dt.now.return_value.weekday.return_value = 5  # Saturday
            assert DayFactor().get_bucket({}) == "weekend"

            mock_dt.now.return_value.weekday.return_value = 6  # Sunday
            assert DayFactor().get_bucket({}) == "weekend"


class TestPowerFactor:
    """Tests for PowerFactor."""

    def test_name(self):
        assert PowerFactor().name == "power"

    def test_valid_power_sources(self):
        factor = PowerFactor()
        assert factor.get_bucket({"power_source": "ac"}) == "ac"
        assert factor.get_bucket({"power_source": "battery_high"}) == "battery_high"
        assert factor.get_bucket({"power_source": "battery_low"}) == "battery_low"

    def test_invalid_returns_unknown(self):
        factor = PowerFactor()
        assert factor.get_bucket({"power_source": "charging"}) == "unknown"
        assert factor.get_bucket({}) == "unknown"


class TestFontFactor:
    """Tests for FontFactor."""

    def test_name(self):
        assert FontFactor().name == "font"

    def test_normalizes_font_name(self):
        factor = FontFactor()
        assert factor.get_bucket({"font": "Rec Mono Semicasual"}) == "rec_mono_semicasual"
        assert factor.get_bucket({"font": "JetBrains Mono"}) == "jetbrains_mono"
        assert factor.get_bucket({"font": "Monaco"}) == "monaco"

    def test_empty_font_returns_unknown(self):
        factor = FontFactor()
        assert factor.get_bucket({"font": ""}) == "unknown"
        assert factor.get_bucket({"font": None}) == "unknown"
        assert factor.get_bucket({}) == "unknown"


class TestCircadianFactor:
    """Tests for CircadianFactor."""

    def test_name(self):
        assert CircadianFactor().name == "circadian"

    def test_required_context_keys(self):
        assert CircadianFactor().required_context_keys == {"sunrise", "sunset"}

    def test_missing_sunrise_sunset_returns_unknown(self):
        factor = CircadianFactor()
        assert factor.get_bucket({}) == "unknown"
        assert factor.get_bucket({"sunrise": datetime(2024, 1, 15, 7, 0)}) == "unknown"
        assert factor.get_bucket({"sunset": datetime(2024, 1, 15, 17, 0)}) == "unknown"

    def test_night_before_pre_dawn(self):
        """More than 1hr before sunrise = night."""
        factor = CircadianFactor()
        sunrise = datetime(2024, 1, 15, 7, 0)
        sunset = datetime(2024, 1, 15, 17, 0)
        now = datetime(2024, 1, 15, 5, 0)  # 2hrs before sunrise
        assert factor.get_bucket({"sunrise": sunrise, "sunset": sunset, "datetime": now}) == "night"

    def test_pre_dawn(self):
        """Within 1hr before sunrise = pre_dawn."""
        factor = CircadianFactor()
        sunrise = datetime(2024, 1, 15, 7, 0)
        sunset = datetime(2024, 1, 15, 17, 0)
        now = datetime(2024, 1, 15, 6, 30)  # 30min before sunrise
        assert factor.get_bucket({"sunrise": sunrise, "sunset": sunset, "datetime": now}) == "pre_dawn"

    def test_golden_hour_morning(self):
        """Within 30min after sunrise = golden_hour_morning."""
        factor = CircadianFactor()
        sunrise = datetime(2024, 1, 15, 7, 0)
        sunset = datetime(2024, 1, 15, 17, 0)
        now = datetime(2024, 1, 15, 7, 15)  # 15min after sunrise
        assert factor.get_bucket({"sunrise": sunrise, "sunset": sunset, "datetime": now}) == "golden_hour_morning"

    def test_solar_day(self):
        """Middle of day = solar_day."""
        factor = CircadianFactor()
        sunrise = datetime(2024, 1, 15, 7, 0)
        sunset = datetime(2024, 1, 15, 17, 0)
        now = datetime(2024, 1, 15, 12, 0)  # Noon
        assert factor.get_bucket({"sunrise": sunrise, "sunset": sunset, "datetime": now}) == "solar_day"

    def test_golden_hour_evening(self):
        """Within 30min of sunset = golden_hour_evening."""
        factor = CircadianFactor()
        sunrise = datetime(2024, 1, 15, 7, 0)
        sunset = datetime(2024, 1, 15, 17, 0)
        now = datetime(2024, 1, 15, 16, 45)  # 15min before sunset
        assert factor.get_bucket({"sunrise": sunrise, "sunset": sunset, "datetime": now}) == "golden_hour_evening"

    def test_twilight(self):
        """Within 1hr after sunset = twilight."""
        factor = CircadianFactor()
        sunrise = datetime(2024, 1, 15, 7, 0)
        sunset = datetime(2024, 1, 15, 17, 0)
        now = datetime(2024, 1, 15, 17, 30)  # 30min after sunset
        assert factor.get_bucket({"sunrise": sunrise, "sunset": sunset, "datetime": now}) == "twilight"

    def test_night_after_twilight(self):
        """More than 1hr after sunset = night."""
        factor = CircadianFactor()
        sunrise = datetime(2024, 1, 15, 7, 0)
        sunset = datetime(2024, 1, 15, 17, 0)
        now = datetime(2024, 1, 15, 19, 0)  # 2hrs after sunset
        assert factor.get_bucket({"sunrise": sunrise, "sunset": sunset, "datetime": now}) == "night"


class TestPressureFactor:
    """Tests for PressureFactor."""

    def test_name(self):
        assert PressureFactor().name == "pressure"

    def test_required_context_keys(self):
        assert PressureFactor().required_context_keys == {"pressure"}

    @pytest.mark.parametrize(
        "pressure,expected",
        [
            (980, "low"),
            (999, "low"),
            (1000, "normal"),
            (1010, "normal"),
            (1020, "normal"),
            (1021, "high"),
            (1040, "high"),
        ],
    )
    def test_bucket_mapping(self, pressure: float, expected: str):
        factor = PressureFactor()
        assert factor.get_bucket({"pressure": pressure}) == expected

    def test_missing_pressure_returns_unknown(self):
        assert PressureFactor().get_bucket({}) == "unknown"
        assert PressureFactor().get_bucket({"pressure": None}) == "unknown"


class TestCloudCoverFactor:
    """Tests for CloudCoverFactor."""

    def test_name(self):
        assert CloudCoverFactor().name == "clouds"

    def test_required_context_keys(self):
        assert CloudCoverFactor().required_context_keys == {"cloud_cover"}

    @pytest.mark.parametrize(
        "cloud_cover,expected",
        [
            (0, "clear"),
            (9, "clear"),
            (10, "few"),
            (24, "few"),
            (25, "scattered"),
            (49, "scattered"),
            (50, "broken"),
            (84, "broken"),
            (85, "overcast"),
            (100, "overcast"),
        ],
    )
    def test_bucket_mapping(self, cloud_cover: int, expected: str):
        factor = CloudCoverFactor()
        assert factor.get_bucket({"cloud_cover": cloud_cover}) == expected

    def test_missing_cloud_cover_returns_unknown(self):
        assert CloudCoverFactor().get_bucket({}) == "unknown"
        assert CloudCoverFactor().get_bucket({"cloud_cover": None}) == "unknown"


class TestUVIndexFactor:
    """Tests for UVIndexFactor."""

    def test_name(self):
        assert UVIndexFactor().name == "uv"

    def test_required_context_keys(self):
        assert UVIndexFactor().required_context_keys == {"uv_index"}

    @pytest.mark.parametrize(
        "uv_index,expected",
        [
            (0, "none"),
            (0.5, "none"),
            (1, "low"),
            (2, "low"),
            (3, "moderate"),
            (5, "moderate"),
            (6, "high"),
            (7, "high"),
            (8, "extreme"),
            (11, "extreme"),
        ],
    )
    def test_bucket_mapping(self, uv_index: float, expected: str):
        factor = UVIndexFactor()
        assert factor.get_bucket({"uv_index": uv_index}) == expected

    def test_missing_uv_index_returns_unknown(self):
        assert UVIndexFactor().get_bucket({}) == "unknown"
        assert UVIndexFactor().get_bucket({"uv_index": None}) == "unknown"
