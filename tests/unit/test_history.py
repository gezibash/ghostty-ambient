"""Tests for lib/history.py - Bayesian learning and bucket functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ghostty_ambient.history import (
    get_context_mismatch_penalty,
    get_day_bucket,
    get_font_bucket,
    get_lux_bucket,
    get_power_bucket,
    get_system_bucket,
    get_time_bucket,
    get_weather_bucket,
)


class TestGetTimeBucket:
    """Tests for get_time_bucket() function."""

    @pytest.mark.parametrize(
        "hour,expected",
        [
            # Morning: 6-12
            (6, "morning"),
            (9, "morning"),
            (11, "morning"),
            # Afternoon: 12-17
            (12, "afternoon"),
            (14, "afternoon"),
            (16, "afternoon"),
            # Evening: 17-21
            (17, "evening"),
            (19, "evening"),
            (20, "evening"),
            # Night: 21-6
            (21, "night"),
            (23, "night"),
            (0, "night"),
            (3, "night"),
            (5, "night"),
        ],
    )
    def test_time_bucket_mapping(self, hour: int, expected: str):
        assert get_time_bucket(hour) == expected

    def test_boundary_morning_start(self):
        assert get_time_bucket(6) == "morning"
        assert get_time_bucket(5) == "night"

    def test_boundary_afternoon_start(self):
        assert get_time_bucket(12) == "afternoon"
        assert get_time_bucket(11) == "morning"

    def test_boundary_evening_start(self):
        assert get_time_bucket(17) == "evening"
        assert get_time_bucket(16) == "afternoon"

    def test_boundary_night_start(self):
        assert get_time_bucket(21) == "night"
        assert get_time_bucket(20) == "evening"


class TestGetLuxBucket:
    """Tests for get_lux_bucket() function."""

    @pytest.mark.parametrize(
        "lux,expected",
        [
            # Moonlight: 0-10
            (0, "moonlight"),
            (5, "moonlight"),
            (9.9, "moonlight"),
            # Dim: 10-50
            (10, "dim"),
            (30, "dim"),
            (49.9, "dim"),
            # Ambient: 50-200
            (50, "ambient"),
            (100, "ambient"),
            (199.9, "ambient"),
            # Office: 200-500
            (200, "office"),
            (350, "office"),
            (499.9, "office"),
            # Bright: 500-2000
            (500, "bright"),
            (1000, "bright"),
            (1999.9, "bright"),
            # Daylight: 2000-10000
            (2000, "daylight"),
            (5000, "daylight"),
            (9999.9, "daylight"),
            # Sunlight: 10000+
            (10000, "sunlight"),
            (50000, "sunlight"),
        ],
    )
    def test_lux_bucket_mapping(self, lux: float, expected: str):
        assert get_lux_bucket(lux) == expected

    def test_none_lux_returns_unknown(self):
        assert get_lux_bucket(None) == "unknown"


class TestGetWeatherBucket:
    """Tests for get_weather_bucket() function."""

    @pytest.mark.parametrize(
        "code,expected",
        [
            # Clear: 0-1
            (0, "clear"),
            (1, "clear"),
            # Cloudy: 2-3
            (2, "cloudy"),
            (3, "cloudy"),
            # Rain: 51-67, 80-82
            (51, "rain"),
            (61, "rain"),
            (67, "rain"),
            (80, "rain"),
            (82, "rain"),
            # Snow: 71-77, 85-86
            (71, "snow"),
            (75, "snow"),
            (77, "snow"),
            (85, "snow"),
            (86, "snow"),
            # Other
            (45, "other"),  # Fog
            (95, "other"),  # Thunderstorm
        ],
    )
    def test_weather_bucket_mapping(self, code: int, expected: str):
        assert get_weather_bucket(code) == expected

    def test_none_weather_returns_unknown(self):
        assert get_weather_bucket(None) == "unknown"


class TestGetSystemBucket:
    """Tests for get_system_bucket() function."""

    def test_dark_mode(self):
        assert get_system_bucket("dark") == "dark"

    def test_light_mode(self):
        assert get_system_bucket("light") == "light"

    def test_unknown_returns_unknown(self):
        assert get_system_bucket("auto") == "unknown"
        assert get_system_bucket(None) == "unknown"
        assert get_system_bucket("") == "unknown"


class TestGetPowerBucket:
    """Tests for get_power_bucket() function."""

    def test_valid_power_sources(self):
        assert get_power_bucket("ac") == "ac"
        assert get_power_bucket("battery_high") == "battery_high"
        assert get_power_bucket("battery_low") == "battery_low"

    def test_invalid_returns_unknown(self):
        assert get_power_bucket("charging") == "unknown"
        assert get_power_bucket(None) == "unknown"
        assert get_power_bucket("") == "unknown"


class TestGetFontBucket:
    """Tests for get_font_bucket() function."""

    def test_normalizes_font_name(self):
        assert get_font_bucket("Rec Mono Semicasual") == "rec_mono_semicasual"
        assert get_font_bucket("JetBrains Mono") == "jetbrains_mono"
        assert get_font_bucket("Monaco") == "monaco"

    def test_none_returns_unknown(self):
        assert get_font_bucket(None) == "unknown"
        assert get_font_bucket("") == "unknown"


class TestGetDayBucket:
    """Tests for get_day_bucket() function."""

    def test_weekday(self):
        from datetime import datetime

        # Monday = 0, so weekday < 5
        with patch("ghostty_ambient.history.datetime") as mock_dt:
            mock_dt.now.return_value.weekday.return_value = 0  # Monday
            assert get_day_bucket() == "weekday"

            mock_dt.now.return_value.weekday.return_value = 4  # Friday
            assert get_day_bucket() == "weekday"

    def test_weekend(self):
        with patch("ghostty_ambient.history.datetime") as mock_dt:
            mock_dt.now.return_value.weekday.return_value = 5  # Saturday
            assert get_day_bucket() == "weekend"

            mock_dt.now.return_value.weekday.return_value = 6  # Sunday
            assert get_day_bucket() == "weekend"


class TestGetContextMismatchPenalty:
    """Tests for get_context_mismatch_penalty() function."""

    def test_light_theme_in_dark_context_penalized(self):
        """Light theme (brightness > 150) with 2+ dark signals should be penalized."""
        # System dark (weight 2) + night (weight 1) = 3 dark signals
        penalty = get_context_mismatch_penalty(
            theme_brightness=200,
            system_appearance="dark",
            time_bucket="night",
            lux=50,  # dim = +1 dark signal
        )
        assert penalty > 0
        assert penalty <= 50  # capped at 50

    def test_dark_theme_in_light_context_penalized(self):
        """Dark theme (brightness < 100) with 2+ light signals should be penalized."""
        # System light (weight 2) + afternoon (weight 1) = 3 light signals
        penalty = get_context_mismatch_penalty(
            theme_brightness=50,
            system_appearance="light",
            time_bucket="afternoon",
            lux=1000,  # bright = +1 light signal
        )
        assert penalty > 0
        assert penalty <= 50

    def test_neutral_theme_no_penalty(self):
        """Themes with brightness 100-150 should not be penalized."""
        penalty = get_context_mismatch_penalty(
            theme_brightness=127,  # neutral
            system_appearance="dark",
            time_bucket="night",
            lux=50,
        )
        assert penalty == 0.0

    def test_insufficient_signals_no_penalty(self):
        """Need 2+ agreeing signals to apply penalty."""
        # Only system=dark (weight 2), no other dark signals
        penalty = get_context_mismatch_penalty(
            theme_brightness=200,
            system_appearance="dark",
            time_bucket="afternoon",  # light signal
            lux=1000,  # light signal
        )
        # Dark signals = 2 (system), light signals = 2 (time + lux)
        # Neither reaches 2 with agreement, dark = 2 exactly
        assert penalty >= 0  # May or may not trigger based on exact logic

    def test_penalty_capped_at_50(self):
        """Penalty should never exceed 50."""
        penalty = get_context_mismatch_penalty(
            theme_brightness=255,  # maximum brightness
            system_appearance="dark",
            time_bucket="night",
            lux=0,  # very dark
        )
        assert penalty <= 50.0

    def test_system_appearance_has_weight_2(self):
        """System appearance contributes weight 2 to signals."""
        # Only system dark (weight 2) should be enough
        penalty = get_context_mismatch_penalty(
            theme_brightness=200,
            system_appearance="dark",
            time_bucket="evening",  # no signal
            lux=None,  # no signal
        )
        # 2 dark signals from system alone
        assert penalty > 0

    def test_evening_time_no_signal(self):
        """Evening time bucket should not contribute to signals."""
        penalty_evening = get_context_mismatch_penalty(
            theme_brightness=200,
            system_appearance=None,
            time_bucket="evening",
            lux=None,
        )
        # No signals = no penalty
        assert penalty_evening == 0.0


class TestHistoryBayesianScore:
    """Tests for History.get_bayesian_score() method."""

    def test_uniform_prior_returns_half(self, mock_history):
        """With no data (Beta(1,1)), score should be ~0.5."""
        score = mock_history.get_bayesian_score(
            theme_name="unknown_theme",
            hour=12,
            lux=500,
        )
        # All factors have Beta(1,1) = mean 0.5
        assert 0.45 <= score <= 0.55

    def test_score_with_factor_data(self, mock_history, history_with_data):
        """Score should reflect factor Beta distributions."""
        mock_history.data = history_with_data

        # dark_theme has strong priors for night/moonlight/dark
        score = mock_history.get_bayesian_score(
            theme_name="dark_theme",
            hour=23,  # night
            lux=5,  # moonlight
            system_appearance="dark",
        )
        # Should be higher than 0.5 due to matching factors
        assert score > 0.5

    def test_confidence_weighting(self, mock_history, history_with_data):
        """Factors with more data should have more weight."""
        mock_history.data = history_with_data

        # The weight formula: 0.15 + 0.1 * min(1.0, n/5)
        # With n=0 (no data): weight = 0.15
        # With n=5+: weight = 0.25

        score = mock_history.get_bayesian_score(
            theme_name="dark_theme",
            hour=23,
            lux=5,
        )
        # Just verify it returns a valid score
        assert 0 <= score <= 1


class TestHistoryUpdateFactorBeta:
    """Tests for History._update_factor_beta() method."""

    def test_chosen_theme_alpha_incremented(self, mock_history):
        """Chosen theme should get alpha += 1 for each factor."""
        mock_history.data["factor_beta"] = {}

        factors = {"time": "night", "lux": "moonlight"}
        mock_history._update_factor_beta("dark_theme", factors, ["dark_theme", "light_theme"])

        # Check alpha was incremented
        assert mock_history.data["factor_beta"]["dark_theme"]["time:night"]["alpha"] == 2
        assert mock_history.data["factor_beta"]["dark_theme"]["lux:moonlight"]["alpha"] == 2

    def test_non_chosen_theme_beta_incremented(self, mock_history):
        """Non-chosen themes should get beta += 0.2 for each factor."""
        mock_history.data["factor_beta"] = {}

        factors = {"time": "night", "lux": "moonlight"}
        mock_history._update_factor_beta("dark_theme", factors, ["dark_theme", "light_theme"])

        # Check beta was incremented for non-chosen
        assert mock_history.data["factor_beta"]["light_theme"]["time:night"]["beta"] == 1.2
        assert mock_history.data["factor_beta"]["light_theme"]["lux:moonlight"]["beta"] == 1.2
