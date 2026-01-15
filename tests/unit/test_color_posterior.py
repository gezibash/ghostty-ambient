"""Tests for lib/color_posterior.py - Bayesian color preference learning."""

from __future__ import annotations

import numpy as np
import pytest

from ghostty_ambient.color import hex_to_lab, lab_to_hex, lab_to_rgb
from ghostty_ambient.color_posterior import ColorPosterior, ColorPreferenceModel


class TestColorPosterior:
    """Tests for the ColorPosterior class."""

    def test_initial_state(self):
        """Fresh posterior should have neutral prior."""
        p = ColorPosterior()
        assert p.count == 0
        assert p.confidence == 0.0
        # Mean should be neutral gray (L=50, a=0, b=0)
        assert np.allclose(p.mean, [50.0, 0.0, 0.0])

    def test_add_single_observation(self):
        """Single observation should set mean to that observation."""
        p = ColorPosterior()
        lab = (25.0, -5.0, 3.0)
        p.add_observation(lab)

        assert p.count == 1
        assert p.confidence == 0.1  # 1/10
        assert np.allclose(p.mean, lab)

    def test_add_multiple_observations(self):
        """Multiple observations should compute sample mean."""
        p = ColorPosterior()
        observations = [
            (20.0, -2.0, 4.0),
            (24.0, -4.0, 6.0),
            (26.0, -6.0, 8.0),
        ]
        for obs in observations:
            p.add_observation(obs)

        expected_mean = np.mean(observations, axis=0)
        assert p.count == 3
        assert p.confidence == 0.3  # 3/10
        assert np.allclose(p.mean, expected_mean)

    def test_confidence_saturates_at_one(self):
        """Confidence should saturate at 1.0 (10 observations)."""
        p = ColorPosterior()
        for i in range(15):
            p.add_observation((25.0 + i, 0.0, 0.0))

        assert p.count == 15
        assert p.confidence == 1.0  # Saturated

    def test_score_exact_match(self):
        """Scoring the mean should give high score."""
        p = ColorPosterior()
        # Add observations clustered around a point
        for _ in range(5):
            p.add_observation((25.0, -3.0, 5.0))

        score = p.score((25.0, -3.0, 5.0))
        assert score > 90  # Should be close to 100

    def test_score_distant_color(self):
        """Distant colors should get low scores."""
        p = ColorPosterior()
        # Build posterior around dark color
        for _ in range(5):
            p.add_observation((25.0, 0.0, 0.0))

        # Score a very different (light) color
        score = p.score((90.0, 0.0, 0.0))
        assert score < 50  # Should be lower

    def test_covariance_regularization(self):
        """Covariance should be regularized to prevent singularity."""
        p = ColorPosterior()
        # Add identical observations (would cause singular covariance)
        for _ in range(3):
            p.add_observation((25.0, -3.0, 5.0))

        # Should still be able to score without error
        score = p.score((30.0, -2.0, 4.0))
        assert 0 <= score <= 100


class TestColorPreferenceModel:
    """Tests for the ColorPreferenceModel class."""

    def test_empty_model(self):
        """Empty model should return neutral predictions."""
        model = ColorPreferenceModel()
        ideal, confidence = model.predict_ideal({"time": "night"})

        assert np.allclose(ideal, [50.0, 0.0, 0.0])
        assert confidence == 0.0

    def test_record_creates_posteriors(self):
        """Recording should create posteriors for each factor."""
        model = ColorPreferenceModel()
        lab = (25.0, -3.0, 5.0)
        factors = {"time": "night", "system": "dark"}

        model.record(lab, factors)

        assert "time:night" in model.posteriors
        assert "system:dark" in model.posteriors
        assert model.posteriors["time:night"].count == 1
        assert model.posteriors["system:dark"].count == 1

    def test_record_ignores_unknown_buckets(self):
        """Unknown buckets should be ignored."""
        model = ColorPreferenceModel()
        lab = (25.0, -3.0, 5.0)
        factors = {"time": "night", "lux": "unknown", "weather": None}

        model.record(lab, factors)

        assert "time:night" in model.posteriors
        assert "lux:unknown" not in model.posteriors
        assert len(model.posteriors) == 1

    def test_predict_single_factor(self):
        """Prediction with single matching factor."""
        model = ColorPreferenceModel()

        # Record dark colors at night
        for L in [20, 25, 30]:
            model.record((L, -2.0, 4.0), {"time": "night"})

        ideal, confidence = model.predict_ideal({"time": "night"})

        # Should predict around L=25 (mean)
        assert 20 <= ideal[0] <= 30
        assert confidence > 0

    def test_predict_multiple_factors(self):
        """Prediction should combine multiple factors."""
        model = ColorPreferenceModel()

        # Night -> dark colors (L ~ 25)
        for _ in range(5):
            model.record((25.0, 0.0, 0.0), {"time": "night"})

        # Dark mode -> dark colors (L ~ 20)
        for _ in range(5):
            model.record((20.0, 0.0, 0.0), {"system": "dark"})

        # Combined should be between 20 and 25
        ideal, confidence = model.predict_ideal(
            {"time": "night", "system": "dark"}
        )
        assert 18 <= ideal[0] <= 27
        assert confidence >= 0.5  # 5 obs per factor = 0.5 confidence each

    def test_score_theme(self):
        """Theme scoring should work with trained model."""
        model = ColorPreferenceModel()

        # Train on dark colors
        for _ in range(5):
            model.record((25.0, -2.0, 4.0), {"time": "night"})

        # Score a dark theme (close to learned)
        dark_score = model.score_theme((26.0, -3.0, 5.0), {"time": "night"})

        # Score a light theme (far from learned)
        light_score = model.score_theme((90.0, 0.0, 0.0), {"time": "night"})

        assert dark_score > light_score

    def test_score_neutral_without_data(self):
        """Without data, should return neutral score (50)."""
        model = ColorPreferenceModel()
        score = model.score_theme((50.0, 0.0, 0.0), {"time": "morning"})
        assert score == 50.0

    def test_get_ideal_color_requires_confidence(self):
        """get_ideal_color should return None with low confidence."""
        model = ColorPreferenceModel()

        # Single observation = 10% confidence (below 30% threshold)
        model.record((25.0, -2.0, 4.0), {"time": "night"})

        result = model.get_ideal_color({"time": "night"})
        assert result is None

    def test_get_ideal_color_with_sufficient_data(self):
        """get_ideal_color should return ideal with sufficient data."""
        model = ColorPreferenceModel()

        # Add enough observations for >30% confidence
        for _ in range(5):
            model.record((25.0, -2.0, 4.0), {"time": "night"})

        result = model.get_ideal_color({"time": "night"})
        assert result is not None
        ideal, confidence = result
        assert 20 <= ideal[0] <= 30  # L around 25
        assert confidence >= 0.3

    def test_serialization_roundtrip(self):
        """Model should serialize and deserialize correctly."""
        model = ColorPreferenceModel()

        # Add some observations
        model.record((25.0, -2.0, 4.0), {"time": "night"})
        model.record((80.0, 5.0, 10.0), {"time": "afternoon"})

        # Serialize and reload
        data = model.to_dict()
        model2 = ColorPreferenceModel(data)

        # Check posteriors are restored
        assert "time:night" in model2.posteriors
        assert "time:afternoon" in model2.posteriors
        assert model2.posteriors["time:night"].count == 1
        assert model2.posteriors["time:afternoon"].count == 1

    def test_get_stats(self):
        """Stats should report factor count and observations."""
        model = ColorPreferenceModel()

        model.record((25.0, -2.0, 4.0), {"time": "night", "system": "dark"})
        model.record((25.0, -2.0, 4.0), {"time": "night"})

        stats = model.get_stats()
        assert stats["factor_count"] == 2  # time:night, system:dark
        assert stats["total_observations"] == 3  # 2 for time:night, 1 for system:dark


class TestLabRgbRoundtrip:
    """Tests for LAB↔RGB roundtrip conversions."""

    @pytest.mark.parametrize(
        "hex_color",
        [
            "#000000",  # Black
            "#FFFFFF",  # White
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#808080",  # Gray
            "#1a1b26",  # Tokyo Night background
            "#282a36",  # Dracula background
            "#f8f8f2",  # Light foreground
        ],
    )
    def test_hex_roundtrip(self, hex_color: str):
        """hex → LAB → hex should be close to original."""
        lab = hex_to_lab(hex_color)
        result_hex = lab_to_hex(*lab)

        # Parse both to RGB for comparison
        original = hex_color.lstrip("#").lower()
        result = result_hex.lstrip("#").lower()

        # Allow small differences due to floating point
        orig_r, orig_g, orig_b = (
            int(original[0:2], 16),
            int(original[2:4], 16),
            int(original[4:6], 16),
        )
        res_r, res_g, res_b = (
            int(result[0:2], 16),
            int(result[2:4], 16),
            int(result[4:6], 16),
        )

        assert abs(orig_r - res_r) <= 1
        assert abs(orig_g - res_g) <= 1
        assert abs(orig_b - res_b) <= 1

    def test_lab_to_rgb_clamping(self):
        """LAB values outside sRGB gamut should be clamped."""
        # This LAB value is outside sRGB gamut
        r, g, b = lab_to_rgb(50, 100, 100)

        # Should be clamped to valid range
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255


class TestColorPreferenceLearning:
    """Integration tests for color preference learning scenarios."""

    def test_learns_dark_theme_preference(self):
        """Model should learn preference for dark themes at night."""
        model = ColorPreferenceModel()

        # User consistently chooses dark themes at night
        dark_themes = [
            "#1a1b26",  # Tokyo Night
            "#282a36",  # Dracula
            "#2e3440",  # Nord
        ]

        for hex_color in dark_themes:
            lab = hex_to_lab(hex_color)
            model.record(lab, {"time": "night", "system": "dark"})

        # Score a dark theme (should be high)
        dark_score = model.score_theme(
            hex_to_lab("#1e1e2e"), {"time": "night", "system": "dark"}
        )

        # Score a light theme (should be low)
        light_score = model.score_theme(
            hex_to_lab("#f8f8f2"), {"time": "night", "system": "dark"}
        )

        assert dark_score > light_score
        assert dark_score > 60  # Should be reasonably confident

    def test_learns_different_preferences_per_context(self):
        """Model should learn different preferences per context."""
        model = ColorPreferenceModel()

        # Dark themes at night
        for _ in range(3):
            model.record(hex_to_lab("#1a1b26"), {"time": "night"})

        # Light themes in afternoon
        for _ in range(3):
            model.record(hex_to_lab("#f8f8f2"), {"time": "afternoon"})

        # Night should predict dark
        night_ideal, _ = model.predict_ideal({"time": "night"})
        assert night_ideal[0] < 40  # Low L = dark

        # Afternoon should predict light
        afternoon_ideal, _ = model.predict_ideal({"time": "afternoon"})
        assert afternoon_ideal[0] > 80  # High L = light
