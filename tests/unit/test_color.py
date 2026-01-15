"""Tests for lib/color.py - Color space conversions and Delta E."""

from __future__ import annotations

import pytest

from ghostty_ambient.color import delta_e, hex_to_lab, rgb_to_xyz, theme_distance, xyz_to_lab


class TestRgbToXyz:
    """Tests for rgb_to_xyz() function."""

    def test_black(self):
        """Black should produce (0, 0, 0) in XYZ."""
        x, y, z = rgb_to_xyz(0, 0, 0)
        assert abs(x) < 0.01
        assert abs(y) < 0.01
        assert abs(z) < 0.01

    def test_white(self):
        """White should produce D65 reference values (~95, 100, 109)."""
        x, y, z = rgb_to_xyz(255, 255, 255)
        # D65 white point is approximately (95.047, 100.0, 108.883)
        assert 94 < x < 96
        assert 99 < y < 101
        assert 107 < z < 110

    def test_pure_red(self):
        """Pure red (255, 0, 0)."""
        x, y, z = rgb_to_xyz(255, 0, 0)
        # Red has significant X, some Y, minimal Z
        assert x > 30
        assert 10 < y < 30
        assert z < 5

    def test_pure_green(self):
        """Pure green (0, 255, 0)."""
        x, y, z = rgb_to_xyz(0, 255, 0)
        # Green dominates Y
        assert 20 < x < 50
        assert y > 60
        assert 5 < z < 20

    def test_pure_blue(self):
        """Pure blue (0, 0, 255)."""
        x, y, z = rgb_to_xyz(0, 0, 255)
        # Blue has significant Z
        assert 10 < x < 25
        assert 5 < y < 15
        assert z > 80


class TestXyzToLab:
    """Tests for xyz_to_lab() function."""

    def test_black(self):
        """Black in XYZ (0,0,0) should have L=0."""
        L, a, b = xyz_to_lab(0, 0, 0)
        assert L < 1  # Very low lightness

    def test_white(self):
        """D65 white (95.047, 100, 108.883) should have L=100, a=b=0."""
        L, a, b = xyz_to_lab(95.047, 100.0, 108.883)
        assert 99 < L < 101
        assert abs(a) < 1
        assert abs(b) < 1

    def test_lightness_range(self):
        """L should be in range 0-100 for valid inputs."""
        # Gray (mid-point)
        x, y, z = rgb_to_xyz(128, 128, 128)
        L, a, b = xyz_to_lab(x, y, z)
        assert 0 <= L <= 100


class TestHexToLab:
    """Tests for hex_to_lab() function."""

    def test_white(self):
        """#FFFFFF should have L≈100, a≈0, b≈0."""
        L, a, b = hex_to_lab("#FFFFFF")
        assert 99 < L < 101
        assert abs(a) < 1
        assert abs(b) < 1

    def test_black(self):
        """#000000 should have L≈0."""
        L, a, b = hex_to_lab("#000000")
        assert L < 1

    def test_with_hash(self):
        """Should work with or without hash prefix."""
        lab1 = hex_to_lab("#FF0000")
        lab2 = hex_to_lab("FF0000")
        assert lab1 == lab2

    def test_red_positive_a(self):
        """Red should have positive a* (red-green axis)."""
        L, a, b = hex_to_lab("#FF0000")
        assert a > 0  # Red is positive on a* axis

    def test_green_negative_a(self):
        """Green should have negative a* (red-green axis)."""
        L, a, b = hex_to_lab("#00FF00")
        assert a < 0  # Green is negative on a* axis

    def test_blue_negative_b(self):
        """Blue should have negative b* (yellow-blue axis)."""
        L, a, b = hex_to_lab("#0000FF")
        assert b < 0  # Blue is negative on b* axis

    def test_yellow_positive_b(self):
        """Yellow should have positive b* (yellow-blue axis)."""
        L, a, b = hex_to_lab("#FFFF00")
        assert b > 0  # Yellow is positive on b* axis


class TestDeltaE:
    """Tests for delta_e() function (CIE76)."""

    def test_identical_colors_zero_distance(self):
        """Same color should have Delta E = 0."""
        lab = hex_to_lab("#FF5500")
        assert delta_e(lab, lab) == 0.0

    def test_black_vs_white_large_distance(self):
        """Black and white should have large Delta E (~100)."""
        black = hex_to_lab("#000000")
        white = hex_to_lab("#FFFFFF")
        dist = delta_e(black, white)
        assert dist > 90  # Should be ~100

    def test_similar_colors_small_distance(self):
        """Very similar colors should have small Delta E."""
        color1 = hex_to_lab("#FF0000")
        color2 = hex_to_lab("#FE0000")  # Slightly different red
        dist = delta_e(color1, color2)
        assert dist < 2  # Imperceptible/slight difference

    def test_symmetry(self):
        """Delta E should be symmetric: d(a,b) = d(b,a)."""
        lab1 = hex_to_lab("#FF0000")
        lab2 = hex_to_lab("#00FF00")
        assert delta_e(lab1, lab2) == delta_e(lab2, lab1)

    def test_noticeable_difference(self):
        """Clearly different colors should have Delta E > 10."""
        red = hex_to_lab("#FF0000")
        blue = hex_to_lab("#0000FF")
        dist = delta_e(red, blue)
        assert dist > 50  # Different color families


class TestThemeDistance:
    """Tests for theme_distance() function."""

    def test_identical_themes(self):
        """Same theme should have distance 0."""
        theme = {"background": "#1a1b26", "foreground": "#c0caf5"}
        assert theme_distance(theme, theme) == 0.0

    def test_background_only(self):
        """Without foreground, should use only background."""
        theme1 = {"background": "#FFFFFF"}
        theme2 = {"background": "#000000"}
        dist = theme_distance(theme1, theme2)
        assert dist > 90  # Large difference

    def test_with_foreground_weighted(self):
        """With foreground, should be weighted 0.7 bg + 0.3 fg."""
        theme1 = {"background": "#FFFFFF", "foreground": "#000000"}
        theme2 = {"background": "#000000", "foreground": "#FFFFFF"}

        # Both bg and fg are opposite, so distance should be large
        dist = theme_distance(theme1, theme2)
        assert dist > 90

    def test_similar_dark_themes(self):
        """Similar dark themes should have small distance."""
        tokyo_night = {"background": "#1a1b26", "foreground": "#c0caf5"}
        dracula = {"background": "#282a36", "foreground": "#f8f8f2"}
        dist = theme_distance(tokyo_night, dracula)
        # Both are dark themes, should be reasonably similar
        assert dist < 40

    def test_dark_vs_light_large_distance(self):
        """Dark vs light theme should have large distance."""
        dark = {"background": "#1a1b26", "foreground": "#ffffff"}
        light = {"background": "#ffffff", "foreground": "#1a1b26"}
        dist = theme_distance(dark, light)
        assert dist > 70


class TestColorSpacePipeline:
    """Integration tests for the full color conversion pipeline."""

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
            "#c0caf5",  # Tokyo Night foreground
        ],
    )
    def test_conversion_produces_valid_lab(self, hex_color: str):
        """All conversions should produce valid Lab values."""
        L, a, b = hex_to_lab(hex_color)
        # L should be in 0-100 (with small tolerance for floating point)
        assert -0.001 <= L <= 100.001
        # a and b should be in reasonable range (-128 to 128)
        assert -128 <= a <= 128
        assert -128 <= b <= 128
