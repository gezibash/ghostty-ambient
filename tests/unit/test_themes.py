"""Tests for lib/themes.py - Theme parsing and analysis."""

from __future__ import annotations

import pytest

from ghostty_ambient.themes import calculate_brightness, calculate_warmth, hex_to_rgb


class TestHexToRgb:
    """Tests for hex_to_rgb() function."""

    def test_black(self):
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_white(self):
        assert hex_to_rgb("#FFFFFF") == (255, 255, 255)

    def test_red(self):
        assert hex_to_rgb("#FF0000") == (255, 0, 0)

    def test_green(self):
        assert hex_to_rgb("#00FF00") == (0, 255, 0)

    def test_blue(self):
        assert hex_to_rgb("#0000FF") == (0, 0, 255)

    def test_without_hash(self):
        assert hex_to_rgb("FF5500") == (255, 85, 0)

    def test_lowercase(self):
        assert hex_to_rgb("#ff5500") == (255, 85, 0)

    def test_mixed_case(self):
        assert hex_to_rgb("#Ff55aA") == (255, 85, 170)


class TestCalculateBrightness:
    """Tests for calculate_brightness() function (ITU-R BT.601)."""

    def test_white_is_255(self):
        """White should have maximum brightness."""
        assert calculate_brightness("#FFFFFF") == 255

    def test_black_is_0(self):
        """Black should have zero brightness."""
        assert calculate_brightness("#000000") == 0

    def test_red_brightness(self):
        """Red (255, 0, 0): (255*299)/1000 = 76.245 ≈ 76."""
        brightness = calculate_brightness("#FF0000")
        assert brightness == 76

    def test_green_brightness(self):
        """Green (0, 255, 0): (255*587)/1000 = 149.685 ≈ 149."""
        brightness = calculate_brightness("#00FF00")
        assert brightness == 149

    def test_blue_brightness(self):
        """Blue (0, 0, 255): (255*114)/1000 = 29.07 ≈ 29."""
        brightness = calculate_brightness("#0000FF")
        assert brightness == 29

    def test_gray_is_midpoint(self):
        """Mid-gray should have brightness ~127-128."""
        brightness = calculate_brightness("#808080")
        assert 126 <= brightness <= 129

    def test_green_dominates_luminance(self):
        """Green contributes most to perceived brightness (0.587 weight)."""
        red = calculate_brightness("#FF0000")
        green = calculate_brightness("#00FF00")
        blue = calculate_brightness("#0000FF")
        assert green > red > blue

    @pytest.mark.parametrize(
        "color,expected_brightness",
        [
            ("#1a1b26", 27),  # Tokyo Night background (dark)
            ("#c0caf5", 203),  # Tokyo Night foreground (light)
            ("#282a36", 41),  # Dracula background (dark)
            ("#f8f8f2", 246),  # Dracula foreground (light)
        ],
    )
    def test_known_themes(self, color: str, expected_brightness: int):
        """Test brightness calculation for known theme colors."""
        # Allow ±2 tolerance for rounding differences
        brightness = calculate_brightness(color)
        assert abs(brightness - expected_brightness) <= 2


class TestCalculateWarmth:
    """Tests for calculate_warmth() function."""

    def test_red_is_warm(self):
        """Pure red should have warmth close to 1."""
        warmth = calculate_warmth("#FF0000")
        # (255-0)/(255+0+1) = 255/256 ≈ 0.996
        assert warmth > 0.99

    def test_blue_is_cool(self):
        """Pure blue should have warmth close to -1."""
        warmth = calculate_warmth("#0000FF")
        # (0-255)/(0+255+1) = -255/256 ≈ -0.996
        assert warmth < -0.99

    def test_gray_is_neutral(self):
        """Gray (r==b) should have warmth = 0."""
        warmth = calculate_warmth("#808080")
        assert warmth == 0.0

    def test_white_is_neutral(self):
        """White should have warmth = 0."""
        warmth = calculate_warmth("#FFFFFF")
        assert warmth == 0.0

    def test_black_is_neutral(self):
        """Black should have warmth = 0 (handled by +1 in denominator)."""
        warmth = calculate_warmth("#000000")
        assert warmth == 0.0

    def test_orange_is_warm(self):
        """Orange should be warm (r > b)."""
        warmth = calculate_warmth("#FF8800")
        assert warmth > 0.5

    def test_cyan_is_cool(self):
        """Cyan should be cool (b > r)."""
        warmth = calculate_warmth("#00FFFF")
        # (0-255)/(0+255+1) = -255/256 ≈ -0.996
        assert warmth < -0.5

    def test_warmth_range(self):
        """Warmth should always be in [-1, 1]."""
        colors = ["#FF0000", "#0000FF", "#00FF00", "#FFFFFF", "#000000", "#FF5500", "#0055FF"]
        for color in colors:
            warmth = calculate_warmth(color)
            assert -1 <= warmth <= 1


class TestParseTheme:
    """Tests for parse_theme() function."""

    def test_valid_theme(self, tmp_path):
        """Valid theme file should be parsed correctly."""
        from ghostty_ambient.themes import parse_theme

        theme_file = tmp_path / "test_theme"
        theme_file.write_text(
            """
background = #1a1b26
foreground = #c0caf5
palette = 0=#15161e
palette = 1=#f7768e
"""
        )

        theme = parse_theme(theme_file)
        assert theme is not None
        assert theme["name"] == "test_theme"
        assert theme["background"] == "#1a1b26"
        assert theme["foreground"] == "#c0caf5"
        assert "brightness" in theme
        assert "warmth" in theme
        assert theme["palette"][0] == "#15161e"
        assert theme["palette"][1] == "#f7768e"

    def test_missing_background_returns_none(self, tmp_path):
        """Theme without background should return None."""
        from ghostty_ambient.themes import parse_theme

        theme_file = tmp_path / "no_bg_theme"
        theme_file.write_text("foreground = #c0caf5\n")

        assert parse_theme(theme_file) is None

    def test_no_foreground_still_valid(self, tmp_path):
        """Theme with only background should still be valid."""
        from ghostty_ambient.themes import parse_theme

        theme_file = tmp_path / "bg_only"
        theme_file.write_text("background = #1a1b26\n")

        theme = parse_theme(theme_file)
        assert theme is not None
        assert "foreground" not in theme

    def test_nonexistent_file_returns_none(self, tmp_path):
        """Non-existent file should return None."""
        from ghostty_ambient.themes import parse_theme

        theme = parse_theme(tmp_path / "nonexistent")
        assert theme is None
