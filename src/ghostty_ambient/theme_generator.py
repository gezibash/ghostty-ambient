"""
Generate complete Ghostty themes from learned preferences.

Uses Bayesian posteriors to:
1. Get ideal background color
2. Calculate foreground with learned contrast
3. Generate ANSI palette with learned saturation (chroma)
4. Derive cursor/selection colors
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ghostty_ambient.color import lab_to_hex, hex_to_lab

if TYPE_CHECKING:
    from ghostty_ambient.color_posterior import ThemePreferenceModel

# ANSI color hue targets in LAB a/b space
# These define the "direction" of each color - we scale by learned chroma
ANSI_HUES = {
    1: (50, 30),    # red
    2: (-50, 50),   # green
    3: (0, 60),     # yellow
    4: (0, -60),    # blue
    5: (50, -30),   # magenta
    6: (-30, -30),  # cyan
}


class ThemeGenerator:
    """Generate complete Ghostty themes from learned preferences."""

    def __init__(self, model: ThemePreferenceModel):
        """
        Initialize with a trained theme preference model.

        Args:
            model: ThemePreferenceModel with learned color, contrast, and chroma posteriors
        """
        self.model = model

    def generate(
        self,
        factors: dict[str, str],
        name: str = "Generated",
    ) -> str:
        """
        Generate a complete Ghostty theme file.

        Args:
            factors: Context factors (time, lux, system, etc.)
            name: Name for the generated theme

        Returns:
            Complete theme file content as string
        """
        # 1. Background from learned color posterior
        bg_lab, bg_confidence = self.model.predict_ideal(factors)
        bg_hex = lab_to_hex(float(bg_lab[0]), float(bg_lab[1]), float(bg_lab[2]))

        # 2. Get learned contrast and chroma
        target_contrast = self.model.predict_contrast(factors)
        target_chroma = self.model.predict_chroma(factors)

        # 3. Foreground with learned contrast
        fg_lab = self._generate_foreground(bg_lab, target_contrast)
        fg_hex = lab_to_hex(*fg_lab)

        # 4. Palette with learned chroma
        palette = self._generate_palette(bg_lab, target_chroma)

        # 5. Cursor and selection colors
        selection_bg = self._blend(bg_hex, fg_hex, 0.25)

        # Build theme file
        lines = [f"# {name}"]
        lines.append(f"# Generated with confidence={bg_confidence:.0%}")
        lines.append(f"# Learned: contrast={target_contrast:.1f}, chroma={target_chroma:.1f}")
        lines.append("")
        for i, color in enumerate(palette):
            lines.append(f"palette = {i}={color}")
        lines.append(f"background = {bg_hex}")
        lines.append(f"foreground = {fg_hex}")
        lines.append(f"cursor-color = {fg_hex}")
        lines.append(f"cursor-text = {bg_hex}")
        lines.append(f"selection-background = {selection_bg}")
        lines.append(f"selection-foreground = {fg_hex}")

        return "\n".join(lines)

    def _generate_foreground(
        self,
        bg_lab: np.ndarray,
        target_contrast: float,
    ) -> tuple[float, float, float]:
        """
        Generate foreground color with learned contrast.

        Args:
            bg_lab: Background LAB color
            target_contrast: Target Delta E between bg and fg

        Returns:
            Foreground LAB color tuple
        """
        L, a, b = bg_lab

        # For dark backgrounds, go light; for light, go dark
        if L < 50:
            fg_L = min(95, L + target_contrast)
        else:
            fg_L = max(5, L - target_contrast)

        # Keep a/b mostly neutral for text readability, but hint toward bg tint
        return (float(fg_L), float(a * 0.1), float(b * 0.1))

    def _generate_palette(
        self,
        bg_lab: np.ndarray,
        target_chroma: float,
    ) -> list[str]:
        """
        Generate 16-color ANSI palette with learned chroma.

        Args:
            bg_lab: Background LAB color
            target_chroma: Target chroma (saturation) for colors

        Returns:
            List of 16 hex color strings
        """
        is_dark = bg_lab[0] < 50
        palette = []

        for i in range(16):
            base_idx = i % 8
            is_bright = i >= 8

            if base_idx == 0:  # black
                # Derive from background
                L = float(bg_lab[0] * 0.9) if is_dark else 15.0
                a = float(bg_lab[1] * 0.5)
                b = float(bg_lab[2] * 0.5)
            elif base_idx == 7:  # white
                # Derive from background
                L = 85.0 if is_dark else float(bg_lab[0] * 0.6)
                a = float(bg_lab[1] * 0.3)
                b = float(bg_lab[2] * 0.3)
            else:
                # Colored - use ANSI hue with LEARNED chroma
                base_a, base_b = ANSI_HUES[base_idx]
                scale = target_chroma / 60  # Normalize to baseline
                a = float(base_a * scale)
                b = float(base_b * scale)

                # Lightness based on theme darkness and bright variant
                if is_dark:
                    L = 55.0 + (15.0 if is_bright else 0.0)
                else:
                    L = 50.0 - (10.0 if is_bright else 0.0)

            palette.append(lab_to_hex(L, a, b))

        return palette

    def _blend(self, hex1: str, hex2: str, t: float) -> str:
        """
        Blend two colors in LAB space.

        Args:
            hex1: First hex color
            hex2: Second hex color
            t: Blend factor (0 = hex1, 1 = hex2)

        Returns:
            Blended hex color
        """
        lab1 = np.array(hex_to_lab(hex1))
        lab2 = np.array(hex_to_lab(hex2))
        blended = lab1 * (1 - t) + lab2 * t
        return lab_to_hex(float(blended[0]), float(blended[1]), float(blended[2]))

    def get_preview(self, factors: dict[str, str]) -> dict:
        """
        Get a preview of what would be generated.

        Returns dict with predicted values without generating full theme.
        """
        bg_lab, bg_confidence = self.model.predict_ideal(factors)
        target_contrast = self.model.predict_contrast(factors)
        target_chroma = self.model.predict_chroma(factors)

        return {
            "background_lab": (float(bg_lab[0]), float(bg_lab[1]), float(bg_lab[2])),
            "background_hex": lab_to_hex(float(bg_lab[0]), float(bg_lab[1]), float(bg_lab[2])),
            "confidence": bg_confidence,
            "contrast": target_contrast,
            "chroma": target_chroma,
            "is_dark": bg_lab[0] < 50,
        }
