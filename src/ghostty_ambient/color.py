"""Color theory utilities for theme similarity.

Uses CIELAB colorspace and Delta E for perceptual color difference.
This matches how humans perceive color differences, unlike simple RGB distance.
"""

import math


def rgb_to_xyz(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert sRGB to XYZ colorspace."""
    # Normalize to 0-1
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Gamma correction (sRGB)
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    # Linear transformation (D65 illuminant)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return x * 100, y * 100, z * 100


def xyz_to_lab(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert XYZ to CIELAB colorspace."""
    # D65 reference white
    xn, yn, zn = 95.047, 100.0, 108.883

    x, y, z = x / xn, y / yn, z / zn

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    L = (116 * f(y)) - 16
    a = 500 * (f(x) - f(y))
    b = 200 * (f(y) - f(z))

    return L, a, b


def hex_to_lab(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to Lab colorspace."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    x, y, z = rgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z)


def lab_to_xyz(L: float, a: float, b: float) -> tuple[float, float, float]:
    """Convert CIELAB to XYZ colorspace."""
    # D65 reference white
    xn, yn, zn = 95.047, 100.0, 108.883

    # Inverse of f(t) function
    def f_inv(t):
        if t > 0.206893:  # (6/29)^3 threshold
            return t**3
        return (t - 16 / 116) / 7.787

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    x = xn * f_inv(fx)
    y = yn * f_inv(fy)
    z = zn * f_inv(fz)

    return x, y, z


def xyz_to_rgb(x: float, y: float, z: float) -> tuple[int, int, int]:
    """Convert XYZ to sRGB colorspace."""
    # Normalize from 0-100 to 0-1
    x, y, z = x / 100, y / 100, z / 100

    # Inverse linear transformation (D65 illuminant)
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    # Inverse gamma correction (sRGB)
    def gamma(c):
        if c > 0.0031308:
            return 1.055 * (c ** (1 / 2.4)) - 0.055
        return 12.92 * c

    r, g, b = gamma(r), gamma(g), gamma(b)

    # Clamp to 0-1 and convert to 0-255
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))

    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def lab_to_rgb(L: float, a: float, b: float) -> tuple[int, int, int]:
    """Convert CIELAB to sRGB colorspace."""
    x, y, z = lab_to_xyz(L, a, b)
    return xyz_to_rgb(x, y, z)


def lab_to_hex(L: float, a: float, b: float) -> str:
    """Convert CIELAB to hex color string."""
    r, g, b_val = lab_to_rgb(L, a, b)
    return f"#{r:02x}{g:02x}{b_val:02x}"


def delta_e(lab1: tuple, lab2: tuple) -> float:
    """
    Calculate Delta E (CIE76) between two Lab colors.

    Delta E interpretation:
        < 1:    Imperceptible difference
        1-2:    Slight, noticeable by trained eye
        2-10:   Noticeable difference
        10-50:  Similar family of colors
        > 50:   Different colors
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return math.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)


def theme_distance(theme1: dict, theme2: dict) -> float:
    """
    Calculate perceptual distance between two themes.

    Uses background color as primary, with optional weighting
    for foreground color.

    Args:
        theme1: Theme dict with 'background' and optional 'foreground'
        theme2: Theme dict with 'background' and optional 'foreground'

    Returns:
        Delta E distance (lower = more similar)
    """
    bg1 = hex_to_lab(theme1["background"])
    bg2 = hex_to_lab(theme2["background"])

    # Primary: background distance
    bg_dist = delta_e(bg1, bg2)

    # Optional: include foreground if available
    if "foreground" in theme1 and "foreground" in theme2:
        fg1 = hex_to_lab(theme1["foreground"])
        fg2 = hex_to_lab(theme2["foreground"])
        fg_dist = delta_e(fg1, fg2)
        # Weighted average: background matters more
        return 0.7 * bg_dist + 0.3 * fg_dist

    return bg_dist
