"""
Theme embeddings for similarity computation.

Provides a 20-dimensional vector representation of themes that captures
visual characteristics for computing distances and finding similar themes.

Embedding dimensions:
    0-2:   Background LAB (L, a, b)
    3-5:   Foreground LAB (L, a, b)
    6:     Contrast (Delta E between bg/fg)
    7:     Average chroma (palette saturation)
    8:     Brightness (normalized 0-1)
    9:     Warmth (normalized 0-1)
    10-13: Hue quadrants (distribution across 4 quadrants)
    14-17: Harmony scores (complementary, analogous, triadic, split)
    18:    Color variety (std dev of palette hues)
    19:    Lightness range (max-min L in palette)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .color import delta_e, hex_to_lab

EMBEDDING_DIM = 20


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def _lab_to_lch(L: float, a: float, b: float) -> tuple[float, float, float]:
    """Convert LAB to LCH (Lightness, Chroma, Hue)."""
    C = math.sqrt(a**2 + b**2)
    H = math.atan2(b, a) * 180 / math.pi
    if H < 0:
        H += 360
    return L, C, H


def _compute_palette_stats(palette: dict[int, str]) -> dict:
    """Compute statistics from palette colors."""
    if not palette:
        return {
            "avg_chroma": 0.0,
            "hue_quadrants": [0.25, 0.25, 0.25, 0.25],
            "color_variety": 0.0,
            "lightness_range": 0.0,
            "harmony_scores": [0.0, 0.0, 0.0, 0.0],
        }

    # Convert palette to LCH
    lch_colors = []
    for hex_color in palette.values():
        lab = hex_to_lab(hex_color)
        lch = _lab_to_lch(*lab)
        lch_colors.append(lch)

    if not lch_colors:
        return {
            "avg_chroma": 0.0,
            "hue_quadrants": [0.25, 0.25, 0.25, 0.25],
            "color_variety": 0.0,
            "lightness_range": 0.0,
            "harmony_scores": [0.0, 0.0, 0.0, 0.0],
        }

    # Average chroma
    chromas = [c for _, c, _ in lch_colors]
    avg_chroma = np.mean(chromas) if chromas else 0.0

    # Hue quadrant distribution (0-90, 90-180, 180-270, 270-360)
    hue_quadrants = [0, 0, 0, 0]
    hues = [h for _, _, h in lch_colors]
    for h in hues:
        quadrant = min(3, int(h / 90))
        hue_quadrants[quadrant] += 1
    total = sum(hue_quadrants)
    if total > 0:
        hue_quadrants = [q / total for q in hue_quadrants]

    # Color variety (std dev of hues, handling circular nature)
    if len(hues) > 1:
        # Convert to radians for circular std
        hues_rad = [h * math.pi / 180 for h in hues]
        sin_sum = sum(math.sin(h) for h in hues_rad)
        cos_sum = sum(math.cos(h) for h in hues_rad)
        R = math.sqrt(sin_sum**2 + cos_sum**2) / len(hues)
        # Circular variance = 1 - R, convert to std-like measure
        color_variety = math.sqrt(-2 * math.log(max(R, 0.01))) * 180 / math.pi
        color_variety = min(color_variety, 180)  # Cap at 180 degrees
    else:
        color_variety = 0.0

    # Lightness range
    lightnesses = [L for L, _, _ in lch_colors]
    lightness_range = max(lightnesses) - min(lightnesses) if lightnesses else 0.0

    # Harmony scores (how well colors fit common harmony patterns)
    harmony_scores = _compute_harmony_scores(hues)

    return {
        "avg_chroma": avg_chroma,
        "hue_quadrants": hue_quadrants,
        "color_variety": color_variety,
        "lightness_range": lightness_range,
        "harmony_scores": harmony_scores,
    }


def _compute_harmony_scores(hues: list[float]) -> list[float]:
    """
    Compute harmony scores for common color relationships.

    Returns scores for:
    - Complementary (180 degrees apart)
    - Analogous (30 degrees apart)
    - Triadic (120 degrees apart)
    - Split-complementary (150 degrees apart)

    Higher scores indicate better fit to the harmony pattern.
    """
    if len(hues) < 2:
        return [0.0, 0.0, 0.0, 0.0]

    # Compute all pairwise hue differences
    diffs = []
    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            diff = abs(hues[i] - hues[j])
            if diff > 180:
                diff = 360 - diff
            diffs.append(diff)

    if not diffs:
        return [0.0, 0.0, 0.0, 0.0]

    # Score each harmony type based on how close differences are to ideal
    def harmony_score(target: float, tolerance: float = 30) -> float:
        scores = [max(0, 1 - abs(d - target) / tolerance) for d in diffs]
        return np.mean(scores) if scores else 0.0

    return [
        harmony_score(180),  # Complementary
        harmony_score(30),  # Analogous
        harmony_score(120),  # Triadic
        harmony_score(150),  # Split-complementary
    ]


@dataclass
class ThemeEmbedding:
    """
    20-dimensional embedding of a theme's visual characteristics.

    Used for computing distances between themes and finding similar themes.
    """

    name: str
    vector: np.ndarray = field(repr=False)

    @classmethod
    def from_theme(cls, theme: dict) -> ThemeEmbedding:
        """Create embedding from a theme dictionary."""
        vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Background LAB (dims 0-2)
        bg_hex = theme.get("background", "#000000")
        bg_lab = hex_to_lab(bg_hex)
        vector[0:3] = bg_lab

        # Foreground LAB (dims 3-5)
        if "foreground" in theme:
            fg_lab = hex_to_lab(theme["foreground"])
            vector[3:6] = fg_lab
        else:
            # Default to high contrast foreground estimate
            bg_L = bg_lab[0]
            fg_L = 95 if bg_L < 50 else 10  # Light fg for dark bg, dark fg for light bg
            vector[3:6] = [fg_L, 0, 0]

        # Contrast - Delta E between bg and fg (dim 6)
        fg_lab = tuple(vector[3:6])
        vector[6] = delta_e(bg_lab, fg_lab)

        # Palette statistics
        palette = theme.get("palette", {})
        stats = _compute_palette_stats(palette)

        # Average chroma (dim 7) - normalized to 0-1 range (max chroma ~130)
        vector[7] = min(stats["avg_chroma"] / 130, 1.0)

        # Brightness (dim 8) - normalize from 0-255 to 0-1
        brightness = theme.get("brightness", 128)
        vector[8] = brightness / 255.0

        # Warmth (dim 9) - normalize from [-1, 1] to [0, 1]
        warmth = theme.get("warmth", 0.0)
        vector[9] = (warmth + 1) / 2

        # Hue quadrants (dims 10-13)
        vector[10:14] = stats["hue_quadrants"]

        # Harmony scores (dims 14-17)
        vector[14:18] = stats["harmony_scores"]

        # Color variety (dim 18) - normalized by max expected (180 degrees)
        vector[18] = stats["color_variety"] / 180

        # Lightness range (dim 19) - normalized by max (100 L units)
        vector[19] = stats["lightness_range"] / 100

        return cls(name=theme["name"], vector=vector)

    def distance(self, other: ThemeEmbedding) -> float:
        """Compute Euclidean distance to another embedding."""
        return float(np.linalg.norm(self.vector - other.vector))

    def weighted_distance(self, other: ThemeEmbedding, weights: np.ndarray | None = None) -> float:
        """
        Compute weighted Euclidean distance.

        Default weights emphasize perceptually important features:
        - Background L (lightness) is most important for dark/light distinction
        - Contrast matters for readability
        - Chroma and warmth matter for "feel"
        """
        if weights is None:
            weights = np.array(
                [
                    3.0,
                    1.0,
                    1.0,  # bg_L, bg_a, bg_b (L is most important)
                    1.0,
                    0.5,
                    0.5,  # fg_L, fg_a, fg_b
                    2.0,  # contrast
                    1.5,  # avg_chroma
                    2.0,  # brightness
                    1.0,  # warmth
                    0.5,
                    0.5,
                    0.5,
                    0.5,  # hue quadrants
                    0.3,
                    0.3,
                    0.3,
                    0.3,  # harmony scores
                    0.5,  # color variety
                    0.5,  # lightness range
                ],
                dtype=np.float32,
            )

        diff = self.vector - other.vector
        return float(np.sqrt(np.sum(weights * diff**2)))

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "name": self.name,
            "vector": self.vector.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ThemeEmbedding:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            vector=np.array(data["vector"], dtype=np.float32),
        )


class EmbeddingIndex:
    """
    Index of theme embeddings for fast similarity search.
    """

    def __init__(self):
        self.embeddings: dict[str, ThemeEmbedding] = {}
        self._matrix: np.ndarray | None = None
        self._names: list[str] = []

    def add(self, embedding: ThemeEmbedding) -> None:
        """Add an embedding to the index."""
        self.embeddings[embedding.name] = embedding
        self._matrix = None  # Invalidate cache

    def add_theme(self, theme: dict) -> ThemeEmbedding:
        """Create and add embedding for a theme."""
        embedding = ThemeEmbedding.from_theme(theme)
        self.add(embedding)
        return embedding

    def build_from_themes(self, themes: list[dict]) -> None:
        """Build index from a list of themes."""
        for theme in themes:
            self.add_theme(theme)
        self._build_matrix()

    def _build_matrix(self) -> None:
        """Build numpy matrix for vectorized operations."""
        self._names = list(self.embeddings.keys())
        if self._names:
            self._matrix = np.vstack([self.embeddings[name].vector for name in self._names])
        else:
            self._matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    def get(self, name: str) -> ThemeEmbedding | None:
        """Get embedding by theme name."""
        return self.embeddings.get(name)

    def nearest(
        self,
        query: np.ndarray | ThemeEmbedding,
        k: int = 5,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Find k nearest themes to a query embedding.

        Args:
            query: Query embedding vector or ThemeEmbedding
            k: Number of results
            exclude: Theme names to exclude from results

        Returns:
            List of (theme_name, distance) tuples, sorted by distance
        """
        if self._matrix is None:
            self._build_matrix()

        if isinstance(query, ThemeEmbedding):
            query_vec = query.vector
        else:
            query_vec = query

        if self._matrix is None or len(self._matrix) == 0:
            return []

        # Compute distances to all themes
        distances = np.linalg.norm(self._matrix - query_vec, axis=1)

        # Sort by distance
        indices = np.argsort(distances)

        # Filter and collect results
        results = []
        exclude = exclude or set()
        for idx in indices:
            name = self._names[idx]
            if name not in exclude:
                results.append((name, float(distances[idx])))
                if len(results) >= k:
                    break

        return results

    def similar_to(
        self,
        theme_name: str,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find themes similar to a given theme."""
        embedding = self.embeddings.get(theme_name)
        if embedding is None:
            return []
        return self.nearest(embedding, k=k + 1, exclude={theme_name})[:k]

    def to_dict(self) -> dict:
        """Serialize index for storage."""
        return {name: emb.to_dict() for name, emb in self.embeddings.items()}

    @classmethod
    def from_dict(cls, data: dict) -> EmbeddingIndex:
        """Deserialize index from storage."""
        index = cls()
        for name, emb_data in data.items():
            index.embeddings[name] = ThemeEmbedding.from_dict(emb_data)
        index._build_matrix()
        return index
