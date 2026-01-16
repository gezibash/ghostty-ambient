"""Tests for embeddings.py - Theme embedding computation and indexing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ghostty_ambient.embeddings import (
    EMBEDDING_DIM,
    EmbeddingIndex,
    ThemeEmbedding,
    _compute_harmony_scores,
    _compute_palette_stats,
    _hex_to_rgb,
    _lab_to_lch,
)


class TestHexToRgb:
    """Tests for _hex_to_rgb() helper function."""

    @pytest.mark.parametrize(
        "hex_color,expected",
        [
            ("#000000", (0, 0, 0)),
            ("#ffffff", (255, 255, 255)),
            ("#ff0000", (255, 0, 0)),
            ("#00ff00", (0, 255, 0)),
            ("#0000ff", (0, 0, 255)),
            ("#1a1b26", (26, 27, 38)),
            ("282828", (40, 40, 40)),  # Without hash
        ],
    )
    def test_converts_hex_to_rgb(self, hex_color: str, expected: tuple):
        assert _hex_to_rgb(hex_color) == expected

    def test_handles_uppercase(self):
        assert _hex_to_rgb("#FFFFFF") == (255, 255, 255)
        assert _hex_to_rgb("#FF00FF") == (255, 0, 255)

    def test_handles_mixed_case(self):
        assert _hex_to_rgb("#FfFfFf") == (255, 255, 255)


class TestLabToLch:
    """Tests for _lab_to_lch() helper function."""

    def test_converts_neutral_gray(self):
        L, C, H = _lab_to_lch(50, 0, 0)
        assert L == 50
        assert C == pytest.approx(0, abs=0.001)

    def test_converts_red_hue(self):
        L, C, H = _lab_to_lch(50, 50, 0)
        assert L == 50
        assert C == pytest.approx(50)
        assert H == pytest.approx(0, abs=0.1)

    def test_converts_yellow_hue(self):
        L, C, H = _lab_to_lch(50, 0, 50)
        assert L == 50
        assert C == pytest.approx(50)
        assert H == pytest.approx(90, abs=0.1)

    def test_converts_negative_a(self):
        L, C, H = _lab_to_lch(50, -50, 0)
        assert C == pytest.approx(50)
        assert H == pytest.approx(180, abs=0.1)

    def test_converts_negative_b(self):
        L, C, H = _lab_to_lch(50, 0, -50)
        assert C == pytest.approx(50)
        assert H == pytest.approx(270, abs=0.1)

    def test_hue_always_positive(self):
        # All quadrants should result in positive hue
        for a, b in [(-10, -10), (-10, 10), (10, -10), (10, 10)]:
            _, _, H = _lab_to_lch(50, a, b)
            assert 0 <= H < 360


class TestComputePaletteStats:
    """Tests for _compute_palette_stats() helper function."""

    def test_empty_palette_returns_defaults(self):
        stats = _compute_palette_stats({})
        assert stats["avg_chroma"] == 0.0
        assert stats["hue_quadrants"] == [0.25, 0.25, 0.25, 0.25]
        assert stats["color_variety"] == 0.0
        assert stats["lightness_range"] == 0.0
        assert stats["harmony_scores"] == [0.0, 0.0, 0.0, 0.0]

    def test_single_color_palette(self):
        palette = {0: "#ff0000"}  # Red
        stats = _compute_palette_stats(palette)
        assert stats["avg_chroma"] > 0
        assert stats["color_variety"] == 0.0  # Single color has no variety
        assert stats["lightness_range"] == 0.0

    def test_contrasting_colors_have_high_variety(self):
        # Red, cyan (opposite hues)
        palette = {0: "#ff0000", 1: "#00ffff"}
        stats = _compute_palette_stats(palette)
        assert stats["color_variety"] > 50  # High variety

    def test_analogous_colors_have_low_variety(self):
        # Red, orange (similar hues)
        palette = {0: "#ff0000", 1: "#ff7700"}
        stats = _compute_palette_stats(palette)
        assert stats["color_variety"] < 50

    def test_lightness_range_computed(self):
        # Black and white
        palette = {0: "#000000", 1: "#ffffff"}
        stats = _compute_palette_stats(palette)
        assert stats["lightness_range"] > 90  # Near max L difference

    def test_hue_quadrants_distribution(self):
        # Colors in different quadrants
        palette = {
            0: "#ff0000",  # Red (Q1: 0-90)
            1: "#00ff00",  # Green (Q2: 90-180)
            2: "#0000ff",  # Blue (Q3: 180-270)
            3: "#ff00ff",  # Magenta (Q4: 270-360)
        }
        stats = _compute_palette_stats(palette)
        # Should have distribution across quadrants
        assert sum(stats["hue_quadrants"]) == pytest.approx(1.0)
        assert all(q >= 0 for q in stats["hue_quadrants"])


class TestComputeHarmonyScores:
    """Tests for _compute_harmony_scores() helper function."""

    def test_empty_hues_returns_zeros(self):
        scores = _compute_harmony_scores([])
        assert scores == [0.0, 0.0, 0.0, 0.0]

    def test_single_hue_returns_zeros(self):
        scores = _compute_harmony_scores([45])
        assert scores == [0.0, 0.0, 0.0, 0.0]

    def test_complementary_colors(self):
        # Red (0) and Cyan (180) - exactly complementary
        scores = _compute_harmony_scores([0, 180])
        assert scores[0] > 0.9  # Complementary score should be high

    def test_analogous_colors(self):
        # Colors 30 degrees apart
        scores = _compute_harmony_scores([0, 30])
        assert scores[1] > 0.9  # Analogous score should be high

    def test_triadic_colors(self):
        # Colors 120 degrees apart
        scores = _compute_harmony_scores([0, 120, 240])
        assert scores[2] > 0.5  # Triadic score should be decent

    def test_split_complementary(self):
        # Colors 150 degrees apart
        scores = _compute_harmony_scores([0, 150])
        assert scores[3] > 0.9  # Split-complementary score should be high


class TestThemeEmbedding:
    """Tests for ThemeEmbedding class."""

    def test_creates_from_theme_dict(self, dark_theme):
        embedding = ThemeEmbedding.from_theme(dark_theme)
        assert embedding.name == "Test Dark"
        assert embedding.vector.shape == (EMBEDDING_DIM,)
        assert embedding.vector.dtype == np.float32

    def test_embedding_has_correct_dimensions(self, sample_theme_dicts):
        for theme in sample_theme_dicts:
            embedding = ThemeEmbedding.from_theme(theme)
            assert embedding.vector.shape == (EMBEDDING_DIM,)

    def test_dark_theme_has_low_brightness(self, dark_theme):
        embedding = ThemeEmbedding.from_theme(dark_theme)
        brightness = embedding.vector[8]  # Brightness is dim 8
        assert brightness < 0.2

    def test_light_theme_has_high_brightness(self, light_theme):
        embedding = ThemeEmbedding.from_theme(light_theme)
        brightness = embedding.vector[8]
        assert brightness > 0.9

    def test_warm_theme_has_high_warmth(self):
        warm_theme = {
            "name": "Warm",
            "background": "#3a2820",
            "warmth": 0.8,
            "brightness": 50,
        }
        embedding = ThemeEmbedding.from_theme(warm_theme)
        warmth = embedding.vector[9]  # Warmth is dim 9
        assert warmth > 0.8  # (0.8 + 1) / 2 = 0.9

    def test_cool_theme_has_low_warmth(self):
        cool_theme = {
            "name": "Cool",
            "background": "#1a2030",
            "warmth": -0.8,
            "brightness": 50,
        }
        embedding = ThemeEmbedding.from_theme(cool_theme)
        warmth = embedding.vector[9]
        assert warmth < 0.2  # (-0.8 + 1) / 2 = 0.1

    def test_contrast_computed_from_bg_fg(self, dark_theme):
        embedding = ThemeEmbedding.from_theme(dark_theme)
        contrast = embedding.vector[6]  # Contrast is dim 6
        assert contrast > 0  # Should have positive contrast

    def test_missing_foreground_uses_default(self):
        theme = {
            "name": "NoFg",
            "background": "#000000",
            "brightness": 0,
        }
        embedding = ThemeEmbedding.from_theme(theme)
        # Should use light foreground for dark background
        fg_L = embedding.vector[3]  # fg_L is dim 3
        assert fg_L > 80

    def test_missing_palette_handled(self):
        theme = {
            "name": "NoPalette",
            "background": "#282828",
            "brightness": 40,
        }
        embedding = ThemeEmbedding.from_theme(theme)
        assert embedding.vector.shape == (EMBEDDING_DIM,)

    def test_distance_to_self_is_zero(self, dark_theme):
        embedding = ThemeEmbedding.from_theme(dark_theme)
        assert embedding.distance(embedding) == pytest.approx(0.0)

    def test_distance_is_symmetric(self, dark_theme, light_theme):
        dark_emb = ThemeEmbedding.from_theme(dark_theme)
        light_emb = ThemeEmbedding.from_theme(light_theme)
        assert dark_emb.distance(light_emb) == pytest.approx(light_emb.distance(dark_emb))

    def test_distance_satisfies_triangle_inequality(self, sample_theme_dicts):
        embeddings = [ThemeEmbedding.from_theme(t) for t in sample_theme_dicts[:3]]
        d01 = embeddings[0].distance(embeddings[1])
        d12 = embeddings[1].distance(embeddings[2])
        d02 = embeddings[0].distance(embeddings[2])
        assert d02 <= d01 + d12 + 1e-6

    def test_similar_themes_have_small_distance(self):
        # Two dark themes should be closer than dark vs light
        theme1 = {"name": "Dark1", "background": "#1a1a1a", "brightness": 26}
        theme2 = {"name": "Dark2", "background": "#1e1e1e", "brightness": 30}
        theme3 = {"name": "Light", "background": "#fafafa", "brightness": 250}

        emb1 = ThemeEmbedding.from_theme(theme1)
        emb2 = ThemeEmbedding.from_theme(theme2)
        emb3 = ThemeEmbedding.from_theme(theme3)

        assert emb1.distance(emb2) < emb1.distance(emb3)

    def test_weighted_distance_uses_weights(self, dark_theme, light_theme):
        dark_emb = ThemeEmbedding.from_theme(dark_theme)
        light_emb = ThemeEmbedding.from_theme(light_theme)

        # Default weighted distance
        dist_default = dark_emb.weighted_distance(light_emb)

        # Custom weights emphasizing brightness
        brightness_weights = np.ones(EMBEDDING_DIM, dtype=np.float32)
        brightness_weights[8] = 10.0  # Emphasize brightness
        dist_brightness = dark_emb.weighted_distance(light_emb, brightness_weights)

        # Brightness-weighted should be larger (dark vs light differ in brightness)
        assert dist_brightness > dist_default * 0.5

    def test_serialization_roundtrip(self, dark_theme):
        embedding = ThemeEmbedding.from_theme(dark_theme)
        data = embedding.to_dict()

        assert "name" in data
        assert "vector" in data
        assert isinstance(data["vector"], list)

        restored = ThemeEmbedding.from_dict(data)
        assert restored.name == embedding.name
        assert np.allclose(restored.vector, embedding.vector)


class TestEmbeddingIndex:
    """Tests for EmbeddingIndex class."""

    def test_creates_empty_index(self):
        index = EmbeddingIndex()
        assert len(index.embeddings) == 0

    def test_add_single_embedding(self, dark_theme):
        index = EmbeddingIndex()
        embedding = ThemeEmbedding.from_theme(dark_theme)
        index.add(embedding)
        assert len(index.embeddings) == 1
        assert "Test Dark" in index.embeddings

    def test_add_theme_creates_embedding(self, dark_theme):
        index = EmbeddingIndex()
        result = index.add_theme(dark_theme)
        assert isinstance(result, ThemeEmbedding)
        assert result.name == "Test Dark"
        assert "Test Dark" in index.embeddings

    def test_build_from_themes(self, sample_theme_dicts):
        index = EmbeddingIndex()
        index.build_from_themes(sample_theme_dicts)
        assert len(index.embeddings) == len(sample_theme_dicts)

    def test_get_existing_embedding(self, embedding_index_with_themes):
        emb = embedding_index_with_themes.get("Tokyo Night")
        assert emb is not None
        assert emb.name == "Tokyo Night"

    def test_get_nonexistent_returns_none(self, embedding_index_with_themes):
        emb = embedding_index_with_themes.get("Nonexistent Theme")
        assert emb is None

    def test_nearest_returns_sorted_results(self, embedding_index_with_themes):
        query = embedding_index_with_themes.get("Tokyo Night")
        results = embedding_index_with_themes.nearest(query, k=3)

        assert len(results) <= 3
        # Results should be sorted by distance
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_nearest_with_vector_query(self, embedding_index_with_themes):
        # Query with raw vector
        query_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        results = embedding_index_with_themes.nearest(query_vec, k=3)
        assert len(results) <= 3

    def test_nearest_excludes_specified_themes(self, embedding_index_with_themes):
        query = embedding_index_with_themes.get("Tokyo Night")
        results = embedding_index_with_themes.nearest(
            query, k=5, exclude={"Tokyo Night", "Gruvbox Dark"}
        )

        names = [name for name, _ in results]
        assert "Tokyo Night" not in names
        assert "Gruvbox Dark" not in names

    def test_nearest_on_empty_index(self):
        index = EmbeddingIndex()
        results = index.nearest(np.zeros(EMBEDDING_DIM), k=5)
        assert results == []

    def test_similar_to_finds_similar_themes(self, embedding_index_with_themes):
        # Tokyo Night and Nord are both dark themes - should be similar
        similar = embedding_index_with_themes.similar_to("Tokyo Night", k=3)
        assert len(similar) <= 3
        names = [name for name, _ in similar]
        assert "Tokyo Night" not in names  # Should exclude self

    def test_similar_to_nonexistent_returns_empty(self, embedding_index_with_themes):
        similar = embedding_index_with_themes.similar_to("Nonexistent", k=5)
        assert similar == []

    def test_dark_themes_cluster_together(self, sample_theme_dicts):
        """Dark themes should be closer to each other than to light themes."""
        index = EmbeddingIndex()
        index.build_from_themes(sample_theme_dicts)

        tokyo_night = index.get("Tokyo Night")
        gruvbox = index.get("Gruvbox Dark")
        solarized_light = index.get("Solarized Light")

        dark_to_dark = tokyo_night.distance(gruvbox)
        dark_to_light = tokyo_night.distance(solarized_light)

        assert dark_to_dark < dark_to_light

    def test_serialization_roundtrip(self, sample_theme_dicts):
        index = EmbeddingIndex()
        index.build_from_themes(sample_theme_dicts)

        data = index.to_dict()
        assert len(data) == len(sample_theme_dicts)

        restored = EmbeddingIndex.from_dict(data)
        assert len(restored.embeddings) == len(index.embeddings)

        # Verify vectors match
        for name in index.embeddings:
            assert np.allclose(
                restored.embeddings[name].vector,
                index.embeddings[name].vector,
            )

    def test_matrix_built_lazily(self, sample_theme_dicts):
        index = EmbeddingIndex()
        for theme in sample_theme_dicts:
            index.add_theme(theme)

        # Matrix should be None before nearest() call
        assert index._matrix is None

        # After nearest(), matrix should be built
        index.nearest(np.zeros(EMBEDDING_DIM), k=1)
        assert index._matrix is not None
        assert index._matrix.shape == (len(sample_theme_dicts), EMBEDDING_DIM)

    def test_adding_invalidates_matrix_cache(self, sample_theme_dicts):
        index = EmbeddingIndex()
        index.build_from_themes(sample_theme_dicts[:3])
        index.nearest(np.zeros(EMBEDDING_DIM), k=1)  # Build matrix
        assert index._matrix is not None

        # Adding new theme should invalidate cache
        index.add_theme(sample_theme_dicts[3])
        assert index._matrix is None
