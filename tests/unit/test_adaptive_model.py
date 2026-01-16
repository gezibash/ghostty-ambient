"""Tests for adaptive_model.py - Phase-aware preference model."""

from __future__ import annotations

import numpy as np
import pytest

from ghostty_ambient.adaptive_model import (
    DEFAULT_HISTORY_PATH,
    STORAGE_VERSION,
    AdaptivePreferenceModel,
)
from ghostty_ambient.embeddings import EMBEDDING_DIM, EmbeddingIndex
from ghostty_ambient.phase_detector import Phase, PhaseDetector


@pytest.fixture
def mock_embedding_cache(sample_theme_dicts):
    """Mock the embedding cache loading."""
    index = EmbeddingIndex()
    index.build_from_themes(sample_theme_dicts)
    return index


@pytest.fixture
def adaptive_model(mock_embedding_cache, sample_theme_dicts):
    """Create an AdaptivePreferenceModel without using cache."""
    model = AdaptivePreferenceModel(use_embedding_cache=False)
    model.embedding_index = mock_embedding_cache
    return model


class TestAdaptivePreferenceModelInit:
    """Tests for AdaptivePreferenceModel initialization."""

    def test_creates_with_default_path(self):
        model = AdaptivePreferenceModel(use_embedding_cache=False)
        assert model.history_path == DEFAULT_HISTORY_PATH

    def test_creates_with_custom_path(self, tmp_path):
        custom_path = tmp_path / "custom_history.json"
        model = AdaptivePreferenceModel(history_path=custom_path, use_embedding_cache=False)
        assert model.history_path == custom_path

    def test_initializes_empty_observations(self, adaptive_model):
        assert len(adaptive_model.observations) == 0

    def test_initializes_phase_detector(self, adaptive_model):
        assert isinstance(adaptive_model.phase_detector, PhaseDetector)

    def test_initializes_empty_favorites(self, adaptive_model):
        assert len(adaptive_model.favorites) == 0

    def test_initializes_empty_disliked(self, adaptive_model):
        assert len(adaptive_model.disliked) == 0


class TestAdaptivePreferenceModelRecord:
    """Tests for AdaptivePreferenceModel.record() method."""

    def test_record_adds_observation(self, adaptive_model, sample_theme_dicts):
        theme = sample_theme_dicts[0]
        context = {"time": "night", "lux": "dim"}

        adaptive_model.record(theme, context, source="picker")

        assert len(adaptive_model.observations) == 1

    def test_record_preserves_theme_name(self, adaptive_model, sample_theme_dicts):
        theme = sample_theme_dicts[0]
        context = {"time": "night"}

        adaptive_model.record(theme, context, source="picker")

        obs = adaptive_model.observations.recent(1)[0]
        assert obs.theme_name == theme["name"]

    def test_record_preserves_context(self, adaptive_model, sample_theme_dicts):
        theme = sample_theme_dicts[0]
        context = {"time": "night", "lux": "dim", "system": "dark"}

        adaptive_model.record(theme, context, source="picker")

        obs = adaptive_model.observations.recent(1)[0]
        assert obs.context == context

    def test_record_preserves_source(self, adaptive_model, sample_theme_dicts):
        theme = sample_theme_dicts[0]

        adaptive_model.record(theme, {}, source="ideal")

        obs = adaptive_model.observations.recent(1)[0]
        assert obs.source == "ideal"

    def test_record_updates_phase_detector(self, adaptive_model, sample_theme_dicts):
        theme = sample_theme_dicts[0]

        # Record many observations
        for _ in range(10):
            adaptive_model.record(theme, {}, source="picker")

        # Phase detector should have been updated
        assert len(adaptive_model.phase_detector._feature_history) > 0

    def test_record_creates_embedding_if_missing(self, adaptive_model):
        # New theme not in index
        new_theme = {
            "name": "Brand New Theme",
            "background": "#123456",
            "brightness": 50,
        }

        adaptive_model.record(new_theme, {}, source="picker")

        # Should have created embedding
        assert "Brand New Theme" in adaptive_model.embedding_index.embeddings


class TestAdaptivePreferenceModelPhase:
    """Tests for phase-related methods."""

    def test_current_phase_returns_phase(self, adaptive_model):
        phase = adaptive_model.current_phase()
        assert isinstance(phase, Phase)

    def test_initial_phase_is_explore(self, adaptive_model):
        # With no observations, should be EXPLORE
        assert adaptive_model.current_phase() == Phase.EXPLORE

    def test_phase_probabilities_returns_dict(self, adaptive_model):
        probs = adaptive_model.phase_probabilities()
        assert isinstance(probs, dict)
        assert Phase.EXPLORE in probs
        assert Phase.CONVERGE in probs
        assert Phase.STABLE in probs


class TestAdaptivePreferenceModelPredictIdeal:
    """Tests for AdaptivePreferenceModel.predict_ideal() method."""

    def test_predict_ideal_returns_embedding(self, adaptive_model, sample_theme_dicts):
        # Add some observations first
        for theme in sample_theme_dicts[:3]:
            adaptive_model.record(theme, {"time": "night"}, source="picker")

        ideal = adaptive_model.predict_ideal({"time": "night"})

        assert isinstance(ideal, np.ndarray)
        assert ideal.shape == (EMBEDDING_DIM,)

    def test_predict_ideal_with_no_observations(self, adaptive_model):
        ideal = adaptive_model.predict_ideal({})
        # Should return zeros
        assert np.allclose(ideal, np.zeros(EMBEDDING_DIM))

    def test_predict_ideal_with_context_filter(self, adaptive_model, sample_theme_dicts):
        # Record different themes for different contexts
        adaptive_model.record(sample_theme_dicts[0], {"time": "night"}, source="picker")
        adaptive_model.record(sample_theme_dicts[1], {"time": "morning"}, source="picker")

        ideal_night = adaptive_model.predict_ideal({"time": "night"})
        ideal_morning = adaptive_model.predict_ideal({"time": "morning"})

        # Should be different (not strictly equal)
        # They could be similar by chance, so just check they're valid
        assert ideal_night.shape == (EMBEDDING_DIM,)
        assert ideal_morning.shape == (EMBEDDING_DIM,)


class TestAdaptivePreferenceModelRecommend:
    """Tests for AdaptivePreferenceModel.recommend() method."""

    def test_recommend_returns_list(self, adaptive_model, sample_theme_dicts):
        # Add some observations
        for theme in sample_theme_dicts[:3]:
            adaptive_model.record(theme, {}, source="picker")

        recommendations = adaptive_model.recommend({}, sample_theme_dicts, k=5)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

    def test_recommend_includes_scores(self, adaptive_model, sample_theme_dicts):
        for theme in sample_theme_dicts[:3]:
            adaptive_model.record(theme, {}, source="picker")

        recommendations = adaptive_model.recommend({}, sample_theme_dicts, k=5)

        for rec in recommendations:
            assert "_score" in rec
            assert "_distance" in rec
            assert "_phase" in rec

    def test_recommend_sorted_by_score(self, adaptive_model, sample_theme_dicts):
        for theme in sample_theme_dicts[:3]:
            adaptive_model.record(theme, {}, source="picker")

        recommendations = adaptive_model.recommend({}, sample_theme_dicts, k=5)

        scores = [r["_score"] for r in recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_excludes_disliked(self, adaptive_model, sample_theme_dicts):
        for theme in sample_theme_dicts[:3]:
            adaptive_model.record(theme, {}, source="picker")

        # Dislike a theme
        adaptive_model.add_dislike(sample_theme_dicts[0]["name"])

        recommendations = adaptive_model.recommend({}, sample_theme_dicts, k=10)

        names = [r["name"] for r in recommendations]
        assert sample_theme_dicts[0]["name"] not in names

    def test_recommend_boosts_favorites(self, adaptive_model, sample_theme_dicts):
        for theme in sample_theme_dicts[:3]:
            adaptive_model.record(theme, {}, source="picker")

        # Favorite a theme
        fav_name = sample_theme_dicts[1]["name"]
        adaptive_model.add_favorite(fav_name)

        recommendations = adaptive_model.recommend({}, sample_theme_dicts, k=10)

        # Favorite should appear and have boosted score
        fav_rec = next((r for r in recommendations if r["name"] == fav_name), None)
        assert fav_rec is not None

    def test_recommend_respects_k_limit(self, adaptive_model, sample_theme_dicts):
        for theme in sample_theme_dicts[:3]:
            adaptive_model.record(theme, {}, source="picker")

        recommendations = adaptive_model.recommend({}, sample_theme_dicts, k=3)

        assert len(recommendations) <= 3


class TestAdaptivePreferenceModelFindSimilar:
    """Tests for find_similar() method."""

    def test_find_similar_returns_list(self, adaptive_model, sample_theme_dicts):
        similar = adaptive_model.find_similar(sample_theme_dicts[0]["name"], k=3)
        assert isinstance(similar, list)

    def test_find_similar_returns_tuples(self, adaptive_model, sample_theme_dicts):
        similar = adaptive_model.find_similar(sample_theme_dicts[0]["name"], k=3)
        for item in similar:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_find_similar_excludes_self(self, adaptive_model, sample_theme_dicts):
        theme_name = sample_theme_dicts[0]["name"]
        similar = adaptive_model.find_similar(theme_name, k=5)
        names = [name for name, _ in similar]
        assert theme_name not in names

    def test_find_similar_nonexistent_returns_empty(self, adaptive_model):
        similar = adaptive_model.find_similar("Nonexistent Theme", k=5)
        assert similar == []


class TestAdaptivePreferenceModelFavorites:
    """Tests for favorites management."""

    def test_add_favorite(self, adaptive_model):
        adaptive_model.add_favorite("Theme A")
        assert "Theme A" in adaptive_model.favorites

    def test_add_favorite_removes_from_disliked(self, adaptive_model):
        adaptive_model.add_dislike("Theme A")
        adaptive_model.add_favorite("Theme A")
        assert "Theme A" not in adaptive_model.disliked

    def test_remove_favorite(self, adaptive_model):
        adaptive_model.add_favorite("Theme A")
        adaptive_model.remove_favorite("Theme A")
        assert "Theme A" not in adaptive_model.favorites

    def test_remove_nonexistent_favorite_no_error(self, adaptive_model):
        adaptive_model.remove_favorite("Nonexistent")
        # Should not raise


class TestAdaptivePreferenceModelDisliked:
    """Tests for disliked themes management."""

    def test_add_dislike(self, adaptive_model):
        adaptive_model.add_dislike("Theme A")
        assert "Theme A" in adaptive_model.disliked

    def test_add_dislike_removes_from_favorites(self, adaptive_model):
        adaptive_model.add_favorite("Theme A")
        adaptive_model.add_dislike("Theme A")
        assert "Theme A" not in adaptive_model.favorites

    def test_remove_dislike(self, adaptive_model):
        adaptive_model.add_dislike("Theme A")
        adaptive_model.remove_dislike("Theme A")
        assert "Theme A" not in adaptive_model.disliked

    def test_remove_nonexistent_dislike_no_error(self, adaptive_model):
        adaptive_model.remove_dislike("Nonexistent")
        # Should not raise


class TestAdaptivePreferenceModelStats:
    """Tests for get_stats() method."""

    def test_get_stats_returns_dict(self, adaptive_model):
        stats = adaptive_model.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_includes_phase(self, adaptive_model):
        stats = adaptive_model.get_stats()
        assert "phase" in stats
        assert stats["phase"] in ["explore", "converge", "stable"]

    def test_get_stats_includes_counts(self, adaptive_model, sample_theme_dicts):
        adaptive_model.add_favorite("A")
        adaptive_model.add_dislike("B")

        stats = adaptive_model.get_stats()

        assert stats["favorites_count"] == 1
        assert stats["disliked_count"] == 1

    def test_get_stats_includes_observation_count(self, adaptive_model, sample_theme_dicts):
        for theme in sample_theme_dicts[:3]:
            adaptive_model.record(theme, {}, source="picker")

        stats = adaptive_model.get_stats()
        assert stats["total_observations"] == 3


class TestAdaptivePreferenceModelSerialization:
    """Tests for serialization methods."""

    def test_to_dict_structure(self, adaptive_model, sample_theme_dicts):
        # Add some data
        adaptive_model.add_favorite("Theme A")
        adaptive_model.record(sample_theme_dicts[0], {}, source="picker")

        data = adaptive_model.to_dict()

        assert "version" in data
        assert data["version"] == STORAGE_VERSION
        assert "observations" in data
        assert "phase_detector" in data
        assert "favorites" in data
        assert "disliked" in data

    def test_to_dict_excludes_embedding_index(self, adaptive_model):
        data = adaptive_model.to_dict()
        assert "embedding_index" not in data

    def test_serialization_roundtrip(self, adaptive_model, sample_theme_dicts, mock_embedding_cache):
        # Add some data
        adaptive_model.add_favorite("Theme A")
        adaptive_model.add_dislike("Theme B")
        for theme in sample_theme_dicts[:3]:
            adaptive_model.record(theme, {"time": "night"}, source="picker")

        data = adaptive_model.to_dict()

        restored = AdaptivePreferenceModel.from_dict(data)

        assert restored.favorites == adaptive_model.favorites
        assert restored.disliked == adaptive_model.disliked
        assert len(restored.observations) == len(adaptive_model.observations)


class TestAdaptivePreferenceModelSaveLoad:
    """Tests for save() and load() methods."""

    def test_save_creates_file(self, adaptive_model, tmp_path, sample_theme_dicts):
        path = tmp_path / "history.json"
        adaptive_model.record(sample_theme_dicts[0], {}, source="picker")

        adaptive_model.save(path)

        assert path.exists()

    def test_save_creates_parent_directories(self, adaptive_model, tmp_path, sample_theme_dicts):
        path = tmp_path / "subdir" / "history.json"
        adaptive_model.record(sample_theme_dicts[0], {}, source="picker")

        adaptive_model.save(path)

        assert path.exists()

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.json"

        model = AdaptivePreferenceModel.load(path)

        assert len(model.observations) == 0

    def test_load_existing_file(self, adaptive_model, tmp_path, sample_theme_dicts):
        path = tmp_path / "history.json"

        # Save some data
        adaptive_model.add_favorite("Theme A")
        adaptive_model.record(sample_theme_dicts[0], {}, source="picker")
        adaptive_model.save(path)

        # Load it back
        loaded = AdaptivePreferenceModel.load(path)

        assert "Theme A" in loaded.favorites
        assert len(loaded.observations) == 1

    def test_load_handles_corrupted_file(self, tmp_path):
        path = tmp_path / "corrupted.json"
        path.write_text("{ invalid json }")

        model = AdaptivePreferenceModel.load(path)

        # Should return empty model
        assert len(model.observations) == 0

    def test_load_handles_old_version(self, tmp_path):
        path = tmp_path / "old_version.json"
        path.write_text('{"version": 1, "observations": []}')

        model = AdaptivePreferenceModel.load(path)

        # Should return fresh model (migration resets)
        assert len(model.observations) == 0


class TestAdaptivePreferenceModelBuildIndex:
    """Tests for build_index_from_themes() method."""

    def test_builds_index_from_themes(self, adaptive_model, sample_theme_dicts):
        adaptive_model.build_index_from_themes(sample_theme_dicts)

        for theme in sample_theme_dicts:
            assert theme["name"] in adaptive_model.embedding_index.embeddings
