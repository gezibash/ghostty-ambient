"""Tests for embedding_cache.py - Pre-computed theme embedding cache."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from ghostty_ambient.embedding_cache import (
    CACHE_VERSION,
    DEFAULT_CACHE_PATH,
    build_embedding_cache,
    get_cache_info,
    load_embedding_cache,
    save_embedding_cache,
)
from ghostty_ambient.embeddings import EMBEDDING_DIM, EmbeddingIndex


@pytest.fixture
def mock_load_all_themes(sample_theme_dicts):
    """Mock load_all_themes to return test themes."""
    with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
        yield sample_theme_dicts


class TestBuildEmbeddingCache:
    """Tests for build_embedding_cache() function."""

    def test_builds_index_from_provided_themes(self, sample_theme_dicts):
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)

        assert isinstance(index, EmbeddingIndex)
        assert len(index.embeddings) == len(sample_theme_dicts)

    def test_builds_index_from_all_themes_when_none(self, mock_load_all_themes):
        index = build_embedding_cache(themes=None, verbose=False)

        assert len(index.embeddings) == len(mock_load_all_themes)

    def test_all_themes_have_embeddings(self, sample_theme_dicts):
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)

        for theme in sample_theme_dicts:
            assert theme["name"] in index.embeddings

    def test_embeddings_have_correct_dimension(self, sample_theme_dicts):
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)

        for emb in index.embeddings.values():
            assert emb.vector.shape == (EMBEDDING_DIM,)

    def test_verbose_mode_prints_progress(self, sample_theme_dicts, capsys):
        build_embedding_cache(themes=sample_theme_dicts, verbose=True)

        captured = capsys.readouterr()
        assert "Computing embeddings" in captured.out
        assert "Done in" in captured.out


class TestSaveEmbeddingCache:
    """Tests for save_embedding_cache() function."""

    def test_saves_to_default_path(self, sample_theme_dicts, tmp_path):
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)
        path = tmp_path / "cache.json"

        save_embedding_cache(index, path)

        assert path.exists()

    def test_saves_correct_structure(self, sample_theme_dicts, tmp_path):
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)
        path = tmp_path / "cache.json"

        save_embedding_cache(index, path)

        with open(path) as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == CACHE_VERSION
        assert "embedding_dim" in data
        assert data["embedding_dim"] == EMBEDDING_DIM
        assert "theme_count" in data
        assert "embeddings" in data

    def test_creates_parent_directories(self, sample_theme_dicts, tmp_path):
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)
        path = tmp_path / "subdir" / "cache.json"

        save_embedding_cache(index, path)

        assert path.exists()

    def test_saves_all_embeddings(self, sample_theme_dicts, tmp_path):
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)
        path = tmp_path / "cache.json"

        save_embedding_cache(index, path)

        with open(path) as f:
            data = json.load(f)

        assert len(data["embeddings"]) == len(sample_theme_dicts)


class TestLoadEmbeddingCache:
    """Tests for load_embedding_cache() function."""

    def test_loads_existing_cache(self, sample_theme_dicts, tmp_path):
        # Save first
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)
        path = tmp_path / "cache.json"
        save_embedding_cache(index, path)

        # Load
        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
            loaded = load_embedding_cache(path, rebuild_if_stale=False, verbose=False)

        assert len(loaded.embeddings) == len(sample_theme_dicts)

    def test_vectors_match_after_load(self, sample_theme_dicts, tmp_path):
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)
        path = tmp_path / "cache.json"
        save_embedding_cache(index, path)

        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
            loaded = load_embedding_cache(path, rebuild_if_stale=False, verbose=False)

        for name, emb in index.embeddings.items():
            assert np.allclose(emb.vector, loaded.embeddings[name].vector)

    def test_rebuilds_when_cache_missing(self, sample_theme_dicts, tmp_path):
        path = tmp_path / "nonexistent.json"

        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
            index = load_embedding_cache(path, rebuild_if_stale=True, verbose=False)

        assert len(index.embeddings) == len(sample_theme_dicts)
        assert path.exists()  # Should have saved

    def test_returns_empty_when_missing_and_no_rebuild(self, tmp_path):
        path = tmp_path / "nonexistent.json"

        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=[]):
            index = load_embedding_cache(path, rebuild_if_stale=False, verbose=False)

        assert len(index.embeddings) == 0

    def test_rebuilds_on_version_mismatch(self, sample_theme_dicts, tmp_path):
        path = tmp_path / "old_cache.json"

        # Save with old version
        old_data = {
            "version": CACHE_VERSION - 1,
            "embedding_dim": EMBEDDING_DIM,
            "theme_count": 0,
            "embeddings": {},
        }
        with open(path, "w") as f:
            json.dump(old_data, f)

        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
            index = load_embedding_cache(path, rebuild_if_stale=True, verbose=False)

        # Should have rebuilt
        assert len(index.embeddings) == len(sample_theme_dicts)

    def test_rebuilds_when_themes_missing(self, sample_theme_dicts, tmp_path):
        # Save with fewer themes
        fewer_themes = sample_theme_dicts[:3]
        index = build_embedding_cache(themes=fewer_themes, verbose=False)
        path = tmp_path / "cache.json"
        save_embedding_cache(index, path)

        # Load expecting all themes
        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
            loaded = load_embedding_cache(path, rebuild_if_stale=True, verbose=False)

        assert len(loaded.embeddings) == len(sample_theme_dicts)

    def test_handles_corrupted_cache(self, sample_theme_dicts, tmp_path):
        path = tmp_path / "corrupted.json"
        path.write_text("{ invalid json content")

        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
            index = load_embedding_cache(path, rebuild_if_stale=True, verbose=False)

        # Should have rebuilt
        assert len(index.embeddings) == len(sample_theme_dicts)

    def test_verbose_mode_logs_actions(self, sample_theme_dicts, tmp_path, capsys):
        path = tmp_path / "cache.json"

        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
            load_embedding_cache(path, rebuild_if_stale=True, verbose=True)

        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Should have printed something


class TestGetCacheInfo:
    """Tests for get_cache_info() function."""

    def test_returns_not_exists_for_missing(self, tmp_path):
        path = tmp_path / "nonexistent.json"

        info = get_cache_info(path)

        assert info["exists"] is False
        assert "path" in info

    def test_returns_info_for_existing_cache(self, sample_theme_dicts, tmp_path):
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)
        path = tmp_path / "cache.json"
        save_embedding_cache(index, path)

        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
            info = get_cache_info(path)

        assert info["exists"] is True
        assert info["version"] == CACHE_VERSION
        assert info["is_current"] is True
        assert info["theme_count"] == len(sample_theme_dicts)

    def test_detects_missing_themes(self, sample_theme_dicts, tmp_path):
        # Save with fewer themes
        fewer_themes = sample_theme_dicts[:3]
        index = build_embedding_cache(themes=fewer_themes, verbose=False)
        path = tmp_path / "cache.json"
        save_embedding_cache(index, path)

        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
            info = get_cache_info(path)

        assert info["missing_themes"] == len(sample_theme_dicts) - 3

    def test_detects_extra_themes(self, sample_theme_dicts, tmp_path):
        # Save with all themes
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)
        path = tmp_path / "cache.json"
        save_embedding_cache(index, path)

        # But pretend we have fewer themes now
        fewer_themes = sample_theme_dicts[:3]
        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=fewer_themes):
            info = get_cache_info(path)

        assert info["extra_themes"] == len(sample_theme_dicts) - 3

    def test_detects_version_mismatch(self, tmp_path):
        path = tmp_path / "old_cache.json"

        old_data = {
            "version": CACHE_VERSION - 1,
            "embedding_dim": EMBEDDING_DIM,
            "theme_count": 0,
            "embeddings": {},
        }
        with open(path, "w") as f:
            json.dump(old_data, f)

        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=[]):
            info = get_cache_info(path)

        assert info["is_current"] is False
        assert info["version"] == CACHE_VERSION - 1
        assert info["current_version"] == CACHE_VERSION

    def test_includes_file_size(self, sample_theme_dicts, tmp_path):
        index = build_embedding_cache(themes=sample_theme_dicts, verbose=False)
        path = tmp_path / "cache.json"
        save_embedding_cache(index, path)

        with patch("ghostty_ambient.embedding_cache.load_all_themes", return_value=sample_theme_dicts):
            info = get_cache_info(path)

        assert "size_bytes" in info
        assert info["size_bytes"] > 0

    def test_handles_corrupted_cache(self, tmp_path):
        path = tmp_path / "corrupted.json"
        path.write_text("{ invalid json")

        info = get_cache_info(path)

        assert info["exists"] is True
        assert "error" in info


class TestCacheVersion:
    """Tests for cache versioning."""

    def test_cache_version_is_positive(self):
        assert CACHE_VERSION >= 1

    def test_default_path_is_in_config(self):
        assert ".config/ghostty-ambient" in str(DEFAULT_CACHE_PATH)
        assert "theme_embeddings.json" in str(DEFAULT_CACHE_PATH)
