"""
Pre-computed theme embedding cache for fast similarity lookups.

Computes 20D embeddings for all 500+ Ghostty themes once and caches them.
The cache is versioned and rebuilds automatically if the embedding
algorithm changes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .embeddings import EMBEDDING_DIM, EmbeddingIndex, ThemeEmbedding
from .themes import load_all_themes

if TYPE_CHECKING:
    pass

# Cache version - bump this when embedding algorithm changes
CACHE_VERSION = 1

# Default cache path
DEFAULT_CACHE_PATH = Path.home() / ".config/ghostty-ambient/theme_embeddings.json"


def build_embedding_cache(
    themes: list[dict] | None = None,
    verbose: bool = False,
) -> EmbeddingIndex:
    """
    Build embedding index for all themes.

    Args:
        themes: List of theme dicts, or None to load all
        verbose: Print progress

    Returns:
        EmbeddingIndex with all themes indexed
    """
    if themes is None:
        themes = load_all_themes()

    index = EmbeddingIndex()

    if verbose:
        print(f"Computing embeddings for {len(themes)} themes...")
        start = time.time()

    for i, theme in enumerate(themes):
        index.add_theme(theme)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(themes)} themes processed")

    if verbose:
        elapsed = time.time() - start
        print(f"Done in {elapsed:.1f}s ({len(themes)/elapsed:.0f} themes/s)")

    return index


def save_embedding_cache(
    index: EmbeddingIndex,
    path: Path | None = None,
) -> None:
    """
    Save embedding cache to disk.

    Args:
        index: EmbeddingIndex to save
        path: Path to save to (default: ~/.config/ghostty-ambient/theme_embeddings.json)
    """
    path = path or DEFAULT_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": CACHE_VERSION,
        "embedding_dim": EMBEDDING_DIM,
        "theme_count": len(index.embeddings),
        "embeddings": index.to_dict(),
    }

    with open(path, "w") as f:
        json.dump(data, f)


def load_embedding_cache(
    path: Path | None = None,
    rebuild_if_stale: bool = True,
    verbose: bool = False,
) -> EmbeddingIndex:
    """
    Load embedding cache from disk.

    If cache doesn't exist or is outdated, rebuilds it.

    Args:
        path: Path to load from
        rebuild_if_stale: Rebuild if version mismatch or missing themes
        verbose: Print progress

    Returns:
        EmbeddingIndex with all themes
    """
    path = path or DEFAULT_CACHE_PATH

    # Load all themes to check for completeness
    all_themes = load_all_themes()
    theme_names = {t["name"] for t in all_themes}

    # Try to load existing cache
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)

            # Check version
            if data.get("version") != CACHE_VERSION:
                if verbose:
                    print(f"Cache version mismatch (got {data.get('version')}, need {CACHE_VERSION})")
                if rebuild_if_stale:
                    return _rebuild_and_save(all_themes, path, verbose)
                # Return partial cache anyway
                return EmbeddingIndex.from_dict(data.get("embeddings", {}))

            # Check completeness
            cached_names = set(data.get("embeddings", {}).keys())
            missing = theme_names - cached_names

            if missing and rebuild_if_stale:
                if verbose:
                    print(f"Cache missing {len(missing)} themes, rebuilding...")
                return _rebuild_and_save(all_themes, path, verbose)

            # Load from cache
            index = EmbeddingIndex.from_dict(data["embeddings"])
            if verbose:
                print(f"Loaded {len(index.embeddings)} theme embeddings from cache")
            return index

        except (json.JSONDecodeError, KeyError) as e:
            if verbose:
                print(f"Cache corrupted: {e}")
            if rebuild_if_stale:
                return _rebuild_and_save(all_themes, path, verbose)

    # No cache exists
    if rebuild_if_stale:
        return _rebuild_and_save(all_themes, path, verbose)

    # Return empty index
    return EmbeddingIndex()


def _rebuild_and_save(
    themes: list[dict],
    path: Path,
    verbose: bool,
) -> EmbeddingIndex:
    """Rebuild cache and save to disk."""
    index = build_embedding_cache(themes, verbose=verbose)
    save_embedding_cache(index, path)
    if verbose:
        print(f"Saved cache to {path}")
    return index


def get_cache_info(path: Path | None = None) -> dict:
    """
    Get information about the embedding cache.

    Returns:
        Dict with cache metadata
    """
    path = path or DEFAULT_CACHE_PATH

    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
        }

    try:
        with open(path) as f:
            data = json.load(f)

        all_themes = load_all_themes()
        cached_names = set(data.get("embeddings", {}).keys())
        theme_names = {t["name"] for t in all_themes}

        return {
            "exists": True,
            "path": str(path),
            "version": data.get("version"),
            "current_version": CACHE_VERSION,
            "is_current": data.get("version") == CACHE_VERSION,
            "theme_count": len(cached_names),
            "total_themes": len(theme_names),
            "missing_themes": len(theme_names - cached_names),
            "extra_themes": len(cached_names - theme_names),
            "size_bytes": path.stat().st_size,
        }
    except Exception as e:
        return {
            "exists": True,
            "path": str(path),
            "error": str(e),
        }
