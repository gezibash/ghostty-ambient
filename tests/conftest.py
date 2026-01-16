"""Shared pytest fixtures for ghostty-ambient tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest


# =============================================================================
# Adaptive learning fixtures
# =============================================================================

@pytest.fixture
def sample_theme_dicts():
    """
    Sample theme dictionaries with all fields needed for embedding computation.
    Includes background, foreground, palette, brightness, and warmth.
    """
    return [
        {
            "name": "Tokyo Night",
            "background": "#1a1b26",
            "foreground": "#c0caf5",
            "brightness": 26,
            "warmth": -0.2,
            "palette": {
                0: "#15161e", 1: "#f7768e", 2: "#9ece6a", 3: "#e0af68",
                4: "#7aa2f7", 5: "#bb9af7", 6: "#7dcfff", 7: "#a9b1d6",
            },
        },
        {
            "name": "Gruvbox Dark",
            "background": "#282828",
            "foreground": "#ebdbb2",
            "brightness": 40,
            "warmth": 0.4,
            "palette": {
                0: "#282828", 1: "#cc241d", 2: "#98971a", 3: "#d79921",
                4: "#458588", 5: "#b16286", 6: "#689d6a", 7: "#a89984",
            },
        },
        {
            "name": "Solarized Light",
            "background": "#fdf6e3",
            "foreground": "#657b83",
            "brightness": 246,
            "warmth": 0.1,
            "palette": {
                0: "#073642", 1: "#dc322f", 2: "#859900", 3: "#b58900",
                4: "#268bd2", 5: "#d33682", 6: "#2aa198", 7: "#eee8d5",
            },
        },
        {
            "name": "Nord",
            "background": "#2e3440",
            "foreground": "#d8dee9",
            "brightness": 52,
            "warmth": -0.1,
            "palette": {
                0: "#3b4252", 1: "#bf616a", 2: "#a3be8c", 3: "#ebcb8b",
                4: "#81a1c1", 5: "#b48ead", 6: "#88c0d0", 7: "#e5e9f0",
            },
        },
        {
            "name": "Dracula",
            "background": "#282a36",
            "foreground": "#f8f8f2",
            "brightness": 40,
            "warmth": 0.0,
            "palette": {
                0: "#21222c", 1: "#ff5555", 2: "#50fa7b", 3: "#f1fa8c",
                4: "#bd93f9", 5: "#ff79c6", 6: "#8be9fd", 7: "#f8f8f2",
            },
        },
        {
            "name": "One Light",
            "background": "#fafafa",
            "foreground": "#383a42",
            "brightness": 250,
            "warmth": 0.0,
            "palette": {
                0: "#383a42", 1: "#e45649", 2: "#50a14f", 3: "#c18401",
                4: "#4078f2", 5: "#a626a4", 6: "#0184bc", 7: "#a0a1a7",
            },
        },
    ]


@pytest.fixture
def dark_theme():
    """A single dark theme for testing."""
    return {
        "name": "Test Dark",
        "background": "#1e1e1e",
        "foreground": "#d4d4d4",
        "brightness": 30,
        "warmth": 0.0,
        "palette": {
            0: "#000000", 1: "#ff0000", 2: "#00ff00", 3: "#ffff00",
            4: "#0000ff", 5: "#ff00ff", 6: "#00ffff", 7: "#ffffff",
        },
    }


@pytest.fixture
def light_theme():
    """A single light theme for testing."""
    return {
        "name": "Test Light",
        "background": "#ffffff",
        "foreground": "#333333",
        "brightness": 255,
        "warmth": 0.0,
        "palette": {
            0: "#000000", 1: "#c41a16", 2: "#007400", 3: "#826b28",
            4: "#0000ff", 5: "#a90d91", 6: "#318495", 7: "#ffffff",
        },
    }


@pytest.fixture
def sample_embedding():
    """A sample 20D embedding vector for testing."""
    return np.array([
        25.0, -3.5, -8.2,   # bg LAB
        85.0, -2.0, 5.0,    # fg LAB
        60.0,               # contrast
        0.4,                # avg_chroma (normalized)
        0.1,                # brightness (normalized)
        0.4,                # warmth (normalized)
        0.3, 0.2, 0.3, 0.2, # hue quadrants
        0.5, 0.3, 0.4, 0.2, # harmony scores
        0.3,                # color variety
        0.4,                # lightness range
    ], dtype=np.float32)


@pytest.fixture
def sample_observation():
    """A sample Observation instance for testing."""
    from ghostty_ambient.observations import Observation

    return Observation(
        timestamp=datetime.now(),
        theme_name="Tokyo Night",
        embedding=np.zeros(20, dtype=np.float32),
        context={"time": "night", "lux": "dim", "system": "dark"},
        source="picker",
    )


@pytest.fixture
def observation_store_with_data():
    """An ObservationStore with pre-populated observations."""
    from ghostty_ambient.observations import Observation, ObservationStore

    store = ObservationStore()
    now = datetime.now()

    # Add observations spread over time with different contexts
    observations = [
        Observation(
            timestamp=now - timedelta(days=10),
            theme_name="Gruvbox Dark",
            embedding=np.random.randn(20).astype(np.float32),
            context={"time": "evening", "lux": "dim"},
            source="picker",
        ),
        Observation(
            timestamp=now - timedelta(days=5),
            theme_name="Tokyo Night",
            embedding=np.random.randn(20).astype(np.float32),
            context={"time": "night", "lux": "moonlight"},
            source="picker",
        ),
        Observation(
            timestamp=now - timedelta(days=3),
            theme_name="Tokyo Night",
            embedding=np.random.randn(20).astype(np.float32),
            context={"time": "night", "lux": "dim"},
            source="ideal",
        ),
        Observation(
            timestamp=now - timedelta(days=1),
            theme_name="Nord",
            embedding=np.random.randn(20).astype(np.float32),
            context={"time": "morning", "lux": "office"},
            source="picker",
        ),
        Observation(
            timestamp=now - timedelta(hours=6),
            theme_name="Tokyo Night",
            embedding=np.random.randn(20).astype(np.float32),
            context={"time": "afternoon", "lux": "bright"},
            source="manual",
        ),
    ]

    for obs in observations:
        store.add(obs)

    return store


@pytest.fixture
def phase_detector():
    """A fresh PhaseDetector instance."""
    from ghostty_ambient.phase_detector import PhaseDetector

    return PhaseDetector()


@pytest.fixture
def embedding_index_with_themes(sample_theme_dicts):
    """An EmbeddingIndex built from sample themes."""
    from ghostty_ambient.embeddings import EmbeddingIndex

    index = EmbeddingIndex()
    index.build_from_themes(sample_theme_dicts)
    return index
