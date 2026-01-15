"""Shared pytest fixtures for ghostty-ambient tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_history_data():
    """Fresh history data structure for testing."""
    return {
        "events": [],
        "global_beta": {},
        "factor_beta": {},
        "favorites": [],
        "disliked": [],
        "recent_snapshots": [],
    }


@pytest.fixture
def sample_themes():
    """Sample theme dicts for testing."""
    return [
        {"name": "dark_theme", "brightness": 30, "warmth": 0.0},
        {"name": "light_theme", "brightness": 220, "warmth": 0.0},
        {"name": "warm_theme", "brightness": 127, "warmth": 0.5},
        {"name": "cool_theme", "brightness": 127, "warmth": -0.5},
        {"name": "neutral_theme", "brightness": 127, "warmth": 0.0},
    ]


@pytest.fixture
def history_with_data(mock_history_data):
    """History data with some pre-populated Beta distributions."""
    data = mock_history_data.copy()

    # Add some factor_beta data for dark_theme
    data["factor_beta"] = {
        "dark_theme": {
            "time:night": {"alpha": 5, "beta": 1},
            "lux:moonlight": {"alpha": 4, "beta": 1},
            "system:dark": {"alpha": 6, "beta": 2},
        },
        "light_theme": {
            "time:afternoon": {"alpha": 4, "beta": 1},
            "lux:daylight": {"alpha": 5, "beta": 1},
            "system:light": {"alpha": 5, "beta": 2},
        },
    }

    # Add global_beta
    data["global_beta"] = {
        "dark_theme": {"alpha": 10, "beta": 3},
        "light_theme": {"alpha": 8, "beta": 4},
    }

    return data


@pytest.fixture
def mock_history(tmp_path, mock_history_data):
    """Create a History instance with mocked file path."""
    from ghostty_ambient.history import History, HISTORY_FILE

    # Create a temporary history file
    history_file = tmp_path / "history.json"

    with patch.object(History, "_load", return_value=mock_history_data):
        with patch("ghostty_ambient.history.HISTORY_FILE", history_file):
            history = History()
            history.data = mock_history_data
            yield history
