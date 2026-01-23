"""Long-horizon behavioral scenario tests using mock history.json files."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from ghostty_ambient.adaptive_model import AdaptivePreferenceModel
from ghostty_ambient.observations import Observation, ObservationStore
from ghostty_ambient.phase_detector import PhaseDetector

FIXTURE_DIR = Path("tests/fixtures")


def _load_fixture(name: str, tmp_path: Path) -> AdaptivePreferenceModel:
    src = FIXTURE_DIR / name
    dst = tmp_path / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return AdaptivePreferenceModel.load(dst)


def _replay_features_and_beliefs(
    observations: list[Observation],
) -> tuple[list[ObservationStore], list[dict[str, float]]]:
    detector = PhaseDetector()
    snapshots: list[dict[str, float]] = []
    stores: list[ObservationStore] = []
    anchor = datetime.now()
    for idx, obs in enumerate(observations):
        store = ObservationStore()
        shift = anchor - obs.timestamp
        for prev in observations[: idx + 1]:
            shifted = Observation(
                timestamp=prev.timestamp + shift,
                theme_name=prev.theme_name,
                embedding=prev.embedding,
                context=prev.context,
                source=prev.source,
            )
            store.add(shifted)
        features = store.compute_decayed_features(detector.get_config().recency_half_life)
        detector.update(features)
        probs = detector.phase_probabilities()
        snapshots.append({p.value: float(v) for p, v in probs.items()})
        stores.append(store)
    return stores, snapshots


def test_history_tectonic_shift(tmp_path: Path) -> None:
    """Stable -> explore -> stable shift yields recovery to STABLE."""
    model = _load_fixture("history_tectonic_shift.json", tmp_path)
    stores, probs = _replay_features_and_beliefs(model.observations.observations)
    mid_features = stores[69].compute_features(window_size=20)
    explore_features = stores[79].compute_features(window_size=10)
    end_features = stores[-1].compute_features(window_size=20)

    assert explore_features.embedding_variance > mid_features.embedding_variance * 5.0
    assert explore_features.effective_theme_count > mid_features.effective_theme_count
    assert end_features.embedding_variance < explore_features.embedding_variance / 5.0
    assert probs[-1]["stable"] > 0.7


def test_history_relentless_explorer(tmp_path: Path) -> None:
    """Sustained high-variance behavior stays in EXPLORE."""
    model = _load_fixture("history_relentless_explorer.json", tmp_path)
    stores, probs = _replay_features_and_beliefs(model.observations.observations)
    end_features = stores[-1].compute_features(window_size=50)
    assert end_features.embedding_variance > 5.0
    assert probs[-1]["explore"] > 0.7


def test_history_autopilot_lock_in(tmp_path: Path) -> None:
    """Daemon/ideal dominance with low variance stabilizes quickly."""
    model = _load_fixture("history_autopilot_lock_in.json", tmp_path)
    stores, probs = _replay_features_and_beliefs(model.observations.observations)
    end_features = stores[-1].compute_features(window_size=30)
    assert end_features.model_usage_rate > 0.9
    assert end_features.effective_theme_count < 2.5
    assert probs[-1]["stable"] > probs[5]["stable"]


def test_history_context_oscillation(tmp_path: Path) -> None:
    """Two-context oscillation should converge rather than explode."""
    model = _load_fixture("history_context_oscillation.json", tmp_path)
    stores, probs = _replay_features_and_beliefs(model.observations.observations)
    end_features = stores[-1].compute_features(window_size=30)
    assert 1.5 <= end_features.effective_theme_count <= 2.5
    assert end_features.embedding_variance > 5.0
    assert probs[-1]["explore"] > 0.8
