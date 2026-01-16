"""Tests for observations.py - Timestamped observation store."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from ghostty_ambient.embeddings import EMBEDDING_DIM
from ghostty_ambient.observations import (
    Observation,
    ObservationFeatures,
    ObservationStore,
)


class TestObservation:
    """Tests for Observation dataclass."""

    def test_creates_with_required_fields(self):
        obs = Observation(
            timestamp=datetime.now(),
            theme_name="Tokyo Night",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context={"time": "night"},
            source="picker",
        )
        assert obs.theme_name == "Tokyo Night"
        assert obs.source == "picker"

    def test_age_days_returns_correct_value(self):
        now = datetime.now()
        obs = Observation(
            timestamp=now - timedelta(days=5),
            theme_name="Test",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context={},
            source="picker",
        )
        age = obs.age_days(now)
        assert age == pytest.approx(5.0, abs=0.01)

    def test_age_days_uses_current_time_by_default(self):
        obs = Observation(
            timestamp=datetime.now() - timedelta(days=3),
            theme_name="Test",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context={},
            source="picker",
        )
        age = obs.age_days()
        assert age == pytest.approx(3.0, abs=0.1)

    def test_age_hours_returns_correct_value(self):
        now = datetime.now()
        obs = Observation(
            timestamp=now - timedelta(hours=12),
            theme_name="Test",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context={},
            source="picker",
        )
        age = obs.age_hours(now)
        assert age == pytest.approx(12.0, abs=0.01)

    def test_age_hours_uses_current_time_by_default(self):
        obs = Observation(
            timestamp=datetime.now() - timedelta(hours=6),
            theme_name="Test",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context={},
            source="picker",
        )
        age = obs.age_hours()
        assert age == pytest.approx(6.0, abs=0.1)

    def test_serialization_roundtrip(self, sample_observation):
        data = sample_observation.to_dict()

        assert "timestamp" in data
        assert "theme_name" in data
        assert "embedding" in data
        assert "context" in data
        assert "source" in data
        assert isinstance(data["embedding"], list)

        restored = Observation.from_dict(data)
        assert restored.theme_name == sample_observation.theme_name
        assert restored.source == sample_observation.source
        assert restored.context == sample_observation.context
        assert np.allclose(restored.embedding, sample_observation.embedding)

    def test_context_preserves_all_factors(self):
        context = {
            "time": "night",
            "lux": "dim",
            "system": "dark",
            "weather": "clear",
        }
        obs = Observation(
            timestamp=datetime.now(),
            theme_name="Test",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context=context,
            source="picker",
        )
        assert obs.context == context


class TestObservationFeatures:
    """Tests for ObservationFeatures dataclass."""

    def test_creates_with_all_fields(self):
        features = ObservationFeatures(
            embedding_variance=0.5,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=10.0,
            choice_frequency=5.0,
            ideal_usage_rate=0.3,
            manual_rate=0.1,
            unique_themes=5,
            observation_count=20,
        )
        assert features.embedding_variance == 0.5
        assert features.unique_themes == 5


class TestObservationStore:
    """Tests for ObservationStore class."""

    def test_creates_empty_store(self):
        store = ObservationStore()
        assert len(store) == 0
        assert store.observations == []

    def test_add_single_observation(self, sample_observation):
        store = ObservationStore()
        store.add(sample_observation)
        assert len(store) == 1

    def test_add_multiple_observations(self):
        store = ObservationStore()
        for i in range(10):
            obs = Observation(
                timestamp=datetime.now() - timedelta(days=i),
                theme_name=f"Theme {i}",
                embedding=np.random.randn(EMBEDDING_DIM).astype(np.float32),
                context={},
                source="picker",
            )
            store.add(obs)
        assert len(store) == 10

    def test_respects_max_observations(self):
        store = ObservationStore(max_observations=100)
        for i in range(150):
            obs = Observation(
                timestamp=datetime.now(),
                theme_name=f"Theme {i}",
                embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                context={},
                source="picker",
            )
            store.add(obs)
        # Should trim oldest 10% when exceeding max
        assert len(store) <= 100

    def test_trimming_removes_oldest(self):
        store = ObservationStore(max_observations=20)
        for i in range(30):
            obs = Observation(
                timestamp=datetime.now() + timedelta(seconds=i),
                theme_name=f"Theme {i}",
                embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                context={},
                source="picker",
            )
            store.add(obs)

        # Oldest themes should be gone
        names = [o.theme_name for o in store.observations]
        assert "Theme 0" not in names
        assert "Theme 29" in names

    def test_running_mean_updated_incrementally(self):
        store = ObservationStore()

        # Add observation with known embedding
        obs1 = Observation(
            timestamp=datetime.now(),
            theme_name="Test1",
            embedding=np.ones(EMBEDDING_DIM, dtype=np.float32) * 2,
            context={},
            source="picker",
        )
        store.add(obs1)
        assert store._running_mean is not None
        assert np.allclose(store._running_mean, np.ones(EMBEDDING_DIM) * 2)

        # Add second observation
        obs2 = Observation(
            timestamp=datetime.now(),
            theme_name="Test2",
            embedding=np.ones(EMBEDDING_DIM, dtype=np.float32) * 4,
            context={},
            source="picker",
        )
        store.add(obs2)
        # Mean should be (2 + 4) / 2 = 3
        assert np.allclose(store._running_mean, np.ones(EMBEDDING_DIM) * 3)

    def test_recent_returns_n_most_recent(self, observation_store_with_data):
        recent = observation_store_with_data.recent(3)
        assert len(recent) == 3
        # Should be ordered by time (most recent last in list)
        for i in range(len(recent) - 1):
            assert recent[i].timestamp <= recent[i + 1].timestamp

    def test_recent_with_n_greater_than_count(self, observation_store_with_data):
        recent = observation_store_with_data.recent(100)
        assert len(recent) == len(observation_store_with_data)

    def test_recent_on_empty_store(self):
        store = ObservationStore()
        recent = store.recent(5)
        assert recent == []

    def test_in_window_returns_recent_observations(self, observation_store_with_data):
        # Should include observations from last 7 days
        window = observation_store_with_data.in_window(7)
        assert len(window) > 0
        for obs in window:
            assert obs.age_days() <= 7

    def test_in_window_excludes_old_observations(self, observation_store_with_data):
        # 2-day window should exclude older observations
        window = observation_store_with_data.in_window(2)
        for obs in window:
            assert obs.age_days() <= 2

    def test_in_window_on_empty_store(self):
        store = ObservationStore()
        window = store.in_window(30)
        assert window == []

    def test_for_context_returns_matching_observations(self, observation_store_with_data):
        # Query for night observations
        matches = observation_store_with_data.for_context(
            {"time": "night"},
            days=30,
        )
        for obs in matches:
            assert obs.context.get("time") == "night"

    def test_for_context_with_multiple_factors(self, observation_store_with_data):
        # Query with multiple factors - should match if ANY factor matches
        matches = observation_store_with_data.for_context(
            {"time": "night", "lux": "moonlight"},
            days=30,
        )
        for obs in matches:
            time_matches = obs.context.get("time") == "night"
            lux_matches = obs.context.get("lux") == "moonlight"
            assert time_matches or lux_matches

    def test_for_context_with_empty_context_returns_all(self, observation_store_with_data):
        matches = observation_store_with_data.for_context({}, days=30)
        window = observation_store_with_data.in_window(30)
        assert len(matches) == len(window)

    def test_for_context_respects_time_window(self):
        store = ObservationStore()
        now = datetime.now()

        # Old observation matching context
        old_obs = Observation(
            timestamp=now - timedelta(days=60),
            theme_name="Old",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context={"time": "night"},
            source="picker",
        )
        store.add(old_obs)

        # Recent observation matching context
        new_obs = Observation(
            timestamp=now - timedelta(days=1),
            theme_name="New",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context={"time": "night"},
            source="picker",
        )
        store.add(new_obs)

        matches = store.for_context({"time": "night"}, days=30)
        assert len(matches) == 1
        assert matches[0].theme_name == "New"


class TestObservationStoreFeatures:
    """Tests for ObservationStore.compute_features() method."""

    def test_compute_features_with_few_observations(self):
        store = ObservationStore()
        obs = Observation(
            timestamp=datetime.now(),
            theme_name="Test",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context={},
            source="picker",
        )
        store.add(obs)

        features = store.compute_features()
        assert features.observation_count == 1
        assert features.embedding_variance == 0.0

    def test_compute_features_with_many_observations(self, observation_store_with_data):
        features = observation_store_with_data.compute_features()
        assert features.observation_count >= 2
        assert features.embedding_variance >= 0

    def test_compute_features_embedding_variance(self):
        store = ObservationStore()
        now = datetime.now()

        # Add observations with different embeddings
        for i in range(10):
            emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            emb[0] = i  # Vary first dimension
            obs = Observation(
                timestamp=now - timedelta(hours=i),
                theme_name=f"Theme {i}",
                embedding=emb,
                context={},
                source="picker",
            )
            store.add(obs)

        features = store.compute_features()
        assert features.embedding_variance > 0

    def test_compute_features_model_distance(self):
        store = ObservationStore()
        now = datetime.now()

        # Add observations that are distant from running mean
        for i in range(10):
            emb = np.ones(EMBEDDING_DIM, dtype=np.float32) * i
            obs = Observation(
                timestamp=now - timedelta(hours=i),
                theme_name=f"Theme {i}",
                embedding=emb,
                context={},
                source="picker",
            )
            store.add(obs)

        features = store.compute_features()
        assert features.model_distance >= 0

    def test_compute_features_choice_frequency(self):
        store = ObservationStore()
        now = datetime.now()

        # Add 10 observations over 5 days = 2 per day
        for i in range(10):
            obs = Observation(
                timestamp=now - timedelta(days=i / 2),
                theme_name=f"Theme {i}",
                embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                context={},
                source="picker",
            )
            store.add(obs)

        features = store.compute_features()
        assert features.choice_frequency >= 0

    def test_compute_features_source_distribution(self):
        store = ObservationStore()
        now = datetime.now()

        # Add observations with different sources
        sources = ["picker", "picker", "ideal", "ideal", "ideal", "manual"]
        for i, source in enumerate(sources):
            obs = Observation(
                timestamp=now - timedelta(hours=i),
                theme_name=f"Theme {i}",
                embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                context={},
                source=source,
            )
            store.add(obs)

        features = store.compute_features()
        assert features.ideal_usage_rate == pytest.approx(3 / 6)
        assert features.manual_rate == pytest.approx(1 / 6)

    def test_compute_features_unique_themes(self):
        store = ObservationStore()
        now = datetime.now()

        # Add duplicate theme names
        themes = ["A", "A", "B", "B", "B", "C"]
        for i, name in enumerate(themes):
            obs = Observation(
                timestamp=now - timedelta(hours=i),
                theme_name=name,
                embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                context={},
                source="picker",
            )
            store.add(obs)

        features = store.compute_features()
        assert features.unique_themes == 3

    def test_compute_features_window_size(self):
        store = ObservationStore()
        now = datetime.now()

        # Add 100 observations
        for i in range(100):
            obs = Observation(
                timestamp=now - timedelta(hours=i),
                theme_name=f"Theme {i % 20}",
                embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                context={},
                source="picker",
            )
            store.add(obs)

        # Window of 50 should only count last 50
        features = store.compute_features(window_size=50)
        assert features.observation_count == 50


class TestObservationStoreWeightedMean:
    """Tests for ObservationStore.compute_weighted_mean() method."""

    def test_weighted_mean_on_empty_store(self):
        store = ObservationStore()
        mean, weight = store.compute_weighted_mean()
        assert np.allclose(mean, np.zeros(EMBEDDING_DIM))
        assert weight == 0.0

    def test_weighted_mean_single_observation(self):
        store = ObservationStore()
        emb = np.ones(EMBEDDING_DIM, dtype=np.float32) * 5
        obs = Observation(
            timestamp=datetime.now(),
            theme_name="Test",
            embedding=emb,
            context={},
            source="picker",
        )
        store.add(obs)

        mean, weight = store.compute_weighted_mean()
        assert np.allclose(mean, emb, atol=0.01)
        assert weight > 0

    def test_weighted_mean_recent_has_more_weight(self):
        store = ObservationStore()
        now = datetime.now()

        # Old observation
        old_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        store.add(Observation(
            timestamp=now - timedelta(days=30),
            theme_name="Old",
            embedding=old_emb,
            context={},
            source="picker",
        ))

        # Recent observation
        new_emb = np.ones(EMBEDDING_DIM, dtype=np.float32) * 10
        store.add(Observation(
            timestamp=now - timedelta(hours=1),
            theme_name="New",
            embedding=new_emb,
            context={},
            source="picker",
        ))

        mean, _ = store.compute_weighted_mean(half_life_days=7)
        # Mean should be closer to recent observation
        assert mean[0] > 5  # Closer to 10 than to 0

    def test_weighted_mean_respects_half_life(self):
        store = ObservationStore()
        now = datetime.now()

        # Two observations of equal value but different ages
        for days_ago in [0, 7]:
            store.add(Observation(
                timestamp=now - timedelta(days=days_ago),
                theme_name=f"Theme {days_ago}",
                embedding=np.ones(EMBEDDING_DIM, dtype=np.float32),
                context={},
                source="picker",
            ))

        # With 7-day half-life, older should have half the weight
        _, weight = store.compute_weighted_mean(half_life_days=7)
        assert weight == pytest.approx(1.5, abs=0.1)  # 1 + 0.5

    def test_weighted_mean_with_context_filter(self):
        store = ObservationStore()
        now = datetime.now()

        # Night observation
        store.add(Observation(
            timestamp=now - timedelta(hours=1),
            theme_name="Night",
            embedding=np.ones(EMBEDDING_DIM, dtype=np.float32) * 10,
            context={"time": "night"},
            source="picker",
        ))

        # Day observation
        store.add(Observation(
            timestamp=now - timedelta(hours=2),
            theme_name="Day",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context={"time": "afternoon"},
            source="picker",
        ))

        # Filter for night
        mean, _ = store.compute_weighted_mean(context={"time": "night"})
        assert np.allclose(mean, np.ones(EMBEDDING_DIM) * 10, atol=0.1)


class TestObservationStoreSerialization:
    """Tests for ObservationStore serialization."""

    def test_to_dict_structure(self, observation_store_with_data):
        data = observation_store_with_data.to_dict()
        assert "observations" in data
        assert "running_mean" in data
        assert "running_mean_count" in data
        assert isinstance(data["observations"], list)

    def test_serialization_roundtrip(self, observation_store_with_data):
        data = observation_store_with_data.to_dict()
        restored = ObservationStore.from_dict(data)

        assert len(restored) == len(observation_store_with_data)
        assert restored._running_mean_count == observation_store_with_data._running_mean_count

    def test_save_and_load(self, observation_store_with_data, tmp_path):
        path = tmp_path / "observations.json"
        observation_store_with_data.save(path)

        loaded = ObservationStore.load(path)
        assert len(loaded) == len(observation_store_with_data)

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        loaded = ObservationStore.load(path)
        assert len(loaded) == 0

    def test_load_creates_parent_directories(self, tmp_path):
        path = tmp_path / "subdir" / "observations.json"
        store = ObservationStore()
        store.add(Observation(
            timestamp=datetime.now(),
            theme_name="Test",
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            context={},
            source="picker",
        ))
        store.save(path)
        assert path.exists()

    def test_from_dict_recomputes_running_mean_if_missing(self):
        data = {
            "observations": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "theme_name": "Test",
                    "embedding": [1.0] * EMBEDDING_DIM,
                    "context": {},
                    "source": "picker",
                }
            ],
            # No running_mean field
        }
        store = ObservationStore.from_dict(data)
        assert store._running_mean is not None
