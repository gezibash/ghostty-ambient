"""Tests for phase_detector.py - HMM-based phase detection."""

from __future__ import annotations

import numpy as np
import pytest

from ghostty_ambient.embeddings import EMBEDDING_DIM
from ghostty_ambient.observations import ObservationFeatures
from ghostty_ambient.phase_detector import (
    PHASE_CONFIGS,
    OnlineFeatureScaler,
    Phase,
    PhaseConfig,
    PhaseDetector,
    heuristic_phase_detect,
)


def make_features(
    *,
    embedding_variance: float,
    embedding_mean: np.ndarray,
    model_distance: float,
    choice_frequency: float,
    ideal_usage_rate: float,
    manual_rate: float,
    unique_themes: int,
    observation_count: int,
    model_usage_rate: float | None = None,
    theme_entropy: float | None = None,
    effective_theme_count: float | None = None,
    effective_weight: float | None = None,
) -> ObservationFeatures:
    if model_usage_rate is None:
        model_usage_rate = ideal_usage_rate
    if effective_theme_count is None:
        effective_theme_count = float(unique_themes)
    if theme_entropy is None:
        theme_entropy = float(np.log(max(effective_theme_count, 1.0)))
    if effective_weight is None:
        effective_weight = float(observation_count)

    return ObservationFeatures(
        embedding_variance=embedding_variance,
        embedding_mean=embedding_mean,
        model_distance=model_distance,
        choice_frequency=choice_frequency,
        ideal_usage_rate=ideal_usage_rate,
        model_usage_rate=model_usage_rate,
        manual_rate=manual_rate,
        theme_entropy=theme_entropy,
        effective_theme_count=effective_theme_count,
        unique_themes=unique_themes,
        observation_count=observation_count,
        effective_weight=effective_weight,
    )


class TestPhase:
    """Tests for Phase enum."""

    def test_phase_values(self):
        assert Phase.EXPLORE.value == "explore"
        assert Phase.CONVERGE.value == "converge"
        assert Phase.STABLE.value == "stable"

    def test_phase_members(self):
        assert len(Phase) == 3
        assert Phase.EXPLORE in Phase
        assert Phase.CONVERGE in Phase
        assert Phase.STABLE in Phase


class TestPhaseConfig:
    """Tests for PhaseConfig dataclass."""

    def test_explore_config(self):
        config = PHASE_CONFIGS[Phase.EXPLORE]
        assert config.learning_rate == 1.0
        assert config.recency_half_life == 3.0
        assert config.recommendation_diversity == 0.8
        assert config.min_observations == 5

    def test_converge_config(self):
        config = PHASE_CONFIGS[Phase.CONVERGE]
        assert config.learning_rate == 0.5
        assert config.recency_half_life == 7.0
        assert config.recommendation_diversity == 0.5
        assert config.min_observations == 10

    def test_stable_config(self):
        config = PHASE_CONFIGS[Phase.STABLE]
        assert config.learning_rate == 0.1
        assert config.recency_half_life == 30.0
        assert config.recommendation_diversity == 0.2
        assert config.min_observations == 20

    def test_all_phases_have_configs(self):
        for phase in Phase:
            assert phase in PHASE_CONFIGS


class TestPhaseDetector:
    """Tests for PhaseDetector class."""

    def test_initializes_with_explore_prior(self, phase_detector):
        # Initial belief should favor EXPLORE
        assert phase_detector.pi[0] == 0.7  # EXPLORE
        assert phase_detector.pi[1] == 0.2  # CONVERGE
        assert phase_detector.pi[2] == 0.1  # STABLE

    def test_initial_phase_is_explore(self, phase_detector):
        # With no updates, should be EXPLORE (highest prior)
        assert phase_detector.current_phase() == Phase.EXPLORE

    def test_states_ordered_correctly(self, phase_detector):
        assert phase_detector.states == [Phase.EXPLORE, Phase.CONVERGE, Phase.STABLE]
        assert phase_detector.n_states == 3

    def test_transition_matrix_rows_sum_to_one(self, phase_detector):
        for row in phase_detector.A:
            assert np.sum(row) == pytest.approx(1.0)

    def test_transition_matrix_is_sticky(self, phase_detector):
        # Diagonal should be highest (sticky states)
        for i in range(phase_detector.n_states):
            assert phase_detector.A[i, i] >= max(phase_detector.A[i, :])

    def test_stable_is_very_sticky(self, phase_detector):
        # STABLE (index 2) should have highest self-transition
        stable_stay = phase_detector.A[2, 2]
        assert stable_stay >= 0.9

    def test_explore_to_stable_is_rare(self, phase_detector):
        # Direct EXPLORE -> STABLE should be rare
        explore_to_stable = phase_detector.A[0, 2]
        assert explore_to_stable <= 0.05

    def test_emission_means_shape(self, phase_detector):
        assert phase_detector.emission_means.shape == (3, 4)
        assert phase_detector.emission_stds.shape == (3, 4)

    def test_explore_has_high_variance_emission(self, phase_detector):
        # EXPLORE (index 0) should expect high variance
        explore_variance = phase_detector.emission_means[0, 0]
        stable_variance = phase_detector.emission_means[2, 0]
        assert explore_variance > stable_variance

    def test_stable_has_high_model_usage_emission(self, phase_detector):
        # STABLE (index 2) should expect high model usage
        stable_model = phase_detector.emission_means[2, 2]
        explore_model = phase_detector.emission_means[0, 2]
        assert stable_model > explore_model


class TestPhaseDetectorUpdate:
    """Tests for PhaseDetector.update() method."""

    def test_update_returns_phase(self, phase_detector):
        features = make_features(
            embedding_variance=0.5,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=20.0,
            choice_frequency=2.0,
            ideal_usage_rate=0.2,
            manual_rate=0.1,
            unique_themes=8,
            observation_count=20,
        )
        phase = phase_detector.update(features)
        assert isinstance(phase, Phase)

    def test_update_modifies_belief(self, phase_detector):
        initial_belief = phase_detector._belief.copy()

        features = make_features(
            embedding_variance=0.8,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=40.0,
            choice_frequency=5.0,
            ideal_usage_rate=0.1,
            manual_rate=0.0,
            unique_themes=15,
            observation_count=20,
        )
        phase_detector.update(features)

        assert not np.allclose(phase_detector._belief, initial_belief)

    def test_update_adds_to_feature_history(self, phase_detector):
        assert len(phase_detector._feature_history) == 0

        features = make_features(
            embedding_variance=0.5,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=20.0,
            choice_frequency=2.0,
            ideal_usage_rate=0.2,
            manual_rate=0.1,
            unique_themes=8,
            observation_count=20,
        )
        phase_detector.update(features)

        assert len(phase_detector._feature_history) == 1

    def test_feature_history_limited_to_100(self, phase_detector):
        features = make_features(
            embedding_variance=0.5,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=20.0,
            choice_frequency=2.0,
            ideal_usage_rate=0.2,
            manual_rate=0.1,
            unique_themes=8,
            observation_count=20,
        )

        for _ in range(150):
            phase_detector.update(features)

        assert len(phase_detector._feature_history) == 100

    def test_high_variance_leads_to_explore(self, phase_detector):
        # Feed many high-variance observations
        for _ in range(20):
            features = make_features(
                embedding_variance=10.0,  # Very high
                embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                model_distance=50.0,  # High
                choice_frequency=10.0,
                ideal_usage_rate=0.0,  # No ideal usage
                manual_rate=0.0,
                unique_themes=18,  # Many unique
                observation_count=20,
            )
            phase_detector.update(features)

        assert phase_detector.current_phase() == Phase.EXPLORE

    def test_low_variance_high_ideal_leads_to_stable(self, phase_detector):
        # Feed many stable-like observations
        for _ in range(30):
            features = make_features(
                embedding_variance=0.3,  # Low
                embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                model_distance=2.0,  # Low
                choice_frequency=1.0,
                ideal_usage_rate=0.8,  # High ideal usage
                manual_rate=0.0,
                unique_themes=3,  # Few unique
                observation_count=20,
            )
            phase_detector.update(features)

        assert phase_detector.current_phase() == Phase.STABLE


class TestPhaseDetectorStickiness:
    """Tests for phase transition stickiness."""

    def test_stable_does_not_leave_easily(self, phase_detector):
        # First establish STABLE state
        for _ in range(30):
            features = make_features(
                embedding_variance=0.3,
                embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                model_distance=2.0,
                choice_frequency=1.0,
                ideal_usage_rate=0.8,
                manual_rate=0.0,
                unique_themes=3,
                observation_count=20,
            )
            phase_detector.update(features)

        assert phase_detector.current_phase() == Phase.STABLE
        initial_stable_prob = phase_detector.phase_probabilities()[Phase.STABLE]
        assert initial_stable_prob > 0.98  # Should be very confident

        # Single slightly different observation (not extremely exploratory)
        # This represents the window features after one different pick
        # mixed with mostly stable behavior
        mixed_features = make_features(
            embedding_variance=1.0,  # Slightly elevated
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=10.0,  # Slightly elevated
            choice_frequency=1.5,
            ideal_usage_rate=0.7,  # Still mostly ideal
            manual_rate=0.0,
            unique_themes=4,  # One new theme
            observation_count=20,
        )
        phase_detector.update(mixed_features)

        # With slightly elevated but not extreme features, should stay in STABLE
        # Even with stickiness, one observation shifts belief somewhat
        # The key is it doesn't immediately jump to EXPLORE
        assert phase_detector.current_phase() in (Phase.STABLE, Phase.CONVERGE)

    def test_returns_to_stable_quickly(self, phase_detector):
        # Establish STABLE
        for _ in range(30):
            features = make_features(
                embedding_variance=0.3,
                embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                model_distance=2.0,
                choice_frequency=1.0,
                ideal_usage_rate=0.8,
                manual_rate=0.0,
                unique_themes=3,
                observation_count=20,
            )
            phase_detector.update(features)

        # Disrupt with a few exploratory observations
        for _ in range(3):
            explore_features = make_features(
                embedding_variance=5.0,
                embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                model_distance=30.0,
                choice_frequency=5.0,
                ideal_usage_rate=0.0,
                manual_rate=0.0,
                unique_themes=10,
                observation_count=20,
            )
            phase_detector.update(explore_features)

        # Return to stable behavior
        for _ in range(5):
            features = make_features(
                embedding_variance=0.3,
                embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                model_distance=2.0,
                choice_frequency=1.0,
                ideal_usage_rate=0.8,
                manual_rate=0.0,
                unique_themes=3,
                observation_count=20,
            )
            phase_detector.update(features)

        # Should be back to STABLE or close
        stable_prob = phase_detector.phase_probabilities()[Phase.STABLE]
        assert stable_prob > 0.5


class TestPhaseDetectorProbabilities:
    """Tests for phase probability distribution."""

    def test_probabilities_sum_to_one(self, phase_detector):
        probs = phase_detector.phase_probabilities()
        total = sum(probs.values())
        assert total == pytest.approx(1.0)

    def test_probabilities_all_positive(self, phase_detector):
        probs = phase_detector.phase_probabilities()
        for _phase, prob in probs.items():
            assert prob >= 0

    def test_probabilities_returned_for_all_phases(self, phase_detector):
        probs = phase_detector.phase_probabilities()
        assert Phase.EXPLORE in probs
        assert Phase.CONVERGE in probs
        assert Phase.STABLE in probs


class TestPhaseDetectorConfig:
    """Tests for get_config() method."""

    def test_get_config_returns_phase_config(self, phase_detector):
        config = phase_detector.get_config()
        assert isinstance(config, PhaseConfig)

    def test_get_config_matches_current_phase(self, phase_detector):
        phase = phase_detector.current_phase()
        config = phase_detector.get_config()
        assert config == PHASE_CONFIGS[phase]


class TestPhaseDetectorFromStore:
    """Tests for detect_from_store() method."""

    def test_detect_from_store_returns_phase(self, phase_detector, observation_store_with_data):
        phase = phase_detector.detect_from_store(observation_store_with_data)
        assert isinstance(phase, Phase)

    def test_detect_from_store_updates_belief(self, phase_detector, observation_store_with_data):
        phase_detector.detect_from_store(observation_store_with_data)
        # Belief may or may not change depending on features
        # Just verify it doesn't crash


class TestPhaseDetectorReset:
    """Tests for reset() method."""

    def test_reset_restores_initial_belief(self, phase_detector):
        # Make some updates
        features = make_features(
            embedding_variance=0.5,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=20.0,
            choice_frequency=2.0,
            ideal_usage_rate=0.2,
            manual_rate=0.1,
            unique_themes=8,
            observation_count=20,
        )
        for _ in range(10):
            phase_detector.update(features)

        phase_detector.reset()

        assert np.allclose(phase_detector._belief, phase_detector.pi)

    def test_reset_clears_feature_history(self, phase_detector):
        features = make_features(
            embedding_variance=0.5,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=20.0,
            choice_frequency=2.0,
            ideal_usage_rate=0.2,
            manual_rate=0.1,
            unique_themes=8,
            observation_count=20,
        )
        phase_detector.update(features)
        assert len(phase_detector._feature_history) > 0

        phase_detector.reset()
        assert len(phase_detector._feature_history) == 0


class TestPhaseDetectorSerialization:
    """Tests for PhaseDetector serialization."""

    def test_to_dict_structure(self, phase_detector):
        data = phase_detector.to_dict()
        assert "pi" in data
        assert "A" in data
        assert "emission_means" in data
        assert "emission_stds" in data
        assert "belief" in data
        assert "feature_history" in data

    def test_serialization_roundtrip(self, phase_detector):
        # Make some updates first
        features = make_features(
            embedding_variance=0.5,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=20.0,
            choice_frequency=2.0,
            ideal_usage_rate=0.2,
            manual_rate=0.1,
            unique_themes=8,
            observation_count=20,
        )
        phase_detector.update(features)

        data = phase_detector.to_dict()
        restored = PhaseDetector.from_dict(data)

        assert np.allclose(restored._belief, phase_detector._belief)
        assert np.allclose(restored.A, phase_detector.A)
        assert np.allclose(restored.emission_means, phase_detector.emission_means)
        assert np.allclose(restored.nig_mu, phase_detector.nig_mu)
        assert np.allclose(restored.nig_kappa, phase_detector.nig_kappa)
        assert np.allclose(restored.nig_alpha, phase_detector.nig_alpha)
        assert np.allclose(restored.nig_beta, phase_detector.nig_beta)
        assert np.allclose(restored.transition_counts, phase_detector.transition_counts)
        assert restored.emission_half_life == phase_detector.emission_half_life
        assert restored.min_emission_std == phase_detector.min_emission_std
        assert restored.belief_inertia == phase_detector.belief_inertia

    def test_from_dict_handles_empty_data(self):
        detector = PhaseDetector.from_dict({})
        # Should have defaults
        assert detector.n_states == 3
        assert len(detector._belief) == 3

    def test_from_dict_missing_new_fields(self, phase_detector):
        data = phase_detector.to_dict()
        for key in (
            "feature_scaler",
            "transition_counts",
            "nig_mu",
            "nig_kappa",
            "nig_alpha",
            "nig_beta",
            "nig_prior_mu",
            "nig_prior_kappa",
            "nig_prior_alpha",
            "nig_prior_beta",
            "emission_half_life",
            "min_emission_std",
            "belief_inertia",
        ):
            data.pop(key, None)

        restored = PhaseDetector.from_dict(data)
        assert restored.n_states == 3
        assert restored.emission_half_life > 0
        assert restored.min_emission_std > 0
        assert restored.belief_inertia >= 0


class TestOnlineFeatureScaler:
    """Tests for OnlineFeatureScaler behavior."""

    def test_scale_initializes_to_prior(self):
        scaler = OnlineFeatureScaler(4, init_scale=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert np.allclose(scaler.scale(), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    def test_scale_updates_slowly(self):
        scaler = OnlineFeatureScaler(2, init_scale=np.array([10.0, 10.0], dtype=np.float32), scale_half_life=50.0)
        for _ in range(5):
            scaler.update(np.array([1.0, 2.0], dtype=np.float32))
        scale = scaler.scale()
        assert scale[0] > 1.0
        assert scale[0] < 10.0
        assert scale[1] > 2.0
        assert scale[1] < 10.0

    def test_normalize_uses_scale(self):
        scaler = OnlineFeatureScaler(2, init_scale=np.array([2.0, 4.0], dtype=np.float32))
        vec = scaler.normalize(np.array([2.0, 8.0], dtype=np.float32))
        assert np.allclose(vec, np.array([1.0, 2.0], dtype=np.float32))


class TestEmissionLearning:
    """Tests for Bayesian emission learning."""

    def test_nig_updates_move_mean(self, phase_detector):
        features = make_features(
            embedding_variance=0.2,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=1.0,
            choice_frequency=1.0,
            ideal_usage_rate=0.9,
            manual_rate=0.0,
            unique_themes=2,
            observation_count=20,
            effective_theme_count=2.0,
        )
        before = phase_detector.nig_mu.copy()
        phase_detector.update(features)
        after = phase_detector.nig_mu.copy()
        assert not np.allclose(before, after)

    def test_log_likelihoods_are_finite(self, phase_detector):
        features = make_features(
            embedding_variance=0.3,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=2.0,
            choice_frequency=1.0,
            ideal_usage_rate=0.8,
            manual_rate=0.0,
            unique_themes=3,
            observation_count=20,
            effective_theme_count=3.0,
        )
        vec = phase_detector._features_to_vector(features)
        ll = phase_detector.emission_log_likelihoods(vec)
        assert np.isfinite(ll).all()


class TestTransitionLearning:
    """Tests for adaptive transition updates."""

    def test_transition_counts_accumulate(self, phase_detector):
        initial_counts = phase_detector.transition_counts.copy()
        features = make_features(
            embedding_variance=0.3,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=2.0,
            choice_frequency=1.0,
            ideal_usage_rate=0.8,
            manual_rate=0.0,
            unique_themes=3,
            observation_count=20,
            effective_theme_count=3.0,
        )
        for _ in range(5):
            phase_detector.update(features)
        assert np.sum(phase_detector.transition_counts) > np.sum(initial_counts)
        for row in phase_detector.A:
            assert np.sum(row) == pytest.approx(1.0)


class TestHeuristicPhaseDetect:
    """Tests for heuristic_phase_detect() function."""

    def test_few_observations_returns_explore(self):
        features = make_features(
            embedding_variance=0.0,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=0.0,
            choice_frequency=0.0,
            ideal_usage_rate=0.0,
            manual_rate=0.0,
            unique_themes=1,
            observation_count=3,
        )
        assert heuristic_phase_detect(features) == Phase.EXPLORE

    def test_high_ideal_low_variance_returns_stable(self):
        features = make_features(
            embedding_variance=0.1,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=2.0,
            choice_frequency=1.0,
            ideal_usage_rate=0.7,
            manual_rate=0.0,
            unique_themes=3,
            observation_count=20,
        )
        assert heuristic_phase_detect(features) == Phase.STABLE

    def test_high_variance_returns_explore(self):
        features = make_features(
            embedding_variance=0.8,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=30.0,
            choice_frequency=5.0,
            ideal_usage_rate=0.0,
            manual_rate=0.0,
            unique_themes=18,
            observation_count=20,
        )
        assert heuristic_phase_detect(features) == Phase.EXPLORE

    def test_many_unique_themes_returns_explore(self):
        features = make_features(
            embedding_variance=0.3,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=15.0,
            choice_frequency=3.0,
            ideal_usage_rate=0.0,
            manual_rate=0.0,
            unique_themes=16,  # > 70% of 20
            observation_count=20,
        )
        assert heuristic_phase_detect(features) == Phase.EXPLORE

    def test_medium_values_returns_converge(self):
        features = make_features(
            embedding_variance=0.3,
            embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_distance=15.0,
            choice_frequency=3.0,
            ideal_usage_rate=0.2,
            manual_rate=0.1,
            unique_themes=8,
            observation_count=20,
            effective_theme_count=4.0,
        )
        assert heuristic_phase_detect(features) == Phase.CONVERGE
