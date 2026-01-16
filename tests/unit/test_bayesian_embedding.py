"""Tests for bayesian_embedding.py - Gaussian posteriors over embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from ghostty_ambient.bayesian_embedding import (
    DEFAULT_PRIOR_VAR,
    OBSERVATION_VAR,
    ContextualPosterior,
    EmbeddingPosterior,
)
from ghostty_ambient.embeddings import EMBEDDING_DIM


class TestEmbeddingPosteriorInit:
    """Tests for EmbeddingPosterior initialization."""

    def test_creates_with_prior_mean(self):
        posterior = EmbeddingPosterior()
        assert posterior.mean.shape == (EMBEDDING_DIM,)
        assert np.allclose(posterior.mean, 0.0)

    def test_creates_with_high_prior_variance(self):
        posterior = EmbeddingPosterior()
        assert posterior.variance.shape == (EMBEDDING_DIM,)
        assert np.all(posterior.variance == DEFAULT_PRIOR_VAR)

    def test_starts_with_zero_weight(self):
        posterior = EmbeddingPosterior()
        assert posterior.total_weight == 0.0
        assert posterior.observation_count == 0

    def test_initial_confidence_is_low(self):
        posterior = EmbeddingPosterior()
        assert posterior.confidence < 0.1


class TestEmbeddingPosteriorUpdate:
    """Tests for EmbeddingPosterior.update() method."""

    def test_update_shifts_mean_toward_observation(self):
        posterior = EmbeddingPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32) * 50.0

        posterior.update(obs, weight=1.0)

        # Mean should move toward observation
        assert np.all(posterior.mean > 0)

    def test_update_reduces_variance(self):
        posterior = EmbeddingPosterior()
        initial_var = posterior.variance.copy()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32) * 50.0

        posterior.update(obs, weight=1.0)

        assert np.all(posterior.variance < initial_var)

    def test_update_increases_weight(self):
        posterior = EmbeddingPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32)

        posterior.update(obs, weight=2.5)

        assert posterior.total_weight == 2.5
        assert posterior.observation_count == 1

    def test_multiple_updates_accumulate(self):
        posterior = EmbeddingPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32)

        posterior.update(obs, weight=1.0)
        posterior.update(obs, weight=1.0)
        posterior.update(obs, weight=1.0)

        assert posterior.observation_count == 3
        assert posterior.total_weight == 3.0

    def test_higher_weight_has_more_influence(self):
        post_low = EmbeddingPosterior()
        post_high = EmbeddingPosterior()

        obs = np.ones(EMBEDDING_DIM, dtype=np.float32) * 100.0

        post_low.update(obs, weight=0.1)
        post_high.update(obs, weight=10.0)

        # Higher weight should move mean more
        assert np.mean(post_high.mean) > np.mean(post_low.mean)

    def test_zero_weight_has_no_effect(self):
        posterior = EmbeddingPosterior()
        initial_mean = posterior.mean.copy()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32) * 100.0

        posterior.update(obs, weight=0.0)

        assert np.allclose(posterior.mean, initial_mean)
        assert posterior.observation_count == 0


class TestEmbeddingPosteriorConfidence:
    """Tests for confidence computation."""

    def test_confidence_increases_with_observations(self):
        posterior = EmbeddingPosterior()
        obs = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        conf_before = posterior.confidence

        for _ in range(10):
            posterior.update(obs, weight=1.0)

        assert posterior.confidence > conf_before

    def test_confidence_bounded_0_to_1(self):
        posterior = EmbeddingPosterior()

        assert 0 <= posterior.confidence <= 1

        # After many observations
        obs = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        for _ in range(100):
            posterior.update(obs, weight=1.0)

        assert 0 <= posterior.confidence <= 1

    def test_uncertainty_is_inverse_of_confidence(self):
        posterior = EmbeddingPosterior()

        assert posterior.uncertainty == 1.0 - posterior.confidence

    def test_dimension_confidence_per_dimension(self):
        posterior = EmbeddingPosterior()
        dim_conf = posterior.dimension_confidence()

        assert dim_conf.shape == (EMBEDDING_DIM,)
        assert np.all(dim_conf >= 0)
        assert np.all(dim_conf <= 1)


class TestEmbeddingPosteriorSampling:
    """Tests for Thompson sampling."""

    def test_sample_returns_correct_shape(self):
        posterior = EmbeddingPosterior()
        sample = posterior.sample()

        assert sample.shape == (EMBEDDING_DIM,)
        assert sample.dtype == np.float32

    def test_samples_vary(self):
        posterior = EmbeddingPosterior()

        samples = [posterior.sample() for _ in range(10)]

        # Samples should not all be identical
        assert not all(np.allclose(samples[0], s) for s in samples[1:])

    def test_samples_centered_on_mean(self):
        posterior = EmbeddingPosterior()

        # Set a known mean
        posterior.mean = np.ones(EMBEDDING_DIM, dtype=np.float32) * 50.0
        posterior.variance = np.ones(EMBEDDING_DIM, dtype=np.float32) * 1.0

        samples = np.array([posterior.sample() for _ in range(1000)])
        sample_mean = np.mean(samples, axis=0)

        # Should be close to posterior mean
        assert np.allclose(sample_mean, posterior.mean, atol=1.0)

    def test_reproducible_with_rng(self):
        posterior = EmbeddingPosterior()

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        sample1 = posterior.sample(rng1)
        sample2 = posterior.sample(rng2)

        assert np.allclose(sample1, sample2)


class TestEmbeddingPosteriorSerialization:
    """Tests for serialization."""

    def test_to_dict_structure(self):
        posterior = EmbeddingPosterior()
        posterior.update(np.ones(EMBEDDING_DIM, dtype=np.float32), weight=1.0)

        data = posterior.to_dict()

        assert "mean" in data
        assert "variance" in data
        assert "total_weight" in data
        assert "observation_count" in data

    def test_serialization_roundtrip(self):
        posterior = EmbeddingPosterior()
        obs = np.random.randn(EMBEDDING_DIM).astype(np.float32) * 10
        posterior.update(obs, weight=2.5)

        data = posterior.to_dict()
        restored = EmbeddingPosterior.from_dict(data)

        assert np.allclose(restored.mean, posterior.mean)
        assert np.allclose(restored.variance, posterior.variance)
        assert restored.total_weight == posterior.total_weight
        assert restored.observation_count == posterior.observation_count

    def test_from_observations(self):
        embeddings = [
            np.ones(EMBEDDING_DIM, dtype=np.float32) * i
            for i in range(5)
        ]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        posterior = EmbeddingPosterior.from_observations(embeddings, weights)

        assert posterior.observation_count == 5
        assert posterior.total_weight == 5.0


class TestContextualPosteriorInit:
    """Tests for ContextualPosterior initialization."""

    def test_creates_empty(self):
        cp = ContextualPosterior()
        assert len(cp.posteriors) == 0

    def test_has_global_posterior(self):
        cp = ContextualPosterior()
        assert isinstance(cp.global_posterior, EmbeddingPosterior)


class TestContextualPosteriorUpdate:
    """Tests for ContextualPosterior.update() method."""

    def test_update_affects_global(self):
        cp = ContextualPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32) * 50.0

        cp.update(obs, {"time": "night"}, weight=1.0)

        assert cp.global_posterior.observation_count == 1

    def test_update_creates_context_posterior(self):
        cp = ContextualPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32)

        cp.update(obs, {"time": "night", "lux": "dim"}, weight=1.0)

        assert "time=night|lux=dim" in cp.posteriors or "lux=dim|time=night" in cp.posteriors

    def test_update_creates_partial_posteriors(self):
        cp = ContextualPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32)

        cp.update(obs, {"time": "night", "lux": "dim"}, weight=1.0)

        assert "time=night" in cp.posteriors
        assert "lux=dim" in cp.posteriors

    def test_ignores_unknown_values(self):
        cp = ContextualPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32)

        cp.update(obs, {"time": "night", "lux": "unknown"}, weight=1.0)

        assert "lux=unknown" not in cp.posteriors


class TestContextualPosteriorGetPosterior:
    """Tests for ContextualPosterior.get_posterior() method."""

    def test_returns_exact_match(self):
        cp = ContextualPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32) * 50.0

        cp.update(obs, {"time": "night"}, weight=1.0)

        posterior = cp.get_posterior({"time": "night"})

        # Should return the context-specific posterior
        assert np.all(posterior.mean > 0)

    def test_returns_partial_match(self):
        cp = ContextualPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32) * 50.0

        cp.update(obs, {"time": "night", "lux": "dim"}, weight=1.0)

        # Query with only one factor
        posterior = cp.get_posterior({"time": "night"})

        assert posterior.observation_count > 0

    def test_falls_back_to_global(self):
        cp = ContextualPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32) * 50.0

        cp.update(obs, {"time": "night"}, weight=1.0)

        # Query with unrelated context
        posterior = cp.get_posterior({"weather": "rain"})

        # Should fall back to global
        assert posterior.observation_count == 1


class TestContextualPosteriorConfidence:
    """Tests for context-specific confidence."""

    def test_context_confidence_varies(self):
        cp = ContextualPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32)

        # Add more observations for night
        for _ in range(10):
            cp.update(obs, {"time": "night"}, weight=1.0)
        cp.update(obs, {"time": "morning"}, weight=1.0)

        night_conf = cp.context_confidence({"time": "night"})
        morning_conf = cp.context_confidence({"time": "morning"})

        # Night should have higher confidence
        assert night_conf > morning_conf

    def test_least_confident_contexts(self):
        cp = ContextualPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32)

        # Add varying amounts of data
        for _ in range(10):
            cp.update(obs, {"time": "night"}, weight=1.0)
        for _ in range(2):
            cp.update(obs, {"time": "morning"}, weight=1.0)

        least = cp.least_confident_contexts(top_k=5)

        assert isinstance(least, list)
        assert len(least) <= 5
        # Each item should be (context_key, confidence)
        for key, conf in least:
            assert isinstance(key, str)
            assert 0 <= conf <= 1


class TestContextualPosteriorSerialization:
    """Tests for ContextualPosterior serialization."""

    def test_to_dict_structure(self):
        cp = ContextualPosterior()
        obs = np.ones(EMBEDDING_DIM, dtype=np.float32)
        cp.update(obs, {"time": "night"}, weight=1.0)

        data = cp.to_dict()

        assert "global" in data
        assert "contexts" in data

    def test_serialization_roundtrip(self):
        cp = ContextualPosterior()
        obs = np.random.randn(EMBEDDING_DIM).astype(np.float32) * 10

        cp.update(obs, {"time": "night"}, weight=1.0)
        cp.update(obs, {"time": "morning"}, weight=0.5)

        data = cp.to_dict()
        restored = ContextualPosterior.from_dict(data)

        # Check global restored
        assert np.allclose(
            restored.global_posterior.mean,
            cp.global_posterior.mean
        )

        # Check contexts restored
        assert len(restored.posteriors) == len(cp.posteriors)
