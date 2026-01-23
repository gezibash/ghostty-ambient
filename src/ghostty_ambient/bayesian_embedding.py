"""
Bayesian posterior over theme embeddings.

Maintains a Gaussian posterior over the "ideal" embedding vector,
providing both point estimates and uncertainty quantification.

The posterior enables:
- Confidence scores per context
- Thompson sampling for exploration
- Identifying under-explored contexts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .embeddings import EMBEDDING_DIM

# Prior parameters
DEFAULT_PRIOR_MEAN = np.zeros(EMBEDDING_DIM, dtype=np.float32)
DEFAULT_PRIOR_VAR = 100.0  # High initial uncertainty
# Observation noise per dimension (heuristic, tuned to feature scales).
OBSERVATION_STD = np.array(
    [
        10.0,
        15.0,
        15.0,  # bg L, a, b
        10.0,
        15.0,
        15.0,  # fg L, a, b
        15.0,  # contrast (Delta E)
        0.2,
        0.2,
        0.2,  # chroma, brightness, warmth (normalized)
        0.1,
        0.1,
        0.1,
        0.1,  # hue quadrants
        0.1,
        0.1,
        0.1,
        0.1,  # harmony scores
        0.2,  # color variety
        0.2,  # lightness range
    ],
    dtype=np.float32,
)
OBSERVATION_VAR = OBSERVATION_STD**2
MIN_DECAY_FACTOR = 1e-6


@dataclass
class EmbeddingPosterior:
    """
    Gaussian posterior over the ideal embedding.

    Tracks N(μ, Σ) where:
    - μ is the posterior mean (our best estimate of ideal)
    - Σ is diagonal covariance (uncertainty per dimension)

    Uses conjugate Gaussian updates with recency weighting.
    """

    mean: np.ndarray = field(default_factory=lambda: DEFAULT_PRIOR_MEAN.copy())
    variance: np.ndarray = field(default_factory=lambda: np.full(EMBEDDING_DIM, DEFAULT_PRIOR_VAR, dtype=np.float32))
    total_weight: float = 0.0
    observation_count: int = 0

    def update(
        self,
        embedding: np.ndarray,
        weight: float = 1.0,
    ) -> None:
        """
        Update posterior with a new observation.

        Uses weighted Bayesian update:
        - Higher weight = more influence on posterior
        - Posterior variance decreases with more observations

        Args:
            embedding: Observed embedding vector
            weight: Observation weight (e.g., recency weight)
        """
        if weight <= 0:
            return

        # Precision (inverse variance) update
        # posterior_precision = prior_precision + weight * obs_precision
        obs_precision = weight / OBSERVATION_VAR
        prior_precision = 1.0 / self.variance

        new_precision = prior_precision + obs_precision
        new_variance = 1.0 / new_precision

        # Mean update (precision-weighted average)
        # new_mean = (prior_precision * prior_mean + obs_precision * obs) / new_precision
        new_mean = (prior_precision * self.mean + obs_precision * embedding) / new_precision

        self.mean = new_mean.astype(np.float32)
        self.variance = new_variance.astype(np.float32)
        self.total_weight += weight
        self.observation_count += 1

    def update_batch(
        self,
        embeddings: list[np.ndarray],
        weights: list[float],
    ) -> None:
        """Update posterior with multiple observations."""
        for emb, w in zip(embeddings, weights, strict=True):
            self.update(emb, w)

    @property
    def confidence(self) -> float:
        """
        Scalar confidence score in [0, 1].

        Based on how much the posterior variance has shrunk
        from the prior. Higher = more confident.
        """
        # Average variance reduction across dimensions
        # Average per-dimension confidence based on observation noise scale
        conf = 1.0 - (self.variance / (self.variance + OBSERVATION_VAR))
        confidence = np.mean(conf)
        return float(np.clip(confidence, 0.0, 1.0))

    @property
    def uncertainty(self) -> float:
        """
        Scalar uncertainty (inverse of confidence).

        Useful for identifying under-explored contexts.
        """
        return 1.0 - self.confidence

    def dimension_confidence(self) -> np.ndarray:
        """
        Per-dimension confidence scores.

        Useful for understanding which aspects of preference
        are well-established vs uncertain.
        """
        # Confidence per dimension
        conf = 1.0 - (self.variance / (self.variance + OBSERVATION_VAR))
        return conf.astype(np.float32)

    def apply_decay(self, decay_factor: float) -> None:
        """
        Exponentially decay precision to forget older observations.

        Args:
            decay_factor: Multiplier in (0, 1], smaller = faster forgetting.
        """
        if decay_factor <= 0:
            return

        decay_factor = max(decay_factor, MIN_DECAY_FACTOR)
        # Decay precision => inflate variance, but do not exceed the prior.
        self.variance = np.minimum(self.variance / decay_factor, DEFAULT_PRIOR_VAR).astype(np.float32)
        self.total_weight *= decay_factor

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Sample from the posterior (Thompson sampling).

        Returns a plausible ideal embedding, useful for
        exploration in the EXPLORE phase.

        Args:
            rng: Random number generator (for reproducibility)

        Returns:
            Sampled embedding vector
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample from N(mean, diag(variance))
        std = np.sqrt(self.variance)
        sample = rng.normal(self.mean, std)
        return sample.astype(np.float32)

    def credible_interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute credible interval for the ideal embedding.

        Args:
            level: Credible level (e.g., 0.95 for 95% CI)

        Returns:
            (lower_bound, upper_bound) arrays
        """
        from scipy import stats

        z = stats.norm.ppf((1 + level) / 2)
        std = np.sqrt(self.variance)

        lower = self.mean - z * std
        upper = self.mean + z * std

        return lower.astype(np.float32), upper.astype(np.float32)

    def kl_divergence_from_prior(self) -> float:
        """
        KL divergence from posterior to prior.

        Measures how much we've learned. Higher = more learning.
        """
        # KL(posterior || prior) for diagonal Gaussians
        prior_var = DEFAULT_PRIOR_VAR
        kl = 0.5 * np.sum(
            self.variance / prior_var
            + (self.mean - DEFAULT_PRIOR_MEAN) ** 2 / prior_var
            - 1
            + np.log(prior_var / self.variance)
        )
        return float(kl)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "mean": self.mean.tolist(),
            "variance": self.variance.tolist(),
            "total_weight": self.total_weight,
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingPosterior:
        """Deserialize from storage."""
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            variance=np.array(data["variance"], dtype=np.float32),
            total_weight=data.get("total_weight", 0.0),
            observation_count=data.get("observation_count", 0),
        )

    @classmethod
    def from_observations(
        cls,
        embeddings: list[np.ndarray],
        weights: list[float] | None = None,
    ) -> EmbeddingPosterior:
        """
        Create posterior from a list of observations.

        Args:
            embeddings: List of embedding vectors
            weights: Optional weights (default: uniform)

        Returns:
            Updated posterior
        """
        posterior = cls()

        if not embeddings:
            return posterior

        if weights is None:
            weights = [1.0] * len(embeddings)

        posterior.update_batch(embeddings, weights)
        return posterior


class ContextualPosterior:
    """
    Maintains separate posteriors for different contexts.

    Allows different confidence levels for different situations
    (e.g., high confidence for night preferences, low for morning).
    """

    def __init__(self):
        self.posteriors: dict[str, EmbeddingPosterior] = {}
        self.global_posterior = EmbeddingPosterior()
        self.last_updated: datetime | None = None

    def _context_key(self, context: dict[str, str]) -> str:
        """Create a hashable key from context dict."""
        # Use sorted key-value pairs for consistency
        items = sorted(context.items())
        return "|".join(f"{k}={v}" for k, v in items if v != "unknown")

    def update(
        self,
        embedding: np.ndarray,
        context: dict[str, str],
        weight: float = 1.0,
        now: datetime | None = None,
    ) -> None:
        """
        Update posteriors with a new observation.

        Updates both the context-specific and global posteriors.
        """
        # Update global
        self.global_posterior.update(embedding, weight)

        # Update context-specific
        key = self._context_key(context)
        if key not in self.posteriors:
            self.posteriors[key] = EmbeddingPosterior()
        self.posteriors[key].update(embedding, weight)

        # Also update partial context keys (for generalization)
        # e.g., if context is {time: night, lux: dim}, also update
        # posteriors for {time: night} and {lux: dim}
        for k, v in context.items():
            if v != "unknown":
                partial_key = f"{k}={v}"
                if partial_key not in self.posteriors:
                    self.posteriors[partial_key] = EmbeddingPosterior()
                self.posteriors[partial_key].update(embedding, weight * 0.5)

        self.last_updated = now or datetime.now()

    def apply_recency_decay(self, half_life_days: float, now: datetime | None = None) -> None:
        """
        Apply exponential forgetting to all posteriors.

        Args:
            half_life_days: Half-life in days for recency decay.
            now: Timestamp to use as the decay reference.
        """
        now = now or datetime.now()
        if self.last_updated is None:
            self.last_updated = now
            return
        if half_life_days <= 0:
            self.last_updated = now
            return

        delta_days = (now - self.last_updated).total_seconds() / 86400
        if delta_days <= 0:
            self.last_updated = now
            return

        decay = float(np.exp(-delta_days * np.log(2) / half_life_days))
        decay = max(decay, MIN_DECAY_FACTOR)

        self.global_posterior.apply_decay(decay)
        for posterior in self.posteriors.values():
            posterior.apply_decay(decay)

        self.last_updated = now

    # Primary factors that most strongly influence theme preference
    # These are user-controlled or time-based, not ambient environmental
    PRIMARY_FACTORS = {"system", "time", "circadian", "lux"}

    def get_posterior(self, context: dict[str, str]) -> EmbeddingPosterior:
        """
        Get the best posterior for a context.

        Falls back to partial matches or global if exact match unavailable.
        Prioritizes primary factors (system, time, circadian, lux) over ambient ones.
        """
        key = self._context_key(context)

        # Exact match
        if key in self.posteriors:
            return self.posteriors[key]

        # Try to select the best partial match, prioritizing primary factors
        primary_posteriors = []
        secondary_posteriors = []

        for k, v in context.items():
            if v != "unknown":
                partial_key = f"{k}={v}"
                if partial_key in self.posteriors:
                    if k in self.PRIMARY_FACTORS:
                        primary_posteriors.append(self.posteriors[partial_key])
                    else:
                        secondary_posteriors.append(self.posteriors[partial_key])

        # Use primary factors if available, otherwise fall back to all
        if primary_posteriors:
            return self._select_most_confident(primary_posteriors)
        elif secondary_posteriors:
            return self._select_most_confident(secondary_posteriors)

        # Fall back to global
        return self.global_posterior

    def _select_most_confident(
        self,
        posteriors: list[EmbeddingPosterior],
    ) -> EmbeddingPosterior:
        """Select the most confident posterior to avoid double-counting evidence."""
        if not posteriors:
            return EmbeddingPosterior()
        return max(posteriors, key=lambda p: p.confidence)

    def context_confidence(self, context: dict[str, str]) -> float:
        """Get confidence for a specific context."""
        return self.get_posterior(context).confidence

    def least_confident_contexts(self, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Find contexts with lowest confidence.

        Useful for identifying gaps in learning.
        """
        items = [(key, p.confidence) for key, p in self.posteriors.items()]
        items.sort(key=lambda x: x[1])
        return items[:top_k]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "global": self.global_posterior.to_dict(),
            "contexts": {key: p.to_dict() for key, p in self.posteriors.items()},
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextualPosterior:
        """Deserialize from storage."""
        cp = cls()
        if "global" in data:
            cp.global_posterior = EmbeddingPosterior.from_dict(data["global"])
        if "contexts" in data:
            cp.posteriors = {key: EmbeddingPosterior.from_dict(p_data) for key, p_data in data["contexts"].items()}
        last_updated = data.get("last_updated")
        if last_updated:
            cp.last_updated = datetime.fromisoformat(last_updated)
        return cp
