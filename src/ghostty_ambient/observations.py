"""
Timestamped observation store for preference learning.

Stores theme choices with their context and computes features
for phase detection (variance, model distance, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import EMBEDDING_DIM


@dataclass
class Observation:
    """A single theme choice observation with context."""

    timestamp: datetime
    theme_name: str
    embedding: np.ndarray  # 20D theme embedding
    context: dict[str, str]  # factor -> bucket (e.g., {"time": "night", "lux": "dim"})
    source: str  # "picker", "ideal", "manual", "daemon"

    def age_days(self, now: datetime | None = None) -> float:
        """Days since this observation was recorded."""
        now = now or datetime.now()
        return (now - self.timestamp).total_seconds() / 86400

    def age_hours(self, now: datetime | None = None) -> float:
        """Hours since this observation was recorded."""
        now = now or datetime.now()
        return (now - self.timestamp).total_seconds() / 3600

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "theme_name": self.theme_name,
            "embedding": self.embedding.tolist(),
            "context": self.context,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Deserialize from JSON."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            theme_name=data["theme_name"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            context=data["context"],
            source=data["source"],
        )


@dataclass
class ObservationFeatures:
    """Features computed from recent observations for phase detection."""

    embedding_variance: float  # Variance across recent embeddings
    embedding_mean: np.ndarray  # Mean embedding
    model_distance: float  # How far recent choices are from running mean
    choice_frequency: float  # Choices per day
    ideal_usage_rate: float  # Fraction of choices using --ideal
    manual_rate: float  # Fraction of manual (--set) choices
    unique_themes: int  # Number of unique themes in window
    observation_count: int  # Total observations in window


class ObservationStore:
    """
    Store for timestamped observations with feature computation.

    Supports:
    - Adding new observations
    - Querying by recency or time window
    - Computing features for phase detection
    - Persistence to JSON
    """

    def __init__(self, max_observations: int = 5000):
        self.observations: list[Observation] = []
        self.max_observations = max_observations
        self._running_mean: np.ndarray | None = None
        self._running_mean_count: int = 0

    def add(self, obs: Observation) -> None:
        """Add an observation, maintaining max size."""
        self.observations.append(obs)

        # Update running mean incrementally
        if self._running_mean is None:
            self._running_mean = obs.embedding.copy()
            self._running_mean_count = 1
        else:
            self._running_mean_count += 1
            # Incremental mean update
            self._running_mean += (obs.embedding - self._running_mean) / self._running_mean_count

        # Trim if over limit
        if len(self.observations) > self.max_observations:
            # Remove oldest 10%
            trim_count = self.max_observations // 10
            self.observations = self.observations[trim_count:]
            # Recompute running mean after trim
            self._recompute_running_mean()

    def _recompute_running_mean(self) -> None:
        """Recompute running mean from all observations."""
        if not self.observations:
            self._running_mean = None
            self._running_mean_count = 0
        else:
            embeddings = np.array([o.embedding for o in self.observations])
            self._running_mean = np.mean(embeddings, axis=0)
            self._running_mean_count = len(self.observations)

    def recent(self, n: int) -> list[Observation]:
        """Get the n most recent observations."""
        return self.observations[-n:] if self.observations else []

    def in_window(self, days: float) -> list[Observation]:
        """Get observations from the last N days."""
        now = datetime.now()
        cutoff = days * 86400  # seconds
        return [
            o for o in self.observations
            if (now - o.timestamp).total_seconds() <= cutoff
        ]

    def for_context(self, context: dict[str, str], days: float = 30) -> list[Observation]:
        """
        Get observations matching a context within a time window.

        Matches observations where at least one context factor matches.
        """
        candidates = self.in_window(days)
        if not context:
            return candidates

        matching = []
        for obs in candidates:
            # Count matching factors
            matches = sum(
                1 for k, v in context.items()
                if obs.context.get(k) == v
            )
            if matches > 0:
                matching.append(obs)

        return matching

    def compute_features(self, window_size: int = 50) -> ObservationFeatures:
        """
        Compute features from recent observations for phase detection.

        Args:
            window_size: Number of recent observations to analyze

        Returns:
            ObservationFeatures with computed metrics
        """
        recent = self.recent(window_size)

        if len(recent) < 2:
            return ObservationFeatures(
                embedding_variance=0.0,
                embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                model_distance=0.0,
                choice_frequency=0.0,
                ideal_usage_rate=0.0,
                manual_rate=0.0,
                unique_themes=len(recent),
                observation_count=len(recent),
            )

        # Embedding variance
        embeddings = np.array([o.embedding for o in recent])
        embedding_mean = np.mean(embeddings, axis=0)

        # Total variance across all dimensions
        embedding_variance = float(np.var(embeddings))

        # Model distance: how far recent choices deviate from running mean
        if self._running_mean is not None:
            distances = np.linalg.norm(embeddings - self._running_mean, axis=1)
            model_distance = float(np.mean(distances))
        else:
            model_distance = 0.0

        # Choice frequency (choices per day)
        if len(recent) >= 2:
            time_span = (recent[-1].timestamp - recent[0].timestamp).total_seconds()
            if time_span > 0:
                choice_frequency = len(recent) / (time_span / 86400)
            else:
                choice_frequency = 0.0
        else:
            choice_frequency = 0.0

        # Source distribution
        sources = [o.source for o in recent]
        ideal_usage_rate = sources.count("ideal") / len(sources)
        manual_rate = sources.count("manual") / len(sources)

        # Unique themes
        unique_themes = len(set(o.theme_name for o in recent))

        return ObservationFeatures(
            embedding_variance=embedding_variance,
            embedding_mean=embedding_mean,
            model_distance=model_distance,
            choice_frequency=choice_frequency,
            ideal_usage_rate=ideal_usage_rate,
            manual_rate=manual_rate,
            unique_themes=unique_themes,
            observation_count=len(recent),
        )

    def compute_weighted_mean(
        self,
        half_life_days: float = 7.0,
        context: dict[str, str] | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Compute exponentially-weighted mean embedding.

        Args:
            half_life_days: Half-life for exponential decay
            context: Optional context filter

        Returns:
            (weighted_mean_embedding, total_weight)
        """
        if context:
            obs = self.for_context(context)
        else:
            obs = self.observations

        if not obs:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32), 0.0

        now = datetime.now()
        weights = []
        embeddings = []

        for o in obs:
            age = o.age_days(now)
            weight = np.exp(-age * np.log(2) / half_life_days)
            weights.append(weight)
            embeddings.append(o.embedding)

        weights = np.array(weights)
        embeddings = np.array(embeddings)

        total_weight = float(np.sum(weights))
        if total_weight > 0:
            weighted_mean = np.average(embeddings, weights=weights, axis=0)
        else:
            weighted_mean = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        return weighted_mean, total_weight

    def compute_weighted_stats(
        self,
        half_life_days: float = 7.0,
        context: dict[str, str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[float], float]:
        """
        Compute exponentially-weighted mean and variance for Bayesian posterior.

        Args:
            half_life_days: Half-life for exponential decay
            context: Optional context filter

        Returns:
            (weighted_mean, weighted_variance, weights_list, total_weight)
        """
        if context:
            obs = self.for_context(context)
        else:
            obs = self.observations

        if not obs:
            return (
                np.zeros(EMBEDDING_DIM, dtype=np.float32),
                np.ones(EMBEDDING_DIM, dtype=np.float32) * 100.0,  # High variance prior
                [],
                0.0,
            )

        now = datetime.now()
        weights = []
        embeddings = []

        for o in obs:
            age = o.age_days(now)
            weight = np.exp(-age * np.log(2) / half_life_days)
            weights.append(weight)
            embeddings.append(o.embedding)

        weights_arr = np.array(weights)
        embeddings_arr = np.array(embeddings)

        total_weight = float(np.sum(weights_arr))
        if total_weight > 0:
            # Weighted mean
            weighted_mean = np.average(embeddings_arr, weights=weights_arr, axis=0)

            # Weighted variance (using reliability weights formula)
            # Var = sum(w * (x - mean)^2) / sum(w)
            diff = embeddings_arr - weighted_mean
            weighted_var = np.average(diff ** 2, weights=weights_arr, axis=0)

            # Add small floor to prevent zero variance
            weighted_var = np.maximum(weighted_var, 0.01)
        else:
            weighted_mean = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            weighted_var = np.ones(EMBEDDING_DIM, dtype=np.float32) * 100.0

        return (
            weighted_mean.astype(np.float32),
            weighted_var.astype(np.float32),
            weights,
            total_weight,
        )

    def __len__(self) -> int:
        return len(self.observations)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "observations": [o.to_dict() for o in self.observations],
            "running_mean": self._running_mean.tolist() if self._running_mean is not None else None,
            "running_mean_count": self._running_mean_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], max_observations: int = 5000) -> ObservationStore:
        """Deserialize from JSON."""
        store = cls(max_observations=max_observations)
        store.observations = [
            Observation.from_dict(o) for o in data.get("observations", [])
        ]
        if data.get("running_mean") is not None:
            store._running_mean = np.array(data["running_mean"], dtype=np.float32)
            store._running_mean_count = data.get("running_mean_count", len(store.observations))
        else:
            store._recompute_running_mean()
        return store

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path, max_observations: int = 5000) -> ObservationStore:
        """Load from JSON file."""
        if not path.exists():
            return cls(max_observations=max_observations)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data, max_observations=max_observations)
