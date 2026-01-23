"""
Timestamped observation store for preference learning.

Stores theme choices with their context and computes features
for phase detection (variance, model distance, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import EMBEDDING_DIM, EMBEDDING_SCALE


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
    model_usage_rate: float  # Fraction of choices using model (ideal + daemon)
    manual_rate: float  # Fraction of manual (--set) choices
    theme_entropy: float  # Shannon entropy of theme selection
    effective_theme_count: float  # exp(entropy), effective number of themes
    unique_themes: int  # Number of unique themes in window
    observation_count: int  # Total observations in window
    effective_weight: float  # Sum of recency weights (>= observation_count if unweighted)


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
        return [o for o in self.observations if (now - o.timestamp).total_seconds() <= cutoff]

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
            matches = sum(1 for k, v in context.items() if obs.context.get(k) == v)
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
                model_usage_rate=0.0,
                manual_rate=0.0,
                theme_entropy=0.0,
                effective_theme_count=float(len(recent)),
                unique_themes=len(recent),
                observation_count=len(recent),
                effective_weight=float(len(recent)),
            )

        # Embedding variance
        embeddings = np.array([o.embedding for o in recent])
        embedding_mean = np.mean(embeddings, axis=0)

        # Standardize embeddings to avoid scale dominance by LAB dimensions.
        scale = np.where(EMBEDDING_SCALE > 0, EMBEDDING_SCALE, 1.0)
        standardized = embeddings / scale
        standardized_mean = np.mean(standardized, axis=0)

        # Per-dimension variance in standardized space.
        embedding_variance = float(np.mean(np.var(standardized, axis=0)))

        # Model distance: distance from running mean in standardized space.
        if self._running_mean is not None:
            running_mean = self._running_mean / scale
            distances = np.linalg.norm(standardized - running_mean, axis=1)
            model_distance = float(np.mean(distances))
        else:
            distances = np.linalg.norm(standardized - standardized_mean, axis=1)
            model_distance = float(np.mean(distances))

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
        model_usage_rate = sum(1 for s in sources if s in {"ideal", "daemon"}) / len(sources)
        manual_rate = sources.count("manual") / len(sources)

        # Theme entropy and effective theme count
        theme_counts: dict[str, int] = {}
        for o in recent:
            theme_counts[o.theme_name] = theme_counts.get(o.theme_name, 0) + 1
        probs = np.array(list(theme_counts.values()), dtype=np.float32) / len(recent)
        theme_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        effective_theme_count = float(np.exp(theme_entropy))

        # Unique themes
        unique_themes = len({o.theme_name for o in recent})

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
            observation_count=len(recent),
            effective_weight=float(len(recent)),
        )

    def compute_decayed_features(
        self,
        half_life_days: float,
        window_days: float | None = None,
    ) -> ObservationFeatures:
        """
        Compute exponentially time-decayed features for phase detection.

        Args:
            half_life_days: Half-life for exponential decay.
            window_days: Optional time window limit.

        Returns:
            ObservationFeatures with decayed statistics.
        """
        obs = self.in_window(window_days) if window_days is not None else self.observations
        if not obs:
            return ObservationFeatures(
                embedding_variance=0.0,
                embedding_mean=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                model_distance=0.0,
                choice_frequency=0.0,
                ideal_usage_rate=0.0,
                model_usage_rate=0.0,
                manual_rate=0.0,
                theme_entropy=0.0,
                effective_theme_count=0.0,
                unique_themes=0,
                observation_count=0,
                effective_weight=0.0,
            )

        now = datetime.now()
        obs_sorted = sorted(obs, key=lambda o: o.timestamp)
        weights = []
        embeddings = []
        sources = []
        themes = []

        for o in obs_sorted:
            if half_life_days <= 0:
                weight = 1.0
            else:
                age = o.age_days(now)
                weight = np.exp(-age * np.log(2) / half_life_days)
            weights.append(weight)
            embeddings.append(o.embedding)
            sources.append(o.source)
            themes.append(o.theme_name)

        weights_arr = np.array(weights, dtype=np.float32)
        embeddings_arr = np.array(embeddings)
        total_weight = float(np.sum(weights_arr))
        if total_weight <= 0:
            total_weight = 0.0

        scale = np.where(EMBEDDING_SCALE > 0, EMBEDDING_SCALE, 1.0)
        standardized = embeddings_arr / scale

        if total_weight > 0:
            weighted_mean = np.average(standardized, weights=weights_arr, axis=0)
            diff = standardized - weighted_mean
            embedding_variance = float(np.average(diff**2, weights=weights_arr, axis=0).mean())
            distances = np.linalg.norm(diff, axis=1)
            model_distance = float(np.average(distances, weights=weights_arr))
            embedding_mean = np.average(embeddings_arr, weights=weights_arr, axis=0)
        else:
            embedding_variance = 0.0
            model_distance = 0.0
            embedding_mean = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Choice frequency (choices per day)
        if len(obs_sorted) >= 2:
            time_span = (obs_sorted[-1].timestamp - obs_sorted[0].timestamp).total_seconds()
            if time_span > 0:
                choice_frequency = len(obs_sorted) / (time_span / 86400)
            else:
                choice_frequency = 0.0
        else:
            choice_frequency = 0.0

        # Weighted source distribution
        if total_weight > 0:
            ideal_weight = sum(w for w, s in zip(weights_arr, sources, strict=False) if s == "ideal")
            model_weight = sum(w for w, s in zip(weights_arr, sources, strict=False) if s in {"ideal", "daemon"})
            manual_weight = sum(w for w, s in zip(weights_arr, sources, strict=False) if s == "manual")
            ideal_usage_rate = float(ideal_weight / total_weight)
            model_usage_rate = float(model_weight / total_weight)
            manual_rate = float(manual_weight / total_weight)
        else:
            ideal_usage_rate = 0.0
            model_usage_rate = 0.0
            manual_rate = 0.0

        # Weighted theme entropy/effective count
        theme_weights: dict[str, float] = {}
        for w, name in zip(weights_arr, themes, strict=False):
            theme_weights[name] = theme_weights.get(name, 0.0) + float(w)
        if total_weight > 0:
            probs = np.array(list(theme_weights.values()), dtype=np.float32) / total_weight
            theme_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            effective_theme_count = float(np.exp(theme_entropy))
        else:
            theme_entropy = 0.0
            effective_theme_count = 0.0

        unique_themes = len(set(themes))

        return ObservationFeatures(
            embedding_variance=embedding_variance,
            embedding_mean=embedding_mean.astype(np.float32),
            model_distance=model_distance,
            choice_frequency=choice_frequency,
            ideal_usage_rate=ideal_usage_rate,
            model_usage_rate=model_usage_rate,
            manual_rate=manual_rate,
            theme_entropy=theme_entropy,
            effective_theme_count=effective_theme_count,
            unique_themes=unique_themes,
            observation_count=len(obs_sorted),
            effective_weight=total_weight,
        )

    def compute_feature_percentiles(
        self,
        window_size: int = 50,
        step: int = 10,
        percentiles: list[int] | None = None,
    ) -> dict[str, dict[int, float]]:
        """
        Compute percentiles for feature values over sliding windows.

        Useful for calibrating HMM normalization scales.
        """
        percentiles = percentiles or [10, 50, 90, 95]
        observations = self.observations
        if not observations:
            return {}
        if len(observations) < window_size:
            window_size = len(observations)
        values: dict[str, list[float]] = {
            "embedding_variance": [],
            "model_distance": [],
            "effective_theme_count": [],
        }

        for start in range(0, len(observations) - window_size + 1, step):
            window = observations[start : start + window_size]
            temp = ObservationStore()
            temp.observations = window
            temp._recompute_running_mean()
            features = temp.compute_features(window_size=len(window))
            values["embedding_variance"].append(features.embedding_variance)
            values["model_distance"].append(features.model_distance)
            values["effective_theme_count"].append(features.effective_theme_count)

        percentiles_out: dict[str, dict[int, float]] = {}
        for key, vals in values.items():
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float32)
            percentiles_out[key] = {p: float(np.percentile(arr, p)) for p in percentiles}

        return percentiles_out

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
            weighted_var = np.average(diff**2, weights=weights_arr, axis=0)

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
        store.observations = [Observation.from_dict(o) for o in data.get("observations", [])]
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
