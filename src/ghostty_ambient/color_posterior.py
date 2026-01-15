"""
Bayesian color preference learning.

Models user's ideal color as a distribution over LAB color space.
Each chosen theme is an observation that updates the posterior.

Based on Bayes' billiard ball principle: each observation constrains
where the true ideal exists, building a region of confidence.
"""

from __future__ import annotations

import numpy as np
from typing import Any


class ColorPosterior:
    """Gaussian posterior over LAB color space for a single context."""

    def __init__(self):
        self.observations: list[np.ndarray] = []
        # Weak prior centered at neutral gray (L=50, a=0, b=0)
        self.mean: np.ndarray = np.array([50.0, 0.0, 0.0])
        self.cov: np.ndarray = np.eye(3) * 500  # High variance = low confidence

    def add_observation(self, lab: tuple[float, float, float]) -> None:
        """Add a new color observation and update the posterior."""
        self.observations.append(np.array(lab))
        self._update()

    def _update(self) -> None:
        """Recompute posterior parameters from observations."""
        if len(self.observations) < 2:
            if self.observations:
                self.mean = self.observations[0].copy()
            return

        obs = np.array(self.observations)
        self.mean = np.mean(obs, axis=0)
        self.cov = np.cov(obs, rowvar=False)
        # Regularize to prevent singular covariance matrix
        self.cov += 1e-4 * np.eye(3)

    def score(self, lab: tuple[float, float, float]) -> float:
        """
        Score a color by posterior probability.

        Uses Mahalanobis distance (accounts for covariance shape).
        Returns 0-100 score where higher = closer to ideal.
        """
        diff = np.array(lab) - self.mean
        try:
            # Mahalanobis distance: accounts for correlation between L, a, b
            mahal = np.sqrt(diff @ np.linalg.inv(self.cov) @ diff)
        except np.linalg.LinAlgError:
            # Fallback to simple Euclidean if covariance is singular
            mahal = np.linalg.norm(diff) / 10

        # Convert to score: distance 0 → 100, distance 3+ → ~0
        return float(100 * np.exp(-0.5 * mahal))

    @property
    def confidence(self) -> float:
        """
        Confidence in the posterior estimate.

        Based on observation count, saturating at 10 observations.
        Returns 0-1 where higher = more confident.
        """
        return min(1.0, len(self.observations) / 10)

    @property
    def count(self) -> int:
        """Number of observations."""
        return len(self.observations)


class ColorPreferenceModel:
    """
    Manages color posteriors per context factor.

    Each factor:bucket combination (e.g., "time:night", "system:dark")
    has its own posterior distribution. When scoring, we combine
    posteriors from all relevant factors using precision weighting.
    """

    def __init__(self, data: dict[str, Any] | None = None):
        self.posteriors: dict[str, ColorPosterior] = {}
        if data:
            self._load(data)

    def record(
        self, lab: tuple[float, float, float], factors: dict[str, str]
    ) -> None:
        """
        Record a color observation for all active factors.

        Args:
            lab: LAB color of chosen theme (L, a, b)
            factors: Dict of factor:bucket pairs (e.g., {"time": "night"})
        """
        for factor, bucket in factors.items():
            if bucket is None or bucket == "unknown":
                continue
            key = f"{factor}:{bucket}"
            if key not in self.posteriors:
                self.posteriors[key] = ColorPosterior()
            self.posteriors[key].add_observation(lab)

    def predict_ideal(
        self, factors: dict[str, str]
    ) -> tuple[np.ndarray, float]:
        """
        Predict ideal color for given context.

        Combines posteriors from all active factors using precision-weighted
        averaging (more confident posteriors have more influence).

        Args:
            factors: Dict of factor:bucket pairs for current context

        Returns:
            (mean_LAB, confidence) tuple
        """
        relevant: list[tuple[np.ndarray, np.ndarray, float]] = []

        for factor, bucket in factors.items():
            if bucket is None or bucket == "unknown":
                continue
            key = f"{factor}:{bucket}"
            if key in self.posteriors:
                p = self.posteriors[key]
                if p.count > 0:
                    relevant.append((p.mean, p.cov, p.confidence))

        if not relevant:
            # No data - return neutral prior with zero confidence
            return np.array([50.0, 0.0, 0.0]), 0.0

        # Precision-weighted combination of Gaussians
        # More confident posteriors (lower covariance) get more weight
        total_weight = sum(c for _, _, c in relevant)
        if total_weight == 0:
            return np.array([50.0, 0.0, 0.0]), 0.0

        # Simple weighted average of means (weighted by confidence)
        combined_mean = sum(c * m for m, _, c in relevant) / total_weight
        combined_confidence = total_weight / len(relevant)

        return combined_mean, float(combined_confidence)

    def score_theme(
        self, theme_lab: tuple[float, float, float], factors: dict[str, str]
    ) -> float:
        """
        Score a theme by distance to predicted ideal.

        Args:
            theme_lab: LAB color of theme to score
            factors: Current context factors

        Returns:
            Score 0-100 where higher = closer to ideal
        """
        ideal, confidence = self.predict_ideal(factors)
        if confidence == 0:
            return 50.0  # No data, return neutral score

        # Euclidean distance in LAB space (Delta E)
        dist = float(np.linalg.norm(np.array(theme_lab) - ideal))

        # Convert distance to score
        # Delta E of 0 → 100, Delta E of 50 → ~37, Delta E of 100 → ~14
        raw_score = 100 * np.exp(-dist / 50)

        # Blend with neutral (50) based on confidence
        # Low confidence = closer to neutral, high confidence = use learned score
        return float(confidence * raw_score + (1 - confidence) * 50.0)

    def get_ideal_color(
        self, factors: dict[str, str]
    ) -> tuple[tuple[float, float, float], float] | None:
        """
        Get the predicted ideal LAB color for generating new themes.

        Returns None if not enough data for a confident prediction.

        Args:
            factors: Current context factors

        Returns:
            ((L, a, b), confidence) or None if insufficient data
        """
        ideal, confidence = self.predict_ideal(factors)
        if confidence < 0.3:  # Require at least 30% confidence
            return None
        return (float(ideal[0]), float(ideal[1]), float(ideal[2])), confidence

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            key: {
                "observations": [obs.tolist() for obs in p.observations],
                "mean": p.mean.tolist(),
                "cov": p.cov.tolist(),
            }
            for key, p in self.posteriors.items()
        }

    def _load(self, data: dict[str, Any]) -> None:
        """Load from JSON data."""
        for key, pdata in data.items():
            p = ColorPosterior()
            p.observations = [
                np.array(obs) for obs in pdata.get("observations", [])
            ]
            if "mean" in pdata:
                p.mean = np.array(pdata["mean"])
            if "cov" in pdata:
                p.cov = np.array(pdata["cov"])
            self.posteriors[key] = p

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the learned preferences."""
        return {
            "factor_count": len(self.posteriors),
            "total_observations": sum(
                p.count for p in self.posteriors.values()
            ),
            "factors": {
                key: {"count": p.count, "confidence": p.confidence}
                for key, p in self.posteriors.items()
            },
        }


class ScalarPosterior:
    """Gaussian posterior over a 1D scalar (contrast, chroma, etc.)."""

    def __init__(self, prior_mean: float = 50.0, prior_std: float = 30.0):
        self.observations: list[float] = []
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.mean: float = prior_mean
        self.std: float = prior_std

    def add_observation(self, value: float) -> None:
        """Add a new scalar observation and update the posterior."""
        self.observations.append(value)
        self._update()

    def _update(self) -> None:
        """Recompute posterior parameters from observations."""
        if not self.observations:
            self.mean = self.prior_mean
            self.std = self.prior_std
            return
        self.mean = float(np.mean(self.observations))
        if len(self.observations) > 1:
            self.std = float(np.std(self.observations)) + 0.1  # Regularize
        else:
            self.std = self.prior_std * 0.5  # Reduce uncertainty with first obs

    @property
    def confidence(self) -> float:
        """Confidence based on observation count, saturating at 10."""
        return min(1.0, len(self.observations) / 10)

    @property
    def count(self) -> int:
        """Number of observations."""
        return len(self.observations)


class ThemePreferenceModel:
    """
    Manages color + theme property posteriors per context.

    Extends ColorPreferenceModel to also track:
    - Contrast (Delta E between background and foreground)
    - Chroma (saturation of palette colors)
    """

    def __init__(self, data: dict[str, Any] | None = None):
        self.color_posteriors: dict[str, ColorPosterior] = {}
        self.contrast_posteriors: dict[str, ScalarPosterior] = {}
        self.chroma_posteriors: dict[str, ScalarPosterior] = {}
        if data:
            self._load(data)

    def record_theme(
        self,
        bg_lab: tuple[float, float, float],
        fg_lab: tuple[float, float, float] | None,
        palette_chromas: list[float] | None,
        factors: dict[str, str],
    ) -> None:
        """
        Record full theme observation for all active factors.

        Args:
            bg_lab: Background LAB color
            fg_lab: Foreground LAB color (optional)
            palette_chromas: List of palette color chromas (optional)
            factors: Dict of factor:bucket pairs
        """
        # Calculate contrast if we have foreground
        contrast = None
        if fg_lab is not None:
            contrast = float(
                np.linalg.norm(np.array(bg_lab) - np.array(fg_lab))
            )

        # Calculate average chroma if we have palette
        avg_chroma = None
        if palette_chromas:
            avg_chroma = float(np.mean(palette_chromas))

        for factor, bucket in factors.items():
            if bucket is None or bucket == "unknown":
                continue
            key = f"{factor}:{bucket}"

            # Color posterior (background LAB)
            if key not in self.color_posteriors:
                self.color_posteriors[key] = ColorPosterior()
            self.color_posteriors[key].add_observation(bg_lab)

            # Contrast posterior
            if contrast is not None:
                if key not in self.contrast_posteriors:
                    self.contrast_posteriors[key] = ScalarPosterior(
                        prior_mean=80.0, prior_std=20.0
                    )
                self.contrast_posteriors[key].add_observation(contrast)

            # Chroma posterior
            if avg_chroma is not None:
                if key not in self.chroma_posteriors:
                    self.chroma_posteriors[key] = ScalarPosterior(
                        prior_mean=55.0, prior_std=15.0
                    )
                self.chroma_posteriors[key].add_observation(avg_chroma)

    def record(
        self, lab: tuple[float, float, float], factors: dict[str, str]
    ) -> None:
        """
        Record a background color observation (backwards compatible).

        For full theme recording, use record_theme() instead.
        """
        self.record_theme(bg_lab=lab, fg_lab=None, palette_chromas=None, factors=factors)

    def predict_ideal(
        self, factors: dict[str, str]
    ) -> tuple[np.ndarray, float]:
        """Predict ideal background color for given context."""
        relevant: list[tuple[np.ndarray, np.ndarray, float]] = []

        for factor, bucket in factors.items():
            if bucket is None or bucket == "unknown":
                continue
            key = f"{factor}:{bucket}"
            if key in self.color_posteriors:
                p = self.color_posteriors[key]
                if p.count > 0:
                    relevant.append((p.mean, p.cov, p.confidence))

        if not relevant:
            return np.array([50.0, 0.0, 0.0]), 0.0

        total_weight = sum(c for _, _, c in relevant)
        if total_weight == 0:
            return np.array([50.0, 0.0, 0.0]), 0.0

        combined_mean = sum(c * m for m, _, c in relevant) / total_weight
        combined_confidence = total_weight / len(relevant)

        return combined_mean, float(combined_confidence)

    def predict_contrast(self, factors: dict[str, str]) -> float:
        """Predict ideal contrast (Delta E) for context."""
        return self._predict_scalar(
            self.contrast_posteriors, factors, default=80.0
        )

    def predict_chroma(self, factors: dict[str, str]) -> float:
        """Predict ideal palette chroma for context."""
        return self._predict_scalar(
            self.chroma_posteriors, factors, default=55.0
        )

    def _predict_scalar(
        self,
        posteriors: dict[str, ScalarPosterior],
        factors: dict[str, str],
        default: float,
    ) -> float:
        """Predict a scalar property by combining relevant posteriors."""
        relevant: list[tuple[float, float]] = []
        for factor, bucket in factors.items():
            if bucket is None or bucket == "unknown":
                continue
            key = f"{factor}:{bucket}"
            if key in posteriors and posteriors[key].count > 0:
                p = posteriors[key]
                relevant.append((p.mean, p.confidence))

        if not relevant:
            return default

        total_weight = sum(c for _, c in relevant)
        if total_weight == 0:
            return default

        return sum(m * c for m, c in relevant) / total_weight

    def score_theme(
        self, theme_lab: tuple[float, float, float], factors: dict[str, str]
    ) -> float:
        """Score a theme by distance to predicted ideal background."""
        ideal, confidence = self.predict_ideal(factors)
        if confidence == 0:
            return 50.0

        dist = float(np.linalg.norm(np.array(theme_lab) - ideal))
        raw_score = 100 * np.exp(-dist / 50)
        return float(confidence * raw_score + (1 - confidence) * 50.0)

    def get_ideal_color(
        self, factors: dict[str, str]
    ) -> tuple[tuple[float, float, float], float] | None:
        """Get predicted ideal LAB color if confident enough."""
        ideal, confidence = self.predict_ideal(factors)
        if confidence < 0.3:
            return None
        return (float(ideal[0]), float(ideal[1]), float(ideal[2])), confidence

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        result: dict[str, Any] = {}

        # Color posteriors
        for key, p in self.color_posteriors.items():
            if key not in result:
                result[key] = {}
            result[key]["color"] = {
                "observations": [obs.tolist() for obs in p.observations],
                "mean": p.mean.tolist(),
                "cov": p.cov.tolist(),
            }

        # Contrast posteriors
        for key, p in self.contrast_posteriors.items():
            if key not in result:
                result[key] = {}
            result[key]["contrast"] = {
                "observations": p.observations,
                "mean": p.mean,
                "std": p.std,
            }

        # Chroma posteriors
        for key, p in self.chroma_posteriors.items():
            if key not in result:
                result[key] = {}
            result[key]["chroma"] = {
                "observations": p.observations,
                "mean": p.mean,
                "std": p.std,
            }

        return result

    def _load(self, data: dict[str, Any]) -> None:
        """Load from JSON data."""
        for key, pdata in data.items():
            # Load color posterior
            if "color" in pdata:
                p = ColorPosterior()
                cdata = pdata["color"]
                p.observations = [
                    np.array(obs) for obs in cdata.get("observations", [])
                ]
                if "mean" in cdata:
                    p.mean = np.array(cdata["mean"])
                if "cov" in cdata:
                    p.cov = np.array(cdata["cov"])
                self.color_posteriors[key] = p
            # Legacy format: direct observations without "color" key
            elif "observations" in pdata and "mean" in pdata:
                p = ColorPosterior()
                p.observations = [
                    np.array(obs) for obs in pdata.get("observations", [])
                ]
                if "mean" in pdata:
                    p.mean = np.array(pdata["mean"])
                if "cov" in pdata:
                    p.cov = np.array(pdata["cov"])
                self.color_posteriors[key] = p

            # Load contrast posterior
            if "contrast" in pdata:
                p = ScalarPosterior(prior_mean=80.0, prior_std=20.0)
                cdata = pdata["contrast"]
                p.observations = cdata.get("observations", [])
                if "mean" in cdata:
                    p.mean = cdata["mean"]
                if "std" in cdata:
                    p.std = cdata["std"]
                self.contrast_posteriors[key] = p

            # Load chroma posterior
            if "chroma" in pdata:
                p = ScalarPosterior(prior_mean=55.0, prior_std=15.0)
                cdata = pdata["chroma"]
                p.observations = cdata.get("observations", [])
                if "mean" in cdata:
                    p.mean = cdata["mean"]
                if "std" in cdata:
                    p.std = cdata["std"]
                self.chroma_posteriors[key] = p

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the learned preferences."""
        return {
            "color_factors": len(self.color_posteriors),
            "contrast_factors": len(self.contrast_posteriors),
            "chroma_factors": len(self.chroma_posteriors),
            "total_color_obs": sum(
                p.count for p in self.color_posteriors.values()
            ),
            "total_contrast_obs": sum(
                p.count for p in self.contrast_posteriors.values()
            ),
            "total_chroma_obs": sum(
                p.count for p in self.chroma_posteriors.values()
            ),
        }
