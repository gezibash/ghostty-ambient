"""
Hidden Markov Model-based phase detection for preference learning.

Detects three learning phases:
- EXPLORE: User trying new themes, high variance
- CONVERGE: User narrowing down preferences, decreasing variance
- STABLE: User settled on preferences, low variance

Uses observation features (variance, model distance, etc.) to infer
the current phase and adjust learning parameters accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .observations import ObservationFeatures, ObservationStore


class Phase(Enum):
    """Learning phase states."""

    EXPLORE = "explore"
    CONVERGE = "converge"
    STABLE = "stable"


@dataclass
class PhaseConfig:
    """Configuration for each phase."""

    learning_rate: float  # How fast to update preferences
    recency_half_life: float  # Days for exponential decay half-life
    recommendation_diversity: float  # How diverse recommendations should be
    min_observations: int  # Min observations before transitioning out


class OnlineFeatureScaler:
    """Online scale estimator with clipping for robustness."""

    def __init__(
        self,
        n_features: int,
        clip_sigma: float = 4.0,
        min_scale: float = 1e-3,
        scale_half_life: float = 200.0,
        init_scale: np.ndarray | None = None,
    ):
        self.mean = np.zeros(n_features, dtype=np.float32)
        self.var = np.ones(n_features, dtype=np.float32)
        if init_scale is None:
            init_scale = np.ones(n_features, dtype=np.float32)
        self.scale_estimate = np.array(init_scale, dtype=np.float32)
        self.count = 0.0
        self.clip_sigma = clip_sigma
        self.min_scale = min_scale
        self.scale_half_life = scale_half_life

    def update(self, x: np.ndarray, weight: float = 1.0) -> None:
        if weight <= 0:
            return
        x = np.asarray(x, dtype=np.float32)
        if self.count > 0:
            std = np.sqrt(self.var)
            std = np.maximum(std, self.min_scale)
            scale = np.maximum(self.scale_estimate, self.min_scale)
            clip = self.clip_sigma * np.maximum(std, scale)
            x = np.clip(x, self.mean - clip, self.mean + clip)

        total = self.count + weight
        delta = x - self.mean
        mean = self.mean + (weight / total) * delta
        var = (self.count * self.var + weight * delta * (x - mean)) / total
        if self.scale_half_life > 0:
            alpha = 1.0 - 0.5 ** (weight / self.scale_half_life)
        else:
            alpha = 1.0
        self.scale_estimate = (1.0 - alpha) * self.scale_estimate + alpha * np.abs(x)

        self.mean = mean
        self.var = np.maximum(var, self.min_scale**2)
        self.count = total

    def scale(self) -> np.ndarray:
        return np.maximum(self.scale_estimate, self.min_scale)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) / self.scale()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "scale_estimate": self.scale_estimate.tolist(),
            "count": self.count,
            "clip_sigma": self.clip_sigma,
            "min_scale": self.min_scale,
            "scale_half_life": self.scale_half_life,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnlineFeatureScaler:
        n_features = len(data.get("mean", []))
        scaler = cls(
            n_features=n_features or 4,
            clip_sigma=float(data.get("clip_sigma", 4.0)),
            min_scale=float(data.get("min_scale", 1e-3)),
            scale_half_life=float(data.get("scale_half_life", 200.0)),
        )
        if "mean" in data:
            scaler.mean = np.array(data["mean"], dtype=np.float32)
        if "var" in data:
            scaler.var = np.array(data["var"], dtype=np.float32)
        if "scale_estimate" in data:
            scaler.scale_estimate = np.array(data["scale_estimate"], dtype=np.float32)
        if "count" in data:
            scaler.count = float(data["count"])
        return scaler


# Default configurations per phase
PHASE_CONFIGS = {
    Phase.EXPLORE: PhaseConfig(
        learning_rate=1.0,
        recency_half_life=3.0,
        recommendation_diversity=0.8,
        min_observations=5,
    ),
    Phase.CONVERGE: PhaseConfig(
        learning_rate=0.5,
        recency_half_life=7.0,
        recommendation_diversity=0.5,
        min_observations=10,
    ),
    Phase.STABLE: PhaseConfig(
        learning_rate=0.1,
        recency_half_life=30.0,
        recommendation_diversity=0.2,
        min_observations=20,
    ),
}


class PhaseDetector:
    """
    HMM-based phase detector for preference learning.

    Uses a Gaussian Hidden Markov Model to infer the current learning phase
    from observation features. The HMM is initialized with reasonable priors
    and can be updated as more data is collected.

    Observation features used:
    - embedding_variance: Higher in EXPLORE, lower in STABLE (scale-aware)
    - model_distance: Higher when exploring new themes (scale-aware)
    - model_usage_rate: Higher in STABLE (user relies on model/daemon)
    - effective_theme_count: Higher in EXPLORE (entropy-based)

    State transitions:
    - EXPLORE → CONVERGE: Variance decreasing, themes clustering
    - CONVERGE → STABLE: Using --ideal, consistent choices
    - STABLE → EXPLORE: Sudden shift away from pattern
    """

    # Number of features used for phase detection
    N_FEATURES = 4

    def __init__(self):
        # State indices
        self.states = [Phase.EXPLORE, Phase.CONVERGE, Phase.STABLE]
        self.n_states = len(self.states)

        # Initial state probabilities (start in EXPLORE)
        self.pi = np.array([0.7, 0.2, 0.1])

        # Transition matrix (rows = from, cols = to)
        # Designed for stickiness - phases don't change easily
        # EXPLORE: tends to stay, can move to CONVERGE
        # CONVERGE: sticky, moves to STABLE over time
        # STABLE: very sticky, requires sustained change to leave
        self.A = np.array(
            [
                [0.80, 0.18, 0.02],  # From EXPLORE: sticky, rarely jumps to STABLE
                [0.05, 0.75, 0.20],  # From CONVERGE: sticky, gradual to STABLE
                [0.02, 0.08, 0.90],  # From STABLE: very sticky, hard to leave
            ]
        )

        # Emission parameters (Gaussian per state per feature)
        # Features: [variance, model_distance, model_usage, effective_theme_count]
        # Each state has mean and std for each feature
        self.emission_means = np.array(
            [
                [0.75, 0.70, 0.15, 0.70],  # EXPLORE: high variance/distance, low model usage, many themes
                [0.60, 0.60, 0.35, 0.60],  # CONVERGE: mid/high variance, moderate model usage
                [0.20, 0.25, 0.80, 0.35],  # STABLE: low variance/distance, high model usage, few themes
            ]
        )

        self.emission_stds = np.array(
            [
                [0.25, 0.25, 0.2, 0.2],  # EXPLORE
                [0.2, 0.2, 0.2, 0.2],  # CONVERGE
                [0.2, 0.2, 0.25, 0.2],  # STABLE
            ]
        )

        # Online feature normalization (learned per feature).
        self.feature_scaler = OnlineFeatureScaler(
            self.N_FEATURES,
            init_scale=np.array([1.0, 8.0, 1.0, 8.0], dtype=np.float32),
        )

        # Normal-Inverse-Gamma priors for emission learning.
        self._init_emission_priors()

        # Dirichlet counts for adaptive transitions.
        self.transition_prior_strength = 2.0
        self.transition_counts = self.A * self.transition_prior_strength
        self.min_emission_std = 0.2
        self.belief_inertia = 0.5

        # History of state beliefs for Viterbi
        self._belief = self.pi.copy()
        self._feature_history: list[np.ndarray] = []

    def _init_emission_priors(self) -> None:
        """Initialize Normal-Inverse-Gamma priors from current emissions."""
        self.nig_mu = self.emission_means.copy()
        self.nig_kappa = np.full_like(self.emission_means, 0.5)
        self.nig_alpha = np.full_like(self.emission_means, 2.0)
        base_var = self.emission_stds**2
        self.nig_beta = base_var * (self.nig_alpha - 1.0)
        self.nig_prior_mu = self.nig_mu.copy()
        self.nig_prior_kappa = self.nig_kappa.copy()
        self.nig_prior_alpha = self.nig_alpha.copy()
        self.nig_prior_beta = self.nig_beta.copy()

        # Forgetting half-life in updates to prevent overconfidence.
        self.emission_half_life = 200.0

    def _apply_emission_decay(self, weight: float = 1.0) -> None:
        if self.emission_half_life <= 0:
            return
        decay = 0.5 ** (weight / self.emission_half_life)
        self.nig_mu = self.nig_prior_mu + decay * (self.nig_mu - self.nig_prior_mu)
        self.nig_kappa = self.nig_prior_kappa + decay * (self.nig_kappa - self.nig_prior_kappa)
        self.nig_alpha = self.nig_prior_alpha + decay * (self.nig_alpha - self.nig_prior_alpha)
        self.nig_beta = self.nig_prior_beta + decay * (self.nig_beta - self.nig_prior_beta)

    def _raw_features(self, features: ObservationFeatures) -> np.ndarray:
        """Extract raw feature vector from observations."""
        return np.array(
            [
                features.embedding_variance,
                features.model_distance,
                features.model_usage_rate,
                features.effective_theme_count,
            ],
            dtype=np.float32,
        )

    def _features_to_vector(self, features: ObservationFeatures) -> np.ndarray:
        """Convert observation features to normalized feature vector."""
        raw = self._raw_features(features)
        self.feature_scaler.update(raw, weight=1.0)
        return self.feature_scaler.normalize(raw)

    def _student_t_logpdf(
        self,
        x: float,
        mu: float,
        kappa: float,
        alpha: float,
        beta: float,
    ) -> float:
        """Log PDF of Student-t predictive from NIG parameters."""
        kappa = max(kappa, 1e-6)
        alpha = max(alpha, 1e-6)
        beta = max(beta, 1e-6)
        df = 2.0 * alpha
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)
        scale2 = max(scale2, self.min_emission_std**2)
        log_norm = math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0)
        log_norm -= 0.5 * math.log(df * math.pi * scale2)
        log_kernel = -((df + 1.0) / 2.0) * math.log(1.0 + ((x - mu) ** 2) / (df * scale2))
        return log_norm + log_kernel

    def _nig_update(
        self,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        x: np.ndarray,
        weight: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Update NIG parameters with a weighted observation."""
        if weight <= 0:
            return mu, kappa, alpha, beta
        kappa_n = kappa + weight
        delta = x - mu
        mu_n = mu + (weight / kappa_n) * delta
        alpha_n = alpha + 0.5 * weight
        beta_n = beta + 0.5 * weight * (delta**2) * (kappa / kappa_n)
        return mu_n, kappa_n, alpha_n, beta_n

    def emission_log_likelihoods(self, feature_vec: np.ndarray) -> np.ndarray:
        """
        Per-state, per-feature log-likelihood contributions.

        Returns array shape (n_states, n_features).
        """
        out = np.zeros((self.n_states, self.N_FEATURES), dtype=np.float32)
        for state_idx in range(self.n_states):
            for f_idx in range(self.N_FEATURES):
                out[state_idx, f_idx] = self._student_t_logpdf(
                    feature_vec[f_idx],
                    float(self.nig_mu[state_idx, f_idx]),
                    float(self.nig_kappa[state_idx, f_idx]),
                    float(self.nig_alpha[state_idx, f_idx]),
                    float(self.nig_beta[state_idx, f_idx]),
                )
        return out

    def diagnose(self, features: ObservationFeatures) -> dict[str, Any]:
        """
        Compute diagnostics without mutating detector state.
        """
        raw_features = self._raw_features(features)
        feature_vec = self.feature_scaler.normalize(raw_features)
        per_feature = self.emission_log_likelihoods(feature_vec)
        total_log = per_feature.sum(axis=1)
        emissions = np.exp(total_log)
        predicted = self.A.T @ self._belief
        updated = predicted * emissions
        norm = np.sum(updated)
        if norm > 0:
            updated = updated / norm
        else:
            updated = self.pi.copy()
        return {
            "raw_features": raw_features,
            "feature_vector": feature_vec,
            "per_feature_log_likelihoods": per_feature,
            "state_log_likelihoods": total_log,
            "emissions": emissions,
            "predicted_belief": predicted,
            "updated_belief": updated,
            "feature_scale": self.feature_scaler.scale(),
        }

    def update(self, features: ObservationFeatures) -> Phase:
        """
        Update belief state with new observation and return current phase.

        Uses forward algorithm to update state probabilities.
        """
        feature_vec = self._features_to_vector(features)
        self._feature_history.append(feature_vec)

        # Keep only recent history
        if len(self._feature_history) > 100:
            self._feature_history = self._feature_history[-100:]

        # Forward step: P(state_t | observations_1:t)
        # new_belief[j] = sum_i(belief[i] * A[i,j]) * emission[j]

        # Transition
        predicted = self.A.T @ self._belief

        # Emission (posterior predictive)
        self._apply_emission_decay()
        per_feature = self.emission_log_likelihoods(feature_vec)
        log_emissions = per_feature.sum(axis=1)
        log_emissions -= np.max(log_emissions)
        emissions = np.exp(log_emissions)

        # Update
        new_belief = predicted * emissions
        norm = np.sum(new_belief)
        if norm > 0:
            new_belief /= norm
        else:
            new_belief = self.pi.copy()

        # Gate phases until sufficient evidence is collected.
        effective_weight = getattr(features, "effective_weight", features.observation_count)
        gated = new_belief.copy()
        for i, phase in enumerate(self.states):
            if phase is Phase.EXPLORE:
                continue
            if effective_weight < PHASE_CONFIGS[phase].min_observations:
                gated[i] = 0.0
        gated_norm = np.sum(gated)
        if gated_norm > 0:
            new_belief = gated / gated_norm

        inertia = self.belief_inertia if self.current_phase() == Phase.STABLE else 0.0
        if inertia > 0:
            new_belief = (1.0 - inertia) * new_belief + inertia * self._belief
            new_belief = new_belief / np.sum(new_belief)

        # Update transitions with expected transition counts.
        transition_weights = (self._belief[:, None] * self.A) * emissions[None, :]
        transition_norm = float(np.sum(transition_weights))
        if transition_norm > 0:
            transition_weights = transition_weights / transition_norm
            self.transition_counts += transition_weights
            row_sums = np.sum(self.transition_counts, axis=1, keepdims=True)
            self.A = np.where(row_sums > 0, self.transition_counts / row_sums, self.A)

        # Update emission parameters with responsibilities.
        for state_idx in range(self.n_states):
            weight = float(new_belief[state_idx])
            (
                self.nig_mu[state_idx],
                self.nig_kappa[state_idx],
                self.nig_alpha[state_idx],
                self.nig_beta[state_idx],
            ) = self._nig_update(
                self.nig_mu[state_idx],
                self.nig_kappa[state_idx],
                self.nig_alpha[state_idx],
                self.nig_beta[state_idx],
                feature_vec,
                weight,
            )

        self.emission_means = self.nig_mu.copy()
        variance_mean = np.maximum(self.nig_beta / np.maximum(self.nig_alpha - 1.0, 1e-6), 1e-6)
        self.emission_stds = np.sqrt(variance_mean)

        self._belief = new_belief

        return self.current_phase()

    def current_phase(self) -> Phase:
        """Return the most likely current phase."""
        return self.states[np.argmax(self._belief)]

    def phase_probabilities(self) -> dict[Phase, float]:
        """Return probability distribution over phases."""
        return {self.states[i]: float(self._belief[i]) for i in range(self.n_states)}

    def get_config(self) -> PhaseConfig:
        """Get configuration for current phase."""
        return PHASE_CONFIGS[self.current_phase()]

    def detect_from_store(self, store: ObservationStore, window_size: int = 50) -> Phase:
        """
        Detect phase from an observation store.

        Computes features and updates belief state.
        """
        config = self.get_config()
        features = store.compute_decayed_features(config.recency_half_life)
        return self.update(features)

    def reset(self) -> None:
        """Reset to initial state."""
        self._belief = self.pi.copy()
        self._feature_history = []
        self.feature_scaler = OnlineFeatureScaler(
            self.N_FEATURES,
            init_scale=np.array([1.0, 8.0, 1.0, 8.0], dtype=np.float32),
        )
        self._init_emission_priors()
        self.transition_counts = self.A * self.transition_prior_strength

    def fit(self, feature_sequences: list[list[np.ndarray]], n_iter: int = 10) -> None:
        """
        Fit HMM parameters using Baum-Welch algorithm.

        This is optional - the default parameters work reasonably well.
        Call this with labeled sequences if you have ground truth data.
        """
        # Simplified Baum-Welch (EM algorithm for HMM)
        # For now, we use fixed parameters based on domain knowledge
        # A full implementation would iterate:
        # 1. E-step: Compute expected state occupancies
        # 2. M-step: Update A, emission_means, emission_stds

        # TODO: Implement full Baum-Welch if needed
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "pi": self.pi.tolist(),
            "A": self.A.tolist(),
            "emission_means": self.emission_means.tolist(),
            "emission_stds": self.emission_stds.tolist(),
            "belief": self._belief.tolist(),
            "feature_history": [f.tolist() for f in self._feature_history[-50:]],
            "feature_scaler": self.feature_scaler.to_dict(),
            "transition_counts": self.transition_counts.tolist(),
            "nig_mu": self.nig_mu.tolist(),
            "nig_kappa": self.nig_kappa.tolist(),
            "nig_alpha": self.nig_alpha.tolist(),
            "nig_beta": self.nig_beta.tolist(),
            "nig_prior_mu": self.nig_prior_mu.tolist(),
            "nig_prior_kappa": self.nig_prior_kappa.tolist(),
            "nig_prior_alpha": self.nig_prior_alpha.tolist(),
            "nig_prior_beta": self.nig_prior_beta.tolist(),
            "emission_half_life": self.emission_half_life,
            "min_emission_std": self.min_emission_std,
            "belief_inertia": self.belief_inertia,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseDetector:
        """Deserialize from storage."""
        detector = cls()
        if "pi" in data:
            detector.pi = np.array(data["pi"])
        if "A" in data:
            detector.A = np.array(data["A"])
        if "emission_means" in data:
            detector.emission_means = np.array(data["emission_means"])
        if "emission_stds" in data:
            detector.emission_stds = np.array(data["emission_stds"])
        had_nig = False
        if "belief" in data:
            detector._belief = np.array(data["belief"])
        if "feature_history" in data:
            detector._feature_history = [np.array(f) for f in data["feature_history"]]
        if "feature_scaler" in data:
            detector.feature_scaler = OnlineFeatureScaler.from_dict(data["feature_scaler"])
        if "transition_counts" in data:
            detector.transition_counts = np.array(data["transition_counts"])
            row_sums = np.sum(detector.transition_counts, axis=1, keepdims=True)
            detector.A = np.where(row_sums > 0, detector.transition_counts / row_sums, detector.A)
        if "nig_mu" in data:
            detector.nig_mu = np.array(data["nig_mu"])
            had_nig = True
        if "nig_kappa" in data:
            detector.nig_kappa = np.array(data["nig_kappa"])
            had_nig = True
        if "nig_alpha" in data:
            detector.nig_alpha = np.array(data["nig_alpha"])
            had_nig = True
        if "nig_beta" in data:
            detector.nig_beta = np.array(data["nig_beta"])
            had_nig = True
        if "nig_prior_mu" in data:
            detector.nig_prior_mu = np.array(data["nig_prior_mu"])
        if "nig_prior_kappa" in data:
            detector.nig_prior_kappa = np.array(data["nig_prior_kappa"])
        if "nig_prior_alpha" in data:
            detector.nig_prior_alpha = np.array(data["nig_prior_alpha"])
        if "nig_prior_beta" in data:
            detector.nig_prior_beta = np.array(data["nig_prior_beta"])
        if "emission_half_life" in data:
            detector.emission_half_life = float(data["emission_half_life"])
        if "min_emission_std" in data:
            detector.min_emission_std = float(data["min_emission_std"])
        if "belief_inertia" in data:
            detector.belief_inertia = float(data["belief_inertia"])
        if not had_nig:
            detector._init_emission_priors()
        else:
            detector.emission_means = detector.nig_mu.copy()
            variance_mean = np.maximum(detector.nig_beta / np.maximum(detector.nig_alpha - 1.0, 1e-6), 1e-6)
            detector.emission_stds = np.sqrt(variance_mean)
            if not hasattr(detector, "nig_prior_mu"):
                detector.nig_prior_mu = detector.nig_mu.copy()
            if not hasattr(detector, "nig_prior_kappa"):
                detector.nig_prior_kappa = detector.nig_kappa.copy()
            if not hasattr(detector, "nig_prior_alpha"):
                detector.nig_prior_alpha = detector.nig_alpha.copy()
            if not hasattr(detector, "nig_prior_beta"):
                detector.nig_prior_beta = detector.nig_beta.copy()
        return detector


def heuristic_phase_detect(features: ObservationFeatures) -> Phase:
    """
    Simple heuristic-based phase detection as fallback.

    Useful when there's not enough data for HMM or as a sanity check.
    """
    # Not enough data - assume exploring
    if features.observation_count < 5:
        return Phase.EXPLORE

    # High model usage and low variance = stable
    if features.model_usage_rate > 0.6 and features.embedding_variance < 0.3 and features.effective_theme_count < 3.5:
        return Phase.STABLE

    # High variance or high theme entropy = exploring
    if features.embedding_variance > 0.8 or features.effective_theme_count > 5.0:
        return Phase.EXPLORE

    # Otherwise converging
    return Phase.CONVERGE
