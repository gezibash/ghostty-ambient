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
    - embedding_variance: Higher in EXPLORE, lower in STABLE
    - model_distance: Higher when exploring new themes
    - ideal_usage_rate: Higher in STABLE (user trusts the model)
    - unique_themes: Higher in EXPLORE

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
        # Features: [variance, model_distance, ideal_usage, unique_ratio]
        # Each state has mean and std for each feature
        self.emission_means = np.array(
            [
                [0.8, 0.7, 0.1, 0.8],  # EXPLORE: high variance, high distance, low ideal, many unique
                [0.4, 0.4, 0.3, 0.5],  # CONVERGE: medium values
                [0.1, 0.2, 0.7, 0.2],  # STABLE: low variance, low distance, high ideal, few unique
            ]
        )

        self.emission_stds = np.array(
            [
                [0.3, 0.3, 0.2, 0.2],  # EXPLORE
                [0.2, 0.2, 0.2, 0.2],  # CONVERGE
                [0.2, 0.2, 0.3, 0.2],  # STABLE
            ]
        )

        # History of state beliefs for Viterbi
        self._belief = self.pi.copy()
        self._feature_history: list[np.ndarray] = []

    def _features_to_vector(self, features: ObservationFeatures) -> np.ndarray:
        """Convert observation features to normalized feature vector."""
        # Normalize features to roughly [0, 1] range
        variance_norm = min(features.embedding_variance / 10.0, 1.0)
        distance_norm = min(features.model_distance / 50.0, 1.0)
        ideal_rate = features.ideal_usage_rate
        unique_ratio = min(features.unique_themes / max(features.observation_count, 1), 1.0)

        return np.array([variance_norm, distance_norm, ideal_rate, unique_ratio])

    def _emission_prob(self, state_idx: int, features: np.ndarray) -> float:
        """Compute emission probability P(features | state)."""
        mean = self.emission_means[state_idx]
        std = self.emission_stds[state_idx]

        # Multivariate Gaussian (independent features)
        diff = features - mean
        log_prob = -0.5 * np.sum((diff / std) ** 2)
        log_prob -= np.sum(np.log(std))  # Normalization

        return np.exp(log_prob)

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

        # Emission
        emissions = np.array([self._emission_prob(i, feature_vec) for i in range(self.n_states)])

        # Update
        new_belief = predicted * emissions
        norm = np.sum(new_belief)
        if norm > 0:
            new_belief /= norm
        else:
            new_belief = self.pi.copy()

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
        features = store.compute_features(window_size)
        return self.update(features)

    def reset(self) -> None:
        """Reset to initial state."""
        self._belief = self.pi.copy()
        self._feature_history = []

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
        if "belief" in data:
            detector._belief = np.array(data["belief"])
        if "feature_history" in data:
            detector._feature_history = [np.array(f) for f in data["feature_history"]]
        return detector


def heuristic_phase_detect(features: ObservationFeatures) -> Phase:
    """
    Simple heuristic-based phase detection as fallback.

    Useful when there's not enough data for HMM or as a sanity check.
    """
    # Not enough data - assume exploring
    if features.observation_count < 5:
        return Phase.EXPLORE

    # High ideal usage and low variance = stable
    if features.ideal_usage_rate > 0.5 and features.embedding_variance < 0.2:
        return Phase.STABLE

    # High variance or many unique themes = exploring
    if features.embedding_variance > 0.5 or features.unique_themes > features.observation_count * 0.7:
        return Phase.EXPLORE

    # Otherwise converging
    return Phase.CONVERGE
