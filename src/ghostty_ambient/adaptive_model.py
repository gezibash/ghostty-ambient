"""
Adaptive preference model with phase-aware learning.

Combines theme embeddings, observation store, and phase detection
to provide context-aware theme recommendations that adapt to the
user's current learning phase.

Includes Bayesian uncertainty quantification via Gaussian posteriors
over the ideal embedding, enabling:
- Confidence scores per context
- Thompson sampling for exploration
- Identification of under-explored contexts
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .bayesian_embedding import ContextualPosterior, EmbeddingPosterior
from .embeddings import EMBEDDING_DIM, EmbeddingIndex, ThemeEmbedding
from .observations import Observation, ObservationStore
from .phase_detector import Phase, PhaseDetector, PHASE_CONFIGS


# Storage version for migration
STORAGE_VERSION = 3  # Bumped for Bayesian posterior addition

# Default storage path
DEFAULT_HISTORY_PATH = Path.home() / ".config/ghostty-ambient/history.json"


class AdaptivePreferenceModel:
    """
    Phase-aware preference model for theme recommendations.

    Features:
    - Stores timestamped observations with context
    - Detects learning phase (EXPLORE/CONVERGE/STABLE)
    - Adapts learning rate and recommendations based on phase
    - Computes weighted ideal embeddings with recency decay

    Usage:
        model = AdaptivePreferenceModel.load()

        # Record a choice
        model.record(theme_dict, context_dict, source="picker")

        # Get recommendations
        recommendations = model.recommend(context_dict, all_themes)

        # Get ideal theme embedding
        ideal = model.predict_ideal(context_dict)

        # Save
        model.save()
    """

    def __init__(self, history_path: Path | None = None, use_embedding_cache: bool = True):
        self.history_path = history_path or DEFAULT_HISTORY_PATH
        self.observations = ObservationStore()
        self.phase_detector = PhaseDetector()
        self.favorites: set[str] = set()
        self.disliked: set[str] = set()

        # Bayesian posterior for uncertainty quantification
        self.posterior = ContextualPosterior()

        # Random generator for Thompson sampling
        self._rng = np.random.default_rng()

        # Load embedding index from cache if available
        if use_embedding_cache:
            from .embedding_cache import load_embedding_cache
            self.embedding_index = load_embedding_cache(rebuild_if_stale=True, verbose=False)
        else:
            self.embedding_index = EmbeddingIndex()

    def record(
        self,
        theme: dict,
        context: dict[str, str],
        source: str = "picker",
    ) -> None:
        """
        Record a theme choice.

        Args:
            theme: Theme dictionary with at least 'name' and 'background'
            context: Context factors (e.g., {"time": "night", "lux": "dim"})
            source: How the theme was chosen ("picker", "ideal", "manual", "daemon")
        """
        # Ensure we have an embedding for this theme
        embedding = self.embedding_index.get(theme["name"])
        if embedding is None:
            embedding = self.embedding_index.add_theme(theme)

        # Create observation
        obs = Observation(
            timestamp=datetime.now(),
            theme_name=theme["name"],
            embedding=embedding.vector,
            context=context,
            source=source,
        )

        # Add to store
        self.observations.add(obs)

        # Update Bayesian posterior
        # Weight by recency (recent observations have more impact)
        config = self.phase_detector.get_config()
        self.posterior.update(embedding.vector, context, weight=config.learning_rate)

        # Update phase detection
        self.phase_detector.detect_from_store(self.observations)

    def current_phase(self) -> Phase:
        """Get the current learning phase."""
        return self.phase_detector.current_phase()

    def phase_probabilities(self) -> dict[Phase, float]:
        """Get probability distribution over phases."""
        return self.phase_detector.phase_probabilities()

    def predict_ideal(
        self,
        context: dict[str, str] | None = None,
        return_confidence: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, float]:
        """
        Predict ideal theme embedding for current context.

        Uses exponential recency weighting with half-life determined
        by the current learning phase. Optionally returns Bayesian
        confidence estimate.

        Args:
            context: Optional context filter
            return_confidence: If True, return (embedding, confidence) tuple

        Returns:
            20D embedding vector, or (embedding, confidence) if return_confidence=True
        """
        config = self.phase_detector.get_config()
        half_life = config.recency_half_life

        weighted_mean, total_weight = self.observations.compute_weighted_mean(
            half_life_days=half_life,
            context=context,
        )

        if return_confidence:
            # Get confidence from Bayesian posterior
            posterior = self.posterior.get_posterior(context or {})
            confidence = posterior.confidence
            return weighted_mean, confidence

        return weighted_mean

    def sample_ideal(self, context: dict[str, str] | None = None) -> np.ndarray:
        """
        Sample from the posterior over ideal embeddings (Thompson sampling).

        Useful for exploration in the EXPLORE phase - samples a plausible
        ideal embedding rather than using the point estimate.

        Args:
            context: Optional context filter

        Returns:
            Sampled 20D embedding vector
        """
        posterior = self.posterior.get_posterior(context or {})
        return posterior.sample(self._rng)

    def get_confidence(self, context: dict[str, str] | None = None) -> float:
        """
        Get confidence score for a context.

        Args:
            context: Context to check

        Returns:
            Confidence in [0, 1], higher = more certain
        """
        return self.posterior.context_confidence(context or {})

    def recommend(
        self,
        context: dict[str, str],
        all_themes: list[dict],
        k: int = 10,
    ) -> list[dict]:
        """
        Get theme recommendations based on current phase.

        - EXPLORE: Use Thompson sampling from posterior for diversity
        - CONVERGE: Recommend themes moving toward emerging preference
        - STABLE: Recommend refined ideal + close alternatives

        Args:
            context: Current context factors
            all_themes: List of all available themes
            k: Number of recommendations

        Returns:
            List of theme dicts with added '_score', '_distance', '_confidence' keys
        """
        # Ensure all themes are indexed
        for theme in all_themes:
            if theme["name"] not in self.embedding_index.embeddings:
                self.embedding_index.add_theme(theme)

        phase = self.current_phase()
        config = PHASE_CONFIGS[phase]

        # Get confidence for this context
        confidence = self.get_confidence(context)

        # Compute reference embedding based on phase
        if phase == Phase.EXPLORE:
            # Thompson sampling: sample from posterior for natural exploration
            if confidence < 0.3:
                # Very low confidence - sample from posterior
                center = self.sample_ideal(context)
            else:
                # Some confidence - mix sampled and mean
                sampled = self.sample_ideal(context)
                mean = self.predict_ideal(context)
                # Blend: more sampling when less confident
                blend = confidence
                center = blend * mean + (1 - blend) * sampled
        else:
            # CONVERGE/STABLE: use point estimate
            center = self.predict_ideal(context)

        # Find nearest themes
        nearest = self.embedding_index.nearest(
            center,
            k=k * 2,  # Get extra to filter
            exclude=self.disliked,
        )

        # Score and sort
        results = []
        for theme_name, distance in nearest:
            theme = next((t for t in all_themes if t["name"] == theme_name), None)
            if theme is None:
                continue

            # Base score inversely proportional to distance
            score = 100 * np.exp(-distance / 20)

            # Boost favorites
            if theme_name in self.favorites:
                score += 15

            # In EXPLORE phase, add diversity bonus for less similar themes
            if phase == Phase.EXPLORE:
                diversity_bonus = config.recommendation_diversity * distance * 0.5
                score += diversity_bonus

            theme["_score"] = score
            theme["_distance"] = distance
            theme["_phase"] = phase.value
            theme["_confidence"] = confidence
            results.append(theme)

        # Sort by score (descending)
        results.sort(key=lambda t: t["_score"], reverse=True)

        return results[:k]

    def find_similar(self, theme_name: str, k: int = 5) -> list[tuple[str, float]]:
        """Find themes similar to a given theme."""
        return self.embedding_index.similar_to(theme_name, k=k)

    def add_favorite(self, theme_name: str) -> None:
        """Mark a theme as favorite."""
        self.favorites.add(theme_name)
        self.disliked.discard(theme_name)

    def remove_favorite(self, theme_name: str) -> None:
        """Remove a theme from favorites."""
        self.favorites.discard(theme_name)

    def add_dislike(self, theme_name: str) -> None:
        """Mark a theme as disliked."""
        self.disliked.add(theme_name)
        self.favorites.discard(theme_name)

    def remove_dislike(self, theme_name: str) -> None:
        """Remove a theme from disliked."""
        self.disliked.discard(theme_name)

    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        features = self.observations.compute_features()
        phase = self.current_phase()
        probs = self.phase_probabilities()

        # Get global confidence
        global_confidence = self.posterior.global_posterior.confidence

        # Find least confident contexts
        least_confident = self.posterior.least_confident_contexts(top_k=3)

        return {
            "phase": phase.value,
            "phase_probabilities": {p.value: prob for p, prob in probs.items()},
            "total_observations": len(self.observations),
            "recent_variance": features.embedding_variance,
            "recent_unique_themes": features.unique_themes,
            "ideal_usage_rate": features.ideal_usage_rate,
            "favorites_count": len(self.favorites),
            "disliked_count": len(self.disliked),
            "indexed_themes": len(self.embedding_index.embeddings),
            # Bayesian confidence stats
            "global_confidence": global_confidence,
            "least_confident_contexts": least_confident,
            "posterior_observation_count": self.posterior.global_posterior.observation_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize model for storage."""
        return {
            "version": STORAGE_VERSION,
            "observations": self.observations.to_dict(),
            "phase_detector": self.phase_detector.to_dict(),
            "posterior": self.posterior.to_dict(),
            # Note: embedding_index is NOT stored here - it's loaded from the global cache
            "favorites": list(self.favorites),
            "disliked": list(self.disliked),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], history_path: Path | None = None) -> AdaptivePreferenceModel:
        """Deserialize model from storage."""
        model = cls(history_path=history_path)

        version = data.get("version", 1)
        if version < 2:
            # Old v1 format - need migration
            # Return empty model (user chose "reset and start fresh")
            return model

        if "observations" in data:
            model.observations = ObservationStore.from_dict(data["observations"])
        if "phase_detector" in data:
            model.phase_detector = PhaseDetector.from_dict(data["phase_detector"])
        if "posterior" in data:
            model.posterior = ContextualPosterior.from_dict(data["posterior"])
        elif version == 2:
            # Upgrading from v2 to v3: rebuild posterior from observations
            model._rebuild_posterior_from_observations()
        # Note: embedding_index is loaded from global cache in __init__, not from history
        if "favorites" in data:
            model.favorites = set(data["favorites"])
        if "disliked" in data:
            model.disliked = set(data["disliked"])

        return model

    def _rebuild_posterior_from_observations(self) -> None:
        """Rebuild Bayesian posterior from stored observations."""
        config = self.phase_detector.get_config()
        for obs in self.observations.observations:
            self.posterior.update(obs.embedding, obs.context, weight=config.learning_rate)

    def save(self, path: Path | None = None) -> None:
        """Save model to JSON file."""
        path = path or self.history_path
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | None = None) -> AdaptivePreferenceModel:
        """Load model from JSON file."""
        path = path or DEFAULT_HISTORY_PATH

        if not path.exists():
            return cls(history_path=path)

        try:
            with open(path) as f:
                data = json.load(f)

            # Check version
            version = data.get("version", 1)
            if version < 2:
                # v1 format (old Bayesian model) - backup and start fresh
                backup_path = path.with_suffix(".v1.bak")
                if not backup_path.exists():
                    import shutil
                    shutil.copy(path, backup_path)
                    print(f"Backed up old history to {backup_path}")

                print("Learning data reset for new adaptive model.")
                return cls(history_path=path)

            # v2 or v3 - load and migrate if needed
            model = cls.from_dict(data, history_path=path)

            # If upgrading from v2 to v3, save to persist the upgrade
            if version == 2:
                print("Upgraded history to v3 with Bayesian posteriors.")
                model.save(path)

            return model

        except (json.JSONDecodeError, KeyError) as e:
            # Corrupted file - backup and start fresh
            backup_path = path.with_suffix(".corrupted.bak")
            if path.exists() and not backup_path.exists():
                import shutil
                shutil.copy(path, backup_path)
            return cls(history_path=path)

    def build_index_from_themes(self, themes: list[dict]) -> None:
        """Build embedding index from a list of themes."""
        self.embedding_index.build_from_themes(themes)
