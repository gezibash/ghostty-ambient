"""
History - Theme preference learning using AdaptivePreferenceModel.

Provides tracking of theme choices, favorites, dislikes, and
phase-aware preference learning with embedding-based recommendations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adaptive_model import DEFAULT_HISTORY_PATH, AdaptivePreferenceModel
from .adaptive_scorer import build_context


class History:
    """
    Theme preference learning using AdaptivePreferenceModel.

    Tracks theme choices, favorites, dislikes, and provides
    phase-aware preference learning with embedding-based recommendations.
    """

    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_HISTORY_PATH
        self.model = AdaptivePreferenceModel.load(self.path)
        self._themes_indexed = False

    def _ensure_indexed(self, themes: list[dict] | None = None) -> None:
        """Ensure themes are indexed in the model."""
        if themes and not self._themes_indexed:
            self.model.build_index_from_themes(themes)
            self._themes_indexed = True

    def record_choice(
        self,
        theme_name: str,
        lux: float | None,
        hour: int,
        available: list[str],
        source: str = "recommended",
        weather_code: int | None = None,
        system_appearance: str | None = None,
        power_source: str | None = None,
        font: str | None = None,
        background_hex: str | None = None,
    ) -> None:
        """Record a theme choice (compatible with old History interface)."""
        # Build a minimal theme dict
        theme = {
            "name": theme_name,
            "background": background_hex or "#000000",
            "brightness": 128,
            "warmth": 0.0,
        }

        # Build context
        context = build_context(
            lux=lux,
            hour=hour,
            weather_code=weather_code,
            system_appearance=system_appearance,
            power_source=power_source,
            font=font,
        )

        # Map source names
        source_map = {
            "recommended": "picker",
            "browsed": "picker",
            "explore": "picker",
            "direct": "manual",
            "favorite": "picker",
            "picker": "picker",
            "ideal": "ideal",
            "daemon": "daemon",
        }
        mapped_source = source_map.get(source, "picker")

        # Record in model
        self.model.record(theme, context, source=mapped_source)

        # Auto-save
        self._save()

    def record_snapshot(
        self,
        theme_name: str,
        factors: dict[str, str],
        background_hex: str | None = None,
        foreground_hex: str | None = None,
        palette_chromas: list[float] | None = None,
    ) -> None:
        """Record a passive observation (daemon snapshot)."""
        theme = {
            "name": theme_name,
            "background": background_hex or "#000000",
            "brightness": 128,
            "warmth": 0.0,
        }
        if foreground_hex:
            theme["foreground"] = foreground_hex

        self.model.record(theme, factors, source="daemon")

        # Auto-save (debounced in daemon, but safe to call)
        self._save()

    def add_favorite(self, theme_name: str) -> None:
        """Mark a theme as favorite."""
        self.model.add_favorite(theme_name)
        self._save()

    def remove_favorite(self, theme_name: str) -> None:
        """Remove a theme from favorites."""
        self.model.remove_favorite(theme_name)
        self._save()

    def add_dislike(self, theme_name: str) -> None:
        """Mark a theme as disliked."""
        self.model.add_dislike(theme_name)
        self._save()

    def remove_dislike(self, theme_name: str) -> None:
        """Remove a theme from disliked."""
        self.model.remove_dislike(theme_name)
        self._save()

    def is_favorite(self, theme_name: str) -> bool:
        """Check if theme is a favorite."""
        return theme_name in self.model.favorites

    def is_disliked(self, theme_name: str) -> bool:
        """Check if theme is disliked."""
        return theme_name in self.model.disliked

    def get_favorites(self) -> list[str]:
        """Get list of favorite themes."""
        return list(self.model.favorites)

    def get_disliked(self) -> list[str]:
        """Get list of disliked themes."""
        return list(self.model.disliked)

    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        stats = self.model.get_stats()

        # Add compatibility fields
        return {
            "phase": stats["phase"],
            "phase_probabilities": stats["phase_probabilities"],
            "total_observations": stats["total_observations"],
            "favorites_count": stats["favorites_count"],
            "disliked_count": stats["disliked_count"],
            "indexed_themes": stats["indexed_themes"],
            "recent_variance": stats["recent_variance"],
            "ideal_usage_rate": stats["ideal_usage_rate"],
            # Legacy compatibility
            "total_choices": stats["total_observations"],
            "unique_themes": stats.get("recent_unique_themes", 0),
        }

    def get_theme_model_stats(self) -> dict[str, Any]:
        """Get theme model statistics (compatibility)."""
        stats = self.model.get_stats()
        return {
            "phase": stats["phase"],
            "observations": stats["total_observations"],
            "confidence": min(1.0, stats["total_observations"] / 50),
        }

    def reset_learning(self, keep_favorites: bool = True) -> dict[str, int]:
        """Reset learning data."""
        old_obs = len(self.model.observations)

        # Clear observations
        self.model.observations = type(self.model.observations)()
        self.model.phase_detector.reset()

        if not keep_favorites:
            self.model.favorites = set()
            self.model.disliked = set()

        self._save()

        return {
            "observations_cleared": old_obs,
            "favorites_kept": len(self.model.favorites) if keep_favorites else 0,
            "disliked_kept": len(self.model.disliked) if keep_favorites else 0,
            # Legacy fields
            "color_posteriors_cleared": old_obs,
            "factor_betas_cleared": 0,
            "events_cleared": old_obs,
            "snapshots_cleared": 0,
        }

    def _save(self) -> None:
        """Save model to disk."""
        self.model.save(self.path)

    @property
    def data(self) -> dict[str, Any]:
        """Compatibility property for accessing raw data."""
        # Build a compatible structure for show_stats
        stats = self.model.get_stats()

        # Create a minimal theme_posteriors-like structure for compatibility
        # This signals to show_stats that we have data
        theme_posteriors = {}
        if stats["total_observations"] > 0:
            theme_posteriors["_adaptive_model"] = {"has_data": True}

        return {
            "favorites": list(self.model.favorites),
            "disliked": list(self.model.disliked),
            "version": 2,
            "theme_posteriors": theme_posteriors,
            "_adaptive_stats": stats,
        }

    def show_stats(self) -> None:
        """Display learning statistics for v2 model."""
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        stats = self.model.get_stats()

        if stats["total_observations"] == 0:
            console.print("No learning data yet. Run: ghostty-ambient --daemon")
            return

        console.print()

        # Phase info
        phase = stats["phase"]
        probs = stats["phase_probabilities"]

        phase_colors = {
            "explore": "yellow",
            "converge": "cyan",
            "stable": "green",
        }

        console.print(f"[bold]Learning Phase:[/] [{phase_colors.get(phase, 'white')}]{phase.upper()}[/]")
        console.print(
            f"  Explore: {probs.get('explore', 0):.0%}  Converge: {probs.get('converge', 0):.0%}  Stable: {probs.get('stable', 0):.0%}"
        )
        console.print()

        # Confidence info
        global_conf = stats.get("global_confidence", 0)
        conf_color = "green" if global_conf > 0.6 else "yellow" if global_conf > 0.3 else "red"
        console.print(f"[bold]Confidence:[/] [{conf_color}]{global_conf:.0%}[/]")

        # Show least confident contexts if any
        least_confident = stats.get("least_confident_contexts", [])
        if least_confident and least_confident[0][1] < 0.5:
            low_conf = [ctx for ctx, conf in least_confident[:2] if conf < 0.5]
            if low_conf:
                console.print(f"  [dim]Needs more data for: {', '.join(low_conf)}[/]")
        console.print()

        # Stats table
        table = Table(title="Learning Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Total Observations", str(stats["total_observations"]))
        table.add_row("Bayesian Updates", str(stats.get("posterior_observation_count", 0)))
        table.add_row("Indexed Themes", str(stats["indexed_themes"]))
        table.add_row("Recent Variance", f"{stats['recent_variance']:.3f}")
        table.add_row("Ideal Usage Rate", f"{stats['ideal_usage_rate']:.0%}")
        table.add_row("Favorites", str(stats["favorites_count"]))
        table.add_row("Disliked", str(stats["disliked_count"]))

        console.print(table)
        console.print()

        # Recent observations
        recent = self.model.observations.recent(10)
        if recent:
            obs_table = Table(title="Recent Observations", box=box.SIMPLE)
            obs_table.add_column("Time", style="dim", width=8)
            obs_table.add_column("Theme", style="bold", width=25)
            obs_table.add_column("Source", width=8)
            obs_table.add_column("Context", ratio=1)

            for obs in reversed(recent):
                time_str = obs.timestamp.strftime("%H:%M:%S")
                # Show values only (no keys), skip unknowns, prioritized order
                priority_keys = ["time", "lux", "weather", "system", "power", "circadian", "day"]
                ctx_parts = []
                for key in priority_keys:
                    val = obs.context.get(key, "")
                    if val and val != "unknown":
                        ctx_parts.append(val)
                ctx_str = " · ".join(ctx_parts)
                obs_table.add_row(time_str, obs.theme_name[:25], obs.source, ctx_str)

            console.print(obs_table)

    @property
    def theme_model(self):
        """Compatibility property for theme model access."""
        # Return a mock object that provides the interface needed by ThemeGenerator
        return _ThemeModelCompat(self.model)


class _ThemeModelCompat:
    """Compatibility wrapper for theme model interface."""

    def __init__(self, model: AdaptivePreferenceModel):
        self._model = model

    def predict_ideal(self, factors: dict[str, str]) -> tuple[tuple[float, float, float], float]:
        """
        Get ideal background color for context.

        Returns:
            (lab_tuple, confidence) where lab_tuple is (L, a, b)
        """
        if len(self._model.observations) < 3:
            # Return neutral dark gray as default
            return (25.0, 0.0, 0.0), 0.0

        # Use the new return_confidence parameter
        ideal, confidence = self._model.predict_ideal(factors, return_confidence=True)

        # Extract LAB values from embedding (first 3 dimensions are background LAB)
        L, a, b = float(ideal[0]), float(ideal[1]), float(ideal[2])

        return (L, a, b), confidence

    def predict_contrast(self, factors: dict[str, str]) -> float:
        """Get ideal contrast for context."""
        if len(self._model.observations) < 3:
            return 70.0  # Default contrast

        ideal = self._model.predict_ideal(factors)

        # Contrast is dimension 6
        return float(ideal[6])

    def predict_chroma(self, factors: dict[str, str]) -> float:
        """Get ideal chroma for context."""
        if len(self._model.observations) < 3:
            return 40.0  # Default chroma

        ideal = self._model.predict_ideal(factors)

        # Avg chroma is dimension 7, denormalized (was /130)
        return float(ideal[7]) * 130
