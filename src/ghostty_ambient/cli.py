"""
ghostty-ambient - Ambient light-aware Ghostty theme selector with learning.

Reads ambient light sensor and proposes themes based on:
- Current lighting conditions (lux) - cross-platform sensor support
- Time of day and sun position
- Weather conditions (via Open-Meteo)
- Learned color preferences (Bayesian color posterior model)
- System dark/light mode preference

Usage:
    ghostty-ambient              # Show picker with recommendations
    ghostty-ambient --ideal      # Generate and apply optimal theme for current context
    ghostty-ambient --set NAME   # Set theme by name directly
    ghostty-ambient --apply N    # Apply recommendation N (1-5 or a-e for explore)
    ghostty-ambient --json       # Output as JSON
    ghostty-ambient --stats      # Show learning statistics
    ghostty-ambient --favorite   # Mark current theme as favorite
    ghostty-ambient --dislike    # Mark current theme as disliked
    ghostty-ambient --sensors    # Show available sensor backends
"""

import argparse
import json
import sys
from datetime import datetime

from .history import History
from .scorer import score_themes
from .sensor import (
    CONFIG_FILE,
    WeatherData,
    get_geolocation,
    get_lux,
    get_power_source,
    get_sun_phase,
    get_system_appearance,
    get_weather,
    load_config,
    lux_to_condition,
)
from .sensors import discover_backends, get_best_backend
from .themes import (
    GHOSTTY_CONFIG,
    apply_theme,
    get_current_font,
    get_current_theme,
    load_all_themes,
)


def parse_interval(value: str) -> int | None:
    """
    Parse interval string to seconds.

    Supports:
        - Plain numbers: "300" → 300 seconds
        - Seconds: "30s" → 30 seconds
        - Minutes: "5m" → 300 seconds
        - Hours: "1h" → 3600 seconds

    Returns None if invalid format.
    """
    value = value.strip().lower()
    if not value:
        return None

    try:
        # Plain number (seconds)
        if value.isdigit():
            return int(value)

        # With suffix
        if value.endswith("s"):
            return int(value[:-1])
        elif value.endswith("m"):
            return int(value[:-1]) * 60
        elif value.endswith("h"):
            return int(value[:-1]) * 3600
        else:
            return None
    except ValueError:
        return None


def format_output(
    weather: WeatherData,
    lux: float | None,
    recommendations: list[dict],
    history: History | None = None,
    sensor_name: str | None = None,
    system_appearance: str | None = None,
    power_source: str | None = None,
    current_theme: dict | None = None,
    location: str | None = None,
    timezone: str | None = None,
) -> str:
    """Format the output for display."""
    now = datetime.now()
    lines = []

    # Header box
    lines.append("╭" + "─" * 50 + "╮")

    # Location info
    if location:
        tz_info = f" ({timezone})" if timezone else ""
        lines.append(f"│ Location: {location}{tz_info}".ljust(51) + "│")

    # Weather info
    if weather.temperature is not None:
        lines.append(f"│ Weather: {weather.temperature:.0f}°C, {weather.condition}".ljust(51) + "│")

    # Lux info with sensor name
    if lux is not None:
        condition, _, _ = lux_to_condition(lux)
        sensor_info = f" via {sensor_name}" if sensor_name else ""
        lines.append(f"│ Ambient: {lux:.0f} lux ({condition}){sensor_info}".ljust(51) + "│")
    else:
        lines.append("│ Ambient: unavailable".ljust(51) + "│")

    # Time info
    if weather.sunrise and weather.sunset:
        sun_phase, _ = get_sun_phase(now, weather.sunrise, weather.sunset)
        lines.append(f"│ Time: {now.strftime('%H:%M')} ({sun_phase})".ljust(51) + "│")
        lines.append(
            f"│ Sun: rises {weather.sunrise.strftime('%H:%M')}, sets {weather.sunset.strftime('%H:%M')}".ljust(51) + "│"
        )
    else:
        lines.append(f"│ Time: {now.strftime('%H:%M')}".ljust(51) + "│")

    # System info
    sys_info_parts = []
    if system_appearance and system_appearance != "unknown":
        sys_info_parts.append(f"{system_appearance} mode")
    if power_source and power_source != "unknown":
        power_display = {"ac": "AC", "battery_high": "battery", "battery_low": "low battery"}.get(power_source, power_source)
        sys_info_parts.append(power_display)
    if sys_info_parts:
        lines.append(f"│ System: {', '.join(sys_info_parts)}".ljust(51) + "│")

    lines.append("╰" + "─" * 50 + "╯")
    lines.append("")

    # Current theme
    if current_theme:
        lines.append(f"Current: {current_theme['name']}")

    return "\n".join(lines)


def setup_config():
    """Interactive setup for configuration."""
    print("ghostty-ambient setup")
    print("=" * 40)
    print()
    print("Configure your location for accurate sunrise/sunset times.")
    print("Weather data is provided by Open-Meteo (free, no API key required).")
    print()

    # Get current location from IP as default
    geo = get_geolocation()
    if geo.city and geo.country:
        print(f"Detected location: {geo.city}, {geo.country} ({geo.lat:.4f}, {geo.lon:.4f})")
        custom = input("Use different location? [y/N]: ").strip().lower()
    else:
        print("Could not detect location from IP.")
        custom = "y"

    lat, lon = geo.lat, geo.lon
    if custom == "y":
        try:
            lat = float(input("Latitude: ").strip())
            lon = float(input("Longitude: ").strip())
        except ValueError:
            print("Invalid coordinates, keeping detected location.")

    if lat is None or lon is None:
        print("Error: No location available.", file=sys.stderr)
        return

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config = {"lat": lat, "lon": lon}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Config saved to {CONFIG_FILE}")


def show_sensors():
    """Display available sensor backends."""
    backends = discover_backends()

    print("Available sensor backends:")
    print()

    if not backends:
        print("  No sensor backends found for this platform.")
        return

    for backend in backends:
        status = "available" if backend["available"] else "not available"
        print(f"  {backend['name']} ({backend['platform']})")
        print(f"    Status: {status}")
        print()

    # Show which one would be used
    best = get_best_backend()
    if best:
        print(f"Active backend: {best.name}")
        if best.is_available():
            reading = best.read()
            if reading.lux is not None:
                print(f"Current reading: {reading.lux:.1f} lux")
            elif reading.error:
                print(f"Error: {reading.error}")


def show_snapshots(history: History):
    """Display recent learning snapshots in a rich table."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    snapshots = history.data.get("recent_snapshots", [])

    if not snapshots:
        console.print("[yellow]No snapshots recorded yet.[/]")
        console.print("Run: [bold]ghostty-ambient --daemon[/]")
        return

    table = Table(
        title=f"Recent Snapshots ({len(snapshots)} total)",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("ts", style="dim", no_wrap=True, width=8)
    table.add_column("theme", style="bold", no_wrap=True, width=20)
    table.add_column("factors", no_wrap=False, ratio=1)

    for snap in reversed(snapshots[-15:]):  # Show last 15, newest first
        ts = snap["timestamp"][11:19]  # HH:MM:SS
        theme = snap["theme"]
        f = snap.get("factors", {})

        # Build compact factor string, skip unknowns
        parts = []
        for key in ["time", "lux", "weather", "system", "day", "power", "circadian", "pressure", "clouds", "uv"]:
            val = f.get(key, "-")
            if val and val != "-" and val != "unknown":
                parts.append(f"[cyan]{key}[/]={val}")
        # Font separate
        font = f.get("font", "")
        if font and font != "unknown":
            parts.append(f"[dim]font={font}[/]")

        table.add_row(ts, theme[:22], " ".join(parts))

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Ambient light-aware Ghostty theme selector with learning"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--set", type=str, metavar="NAME", help="Set theme by name directly")
    parser.add_argument("--apply", type=str, metavar="N", help="Apply theme (1-5 for recommended, a-e for explore)")
    parser.add_argument("--count", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--lux", type=float, help="Override lux value (for testing)")
    parser.add_argument("--setup", action="store_true", help="Configure location")
    parser.add_argument("--favorite", action="store_true", help="Mark current theme as favorite")
    parser.add_argument("--dislike", action="store_true", help="Mark current theme as disliked")
    parser.add_argument("--unfavorite", action="store_true", help="Remove current theme from favorites")
    parser.add_argument("--undislike", action="store_true", help="Remove current theme from disliked")
    parser.add_argument("--sensors", action="store_true", help="Show available sensor backends")
    parser.add_argument("--reset-learning", action="store_true", help="Clear learned preferences and start fresh")
    parser.add_argument("--clean", action="store_true", help="Clear recent snapshots (keeps learned preferences)")
    parser.add_argument("--ideal", action="store_true", help="Generate and apply optimal theme for current context")
    parser.add_argument("--daemon", action="store_true", help="Run background learning daemon")
    parser.add_argument("--interval", type=str, default="5m", help="Daemon snapshot interval (e.g., 30s, 5m, 1h)")
    parser.add_argument("--snapshots", action="store_true", help="Show recent learning snapshots")
    parser.add_argument("--stats", action="store_true", help="Show learning statistics")
    parser.add_argument("--export-profile", type=str, metavar="FILE", help="Export learned preferences to file")
    parser.add_argument("--import-profile", type=str, metavar="FILE", help="Import learned preferences from file")
    # Daemon management
    parser.add_argument("--status", action="store_true", help="Show daemon status")
    parser.add_argument("--start", action="store_true", help="Start the daemon")
    parser.add_argument("--stop", action="store_true", help="Stop the daemon")
    parser.add_argument("--restart", action="store_true", help="Restart the daemon")
    parser.add_argument("--logs", action="store_true", help="Tail daemon logs")
    parser.add_argument("--freq", type=str, default="5m", help="Snapshot interval for --start (e.g. 30s, 1m, 5m, 1h)")
    args = parser.parse_args()

    # Daemon management commands
    if args.status:
        from .daemon_manager import daemon_status
        daemon_status()
        return
    if args.start:
        from .daemon_manager import daemon_start
        daemon_start(freq=args.freq)
        return
    if args.stop:
        from .daemon_manager import daemon_stop
        daemon_stop()
        return
    if args.restart:
        from .daemon_manager import daemon_restart
        daemon_restart()
        return
    if args.logs:
        from .daemon_manager import daemon_logs
        daemon_logs()
        return

    # Setup mode
    if args.setup:
        setup_config()
        return

    # Show sensors
    if args.sensors:
        show_sensors()
        return

    # Reset learning
    if args.reset_learning:
        history = History()
        stats = history.reset_learning(keep_favorites=True)
        print("Learning data cleared:")
        print(f"  Color posteriors: {stats['color_posteriors_cleared']}")
        print(f"  Factor preferences: {stats['factor_betas_cleared']}")
        print(f"  Events: {stats['events_cleared']}")
        print(f"  Snapshots: {stats['snapshots_cleared']}")
        print()
        print("Kept: favorites and disliked lists")
        print("Run the daemon to start fresh learning: ghostty-ambient --daemon")
        return

    # Clean recent snapshots only
    if args.clean:
        history = History()
        count = len(history.data.get("recent_snapshots", []))
        history.data["recent_snapshots"] = []
        history._save()
        print(f"Cleared {count} recent snapshots")
        print("Learning preferences kept intact")
        return

    # Generate and apply ideal theme for current context
    if args.ideal:
        from pathlib import Path
        from .theme_generator import ThemeGenerator
        from .factors import FactorRegistry

        history = History()

        # Get current context
        weather = get_weather()
        system_appearance = get_system_appearance()
        power_source = get_power_source()
        font = get_current_font()
        lux = None
        backend = get_best_backend()
        if backend and backend.is_available():
            reading = backend.read()
            lux = reading.lux

        # Build factors for current context
        now = datetime.now()
        context = {
            "hour": now.hour,
            "lux": lux,
            "weather_code": weather.weather_code,
            "system_appearance": system_appearance,
            "power_source": power_source,
            "font": font,
        }
        factors = FactorRegistry.get_all_buckets(context)

        # Generate theme
        generator = ThemeGenerator(history.theme_model)
        preview = generator.get_preview(factors)

        if preview["confidence"] < 0.1:
            print("Not enough learning data to generate a theme.", file=sys.stderr)
            print("Run the daemon to learn your preferences: ghostty-ambient --daemon", file=sys.stderr)
            sys.exit(1)

        theme_content = generator.generate(factors, name="Ideal")

        # Ensure user themes directory exists
        theme_dir = Path.home() / ".config/ghostty/themes"
        theme_dir.mkdir(parents=True, exist_ok=True)

        # Save theme (overwrite each time - it's context-dependent)
        theme_path = theme_dir / "Ideal"
        theme_path.write_text(theme_content)

        # Apply it immediately
        if apply_theme("Ideal"):
            mode = "dark" if preview["is_dark"] else "light"
            print(f"Applied: Ideal ({mode}, {preview['confidence']:.0%} confidence)")
            print(f"  contrast={preview['contrast']:.1f}, chroma={preview['chroma']:.1f}")
        else:
            print("Error applying theme", file=sys.stderr)
            sys.exit(1)
        return

    # Run daemon mode
    if args.daemon:
        from .daemon import run_daemon
        interval = parse_interval(args.interval)
        if interval is None:
            print(f"Invalid interval: {args.interval}", file=sys.stderr)
            print("Use formats like: 30s, 5m, 1h, or plain seconds (300)", file=sys.stderr)
            sys.exit(1)
        run_daemon(interval=interval)
        return

    # Initialize history
    history = History()

    # Set theme by name directly
    if args.set:
        from difflib import get_close_matches

        themes = load_all_themes()
        theme_names = [t["name"] for t in themes]

        # Find theme by name (case-insensitive exact match)
        theme_name_lower = args.set.lower()
        theme = next(
            (t for t in themes if t["name"].lower() == theme_name_lower),
            None
        )
        if not theme:
            # Try partial match (substring)
            matches = [t for t in themes if theme_name_lower in t["name"].lower()]
            if len(matches) == 1:
                theme = matches[0]
            elif len(matches) > 1:
                print(f"Multiple themes match '{args.set}':", file=sys.stderr)
                for m in matches[:5]:
                    print(f"  {m['name']}", file=sys.stderr)
                sys.exit(1)
            else:
                # Try fuzzy match
                close = get_close_matches(args.set, theme_names, n=3, cutoff=0.6)
                if close:
                    print(f"Theme not found: {args.set}", file=sys.stderr)
                    print(f"Did you mean:", file=sys.stderr)
                    for name in close:
                        print(f"  {name}", file=sys.stderr)
                else:
                    print(f"Theme not found: {args.set}", file=sys.stderr)
                sys.exit(1)

        # Apply the theme
        if apply_theme(theme["name"]):
            # Record in history with current context
            weather = get_weather()
            system_appearance = get_system_appearance()
            power_source = get_power_source()
            font = get_current_font()
            lux = None
            backend = get_best_backend()
            if backend and backend.is_available():
                reading = backend.read()
                lux = reading.lux

            now = datetime.now()
            history.record_choice(
                theme["name"],
                lux,
                now.hour,
                [theme["name"]],  # Only this theme was "available"
                source="direct",
                weather_code=weather.weather_code,
                system_appearance=system_appearance,
                power_source=power_source,
                font=font,
                background_hex=theme.get("background"),
            )
            print(f"Applied: {theme['name']}")
        else:
            print(f"Error applying theme", file=sys.stderr)
            sys.exit(1)
        return

    # Show snapshots
    if args.snapshots:
        show_snapshots(history)
        return

    # Show stats
    if args.stats:
        from .browser import show_stats
        show_stats(history)
        return

    # Export profile
    if args.export_profile:
        from pathlib import Path
        import platform

        profile = {
            "version": 1,
            "exported": datetime.now().isoformat(),
            "device": platform.node(),
            "theme_posteriors": history.data.get("theme_posteriors", {}),
            "favorites": history.data.get("favorites", []),
            "disliked": history.data.get("disliked", []),
        }

        # Count observations for summary
        total_obs = sum(
            len(ctx.get("color", {}).get("observations", []))
            for ctx in profile["theme_posteriors"].values()
        )
        contexts = len(profile["theme_posteriors"])

        output_path = Path(args.export_profile)
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2)

        print(f"Exported profile to: {output_path}")
        print(f"  {contexts} contexts, {total_obs} observations")
        print(f"  {len(profile['favorites'])} favorites, {len(profile['disliked'])} disliked")
        return

    # Import profile
    if args.import_profile:
        from pathlib import Path

        input_path = Path(args.import_profile)
        if not input_path.exists():
            print(f"File not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        with open(input_path) as f:
            profile = json.load(f)

        if "theme_posteriors" not in profile:
            print("Invalid profile: missing theme_posteriors", file=sys.stderr)
            sys.exit(1)

        # Merge into current history
        imported_posteriors = profile.get("theme_posteriors", {})
        current_posteriors = history.data.get("theme_posteriors", {})

        # Merge observations (append imported to current)
        for ctx, ctx_data in imported_posteriors.items():
            if ctx not in current_posteriors:
                current_posteriors[ctx] = ctx_data
            else:
                # Merge observations for each type
                for obs_type in ["color", "contrast", "chroma"]:
                    if obs_type in ctx_data:
                        if obs_type not in current_posteriors[ctx]:
                            current_posteriors[ctx][obs_type] = {"observations": []}
                        current_posteriors[ctx][obs_type]["observations"].extend(
                            ctx_data[obs_type].get("observations", [])
                        )

        history.data["theme_posteriors"] = current_posteriors

        # Merge favorites and disliked (union)
        existing_favs = set(history.data.get("favorites", []))
        existing_disliked = set(history.data.get("disliked", []))
        history.data["favorites"] = list(existing_favs | set(profile.get("favorites", [])))
        history.data["disliked"] = list(existing_disliked | set(profile.get("disliked", [])))

        history._save()

        # Reload the model with merged data
        from .color_posterior import ThemePreferenceModel
        history.theme_model = ThemePreferenceModel(history.data.get("theme_posteriors"))

        imported_obs = sum(
            len(ctx.get("color", {}).get("observations", []))
            for ctx in imported_posteriors.values()
        )
        print(f"Imported profile from: {input_path}")
        print(f"  Source: {profile.get('device', 'unknown')} ({profile.get('exported', 'unknown')[:10]})")
        print(f"  Merged {len(imported_posteriors)} contexts, {imported_obs} observations")
        return

    # Handle favorite/dislike for current theme
    current = get_current_theme()

    if args.favorite:
        if current:
            history.add_favorite(current)
            print(f"★ Added to favorites: {current}")
        else:
            print("No theme currently set", file=sys.stderr)
        return

    if args.dislike:
        if current:
            history.add_dislike(current)
            print(f"✗ Marked as disliked: {current}")
        else:
            print("No theme currently set", file=sys.stderr)
        return

    if args.unfavorite:
        if current:
            history.remove_favorite(current)
            print(f"Removed from favorites: {current}")
        else:
            print("No theme currently set", file=sys.stderr)
        return

    if args.undislike:
        if current:
            history.remove_dislike(current)
            print(f"Removed from disliked: {current}")
        else:
            print("No theme currently set", file=sys.stderr)
        return

    # Get sensor and weather data
    sensor_name = None
    if args.lux is not None:
        lux = args.lux
    else:
        # Try to get lux from sensor registry
        backend = get_best_backend()
        if backend and backend.is_available():
            reading = backend.read()
            lux = reading.lux
            sensor_name = backend.name
            if lux is None and reading.error:
                # Sensor available but failed - fall back to legacy
                lux = get_lux()
                sensor_name = "legacy"
        else:
            # No sensor backend available, use legacy method
            lux = get_lux()
            if lux is not None:
                sensor_name = "legacy"

    weather = get_weather()

    # Get location info for display
    config = load_config()
    location_str = None
    timezone_str = None
    if config.get("city") and config.get("country"):
        location_str = f"{config['city']}, {config['country']}"
        timezone_str = config.get("timezone")

    # Detect system context factors
    system_appearance = get_system_appearance()
    power_source = get_power_source()
    font = get_current_font()

    # Load and score themes
    themes = load_all_themes()
    if not themes:
        print("Error: No themes found", file=sys.stderr)
        sys.exit(1)

    # Score themes using factorized Bayesian preference
    scored = score_themes(
        themes,
        lux=lux,
        weather_code=weather.weather_code,
        history=history,
        system_appearance=system_appearance,
        power_source=power_source,
        font=font,
    )

    # Get recommendations with scores for UI
    # Note: theme already has _final_score from scorer, but we also set _score for TUI
    recommendations = []
    for score, theme in scored[:args.count]:
        theme["_score"] = score  # Score is already in ~0-100 range
        recommendations.append(theme)

    # Apply mode
    if args.apply:
        theme_to_apply = None
        source = "recommended"

        # Check if it's a number (1-5 for recommendations)
        if args.apply.isdigit():
            idx = int(args.apply)
            if 1 <= idx <= len(recommendations):
                theme_to_apply = recommendations[idx - 1]
            else:
                print(f"Error: Invalid selection {args.apply}", file=sys.stderr)
                sys.exit(1)

        # Check if it's a letter (a-e for explore themes)
        elif args.apply.lower() in "abcde":
            # Get explore themes (lowest-scoring = most different from usual)
            if len(scored) > args.count + 5:
                rec_names = {r["name"] for r in recommendations}
                explore = [t for _, t in scored[-15:-2] if t["name"] not in rec_names][:5]
                letter_idx = ord(args.apply.lower()) - ord('a')
                if letter_idx < len(explore):
                    theme_to_apply = explore[letter_idx]
                    source = "explore"
                else:
                    print(f"Error: No explore theme at '{args.apply}'", file=sys.stderr)
                    sys.exit(1)
            else:
                print("Error: Not enough themes for explore mode", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Error: Invalid selection '{args.apply}' (use 1-5 or a-e)", file=sys.stderr)
            sys.exit(1)

        if theme_to_apply:
            theme_name = theme_to_apply["name"]

            # Record the choice in history
            available = [t["name"] for _, t in scored[:20]]
            now = datetime.now()
            history.record_choice(
                theme_name,
                lux,
                now.hour,
                available,
                source=source,
                weather_code=weather.weather_code,
                system_appearance=system_appearance,
                power_source=power_source,
                font=font,
                background_hex=theme_to_apply.get("background"),
            )

            # Apply the theme
            if apply_theme(theme_name):
                print(f"Applied: {theme_name}")
            else:
                print(f"Error applying theme", file=sys.stderr)
                sys.exit(1)
            return

    # JSON output
    if args.json:
        output = {
            "lux": lux,
            "weather": {
                "temperature": weather.temperature,
                "condition": weather.condition,
                "is_day": weather.is_day,
            },
            "sunrise": weather.sunrise.isoformat() if weather.sunrise else None,
            "sunset": weather.sunset.isoformat() if weather.sunset else None,
            "current_theme": current,
            "recommendations": [
                {
                    "name": t["name"],
                    "background": t["background"],
                    "brightness": t["brightness"],
                    "score": t.get("_final_score", 0),
                    "color_score": t.get("_color_score", 50),
                    "familiarity_boost": t.get("_familiarity_boost", 0),
                    "context_penalty": t.get("_context_penalty", 0),
                    "choice_count": t.get("_choice_count", 0),
                    "is_favorite": t.get("_is_favorite", False),
                }
                for t in recommendations
            ],
            "stats": history.get_stats(),
            "theme_model_stats": history.get_theme_model_stats(),
        }
        print(json.dumps(output, indent=2))
    else:
        # Find current theme dict
        current_theme_name = get_current_theme()
        current_theme_dict = None
        if current_theme_name:
            current_theme_dict = next(
                (t for t in themes if t["name"] == current_theme_name), None
            )

        # Get explore themes: well-scoring themes you haven't tried much
        # These are "fresh discoveries" - good themes waiting to be explored
        explore = None
        if len(scored) > args.count + 5 and history:
            top_score = scored[0][0] if scored else 100
            threshold = top_score * 0.7  # Must score at least 70% of top
            rec_names = {r["name"] for r in recommendations}

            # Find themes that score well but have low choice count
            candidates = []
            for score, theme in scored:
                if theme["name"] in rec_names:
                    continue
                if score < threshold:
                    continue
                choice_count = theme.get("_choice_count", 0)
                # Prioritize rarely chosen themes
                candidates.append((choice_count, score, theme))

            # Sort by choice count (ascending), then by score (descending)
            candidates.sort(key=lambda x: (x[0], -x[1]))

            explore = []
            for choice_count, score, theme in candidates[:10]:
                theme["_score"] = score
                explore.append(theme)

        # Build status header
        header = format_output(
            weather, lux, recommendations, history, sensor_name,
            system_appearance, power_source, current_theme_dict,
            location_str, timezone_str
        )

        # Show interactive picker with header
        from .tui import pick_theme
        selected = pick_theme(recommendations, explore, current_theme_name, header=header)

        if selected:
            theme_name = selected["name"]

            # Record the choice
            now = datetime.now()
            available = [t["name"] for _, t in scored[:20]]
            history.record_choice(
                theme_name,
                lux,
                now.hour,
                available,
                source="picker",
                weather_code=weather.weather_code,
                system_appearance=system_appearance,
                power_source=power_source,
                font=font,
                background_hex=selected.get("background"),
            )

            # Apply the theme
            if apply_theme(theme_name):
                print(f"Applied: {theme_name}")


if __name__ == "__main__":
    main()
