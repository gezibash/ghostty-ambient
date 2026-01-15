"""Ambient light sensor and weather utilities using Open-Meteo."""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

# Path to ambient light sensor binary (project root)
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
ALS_BINARY = SCRIPT_DIR / "als"

# Config paths
CONFIG_FILE = Path.home() / ".config/ghostty-ambient/config.json"
WEATHER_CACHE_FILE = Path.home() / ".config/ghostty-ambient/weather_cache.json"
GEOLOCATION_CACHE_FILE = Path.home() / ".config/ghostty-ambient/geolocation_cache.json"
WEATHER_CACHE_TTL = 15 * 60  # 15 minutes in seconds
GEOLOCATION_CACHE_TTL = 24 * 60 * 60  # 24 hours (location rarely changes)

# WMO Weather codes to descriptions
WEATHER_CODES = {
    0: "Clear",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Foggy",
    51: "Light Drizzle",
    53: "Drizzle",
    55: "Heavy Drizzle",
    56: "Freezing Drizzle",
    57: "Freezing Drizzle",
    61: "Light Rain",
    63: "Rain",
    65: "Heavy Rain",
    66: "Freezing Rain",
    67: "Freezing Rain",
    71: "Light Snow",
    73: "Snow",
    75: "Heavy Snow",
    77: "Snow Grains",
    80: "Light Showers",
    81: "Showers",
    82: "Heavy Showers",
    85: "Snow Showers",
    86: "Heavy Snow Showers",
    95: "Thunderstorm",
    96: "Thunderstorm",
    99: "Thunderstorm",
}


@dataclass
class WeatherData:
    """Weather data from Open-Meteo."""
    temperature: float | None = None
    weather_code: int | None = None
    condition: str = ""
    is_day: bool = True
    cloud_cover: int | None = None
    sunrise: datetime | None = None
    sunset: datetime | None = None
    pressure: float | None = None  # Surface pressure in hPa
    uv_index: float | None = None  # UV index (daily max)


@dataclass
class GeoLocation:
    """Geolocation data from IP lookup."""
    lat: float | None = None
    lon: float | None = None
    city: str | None = None
    country: str | None = None
    timezone: str | None = None


def _load_geolocation_cache() -> dict | None:
    """Load geolocation from cache if still valid."""
    if not GEOLOCATION_CACHE_FILE.exists():
        return None

    try:
        with open(GEOLOCATION_CACHE_FILE) as f:
            cache = json.load(f)

        cached_at = cache.get("cached_at", 0)
        if datetime.now().timestamp() - cached_at > GEOLOCATION_CACHE_TTL:
            return None

        return cache.get("data")
    except Exception:
        return None


def _save_geolocation_cache(data: dict):
    """Save geolocation to cache."""
    try:
        GEOLOCATION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "cached_at": datetime.now().timestamp(),
            "data": data,
        }
        with open(GEOLOCATION_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def get_geolocation() -> GeoLocation:
    """
    Get geolocation from IP address using ipwho.is.

    Free, no API key required. Cached for 24 hours.
    """
    # Try cache first
    cached = _load_geolocation_cache()
    if cached:
        return GeoLocation(
            lat=cached.get("lat"),
            lon=cached.get("lon"),
            city=cached.get("city"),
            country=cached.get("country"),
            timezone=cached.get("timezone"),
        )

    try:
        url = "https://ipwho.is/?fields=success,latitude,longitude,city,country,timezone.abbr"
        with urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))

            if not data.get("success", False):
                return GeoLocation()

            # Get timezone abbreviation (CET, CEST, etc.) instead of IANA ID
            tz_data = data.get("timezone", {})
            tz_abbr = tz_data.get("abbr") if isinstance(tz_data, dict) else None

            # Cache the result
            cache_data = {
                "lat": data.get("latitude"),
                "lon": data.get("longitude"),
                "city": data.get("city"),
                "country": data.get("country"),
                "timezone": tz_abbr,
            }
            _save_geolocation_cache(cache_data)

            return GeoLocation(
                lat=cache_data["lat"],
                lon=cache_data["lon"],
                city=cache_data["city"],
                country=cache_data["country"],
                timezone=cache_data["timezone"],
            )
    except Exception:
        return GeoLocation()


def load_config() -> dict:
    """Load configuration from file, falling back to IP geolocation."""
    config = {}

    # Try user config first
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
        except Exception:
            pass

    # If no lat/lon configured, use IP geolocation
    if "lat" not in config or "lon" not in config:
        geo = get_geolocation()
        if geo.lat is not None and geo.lon is not None:
            config["lat"] = geo.lat
            config["lon"] = geo.lon
            config["city"] = geo.city
            config["country"] = geo.country
            config["timezone"] = geo.timezone

    return config


def get_lux() -> float | None:
    """Read ambient light sensor value in lux."""
    try:
        result = subprocess.run(
            [str(ALS_BINARY)], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


def _load_weather_cache() -> dict | None:
    """Load weather data from cache if still valid."""
    if not WEATHER_CACHE_FILE.exists():
        return None

    try:
        with open(WEATHER_CACHE_FILE) as f:
            cache = json.load(f)

        # Check if cache is still valid
        cached_at = cache.get("cached_at", 0)
        if datetime.now().timestamp() - cached_at > WEATHER_CACHE_TTL:
            return None

        return cache.get("data")
    except Exception:
        return None


def _save_weather_cache(data: dict):
    """Save weather data to cache."""
    try:
        WEATHER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "cached_at": datetime.now().timestamp(),
            "data": data,
        }
        with open(WEATHER_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def get_weather() -> WeatherData:
    """Get weather data from Open-Meteo API (free, no API key required).

    Uses a 15-minute cache to avoid repeated API calls.
    """
    # Try cache first
    cached = _load_weather_cache()
    if cached:
        return WeatherData(
            temperature=cached.get("temperature"),
            weather_code=cached.get("weather_code"),
            condition=cached.get("condition"),
            is_day=cached.get("is_day", True),
            cloud_cover=cached.get("cloud_cover"),
            sunrise=datetime.fromisoformat(cached["sunrise"]) if cached.get("sunrise") else None,
            sunset=datetime.fromisoformat(cached["sunset"]) if cached.get("sunset") else None,
            pressure=cached.get("pressure"),
            uv_index=cached.get("uv_index"),
        )

    config = load_config()
    lat = config.get("lat")
    lon = config.get("lon")

    # No location available
    if lat is None or lon is None:
        return WeatherData()

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,weather_code,is_day,cloud_cover,surface_pressure"
            f"&daily=sunrise,sunset,uv_index_max"
            f"&timezone=auto"
        )
        with urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

            current = data.get("current", {})
            daily = data.get("daily", {})

            # Parse sunrise/sunset
            sunrise = None
            sunset = None
            if daily.get("sunrise"):
                sunrise = datetime.fromisoformat(daily["sunrise"][0])
            if daily.get("sunset"):
                sunset = datetime.fromisoformat(daily["sunset"][0])

            # Parse UV index (daily max)
            uv_index = None
            if daily.get("uv_index_max"):
                uv_index = daily["uv_index_max"][0]

            # Parse weather code
            weather_code = current.get("weather_code")
            condition = WEATHER_CODES.get(weather_code, "Unknown")

            # Save to cache
            cache_data = {
                "temperature": current.get("temperature_2m"),
                "weather_code": weather_code,
                "condition": condition,
                "is_day": current.get("is_day", 1) == 1,
                "cloud_cover": current.get("cloud_cover"),
                "sunrise": sunrise.isoformat() if sunrise else None,
                "sunset": sunset.isoformat() if sunset else None,
                "pressure": current.get("surface_pressure"),
                "uv_index": uv_index,
            }
            _save_weather_cache(cache_data)

            return WeatherData(
                temperature=current.get("temperature_2m"),
                weather_code=weather_code,
                condition=condition,
                is_day=current.get("is_day", 1) == 1,
                cloud_cover=current.get("cloud_cover"),
                sunrise=sunrise,
                sunset=sunset,
                pressure=current.get("surface_pressure"),
                uv_index=uv_index,
            )
    except Exception:
        return WeatherData()


def get_sun_times() -> tuple[datetime | None, datetime | None]:
    """Get sunrise and sunset times from Open-Meteo API."""
    weather = get_weather()
    return weather.sunrise, weather.sunset


def lux_to_condition(lux: float) -> tuple[str, int, int]:
    """
    Convert lux to condition description and recommended brightness range.

    Buckets aligned with history.get_lux_bucket():
        0-10:       moonlight (night, screen-only)
        10-50:      dim (candlelit, nightlight)
        50-200:     ambient (evening home, cozy)
        200-500:    office (typical workspace)
        500-2000:   bright (near window, well-lit)
        2000-10000: daylight (overcast outdoor)
        10000+:     sunlight (direct outdoor)
    """
    if lux < 10:
        return "moonlight", 0, 50
    elif lux < 50:
        return "dim", 20, 60
    elif lux < 200:
        return "ambient", 25, 80
    elif lux < 500:
        return "office", 40, 120
    elif lux < 2000:
        return "bright", 100, 200
    elif lux < 10000:
        return "daylight", 150, 240
    else:
        return "sunlight", 200, 255


def get_sun_phase(
    now: datetime, sunrise: datetime | None, sunset: datetime | None
) -> tuple[str, bool]:
    """Determine sun phase and whether to prefer dark themes."""
    if not sunrise or not sunset:
        return "unknown", False

    if now < sunrise:
        minutes_to_sunrise = (sunrise - now).seconds // 60
        if minutes_to_sunrise < 60:
            return f"predawn", True
        return "night", True
    elif now > sunset:
        minutes_since_sunset = (now - sunset).seconds // 60
        if minutes_since_sunset < 60:
            return "dusk", True
        return "night", True
    else:
        minutes_since_sunrise = (now - sunrise).seconds // 60
        minutes_to_sunset = (sunset - now).seconds // 60
        if minutes_since_sunrise < 60:
            return "golden hour", False
        elif minutes_to_sunset < 60:
            return "golden hour", False
        return "daytime", False


def get_system_appearance() -> str:
    """
    Get system dark/light mode setting.

    Returns:
        "dark", "light", or "unknown"
    """
    import platform

    system = platform.system()

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0 and "Dark" in result.stdout:
                return "dark"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "light"

    elif system == "Linux":
        try:
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if "dark" in result.stdout.lower():
                return "dark"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "light"

    return "unknown"


def get_power_source() -> str:
    """
    Get power source and battery level.

    Returns:
        "ac", "battery_high", "battery_low", or "unknown"
    """
    import platform
    import re

    system = platform.system()

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            output = result.stdout
            if "AC Power" in output:
                return "ac"
            # Parse battery percentage
            match = re.search(r"(\d+)%", output)
            if match:
                pct = int(match.group(1))
                return "battery_low" if pct < 20 else "battery_high"
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

    elif system == "Linux":
        try:
            status_path = Path("/sys/class/power_supply/BAT0/status")
            capacity_path = Path("/sys/class/power_supply/BAT0/capacity")
            if status_path.exists():
                status = status_path.read_text().strip()
                if status == "Charging":
                    return "ac"
                if capacity_path.exists():
                    capacity = int(capacity_path.read_text().strip())
                    return "battery_low" if capacity < 20 else "battery_high"
        except (ValueError, OSError):
            pass

    return "unknown"
