"""
Weather tool: HTTP client for OpenWeatherMap. Input: city, country (validated/sanitized).
Uses /weather for current conditions and /forecast for accurate daily min/max.
Retry/backoff on rate-limit; friendly messages on network errors.
"""
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.base import WeatherResult

logger = logging.getLogger(__name__)

OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
MAX_RETRIES = 3
BACKOFF_SEC = 2.0


def _sanitize_location(text: Optional[str]) -> str:
    """Allow only letters, spaces, hyphens; max length 100."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = re.sub(r"[^\w\s\-]", "", text.strip())[:100]
    return cleaned.strip() or ""


def _get_daily_minmax(forecast_data: dict, target_date: Optional[str] = None) -> dict[str, Optional[float]]:
    """
    Extract accurate daily min/max from /forecast 3-hour blocks.
    target_date: 'YYYY-MM-DD' string. If None, uses today (based on forecast timezone).
    Returns {"temp_min": float, "temp_max": float} or Nones.
    """
    forecasts = forecast_data.get("list", [])
    if not forecasts:
        return {"temp_min": None, "temp_max": None}

    tz_offset = forecast_data.get("city", {}).get("timezone", 0)

    if target_date is None:
        now_utc = datetime.now(timezone.utc)
        local_ts = now_utc.timestamp() + tz_offset
        target_date = datetime.fromtimestamp(local_ts, tz=timezone.utc).strftime("%Y-%m-%d")

    temps = []
    for entry in forecasts:
        dt_utc = entry.get("dt", 0)
        local_ts = dt_utc + tz_offset
        entry_date = datetime.fromtimestamp(local_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if entry_date == target_date:
            main = entry.get("main", {})
            temps.append(main.get("temp_min", main.get("temp")))
            temps.append(main.get("temp_max", main.get("temp")))

    temps = [t for t in temps if t is not None]
    if not temps:
        return {"temp_min": None, "temp_max": None}

    return {"temp_min": round(min(temps), 1), "temp_max": round(max(temps), 1)}


def _http_get(url: str, params: dict) -> Optional[dict]:
    """HTTP GET with retry/backoff. Returns parsed JSON or None."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(url, params=params)
            if r.status_code == 429:
                last_error = "Rate limit reached. Please try again later."
                time.sleep(BACKOFF_SEC * (attempt + 1))
                continue
            if r.status_code != 200:
                try:
                    body = r.json()
                    msg = body.get("message", r.text)
                except Exception:
                    msg = r.text
                logger.warning("Weather API %s error: %s", url, msg)
                return None
            return r.json()
        except httpx.TimeoutException as e:
            last_error = "Request timed out."
            logger.warning("Weather API timeout: %s", e)
        except httpx.RequestError as e:
            last_error = "Network error."
            logger.warning("Weather API request error: %s", e)
        time.sleep(BACKOFF_SEC * (attempt + 1))
    logger.error("Weather API exhausted retries: %s", last_error)
    return None


class WeatherInput(BaseModel):
    """Structured input for weather: city and optional country."""
    city: str = Field(description="City name")
    country: Optional[str] = Field(default=None, description="Country code or name (optional)")


def get_weather_impl(
    city: str,
    country: Optional[str] = None,
    api_key: Optional[str] = None,
) -> WeatherResult:
    """
    Fetch current weather + forecast for accurate daily min/max.
    Uses lang=pt_br for Portuguese descriptions from the API.
    """
    api_key = api_key or __import__("os").environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return WeatherResult(summary="API do clima não configurada. Defina OPENWEATHERMAP_API_KEY no .env.")
    city = _sanitize_location(city)
    country = _sanitize_location(country) if country else ""
    if not city:
        return WeatherResult(summary="Por favor, forneça um nome de cidade válido.")

    q = f"{city},{country}" if country else city
    base_params = {"q": q, "appid": api_key, "units": "metric", "lang": "pt_br"}

    current = _http_get(OPENWEATHER_CURRENT_URL, base_params)
    if not current:
        return WeatherResult(summary="Não foi possível obter dados do clima. Tente novamente.")

    forecast = _http_get(OPENWEATHER_FORECAST_URL, base_params)

    daily = {"temp_min": None, "temp_max": None}
    if forecast:
        daily = _get_daily_minmax(forecast)

    current.setdefault("daily", {})
    current["daily"]["temp_min"] = daily["temp_min"]
    current["daily"]["temp_max"] = daily["temp_max"]

    temp = current.get("main", {}).get("temp")
    desc = (current.get("weather") or [{}])[0].get("description", "")
    name = current.get("name", city)
    summary = f"Em {name}: {desc}, {temp}°C."

    return WeatherResult(summary=summary, raw_data=current)


@tool(args_schema=WeatherInput)
def weather_api(city: str, country: Optional[str] = None) -> str:
    """
    Get current weather for a city. Use when the user asks about weather, temperature, or climate.
    Input: city (required), country (optional). Returns a concise natural language summary.
    """
    result = get_weather_impl(city, country)

    if result.raw_data:
        main = result.raw_data.get("main", {})
        daily = result.raw_data.get("daily", {})
        temp = main.get("temp", "N/A")
        desc = (result.raw_data.get("weather") or [{}])[0].get("description", "").title()
        feels_like = main.get("feels_like")
        humidity = main.get("humidity")
        wind_speed = result.raw_data.get("wind", {}).get("speed")
        temp_min = daily.get("temp_min")
        temp_max = daily.get("temp_max")

        parts = [f"Em {city}: {desc}, {temp}°C"]
        if temp_min is not None and temp_max is not None:
            parts.append(f"(mín {temp_min}°C / máx {temp_max}°C)")
        if feels_like:
            parts.append(f"Sensação térmica: {feels_like}°C")
        if humidity:
            parts.append(f"Umidade: {humidity}%")
        if wind_speed:
            parts.append(f"Vento: {wind_speed} m/s")

        return ". ".join(parts) + "."

    return result.summary


def get_weather_tool():
    return weather_api
