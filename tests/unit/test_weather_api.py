"""Unit tests for weather_api: mock HTTP; success, rate-limit, network error, forecast."""
import pytest
from unittest.mock import patch, MagicMock, call
import httpx

from tools.weather_api import get_weather_impl, _sanitize_location, _get_daily_minmax


def test_sanitize_location():
    assert _sanitize_location("  São Paulo  ") == "São Paulo"
    assert _sanitize_location("") == ""
    assert _sanitize_location(None) == ""


@patch("tools.weather_api._http_get")
def test_weather_success_with_forecast(mock_http):
    current_data = {
        "main": {"temp": 22.5, "feels_like": 23, "humidity": 60},
        "weather": [{"description": "céu limpo"}],
        "name": "London",
        "wind": {"speed": 3.0},
    }
    forecast_data = {
        "city": {"timezone": 0},
        "list": [
            {"dt": 1700000000, "main": {"temp": 18, "temp_min": 15, "temp_max": 20}},
            {"dt": 1700010800, "main": {"temp": 25, "temp_min": 23, "temp_max": 28}},
        ],
    }

    mock_http.side_effect = [current_data, forecast_data]

    result = get_weather_impl("London", api_key="fake")
    assert "London" in result.summary
    assert result.raw_data is not None
    assert result.raw_data.get("daily") is not None


@patch("tools.weather_api._http_get")
def test_weather_forecast_failure_still_works(mock_http):
    """If forecast call fails, current weather still returned (no daily min/max)."""
    current_data = {
        "main": {"temp": 22.5},
        "weather": [{"description": "nublado"}],
        "name": "Paris",
    }
    mock_http.side_effect = [current_data, None]

    result = get_weather_impl("Paris", api_key="fake")
    assert "Paris" in result.summary
    assert result.raw_data["daily"]["temp_min"] is None
    assert result.raw_data["daily"]["temp_max"] is None


def test_weather_not_configured():
    result = get_weather_impl("Paris", api_key=None)
    assert "não configurada" in result.summary.lower() or "not configured" in result.summary.lower()


@patch("tools.weather_api._http_get")
def test_weather_current_fails(mock_http):
    mock_http.return_value = None
    result = get_weather_impl("Berlin", api_key="fake")
    assert "não foi possível" in result.summary.lower() or "tente novamente" in result.summary.lower()


@patch("tools.weather_api._http_get")
def test_weather_uses_lang_ptbr(mock_http):
    """Verify that API calls use lang=pt_br parameter."""
    current_data = {
        "main": {"temp": 20},
        "weather": [{"description": "céu limpo"}],
        "name": "Rome",
    }
    mock_http.side_effect = [current_data, {"city": {"timezone": 0}, "list": []}]

    get_weather_impl("Rome", api_key="fake")

    calls = mock_http.call_args_list
    assert len(calls) == 2
    for c in calls:
        params = c[0][1]
        assert params["lang"] == "pt_br"
