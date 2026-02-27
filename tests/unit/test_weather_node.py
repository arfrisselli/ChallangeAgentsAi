"""
Unit tests for weather_node, _extract_city, _translate_weather_desc.
All responses are always in PT-BR. Min/max from forecast (daily).
"""
import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage

from graph.nodes import weather_node, _extract_city, _translate_weather_desc
from tools.base import WeatherResult


class TestExtractCity:
    """Test the _extract_city helper that parses city names from queries."""

    def test_portuguese_clima_em(self):
        assert _extract_city("Como será o clima em Londrina?") == ("Londrina", None)

    def test_portuguese_tempo_em(self):
        assert _extract_city("Qual o tempo em São Paulo?") == ("São Paulo", None)

    def test_english_weather_in(self):
        assert _extract_city("What's the weather in London?") == ("London", None)

    def test_english_temperature_in(self):
        assert _extract_city("Temperature in New York?") == ("New York", None)

    def test_portuguese_clima_de(self):
        city, _ = _extract_city("Previsão do clima de Curitiba")
        assert city == "Curitiba"

    def test_city_with_slash_country(self):
        city, country = _extract_city("Clima em Londrina/PR")
        assert city == "Londrina"
        assert country == "PR"

    def test_city_with_comma_country(self):
        city, country = _extract_city("Weather in Paris, France")
        assert city == "Paris"
        assert country == "France"

    def test_fallback_capitalized_word(self):
        city, _ = _extract_city("Clima Berlim")
        assert city == "Berlim"

    def test_skips_common_pt_words(self):
        city, country = _extract_city("Como está o clima?")
        assert city is None

    def test_skips_common_en_words(self):
        city, country = _extract_city("What is the weather?")
        assert city is None


class TestTranslateWeatherDesc:
    """Test PT-BR translation of OpenWeatherMap descriptions."""

    def test_clear_sky(self):
        assert _translate_weather_desc("clear sky") == "Céu limpo"

    def test_few_clouds(self):
        assert _translate_weather_desc("few clouds") == "Poucas nuvens"

    def test_light_rain(self):
        assert _translate_weather_desc("light rain") == "Chuva leve"

    def test_case_insensitive(self):
        assert _translate_weather_desc("Clear Sky") == "Céu limpo"

    def test_unknown_fallback_to_title(self):
        assert _translate_weather_desc("very unusual weather") == "Very Unusual Weather"

    def test_partial_match(self):
        assert _translate_weather_desc("heavy thunderstorm") == "Tempestade"

    def test_already_portuguese(self):
        """When API returns PT-BR via lang=pt_br, _translate title-cases it."""
        assert _translate_weather_desc("céu limpo") == "Céu Limpo"


class TestGetDailyMinmax:
    """Test daily min/max extraction from forecast data."""

    def test_extracts_minmax_from_forecast_entries(self):
        from tools.weather_api import _get_daily_minmax

        forecast = {
            "city": {"timezone": 0},
            "list": [
                {"dt": 1700006400, "main": {"temp": 22, "temp_min": 18, "temp_max": 24}},
                {"dt": 1700017200, "main": {"temp": 28, "temp_min": 26, "temp_max": 31}},
                {"dt": 1700028000, "main": {"temp": 19, "temp_min": 15, "temp_max": 21}},
            ],
        }
        from datetime import datetime, timezone as tz
        target = datetime.fromtimestamp(1700006400, tz=tz.utc).strftime("%Y-%m-%d")

        result = _get_daily_minmax(forecast, target_date=target)
        assert result["temp_min"] == 15.0
        assert result["temp_max"] == 31.0

    def test_empty_forecast(self):
        from tools.weather_api import _get_daily_minmax

        result = _get_daily_minmax({"list": []})
        assert result["temp_min"] is None
        assert result["temp_max"] is None

    def test_no_matching_date(self):
        from tools.weather_api import _get_daily_minmax

        forecast = {
            "city": {"timezone": 0},
            "list": [
                {"dt": 1700000000, "main": {"temp": 22, "temp_min": 20, "temp_max": 24}},
            ],
        }
        result = _get_daily_minmax(forecast, target_date="2099-01-01")
        assert result["temp_min"] is None
        assert result["temp_max"] is None


class TestWeatherNode:
    """Test the weather_node - always responds in PT-BR with accurate daily min/max."""

    def _make_raw_data(self, **overrides):
        base = {
            "main": {"temp": 25, "feels_like": 27, "humidity": 70},
            "weather": [{"description": "céu limpo"}],
            "name": "São Paulo",
            "wind": {"speed": 3.5},
            "daily": {"temp_min": 18.2, "temp_max": 31.5},
        }
        base.update(overrides)
        return base

    def test_full_flow_ptbr_with_daily_minmax(self):
        state = {"messages": [HumanMessage(content="Como será o clima em São Paulo?")]}

        with patch("tools.weather_api.get_weather_impl") as mock:
            mock.return_value = WeatherResult(summary="Test", raw_data=self._make_raw_data())
            result = weather_node(state)

            mock.assert_called_once()
            content = result["messages"][0].content
            assert "São Paulo" in content
            assert "mín 18.2°C" in content
            assert "máx 31.5°C" in content
            assert "Sensação térmica" in content
            assert "Umidade" in content
            assert "Vento" in content

    def test_ptbr_description_from_api(self):
        """When API returns PT-BR description via lang=pt_br, it should be used."""
        state = {"messages": [HumanMessage(content="clima em Curitiba")]}

        with patch("tools.weather_api.get_weather_impl") as mock:
            mock.return_value = WeatherResult(
                summary="Test",
                raw_data={
                    "main": {"temp": 10, "feels_like": 8, "humidity": 90},
                    "weather": [{"description": "chuva leve"}],
                    "name": "Curitiba",
                    "wind": {"speed": 2.0},
                    "daily": {"temp_min": 5.3, "temp_max": 14.8},
                },
            )
            result = weather_node(state)
            content = result["messages"][0].content
            assert "Chuva leve" in content or "Chuva Leve" in content
            assert "mín 5.3°C" in content
            assert "máx 14.8°C" in content

    def test_english_query_still_ptbr_response(self):
        state = {"messages": [HumanMessage(content="What's the weather in London?")]}

        with patch("tools.weather_api.get_weather_impl") as mock:
            mock.return_value = WeatherResult(
                summary="Test",
                raw_data={
                    "main": {"temp": 15, "feels_like": 13, "humidity": 80},
                    "weather": [{"description": "nublado"}],
                    "name": "London",
                    "wind": {"speed": 5.0},
                    "daily": {"temp_min": 10.0, "temp_max": 17.0},
                },
            )
            result = weather_node(state)
            content = result["messages"][0].content
            assert "London" in content
            assert "Sensação térmica" in content
            assert "mín 10.0°C" in content

    def test_no_city_returns_error_ptbr(self):
        state = {"messages": [HumanMessage(content="Como está o clima?")]}

        with patch("tools.weather_api.get_weather_impl") as mock:
            result = weather_node(state)
            mock.assert_not_called()
            assert "não consegui identificar a cidade" in result["messages"][0].content.lower()

    def test_no_city_en_query_still_ptbr_error(self):
        state = {"messages": [HumanMessage(content="What is the weather?")]}

        with patch("tools.weather_api.get_weather_impl") as mock:
            result = weather_node(state)
            mock.assert_not_called()
            assert "não consegui identificar a cidade" in result["messages"][0].content.lower()

    def test_no_daily_data_omits_minmax(self):
        """If forecast fails, daily min/max should be omitted (not crash)."""
        state = {"messages": [HumanMessage(content="clima em Rio")]}

        with patch("tools.weather_api.get_weather_impl") as mock:
            mock.return_value = WeatherResult(
                summary="Test",
                raw_data={
                    "main": {"temp": 30, "feels_like": 32, "humidity": 65},
                    "weather": [{"description": "ensolarado"}],
                    "name": "Rio de Janeiro",
                    "wind": {"speed": 2.0},
                    "daily": {"temp_min": None, "temp_max": None},
                },
            )
            result = weather_node(state)
            content = result["messages"][0].content
            assert "Rio de Janeiro" in content
            assert "mín" not in content
            assert "máx" not in content

    def test_response_includes_emoji(self):
        state = {"messages": [HumanMessage(content="clima em Rio")]}

        with patch("tools.weather_api.get_weather_impl") as mock:
            mock.return_value = WeatherResult(
                summary="Test",
                raw_data={
                    "main": {"temp": 30},
                    "weather": [{"description": "ensolarado"}],
                    "name": "Rio de Janeiro",
                    "daily": {"temp_min": 25.0, "temp_max": 35.0},
                },
            )
            result = weather_node(state)
            assert "🌤️" in result["messages"][0].content

    def test_api_error_returns_summary(self):
        state = {"messages": [HumanMessage(content="Weather in InvalidCity")]}

        with patch("tools.weather_api.get_weather_impl") as mock:
            mock.return_value = WeatherResult(summary="Cidade não encontrada", raw_data=None)
            result = weather_node(state)
            assert "Cidade não encontrada" in result["messages"][0].content

    def test_no_llm_call(self):
        state = {"messages": [HumanMessage(content="Weather in Paris")]}

        with patch("tools.weather_api.get_weather_impl") as mock_weather:
            mock_weather.return_value = WeatherResult(
                summary="Test",
                raw_data={
                    "main": {"temp": 18},
                    "weather": [{"description": "chuva"}],
                    "name": "Paris",
                    "daily": {"temp_min": 14.0, "temp_max": 20.0},
                },
            )
            with patch("graph.nodes._get_llm") as mock_llm:
                weather_node(state)
                mock_llm.assert_not_called()
                mock_weather.assert_called_once()
