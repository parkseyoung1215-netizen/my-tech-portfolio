"""브리핑: 날씨 + 일정 + 할 일 + 관심 종목 + 경제 헤드라인"""
import datetime
import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request

from config import *
from jarvis.core import S, client, run_applescript, ssl_context, home_now, load_state, save_state, \
    press_media_key, get_volume, set_muted, is_muted
from jarvis.registry import register, execute, _tool, _s, _i
from jarvis.voice import speak
from skills.calendar import get_reminders
from skills.news import get_news
from skills.market import get_quotes
from skills.memory import memory_prompt, watchlist_symbols

WEATHER_CODES = {0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
                 45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
                 61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
                 75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "heavy showers",
                 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def get_weather() -> str:
    """오늘 날씨 (Open-Meteo: 무료, 키 필요 없음)."""
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={HOME_LAT}&longitude={HOME_LON}"
           "&current=temperature_2m,weather_code"
           "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
           "&timezone=auto&forecast_days=1")
    try:
        with urllib.request.urlopen(url, timeout=10, context=ssl_context()) as response:
            data = json.loads(response.read())
        now = data["current"]
        day = data["daily"]
        return (f"{HOME_CITY}: now {now['temperature_2m']:.0f}°C, "
                f"{WEATHER_CODES.get(now['weather_code'], 'mixed weather')}; "
                f"today {day['temperature_2m_min'][0]:.0f}~{day['temperature_2m_max'][0]:.0f}°C, "
                f"chance of rain {day['precipitation_probability_max'][0]}%")
    except Exception as e:
        print(f"(날씨를 못 가져왔어요: {e})")
        return "weather unavailable"


def gather_briefing() -> str:
    """브리핑 재료 모으기."""
    events = execute("today_events", {})
    events = events if isinstance(events, str) else "(couldn't read the calendar)"
    reminders = get_reminders() or "(none)"
    now = home_now()
    text = (f"Local time: {now.strftime('%A %H:%M')}\n"
            f"Weather: {get_weather()}\n"
            f"Today's events:\n{events}\n"
            f"Open reminders:\n{reminders}")
    memories = memory_prompt()
    text += ("\nWhat you remember about the user (mention only what matters today, e.g. a deadline "
             "today or tomorrow):\n" + memories)
    symbols = watchlist_symbols()
    if symbols:
        try:
            text += "\nUser's watchlist (mention briefly, especially big moves):\n" + get_quotes(symbols)
        except Exception:
            pass
    if BRIEFING_NEWS:
        try:
            text += "\nTop business headlines (mention only the 1-2 most important, briefly):\n" + \
                    get_news("global", "business", "")
        except Exception:
            pass
    return text


def compose_briefing(data: str) -> str:
    """재료를 자비스 말투의 짧은 브리핑으로."""
    response = client.chat.completions.create(
        model=BRAIN_MODEL, temperature=0.5,
        messages=[
            {"role": "system", "content":
             "You are Jarvis, Tony Stark's calm, capable, slightly witty AI butler. "
             "Give a spoken morning briefing in 2-4 short English sentences: greeting with 'sir' "
             "fitting the local time, weather, today's events (time and name, keep names as written), "
             "and open reminders if any. Mention an umbrella if rain chance is high. "
             "Do not invent anything. No questions at the end."},
            {"role": "user", "content": data},
        ])
    return response.choices[0].message.content.strip()


def maybe_morning_briefing() -> None:
    """아침 시간대에 오늘 아직 브리핑을 안 했으면 먼저 브리핑."""
    if not BRIEFING:
        return
    now = home_now()
    today = now.strftime("%Y-%m-%d")
    state = load_state()
    if not (MORNING_HOURS[0] <= now.hour < MORNING_HOURS[1]) or state.get("last_briefing") == today:
        return
    state["last_briefing"] = today
    save_state(state)
    try:
        print("(아침 브리핑)")
        speak(compose_briefing(gather_briefing()))
    except Exception as e:
        print(f"(브리핑 실패: {e})")


def handle_briefing(args):
    return gather_briefing()

# briefing: 브리핑: 날씨 + 오늘 일정 + 할 일 (읽기만 함)
register(_tool("briefing", "날씨·오늘 일정·할 일을 모아서 가져오기. '브리핑해줘', '오늘 어때?', '날씨 어때?' 같은 말에. 결과를 보고 자비스처럼 짧게 브리핑한다."),
         'auto', handle_briefing)
