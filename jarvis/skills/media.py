"""음악·영상: 유튜브, 스포티파이, 지금 나오는 소리 제어"""
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
import base64

spotify_token = {"value": "", "expires": 0.0}


def first_youtube_video(query: str) -> str:
    """유튜브 검색 결과의 첫 번째 영상 주소. 못 찾으면 빈 글자."""
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
    # 유럽(독일)에서는 쿠키 동의 화면이 먼저 나와서 영상 목록이 안 보임 -> 동의 쿠키를 같이 보냄
    request = urllib.request.Request(url, headers={
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "ko,en;q=0.8",
        "Cookie": "SOCS=CAI; CONSENT=YES+1",
    })
    try:
        with urllib.request.urlopen(request, timeout=10, context=ssl_context()) as response:
            page = response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"(유튜브 검색 실패: {e})")
        return ""
    # 광고 영상을 건너뛰고 일반 검색 결과(videoRenderer)의 첫 영상만 고름
    match = (re.search(r'"videoRenderer":\{"videoId":"([\w-]{11})"', page)
             or re.search(r'"videoId":"([\w-]{11})"', page))
    if not match:
        print("(유튜브 결과에서 영상을 못 찾아서 검색 페이지를 엽니다)")
        return ""
    return f"https://www.youtube.com/watch?v={match.group(1)}&autoplay=1"


def music_app_playing() -> bool:
    """맥 음악 앱이 켜져 있고 재생 중인지 (꺼져 있으면 새로 켜지 않음)."""
    result = run_applescript([
        'if application "Music" is running then',
        'tell application "Music" to return (player state is playing) as text',
        'end if',
        'return "false"'])
    return "true" in result.stdout


def spotify_play_uri(uri: str) -> bool:
    """스포티파이 앱에서 주소(곡/가수/앨범/플레이리스트)를 틀고, 실제로 재생되는지 확인."""
    result = run_applescript([
        'on run argv',
        'tell application "Spotify"',
        'play track (item 1 of argv)',
        'delay 1.5',
        'if player state is not playing then play',
        'delay 0.5',
        'return (player state is playing) as text',
        'end tell',
        'end run'], [uri])
    if result.returncode != 0:
        print(f"(스포티파이 재생 실패: {result.stderr.strip()})")
        return False
    return "true" in result.stdout


def spotify_search(query: str, kind: str) -> str:
    """스포티파이에서 검색해서 첫 결과의 주소(spotify:track:...)를 돌려줌.
    .env에 SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET이 있어야 함. 없거나 실패하면 빈 글자."""
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("(.env에 SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET이 없어서 검색 화면만 엽니다)")
        return ""
    try:
        now = datetime.datetime.now().timestamp()
        if not spotify_token["value"] or now >= spotify_token["expires"]:
            auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            request = urllib.request.Request(
                "https://accounts.spotify.com/api/token",
                data=b"grant_type=client_credentials",
                headers={"Authorization": f"Basic {auth}",
                         "Content-Type": "application/x-www-form-urlencoded"},
                method="POST")
            with urllib.request.urlopen(request, timeout=10, context=ssl_context()) as response:
                data = json.loads(response.read())
            spotify_token["value"] = data["access_token"]
            spotify_token["expires"] = now + data.get("expires_in", 3600) - 60
        url = ("https://api.spotify.com/v1/search?limit=1&type=" + kind
               + "&q=" + urllib.parse.quote(query))
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {spotify_token['value']}"})
        with urllib.request.urlopen(request, timeout=10, context=ssl_context()) as response:
            data = json.loads(response.read())
        items = data.get(kind + "s", {}).get("items") or []
        items = [i for i in items if i]
        if not items:
            print("(스포티파이에서 결과를 못 찾았어요)")
            return ""
        print(f"(스포티파이: {items[0].get('name')})")
        return items[0]["uri"]
    except Exception as e:
        print(f"(스포티파이 검색 실패: {e})")
        return ""


def handle_youtube_play(args):
    query = str(args.get("query", "")).strip()
    if not query:
        return False
    url = first_youtube_video(query)
    if not url:  # 영상을 못 찾으면 검색 결과라도 열기
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
    S.last_opened_url = url
    return subprocess.run(["open", url]).returncode == 0


def handle_youtube_search(args):
    query = str(args.get("query", "")).strip()
    if not query:
        return False
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
    S.last_opened_url = url
    return subprocess.run(["open", url]).returncode == 0


def handle_spotify_play(args):
    query = str(args.get("query", "")).strip()
    kind = str(args.get("type", "track")).strip().lower()
    if kind == "liked":
        user = os.environ.get("SPOTIFY_USERNAME", SPOTIFY_USERNAME)
        for uri in (f"spotify:user:{user}:collection", "spotify:collection:tracks"):
            if spotify_play_uri(uri):
                print("(스포티파이: 좋아요 표시한 곡)")
                return True
        print("(좋아요 표시한 곡을 틀지 못했어요)")
        return False
    if kind not in ("track", "artist", "album", "playlist"):
        kind = "track"
    if not query:
        return run_applescript(['tell application "Spotify" to play']).returncode == 0
    uri = spotify_search(query, kind)
    if not uri:
        # 키가 없거나 못 찾으면: 스포티파이 앱에서 검색 화면이라도 열기
        subprocess.run(["open", "spotify:search:" + urllib.parse.quote(query)])
        return True
    return spotify_play_uri(uri)


def handle_music(args):
    command = str(args.get("command", "play")).strip().lower()
    query = str(args.get("query", "")).strip()
    if command == "play" and query:
        result = run_applescript([
            'on run argv',
            'tell application "Music"',
            'set found to (every track of playlist "Library" whose name contains (item 1 of argv) or artist contains (item 1 of argv))',
            'if (count of found) is 0 then return "none"',
            'play item 1 of found',
            'end tell',
            'return "ok"',
            'end run'], [query])
        return result.returncode == 0 and "ok" in result.stdout
    if command not in ("play", "pause", "next", "previous"):
        return False
    # 맥 음악 앱이 재생 중이면 음악 앱을 직접 제어
    if command == "pause" and music_app_playing():
        return run_applescript(['tell application "Music" to pause']).returncode == 0
    # 그 외(유튜브, 스포티파이 등 지금 소리 나는 것)는 키보드의 재생/정지 키를 누른 것처럼 처리
    key = {"play": "playpause", "pause": "playpause",
           "next": "next", "previous": "previous"}[command]
    if press_media_key(key):
        return True
    # 미디어 키를 못 쓰면 음악 앱이라도 제어
    scripts = {"play": "play", "pause": "pause", "next": "next track",
               "previous": "previous track"}
    return run_applescript([f'tell application "Music" to {scripts[command]}']).returncode == 0


# youtube_play: 유튜브 첫 번째 영상 바로 틀기
register(_tool("youtube_play", "유튜브에서 찾아서 첫 번째 영상을 바로 재생", {"query": _s("검색어. 사용자가 말한 그대로")}, ["query"]),
         'auto', handle_youtube_play)

# youtube_search: 유튜브 검색 결과 열기
register(_tool("youtube_search", "유튜브 검색 결과 목록만 열기", {"query": _s("검색어")}, ["query"]),
         'auto', handle_youtube_search)

# spotify_play: 스포티파이에서 곡/가수/앨범/플레이리스트 찾아 틀기
register(_tool("spotify_play", "스포티파이에서 찾아서 재생. 사용자는 음악을 주로 스포티파이로 듣는다.",
          {"query": _s("곡/가수/앨범/플레이리스트 이름. type이 liked면 빈 문자열"),
           "type": {"type": "string", "enum": ["track", "artist", "album", "playlist", "liked"],
                    "description": "liked = 좋아요 표시한 곡"}}, ["query", "type"]),
         'auto', handle_spotify_play)

# music: 지금 나오는 소리 재생/정지/다음/이전, 맥 음악 앱에서 곡 찾기
register(_tool("music", "지금 나오는 소리(유튜브·스포티파이·음악 앱 무엇이든) 재생/정지/다음/이전",
          {"command": {"type": "string", "enum": ["play", "pause", "next", "previous"]},
           "query": _s("보통 빈 문자열. 사용자가 '음악 앱'에서 특정 곡을 틀라고 할 때만")}, ["command", "query"]),
         'auto', handle_music)
