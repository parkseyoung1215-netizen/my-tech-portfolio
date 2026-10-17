"""
자비스 설정 파일
조절하고 싶은 값은 대부분 여기에 있어요. 바꾸고 나서 자비스를 다시 켜면 적용돼요.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # assistant 폴더

# ---- 이름 ----------------------------------------------------------
ASSISTANT_NAME = "자비스"

# ---- 목소리 ---------------------------------------------------------
VOICE_ENGINE = "mac"
ELEVEN_VOICE_ID = "여기에_보이스_ID"   # ElevenLabs에서 고른 목소리의 Voice ID
ELEVEN_MODEL = "eleven_multilingual_v2"  # 한국어 지원. 더 빠르게는 "eleven_flash_v2_5"
OPENAI_VOICE = "onyx"   # 후보: onyx(낮은 남성), ash, echo, fable, sage, alloy, nova, shimmer, coral
VOICE_STYLE = ("실제 사람 비서가 옆에서 말하듯 자연스럽게 말한다. 차분하고 따뜻한 톤, "
               "편안한 속도, 문장 사이에 자연스러운 짧은 쉼. 기계처럼 단어마다 끊거나 "
               "책 읽듯 딱딱하게 말하지 않는다. 한국어 억양을 자연스럽게.")
MAC_VOICE = "Ava"         # 영어 목소리 (Premium/Enhanced 버전이 있으면 자동으로 그걸 씀)
KOREAN_VOICE = "Yuna"     # 한국어 대답일 때 쓸 목소리 (Premium 버전 자동 선택)

# ---- 호출어 ("헤이 자비스") ----------------------------------------------
WAKE_THRESHOLD = 0.2   # 0~1. 잘 안 깨면 낮추고, 엉뚱하게 깨면 높이기
SHOW_WAKE_SCORE = True # 부를 때 점수를 화면에 보여줌 (기준값 맞추는 용도)
WAKE_SOFT = 0.15       # 이 점수를 연달아 넘으면(길게 부른 경우) 기준 미달이어도 깨어남
WAKE_SOFT_FRAMES = 3   # 몇 번 연달아 넘어야 하는지
MUTE_WHILE_LISTENING = True  # 깨어나서 듣는 동안 맥 소리를 완전히 끔 (음악이 같이 녹음되는 것 방지)
WAKE_GAIN = 3.0        # 호출어 들을 때 소리를 몇 배로 키울지. 작게 말해도 안 깨면 높이고, 잡음에 깨면 낮추기
WAKE_FRAME = 1280      # 80ms 단위로 듣기 (16kHz 기준)

# ---- 목소리 잠금 ------------------------------------------------------
VOICE_LOCK = True
VERIFIER_PATH = os.path.join(BASE_DIR, "voice", "verifier.pkl")
VERIFY_FROM = 0.1      # 호출어 점수가 이 이상이면 목소리 검사를 함

# ---- 듣기 ----------------------------------------------------------
TRANSCRIBE_MODEL = "gpt-4o-transcribe"
TRANSCRIBE_HINT = "개인 비서에게 하는 짧은 명령입니다."  # 단어 목록을 넣으면 그 단어로 잘못 알아듣는 경우가 많아서 문장 하나만
SAMPLE_RATE = 16000
RECORD_SECONDS = 5       # (Enter 방식 안내 문구용)
START_WAIT = 4.0         # 깨운 뒤 이 시간 안에 말을 시작 안 하면 취소
END_SILENCE = 0.9        # 말이 끝나고 이만큼 조용하면 듣기 종료
MAX_LISTEN = 12.0        # 한 번에 최대 듣는 시간
MIN_SPEECH_LEVEL = 300   # 말소리로 볼 최소 크기 (주변 소음에 맞춰 자동으로 올라감)
MAX_SPEECH_LEVEL = 700   # 음악 때문에 소음 기준이 너무 올라가도 이 이상은 안 올림
FOLLOW_UP = True         # 대답한 뒤 몇 초 동안 호출어 없이 바로 이어서 말할 수 있게
FOLLOW_UP_WAIT = 4.0     # 이어 말하기를 기다리는 시간
CONFIRM_SECONDS = 4      # "Shall I go ahead?" 뒤에 대답 시작을 기다리는 시간

# ---- 두뇌 ----------------------------------------------------------
BRAIN_MODEL = "gpt-4.1-mini"
MAX_STEPS = 6            # 한 번 명령에 도구를 최대 몇 번까지 이어서 쓸지

# ---- 대화 기억 (짧은 기억) -----------------------------------------------
MEMORY_TURNS = 8         # 최근 몇 번의 대화를 기억할지
MEMORY_MINUTES = 15      # 이 시간 동안 말이 없으면 기억을 비움 (새 대화로 시작)

# ---- 오래 기억하기 -----------------------------------------------------
MEMORY_FILE = os.path.join(BASE_DIR, "jarvis_memory.json")
MEMORY_LIMIT = 60        # 최대 기억 개수 (너무 많으면 오래된 것부터 정리 필요)

# ---- 종료 ----------------------------------------------------------
QUIT_WORDS = ("종료", "그만", "꺼져", "끝", "stop", "quit", "exit", "goodbye", "shutdown")
QUIT_EXITS_PROGRAM = False

# ---- 아침 브리핑 ------------------------------------------------------
BRIEFING = True
BRIEFING_NEWS = True             # 아침 브리핑에 경제 헤드라인도 1~2개
HOME_CITY = "Bremen"
HOME_LAT, HOME_LON = 53.08, 8.80
HOME_TZ = "Europe/Berlin"        # 노트북 시계가 다른 나라로 돼 있어도 '아침'은 이 시간대로 판단
MORNING_HOURS = (5, 12)          # 이 시간대의 첫 대화 때 먼저 브리핑

# ---- 웹 검색 ----
SEARCH_MODEL = "gpt-4.1-mini"    # 웹 검색할 때 쓰는 모델
SEARCH_DAILY_LIMIT = 30          # 하루 최대 검색 횟수 (한 번에 약 1센트)

# ---- 먼저 말 거는 알림 ----
EVENT_HEADS_UP = 10              # 캘린더 일정 몇 분 전에 알려줄지 (0이면 안 함)

# ---- 뉴스 ----------------------------------------------------------
NEWS_MAX_PER_FEED = 12
NEWS_HOURS = 24          # 이 시간 안의 기사 위주로 (부족하면 최신순으로 채움)

# ---- 스포티파이 -------------------------------------------------------
SPOTIFY_USERNAME = "qkrtpdud"   # 스포티파이 사용자 이름 (좋아요 표시한 곡 틀 때 씀)

# ---- 기본법 (안전 규칙) -------------------------------------------------
PROTECTED_APPS = {"code", "visual studio code", "terminal", "iterm", "iterm2", "python", "finder"}
UNDO_MINUTES = 30
# 지문(Touch ID)으로만 허락하는 행동. 말로 "yes" 해도 통과 안 됨.
# 여기 넣으면 허락 수준이 올라가기만 함 (기능 파일에서 정한 수준보다 낮아지는 일은 없음)
SECURE_TOOLS = {"forget_fact"}   # 기억 지우기
SECURE_TIMEOUT = 30              # 이 시간(초) 안에 지문을 안 대면 거절        # 이 시간이 지난 행동은 되돌리지 않음

# ---- 화면 (HUD) -----------------------------------------------------
HUD_ENABLED = True               # 화면 원 띄우기 (pip install pywebview 필요, 없으면 화면만 빠지고 정상 작동)
HUD_POSITION = "bottom-right"    # bottom-right / bottom-left / top-right / top-left / center
HUD_MODE = "everywhere"          # 처음 보이기 방식: everywhere(유튜브·전체화면 위에도 항상) / auto(부를 때만) / off
HUD_PORT = 8765
HANDS_PORT = 8800
HAND_REMOTE_PORT = 8767           # 손 리모컨 → 자비스 손짓 신호 (이 컴퓨터 안에서만)                # 손 모드 화면 주소 (http://localhost:8800, 이 컴퓨터 안에서만)                  # 화면과 주고받는 번호 (이 컴퓨터 안에서만)

# ---- 파일 위치 -------------------------------------------------------
LOG_FILE = os.path.expanduser("~/assistant_log.jsonl")
STATE_FILE = os.path.expanduser("~/.jarvis_state.json")
