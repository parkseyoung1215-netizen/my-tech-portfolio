# Jarvis: a personal voice assistant for the Mac (work in progress)

> Status (2026-10-18): built for the author's own use on his MacBook. Still being built. Not tested with other users.

Jarvis listens for "hey Jarvis", hears a spoken command, decides which tool to use, acts on the Mac and answers out loud.
The part this portfolio cares about is **how it is kept within rules**: what it may do, when it must ask, and whether it reports honestly what it did.

## What it does
- **Wake word and voice lock:** "hey Jarvis" is detected locally with openWakeWord. A small speaker check (trained on the author's own recordings) ignores other voices.
- **Speech to action:** speech is transcribed (OpenAI `gpt-4o-transcribe`), a model (`gpt-4.1-mini`) chooses tools, and the Mac answers with a built-in voice.
- **Tools:** apps and web pages, timers and reminders, calendar, music (Spotify, YouTube), morning briefing, economic news and market prices, long-term memory, web lookup, alerts, an on-screen HUD and hand-gesture control.

## How it is kept within rules
The same rules live in two places: in the prompt (so the model plans inside them) and in code (so the code has the final say, whatever the model writes).

- **Allow-list:** a tool that is not registered cannot run.
- **Three permission levels:** some tools run directly, some need a spoken "yes", and actions that are hard to undo (for example deleting memories) need **Touch ID**. A spoken "yes" can come from someone else or from a video playing nearby; a fingerprint cannot. If Touch ID fails, times out or errors, the action is refused.
- **Action log and self-report:** every action is logged. "What did you do today?" reports what was executed, allowed, refused, ignored or failed. "Why did you do that?" explains from the log (what was heard, which tool was chosen) and admits a mishearing. "Undo that" reverses recent actions within 30 minutes.
- **No false "done" reports (v5.5):** Jarvis once answered "Volume turned down" without calling any tool, copying an old answer from its conversation memory. A code check now catches a turn where no tool ran but the reply claims an action, tells the model that nothing was executed, and makes it decide again. Past turns are marked as past records.

The full history, including the problems found and how each was fixed, is in [CHANGELOG.md](CHANGELOG.md) (in Korean).

## Files
| Path | Role |
|---|---|
| `assistant.py` | Entry point |
| `config.py` | All settings (voices, wake-word thresholds, models, permission levels) |
| `jarvis/` | Core: hearing, voice, brain (tool calling), tool registry, Touch ID (`secure.py`), log/explain/undo (`trust.py`), HUD |
| `skills/` | One file per group of tools |
| `hud/` | On-screen display and hand-gesture remote |
| `enroll_voice.py` | Records the owner's voice to train the voice lock |
| `jarvis.sh`, `start_jarvis.command` | Run in the background / start at login |

## Not included in this repository
API keys (`.env`), the voice recordings and the trained voice model (`voice/`), and the personal memory file are kept only on the author's computer.

## Limits
- Built and tested by one person on one Mac. No measurements of accuracy, false wake-ups or blocked actions yet.
- The speaker check is small and was trained on a small set of recordings.
- No `requirements.txt` yet. Main libraries: `openai`, `openwakeword`, `sounddevice`, `numpy`, `scipy`, `scikit-learn`, `yfinance`, `pyobjc` (macOS).

