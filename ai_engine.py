"""FluentPath AI text service: Gemini, OpenRouter, then Groq."""

import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _setting(name, default=""):
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        import streamlit as st
        return str(st.secrets.get(name, default)).strip()
    except (ImportError, AttributeError, KeyError, FileNotFoundError):
        return default


def _request(url, headers, payload):
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        return response.json(), None
    except requests.Timeout:
        return None, "request timed out"
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        return None, f"HTTP {code}"
    except (requests.RequestException, ValueError) as exc:
        return None, type(exc).__name__


def _gemini(system_prompt, user_prompt, key):
    model = _setting("GEMINI_MODEL", "gemini-2.5-flash")
    data, error = _request(
        GEMINI_URL.format(model),
        {"x-goog-api-key": key, "Content-Type": "application/json"},
        {
            "systemInstruction": {"parts": [{"text": str(system_prompt)}]},
            "contents": [{"role": "user", "parts": [{"text": str(user_prompt)}]}],
            "generationConfig": {"temperature": 0.3},
        },
    )
    if error:
        return None, error
    try:
        parts = data["candidates"][0]["content"]["parts"]
        answer = "\n".join(part["text"] for part in parts if part.get("text")).strip()
        return (answer, None) if answer else (None, "empty response")
    except (KeyError, IndexError, TypeError):
        return None, "no usable text in response"


def _openrouter(system_prompt, user_prompt, key):
    data, error = _request(
        OPENROUTER_URL,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {
            "model": _setting("OPENROUTER_MODEL", "openrouter/free"),
            "messages": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "temperature": 0.3,
        },
    )
    if error:
        return None, error
    try:
        content = data["choices"][0]["message"]["content"]
        answer = content.strip() if isinstance(content, str) else ""
        return (answer, None) if answer else (None, "empty response")
    except (KeyError, IndexError, TypeError):
        return None, "no usable text in response"


def _groq(system_prompt, user_prompt, key):
    data, error = _request(
        GROQ_URL,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {
            "model": _setting("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "messages": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "temperature": 0.3,
        },
    )
    if error:
        return None, error
    try:
        content = data["choices"][0]["message"]["content"]
        answer = content.strip() if isinstance(content, str) else ""
        return (answer, None) if answer else (None, "empty response")
    except (KeyError, IndexError, TypeError):
        return None, "no usable text in response"


def _valid_answer(answer, system_prompt):
    if not answer or answer.strip().lower() in {"user safety: safe", "safe"}:
        return False
    if "you are smartspeak" in str(system_prompt).lower():
        if "evaluator" in str(system_prompt).lower():
            score = re.search(
                r"Total Score:\s*(\d{1,3})\s*/\s*100", answer, re.IGNORECASE
            )
            return bool(score and 0 <= int(score.group(1)) <= 100)
        sections = ("correct english", "natural english", "professional version",
                    "best version for this situation")
        return all(section in answer.lower() for section in sections)
    return True


def ask_ai(system_prompt, user_prompt):
    """Return text via Gemini, OpenRouter, or Groq in that order."""
    if not str(user_prompt).strip():
        return "ERROR: Please enter a message."

    errors = []
    gemini_key = _setting("GEMINI_API_KEY")
    openrouter_key = _setting("OPENROUTER_API_KEY")
    groq_key = _setting("GROQ_API_KEY")
    if gemini_key:
        answer, error = _gemini(system_prompt, user_prompt, gemini_key)
        if _valid_answer(answer, system_prompt):
            return answer
        errors.append(f"Gemini: {error or 'incomplete response'}")
    if openrouter_key:
        answer, error = _openrouter(system_prompt, user_prompt, openrouter_key)
        if _valid_answer(answer, system_prompt):
            return answer
        errors.append(f"OpenRouter: {error or 'incomplete response'}")
    if groq_key:
        answer, error = _groq(system_prompt, user_prompt, groq_key)
        if _valid_answer(answer, system_prompt):
            return answer
        errors.append(f"Groq: {error or 'incomplete response'}")
    if not errors:
        return "ERROR: Set GEMINI_API_KEY, OPENROUTER_API_KEY, or GROQ_API_KEY in your secrets."
    return "ERROR: AI service unavailable (" + "; ".join(errors) + "). Please try again."
