from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

# Field names whose values must never reach disk, whatever they contain.
# `total_tokens` / `prompt_tokens` are counters and intentionally do not match.
_SENSITIVE_KEY = re.compile(
    r"^(authorization|proxy-authorization|cookie|set-cookie|password|passwd|pwd|"
    r"secret|token|credentials?|"
    r"(?:[a-z0-9]+[_-])*(?:api[_-]?key|apikey|private[_-]?key|access[_-]?key|"
    r"secret|password|token))$",
    re.IGNORECASE,
)

_KEY_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|apikey|password|passwd|pwd|client[_-]?secret|secret|"
    r"access[_-]?token|auth[_-]?token|refresh[_-]?token|token|authorization)\b"
    r"(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;&\"']+)"
)
_TEXT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"), REDACTED),
    (re.compile(r"sk-[A-Za-z0-9_\-*.]{6,}"), REDACTED),
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-~+/=]+"), REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"), REDACTED),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), REDACTED),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
)


def redact(text: str) -> str:
    """Mask credentials and personal identifiers inside free text."""
    for pattern, replacement in _TEXT_PATTERNS:
        text = pattern.sub(replacement, text)
    return _KEY_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)


def redact_value(value: Any) -> Any:
    """Recursively redact strings, and any value stored under a sensitive key."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {
            key: REDACTED
            if isinstance(key, str) and _SENSITIVE_KEY.match(key)
            else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [redact_value(item) for item in value]
    return value
