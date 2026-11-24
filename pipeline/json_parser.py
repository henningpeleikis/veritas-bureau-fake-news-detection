import json
import re
import logging

logger = logging.getLogger(__name__)


def _strip_trailing_commas(text: str) -> str:
    text = re.sub(r",\s*(\})", r"\1", text)
    text = re.sub(r",\s*(\])", r"\1", text)
    return text


def _fix_unclosed_array(text: str) -> str:
    """Fix qwen2.5 pattern: array missing ] before final } e.g. {"key": [...}}"""
    stripped = text.rstrip()
    if not stripped.endswith("}"):
        return text
    # Count unmatched [ vs ]
    open_brackets = 0
    in_string = False
    escape_next = False
    for ch in stripped:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            open_brackets += 1
        elif ch == "]":
            open_brackets -= 1
    if open_brackets > 0:
        # Insert the missing ] before the last }
        idx = stripped.rfind("}")
        return stripped[:idx] + ("]" * open_brackets) + stripped[idx:]
    return text


def _fix_truncated(text: str) -> str:
    """Recover from output truncated mid-string by rolling back to the last complete value."""
    stripped = text.rstrip()
    # If it ends cleanly, nothing to do
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    # Roll back to the last comma or bracket that closes a complete value
    for i in range(len(stripped) - 1, 0, -1):
        candidate = stripped[:i].rstrip().rstrip(",").rstrip()
        # Try closing all open structures
        opens = candidate.count("{") - candidate.count("}")
        arr_opens = candidate.count("[") - candidate.count("]")
        closing = ("]" * arr_opens) + ("}" * opens)
        try:
            json.loads(candidate + closing)
            return candidate + closing
        except json.JSONDecodeError:
            continue
    return text


def _candidates(text: str):
    """Yield progressively fixed versions of the text."""
    yield text
    yield _strip_trailing_commas(text)
    fixed_arr = _fix_unclosed_array(text)
    yield fixed_arr
    yield _strip_trailing_commas(fixed_arr)
    yield _fix_truncated(text)


def _try_parse(text: str) -> dict | None:
    for candidate in _candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def extract_json(text: str, default: dict = None) -> dict:
    """Robustly extract JSON from LLM output that may contain extra text or markdown."""
    if default is None:
        default = {}

    if not text or not text.strip():
        logger.warning("extract_json: empty input, returning default")
        return default

    # Attempt 1: direct parse
    result = _try_parse(text.strip())
    if result is not None:
        return result

    # Attempt 2: strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    result = _try_parse(cleaned)
    if result is not None:
        return result

    # Attempt 3: extract outermost {...} block by finding first { to last }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace:last_brace + 1]
        result = _try_parse(candidate)
        if result is not None:
            return result

    # Attempt 4: from first { to end (in case last } is missing entirely)
    if first_brace != -1:
        result = _try_parse(cleaned[first_brace:])
        if result is not None:
            return result

    logger.warning(
        f"extract_json: all parsing attempts failed, returning default. "
        f"Input preview: {text[:200]!r}"
    )
    return default
