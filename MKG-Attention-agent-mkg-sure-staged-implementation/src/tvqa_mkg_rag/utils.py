from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_str: str | Path, base: Path | None = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    anchor = base or project_root()
    return (anchor / path).resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    ensure_directory(path.parent)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def read_seen_values(path: Path, key: str) -> set[Any]:
    values: set[Any] = set()
    for row in iter_jsonl(path):
        if key in row:
            values.add(row[key])
    return values


def stable_hash(*parts: Any, limit: int = 12) -> str:
    digest = hashlib.md5(
        "||".join(normalize_whitespace(str(part)) for part in parts).encode("utf-8")
    ).hexdigest()
    return digest[:limit]


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall((text or "").lower()) if token not in _STOPWORDS]


def lexical_overlap_score(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def question_reasoning_type(question: str) -> str:
    normalized = normalize_whitespace(question).lower()
    if normalized.startswith("why ") or " why " in normalized:
        return "why"
    if (
        normalized.startswith("how did ")
        or normalized.startswith("how was ")
        or normalized.startswith("how does ")
        or normalized.startswith("how do ")
        or " feel " in normalized
        or " feeling " in normalized
        or " felt " in normalized
        or " mood " in normalized
        or " emotion " in normalized
    ):
        return "feeling"
    if normalized.startswith("who "):
        return "who"
    if normalized.startswith("where "):
        return "where"
    if normalized.startswith("when "):
        return "when"
    if normalized.startswith("what "):
        return "what"
    return "other"


def is_causal_or_affective_question(question: str) -> bool:
    return question_reasoning_type(question) in {"why", "feeling"}


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = normalize_whitespace(item)
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(normalized)
    return ordered


def bounded(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def first_json_object(raw_text: str) -> dict[str, Any] | None:
    if not raw_text:
        return None
    raw_text = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None
