from __future__ import annotations

from pathlib import Path

from ..config import PromptPaths


class PromptRepository:
    def __init__(self, prompt_paths: PromptPaths) -> None:
        self._paths = {
            "triplet_text": Path(prompt_paths.triplet_text),
            "triplet_visual": Path(prompt_paths.triplet_visual),
            "answer_mc": Path(prompt_paths.answer_mc),
            "verify_answer": Path(prompt_paths.verify_answer),
        }
        self._cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        if name not in self._cache:
            self._cache[name] = self._paths[name].read_text(encoding="utf-8").strip()
        return self._cache[name]

    def render(self, name: str, **values: str) -> str:
        template = self.get(name)
        normalized_values = {
            key: value.strip() if isinstance(value, str) else value for key, value in values.items()
        }
        return template.format(**normalized_values)

    @staticmethod
    def format_options(options: list[str]) -> str:
        return "\n".join(f"{index}. {option}" for index, option in enumerate(options))
