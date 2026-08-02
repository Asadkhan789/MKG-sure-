from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .io_utils import stable_id, tokenize
from .schemas import Fact, SourceUnit


class FactExtractor(ABC):
    @abstractmethod
    def extract(self, source: SourceUnit) -> list[Fact]: ...


class HeuristicFactExtractor(FactExtractor):
    """Query-independent fact extractor used for smoke tests and offline fallback."""

    VERBS = {"is", "are", "was", "were", "sits", "stands", "enters", "leaves", "gives", "give", "hands", "takes", "brings", "brought", "calls", "rings", "thanks", "reads", "smiles", "says", "carrying", "carried"}

    def extract(self, source: SourceUnit) -> list[Fact]:
        text = source.text.strip()
        tokens = tokenize(text)
        if not tokens:
            return []
        facts: list[tuple[str, str, str]] = []
        if source.source_type in {"visual_concept", "region", "frame"}:
            facts.append((tokens[0], "visible_in", source.video_id))
        else:
            verb_index = next((index for index, token in enumerate(tokens) if token in self.VERBS), None)
            if verb_index is not None and verb_index > 0 and verb_index + 1 < len(tokens):
                head = " ".join(tokens[max(0, verb_index - 2):verb_index])
                relation = tokens[verb_index]
                tail = " ".join(tokens[verb_index + 1:verb_index + 5])
                facts.append((head, relation, tail))
            content = [token for token in tokens if len(token) > 2][:5]
            for left, right in zip(content, content[1:]):
                facts.append((left, "co_occurs_with", right))
        output: list[Fact] = []
        for head, relation, tail in facts:
            output.append(Fact(
                fact_id=stable_id(source.video_id, source.source_id, head, relation, tail),
                video_id=source.video_id,
                head=head,
                relation=relation,
                tail=tail,
                source_type=source.source_type,
                source_ids=[source.source_id],
                extraction_log_likelihood=-0.1 * max(1, len(tokens)),
                reliability=min(0.95, max(0.2, source.confidence)),
                start_time=source.start_time,
                end_time=source.end_time,
            ))
        return output


class Qwen3VLFactExtractor(FactExtractor):
    """Optional frozen Qwen3-VL structured extractor using no question or options."""

    def __init__(self, model_name: str, device_map: str = "auto", dtype: str = "auto"):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install mkg-sure[real] for Qwen3-VL extraction") from exc
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, device_map=device_map, dtype=dtype).eval()
        self.model.requires_grad_(False)

    def extract(self, source: SourceUnit) -> list[Fact]:
        prompt = (
            "Extract query-independent factual relations. Return JSON only as "
            '{"facts":[{"head":"...","relation":"...","tail":"..."}]}. '
            f"Source type: {source.source_type}. Source: {source.text}"
        )
        content = []
        if source.media_path:
            media_type = "video" if source.source_type == "clip" else "image"
            content.append({"type": media_type, media_type: source.media_path})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=256)
        raw = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return []
        try:
            rows = json.loads(match.group(0)).get("facts", [])
        except json.JSONDecodeError:
            return []
        facts = []
        for row in rows:
            head, relation, tail = str(row.get("head", "")).strip(), str(row.get("relation", "")).strip(), str(row.get("tail", "")).strip()
            if head and relation and tail:
                facts.append(Fact(
                    fact_id=stable_id(source.video_id, source.source_id, head, relation, tail), video_id=source.video_id,
                    head=head, relation=relation, tail=tail, source_type=source.source_type, source_ids=[source.source_id],
                    reliability=source.confidence, start_time=source.start_time, end_time=source.end_time,
                ))
        return facts
