from __future__ import annotations

from dataclasses import dataclass

from .schemas import GraphPath, SourceUnit


@dataclass(slots=True)
class PackedEvidence:
    text: str
    source_ids: list[str]
    media_paths: list[str]
    text_tokens: int
    visual_tokens: int


def pack_paths(paths: list[GraphPath], sources: dict[str, SourceUnit]) -> PackedEvidence:
    source_ids = sorted({source_id for path in paths for source_id in path.source_ids})
    source_rows = [sources[source_id] for source_id in source_ids if source_id in sources]
    source_rows.sort(key=lambda item: (item.start_time is None, item.start_time or 0.0, item.source_id))
    lines = ["[GRAPH PATHS]"]
    for path in paths:
        lines.append(f"- {path.serialized_text}")
    lines.append("[ORIGINAL SOURCES]")
    seen_text = set()
    for source in source_rows:
        if source.text in seen_text:
            continue
        seen_text.add(source.text)
        timing = "" if source.start_time is None else f" @{source.start_time:.2f}-{(source.end_time or source.start_time):.2f}s"
        lines.append(f"- [{source.source_type}:{source.source_id}{timing}] {source.text}")
    media_paths = sorted({source.media_path for source in source_rows if source.media_path})
    text = "\n".join(lines)
    return PackedEvidence(text=text, source_ids=source_ids, media_paths=media_paths, text_tokens=len(text.split()), visual_tokens=sum(1 for source in source_rows if source.source_type in {"clip", "frame", "region", "visual_concept"}))
