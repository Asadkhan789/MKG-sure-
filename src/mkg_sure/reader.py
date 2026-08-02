from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from .io_utils import lexical_overlap, tokenize


class FrozenReader(ABC):
    @abstractmethod
    def score_options(self, question: str, options: list[str], context: str, media_paths: list[str] | None = None) -> list[float]: ...

    def predict(self, question: str, options: list[str], context: str, media_paths: list[str] | None = None) -> tuple[int, list[float]]:
        scores = self.score_options(question, options, context, media_paths)
        return int(max(range(len(scores)), key=lambda index: scores[index])), scores


class LexicalFrozenReader(FrozenReader):
    """Deterministic frozen reader for smoke tests only."""

    def score_options(self, question: str, options: list[str], context: str, media_paths: list[str] | None = None) -> list[float]:
        context_tokens = tokenize(context)
        context_set = set(context_tokens)
        scores = []
        for option in options:
            option_tokens = tokenize(option)
            option_set = set(option_tokens)
            coverage = len(option_set & context_set) / max(1, len(option_set))
            repeated_support = sum(context_tokens.count(token) for token in option_set) / max(1, len(option_set))
            scores.append(1.8 * coverage + 0.18 * repeated_support + 0.4 * lexical_overlap(question, option) + 0.1 * lexical_overlap(question, context))
        return scores


class Qwen3VLFrozenReader(FrozenReader):
    """Frozen Qwen3-VL option log-likelihood scorer."""

    def __init__(self, model_name: str, device_map: str = "auto", dtype: str = "auto", flash_attention_2: bool = True):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install mkg-sure[real] for Qwen3-VL reader support") from exc
        self.processor = AutoProcessor.from_pretrained(model_name)
        kwargs = {"device_map": device_map, "dtype": dtype}
        if flash_attention_2:
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs).eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False

    @torch.inference_mode()
    def score_options(self, question: str, options: list[str], context: str, media_paths: list[str] | None = None) -> list[float]:
        scores = []
        media_paths = [path for path in (media_paths or []) if Path(path).exists()]
        for option in options:
            prompt = f"Evidence:\n{context}\n\nQuestion: {question}\nChoose the correct answer."
            if media_paths:
                content = [{"type": "video", "video": path} for path in media_paths[:1]]
                content.append({"type": "text", "text": prompt})
                prompt_messages = [{"role": "user", "content": content}]
                full_messages = prompt_messages + [{"role": "assistant", "content": [{"type": "text", "text": option}]}]
                prompt_inputs = self.processor.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
                full_inputs = self.processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
                labels = full_inputs["input_ids"].clone()
                prompt_length = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
                labels[:, :prompt_length] = -100
                full_inputs["labels"] = labels
                outputs = self.model(**full_inputs)
                token_count = max(1, int((labels != -100).sum().item()))
            else:
                tokenizer = self.processor.tokenizer
                prompt_text = prompt + "\nAnswer:"
                prompt_ids = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(self.model.device)
                option_ids = tokenizer(option, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device)
                input_ids = torch.cat([prompt_ids, option_ids], dim=1)
                labels = input_ids.clone()
                labels[:, :prompt_ids.shape[1]] = -100
                outputs = self.model(input_ids=input_ids, labels=labels)
                token_count = max(1, option_ids.shape[1])
            scores.append(float(-outputs.loss.item() / token_count))
        return scores


def correct_margin(scores: list[float], answer_idx: int) -> float:
    correct = scores[answer_idx]
    others = [score for index, score in enumerate(scores) if index != answer_idx]
    return correct - max(others)
