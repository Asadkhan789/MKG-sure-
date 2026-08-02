from __future__ import annotations

from pathlib import Path

from .io_utils import stable_id
from .schemas import SourceUnit


def dummy_regions(sources: list[SourceUnit]) -> list[SourceUnit]:
    output = list(sources)
    for source in sources:
        if source.source_type != "visual_concept": continue
        output.append(SourceUnit(source_id=f"{source.video_id}:pred_region:{stable_id(source.source_id)}", video_id=source.video_id, source_type="region", text=source.text, start_time=source.start_time, end_time=source.end_time, frame_id=source.frame_id, confidence=min(0.9, max(0.3, source.confidence)), oracle_annotation=False, metadata={"detector": "dummy_from_visual_concept"}))
    return output


class GroundingDINORunner:
    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-base", device: str = "auto"):
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install mkg-sure[real] for Grounding DINO") from exc
        self.torch, self.Image = torch, Image
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        resolved = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.device = torch.device(resolved); self.model.to(self.device).eval().requires_grad_(False); self.model_name = model_name

    def predict_image(self, image_path: str, labels: list[str], frame_id: int, video_id: str) -> list[SourceUnit]:
        image = self.Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=". ".join(labels) + ".", return_tensors="pt").to(self.device)
        with self.torch.inference_mode(): outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(outputs, inputs["input_ids"], threshold=0.25, text_threshold=0.20, target_sizes=[image.size[::-1]])[0]
        rows = []
        for index, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"])):
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            rows.append(SourceUnit(source_id=f"{video_id}:pred_region:{frame_id}:{index}", video_id=video_id, source_type="region", text=str(label), frame_id=frame_id, bbox=(x1, y1, x2, y2), media_path=image_path, confidence=float(score), oracle_annotation=False, metadata={"detector": self.model_name}))
        return rows


def sample_video_frames(video_path: str, output_dir: str | Path, count: int = 4) -> list[tuple[int, float, str]]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install mkg-sure[real] for raw-video frame extraction") from exc
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened(): raise RuntimeError(f"Could not open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    indices = np.linspace(0, max(0, total - 1), max(1, count)).astype(int).tolist(); rows = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index); ok, frame = capture.read()
        if not ok: continue
        path = output / f"frame_{frame_index:06d}.jpg"; cv2.imwrite(str(path), frame); rows.append((frame_index, frame_index / fps, str(path)))
    capture.release(); return rows


def extract_raw_video_regions(clip_sources: list[SourceUnit], prompt_labels: dict[str, list[str]], runner: GroundingDINORunner, output_dir: str | Path, frames_per_clip: int = 4) -> list[SourceUnit]:
    regions = []
    for clip in clip_sources:
        if not clip.media_path: continue
        frames = sample_video_frames(clip.media_path, Path(output_dir) / clip.video_id, count=frames_per_clip)
        labels = prompt_labels.get(clip.video_id) or ["person", "object", "room"]
        for frame_index, timestamp, image_path in frames:
            predicted = runner.predict_image(image_path, labels, frame_index, clip.video_id)
            for region in predicted: region.start_time = timestamp; region.end_time = timestamp
            regions.extend(predicted)
    return regions
