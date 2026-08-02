from __future__ import annotations

from statistics import mean

from ..types import PredictionRecord, QAExample


def build_report_payload(run_name: str, predictions: list[PredictionRecord]) -> dict:
    total = len(predictions)
    labeled = [record for record in predictions if record.gold_idx is not None]
    correct = [record for record in labeled if record.correct]
    high_audit = [record for record in labeled if record.audit_score >= 0.6]
    low_audit = [record for record in labeled if record.audit_score < 0.6]
    confidence_values = [record.confidence for record in predictions if record.confidence is not None]

    def _accuracy(records: list[PredictionRecord]) -> float | None:
        if not records:
            return None
        return len([record for record in records if record.correct]) / len(records)

    return {
        "run_name": run_name,
        "total_predictions": total,
        "labeled_predictions": len(labeled),
        "accuracy": _accuracy(labeled),
        "avg_audit_score": mean(record.audit_score for record in predictions) if predictions else 0.0,
        "avg_confidence": mean(confidence_values) if confidence_values else None,
        "high_audit_accuracy": _accuracy(high_audit),
        "low_audit_accuracy": _accuracy(low_audit),
        "verifier_used": len([record for record in predictions if record.verifier_used]),
        "subtitle_backed_predictions": len([record for record in predictions if record.selected_subtitle_doc_ids]),
        "empty_graph_predictions": len([record for record in predictions if not record.selected_doc_ids]),
        "error_qids": [record.qid for record in labeled if not record.correct][:50],
    }


def _option_text(options: list[str], index: int | None) -> str | None:
    if index is None or not 0 <= index < len(options):
        return None
    return options[index]


def build_prediction_outputs_payload(
    run_name: str,
    examples: list[QAExample],
    predictions: list[PredictionRecord],
) -> dict:
    examples_by_qid = {example.qid: example for example in examples}
    rows: list[dict] = []

    for record in predictions:
        example = examples_by_qid.get(record.qid)
        options = example.options if example is not None else []
        rows.append(
            {
                "qid": record.qid,
                "question": example.question if example is not None else None,
                "predicted_idx": record.predicted_idx,
                "predicted_answer": _option_text(options, record.predicted_idx),
                "gold_idx": record.gold_idx,
                "gold_answer": _option_text(options, record.gold_idx),
                "correct": record.correct,
                "confidence": record.confidence,
                "verifier_used": record.verifier_used,
                "risk_flags": record.risk_flags,
            }
        )

    return {
        "run_name": run_name,
        "total_predictions": len(rows),
        "predictions": rows,
    }


def render_markdown_report(payload: dict) -> str:
    lines = [
        f"# Validation Report: {payload['run_name']}",
        "",
        f"- Total predictions: {payload['total_predictions']}",
        f"- Labeled predictions: {payload['labeled_predictions']}",
        f"- Accuracy: {payload['accuracy']}",
        f"- Average audit score: {payload['avg_audit_score']:.4f}",
        f"- Average confidence: {payload['avg_confidence']}" if payload["avg_confidence"] is not None else "- Average confidence: None",
        f"- High-audit accuracy: {payload['high_audit_accuracy']}",
        f"- Low-audit accuracy: {payload['low_audit_accuracy']}",
        f"- Verifier used: {payload['verifier_used']}",
        f"- Subtitle-backed predictions: {payload['subtitle_backed_predictions']}",
        f"- Empty graph predictions: {payload['empty_graph_predictions']}",
        "",
        "## Error QIDs",
        ", ".join(str(qid) for qid in payload["error_qids"]) if payload["error_qids"] else "None",
    ]
    return "\n".join(lines)
