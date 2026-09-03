from __future__ import annotations

from functools import lru_cache
from typing import Dict, List
import warnings

from sacrebleu import corpus_bleu

from .text_utils import normalize_text, whitespace_words


@lru_cache(maxsize=None)
def _load_evaluate_metric(name: str):
    import evaluate

    return evaluate.load(name)


def compute_corpus_metrics(
    predictions: List[str],
    references: List[str],
    with_text_metrics: bool = False,
    with_bertscore: bool = False,
    bertscore_lang: str = "en",
    lowercase_bleu: bool = True,
    lowercase: bool = False,
) -> Dict[str, float]:
    preds = [normalize_text(x) for x in predictions]
    refs = [normalize_text(x) for x in references]

    if lowercase:
        preds = [p.lower() for p in preds]
        refs = [r.lower() for r in refs]

    bleu = corpus_bleu(preds, [refs], lowercase=lowercase_bleu).score if preds else 0.0
    exact_match = 100.0 * sum(p == r for p, r in zip(preds, refs)) / max(len(preds), 1)
    avg_pred_words = sum(len(whitespace_words(p)) for p in preds) / max(len(preds), 1)
    avg_ref_words = sum(len(whitespace_words(r)) for r in refs) / max(len(refs), 1)

    metrics = {
        "bleu": float(bleu),
        "exact_match": float(exact_match),
        "avg_pred_words": float(avg_pred_words),
        "avg_ref_words": float(avg_ref_words),
    }

    if not preds:
        if with_text_metrics:
            metrics.update(
                {
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "meteor": 0.0,
                }
            )
        if with_bertscore:
            metrics["bertscore_f1"] = 0.0
        return metrics

    if with_text_metrics:
        try:
            rouge = _load_evaluate_metric("rouge")
            meteor = _load_evaluate_metric("meteor")
            out_rouge = rouge.compute(predictions=preds, references=refs)
            out_meteor = meteor.compute(predictions=preds, references=refs)
            metrics.update(
                {
                    "rouge1": float(out_rouge["rouge1"] * 100.0),
                    "rouge2": float(out_rouge["rouge2"] * 100.0),
                    "rougeL": float(out_rouge["rougeL"] * 100.0),
                    "meteor": float(out_meteor["meteor"] * 100.0),
                }
            )
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            warnings.warn(f"ROUGE/METEOR failed and will be skipped: {exc}")

    if with_bertscore:
        try:
            import numpy as np

            bertscore = _load_evaluate_metric("bertscore")
            out_bs = bertscore.compute(predictions=preds, references=refs, lang=bertscore_lang)
            metrics["bertscore_f1"] = float(np.mean(out_bs["f1"]) * 100.0)
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            warnings.warn(f"BERTScore failed and will be skipped: {exc}")

    return metrics
