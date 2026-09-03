from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeConfig:
    model_name: str
    batch_size: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    device: str = "cuda"
    dtype: str = "bfloat16"


def _pick_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = safe_mean(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return float(math.sqrt(var))


def _extract_json_candidate(text: str) -> str:
    text = text.strip()
    match = JSON_RE.search(text)
    return match.group(0) if match else text


def _coerce_int_range(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        score = int(round(float(value)))
    except Exception:
        score = default
    return max(lo, min(hi, score))


def _coerce_float_range(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        score = float(value)
    except Exception:
        score = default
    return max(lo, min(hi, score))


def parse_judge_json(text: str, scale: str = "likert5") -> Dict[str, Any]:
    """Parse JSON judge output.

    scale="likert5": integer sub-scores in [1,5]; overall_score int in [1,5].
    scale="da100":   overall_score float in [0,100] (GEMBA-DA style).
    """
    raw = _extract_json_candidate(text)
    fail_default_overall = 1 if scale == "likert5" else 0.0
    try:
        obj = json.loads(raw)
    except Exception:
        return {
            "overall_score": fail_default_overall,
            "reasoning": raw[:500],
            "parse_failed": True,
        }
    if not isinstance(obj, dict):
        return {
            "overall_score": fail_default_overall,
            "reasoning": str(obj)[:500],
            "parse_failed": True,
        }

    out = dict(obj)
    if scale == "da100":
        out["overall_score"] = _coerce_float_range(
            out.get("overall_score", out.get("score", 0)), 0.0, 100.0, 0.0
        )
    else:
        out["overall_score"] = _coerce_int_range(out.get("overall_score", 1), 1, 5, 1)
        for key, value in list(out.items()):
            if key.endswith("_score") and key != "overall_score":
                out[key] = _coerce_int_range(value, 1, 5, out["overall_score"])
    if "reason" in out and "reasoning" not in out:
        out["reasoning"] = out.pop("reason")
    out.setdefault("parse_failed", False)
    out.setdefault("reasoning", "")
    return out


def _maybe_apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    rendered = []
    for msg in messages:
        rendered.append(f"{msg['role'].upper()}: {msg['content']}")
    rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


_LIBRISQA_RUBRIC = (
    "Rubric (1-5 integer Likert, sub-scores and overall):\n"
    "  5 = Fully correct and complete; no hallucinations; semantically equivalent to the reference (paraphrases OK).\n"
    "  4 = Correct main content with a minor omission or wording imprecision; no contradictions.\n"
    "  3 = Partially correct; one key fact is missing or slightly off, but answer is on-topic.\n"
    "  2 = Mostly wrong or missing most of the reference content, OR contains minor hallucinations.\n"
    "  1 = Contradicts the reference, fully hallucinated, off-topic, or empty.\n"
    "Dimensions to score independently:\n"
    "  correctness_score: factual agreement with the reference answer.\n"
    "  completeness_score: coverage of the key facts in the reference.\n"
    "  hallucination_score: freedom from unsupported or fabricated content (5 = none, 1 = severe).\n"
    "overall_score should reflect a holistic judgment, not a strict average."
)


def _librisqa_prompt(record: Dict[str, Any]) -> str:
    question = str(record.get("question") or record.get("prompt") or "").strip()
    reference = str(record.get("reference") or "").strip()
    prediction = str(record.get("prediction") or "").strip()
    return (
        "You are evaluating a candidate answer for a spoken question-answering task.\n"
        "The reference is the gold answer; accept valid paraphrases and minor wording "
        "differences. Penalize contradictions, missing key facts, and unsupported additions.\n\n"
        f"{_LIBRISQA_RUBRIC}\n\n"
        "Procedure:\n"
        "  1) Think step by step: identify the key facts in the reference, then check whether "
        "the candidate covers each, whether anything contradicts the reference, and whether "
        "anything is hallucinated.\n"
        "  2) Only AFTER reasoning, assign integer sub-scores and the overall_score.\n\n"
        "Output a SINGLE JSON object, no markdown, with keys in this exact order:\n"
        '  {"reasoning": str, "correctness_score": int, "completeness_score": int, '
        '"hallucination_score": int, "overall_score": int}\n\n'
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Candidate answer: {prediction}\n"
    )


_COVOST2_RUBRIC = (
    "Use the WMT Direct Assessment (DA) scale, integer or float in [0, 100]:\n"
    "  0-10   : Random or unrelated output, or empty translation.\n"
    "  11-30  : Major accuracy errors; core meaning lost; many mistranslations or omissions.\n"
    "  31-50  : Some meaning preserved but significant errors (wrong facts, omissions, additions, wrong negation, wrong polarity, wrong entity).\n"
    "  51-70  : Most meaning preserved; noticeable errors (minor mistranslation, awkward phrasing, small omission) but understandable.\n"
    "  71-90  : Meaning fully preserved; minor fluency issues (grammar, word choice, style) only.\n"
    "  91-100 : Perfect or near-perfect translation; adequate and fluent; indistinguishable from a good human translation.\n"
    "Adequacy (meaning) takes precedence over fluency. Paraphrases of the reference are acceptable if meaning is preserved."
)


def _covost2_prompt(record: Dict[str, Any]) -> str:
    source = str(record.get("source") or record.get("sentence") or "").strip()
    reference = str(record.get("reference") or "").strip()
    prediction = str(record.get("prediction") or "").strip()
    src_line = f'English source: "{source}"' if source else "English source: [NOT PROVIDED]"
    return (
        "You are evaluating the quality of a German machine translation using the "
        "GEMBA-DA protocol (Kocmi & Federmann, 2023). Score the candidate translation on "
        "a continuous 0-100 direct-assessment scale.\n\n"
        f"{_COVOST2_RUBRIC}\n\n"
        "Procedure:\n"
        "  1) Think step by step: identify any mistranslations, omissions, additions, wrong "
        "entities, wrong polarity/negation, and fluency issues. Compare meaning against "
        "the source (primary) and the reference translation (supporting).\n"
        "  2) Only AFTER reasoning, assign the final numeric score.\n\n"
        "Output a SINGLE JSON object, no markdown, with keys in this exact order:\n"
        '  {"reasoning": str, "errors": [str, ...], "overall_score": number}\n'
        "where overall_score is a number in [0, 100].\n\n"
        f"{src_line}\n"
        f'Reference German translation: "{reference}"\n'
        f'Candidate German translation: "{prediction}"\n'
    )


TASK_SCALES = {
    "librisqa": "likert5",
    "covost2": "da100",
}


def build_judge_messages(task: str, record: Dict[str, Any]) -> List[Dict[str, str]]:
    scale = TASK_SCALES.get(task, "likert5")
    if scale == "da100":
        system = (
            "You are a professional bilingual translation quality evaluator. "
            "Output a SINGLE valid JSON object ONLY, with no markdown fences and no text "
            "outside the JSON. The overall_score must be a number in [0, 100]."
        )
    else:
        system = (
            "You are a strict but fair evaluator. Reason step by step before scoring. "
            "Output a SINGLE valid JSON object ONLY, with no markdown fences and no text "
            "outside the JSON. All sub-scores and overall_score must be integers in [1, 5]."
        )
    user = _covost2_prompt(record) if task == "covost2" else _librisqa_prompt(record)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_local_judge(config: JudgeConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = _pick_dtype(config.dtype)
    model_kwargs = {}
    if config.device.startswith("cuda") and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            **model_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            attn_implementation="sdpa",
            **model_kwargs,
        )
    model.eval()
    return tokenizer, model


def run_local_llm_judge(
    *,
    task: str,
    records: List[Dict[str, Any]],
    config: JudgeConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not records:
        return {}, []

    from tqdm import tqdm

    tokenizer, model = load_local_judge(config)
    generated_rows: List[Dict[str, Any]] = []
    prompt_texts = [
        _maybe_apply_chat_template(tokenizer, build_judge_messages(task, record))
        for record in records
    ]

    pbar = tqdm(total=len(prompt_texts), desc=f"judge[{task}]", unit="ex", dynamic_ncols=True)
    for start in range(0, len(prompt_texts), config.batch_size):
        stop = min(len(prompt_texts), start + config.batch_size)
        batch_prompts = prompt_texts[start:stop]
        batch_records = records[start:stop]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        model_device = next(model.parameters()).device
        inputs = {
            key: value.to(model_device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        do_sample = config.temperature > 0.0
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=do_sample,
            top_p=config.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            generate_kwargs["temperature"] = max(config.temperature, 1e-5)
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        continuations = outputs[:, prompt_len:]
        texts = tokenizer.batch_decode(continuations, skip_special_tokens=True)
        scale = TASK_SCALES.get(task, "likert5")
        for record, text in zip(batch_records, texts):
            parsed = parse_judge_json(text, scale=scale)
            row = dict(record)
            row["judge_raw"] = text
            row.update(parsed)
            generated_rows.append(row)
        pbar.update(stop - start)
    pbar.close()

    numeric_keys = sorted(
        {
            key
            for row in generated_rows
            for key, value in row.items()
            if key.endswith("_score") or key == "overall_score"
        }
    )

    summary: Dict[str, Any] = {
        "judge_num_examples": len(generated_rows),
        "judge_parse_fail_rate": float(sum(1 for row in generated_rows if row.get("parse_failed")) / max(len(generated_rows), 1)),
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in generated_rows if key in row]
        if not values:
            continue
        summary[f"judge_{key}_mean"] = round(safe_mean(values), 6)
        summary[f"judge_{key}_std"] = round(safe_std(values), 6)
    scale = TASK_SCALES.get(task, "likert5")
    if "judge_overall_score_mean" in summary:
        score = float(summary["judge_overall_score_mean"])
        if scale == "da100":
            summary["judge_overall_pct"] = round(score, 6)
        else:
            summary["judge_overall_pct"] = round(100.0 * (score - 1.0) / 4.0, 6)
    summary["judge_scale"] = scale
    return summary, generated_rows


def save_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Shared jsonl / LibriSQA-passage helpers used by the judge producers.
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _norm(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().split())


def build_passage_lookup(json_paths: List[str]) -> Dict[str, Dict[str, str]]:
    """Return dict with two maps built from LibriSQA source JSON(s):
      - "qa": (question,answer) -> passage  [primary, ~unique]
      - "q" : question          -> passage  [fallback]
    Matching by (question, answer) avoids ambiguous hits when the same question
    appears on multiple passages (e.g. 'Who is speaking?').
    """
    from datasets import load_dataset

    data_files = {f"s{i}": p for i, p in enumerate(json_paths)}
    ds = load_dataset("json", data_files=data_files)
    qa_map: Dict[str, str] = {}
    q_map: Dict[str, str] = {}
    q_ambiguous: set = set()
    text_keys = ("text", "passage", "transcript", "transcription")
    ans_keys = ("answer", "reference")
    for split in ds:
        for ex in ds[split]:
            q = _norm(ex.get("question"))
            if not q:
                continue
            passage = ""
            for k in text_keys:
                v = ex.get(k)
                if v:
                    passage = str(v).strip()
                    break
            if not passage:
                continue
            a = ""
            for k in ans_keys:
                v = ex.get(k)
                if v:
                    a = _norm(v)
                    break
            qa_map[f"{q}||{a}"] = passage
            if q in q_map and q_map[q] != passage:
                q_ambiguous.add(q)
            else:
                q_map.setdefault(q, passage)
    for q in q_ambiguous:
        q_map.pop(q, None)
    print(f"  built qa-map={len(qa_map)}, unique-q-map={len(q_map)}, ambiguous-q-dropped={len(q_ambiguous)}")
    return {"qa": qa_map, "q": q_map}


def _lookup_passage(lookups: Dict[str, Dict[str, str]], question: Any, reference: Any) -> str:
    q = _norm(question)
    a = _norm(reference)
    if not q:
        return ""
    p = lookups["qa"].get(f"{q}||{a}")
    if p:
        return p
    return lookups["q"].get(q, "")
