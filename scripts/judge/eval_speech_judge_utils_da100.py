"""Grounded LibriSQA judge on a continuous 0-100 Direct Assessment scale
(GEMBA-DA style adapted to speech-QA). Breaks the Likert-5 ceiling that
saturates on easy factual answers.
"""
from __future__ import annotations

from typing import Any, Dict, List


_LIBRISQA_DA100_RUBRIC = (
    "Score the candidate answer on a continuous 0-100 scale (integer or float):\n"
    "  0-10   : Empty, unrelated, or contradicts the passage/reference.\n"
    "  11-30  : Core meaning lost; severe factual error or heavy hallucination.\n"
    "  31-50  : Partially correct; at least one key fact wrong, missing, or unsupported.\n"
    "  51-70  : Most meaning preserved; minor factual imprecision, small omission, or minor unsupported addition.\n"
    "  71-90  : Correct and grounded; minor wording imprecision or small completeness gap only.\n"
    "  91-100 : Fully correct, complete, grounded in the passage, free of hallucination. A valid\n"
    "           paraphrase of the reference that is supported by the passage belongs here.\n"
    "Adequacy (factual agreement with reference AND passage) takes precedence over fluency.\n"
    "Do NOT penalize valid paraphrases. Do penalize contradictions, missing key facts,\n"
    "and unsupported additions."
)


def _librisqa_da100_prompt(record: Dict[str, Any]) -> str:
    question = str(record.get("question") or record.get("prompt") or "").strip()
    passage = str(record.get("passage") or record.get("text") or "").strip()
    reference = str(record.get("reference") or "").strip()
    prediction = str(record.get("prediction") or "").strip()
    passage_line = (
        f"Spoken passage (transcript):\n\"\"\"\n{passage}\n\"\"\""
        if passage else "Spoken passage (transcript): [NOT PROVIDED]"
    )
    return (
        "You are evaluating a candidate answer for a spoken question-answering task using a\n"
        "GEMBA-style Direct Assessment protocol. You are given the full passage transcript,\n"
        "the question, the gold reference answer, and the candidate. Score the candidate against\n"
        "BOTH the reference and the passage.\n\n"
        f"{_LIBRISQA_DA100_RUBRIC}\n\n"
        "Procedure:\n"
        "  1) List the key facts in the reference, list any errors/omissions/unsupported content\n"
        "     in the candidate, and judge grounding against the passage.\n"
        "  2) Then assign the final overall_score as a number in [0, 100].\n\n"
        "Output a SINGLE JSON object, no markdown, with keys in this exact order:\n"
        '  {"reasoning": str, "errors": [str, ...], "overall_score": number}\n\n'
        f"{passage_line}\n"
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Candidate answer: {prediction}\n"
    )


_MT_DA100_RUBRIC = (
    "Score the candidate translation on a continuous 0-100 scale (integer or float).\n"
    "This is the standard WMT Direct Assessment scale used by GEMBA (Kocmi & Federmann 2023):\n"
    "  0-10   : Nonsense / no meaningful relation to the source. Off-topic, untranslated,\n"
    "           a fragment in a different language, or essentially empty.\n"
    "  11-30  : Major adequacy errors. Core meaning of the source is lost or contradicted;\n"
    "           large omissions; multiple hallucinated entities/facts.\n"
    "  31-50  : Disfluent and inaccurate but recognisable. Some content from the source is\n"
    "           preserved but with severe grammar/word-choice problems or significant omissions.\n"
    "  51-70  : Most source meaning preserved; minor adequacy errors (small omissions, a\n"
    "           wrong word, awkward construction) or moderate fluency issues.\n"
    "  71-90  : Adequate translation with minor wording or fluency imprecision. A valid\n"
    "           paraphrase of the reference belongs here.\n"
    "  91-100 : Fully adequate and fluent. Conveys all of the source content; native-quality\n"
    "           target-language style; trivial-or-no differences from the reference.\n"
    "Adequacy (preservation of source meaning) takes precedence over fluency.\n"
    "Do NOT penalise valid paraphrases of the reference. Do penalise mistranslations,\n"
    "untranslated source words, content from outside the source, and ungrammatical output."
)


def _mt_da100_prompt(record: Dict[str, Any]) -> str:
    source     = str(record.get("source")     or "").strip()
    reference  = str(record.get("reference")  or "").strip()
    prediction = str(record.get("prediction") or "").strip()
    src_line = (
        f"Source (English):\n\"\"\"\n{source}\n\"\"\""
        if source else "Source (English): [NOT PROVIDED]"
    )
    return (
        "You are evaluating a German translation of an English source sentence using a\n"
        "GEMBA-style Direct Assessment (WMT DA) protocol. You are given the source, the\n"
        "gold reference translation, and the candidate translation. Score the candidate\n"
        "against BOTH the source (for adequacy) and the reference (for paraphrase quality).\n\n"
        f"{_MT_DA100_RUBRIC}\n\n"
        "Procedure:\n"
        "  1) Briefly identify the key content units in the source and check each against\n"
        "     the candidate. Note adequacy errors (mistranslations, omissions, additions)\n"
        "     and fluency issues (grammar, word choice, naturalness).\n"
        "  2) Then assign the final overall_score as a number in [0, 100].\n\n"
        "Output a SINGLE JSON object, no markdown, with keys in this exact order:\n"
        '  {"reasoning": str, "errors": [str, ...], "overall_score": number}\n\n'
        f"{src_line}\n"
        f"Reference translation (German): {reference}\n"
        f"Candidate translation (German): {prediction}\n"
    )


def build_judge_messages_da100(record: Dict[str, Any], rubric_kind: str = "qa") -> List[Dict[str, str]]:
    """Build the system/user messages for the DA-100 judge.

    rubric_kind:
      "qa" → LibriSQA-style grounded QA rubric (default; what the QA datasets use).
      "mt" → GEMBA-DA MT rubric for translation tasks (CoVoST2 en→de).
    """
    if rubric_kind == "mt":
        system = (
            "You are a professional machine-translation quality evaluator. "
            "Output a SINGLE valid JSON object ONLY, with no markdown fences and no text outside the JSON. "
            "overall_score must be a number in [0, 100]."
        )
        user = _mt_da100_prompt(record)
    else:
        system = (
            "You are a professional speech-QA quality evaluator. "
            "Output a SINGLE valid JSON object ONLY, with no markdown fences and no text outside the JSON. "
            "overall_score must be a number in [0, 100]."
        )
        user = _librisqa_da100_prompt(record)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
