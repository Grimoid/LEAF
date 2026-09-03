"""M-Prometheus (Unbabel/M-Prometheus-*) prompt + parser for absolute Likert-5
scoring.

M-Prometheus uses the Prometheus-2 prompt template (with reference answer +
score rubric) and emits a single line in the form:

    Feedback: <reasoning> [RESULT] <integer 1..5>

We expose:
  * build_mprometheus_prompt(record, rubric=...)
  * parse_mprometheus(text) -> dict with `score`, `parse_failed`, `reasoning`,
    `winner` (always "tie" since this is absolute scoring — kept for schema
    compatibility with the pairwise aggregator if needed).

References:
  - Prometheus-2 paper:    https://arxiv.org/abs/2405.01535
  - M-Prometheus paper:    https://arxiv.org/abs/2504.04953
  - Model card template:   https://huggingface.co/Unbabel/M-Prometheus-14B
"""
from __future__ import annotations

import re
from typing import Any, Dict


# Default rubric for open-ended speech QA tasks (LibriSQA / DailyTalk /
# SpeechLongAudio). Mirrors the spirit of the existing grounded Likert-5
# protocol (correctness, grounding, completeness, hallucination) but fused
# into a single 1-5 scale per the Prometheus-2 design.
DEFAULT_QA_RUBRIC = """[Evaluate the response as an answer to a spoken-content question. Judge correctness, completeness, and absence of hallucinations against the reference answer. Ignore surface stylistic differences and length — a correct paraphrase of the reference is just as good as the reference itself.]
Score 1: The response is fully incorrect, off-topic, empty, or contradicts the reference. It contains hallucinated content not supported by the reference, OR it fails to answer the question at all.
Score 2: The response is mostly wrong or missing most of the reference content. It may contain one fact from the reference but is dominated by missing/incorrect content or minor hallucinations.
Score 3: The response captures roughly half of the key facts in the reference, OR captures the main fact but with notable gaps, vagueness, or one minor factual error.
Score 4: The response captures most of the key facts from the reference, with at most one minor omission or wording imprecision; no hallucinations.
Score 5: The response is fully correct and complete: it covers all key facts from the reference (possibly paraphrased), with no hallucinations or contradictions."""


# Default rubric for translation (en -> de etc.). Used for CoVoST2-style MT eval
# when we extend the pipeline later. Adapted from the M-Prometheus model card
# example (Accuracy + Fluency + Style merged into a single rubric).
DEFAULT_MT_RUBRIC = """[Evaluate the machine translation as a faithful rendering of the source sentence in the target language. Judge accuracy (does it preserve the meaning?), fluency (is it natural in the target language?), and absence of additions/omissions. Use the reference translation as the gold; a fluent paraphrase preserving meaning is acceptable.]
Score 1: Translation is incoherent, in the wrong language, empty, or grossly mistranslated. Most of the source meaning is lost.
Score 2: Translation conveys only part of the source meaning. Major omissions, additions, or mistranslations of key content words. Disfluent.
Score 3: Translation conveys the main meaning but has notable errors: a missing/added clause, a wrong term, or awkward target-language grammar.
Score 4: Translation conveys all the source meaning with minor issues (one wording choice, a slight unnaturalness, or a minor stylistic mismatch).
Score 5: Translation is accurate and fluent, with no omissions, additions, or mistranslations. Reads as natural target-language text and preserves the source meaning faithfully."""


_PROMETHEUS_USER_TEMPLATE = """###Task Description: An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
{rubric}

###Feedback:"""


def build_mprometheus_messages(record: Dict[str, Any], rubric: str = DEFAULT_QA_RUBRIC):
    """Return chat-format messages for M-Prometheus (Qwen2.5-14B chat template).

    `record` keys used: `question` (or `prompt`), `reference`, `prediction`.
    """
    instruction = str(record.get("question") or record.get("prompt") or "").strip()
    response = str(record.get("prediction") or "").strip()
    reference = str(record.get("reference") or "").strip()
    user = _PROMETHEUS_USER_TEMPLATE.format(
        instruction=instruction,
        response=response,
        reference=reference,
        rubric=rubric,
    )
    return [{"role": "user", "content": user}]


_RESULT_RE = re.compile(r"\[RESULT\]\s*\(?\s*(\d+)\s*\)?", re.IGNORECASE)


# ─── Pairwise (relative) grading ────────────────────────────────────────
# Prometheus-2 / M-Prometheus relative-grading prompt with reference. Output:
#   Feedback: <text> [RESULT] A    OR    Feedback: <text> [RESULT] B
# There is no native "tie" — we infer ties via AB+BA swap-aggregation upstream.

DEFAULT_PAIRWISE_RUBRIC_QA = (
    "Compare the two answers to the spoken question against the reference. Prefer "
    "the answer that (i) is more correct vs. the reference, (ii) covers more of the "
    "key facts, (iii) introduces fewer hallucinations or unsupported claims, and "
    "(iv) is not unnecessarily verbose. Ignore minor stylistic differences. "
    "If both are essentially equivalent in substance you must still choose the "
    "marginally better one, as the format requires either \"A\" or \"B\"."
)

DEFAULT_PAIRWISE_RUBRIC_MT = (
    "Compare the two German translations of the English source sentence. Prefer "
    "the translation that: (i) more faithfully preserves the meaning of the "
    "source (no omissions, no additions, no mistranslations), (ii) is more "
    "fluent and natural in German (correct grammar, appropriate register), and "
    "(iii) better matches the reference translation when paraphrases are "
    "otherwise equivalent. Ignore length differences and stylistic preferences "
    "that do not change meaning. If both are essentially equivalent you must "
    "still choose the marginally better one."
)

PAIRWISE_RUBRIC_PRESETS = {
    "qa": DEFAULT_PAIRWISE_RUBRIC_QA,
    "mt": DEFAULT_PAIRWISE_RUBRIC_MT,
}

_PAIRWISE_USER_TEMPLATE = """###Task Description:

An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), a reference answer, and an evaluation criteria are given.

1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.

2. Make comparisons between Response A, Response B, and the Reference Answer. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.

3. After writing the feedback, indicate the better response, either "A" or "B".

4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either \\"A\\" or \\"B\\")"

5. Please do not generate any other opening, closing, and explanations.

###Instruction:

{instruction}

###Response A:

{response_A}

###Response B:

{response_B}

###Reference Answer:

{reference_answer}

###Score Rubric:

{rubric}

###Feedback: """


def build_mprometheus_pairwise_messages(record: Dict[str, Any], rubric: str) -> Any:
    """Return chat-format messages for M-Prometheus relative grading.

    `record` keys used: `question` (or `prompt`), `reference`,
    `candidate_a`, `candidate_b`.
    """
    instruction = str(record.get("question") or record.get("prompt") or "").replace("<|audio|>", "").strip()
    a = str(record.get("candidate_a") or "").strip()
    b = str(record.get("candidate_b") or "").strip()
    reference = str(record.get("reference") or "").strip()
    user = _PAIRWISE_USER_TEMPLATE.format(
        instruction=instruction, response_A=a, response_B=b,
        reference_answer=reference, rubric=rubric,
    )
    return [{"role": "user", "content": user}]


_PAIRWISE_RESULT_RE = re.compile(r"\[RESULT\]\s*\(?\s*[\"']?([AB])[\"']?\s*\)?", re.IGNORECASE)


def parse_mprometheus_pairwise(text: str) -> Dict[str, Any]:
    """Parse Prometheus-2 relative-grading output.

    Returns dict with keys: winner ('a'|'b'|'tie'), correctness_winner,
    grounding_winner, completeness_winner, hallucination_winner (all set to
    the same overall verdict — Prometheus pairwise doesn't grade
    sub-dimensions), parse_failed, reasoning.

    'tie' is only used as a fallback when neither [RESULT] A nor [RESULT] B
    is parseable. Real ties bubble up from AB+BA disagreement (inconsistent)
    in the aggregator.
    """
    if not text:
        return {"winner": "tie", "correctness_winner": "tie", "grounding_winner": "tie",
                "completeness_winner": "tie", "hallucination_winner": "tie",
                "parse_failed": True, "reasoning": ""}
    m = _PAIRWISE_RESULT_RE.search(text)
    if m:
        v = m.group(1).lower()
        return {"winner": v, "correctness_winner": v, "grounding_winner": v,
                "completeness_winner": v, "hallucination_winner": v,
                "parse_failed": False, "reasoning": text.strip()[-1500:]}
    # Fallback: a lone A or B at the very end of the output (clipped early).
    tail = text.strip().rstrip(".\"' \t\n").upper()
    if tail.endswith("A") or tail.endswith("B"):
        v = tail[-1].lower()
        return {"winner": v, "correctness_winner": v, "grounding_winner": v,
                "completeness_winner": v, "hallucination_winner": v,
                "parse_failed": False, "reasoning": text.strip()[-1500:]}
    return {"winner": "tie", "correctness_winner": "tie", "grounding_winner": "tie",
            "completeness_winner": "tie", "hallucination_winner": "tie",
            "parse_failed": True, "reasoning": text.strip()[-1500:]}


def parse_mprometheus(text: str) -> Dict[str, Any]:
    """Extract the integer score from a Prometheus-2 style output.

    Returns dict with keys:
      score:        int in [1..5] or -1 on parse fail
      parse_failed: bool
      reasoning:    the text before [RESULT] (truncated)
      winner:       always "tie" (absolute mode; we keep the field for
                    downstream schema compatibility)
    """
    if not text:
        return {"score": -1, "parse_failed": True, "reasoning": "", "winner": "tie"}
    m = _RESULT_RE.search(text)
    if not m:
        return {"score": -1, "parse_failed": True, "reasoning": text[:1500], "winner": "tie"}
    try:
        score = int(m.group(1))
    except ValueError:
        return {"score": -1, "parse_failed": True, "reasoning": text[:1500], "winner": "tie"}
    if score < 1 or score > 5:
        # clamp + flag
        return {"score": max(1, min(5, score)), "parse_failed": True,
                "reasoning": text[:1500], "winner": "tie"}
    # Reasoning = text up to (but not including) [RESULT]
    reasoning = text[: m.start()].strip()
    if reasoning.lower().startswith("feedback:"):
        reasoning = reasoning[len("feedback:"):].strip()
    return {"score": score, "parse_failed": False,
            "reasoning": reasoning[:1500], "winner": "tie"}
