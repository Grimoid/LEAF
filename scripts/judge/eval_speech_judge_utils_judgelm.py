"""JudgeLM (BAAI/JudgeLM-*-v1.0) prompt + parser for pairwise judging.

JudgeLM uses a Vicuna v1.1 chat format with a specific template that places
the two candidate answers between "[The Start of Assistant N's Answer]" / "[The
End of Assistant N's Answer]" markers, and asks the judge to emit two
space-separated scores (Assistant 1, Assistant 2) on the first line, followed
by free-form reasoning.

Reference:
  https://github.com/baaivision/JudgeLM/blob/main/judgelm/llm_judge/common.py
"""
from __future__ import annotations

from typing import Any, Dict


JUDGELM_SYSTEM = (
    "You are a helpful and precise assistant for checking the quality of the answer."
)

# Verbatim from JudgeLM's eval/with_reference template.
JUDGELM_RUBRIC = (
    "Please rate the helpfulness, relevance, accuracy, level of details of their "
    "responses. Each assistant receives an overall score on a scale of 1 to 10, "
    "where a higher score indicates better overall performance. Please first output "
    "a single line containing only two values indicating the scores for Assistant 1 "
    "and 2, respectively. The two scores are separated by a space. In the subsequent "
    "line, please provide a comprehensive explanation of your evaluation, avoiding "
    "any potential bias and ensuring that the order in which the responses were "
    "presented does not affect your judgment."
)


def _user_block(question: str, reference: str, a1: str, a2: str) -> str:
    return (
        f"[Question]\n{question}\n\n"
        f"[Reference Answer]\n{reference}\n\n"
        f"[The Start of Assistant 1's Answer]\n{a1}\n\n"
        f"[The End of Assistant 1's Answer]\n\n"
        f"[The Start of Assistant 2's Answer]\n{a2}\n\n"
        f"[The End of Assistant 2's Answer]\n\n"
        f"[System]\n{JUDGELM_RUBRIC}\n\n"
    )


def build_judgelm_prompt(record: Dict[str, Any]) -> str:
    """Vicuna v1.1 chat format: 'SYSTEM USER: ... ASSISTANT:'.

    `record` keys used: question, reference, candidate_a, candidate_b.
    """
    question = str(record.get("question") or record.get("prompt") or "").strip()
    reference = str(record.get("reference") or "").strip()
    a1 = str(record.get("candidate_a") or "").strip()
    a2 = str(record.get("candidate_b") or "").strip()
    user = _user_block(question, reference, a1, a2)
    return f"{JUDGELM_SYSTEM} USER: {user}ASSISTANT:"


def parse_judgelm(text: str) -> Dict[str, Any]:
    """Extract (score_1, score_2) from JudgeLM's first line.

    Returns a dict with the pairwise-verdict key set:
      - score_1, score_2 (floats; -1.0 on parse fail)
      - winner: 'a' | 'b' | 'tie'
      - correctness_winner / grounding_winner / completeness_winner / hallucination_winner:
          all set to `winner` because JudgeLM emits a single score per assistant
          (not 4 sub-dimensions). Filled in so downstream aggregation works.
      - parse_failed: bool
      - reasoning: free-text body
    """
    if not text:
        return {"score_1": -1.0, "score_2": -1.0, "winner": "tie",
                "correctness_winner": "tie", "grounding_winner": "tie",
                "completeness_winner": "tie", "hallucination_winner": "tie",
                "parse_failed": True, "reasoning": ""}

    first_line = text.split("\n", 1)[0].replace(",", " ").strip()
    parts = [p for p in first_line.split(" ") if p]
    score_1 = score_2 = -1.0
    parse_failed = True
    try:
        if len(parts) >= 2:
            score_1 = float(parts[0])
            score_2 = float(parts[1])
            parse_failed = False
    except ValueError:
        pass

    if parse_failed:
        winner = "tie"
    elif score_1 > score_2:
        winner = "a"
    elif score_2 > score_1:
        winner = "b"
    else:
        winner = "tie"

    reasoning_body = text.split("\n", 1)[1].strip() if "\n" in text else ""

    return {
        "score_1": score_1,
        "score_2": score_2,
        "winner": winner,
        # JudgeLM doesn't grade per-dimension; mirror the overall winner so
        # the downstream aggregation schema is preserved.
        "correctness_winner": winner,
        "grounding_winner": winner,
        "completeness_winner": winner,
        "hallucination_winner": winner,
        "parse_failed": parse_failed,
        "reasoning": reasoning_body[:1500],
    }
