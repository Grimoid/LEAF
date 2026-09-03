from __future__ import annotations

from dataclasses import dataclass

from sacrebleu import sentence_bleu

from .text_utils import normalize_text


@dataclass
class RewardConfig:
    mode: str = "bleu"
    lowercase_bleu: bool = True


@dataclass
class BinaryRewardConfig:
    exact_match: bool = True


def bleu_reward(reference: str, hypothesis: str, lowercase: bool = True) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    return float(sentence_bleu(hyp, [ref], lowercase=lowercase).score / 100.0)


def binary_reward(reference: str, hypothesis: str, cfg: BinaryRewardConfig | None = None) -> float:
    _ = cfg
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    return 1.0 if hyp == ref else 0.0


def compute_reward(
    reference: str,
    hypothesis: str,
    cfg: RewardConfig,
    binary_cfg: BinaryRewardConfig | None = None,
) -> float:
    if cfg.mode == "binary":
        return binary_reward(reference, hypothesis, binary_cfg)
    return bleu_reward(reference, hypothesis, lowercase=cfg.lowercase_bleu)
