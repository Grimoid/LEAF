# VoiceBench open-ended QA (`scripts/voicebench/`)

Cross-benchmark generalisation probe (paper Figure 3): the policies are trained on spoken-content QA
(LibriSQA-style), VoiceBench (Chen et al., TACL 2026, arXiv:2410.17196) measures spoken
*instruction following*. Both share the free-form text answer format, which is what makes
the comparison valid for a policy rewarded on free-form text.

## Subsets

Only the subsets VoiceBench itself scores with its `open` (1–5 LLM-judge) evaluator:

| subset | items | audio | task |
|---|---|---|---|
| `alpacaeval` | 199 | TTS | open-ended QA |
| `commoneval` | 200 | human-recorded | open-ended QA |
| `wildvoice` | 1000 | human-recorded, diverse accents | open-ended QA |

The constrained-format subsets (`openbookqa`, `mmsu` multiple choice; `ifeval`; `advbench`;
`mtbench` multi-turn) are deliberately excluded: their answer format does not match the
training objective.

## Pipeline

1. `voicebench_generate.py` — loads `hlt-lab/voicebench` (HF hub), wraps Granite Speech +
   adapter, prompt `"You will hear a request in <|audio|>. Respond to it."` (same chat template
   as training), greedy decoding, `max_new_tokens=256`; writes `<method>__<subset>.jsonl`.
2. `voicebench_judge_prometheus.py` — scores each `(instruction, response)` with
   Unbabel/M-Prometheus-14B (vLLM, bf16, greedy) using VoiceBench's **verbatim** open-QA rubric
   (1–5); parses plain / `[[n]]` / `[RESULT] n` outputs. Writes `*.scored.jsonl`.
3. `voicebench_aggregate.py` — mean score per (method, subset) and the LEAF−GRPO difference
   → `VOICEBENCH_RESULTS.md`.

`run_voicebench.sh` runs all three (`LEAF_ADAPTER`, `GRPO_ADAPTER`, `OUT_DIR`, `JUDGE_PYTHON`).

## Caveats

* The judge is an open model instead of VoiceBench's `gpt-4o-mini`, so absolute scores are
  **not** comparable with the public leaderboard; the LEAF-vs-GRPO comparison is internally
  valid (identical judge, prompt and decoding for both).
* Judge sampling also differs from the official protocol: VoiceBench queries its API judge
  3 times at T=0.5 / top-p 0.95 and averages all three scores per item; here the open judge is
  called once, greedily (deterministic, slightly less noisy). Score parsing additionally accepts
  Prometheus's `[RESULT] n` format (the official parser only handles a bare number or `[[n]]`).
* VoiceBench has no Granite wrapper, so the model side (chat-template prompt above, greedy,
  `max_new_tokens=256`) is defined here; every VoiceBench model uses its own wrapper anyway.
* Task shift, not format shift: absolute scores reflect generalisation, not in-distribution
  quality.
