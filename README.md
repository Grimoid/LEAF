# LEAF: Growing Trees Without Branching for Speech-Aware LLM Post-Training

Code for **LEAF** (**L**ow-rank **E**xploration with **A**daptive **F**orking), a retrospective
tree-based RL method for speech-aware large language model (SALLM) post-training, and the
**GRPO** baseline it is compared against — on IBM Granite Speech 3.3 2B/8B, Granite-4.0-1B-Speech
and Qwen2-Audio-7B, for spoken question answering (LibriSQA, DailyTalk, LongAudio) and speech
translation (CoVoST2 En→De). This repository accompanies the EMNLP 2026 paper
*"LEAF: Growing Trees Without Branching for Speech-Aware Large Language Model Post-Training"*.

The training stack is Ray + DeepSpeed (ZeRO-2) + LoRA with in-actor HuggingFace generation.
This is the code that produced the results reported in the paper; the default configuration
(`scripts/configs/defaults.yaml`) is the configuration of those runs.

---

## Methods

| `--method` | What it does | Knobs |
|---|---|---|
| `leaf` | **LEAF.** Sample **K** complete responses i.i.d. per prompt (the rollout budget, identical to GRPO's) and record token surprisals. Greedily select up to **B** fork boundaries (the fork budget) in descending surprisal, subject to the separation rule `Δ = max(2, ⌊ℓ_min/(B+1)⌋)`. At each selected boundary, group responses by exact token-prefix equality; non-singleton groups become retained prefix nodes with value `V̂(v)` = mean terminal reward of their descendants. The retained nodes on each response partition it into spans: an internal span ending at node `v` gets the advantage `(GA + LA)/√n(v)` with `GA = V̂(v) − V̂(v₀)` (root-relative) and `LA = V̂(v) − V̂(parent)`, where `1/√n(v)` is the multiplicity correction; the tail span gets `(r − V̂(v₀)) + (r − V̂(v_last))`. Raw span advantages are z-score normalised across the sampled batch, then optimised with the clipped token loss (`ε = 0.2`) plus an explicit KL penalty to the reference model. | `--rollout_budget_K` (K = 8), `--fork_budget_B` (B = 2) |
| `grpo` | **GRPO** baseline. The same K i.i.d. responses per prompt; one group-normalised terminal-reward advantage broadcast to every token (`std` floored at 1/3); same clipped loss + explicit KL. | `--rollout_budget_K` (K = 8) |

Shared hyperparameters (paper Table 7): sentence-BLEU reward (**case-sensitive** during
training; all reported evaluations are **case-insensitive**), LoRA rank 64 / alpha 128 /
dropout 0.05, learning rate 5e-6, KL coefficient 0.02, rollout batch 6 / training batch 6 /
micro-batch 1, max prompt length 256, max response length 200, sampling temperature 1.0,
top-p 0.9. Epochs per (model, dataset) follow paper Table 9 and are run as warm restarts.

---

## Setup

```bash
conda create -p ./.conda python=3.10 -y && conda activate ./.conda
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .                      # or: export PYTHONPATH=$PWD (speechrl.sh does this)
```

A second environment with vLLM is used for the judge-based evaluations
(`requirements-judge.txt`), so the judge models do not constrain the training stack.

### Data (not included)

**No datasets are distributed with this repository** (their licenses do not permit
redistribution). The scripts in [`scripts/data_prep/`](scripts/data_prep/README.md) download
each corpus from its original source (LibriSpeech + LibriSQA; DailyTalk and the Europarl-ST /
VoxPopuli LongAudio splits; CoVoST2 / Common Voice) and build the HuggingFace `save_to_disk`
directories that training and evaluation consume: a `dataset/` subfolder whose rows are
`{"prompt", "reference", "audio"}`. Obtain the source corpora under their own licenses
(Common Voice requires an account/API key), then point `SPEECH_DATA_ROOT` at the directory
holding the prepared datasets (default `./data`). The dataset registry is
`scripts/configs/datasets.yaml` (`librisqa`, `dailytalk`, `longaudio`, `covost2`).

---

## Training

```bash
export SPEECH_DATA_ROOT=/path/to/prepared/datasets
# LEAF, Granite Speech 3.3 2B, LibriSQA (3 GPUs: 2 actors + 1 reference)
bash scripts/speechrl.sh train --method leaf --model granite3_2b --dataset librisqa --out ckpts/leaf_librisqa --log
# GRPO baseline, same everything
bash scripts/speechrl.sh train --method grpo --model granite3_2b --dataset librisqa --out ckpts/grpo_librisqa --log

# override any knob; --dry-run prints the exact python command
bash scripts/speechrl.sh train --method leaf --model granite3_2b --dataset librisqa \
     --set rollout_budget_K=8 --set fork_budget_B=4 --set num_episodes=3 --dry-run
```

* Re-running with the same `--out` dir **resumes** from the latest DeepSpeed checkpoint.
* **Next epoch as a warm restart** (the paper protocol: fresh optimiser and LR schedule,
  continue the LoRA adapter, KL reference stays the base model):
  `cp -r <run>/_actor_global_step<LAST> <run>_e1_final` then
  `bash scripts/speechrl.sh train ... --warm-start <run>_e1_final --out <run>_ep2`.
  (Copy the adapter to a directory **not** ending in `_actor_global_step<N>`; that suffix is
  interpreted as "prompt batches already consumed".)
* `--fork-stats` logs per-rollout fork-fire statistics (`fork_position_stats_rank*.jsonl`) for the
  fork-fire analyses; it does not change training.
* `scripts/slurm/train.sbatch` is a Slurm template (3 GPUs, resumable).
* Models: `granite3_2b`, `granite3_8b`, `granite4_1b`, `qwen2_7b` (`scripts/configs/models.yaml`).
  The Qwen2-Audio path uses its own loaders/actors (`--model_family qwen`).

Every `eval_steps` steps the trainer runs a sampled validation pass (`eval_limit` items) and
writes `<run>/eval_log.jsonl` (BLEU, exact match, response length), saving an adapter
`_actor_global_step<N>`; `save_steps` adds DeepSpeed resume checkpoints.

The underlying entrypoint is `train_reinforce_ray_speech.py`; its docstring lists the
`--method` → internal-flag mapping. All defaults (`scripts/configs/defaults.yaml`) are the
values used for the reported runs; only the GPU layout is hardware-specific.

---

## Evaluation

### 1. Text metrics on checkpoints
```bash
bash scripts/speechrl.sh eval --model granite3_2b --dataset librisqa --run ckpts/leaf_librisqa --mode all
bash scripts/speechrl.sh eval --model granite3_2b --dataset librisqa --run ckpts/leaf_librisqa --mode topk --top-k 5
bash scripts/speechrl.sh eval --model granite3_2b --dataset librisqa --checkpoint <ckpt> --out r.json
```
`scripts/eval/eval_librisqa.py` / `eval_covost2.py` (+ `_qwen`) report BLEU, ROUGE-1/2/L,
METEOR and BERTScore-F1, averaged over `num_runs=5` sampled passes (T=0.9, top-p 0.9);
`eval_covost2_plus.py` additionally stages per-item translation predictions for the judging
phase. Multi-GPU sharding is automatic.

### 2. LLM-as-judge (spoken QA) — `scripts/judge/`
Paper protocol for one checkpoint (predictions → GEMBA-style **DA-100** with
Qwen2.5-14B-Instruct, 5 samples at T=0.7, median → **Likert-5** with M-Prometheus-14B):
```bash
RUN_DIR=ckpts/leaf_librisqa STEP=11200 TRAIN_PYTHON=.conda/bin/python JUDGE_PYTHON=.conda.judge/bin/python \
bash scripts/judge/judge_checkpoint.sh
```
**Pairwise** head-to-head (JudgeLM-13B or M-Prometheus-14B, both A/B orderings judged, a verdict
counts only when the two orderings agree):
```bash
PRED_A=<leaf>/librisqa_judge_predictions/step_11200.jsonl PRED_B=<grpo>/librisqa_judge_predictions/step_13800.jsonl \
LABEL_A=leaf LABEL_B=grpo RESULTS_JSONL=results/pairwise_leaf_vs_grpo.jsonl \
bash scripts/judge/run_judge_predictions_pairwise_judgelm.sh        # or ..._pairwise_mprometheus.sh (RUBRIC_KEY=qa|mt)
```
See [`scripts/judge/README.md`](scripts/judge/README.md) for the full list of judges and drivers.

### 3. VoiceBench open-ended QA (out-of-distribution) — `scripts/voicebench/`
Generates responses for the `alpacaeval`, `commoneval` and `wildvoice` subsets of
[VoiceBench](https://github.com/MatthewCYM/VoiceBench) and scores them with an open judge
(M-Prometheus-14B) using VoiceBench's own 1–5 open-QA rubric:
```bash
LEAF_ADAPTER=<leaf>/_actor_global_step11200 GRPO_ADAPTER=<grpo>/_actor_global_step13800 \
JUDGE_PYTHON=.conda.judge/bin/python OUT_DIR=results/voicebench bash scripts/voicebench/run_voicebench.sh
```
See [`scripts/voicebench/README.md`](scripts/voicebench/README.md).

### 4. Forkability / fork-fire statistics — `scripts/fork_stats/`
Measures the paper's *forkability* property: the usable fork-fire rate (how often a selected
boundary yields a non-singleton prefix group with non-trivial reward variation) and where along
the response fork-fires occur (`collect_fork_fire_stats.py` on a checkpoint, or `summarize_*`
on `--fork-stats` training logs), plus the plotting scripts for the fork-fire-rate figures.
See [`scripts/fork_stats/README.md`](scripts/fork_stats/README.md).

---

## Repository layout

```
train_reinforce_ray_speech.py   training entrypoint (--method leaf|grpo, --model_family granite|qwen)
openrlhf/
  speech_leaf/                the speech RL package
    sampler.py                  K i.i.d. rollouts + retrospective prefix-tree construction (LEAF)
    experience_maker_openrlhf.py rollouts -> rewards -> advantages -> KL (leaf / grpo)
    trainer_openrlhf.py         training step (clipped loss + explicit KL), validation, checkpoints
    loss.py, reward.py          leaf_policy_loss, sentence-BLEU reward
    ray_openrlhf.py, actor.py, loaders.py, replay_buffer.py, model_utils.py, metrics.py
    *_qwen.py                   Qwen2-Audio variants
    fork_analysis.py, fork_position_logging.py   fork statistics (analysis only)
  trainer/, models/, utils/, datasets/   RL framework core (Ray actor groups, DeepSpeed strategy, ReinforceTrainer)
scripts/
  speechrl.py, speechrl.sh, configs/   config-driven launcher (methods / models / datasets / defaults)
  data_prep/                    dataset construction
  eval/                         text-metric evaluation
  judge/                        LLM-as-judge (DA-100, Likert, pairwise)
  voicebench/                   VoiceBench open-ended QA evaluation
  fork_stats/                   fork-fire statistics and figures
  slurm/                        Slurm template
```

The `openrlhf/trainer`, `openrlhf/models`, `openrlhf/utils` and `openrlhf/datasets` packages
are the underlying RL-framework core needed by the Ray + DeepSpeed training loop; the speech
code does not use their math/text-reasoning parts (`parallel_mcts`, `evaluation`, …), which are
kept only because the trainer package imports them.

## Author, license and acknowledgements

Code by **Argyrios Gerogiannis**. Licensed under Apache 2.0 (see `LICENSE` and `NOTICE`).
The Ray/DeepSpeed RL infrastructure builds on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
and [TreeRL](https://github.com/THUDM/TreeRL) (Hou et al., ACL 2025), which are gratefully
acknowledged. The GRPO baseline follows Elmakies et al., "Advancing Speech Understanding in
Speech-Aware Language Models with GRPO"; VoiceBench is Chen et al., TACL 2026.
