"""LEAF / GRPO training entrypoint for speech-aware LLMs (Ray + DeepSpeed + LoRA).

Two methods are supported, selected with ``--method``:

* ``leaf`` — Low-rank Exploration with Adaptive Forking: sample K complete responses
  i.i.d. per prompt (the rollout budget, shared with GRPO), select up to B high-surprisal
  fork boundaries (the fork budget), group responses by exact shared prefixes into a
  retrospective prefix tree, and assign span-level advantages (GA + LA, multiplicity-
  corrected by 1/sqrt(n), z-score normalised across the batch) optimised with a clipped
  policy loss plus an explicit KL term.  Knobs: ``--rollout_budget_K``, ``--fork_budget_B``.
* ``grpo`` — GRPO baseline: the same K rollouts per prompt with a single group-normalised
  terminal-reward advantage broadcast to every token, same clipped loss + explicit KL.

``--method`` resolves to the internal selectors read by the experience maker and trainer
(set in ``_set_defaults``):

    leaf : speech_tree_method=retro_batch, speech_tree_advantage_norm=zscore
    grpo : speech_tree_method=grpo_paper,  speech_tree_advantage_norm=none
    both : use_mcts=True, process_supervision=True, speech_tree_explicit_kl=True
"""
import argparse
import os
from pathlib import Path

import ray
import torch

from openrlhf.trainer.ray.launcher_reinforce import ReinforceRayActorGroup
from openrlhf.utils import get_strategy

METHODS = {
    # method -> internal (tree method, advantage normalisation)
    "leaf": ("retro_batch", "zscore"),
    "grpo": ("grpo_paper", "none"),
}


def _set_defaults(args):
    # Internal algorithm selectors derived from --method (see module docstring).
    tree_method, adv_norm = METHODS[args.method]
    args.speech_tree_method = tree_method
    args.speech_tree_advantage_norm = adv_norm
    args.use_mcts = True
    args.process_supervision = True
    args.speech_tree_explicit_kl = True

    defaults = {
        "adam_offload": True,
        "actor_init_on_gpu": False,
        "aux_loss_coef": 0.0,
        "bf16": True,
        "disable_trace_cache": False,
        "enable_ema": False,
        "eval_steps": -1,
        "eval_limit": 64,
        "eval_split": "validation",
        "eval_do_sample": True,
        "eval_temperature": 0.9,
        "eval_top_p": 0.9,
        "eval_max_new_tokens": 200,
        "flash_attn": False,
        "grad_accum_dtype": "fp32",
        "gradient_checkpointing_use_reentrant": False,
        "input_template": None,
        "kl_target": None,
        "l2_logits_loss_coeff": 0.0,
        "load_in_4bit": False,
        "load_ckpt": False,
        "local_rank": -1,
        "logging_steps": 1,
        "max_ckpt_mem": 1000,
        "max_ckpt_num": 3,
        "normalize_advantage": False,
        "perf": False,
        "pretrain_data": None,
        "pretrain_data_probs": "1.0",
        "print_rollout_samples": False,
        "ptx_coef": 0.0,
        "save_ckpt": False,
        "save_steps": -1,
        "top_k": 0,
        "use_mpi_init": False,
        "use_vinevalue": False,
        "use_wandb": None,
        "wandb_group": None,
        "wandb_id": None,
        "wandb_org": None,
        "wandb_project": "leaf_speech",
        "wandb_run_name": f"{args.method}_speech",
        "zpg": 1,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.load_ckpt = bool(args.load_ckpt)
    args.save_ckpt = bool(args.save_ckpt)
    args.reward_pretrain = args.pretrain
    if args.ref_pretrain is None:
        args.ref_pretrain = args.pretrain
    args.ckpt_path = args.save_path if not args.ckpt_path else args.ckpt_path
    Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)
    if args.max_len is None:
        args.max_len = args.prompt_max_len + args.generate_max_len
    actor_world_size = max(1, int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node))
    min_rollout_batch = actor_world_size * int(args.micro_rollout_batch_size)
    min_train_batch = actor_world_size * int(args.micro_train_batch_size)
    if args.rollout_batch_size < min_rollout_batch:
        print(
            f"Adjusting rollout_batch_size from {args.rollout_batch_size} to {min_rollout_batch} "
            f"to match actor world size {actor_world_size}.",
            flush=True,
        )
        args.rollout_batch_size = min_rollout_batch
    if args.train_batch_size < args.micro_train_batch_size:
        args.train_batch_size = args.micro_train_batch_size
    if args.train_batch_size < min_train_batch:
        print(
            f"Adjusting train_batch_size from {args.train_batch_size} to {min_train_batch} "
            f"to match actor world size {actor_world_size}.",
            flush=True,
        )
        args.train_batch_size = min_train_batch
    return args


def _ray_actor_classes(model_family: str):
    if model_family == "qwen":
        from openrlhf.speech_leaf.ray_openrlhf_qwen import QwenActorModelRayActor, QwenReferenceModelRayActor

        return QwenActorModelRayActor, QwenReferenceModelRayActor
    from openrlhf.speech_leaf.ray_openrlhf import SpeechActorModelRayActor, SpeechReferenceModelRayActor

    return SpeechActorModelRayActor, SpeechReferenceModelRayActor


def train(args):
    args = _set_defaults(args)
    strategy = get_strategy(args)
    if not ray.is_initialized():
        ray_kwargs = {"ignore_reinit_error": True}
        ray_tmpdir = os.environ.get("RAY_TMPDIR")
        if ray_tmpdir:
            Path(ray_tmpdir).mkdir(parents=True, exist_ok=True)
            ray_kwargs["_temp_dir"] = ray_tmpdir
            print(f"Using Ray temp dir: {ray_tmpdir}", flush=True)
        ray.init(**ray_kwargs)

    actor_cls, ref_cls = _ray_actor_classes(args.model_family)
    actor_model = ReinforceRayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        actor_cls,
        num_gpus_per_actor=args.actor_num_gpus_per_actor,
    )
    ref_model = ReinforceRayActorGroup(
        args.ref_num_nodes,
        args.ref_num_gpus_per_node,
        ref_cls,
        num_gpus_per_actor=args.ref_num_gpus_per_actor,
    )

    refs = []
    refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.ref_pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))
    ray.get(refs)

    fit_refs = actor_model.async_fit_actor_model(
        ref_model,
        [],
        reward_fn=lambda rewards: rewards[0] if rewards else torch.tensor(0.0),
        vllm_engines=None,
    )
    ray.get(fit_refs)
    ray.get(actor_model.async_save_actor_model())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── method / model ──
    parser.add_argument("--method", type=str, required=True, choices=sorted(METHODS),
                        help="leaf (LEAF prefix-tree RL) or grpo (paper-exact GRPO chain baseline)")
    parser.add_argument("--model_family", type=str, default="granite", choices=["granite", "qwen"],
                        help="granite: IBM Granite Speech; qwen: Qwen2-Audio (different loaders/actors)")
    parser.add_argument("--pretrain", type=str, default="ibm-granite/granite-speech-3.3-2b",
                        help="Base model, or a saved LoRA adapter dir to warm-start from (with --continue_lora 1)")
    parser.add_argument("--ref_pretrain", type=str, default=None,
                        help="Checkpoint for the frozen KL reference model (defaults to --pretrain)")
    # ── Ray / GPU layout ──
    parser.add_argument("--ref_num_nodes", type=int, default=1)
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=1)
    parser.add_argument("--ref_num_gpus_per_actor", type=float, default=1.0)
    parser.add_argument("--actor_num_nodes", type=int, default=1)
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=1)
    parser.add_argument("--actor_num_gpus_per_actor", type=float, default=1.0)
    # ── data / output ──
    parser.add_argument("--speech_data_dir", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    # ── batching / schedule ──
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of passes over the training prompts")
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1, help="Optimisation epochs per rollout batch")
    parser.add_argument("--prompt_max_len", type=int, default=512)
    parser.add_argument("--generate_max_len", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=None)
    # ── optimisation ──
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--lambd", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--adam_offload", action="store_true", default=None)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--actor_learning_rate", type=float, default=5e-6)
    parser.add_argument("--min_actor_learning_rate_lr", type=float, default=0.1)
    parser.add_argument("--init_kl_coef", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    # ── reward ──
    parser.add_argument("--reward_mode", type=str, default="bleu", choices=["bleu", "binary"])
    parser.add_argument("--lowercase_bleu", type=int, default=1,
                        help="Use case-insensitive BLEU for the reward (1=yes, 0=no)")
    # ── periodic validation / checkpoints ──
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--eval_limit", type=int, default=64)
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--eval_do_sample", type=int, default=1)
    parser.add_argument("--eval_temperature", type=float, default=0.9)
    parser.add_argument("--eval_top_p", type=float, default=0.9)
    parser.add_argument("--eval_max_new_tokens", type=int, default=200)
    parser.add_argument("--eval_lowercase", type=int, default=0,
                        help="Lowercase preds & refs before ALL eval metrics (1=yes, 0=no)")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--load_ckpt", type=int, default=None)
    parser.add_argument("--save_ckpt", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    # ── rollout / fork budgets ──
    parser.add_argument("--rollout_budget_K", type=int, default=8,
                        help="K = complete responses sampled i.i.d. per prompt (both methods)")
    parser.add_argument("--fork_budget_B", type=int, default=2,
                        help="LEAF: B = maximum number of selected fork boundaries")
    parser.add_argument("--fork_min_prefix_tokens", type=int, default=1,
                        help="LEAF: earliest token boundary eligible for fork selection")
    parser.add_argument("--advantage_terms", type=str, default="both",
                        choices=["both", "global", "local"],
                        help="LEAF ablation: span-credit terms (both=GA+LA, the default; "
                             "global=GA only; local=LA only)")
    # ── generation batching (both methods) ──
    parser.add_argument("--gen_prompt_batch_size", type=int, default=1,
                        help="Number of prompts batched together in one generate() call "
                             "(1 = per-prompt generation, the setting used for the paper runs)")
    # ── LoRA ──
    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--continue_lora", type=int, default=1,
                        help="1 = if --pretrain is a LoRA adapter dir, keep training that adapter")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    # ── misc ──
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--print_rollout_samples", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    if args.ckpt_path:
        Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)
    train(args)
