from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


_PATCH_LOCK = threading.Lock()
_THREAD_STATE = threading.local()
_PATCHED = False


@dataclass
class PromptForkStats:
    attempted_positions: int
    selected_positions: int
    selected_position_indices: List[int]
    usable_positions: int
    usable_position_indices: List[int]
    group_sizes: List[int]
    reward_spreads: List[float]
    num_sequences: int


class BatchForkStatsCollector:
    def __init__(self, maker, prompts, tree_cfg) -> None:
        args = maker.strategy.args
        self.save_path = Path(args.save_path)
        self.rank = int(maker.strategy.get_rank()) if hasattr(maker.strategy, "get_rank") else int(getattr(args, "rank", 0))
        self.task_name = os.environ.get("FORK_GROUP_TASK_NAME") or Path(str(getattr(args, "speech_data_dir", "unknown"))).name
        self.num_prompts = len(prompts)
        self.attempted_per_prompt = int(tree_cfg.num_forks)   # = B (fork boundaries)
        self.rollout_budget_K = int(tree_cfg.total_seqs)           # rollout budget K
        self.fork_budget_B = int(tree_cfg.num_forks)               # fork budget B
        self.model_name = str(getattr(args, "pretrain", ""))
        self.call_index = int(getattr(maker, "_fork_group_log_counter", 0)) + 1
        maker._fork_group_log_counter = self.call_index
        self.prompt_stats: list[PromptForkStats] = []

    def record(self, prompt_stats: PromptForkStats) -> None:
        self.prompt_stats.append(prompt_stats)

    def finalize(self) -> dict:
        usable_positions = sum(item.usable_positions for item in self.prompt_stats)
        selected_positions = sum(item.selected_positions for item in self.prompt_stats)
        selected_position_indices = [
            position for item in self.prompt_stats for position in item.selected_position_indices
        ]
        usable_position_indices = [
            position for item in self.prompt_stats for position in item.usable_position_indices
        ]
        group_sizes = [size for item in self.prompt_stats for size in item.group_sizes]
        reward_spreads = [spread for item in self.prompt_stats for spread in item.reward_spreads]

        avg_group_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0.0
        avg_reward_spread = sum(reward_spreads) / len(reward_spreads) if reward_spreads else 0.0
        avg_selected_position = sum(selected_position_indices) / len(selected_position_indices) if selected_position_indices else 0.0
        avg_usable_position = sum(usable_position_indices) / len(usable_position_indices) if usable_position_indices else 0.0

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_name": self.task_name,
            "rank": self.rank,
            "call_index": self.call_index,
            "num_prompts": self.num_prompts,
            "attempted_fork_positions": self.num_prompts * self.attempted_per_prompt,
            "attempted_fork_positions_per_prompt": self.attempted_per_prompt,
            "selected_fork_positions": selected_positions,
            "usable_fork_positions": usable_positions,
            "usable_fork_positions_per_prompt": (usable_positions / self.num_prompts) if self.num_prompts else 0.0,
            "usable_fork_fraction": (
                usable_positions / (self.num_prompts * self.attempted_per_prompt)
                if self.num_prompts and self.attempted_per_prompt
                else 0.0
            ),
            "usable_group_count": len(group_sizes),
            "avg_group_size": avg_group_size,
            "max_group_size": max(group_sizes) if group_sizes else 0,
            "avg_selected_position": avg_selected_position,
            "max_selected_position": max(selected_position_indices) if selected_position_indices else -1,
            "avg_usable_position": avg_usable_position,
            "max_usable_position": max(usable_position_indices) if usable_position_indices else -1,
            "avg_reward_spread": avg_reward_spread,
            "max_reward_spread": max(reward_spreads) if reward_spreads else 0.0,
            "group_sizes": group_sizes,
            "selected_position_indices": selected_position_indices,
            "usable_position_indices": usable_position_indices,
            "reward_spreads": reward_spreads,
            "rollout_budget_K": self.rollout_budget_K,
            "fork_budget_B": self.fork_budget_B,
            "model_name": self.model_name,
        }

    def log(self) -> None:
        record = self.finalize()
        self.save_path.mkdir(parents=True, exist_ok=True)
        log_path = self.save_path / f"fork_position_stats_rank{self.rank}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(
            "[fork-position-stats] "
            f"task={record['task_name']} "
            f"rank={record['rank']} "
            f"call={record['call_index']} "
            f"prompts={record['num_prompts']} "
            f"attempted={record['attempted_fork_positions']} "
            f"selected_positions={record['selected_fork_positions']} "
            f"usable_positions={record['usable_fork_positions']} "
            f"usable_groups={record['usable_group_count']} "
            f"avg_group_size={record['avg_group_size']:.3f} "
            f"avg_selected_position={record['avg_selected_position']:.3f} "
            f"max_group_size={record['max_group_size']} "
            f"avg_position={record['avg_usable_position']:.3f} "
            f"avg_reward_spread={record['avg_reward_spread']:.6f}",
            flush=True,
        )


def _get_collector() -> Optional[BatchForkStatsCollector]:
    return getattr(_THREAD_STATE, "collector", None)


def _set_collector(collector: Optional[BatchForkStatsCollector]) -> None:
    _THREAD_STATE.collector = collector


def _compute_prompt_fork_stats(all_trajs, rewards, cfg) -> PromptForkStats:
    num_forks = int(cfg.num_forks)
    if not all_trajs:
        return PromptForkStats(
            attempted_positions=num_forks,
            selected_positions=0,
            selected_position_indices=[],
            usable_positions=0,
            usable_position_indices=[],
            group_sizes=[],
            reward_spreads=[],
            num_sequences=0,
        )

    candidates = []
    for seq_idx, traj in enumerate(all_trajs):
        surprisal = traj.surprisal
        if surprisal is None:
            continue
        gen_len = int(traj.gen_token_ids.numel())
        for pos in range(max(int(cfg.min_prefix_tokens), 1), gen_len):
            candidates.append((float(surprisal[pos]), seq_idx, pos))

    if not candidates:
        return PromptForkStats(
            attempted_positions=num_forks,
            selected_positions=0,
            selected_position_indices=[],
            usable_positions=0,
            usable_position_indices=[],
            group_sizes=[],
            reward_spreads=[],
            num_sequences=len(all_trajs),
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    gen_lens = [int(traj.gen_token_ids.numel()) for traj in all_trajs if traj.gen_token_ids.numel() > 0]
    min_gen_len = min(gen_lens) if gen_lens else 1
    min_fork_sep = max(2, min_gen_len // (max(num_forks, 1) + 1))

    fork_positions = []
    for _, _, pos in candidates:
        if all(abs(pos - existing_pos) >= min_fork_sep for existing_pos in fork_positions):
            fork_positions.append(pos)
            if len(fork_positions) >= num_forks:
                break
    fork_positions.sort()

    usable_position_indices = []
    group_sizes = []
    reward_spreads = []

    for fork_pos in fork_positions:
        groups = {}
        for seq_idx, traj in enumerate(all_trajs):
            gen_ids = traj.gen_token_ids
            if int(gen_ids.numel()) <= fork_pos:
                prefix_key = tuple(gen_ids.tolist())
            else:
                prefix_key = tuple(gen_ids[:fork_pos].tolist())
            groups.setdefault(prefix_key, []).append(seq_idx)

        usable_here = False
        for members in groups.values():
            if len(members) <= 1:
                continue
            usable_here = True
            group_sizes.append(len(members))
            member_rewards = [float(rewards[idx]) for idx in members]
            reward_spreads.append(max(member_rewards) - min(member_rewards))
        if usable_here:
            usable_position_indices.append(fork_pos)

    return PromptForkStats(
        attempted_positions=num_forks,
        selected_positions=len(fork_positions),
        selected_position_indices=fork_positions,
        usable_positions=len(usable_position_indices),
        usable_position_indices=usable_position_indices,
        group_sizes=group_sizes,
        reward_spreads=reward_spreads,
        num_sequences=len(all_trajs),
    )


def patch_experience_maker_with_fork_position_logging() -> None:
    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return

        from openrlhf.speech_leaf import sampler as sampler_mod
        import openrlhf.speech_leaf.experience_maker_openrlhf as exp_mod

        original_build = sampler_mod.build_prefix_tree_from_trajectories
        original_generate_batch = sampler_mod.generate_batch
        original_make_experience = exp_mod.SpeechRemoteExperienceMaker.make_experience

        def build_with_logging(all_trajs, reward_fn: Callable[[str], float], prefix_tree_cfg=None):
            cfg = prefix_tree_cfg or sampler_mod.PrefixTreeConfig()
            rewards = [float(reward_fn(traj.text)) for traj in all_trajs] if all_trajs else []
            collector = _get_collector()
            if collector is not None:
                collector.record(_compute_prompt_fork_stats(all_trajs, rewards, cfg))
            return original_build(all_trajs, reward_fn, prefix_tree_cfg)

        def sample_with_logging(
            model,
            tokenizer,
            model_inputs,
            gen_cfg,
            reward_fn: Callable[[str], float],
            prefix_tree_cfg=None,
        ):
            cfg = prefix_tree_cfg or sampler_mod.PrefixTreeConfig()
            total_seqs = cfg.total_seqs
            all_trajs = original_generate_batch(
                model=model,
                tokenizer=tokenizer,
                model_inputs=model_inputs,
                cfg=gen_cfg,
                num_return_sequences=total_seqs,
                compute_surprisal=True,
            )
            rewards = [float(reward_fn(traj.text)) for traj in all_trajs] if all_trajs else []
            collector = _get_collector()
            if collector is not None:
                collector.record(_compute_prompt_fork_stats(all_trajs, rewards, cfg))
            return original_build(all_trajs, reward_fn, cfg)

        def make_experience_with_logging(self, prompts, use_mcts: bool, use_vinevalue: bool = False, **generate_kwargs):
            should_log = bool(use_mcts) and str(getattr(self.strategy.args, "speech_tree_method", "retro_batch")) == "retro_batch"
            if not should_log:
                return original_make_experience(self, prompts, use_mcts, use_vinevalue=use_vinevalue, **generate_kwargs)

            collector = BatchForkStatsCollector(self, prompts, self._tree_cfg())
            previous_collector = _get_collector()
            _set_collector(collector)
            try:
                experience = original_make_experience(
                    self,
                    prompts,
                    use_mcts,
                    use_vinevalue=use_vinevalue,
                    **generate_kwargs,
                )
            finally:
                _set_collector(previous_collector)
            collector.log()
            return experience

        exp_mod.build_prefix_tree_from_trajectories = build_with_logging
        exp_mod.sample_prefix_tree_retro_batch = sample_with_logging
        exp_mod.SpeechRemoteExperienceMaker.make_experience = make_experience_with_logging

        _PATCHED = True
