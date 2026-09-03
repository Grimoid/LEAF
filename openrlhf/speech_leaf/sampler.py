from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _combine_adv(ga: float, la: float, terms: str = "both") -> float:
    """Which credit terms enter the LEAF tree process reward.
    'global' = GA only (node vs root), 'local' = LA only (node vs parent), else GA+LA (the default)."""
    if terms == "global":
        return ga
    if terms == "local":
        return la
    return ga + la


@dataclass
class GenConfig:
    max_new_tokens: int = 200
    temperature: float = 1.2
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True


@dataclass
class PrefixTreeConfig:
    # LEAF is parametrised by exactly two independent knobs (paper notation):
    #   K = rollout budget: complete responses sampled i.i.d. per prompt (= total_seqs)
    #   B = fork budget: maximum number of selected fork boundaries      (= num_forks)
    K: int = 8
    B: int = 2
    min_prefix_tokens: int = 1
    advantage_terms: str = "both"  # both=GA+LA (default) | global=GA only | local=LA only

    @property
    def total_seqs(self) -> int:
        """Rollout budget K: complete responses sampled i.i.d. per prompt."""
        return max(1, int(self.K))

    @property
    def num_forks(self) -> int:
        """Fork budget B: maximum number of selected fork boundaries."""
        return max(0, int(self.B))


@dataclass
class PrefixTreeTrainStep:
    step_start: int
    step_end: int
    advantage: float


@dataclass
class PrefixTreeTrainSequence:
    gen_token_ids: torch.Tensor
    logp: torch.Tensor
    text: str
    steps: List[PrefixTreeTrainStep]
    leaf_reward: float


@dataclass
class Trajectory:
    text: str
    gen_token_ids: torch.Tensor
    logp: torch.Tensor
    surprisal: Optional[torch.Tensor] = None


@dataclass
class _PrefixTreeNode:
    trajectory: Trajectory
    tree_idx: int
    prefix_len: int = 0
    reward: float = 0.0
    mask: List[bool] = field(default_factory=list)

    def __post_init__(self):
        n_tokens = int(self.trajectory.gen_token_ids.numel())
        if not self.mask:
            self.mask = [False] * n_tokens
        for pos in range(min(self.prefix_len, n_tokens)):
            self.mask[pos] = True
        if self.prefix_len < n_tokens and self.prefix_len > 0:
            self.mask[self.prefix_len] = True

    def prefix_ids(self, split_pos: int) -> torch.Tensor:
        return self.trajectory.gen_token_ids[:split_pos].clone()


@torch.no_grad()
def _repeat_model_inputs(model_inputs: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    if n <= 1:
        return model_inputs
    repeated: Dict[str, torch.Tensor] = {}
    for key, value in model_inputs.items():
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == 1:
            repeated[key] = value.repeat(n, *([1] * (value.dim() - 1)))
        else:
            repeated[key] = value
    return repeated


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    model_inputs: Dict[str, torch.Tensor],
    cfg: GenConfig,
    num_return_sequences: int,
    compute_surprisal: bool,
) -> List[Trajectory]:
    batched_inputs = _repeat_model_inputs(model_inputs, num_return_sequences)
    outputs = model.generate(
        **batched_inputs,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        max_new_tokens=cfg.max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )

    prompt_len = batched_inputs["input_ids"].shape[-1]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_ids = set()
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_ids = {int(x) for x in eos_token_id if x is not None}
    else:
        eos_ids = {int(eos_token_id)}
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    pad_token_id = int(pad_token_id) if pad_token_id is not None else None

    trajectories: List[Trajectory] = []
    for batch_idx in range(int(outputs.sequences.shape[0])):
        raw_gen_ids = outputs.sequences[batch_idx, prompt_len:]
        cut = int(raw_gen_ids.numel())
        for token_idx, token in enumerate(raw_gen_ids.tolist()):
            if token in eos_ids or (pad_token_id is not None and token == pad_token_id):
                cut = token_idx + 1
                break
        gen_ids = raw_gen_ids[:cut]

        logp_steps = []
        surprisal_steps = []
        for step_idx in range(int(gen_ids.numel())):
            logits = outputs.scores[step_idx][batch_idx]
            token = int(gen_ids[step_idx].item())
            logp = F.log_softmax(logits, dim=-1)[token]
            logp_steps.append(logp)
            if compute_surprisal:
                surprisal_steps.append(-logp)

        logp_tensor = torch.stack(logp_steps, dim=0).detach().cpu() if logp_steps else torch.empty(0, dtype=torch.float32)
        surprisal_tensor = (
            torch.stack(surprisal_steps, dim=0).detach().cpu()
            if compute_surprisal and surprisal_steps
            else None
        )
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        trajectories.append(
            Trajectory(
                text=text,
                gen_token_ids=gen_ids.detach().cpu(),
                logp=logp_tensor,
                surprisal=surprisal_tensor,
            )
        )

    return trajectories


@torch.no_grad()
def generate_batch_multi_prompt(
    model,
    tokenizer,
    model_inputs_list: List[Dict[str, torch.Tensor]],
    cfg: GenConfig,
    num_return_sequences: int,
    compute_surprisal: bool,
    audio_token_id: int = -1,
) -> List[List[Trajectory]]:
    """Generate completions for multiple prompts in one batched call.

    Returns a list of trajectory lists, one per prompt.
    """
    from openrlhf.speech_leaf.model_utils import sync_audio_features
    from openrlhf.speech_leaf.replay_buffer import zero_pad_last_dim, zero_pad_sequences

    n_prompts = len(model_inputs_list)
    if n_prompts == 0:
        return []
    if n_prompts == 1:
        return [generate_batch(model, tokenizer, model_inputs_list[0], cfg, num_return_sequences, compute_surprisal)]

    device = model_inputs_list[0]["input_ids"].device
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # Left-pad input_ids and attention_mask (so generation starts at the same column)
    input_ids_list = [mi["input_ids"].squeeze(0) for mi in model_inputs_list]
    attn_mask_list = [mi["attention_mask"].squeeze(0) for mi in model_inputs_list]
    batched_ids = zero_pad_sequences(input_ids_list, side="left", value=pad_token_id)
    batched_attn = zero_pad_sequences(attn_mask_list, side="left", value=0)

    batched: Dict[str, torch.Tensor] = {"input_ids": batched_ids, "attention_mask": batched_attn}

    # Right-pad other keys (input_features, input_features_mask, etc.)
    other_keys = [k for k in model_inputs_list[0].keys() if k not in ("input_ids", "attention_mask")]
    for key in other_keys:
        values = [mi[key].squeeze(0) for mi in model_inputs_list]
        batched[key] = zero_pad_last_dim(values)

    # Sync audio features for the padded batch
    if audio_token_id >= 0 and "input_features_mask" in batched:
        batched = sync_audio_features(batched, audio_token_id)

    # Repeat each prompt K times: [p0,p0,...,p1,p1,...] for num_return_sequences
    if num_return_sequences > 1:
        for key, value in batched.items():
            batched[key] = torch.repeat_interleave(value, num_return_sequences, dim=0)

    outputs = model.generate(
        **batched,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        max_new_tokens=cfg.max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )

    prompt_len = batched["input_ids"].shape[-1]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_ids = set()
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_ids = {int(x) for x in eos_token_id if x is not None}
    else:
        eos_ids = {int(eos_token_id)}

    all_trajectories: List[List[Trajectory]] = [[] for _ in range(n_prompts)]
    for seq_idx in range(int(outputs.sequences.shape[0])):
        prompt_idx = seq_idx // num_return_sequences
        raw_gen_ids = outputs.sequences[seq_idx, prompt_len:]
        cut = int(raw_gen_ids.numel())
        for token_idx, token in enumerate(raw_gen_ids.tolist()):
            if token in eos_ids or token == pad_token_id:
                cut = token_idx + 1
                break
        gen_ids = raw_gen_ids[:cut]

        logp_steps = []
        surprisal_steps = []
        for step_idx in range(int(gen_ids.numel())):
            logits = outputs.scores[step_idx][seq_idx]
            token = int(gen_ids[step_idx].item())
            logp = F.log_softmax(logits, dim=-1)[token]
            logp_steps.append(logp)
            if compute_surprisal:
                surprisal_steps.append(-logp)

        logp_tensor = torch.stack(logp_steps, dim=0).detach().cpu() if logp_steps else torch.empty(0, dtype=torch.float32)
        surprisal_tensor = (
            torch.stack(surprisal_steps, dim=0).detach().cpu()
            if compute_surprisal and surprisal_steps
            else None
        )
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        all_trajectories[prompt_idx].append(
            Trajectory(
                text=text,
                gen_token_ids=gen_ids.detach().cpu(),
                logp=logp_tensor,
                surprisal=surprisal_tensor,
            )
        )

    return all_trajectories


def _append_tokens(model_inputs: Dict[str, torch.Tensor], extra_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    updated: Dict[str, torch.Tensor] = {}
    for key, value in model_inputs.items():
        if not torch.is_tensor(value):
            updated[key] = value
            continue
        if value.dim() == 2 and value.shape[0] == 1 and key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            if key == "input_ids":
                updated[key] = torch.cat([value, extra_ids[None, :].to(value.device)], dim=-1)
            elif key == "attention_mask":
                extra = torch.ones((1, extra_ids.numel()), dtype=value.dtype, device=value.device)
                updated[key] = torch.cat([value, extra], dim=-1)
            elif key == "position_ids":
                start = value[0, -1].item() + 1
                extra = torch.arange(start, start + extra_ids.numel(), device=value.device, dtype=value.dtype)[None, :]
                updated[key] = torch.cat([value, extra], dim=-1)
            else:
                extra = value[:, -1:].repeat(1, extra_ids.numel())
                updated[key] = torch.cat([value, extra], dim=-1)
        else:
            updated[key] = value
    return updated


@torch.no_grad()
def _make_full_trajectories(
    model,
    tokenizer,
    model_inputs: Dict[str, torch.Tensor],
    prefix_ids: torch.Tensor,
    cfg: GenConfig,
    num_samples: int,
) -> List[Trajectory]:
    prefix_inputs = _append_tokens(model_inputs, prefix_ids.to(model_inputs["input_ids"].device)) if prefix_ids.numel() > 0 else model_inputs
    suffixes = generate_batch(
        model=model,
        tokenizer=tokenizer,
        model_inputs=prefix_inputs,
        cfg=cfg,
        num_return_sequences=num_samples,
        compute_surprisal=True,
    )

    trajectories: List[Trajectory] = []
    for suffix in suffixes:
        full_ids = torch.cat([prefix_ids.cpu(), suffix.gen_token_ids.cpu()], dim=0) if prefix_ids.numel() > 0 else suffix.gen_token_ids.cpu()
        if suffix.surprisal is not None and prefix_ids.numel() > 0:
            prefix_surp = torch.full((prefix_ids.numel(),), -1e9, dtype=suffix.surprisal.dtype)
            surprisal = torch.cat([prefix_surp, suffix.surprisal.cpu()], dim=0)
        else:
            surprisal = suffix.surprisal.cpu() if suffix.surprisal is not None else None

        if prefix_ids.numel() > 0 and suffix.logp.numel() > 0:
            prefix_logp = torch.zeros(prefix_ids.numel(), dtype=suffix.logp.dtype)
            logp = torch.cat([prefix_logp, suffix.logp.cpu()], dim=0)
        else:
            logp = suffix.logp.cpu()

        text = tokenizer.decode(full_ids, skip_special_tokens=True)
        trajectories.append(Trajectory(text=text, gen_token_ids=full_ids, logp=logp, surprisal=surprisal))
    return trajectories


@torch.no_grad()
def _make_full_trajectories_batched(
    model,
    tokenizer,
    model_inputs: Dict[str, torch.Tensor],
    prefix_ids_list: List[torch.Tensor],
    cfg: GenConfig,
    num_samples_per_prefix: int,
) -> List[List[Trajectory]]:
    """Generate completions from multiple prefixes in one batched generate() call."""
    if not prefix_ids_list:
        return []

    device = model_inputs["input_ids"].device
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
    prompt_ids = model_inputs["input_ids"][0]
    prompt_attn = model_inputs["attention_mask"][0]

    # Build extended input_ids/attention_mask for each (prefix, sample) pair
    extended_ids: List[torch.Tensor] = []
    extended_attn: List[torch.Tensor] = []
    for prefix_ids in prefix_ids_list:
        prefix_ids_dev = prefix_ids.to(device)
        plen = prefix_ids_dev.numel()
        ids = torch.cat([prompt_ids, prefix_ids_dev]) if plen > 0 else prompt_ids.clone()
        attn = torch.cat([prompt_attn, torch.ones(plen, dtype=prompt_attn.dtype, device=device)]) if plen > 0 else prompt_attn.clone()
        for _ in range(num_samples_per_prefix):
            extended_ids.append(ids)
            extended_attn.append(attn)

    # Left-pad to uniform length for batched generation
    max_len = max(ids.numel() for ids in extended_ids)
    padded_ids = []
    padded_attn = []
    for ids, attn in zip(extended_ids, extended_attn):
        pad_len = max_len - ids.numel()
        padded_ids.append(F.pad(ids, (pad_len, 0), value=pad_token_id))
        padded_attn.append(F.pad(attn, (pad_len, 0), value=0))

    batch_size = len(padded_ids)
    batched: Dict[str, torch.Tensor] = {}
    for key, value in model_inputs.items():
        if key == "input_ids":
            batched[key] = torch.stack(padded_ids, dim=0)
        elif key == "attention_mask":
            batched[key] = torch.stack(padded_attn, dim=0)
        elif torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == 1:
            batched[key] = value.expand(batch_size, *value.shape[1:])
        else:
            batched[key] = value

    outputs = model.generate(
        **batched,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k if cfg.top_k > 0 else None,
        max_new_tokens=cfg.max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_ids = set()
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_ids = {int(x) for x in eos_token_id if x is not None}
    else:
        eos_ids = {int(eos_token_id)}

    # Parse outputs back into per-prefix groups of Trajectories
    results: List[List[Trajectory]] = []
    idx = 0
    for prefix_ids in prefix_ids_list:
        prefix_cpu = prefix_ids.cpu()
        prefix_trajs: List[Trajectory] = []
        for _ in range(num_samples_per_prefix):
            raw_gen_ids = outputs.sequences[idx, max_len:]
            cut = int(raw_gen_ids.numel())
            for token_idx, token in enumerate(raw_gen_ids.tolist()):
                if token in eos_ids or token == pad_token_id:
                    cut = token_idx + 1
                    break
            gen_ids = raw_gen_ids[:cut]

            logp_steps = []
            surprisal_steps = []
            for step_idx in range(int(gen_ids.numel())):
                logits = outputs.scores[step_idx][idx]
                token = int(gen_ids[step_idx].item())
                logp = F.log_softmax(logits, dim=-1)[token]
                logp_steps.append(logp)
                surprisal_steps.append(-logp)

            logp_tensor = torch.stack(logp_steps, dim=0).detach().cpu() if logp_steps else torch.empty(0, dtype=torch.float32)
            surprisal_tensor = torch.stack(surprisal_steps, dim=0).detach().cpu() if surprisal_steps else None

            full_ids = torch.cat([prefix_cpu, gen_ids.detach().cpu()]) if prefix_cpu.numel() > 0 else gen_ids.detach().cpu()
            if surprisal_tensor is not None and prefix_cpu.numel() > 0:
                prefix_surp = torch.full((prefix_cpu.numel(),), -1e9, dtype=surprisal_tensor.dtype)
                surprisal_tensor = torch.cat([prefix_surp, surprisal_tensor])
            if prefix_cpu.numel() > 0 and logp_tensor.numel() > 0:
                prefix_logp = torch.zeros(prefix_cpu.numel(), dtype=logp_tensor.dtype)
                logp_tensor = torch.cat([prefix_logp, logp_tensor])

            text = tokenizer.decode(full_ids, skip_special_tokens=True)
            prefix_trajs.append(Trajectory(text=text, gen_token_ids=full_ids, logp=logp_tensor, surprisal=surprisal_tensor))
            idx += 1
        results.append(prefix_trajs)

    return results


def normalize_step_advantages_zscore(
    train_sequences: List[PrefixTreeTrainSequence],
    eps: float = 1e-6,
) -> List[PrefixTreeTrainSequence]:
    all_advs = [step.advantage for seq in train_sequences for step in seq.steps]
    if not all_advs:
        return train_sequences

    adv_mean = sum(all_advs) / len(all_advs)
    adv_var = sum((adv - adv_mean) ** 2 for adv in all_advs) / max(len(all_advs) - 1, 1)
    adv_std = max(adv_var**0.5, eps)

    for seq in train_sequences:
        for step in seq.steps:
            step.advantage = (step.advantage - adv_mean) / adv_std
    return train_sequences


@torch.no_grad()
def sample_prefix_tree_retro_batch(
    model,
    tokenizer,
    model_inputs: Dict[str, torch.Tensor],
    gen_cfg: GenConfig,
    reward_fn: Callable[[str], float],
    prefix_tree_cfg: Optional[PrefixTreeConfig] = None,
) -> List[PrefixTreeTrainSequence]:
    import math

    cfg = prefix_tree_cfg or PrefixTreeConfig()
    total_seqs = cfg.total_seqs

    all_trajs = generate_batch(
        model=model,
        tokenizer=tokenizer,
        model_inputs=model_inputs,
        cfg=gen_cfg,
        num_return_sequences=total_seqs,
        compute_surprisal=True,
    )
    if not all_trajs:
        return []

    rewards = [float(reward_fn(traj.text)) for traj in all_trajs]
    num_forks = cfg.num_forks

    candidates: List[Tuple[float, int, int]] = []
    for seq_idx, traj in enumerate(all_trajs):
        surprisal = traj.surprisal
        if surprisal is None:
            continue
        gen_len = int(traj.gen_token_ids.numel())
        for pos in range(max(cfg.min_prefix_tokens, 1), gen_len):
            candidates.append((float(surprisal[pos]), seq_idx, pos))

    if not candidates:
        root_value = sum(rewards) / max(len(rewards), 1)
        train_sequences: List[PrefixTreeTrainSequence] = []
        for seq_idx, traj in enumerate(all_trajs):
            gen_ids = traj.gen_token_ids
            gen_len = int(gen_ids.numel())
            if gen_len == 0:
                continue
            steps = [
                PrefixTreeTrainStep(
                    step_start=0,
                    step_end=gen_len,
                    advantage=rewards[seq_idx] - root_value,
                )
            ]
            train_sequences.append(
                PrefixTreeTrainSequence(
                    gen_token_ids=gen_ids.cpu(),
                    logp=traj.logp.cpu() if traj.logp is not None else torch.empty(0),
                    text=traj.text,
                    steps=steps,
                    leaf_reward=rewards[seq_idx],
                )
            )
        return train_sequences

    candidates.sort(key=lambda item: item[0], reverse=True)
    gen_lens = [int(traj.gen_token_ids.numel()) for traj in all_trajs if traj.gen_token_ids.numel() > 0]
    min_gen_len = min(gen_lens) if gen_lens else 1
    min_fork_sep = max(2, min_gen_len // (max(num_forks, 1) + 1))

    fork_positions: List[int] = []
    for _, _, pos in candidates:
        if all(abs(pos - existing_pos) >= min_fork_sep for existing_pos in fork_positions):
            fork_positions.append(pos)
            if len(fork_positions) >= num_forks:
                break
    fork_positions.sort()

    forks: List[dict] = []
    leaf_fork_path: List[List[int]] = [[] for _ in range(len(all_trajs))]
    for fork_pos in fork_positions:
        groups: Dict[tuple, List[int]] = {}
        for seq_idx, traj in enumerate(all_trajs):
            gen_ids = traj.gen_token_ids
            if int(gen_ids.numel()) <= fork_pos:
                prefix_key = tuple(gen_ids.tolist())
            else:
                prefix_key = tuple(gen_ids[:fork_pos].tolist())
            groups.setdefault(prefix_key, []).append(seq_idx)

        for prefix_key, members in groups.items():
            if len(members) <= 1:
                continue
            fork_id = len(forks)
            member_rewards = [rewards[idx] for idx in members]
            forks.append(
                {
                    "position": fork_pos,
                    "prefix_key": prefix_key,
                    "leaf_indices": members,
                    "value": sum(member_rewards) / len(member_rewards),
                    "n_descendants": len(member_rewards),
                }
            )
            for seq_idx in members:
                leaf_fork_path[seq_idx].append(fork_id)

    root_value = sum(rewards) / max(len(rewards), 1)
    train_sequences: List[PrefixTreeTrainSequence] = []
    for seq_idx, traj in enumerate(all_trajs):
        gen_ids = traj.gen_token_ids
        gen_len = int(gen_ids.numel())
        if gen_len == 0:
            continue

        path_forks = [(forks[fork_id]["position"], fork_id) for fork_id in leaf_fork_path[seq_idx]]
        path_forks.sort(key=lambda item: item[0])

        steps: List[PrefixTreeTrainStep] = []
        prev_pos = 0
        prev_value = root_value
        for fork_pos, fork_id in path_forks:
            fork = forks[fork_id]
            node_value = fork["value"]
            n_desc = fork["n_descendants"]
            ga = node_value - root_value
            la = node_value - prev_value
            process_reward = _combine_adv(ga, la, cfg.advantage_terms) / math.sqrt(max(n_desc, 1))

            if fork_pos > prev_pos:
                steps.append(
                    PrefixTreeTrainStep(
                        step_start=prev_pos,
                        step_end=fork_pos,
                        advantage=process_reward,
                    )
                )
            prev_pos = fork_pos
            prev_value = node_value

        if prev_pos < gen_len:
            ga = rewards[seq_idx] - root_value
            la = rewards[seq_idx] - prev_value
            steps.append(
                PrefixTreeTrainStep(
                    step_start=prev_pos,
                    step_end=gen_len,
                    advantage=_combine_adv(ga, la, cfg.advantage_terms),
                )
            )

        train_sequences.append(
            PrefixTreeTrainSequence(
                gen_token_ids=gen_ids.cpu(),
                logp=traj.logp.cpu() if traj.logp is not None else torch.empty(0),
                text=traj.text,
                steps=steps,
                leaf_reward=rewards[seq_idx],
            )
        )

    return train_sequences


def build_prefix_tree_from_trajectories(
    all_trajs: List[Trajectory],
    reward_fn: Callable[[str], float],
    prefix_tree_cfg: Optional[PrefixTreeConfig] = None,
) -> List[PrefixTreeTrainSequence]:
    """Build PrefixTree train sequences from pre-generated trajectories (no generation).

    This is the tree-construction part of sample_prefix_tree_retro_batch, separated
    so that generation can be batched across multiple prompts.
    """
    import math

    cfg = prefix_tree_cfg or PrefixTreeConfig()
    if not all_trajs:
        return []

    rewards = [float(reward_fn(traj.text)) for traj in all_trajs]
    num_forks = cfg.num_forks

    candidates: List[Tuple[float, int, int]] = []
    for seq_idx, traj in enumerate(all_trajs):
        surprisal = traj.surprisal
        if surprisal is None:
            continue
        gen_len = int(traj.gen_token_ids.numel())
        for pos in range(max(cfg.min_prefix_tokens, 1), gen_len):
            candidates.append((float(surprisal[pos]), seq_idx, pos))

    root_value = sum(rewards) / max(len(rewards), 1)

    if not candidates:
        train_seqs: List[PrefixTreeTrainSequence] = []
        for seq_idx, traj in enumerate(all_trajs):
            gen_ids = traj.gen_token_ids
            gen_len = int(gen_ids.numel())
            if gen_len == 0:
                continue
            train_seqs.append(
                PrefixTreeTrainSequence(
                    gen_token_ids=gen_ids.cpu(),
                    logp=traj.logp.cpu() if traj.logp is not None else torch.empty(0),
                    text=traj.text,
                    steps=[PrefixTreeTrainStep(step_start=0, step_end=gen_len, advantage=rewards[seq_idx] - root_value)],
                    leaf_reward=rewards[seq_idx],
                )
            )
        return train_seqs

    candidates.sort(key=lambda item: item[0], reverse=True)
    gen_lens = [int(traj.gen_token_ids.numel()) for traj in all_trajs if traj.gen_token_ids.numel() > 0]
    min_gen_len = min(gen_lens) if gen_lens else 1
    min_fork_sep = max(2, min_gen_len // (max(num_forks, 1) + 1))

    fork_positions: List[int] = []
    for _, _, pos in candidates:
        if all(abs(pos - ep) >= min_fork_sep for ep in fork_positions):
            fork_positions.append(pos)
            if len(fork_positions) >= num_forks:
                break
    fork_positions.sort()

    forks: List[dict] = []
    leaf_fork_path: List[List[int]] = [[] for _ in range(len(all_trajs))]
    for fork_pos in fork_positions:
        groups: Dict[tuple, List[int]] = {}
        for seq_idx, traj in enumerate(all_trajs):
            gen_ids = traj.gen_token_ids
            prefix_key = tuple(gen_ids[:fork_pos].tolist()) if int(gen_ids.numel()) > fork_pos else tuple(gen_ids.tolist())
            groups.setdefault(prefix_key, []).append(seq_idx)
        for prefix_key, members in groups.items():
            if len(members) <= 1:
                continue
            fork_id = len(forks)
            member_rewards = [rewards[idx] for idx in members]
            forks.append({
                "position": fork_pos, "leaf_indices": members,
                "value": sum(member_rewards) / len(member_rewards),
                "n_descendants": len(member_rewards),
            })
            for seq_idx in members:
                leaf_fork_path[seq_idx].append(fork_id)

    train_seqs: List[PrefixTreeTrainSequence] = []
    for seq_idx, traj in enumerate(all_trajs):
        gen_ids = traj.gen_token_ids
        gen_len = int(gen_ids.numel())
        if gen_len == 0:
            continue
        path_forks = sorted(
            [(forks[fid]["position"], fid) for fid in leaf_fork_path[seq_idx]],
            key=lambda x: x[0],
        )
        steps: List[PrefixTreeTrainStep] = []
        prev_pos, prev_value = 0, root_value
        for fork_pos, fork_id in path_forks:
            fork = forks[fork_id]
            nv, nd = fork["value"], fork["n_descendants"]
            if cfg.advantage_terms == "both":
                # default: full GA + LA credit, scaled by the node's descendant count
                pr = (nv - root_value + nv - prev_value) / math.sqrt(max(nd, 1))
            else:
                pr = _combine_adv(nv - root_value, nv - prev_value, cfg.advantage_terms) / math.sqrt(max(nd, 1))
            if fork_pos > prev_pos:
                steps.append(PrefixTreeTrainStep(step_start=prev_pos, step_end=fork_pos, advantage=pr))
            prev_pos, prev_value = fork_pos, nv
        if prev_pos < gen_len:
            steps.append(PrefixTreeTrainStep(
                step_start=prev_pos, step_end=gen_len,
                advantage=_combine_adv(rewards[seq_idx] - root_value, rewards[seq_idx] - prev_value, cfg.advantage_terms),
            ))
        train_seqs.append(PrefixTreeTrainSequence(
            gen_token_ids=gen_ids.cpu(),
            logp=traj.logp.cpu() if traj.logp is not None else torch.empty(0),
            text=traj.text, steps=steps, leaf_reward=rewards[seq_idx],
        ))

    return train_seqs

