from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

from openrlhf.speech_leaf.sampler import PrefixTreeConfig, Trajectory


def analyze_trajectories(
    all_trajs: List[Trajectory],
    reward_fn: Callable[[str], float],
    prefix_tree_cfg: Optional[PrefixTreeConfig] = None,
) -> Dict[str, object]:
    cfg = prefix_tree_cfg or PrefixTreeConfig()
    rewards = [float(reward_fn(traj.text)) for traj in all_trajs]
    num_forks = int(cfg.num_forks)
    root_value = sum(rewards) / max(len(rewards), 1)

    candidates: List[Tuple[float, int, int]] = []
    for seq_idx, traj in enumerate(all_trajs):
        surprisal = traj.surprisal
        if surprisal is None:
            continue
        gen_len = int(traj.gen_token_ids.numel())
        for pos in range(max(int(cfg.min_prefix_tokens), 1), gen_len):
            candidates.append((float(surprisal[pos]), seq_idx, pos))

    selected_positions: List[int] = []
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        gen_lens = [int(traj.gen_token_ids.numel()) for traj in all_trajs if traj.gen_token_ids.numel() > 0]
        min_gen_len = min(gen_lens) if gen_lens else 1
        min_fork_sep = max(2, min_gen_len // (max(num_forks, 1) + 1))
        for _, _, pos in candidates:
            if all(abs(pos - existing_pos) >= min_fork_sep for existing_pos in selected_positions):
                selected_positions.append(pos)
                if len(selected_positions) >= num_forks:
                    break
        selected_positions.sort()

    positions: List[Dict[str, object]] = []
    forks: List[Dict[str, object]] = []
    leaf_fork_path: List[List[int]] = [[] for _ in range(len(all_trajs))]

    for fork_pos in selected_positions:
        groups: Dict[tuple, List[int]] = {}
        for seq_idx, traj in enumerate(all_trajs):
            gen_ids = traj.gen_token_ids
            if int(gen_ids.numel()) <= fork_pos:
                prefix_key = tuple(gen_ids.tolist())
            else:
                prefix_key = tuple(gen_ids[:fork_pos].tolist())
            groups.setdefault(prefix_key, []).append(seq_idx)

        position_entry = {
            "position": fork_pos,
            "usable": False,
            "groups": [],
        }

        for prefix_key, members in groups.items():
            if len(members) <= 1:
                continue
            member_rewards = [rewards[idx] for idx in members]
            fork_id = len(forks)
            value = sum(member_rewards) / len(member_rewards)
            spread = max(member_rewards) - min(member_rewards)
            group_entry = {
                "fork_id": fork_id,
                "position": fork_pos,
                "prefix_token_ids": list(prefix_key),
                "member_indices": members,
                "member_rewards": member_rewards,
                "group_size": len(members),
                "value": value,
                "reward_spread": spread,
                "n_descendants": len(members),
            }
            forks.append(group_entry)
            position_entry["usable"] = True
            position_entry["groups"].append(group_entry)
            for seq_idx in members:
                leaf_fork_path[seq_idx].append(fork_id)

        positions.append(position_entry)

    trajectory_entries: List[Dict[str, object]] = []
    for seq_idx, traj in enumerate(all_trajs):
        gen_ids = traj.gen_token_ids
        gen_len = int(gen_ids.numel())
        path_forks = sorted(
            [(forks[fid]["position"], fid) for fid in leaf_fork_path[seq_idx]],
            key=lambda item: item[0],
        )

        steps = []
        prev_pos = 0
        prev_value = root_value
        for fork_pos, fork_id in path_forks:
            fork = forks[fork_id]
            node_value = float(fork["value"])
            n_desc = int(fork["n_descendants"])
            ga = node_value - root_value
            la = node_value - prev_value
            advantage = (ga + la) / math.sqrt(max(n_desc, 1))
            if fork_pos > prev_pos:
                steps.append(
                    {
                        "kind": "fork_span",
                        "fork_id": fork_id,
                        "position": fork_pos,
                        "step_start": prev_pos,
                        "step_end": fork_pos,
                        "advantage": advantage,
                        "token_ids": gen_ids[prev_pos:fork_pos].tolist(),
                    }
                )
            prev_pos = fork_pos
            prev_value = node_value

        if prev_pos < gen_len:
            ga = rewards[seq_idx] - root_value
            la = rewards[seq_idx] - prev_value
            steps.append(
                {
                    "kind": "leaf_span",
                    "fork_id": None,
                    "position": gen_len,
                    "step_start": prev_pos,
                    "step_end": gen_len,
                    "advantage": ga + la,
                    "token_ids": gen_ids[prev_pos:gen_len].tolist(),
                }
            )

        trajectory_entries.append(
            {
                "trajectory_index": seq_idx,
                "text": traj.text,
                "gen_token_ids": gen_ids.tolist(),
                "reward": rewards[seq_idx],
                "steps": steps,
                "path_fork_ids": [fork_id for _, fork_id in path_forks],
            }
        )

    usable_positions = [entry["position"] for entry in positions if entry["usable"]]
    group_sizes = [int(fork["group_size"]) for fork in forks]
    reward_spreads = [float(fork["reward_spread"]) for fork in forks]

    return {
        "attempted_positions": num_forks,
        "selected_positions": selected_positions,
        "usable_positions": usable_positions,
        "usable_position_count": len(usable_positions),
        "usable_group_count": len(forks),
        "group_sizes": group_sizes,
        "reward_spreads": reward_spreads,
        "root_value": root_value,
        "rewards": rewards,
        "positions": positions,
        "forks": forks,
        "trajectories": trajectory_entries,
    }
