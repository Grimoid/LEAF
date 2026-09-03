# Forkability / fork-fire statistics (`scripts/fork_stats/`)

Measures the paper's *forkability* property. A **fork-fire** occurs at a selected boundary when
at least one prefix group has size ≥ 2; it is **usable** when such a group also has non-trivial
reward variation among its descendants. These scripts produce the usable fork-fire rate and its
distribution over token positions.

| Script | Role |
|---|---|
| `collect_fork_fire_stats.py` | **producer (GPU)**: run a checkpoint over a split with `--rollout_budget_K/--fork_budget_B`, record selected / usable fork positions → `<model>_<task>_fork_fire_stats.json` |
| `summarize_fork_group_stats.py` | group-level summary of a training run launched with `speechrl train --fork-stats` (`fork_position_stats_rank*.jsonl`) |
| `summarize_fork_fire_rate_from_training.py` | fork-fire rate vs token position from the same training logs |
| `aggregate_speech_longaudio_fork_stats.py` | merge training-log statistics of several runs into the `*_fork_fire_stats.json` format |
| `plot_fork_fire_rate.py` | per-position fire-rate plot (PNG/SVG) from a stats JSON |
| `plot_fork_fire_clean.py` | multi-task / multi-backbone fire-rate figure grids with Wilson CIs (`FORK_STATS_DIR=<dir with the JSONs>`) |

```bash
python scripts/fork_stats/collect_fork_fire_stats.py --model_name <ckpt> --data_dir $SPEECH_DATA_ROOT/librisqa_part1 \
    --split validation --rollout_budget_K 8 --fork_budget_B 2 --output_json results/fork_stats/granite3_2b_librisqa_fork_fire_stats.json
python scripts/fork_stats/summarize_fork_group_stats.py --run_dir ckpts/leaf_librisqa
```
