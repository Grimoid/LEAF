#!/usr/bin/env python3
"""
Config-driven launcher for the LEAF / GRPO speech experiments (no SFT stage: policies are trained directly from the base model).

Pick a method + model + dataset; the launcher resolves the configuration from
scripts/configs/{defaults,methods,models,datasets}.yaml and builds the training /
evaluation command.

    speechrl.py train --method leaf --model granite3_2b --dataset librisqa
    speechrl.py train --method grpo --model granite3_2b --dataset covost2 --set generate_max_len=100
    speechrl.py train --method leaf ... --resume <run_dir>          # resume into an existing run dir
    speechrl.py train --method leaf ... --warm-start <adapter_dir>  # next epoch: continue the LoRA, fresh optimizer
    speechrl.py eval  --model granite3_2b --dataset librisqa --run <run_dir> --mode all
    speechrl.py eval  --model granite3_2b --dataset librisqa --checkpoint <ckpt> --out r.json
    <any command> --dry-run                                         # print the command without running

Environment setup (conda env, CUDA, NCCL, PYTHONPATH) is done by speechrl.sh.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent          # scripts/
REPO_ROOT = HERE.parent                          # repo root
CONFIG_DIR = HERE / "configs"

# Value-args emitted for every RL method (flag name == config key).
REINFORCE_BASE_KEYS = [
    "train_split", "max_samples",
    "actor_num_gpus_per_node", "ref_num_gpus_per_node",
    "actor_num_gpus_per_actor", "ref_num_gpus_per_actor",
    "micro_rollout_batch_size", "micro_train_batch_size",
    "rollout_batch_size", "train_batch_size",
    "num_episodes", "max_epochs",
    "eval_steps", "eval_limit", "eval_split", "eval_do_sample",
    "eval_temperature", "eval_top_p", "eval_max_new_tokens",
    "save_steps", "logging_steps",
    "prompt_max_len", "generate_max_len", "temperature", "top_p",
    "actor_learning_rate", "init_kl_coef", "reward_mode",
    "rollout_budget_K",
    "lowercase_bleu", "eval_lowercase",
    "use_lora", "continue_lora", "lora_rank", "lora_alpha", "lora_dropout",
    "load_ckpt", "save_ckpt", "gen_prompt_batch_size",
]
EVAL_KEYS = [
    "split", "limit", "random_subset_seed", "num_runs", "base_seed",
    "temperature", "top_p", "do_sample", "max_new_tokens", "eval_batch_size",
    "with_bertscore", "bertscore_lang", "lowercase_bleu", "eval_lowercase",
]


# ---------------------------------------------------------------- config ----
def load_yaml(name):
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def fmt(v):
    """Render a config value as a CLI token."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float) and v.is_integer():
        return repr(v)          # keep e.g. 1.0 for gpu fractions
    return str(v)


def subst_env(s):
    """Expand ${SPEECH_DATA_ROOT} (and any env var) in a config string."""
    if not isinstance(s, str):
        return s
    os.environ.setdefault("SPEECH_DATA_ROOT", str(REPO_ROOT / "data"))
    return os.path.expandvars(s)


def parse_set(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            sys.exit(f"--set expects key=value, got: {p}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_train_cfg(method_name, model_name, dataset_name, overrides):
    defaults = load_yaml("defaults.yaml")
    methods = load_yaml("methods.yaml")
    datasets = load_yaml("datasets.yaml")
    models = load_yaml("models.yaml")
    if method_name not in methods:
        sys.exit(f"unknown method '{method_name}'. known: {', '.join(methods)}")
    if dataset_name not in datasets:
        sys.exit(f"unknown dataset '{dataset_name}'. known: {', '.join(datasets)}")
    if model_name not in models:
        sys.exit(f"unknown model '{model_name}'. known: {', '.join(models)}")
    mspec = methods[method_name]
    dspec = datasets[dataset_name]
    modelspec = models[model_name]
    section = "train"
    cfg = dict(defaults[section])
    cfg.update(dspec.get("defaults", {}) or {})           # dataset...
    cfg.update(mspec.get("defaults", {}) or {})           # ...then method...
    cfg.update(modelspec.get("defaults", {}) or {})       # ...then model...
    cfg.update(overrides)                                  # ...then CLI --set
    return cfg, mspec, dspec, modelspec, section


# --------------------------------------------------------------- builders ----
def build_train_command(args):
    overrides = parse_set(args.set)
    cfg, mspec, dspec, modelspec, section = build_train_cfg(args.method, args.model, args.dataset, overrides)
    hf_id = modelspec["hf_id"]
    family = modelspec["family"]
    data_dir = subst_env(dspec["data_dir"])
    out_dir = resolve_out_dir(args, default_tag=f"{args.method}_{args.model}_{args.dataset}")

    # RL family
    pretrain = hf_id
    if args.warm_start:
        # next-epoch warm restart: continue the saved LoRA adapter with a fresh optimizer/schedule;
        # the KL reference stays the base model.
        pretrain = str(args.warm_start)
        cfg["continue_lora"] = 1
        cfg["load_ckpt"] = 0
        if not cfg.get("ref_pretrain"):
            cfg["ref_pretrain"] = hf_id
        if re.search(r"_actor_global_step\d+$", pretrain.rstrip("/")):
            print("[speechrl] WARNING: --warm-start points at a _actor_global_step<N> dir; the trainer would "
                  "treat <N> as already-consumed prompt batches. Copy the adapter to a dir without that suffix.",
                  file=sys.stderr)
    if args.resume or _has_checkpoint(out_dir):
        cfg["load_ckpt"] = 1
    cmd = ["$PYTHON_BIN", "train_reinforce_ray_speech.py",
           "--method", args.method, "--model_family", family,
           "--pretrain", pretrain, "--speech_data_dir", data_dir, "--save_path", out_dir]
    for k in REINFORCE_BASE_KEYS:
        cmd += [f"--{k}", fmt(cfg[k])]
    for k in mspec.get("args", []) or []:
        cmd += [f"--{k}", fmt(cfg[k])]
    cmd.append("--bf16")
    if str(cfg.get("adam_offload", 0)) == "1":
        cmd.append("--adam_offload")
    if cfg.get("ref_pretrain"):
        cmd += ["--ref_pretrain", str(cfg["ref_pretrain"])]
    if str(cfg.get("print_rollout_samples", 0)) == "1":
        cmd.append("--print_rollout_samples")
    return cmd, out_dir, section


def detect_num_gpus():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return max(1, len([x for x in cvd.split(",") if x.strip()]))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        n = len([ln for ln in out.stdout.splitlines() if ln.strip()])
        if n > 0:
            return n
    except Exception:
        pass
    return 1


def resolve_num_gpus(args, cfg):
    if getattr(args, "num_gpus", None):
        return max(1, int(args.num_gpus))
    v = str(cfg.get("num_gpus", "auto")).strip().lower()
    if v in ("auto", "max", "all", "", "none", "0"):
        return detect_num_gpus()
    return max(1, int(float(v)))


def eval_entrypoint(family, task):
    base = "eval_covost2" if task == "translation" else "eval_librisqa"
    suffix = "_qwen" if family == "qwen" else ""
    return f"scripts/eval/{base}{suffix}.py"


def build_eval_command(args, ckpt, output_json):
    datasets = load_yaml("datasets.yaml")
    models = load_yaml("models.yaml")
    defaults = load_yaml("defaults.yaml")
    dspec = datasets[args.dataset]
    family = models[args.model]["family"]
    cfg = dict(defaults["eval"])
    if "bertscore_lang" in dspec:
        cfg["bertscore_lang"] = dspec["bertscore_lang"]
    cfg.update(parse_set(args.set))
    data_dir = subst_env(dspec.get("eval_data_dir", dspec["data_dir"]))
    ep = eval_entrypoint(family, dspec["task"])
    ng = resolve_num_gpus(args, cfg)
    if int(ng) > 1:
        port = 29500 + (os.getpid() % 10000)
        launcher = ["$PYTHON_BIN", "-m", "torch.distributed.run",
                    f"--nproc_per_node={ng}", f"--master_port={port}"]
    else:
        launcher = ["$PYTHON_BIN"]
    cmd = launcher + [ep, "--model_name", str(ckpt), "--data_dir", data_dir]
    for k in EVAL_KEYS:
        cmd += [f"--{k}", fmt(cfg[k])]
    cmd += ["--output_json", str(output_json)]
    return cmd


# --------------------------------------------------------------- out / resume ----
def resolve_out_dir(args, default_tag):
    if args.resume:
        return str(args.resume)
    if args.out:
        return str(args.out)
    root = args.out_root or str(REPO_ROOT / "ckpts" / default_tag)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(root) / f"run_{tag}")


def _has_checkpoint(d):
    p = Path(d)
    return p.is_dir() and any(p.glob("_actor_ckpt_global_step*"))


# --------------------------------------------------------------- run helpers ----
def show(cmd):
    print(" \\\n  ".join(cmd))


def run(cmd, cwd=REPO_ROOT, log_path=None):
    cmd = [os.environ.get("PYTHON_BIN", sys.executable) if c == "$PYTHON_BIN" else c for c in cmd]
    print("[speechrl] running:", " ".join(cmd), flush=True)
    if not log_path:
        return subprocess.call(cmd, cwd=str(cwd))
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[speechrl] logging to {log_path}", flush=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n# speechrl {datetime.now():%Y-%m-%d %H:%M:%S}: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        return proc.wait()


# --------------------------------------------------------------- eval sweep ----
def list_checkpoints(run_dir):
    return sorted(Path(run_dir).glob("_actor_global_step*"),
                  key=lambda p: int(re.sub(r"\D", "", p.name) or 0))


def metric_of(rec):
    for k in ("bleu", "BLEU", "sacrebleu", "bleu_mean"):
        if k in rec:
            return rec[k]
    return rec.get("metrics", {}).get("bleu", -1)


def cmd_eval(args):
    out_root = Path(args.run) if args.run else None
    if args.checkpoint:
        out_json = args.out or "eval_result.json"
        cmd = build_eval_command(args, args.checkpoint, out_json)
        if args.dry_run:
            show(cmd)
            return 0
        return run(cmd)
    if not out_root:
        sys.exit("eval needs either --checkpoint <path> or --run <run_dir>")
    results = Path(args.results or (out_root / ("eval_all_ckpts.jsonl" if args.mode == "all"
                                                else f"eval_top{args.top_k}_ckpts.jsonl")))
    ckpts = list_checkpoints(out_root)
    if not ckpts:
        sys.exit(f"no _actor_global_step* checkpoints in {out_root}")
    if args.mode == "topk":
        src = Path(args.source_jsonl or (out_root / "eval_all_ckpts.jsonl"))
        if not src.exists():
            sys.exit(f"--mode topk needs a ranked source jsonl (missing {src})")
        ranked = sorted((json.loads(l) for l in src.read_text().splitlines() if l.strip()),
                        key=metric_of, reverse=True)
        keep = {str(r.get("checkpoint")) for r in ranked[: args.top_k]}
        ckpts = [c for c in ckpts if str(c) in keep]

    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                done.add(json.loads(line).get("global_step"))
            except Exception:
                pass
    for i, ck in enumerate(ckpts, 1):
        step = int(re.sub(r"\D", "", ck.name) or 0)
        if args.mode == "all" and step in done:
            print(f"[{i}/{len(ckpts)}] step={step} already evaluated, skipping")
            continue
        tmp = out_root / f".speechrl_eval_{os.getpid()}_{step}.json"
        cmd = build_eval_command(args, ck, tmp)
        if args.dry_run:
            print(f"# step {step}")
            show(cmd)
            print()
            continue
        if run(cmd) == 0 and tmp.exists():
            rec = json.loads(tmp.read_text())
            rec["global_step"] = step
            rec["checkpoint"] = str(ck)
            with open(results, "a") as f:
                f.write(json.dumps(rec) + "\n")
            tmp.unlink(missing_ok=True)
            print(f"[{i}/{len(ckpts)}] step={step} -> appended to {results}")
        else:
            print(f"[{i}/{len(ckpts)}] step={step} FAILED")
    print(f"done. results in {results}")
    return 0


def cmd_train(args):
    cmd, out_dir, section = build_train_command(args)
    if getattr(args, "fork_stats", False):
        os.environ["SPEECH_FORK_STATS"] = "1"     # propagated to the Ray actors (global_envs.py)
    if args.dry_run:
        show(cmd)
        if getattr(args, "fork_stats", False):
            print("\n# fork-stats logging: on (SPEECH_FORK_STATS=1)", file=sys.stderr)
        print(f"# save_path: {out_dir}", file=sys.stderr)
        return 0
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if getattr(args, "fork_stats", False):
        print("[speechrl] fork-stats logging enabled -> fork_position_stats_rank*.jsonl")
    _print_step_banner(args)
    log_path = Path(out_dir) / "train.log" if args.log else None
    return run(cmd, log_path=log_path)


def _print_step_banner(args):
    try:
        cfg = build_train_cfg(args.method, args.model, args.dataset, parse_set(args.set))[0]
        g, mrb, rb = int(cfg["actor_num_gpus_per_node"]), int(cfg["micro_rollout_batch_size"]), int(cfg["rollout_batch_size"])
        ut = max(1, rb // (g * mrb))
        es, ss = int(cfg["eval_steps"]), int(cfg["save_steps"])
        print(f"[speechrl] eval_steps/save_steps count TRAINING STEPS; 1 step = {ut} prompts "
              f"(= rollout_batch_size {rb} / (actor_gpus {g} * micro_rollout {mrb})).")
        print(f"[speechrl]   eval_steps={es}: validation + checkpoint every {es} steps (~{es * ut} prompts).")
        if ss > 0:
            print(f"[speechrl]   save_steps={ss}: checkpoint every {ss} steps (~{ss * ut} prompts).")
        print("[speechrl]   checkpoints are named _actor_global_step<STEP>.")
    except Exception:
        pass


# --------------------------------------------------------------- cli ----
def main():
    p = argparse.ArgumentParser(prog="speechrl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train or resume a LEAF / GRPO run")
    pt.add_argument("--method", required=True, help="leaf | grpo (scripts/configs/methods.yaml)")
    pt.add_argument("--model", required=True, help="scripts/configs/models.yaml key")
    pt.add_argument("--dataset", required=True, help="scripts/configs/datasets.yaml key")
    pt.add_argument("--out", help="exact save_path dir")
    pt.add_argument("--out-root", dest="out_root", help="root dir; a run_<timestamp> subdir is appended")
    pt.add_argument("--resume", help="resume into this existing run dir (forces load_ckpt=1)")
    pt.add_argument("--warm-start", dest="warm_start",
                    help="LoRA adapter dir to continue from with a fresh optimizer (next epoch)")
    pt.add_argument("--fork-stats", dest="fork_stats", action="store_true",
                    help="log per-rollout fork statistics to <run_dir>/fork_position_stats_rank*.jsonl")
    pt.add_argument("--set", action="append", help="override any config key, e.g. --set fork_budget_B=4")
    pt.add_argument("--log", action="store_true", help="tee output to <save_path>/train.log")
    pt.add_argument("--dry-run", dest="dry_run", action="store_true")
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="evaluate a checkpoint or sweep a run dir (text metrics)")
    pe.add_argument("--model", required=True)
    pe.add_argument("--dataset", required=True)
    pe.add_argument("--run", help="run dir containing _actor_global_step* checkpoints")
    pe.add_argument("--checkpoint", help="single checkpoint/model to evaluate")
    pe.add_argument("--mode", choices=["all", "topk"], default="all")
    pe.add_argument("--top-k", dest="top_k", type=int, default=5)
    pe.add_argument("--results", help="output jsonl (sweep modes)")
    pe.add_argument("--source-jsonl", dest="source_jsonl", help="ranking source for --mode topk")
    pe.add_argument("--out", help="output json (single checkpoint)")
    pe.add_argument("--num-gpus", dest="num_gpus", type=int, help="GPUs for eval (default: all visible)")
    pe.add_argument("--set", action="append")
    pe.add_argument("--dry-run", dest="dry_run", action="store_true")
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
