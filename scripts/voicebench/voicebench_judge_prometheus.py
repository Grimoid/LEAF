#!/usr/bin/env python
"""Score VoiceBench open-ended QA responses with a local open judge (default M-Prometheus-14B via vLLM).

Uses VoiceBench's *exact* open-QA judge prompt (1-5 scale; ``meta_prompt_open`` in the
VoiceBench repo's ``api_judge.py``), replacing the API judge (gpt-4o-mini) with an
open-source evaluator so the comparison is fully reproducible and API-free.
Because the judge differs from the public leaderboard's, absolute scores are not
leaderboard-comparable; LEAF-vs-GRPO comparisons (same judge, prompt, decoding) are.

    python scripts/voicebench/voicebench_judge_prometheus.py --src out/leaf__alpacaeval.jsonl --tp 1

Input: a responses jsonl with {prompt, response} (from voicebench_generate.py).
Output: <src>.scored.jsonl with added ``judge_raw`` and ``score`` (int 1-5 or null).
"""
import argparse
import json
import re

# Verbatim from VoiceBench api_judge.py (meta_prompt_open), fields {prompt} and {response}.
META_PROMPT_OPEN = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user’s instruction and models’ response:
### [Instruction]: {prompt}
### [Response]: {response}

After evaluating, please output the score only without anything else.
You don’t need to provide any explanations.
"""


def extract_score(txt):
    m = re.search(r"\[RESULT\]\s*([1-5])", txt)   # Prometheus feedback+score format
    if m:
        return int(m.group(1))
    m = re.search(r"\[\[([1-5])\]\]", txt)         # VoiceBench [[n]] form
    if m:
        return int(m.group(1))
    ms = re.findall(r"\b([1-5])\b", txt)           # else the LAST standalone 1-5 (score usually last)
    return int(ms[-1]) if ms else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--judge_model", default="Unbabel/M-Prometheus-14B")
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.8, help="vLLM gpu_memory_utilization (lower if GPUs are shared)")
    ap.add_argument("--max_tokens", type=int, default=512, help="Prometheus emits feedback + score; give it room")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.src) if l.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.judge_model)
    llm = LLM(model=args.judge_model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, dtype="bfloat16")

    prompts = []
    for r in rows:
        content = META_PROMPT_OPEN.replace("{prompt}", str(r.get("prompt", ""))).replace(
            "{response}", str(r.get("response", "")))
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))

    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    scores, fails = [], 0
    out_path = args.out or (args.src[:-6] + ".scored.jsonl" if args.src.endswith(".jsonl") else args.src + ".scored.jsonl")
    with open(out_path, "w") as f:
        for r, o in zip(rows, outs):
            txt = o.outputs[0].text.strip()
            s = extract_score(txt)
            if s is None:
                fails += 1
            else:
                scores.append(s)
            r["judge_raw"], r["score"] = txt, s
            f.write(json.dumps(r) + "\n")
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"mean_score={mean:.3f}  n={len(scores)}  parse_fail={fails}  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
