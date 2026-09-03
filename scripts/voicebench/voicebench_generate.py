#!/usr/bin/env python
"""Generate responses for the VoiceBench open-ended QA subsets with a Granite-Speech policy.

Self-contained re-implementation of the VoiceBench generation loop
(https://github.com/MatthewCYM/VoiceBench, ``main.py`` + a ``VoiceAssistant`` wrapper)
for IBM Granite Speech + an optional LoRA adapter (the LEAF / GRPO policies).

VoiceBench's audio *is* the spoken instruction, so the model is prompted to respond to it
with the same chat template used for training. Decoding is greedy (``max_new_tokens=256``).

    python scripts/voicebench/voicebench_generate.py --data alpacaeval \
        --adapter <run>/_actor_global_step11200 --output out/leaf__alpacaeval.jsonl

Output: one JSON line per item with all non-audio dataset columns + ``response``
(the format expected by ``voicebench_judge_prometheus.py``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openrlhf.speech_leaf import (  # noqa: E402
    load_audio_array,
    load_model_maybe_peft,
    load_processor,
    sync_audio_features,
)

DEFAULT_BASE = "ibm-granite/granite-speech-3.3-2b"
DEFAULT_PROMPT = "You will hear a request in <|audio|>. Respond to it."
OPEN_ENDED_SUBSETS = ("alpacaeval", "commoneval", "wildvoice")


class GraniteAssistant:
    """Granite Speech + optional LoRA adapter as a VoiceBench-style ``generate_audio`` model."""

    def __init__(self, model_name: str, raw_prompt: str = DEFAULT_PROMPT, device: str = "cuda"):
        self.device = device
        self.processor = load_processor(model_name)
        self.tokenizer = self.processor.tokenizer
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = load_model_maybe_peft(model_name, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        audio_token = getattr(self.processor, "audio_token", "<|audio|>")
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(audio_token)
        self.raw_prompt = raw_prompt
        print(f"[voicebench] loaded {model_name}", flush=True)

    @torch.no_grad()
    def generate_audio(self, audio, max_new_tokens: int = 256) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": self.raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.raw_prompt
        arr = load_audio_array(audio, target_sampling_rate=16000)[0]
        inputs = self.processor(text=[text], audio=[arr], return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        inputs = sync_audio_features(inputs, self.audio_token_id)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
        prompt_len = inputs["input_ids"].shape[-1]
        gen = out.sequences[:, prompt_len:]
        return self.tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="alpacaeval", help=f"VoiceBench subset, e.g. {', '.join(OPEN_ENDED_SUBSETS)}")
    parser.add_argument("--split", default="test")
    parser.add_argument("--adapter", default="", help="LoRA adapter dir (a saved _actor_global_step<N>); empty = base model")
    parser.add_argument("--base_model", default=DEFAULT_BASE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from datasets import Audio, load_dataset
    from tqdm import tqdm

    data = load_dataset("hlt-lab/voicebench", args.data, split=args.split)
    data = data.cast_column("audio", Audio(sampling_rate=16_000))
    if args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))

    model = GraniteAssistant(args.adapter.strip() or args.base_model, raw_prompt=args.prompt)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in tqdm(data, total=len(data), desc=f"voicebench/{args.data}"):
            record = {k: v for k, v in item.items() if k != "audio"}
            record["response"] = model.generate_audio(item["audio"], max_new_tokens=args.max_new_tokens)
            f.write(json.dumps(record) + "\n")
    print(f"[voicebench] wrote {len(data)} responses -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
