"""Qwen2-Audio REINFORCE trainer.

Subclasses SpeechReinforceTrainer and overrides:
  - experience maker: uses QwenRemoteExperienceMaker (audios=, no sync_audio_features)
  - evaluate_bleu:   uses audios=, skips sync_audio_features, replaces audio token
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from openrlhf.speech_leaf import compute_corpus_metrics, load_audio_array
from openrlhf.speech_leaf.experience_maker_openrlhf_qwen import (
    QwenRemoteExperienceMaker,
    _GRANITE_AUDIO_TOKEN,
    _QWEN_AUDIO_PLACEHOLDER,
)
from openrlhf.speech_leaf.replay_buffer import SpeechReplayBuffer
from openrlhf.speech_leaf.text_utils import normalize_text
from openrlhf.speech_leaf.trainer_openrlhf import SpeechReinforceTrainer


class QwenReinforceTrainer(SpeechReinforceTrainer):
    """SpeechReinforceTrainer adapted for Qwen2-Audio."""

    def __init__(
        self,
        *args,
        processor,
        reward_mode: str = "bleu",
        lowercase_bleu: bool = True,
        eval_dataset=None,
        **kwargs,
    ) -> None:
        # Call parent init (creates SpeechRemoteExperienceMaker internally).
        super().__init__(
            *args,
            processor=processor,
            reward_mode=reward_mode,
            lowercase_bleu=lowercase_bleu,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        # Replace the Granite experience maker with the Qwen one.
        self.experience_maker = QwenRemoteExperienceMaker(
            actor=self.actor,
            initial_model=self.initial_model,
            processor=processor,
            tokenizer=self.tokenizer,
            prompt_max_len=self.prompt_max_len,
            kl_controller=self.kl_ctl,
            strategy=self.strategy,
            reward_mode=reward_mode,
            lowercase_bleu=lowercase_bleu,
        )

    def _format_prompt(self, raw_prompt: str) -> str:
        prompt = raw_prompt.replace(_GRANITE_AUDIO_TOKEN, _QWEN_AUDIO_PLACEHOLDER)
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        system_prompt = getattr(self.strategy.args, "system_prompt", None)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def evaluate_bleu(self, global_step: int) -> Dict[str, float]:
        if not self.strategy.is_rank_0() or self.eval_dataset is None or len(self.eval_dataset) == 0:
            return {}

        args = self.strategy.args
        eval_limit = int(getattr(args, "eval_limit", 0))
        n_samples = min(eval_limit, len(self.eval_dataset)) if eval_limit > 0 else len(self.eval_dataset)
        if n_samples <= 0:
            return {}

        device = torch.device("cuda", torch.cuda.current_device())
        policy_model = self._unwrap_policy_model()
        policy_model.eval()

        predictions = []
        references = []

        for idx in range(n_samples):
            example = self.eval_dataset[idx]
            prompt = self._format_prompt(example["prompt"])
            reference = normalize_text(example["reference"])
            audio_array, _ = load_audio_array(example["audio"], target_sampling_rate=16000)
            model_inputs = self.processor(
                text=prompt,
                audios=[audio_array],  # Qwen: plural list
                return_tensors="pt",
                truncation=True,
                max_length=self.prompt_max_len,
            )
            model_inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in model_inputs.items()
            }
            # No sync_audio_features for Qwen2-Audio

            outputs = policy_model.generate(
                **model_inputs,
                max_new_tokens=int(getattr(args, "eval_max_new_tokens", self.generate_kwargs.get("max_new_tokens", 200))),
                do_sample=bool(getattr(args, "eval_do_sample", 1)),
                temperature=float(getattr(args, "eval_temperature", 0.9)),
                top_p=float(getattr(args, "eval_top_p", 0.9)),
                return_dict_in_generate=True,
            )
            seq = outputs.sequences[0]
            prompt_len = model_inputs["input_ids"].shape[-1]
            hypothesis = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            predictions.append(normalize_text(hypothesis))
            references.append(reference)

        metrics = compute_corpus_metrics(
            predictions=predictions,
            references=references,
            lowercase=bool(getattr(args, "eval_lowercase", 0)),
        )
        record = {"global_step": global_step, **metrics}
        self.strategy.print(
            "[eval] step="
            f"{global_step} " + " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        )
        if self.strategy.is_rank_0():
            self.eval_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.eval_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if self._wandb is not None:
                self._wandb.log(
                    {
                        "eval/epoch": global_step,
                        **{f"eval/{key}": value for key, value in metrics.items()},
                    }
                )
        self.actor.train()
        return metrics
