"""Qwen2-Audio experience maker for RL training.

Subclasses SpeechRemoteExperienceMaker and overrides the three Granite-specific
points:
  1. Audio token replacement: dataset prompts contain ``<|audio|>`` (Granite);
     Qwen2-Audio expects ``<|audio_bos|><|AUDIO|><|audio_eos|>``.
  2. Processor call: Qwen uses ``audios=[...]`` (plural list) not ``audio=...``.
  3. sync_audio_features is a no-op for Qwen (no ``input_features_mask``);
     we skip it entirely by setting audio_token_id=-1.
"""
from __future__ import annotations

from typing import Dict

import torch

from openrlhf.speech_leaf.audio_utils import load_audio_array
from openrlhf.speech_leaf.experience_maker_openrlhf import SpeechRemoteExperienceMaker

_GRANITE_AUDIO_TOKEN = "<|audio|>"
_QWEN_AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"


class QwenRemoteExperienceMaker(SpeechRemoteExperienceMaker):
    """Experience maker adapted for Qwen2-Audio models."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Disable sync_audio_features: Qwen has no input_features_mask.
        # The guards in the parent already check `audio_token_id >= 0`, so
        # setting -1 is enough, but we also ensure it doesn't accidentally
        # match a real token.
        self.audio_token_id = -1

    def _format_prompt(self, raw_prompt: str) -> str:
        # Replace Granite audio placeholder before applying the chat template.
        prompt = raw_prompt.replace(_GRANITE_AUDIO_TOKEN, _QWEN_AUDIO_PLACEHOLDER)
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        system_prompt = getattr(self.strategy.args, "system_prompt", None)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _prepare_prompt_inputs(self, example: dict, device: torch.device) -> Dict[str, torch.Tensor]:
        audio_array, _ = load_audio_array(example["audio"], target_sampling_rate=16000)
        prompt = self._format_prompt(example["prompt"])
        inputs = self.processor(
            text=prompt,
            audios=[audio_array],  # Qwen: plural list
            return_tensors="pt",
            truncation=True,
            max_length=self.prompt_max_len,
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        # No sync_audio_features for Qwen2-Audio
        return inputs
