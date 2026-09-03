from __future__ import annotations

import random
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class SpeechExperience:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    info: Optional[dict] = None
    kl: Optional[torch.Tensor] = None
    model_inputs: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        if self.base_action_log_probs is not None:
            self.base_action_log_probs = self.base_action_log_probs.to(device)
        if self.values is not None:
            self.values = self.values.to(device)
        if self.returns is not None:
            self.returns = self.returns.to(device)
        if self.advantages is not None:
            self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)
        if self.kl is not None:
            self.kl = self.kl.to(device)
        if self.model_inputs is not None:
            self.model_inputs = {k: v.to(device) for k, v in self.model_inputs.items()}

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        if self.base_action_log_probs is not None:
            self.base_action_log_probs = self.base_action_log_probs.pin_memory()
        if self.values is not None:
            self.values = self.values.pin_memory()
        if self.returns is not None:
            self.returns = self.returns.pin_memory()
        if self.advantages is not None:
            self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        if self.kl is not None:
            self.kl = self.kl.pin_memory()
        if self.model_inputs is not None:
            self.model_inputs = {k: v.pin_memory() for k, v in self.model_inputs.items()}
        return self


@dataclass
class SpeechBufferItem:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    info: Optional[dict] = None
    kl: Optional[torch.Tensor] = None
    model_inputs: Optional[Dict[str, torch.Tensor]] = None


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left", value: int = 0) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        orig_dtype = seq.dtype
        seq_for_pad = seq.to(torch.int64) if orig_dtype == torch.bool else seq
        seq_for_pad = F.pad(seq_for_pad, padding, value=value)
        padded.append(seq_for_pad.to(orig_dtype))
    return torch.stack(padded, dim=0)


def zero_pad_last_dim(tensors: List[torch.Tensor]) -> torch.Tensor:
    if tensors[0].dim() == 0:
        return torch.stack(tensors, dim=0)
    # Pad ALL dimensions to the max size (not just the last).
    # This handles e.g. input_features with varying seq_len: [232,160] vs [684,160].
    ndim = tensors[0].dim()
    max_sizes = [max(t.shape[d] for t in tensors) for d in range(ndim)]
    padded = []
    for tensor in tensors:
        orig_dtype = tensor.dtype
        tensor_for_pad = tensor.to(torch.int64) if orig_dtype == torch.bool else tensor
        # F.pad expects pairs in reverse dimension order: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        pad_args = []
        for d in reversed(range(ndim)):
            pad_args.extend([0, max_sizes[d] - tensor.shape[d]])
        if any(p > 0 for p in pad_args):
            tensor_for_pad = F.pad(tensor_for_pad, pad_args)
        padded.append(tensor_for_pad.to(orig_dtype))
    return torch.stack(padded, dim=0)


def split_experience_batch(experience: SpeechExperience) -> List[SpeechBufferItem]:
    batch_size = experience.sequences.size(0)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "kl",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            continue
        vals = torch.unbind(value)
        for idx, val in enumerate(vals):
            batch_kwargs[idx][key] = val

    if experience.model_inputs:
        for idx in range(batch_size):
            batch_kwargs[idx]["model_inputs"] = {}
        for key, value in experience.model_inputs.items():
            vals = torch.unbind(value)
            for idx, val in enumerate(vals):
                batch_kwargs[idx]["model_inputs"][key] = val

    for idx in range(batch_size):
        batch_kwargs[idx]["info"] = {}
    for key, value in experience.info.items():
        vals = torch.unbind(value)
        for idx, val in enumerate(vals):
            batch_kwargs[idx]["info"][key] = val.item() if val.dim() == 0 else val[0].item()

    return [SpeechBufferItem(**kwargs) for kwargs in batch_kwargs]


def make_experience_batch(items: List[SpeechBufferItem]) -> SpeechExperience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "kl",
    )
    keys = [key for key in keys if getattr(items[0], key) is not None]

    for key in keys:
        values = [getattr(item, key) for item in items]
        pad_value = 0
        side = "left" if key in ("sequences", "attention_mask") else "right"
        kwargs[key] = zero_pad_sequences(values, side=side, value=pad_value)

    kwargs["info"] = {}
    for key in items[0].info.keys():
        kwargs["info"][key] = torch.tensor([item.info[key] for item in items], dtype=torch.float32)

    if items[0].model_inputs is not None:
        kwargs["model_inputs"] = {}
        for key in items[0].model_inputs.keys():
            values = [item.model_inputs[key] for item in items]
            kwargs["model_inputs"][key] = zero_pad_last_dim(values)

    return SpeechExperience(**kwargs)


def remove_padding_in_sequences(items: List[SpeechBufferItem]) -> List[SpeechBufferItem]:
    for item in items:
        # Action-level right padding (for action_log_probs, action_mask, etc.)
        action_right_pad = (1 - item.action_mask.long()).sum().item()
        action_right = None if action_right_pad == 0 else -action_right_pad

        # Sequence-level padding: use attention_mask which correctly marks
        # both left padding AND right padding in the full sequence.
        left_pad = item.attention_mask.long().argmax().item()
        actual_content = item.attention_mask.long().sum().item()
        seq_right_pad = item.sequences.size(0) - left_pad - actual_content
        seq_right = None if seq_right_pad == 0 else -seq_right_pad

        item.sequences = item.sequences[left_pad:seq_right]
        item.attention_mask = item.attention_mask[left_pad:seq_right]
        item.action_log_probs = item.action_log_probs[:action_right]
        item.action_mask = item.action_mask[:action_right]
        if item.base_action_log_probs is not None:
            item.base_action_log_probs = item.base_action_log_probs[:action_right]
        if item.values is not None:
            item.values = item.values[:action_right]
        if item.returns is not None:
            item.returns = item.returns[:action_right]
        if item.advantages is not None:
            item.advantages = item.advantages[:action_right]
        if item.kl is not None:
            item.kl = item.kl[:action_right]
    return items


class SpeechReplayBuffer(ABC):
    def __init__(self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[SpeechBufferItem] = []

    @torch.no_grad()
    def append(self, experience: SpeechExperience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = remove_padding_in_sequences(split_experience_batch(experience))
        self.items.extend(items)
        if self.limit > 0:
            extra = len(self.items) - self.limit
            if extra > 0:
                self.items = self.items[extra:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> SpeechExperience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SpeechBufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> SpeechExperience:
        return make_experience_batch(batch)

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        values = [getattr(item, attribute) for item in self.items]
        masks = [item.action_mask for item in self.items]
        value_vec = torch.cat(values).float().flatten()
        mask_vec = torch.cat(masks).flatten()
        sum_and_count = torch.tensor([value_vec.sum(), mask_vec.sum()], device=value_vec.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        std = ((value_vec - mean).pow(2) * mask_vec).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()
        for idx, item in enumerate(self.items):
            setattr(item, attribute, (values[idx] - mean) * rstd)
