"""Qwen2-Audio actor classes for RL training.

Mirrors actor.py but uses load_model_maybe_peft_qwen (AutoModelForConditionalGeneration)
instead of the Granite-specific loader.  The generate() and forward() methods are
inherited unchanged from SpeechActor / FrozenSpeechActor.
"""
from __future__ import annotations

from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from openrlhf.speech_leaf.actor import (
    FrozenSpeechActor,
    SpeechActor,
    _freeze_model,
    _maybe_merge_peft,
)
from openrlhf.speech_leaf.loaders_qwen import load_model_maybe_peft_qwen


def build_trainable_qwen_model(
    pretrain_or_model,
    *,
    bf16: bool = True,
    use_lora: bool = True,
    continue_lora: bool = True,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
):
    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float32
    if isinstance(pretrain_or_model, str):
        model = load_model_maybe_peft_qwen(pretrain_or_model, dtype=dtype)
        model_name = pretrain_or_model
    else:
        model = pretrain_or_model
        model_name = ""

    print(
        "build_trainable_qwen_model: "
        f"use_lora={use_lora} continue_lora={continue_lora} "
        f"model_type={type(model).__name__} model_name={model_name}",
        flush=True,
    )

    if use_lora:
        if continue_lora and isinstance(model, PeftModel):
            print("Continuing existing LoRA adapter without reinjection.", flush=True)
            for name, param in model.named_parameters():
                param.requires_grad_("lora_" in name)
            return model

        print(
            "Injecting a new LoRA adapter. "
            f"continue_lora={continue_lora} is_peft_model={isinstance(model, PeftModel)}",
            flush=True,
        )
        model = _maybe_merge_peft(model, model_name) if model_name else model
        task_type = "SEQ_2_SEQ_LM" if getattr(model.config, "is_encoder_decoder", False) else "CAUSAL_LM"
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules=target_modules if target_modules else None,
        )
        model = get_peft_model(model, config)
        return model

    model = _maybe_merge_peft(model, model_name) if model_name else model
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def build_reference_qwen_model(pretrain_or_model, *, bf16: bool = True):
    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float32
    model = load_model_maybe_peft_qwen(pretrain_or_model, dtype=dtype) if isinstance(pretrain_or_model, str) else pretrain_or_model
    model = _maybe_merge_peft(model, pretrain_or_model) if isinstance(pretrain_or_model, str) else model
    _freeze_model(model)
    return model


class QwenActor(SpeechActor):
    """SpeechActor variant that loads Qwen2-Audio via AutoModelForConditionalGeneration."""

    def __init__(
        self,
        pretrain_or_model,
        *,
        bf16: bool = True,
        use_lora: bool = True,
        continue_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.model = build_trainable_qwen_model(
            pretrain_or_model,
            bf16=bf16,
            use_lora=use_lora,
            continue_lora=continue_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )


class FrozenQwenActor(FrozenSpeechActor):
    """FrozenSpeechActor variant that loads Qwen2-Audio via AutoModelForConditionalGeneration."""

    def __init__(self, pretrain_or_model, *, bf16: bool = True) -> None:
        torch.nn.Module.__init__(self)
        self.model = build_reference_qwen_model(pretrain_or_model, bf16=bf16)
