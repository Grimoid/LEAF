from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from openrlhf.models.actor import Actor
from openrlhf.models.utils import log_probs_from_logits
from openrlhf.speech_leaf.loaders import load_model_maybe_peft


def _maybe_merge_peft(model, name_or_path: str):
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
    elif Path(name_or_path).exists() and (Path(name_or_path) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, name_or_path)
        model = model.merge_and_unload()

    if hasattr(model, "peft_config"):
        del model.peft_config
    if hasattr(model, "_hf_peft_config_loaded"):
        model._hf_peft_config_loaded = False
    return model


def _freeze_model(model) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def build_trainable_speech_model(
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
        model = load_model_maybe_peft(pretrain_or_model, dtype=dtype)
        model_name = pretrain_or_model
    else:
        model = pretrain_or_model
        model_name = ""

    print(
        "build_trainable_speech_model: "
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


def build_reference_speech_model(pretrain_or_model, *, bf16: bool = True):
    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float32
    model = load_model_maybe_peft(pretrain_or_model, dtype=dtype) if isinstance(pretrain_or_model, str) else pretrain_or_model
    model = _maybe_merge_peft(model, pretrain_or_model) if isinstance(pretrain_or_model, str) else model
    _freeze_model(model)
    return model


class SpeechActor(Actor):
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
        self.model = build_trainable_speech_model(
            pretrain_or_model,
            bf16=bf16,
            use_lora=use_lora,
            continue_lora=continue_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        extra_model_inputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", None),
            "num_beams": kwargs.get("num_beams", 1),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "use_cache": True,
        }
        if kwargs.get("max_new_tokens") is not None:
            generate_args["max_new_tokens"] = kwargs["max_new_tokens"]
        if kwargs.get("max_length") is not None:
            generate_args["max_length"] = kwargs["max_length"]
        if extra_model_inputs:
            generate_args.update(extra_model_inputs)

        sequences = self.model.generate(**generate_args)
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        for seq in sequences:
            if seq[-1].item() != pad_token_id:
                seq[-1] = eos_token_id

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        extra_model_inputs: Optional[Dict[str, torch.Tensor]] = None,
        return_output: bool = False,
    ) -> torch.Tensor:
        inputs = {
            "input_ids": sequences,
            "attention_mask": attention_mask,
        }
        if extra_model_inputs:
            inputs.update(extra_model_inputs)
        output = self.model(**inputs)
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])
        if return_output:
            return output if num_actions is None else (log_probs[:, -num_actions:], output)
        return log_probs[:, -num_actions:]


class FrozenSpeechActor(Actor):
    def __init__(self, pretrain_or_model, *, bf16: bool = True) -> None:
        torch.nn.Module.__init__(self)
        self.model = build_reference_speech_model(pretrain_or_model, bf16=bf16)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        extra_model_inputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        generate_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", None),
            "num_beams": kwargs.get("num_beams", 1),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "use_cache": True,
        }
        if kwargs.get("max_new_tokens") is not None:
            generate_args["max_new_tokens"] = kwargs["max_new_tokens"]
        if kwargs.get("max_length") is not None:
            generate_args["max_length"] = kwargs["max_length"]
        if extra_model_inputs:
            generate_args.update(extra_model_inputs)
        sequences = self.model.generate(**generate_args)
        return self.process_sequences(
            sequences,
            input_ids.size(1),
            generate_args["eos_token_id"],
            generate_args["pad_token_id"],
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        extra_model_inputs: Optional[Dict[str, torch.Tensor]] = None,
        return_output: bool = False,
    ):
        inputs = {
            "input_ids": sequences,
            "attention_mask": attention_mask,
        }
        if extra_model_inputs:
            inputs.update(extra_model_inputs)
        output = self.model(**inputs)
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])
        if return_output:
            return output if num_actions is None else (log_probs[:, -num_actions:], output)
        return log_probs[:, -num_actions:]
