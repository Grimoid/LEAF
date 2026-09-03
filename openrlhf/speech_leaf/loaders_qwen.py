"""Qwen2-Audio model and processor loading utilities.

Parallel to loaders.py but uses AutoModelForConditionalGeneration
instead of AutoModelForSpeechSeq2Seq / AutoModelForCausalLM, and
omits the Granite-specific transformers version check.
"""
from __future__ import annotations

from pathlib import Path

from peft import PeftConfig, PeftModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


def _adapter_base_model(model_or_adapter: str) -> str | None:
    path = Path(model_or_adapter)
    if not path.exists() or not (path / "adapter_config.json").exists():
        return None
    cfg = PeftConfig.from_pretrained(model_or_adapter)
    return cfg.base_model_name_or_path


def load_processor_qwen(model_or_adapter: str):
    processor_source = _adapter_base_model(model_or_adapter) or model_or_adapter
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    adapter_path = Path(model_or_adapter)
    chat_template_path = adapter_path / "chat_template.jinja"
    if adapter_path.exists() and chat_template_path.exists() and getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif tokenizer.bos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.bos_token_id
    return processor


def _load_base_model_qwen(model_name_or_path: str, dtype=None):
    return Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )


def load_model_maybe_peft_qwen(model_or_adapter: str, dtype=None):
    base_model = _adapter_base_model(model_or_adapter)
    if base_model is not None:
        cfg = PeftConfig.from_pretrained(model_or_adapter)
        base = _load_base_model_qwen(cfg.base_model_name_or_path, dtype=dtype)
        return PeftModel.from_pretrained(base, model_or_adapter)
    return _load_base_model_qwen(model_or_adapter, dtype=dtype)
