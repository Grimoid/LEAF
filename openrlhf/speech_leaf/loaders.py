from __future__ import annotations

from pathlib import Path

from peft import PeftConfig, PeftModel
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for chunk in version.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def _require_granite_speech_support(model_or_adapter: str) -> None:
    name_lower = model_or_adapter.lower()
    if not ("granite" in name_lower and "speech" in name_lower):
        return
    min_version = (4, 52, 4)
    cur_version = _version_tuple(transformers.__version__)
    if cur_version >= min_version:
        return
    raise RuntimeError(
        "Granite Speech models require transformers>=4.52.4. "
        f"Current version is {transformers.__version__}. "
        "Upgrade the environment before running the LEAF speech scripts."
    )


def _adapter_base_model(model_or_adapter: str) -> str | None:
    path = Path(model_or_adapter)
    if not path.exists() or not (path / "adapter_config.json").exists():
        return None
    cfg = PeftConfig.from_pretrained(model_or_adapter)
    return cfg.base_model_name_or_path


def load_processor(model_or_adapter: str):
    processor_source = _adapter_base_model(model_or_adapter) or model_or_adapter
    _require_granite_speech_support(processor_source)
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


def _load_base_model(model_name_or_path: str, dtype=None):
    _require_granite_speech_support(model_name_or_path)
    last_err = None
    for cls in (AutoModelForSpeechSeq2Seq, AutoModelForCausalLM):
        try:
            return cls.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            last_err = exc
    raise RuntimeError(f"Failed to load model: {model_name_or_path}") from last_err


def load_model_maybe_peft(model_or_adapter: str, dtype=None):
    base_model = _adapter_base_model(model_or_adapter)
    if base_model is not None:
        cfg = PeftConfig.from_pretrained(model_or_adapter)
        base = _load_base_model(cfg.base_model_name_or_path, dtype=dtype)
        return PeftModel.from_pretrained(base, model_or_adapter)
    return _load_base_model(model_or_adapter, dtype=dtype)
