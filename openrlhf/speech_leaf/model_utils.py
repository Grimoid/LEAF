from __future__ import annotations

import warnings
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def sync_audio_features(model_inputs: Dict[str, torch.Tensor], audio_token_id: int) -> Dict[str, torch.Tensor]:
    if "input_features_mask" not in model_inputs or "input_ids" not in model_inputs:
        return model_inputs

    input_ids = model_inputs["input_ids"]
    mask = model_inputs["input_features_mask"]

    n_audio = (input_ids == audio_token_id).sum(dim=-1)
    n_feat = mask.int().sum(dim=-1)

    if torch.all(n_audio == n_feat):
        return model_inputs

    new_mask = mask.clone()
    for batch_idx in range(mask.shape[0]):
        need = int(n_audio[batch_idx].item())
        have = int(n_feat[batch_idx].item())
        if need < have:
            true_pos = mask[batch_idx].nonzero(as_tuple=True)[0]
            new_mask[batch_idx, true_pos[need:]] = False
        elif need > have:
            # Pad input_features_mask with True values so the model sees
            # enough audio features.  The underlying input_features tensor
            # is zero-padded, so the extra slots are silent but keep shapes
            # consistent.
            false_pos = (~mask[batch_idx]).nonzero(as_tuple=True)[0]
            extra_needed = need - have
            if len(false_pos) >= extra_needed:
                new_mask[batch_idx, false_pos[:extra_needed]] = True
            else:
                warnings.warn(
                    f"Batch element {batch_idx}: {need} audio tokens but only {have} audio features "
                    f"and not enough mask slots to pad ({len(false_pos)} available).",
                    stacklevel=2,
                )

    updated = dict(model_inputs)
    updated["input_features_mask"] = new_mask
    return updated


def build_full_inputs(model_inputs: Dict[str, torch.Tensor], gen_ids: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], int]:
    prompt_len = model_inputs["input_ids"].shape[-1]
    device = model_inputs["input_ids"].device
    gen_ids = gen_ids.to(device)
    input_ids = torch.cat([model_inputs["input_ids"], gen_ids[None, :]], dim=-1)
    attention_mask = torch.cat(
        [
            model_inputs["attention_mask"],
            torch.ones((1, gen_ids.numel()), dtype=model_inputs["attention_mask"].dtype, device=device),
        ],
        dim=-1,
    )

    full_inputs: Dict[str, torch.Tensor] = {}
    for key, value in model_inputs.items():
        if key == "input_ids":
            full_inputs[key] = input_ids
        elif key == "attention_mask":
            full_inputs[key] = attention_mask
        else:
            full_inputs[key] = value
    return full_inputs, prompt_len


def score_tokens_logp(
    model,
    full_inputs: Dict[str, torch.Tensor],
    prompt_len: int,
    gen_len: int,
    detach: bool = True,
) -> torch.Tensor:
    outputs = model(**full_inputs)
    logits = outputs.logits
    start = prompt_len - 1
    end = prompt_len + gen_len - 1
    logits_slice = logits[0, start:end, :]
    target = full_inputs["input_ids"][0, prompt_len : prompt_len + gen_len]
    logp = F.log_softmax(logits_slice, dim=-1).gather(1, target[:, None]).squeeze(1)
    return logp.detach().cpu() if detach else logp


def build_batched_full_inputs(
    model_inputs: Dict[str, torch.Tensor],
    gen_ids_list: list,
    pad_token_id: int = 0,
) -> Tuple[Dict[str, torch.Tensor], int, list]:
    device = model_inputs["input_ids"].device
    prompt_len = model_inputs["input_ids"].shape[-1]
    gen_lens = [int(g.numel()) for g in gen_ids_list]
    max_gen = max(gen_lens) if gen_lens else 0

    input_id_rows = []
    attn_rows = []
    for gen_ids, gen_len in zip(gen_ids_list, gen_lens):
        gen_ids = gen_ids.to(device)
        pad_len = max_gen - gen_len
        ids = torch.cat(
            [
                model_inputs["input_ids"][0],
                gen_ids,
                torch.full((pad_len,), pad_token_id, dtype=gen_ids.dtype, device=device),
            ]
        )
        mask = torch.cat(
            [
                model_inputs["attention_mask"][0],
                torch.ones(gen_len, dtype=model_inputs["attention_mask"].dtype, device=device),
                torch.zeros(pad_len, dtype=model_inputs["attention_mask"].dtype, device=device),
            ]
        )
        input_id_rows.append(ids)
        attn_rows.append(mask)

    batched: Dict[str, torch.Tensor] = {}
    batch_size = len(gen_ids_list)
    for key, value in model_inputs.items():
        if key == "input_ids":
            batched[key] = torch.stack(input_id_rows, dim=0)
        elif key == "attention_mask":
            batched[key] = torch.stack(attn_rows, dim=0)
        else:
            batched[key] = value.expand(batch_size, *value.shape[1:])
    return batched, prompt_len, gen_lens


def score_tokens_logp_batched(
    model,
    batched_inputs: Dict[str, torch.Tensor],
    prompt_len: int,
    gen_lens: list,
    detach: bool = True,
) -> list:
    outputs = model(**batched_inputs)
    logits = outputs.logits

    results = []
    for batch_idx, gen_len in enumerate(gen_lens):
        if gen_len == 0:
            results.append(None)
            continue
        start = prompt_len - 1
        end = prompt_len + gen_len - 1
        logits_slice = logits[batch_idx, start:end, :]
        target = batched_inputs["input_ids"][batch_idx, prompt_len : prompt_len + gen_len]
        logp = F.log_softmax(logits_slice, dim=-1).gather(1, target[:, None]).squeeze(1)
        results.append(logp.detach().cpu() if detach else logp)
    return results
