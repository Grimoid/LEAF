from __future__ import annotations

from typing import Dict, List, Optional

import ray
import torch

from openrlhf.models.utils import compute_approx_kl, compute_reward_naive, masked_mean
from openrlhf.speech_leaf.audio_utils import load_audio_array
from openrlhf.speech_leaf.model_utils import build_batched_full_inputs, sync_audio_features
from openrlhf.speech_leaf.replay_buffer import (
    SpeechExperience,
    zero_pad_last_dim,
    zero_pad_sequences,
)
from openrlhf.speech_leaf.reward import RewardConfig, compute_reward
from openrlhf.speech_leaf.sampler import (
    GenConfig,
    PrefixTreeConfig,
    PrefixTreeTrainSequence,
    PrefixTreeTrainStep,
    build_prefix_tree_from_trajectories,
    generate_batch,
    generate_batch_multi_prompt,
    normalize_step_advantages_zscore,
    sample_prefix_tree_retro_batch,
)
from openrlhf.speech_leaf.text_utils import normalize_text


class SpeechRemoteExperienceMaker:
    def __init__(
        self,
        actor,
        initial_model,
        processor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy,
        reward_mode: str = "bleu",
        lowercase_bleu: bool = True,
    ) -> None:
        self.actor = actor
        self.initial_model = initial_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_cfg = RewardConfig(mode=reward_mode, lowercase_bleu=lowercase_bleu)
        self.audio_token = getattr(processor, "audio_token", "<|audio|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)

    def _unwrap_model(self, model):
        return model.model.module if hasattr(model.model, "module") else model.model

    def _format_prompt(self, raw_prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return raw_prompt
        messages = [{"role": "user", "content": raw_prompt}]
        system_prompt = getattr(self.strategy.args, "system_prompt", None)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _prepare_prompt_inputs(self, example: dict, device: torch.device) -> Dict[str, torch.Tensor]:
        audio_array, _ = load_audio_array(example["audio"], target_sampling_rate=16000)
        prompt = self._format_prompt(example["prompt"])
        inputs = self.processor(
            text=prompt,
            audio=audio_array,
            return_tensors="pt",
            truncation=True,
            max_length=self.prompt_max_len,
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        return sync_audio_features(inputs, self.audio_token_id)

    def _leaf_reward(self, reference: str, hypothesis: str) -> float:
        return float(compute_reward(reference, hypothesis, self.reward_cfg))

    def _gen_cfg(self, generate_kwargs) -> GenConfig:
        return GenConfig(
            max_new_tokens=int(generate_kwargs.get("max_new_tokens", self.strategy.args.generate_max_len)),
            temperature=float(generate_kwargs.get("temperature", self.strategy.args.temperature)),
            top_p=float(generate_kwargs.get("top_p", self.strategy.args.top_p)),
            top_k=int(generate_kwargs.get("top_k", getattr(self.strategy.args, "top_k", 0) or 0)),
            do_sample=bool(generate_kwargs.get("do_sample", True)),
        )

    def _tree_cfg(self) -> PrefixTreeConfig:
        # LEAF tree budget: K = total rollouts per prompt, B = number of fork points.
        args = self.strategy.args
        return PrefixTreeConfig(
            K=int(args.rollout_budget_K),
            B=int(args.fork_budget_B),
            min_prefix_tokens=int(args.fork_min_prefix_tokens),
            advantage_terms=str(getattr(args, "advantage_terms", "both")),
        )

    def _tree_method(self) -> str:
        return str(getattr(self.strategy.args, "speech_tree_method", "retro_batch"))

    def _tree_advantage_norm(self) -> str:
        return str(getattr(self.strategy.args, "speech_tree_advantage_norm", "none"))

    def _sample_iid_sequences(
        self,
        policy_model,
        model_inputs: Dict[str, torch.Tensor],
        reference: str,
        num_samples: int,
        gen_cfg: GenConfig,
    ) -> List[PrefixTreeTrainSequence]:
        trajectories = generate_batch(
            model=policy_model,
            tokenizer=self.tokenizer,
            model_inputs=model_inputs,
            cfg=gen_cfg,
            num_return_sequences=max(1, num_samples),
            compute_surprisal=False,
        )
        if not trajectories:
            return []
        leaf_rewards = torch.tensor(
            [self._leaf_reward(reference, traj.text) for traj in trajectories],
            dtype=torch.float32,
        )
        centered = leaf_rewards - leaf_rewards.mean()
        scale = leaf_rewards.std().clamp(min=1 / 3)
        normalized = centered / (scale + 1e-6)
        return [
            PrefixTreeTrainSequence(
                gen_token_ids=traj.gen_token_ids.cpu(),
                logp=traj.logp.cpu(),
                text=traj.text,
                steps=[
                    PrefixTreeTrainStep(
                        step_start=0,
                        step_end=int(traj.gen_token_ids.numel()),
                        advantage=float(normalized[idx].item()),
                    )
                ],
                leaf_reward=float(leaf_rewards[idx].item()),
            )
            for idx, traj in enumerate(trajectories)
            if int(traj.gen_token_ids.numel()) > 0
        ]

    def _trajs_to_zscore_sequences(
        self,
        trajectories: list,
        reference: str,
    ) -> list:
        """Convert pre-generated trajectories to z-score-advantaged PrefixTreeTrainSequences.

        Same logic as _sample_iid_sequences but takes already-generated trajectories
        (from batched multi-prompt generation) instead of generating them.
        """
        if not trajectories:
            return []
        leaf_rewards = torch.tensor(
            [self._leaf_reward(reference, traj.text) for traj in trajectories],
            dtype=torch.float32,
        )
        centered = leaf_rewards - leaf_rewards.mean()
        scale = leaf_rewards.std().clamp(min=1 / 3)
        normalized = centered / (scale + 1e-6)
        return [
            PrefixTreeTrainSequence(
                gen_token_ids=traj.gen_token_ids.cpu(),
                logp=traj.logp.cpu(),
                text=traj.text,
                steps=[
                    PrefixTreeTrainStep(
                        step_start=0,
                        step_end=int(traj.gen_token_ids.numel()),
                        advantage=float(normalized[idx].item()),
                    )
                ],
                leaf_reward=float(leaf_rewards[idx].item()),
            )
            for idx, traj in enumerate(trajectories)
            if int(traj.gen_token_ids.numel()) > 0
        ]

    def _sample_tree_sequences(
        self,
        policy_model,
        model_inputs: Dict[str, torch.Tensor],
        reference: str,
        gen_cfg: GenConfig,
    ) -> List[PrefixTreeTrainSequence]:
        tree_method = self._tree_method()
        if tree_method == "retro_batch":
            # LEAF: K i.i.d. rollouts, retroactive prefix-tree reconstruction at B fork points
            train_sequences = sample_prefix_tree_retro_batch(
                model=policy_model,
                tokenizer=self.tokenizer,
                model_inputs=model_inputs,
                gen_cfg=gen_cfg,
                reward_fn=lambda hypothesis: self._leaf_reward(reference, hypothesis),
                prefix_tree_cfg=self._tree_cfg(),
            )
        elif tree_method == "grpo_paper":
            # Paper-exact GRPO: i.i.d. samples, z-score advantages, routed
            # through leaf_policy_loss for symmetric clipping + explicit KL.
            train_sequences = self._sample_iid_sequences(
                policy_model,
                model_inputs,
                reference,
                int(getattr(self.strategy.args, "rollout_budget_K", 8)),
                gen_cfg,
            )
        else:
            raise ValueError(f"Unsupported speech_tree_method: {tree_method}")

        adv_norm = self._tree_advantage_norm()
        if adv_norm == "zscore":
            normalize_step_advantages_zscore(train_sequences)
        elif adv_norm != "none":
            raise ValueError(f"Unsupported speech_tree_advantage_norm: {adv_norm}")
        return train_sequences

    def _reward_vector(self, train_sequence: PrefixTreeTrainSequence) -> torch.Tensor:
        reward = torch.zeros(int(train_sequence.gen_token_ids.numel()), dtype=torch.float32)
        for step in train_sequence.steps:
            reward[step.step_start : step.step_end] = float(step.advantage)
        return reward

    @torch.no_grad()
    def make_experience(self, prompts: List[dict], use_mcts: bool, use_vinevalue: bool = False, **generate_kwargs):
        del use_vinevalue
        device = torch.device("cuda", torch.cuda.current_device())
        policy_model = self._unwrap_model(self.actor)
        gen_cfg = self._gen_cfg(generate_kwargs)
        tree_method = self._tree_method()
        process_supervision = bool(getattr(self.strategy.args, "process_supervision", False))
        if use_mcts and tree_method in ("retro_batch", "grpo_paper"):
            process_supervision = True
        explicit_tree_kl = bool(getattr(self.strategy.args, "speech_tree_explicit_kl", False))
        explicit_tree_kl = explicit_tree_kl and use_mcts and process_supervision

        sequence_rows = []
        attention_rows = []
        action_rows = []
        policy_logp_rows = []
        ref_logp_rows = []
        reward_rows = []
        leaf_rewards = []
        extra_inputs_rows: Dict[str, List[torch.Tensor]] = {}

        # Batched generation for retro_batch and grpo_paper: generate for multiple prompts in one call
        use_batched_gen = (
            use_mcts
            and tree_method in ("retro_batch", "grpo_paper")
            and len(prompts) > 1
            and int(getattr(self.strategy.args, "gen_prompt_batch_size", 4)) > 1
        )

        if use_batched_gen and tree_method == "grpo_paper":
            # ── grpo_paper fast path: cross-prompt batched generation + scoring ──
            gen_batch_size = int(getattr(self.strategy.args, "gen_prompt_batch_size", 4))
            total_seqs = int(getattr(self.strategy.args, "rollout_budget_K", 8))
            pad_token_id = self.tokenizer.pad_token_id or 0

            for chunk_start in range(0, len(prompts), gen_batch_size):
                chunk = prompts[chunk_start : chunk_start + gen_batch_size]
                chunk_model_inputs = [self._prepare_prompt_inputs(ex, device=device) for ex in chunk]
                chunk_references = [normalize_text(ex["reference"]) for ex in chunk]

                per_prompt_trajs = generate_batch_multi_prompt(
                    model=policy_model,
                    tokenizer=self.tokenizer,
                    model_inputs_list=chunk_model_inputs,
                    cfg=gen_cfg,
                    num_return_sequences=total_seqs,
                    compute_surprisal=False,
                    audio_token_id=self.audio_token_id,
                )

                # Phase 1: compute z-score advantages for all prompts (fast, CPU-bound)
                flat_prompt_ids = []
                flat_prompt_attn = []
                flat_gen_ids = []
                flat_extra: Dict[str, list] = {}
                flat_train_seqs = []

                for model_inputs, reference, trajs in zip(
                    chunk_model_inputs, chunk_references, per_prompt_trajs
                ):
                    train_sequences = self._trajs_to_zscore_sequences(trajs, reference)
                    train_sequences = [s for s in train_sequences if int(s.gen_token_ids.numel()) > 0]
                    if not train_sequences:
                        continue
                    p_ids = model_inputs["input_ids"][0]
                    p_attn = model_inputs["attention_mask"][0]
                    extra_keys = [k for k in model_inputs if k not in ("input_ids", "attention_mask")]
                    for seq in train_sequences:
                        flat_prompt_ids.append(p_ids)
                        flat_prompt_attn.append(p_attn)
                        flat_gen_ids.append(seq.gen_token_ids.to(device))
                        flat_train_seqs.append(seq)
                        for key in extra_keys:
                            flat_extra.setdefault(key, []).append(model_inputs[key].squeeze(0))

                if not flat_train_seqs:
                    continue

                # Phase 2: left-pad prompts + right-pad gen → all sequences same length
                gen_lens = [int(g.numel()) for g in flat_gen_ids]
                max_gen = max(gen_lens)
                max_prompt = max(p.numel() for p in flat_prompt_ids)
                total_rows = len(flat_train_seqs)

                all_ids = []
                all_attn = []
                for p_ids, p_attn, g_ids, g_len in zip(
                    flat_prompt_ids, flat_prompt_attn, flat_gen_ids, gen_lens
                ):
                    lpad = max_prompt - p_ids.numel()
                    rpad = max_gen - g_len
                    ids = torch.cat([
                        torch.full((lpad,), pad_token_id, dtype=p_ids.dtype, device=device),
                        p_ids,
                        g_ids,
                        torch.full((rpad,), pad_token_id, dtype=p_ids.dtype, device=device),
                    ])
                    attn = torch.cat([
                        torch.zeros(lpad, dtype=p_attn.dtype, device=device),
                        p_attn,
                        torch.ones(g_len, dtype=p_attn.dtype, device=device),
                        torch.zeros(rpad, dtype=p_attn.dtype, device=device),
                    ])
                    all_ids.append(ids)
                    all_attn.append(attn)

                batched_ids = torch.stack(all_ids, dim=0)
                batched_attn = torch.stack(all_attn, dim=0)

                extra_model_inputs = {}
                for key, values in flat_extra.items():
                    extra_model_inputs[key] = zero_pad_last_dim(values)

                # Sync audio features for the cross-prompt batch
                if self.audio_token_id >= 0 and "input_features_mask" in extra_model_inputs:
                    merged = {"input_ids": batched_ids, "attention_mask": batched_attn}
                    merged.update(extra_model_inputs)
                    merged = sync_audio_features(merged, self.audio_token_id)
                    batched_ids = merged["input_ids"]
                    batched_attn = merged["attention_mask"]
                    extra_model_inputs = {
                        k: v for k, v in merged.items() if k not in ("input_ids", "attention_mask")
                    }

                # Phase 3: one actor forward + one ref forward for ALL sequences
                action_log_probs = self.actor(
                    batched_ids, max_gen,
                    attention_mask=batched_attn,
                    extra_model_inputs=extra_model_inputs,
                )
                ref_action_log_probs = ray.get(
                    self.initial_model.forward.remote(
                        batched_ids.cpu(), max_gen,
                        batched_attn.cpu(),
                        {k: v.cpu() for k, v in extra_model_inputs.items()},
                    )
                ).to(device)

                # Phase 4: collect rows
                action_mask = torch.zeros((total_rows, max_gen), dtype=torch.bool, device=device)
                for row_idx, g_len in enumerate(gen_lens):
                    action_mask[row_idx, :g_len] = True

                for row_idx, train_sequence in enumerate(flat_train_seqs):
                    sequence_rows.append(batched_ids[row_idx].detach())
                    attention_rows.append(batched_attn[row_idx].detach())
                    action_rows.append(action_mask[row_idx].detach())
                    policy_logp_rows.append(action_log_probs[row_idx].detach())
                    ref_logp_rows.append(ref_action_log_probs[row_idx].detach())
                    reward_rows.append(self._reward_vector(train_sequence).to(device))
                    leaf_rewards.append(float(train_sequence.leaf_reward))
                    for key, value in extra_model_inputs.items():
                        extra_inputs_rows.setdefault(key, []).append(value[row_idx].detach())

        elif use_batched_gen:
            # ── LEAF (retro_batch): batched generation across prompts, per-prompt tree build ──
            gen_batch_size = int(getattr(self.strategy.args, "gen_prompt_batch_size", 4))
            prefix_tree_cfg = self._tree_cfg()
            total_seqs = prefix_tree_cfg.total_seqs
            adv_norm = self._tree_advantage_norm()

            for chunk_start in range(0, len(prompts), gen_batch_size):
                chunk = prompts[chunk_start : chunk_start + gen_batch_size]
                chunk_model_inputs = [self._prepare_prompt_inputs(ex, device=device) for ex in chunk]
                chunk_references = [normalize_text(ex["reference"]) for ex in chunk]

                per_prompt_trajs = generate_batch_multi_prompt(
                    model=policy_model,
                    tokenizer=self.tokenizer,
                    model_inputs_list=chunk_model_inputs,
                    cfg=gen_cfg,
                    num_return_sequences=total_seqs,
                    compute_surprisal=True,
                    audio_token_id=self.audio_token_id,
                )

                for prompt_idx, (model_inputs, reference, trajs) in enumerate(
                    zip(chunk_model_inputs, chunk_references, per_prompt_trajs)
                ):
                    reward_fn = lambda hyp, ref=reference: self._leaf_reward(ref, hyp)
                    train_sequences = build_prefix_tree_from_trajectories(trajs, reward_fn, prefix_tree_cfg)
                    if adv_norm == "zscore":
                        normalize_step_advantages_zscore(train_sequences)
                    train_sequences = [s for s in train_sequences if int(s.gen_token_ids.numel()) > 0]
                    if not train_sequences:
                        continue

                    gen_ids_list = [seq.gen_token_ids.to(device) for seq in train_sequences]
                    full_inputs, _, gen_lens = build_batched_full_inputs(
                        model_inputs, gen_ids_list, pad_token_id=self.tokenizer.pad_token_id,
                    )
                    extra_model_inputs = {
                        k: v for k, v in full_inputs.items() if k not in ("input_ids", "attention_mask")
                    }
                    max_actions = max(gen_lens)
                    action_mask = torch.zeros((len(gen_lens), max_actions), dtype=torch.bool, device=device)
                    for row_idx, gen_len in enumerate(gen_lens):
                        action_mask[row_idx, :gen_len] = True

                    action_log_probs = self.actor(
                        full_inputs["input_ids"], max_actions,
                        attention_mask=full_inputs["attention_mask"],
                        extra_model_inputs=extra_model_inputs,
                    )
                    ref_action_log_probs = ray.get(
                        self.initial_model.forward.remote(
                            full_inputs["input_ids"].cpu(), max_actions,
                            full_inputs["attention_mask"].cpu(),
                            {k: v.cpu() for k, v in extra_model_inputs.items()},
                        )
                    ).to(device)

                    for row_idx, train_sequence in enumerate(train_sequences):
                        sequence_rows.append(full_inputs["input_ids"][row_idx].detach())
                        attention_rows.append(full_inputs["attention_mask"][row_idx].detach())
                        action_rows.append(action_mask[row_idx].detach())
                        policy_logp_rows.append(action_log_probs[row_idx].detach())
                        ref_logp_rows.append(ref_action_log_probs[row_idx].detach())
                        reward_rows.append(self._reward_vector(train_sequence).to(device))
                        leaf_rewards.append(float(train_sequence.leaf_reward))
                        for key, value in extra_model_inputs.items():
                            extra_inputs_rows.setdefault(key, []).append(value[row_idx].detach())
        else:
            for example in prompts:
                model_inputs = self._prepare_prompt_inputs(example, device=device)
                reference = normalize_text(example["reference"])
                if use_mcts:
                    train_sequences = self._sample_tree_sequences(policy_model, model_inputs, reference, gen_cfg)
                else:
                    train_sequences = self._sample_iid_sequences(
                        policy_model,
                        model_inputs,
                        reference,
                        int(getattr(self.strategy.args, "rollout_budget_K", 1)),
                        gen_cfg,
                    )
                train_sequences = [seq for seq in train_sequences if int(seq.gen_token_ids.numel()) > 0]
                if not train_sequences:
                    continue

                gen_ids_list = [seq.gen_token_ids.to(device) for seq in train_sequences]
                full_inputs, _, gen_lens = build_batched_full_inputs(
                    model_inputs,
                    gen_ids_list,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                extra_model_inputs = {
                    key: value
                    for key, value in full_inputs.items()
                    if key not in ("input_ids", "attention_mask")
                }
                max_actions = max(gen_lens)
                action_mask = torch.zeros((len(gen_lens), max_actions), dtype=torch.bool, device=device)
                for row_idx, gen_len in enumerate(gen_lens):
                    action_mask[row_idx, :gen_len] = True

                action_log_probs = self.actor(
                    full_inputs["input_ids"],
                    max_actions,
                    attention_mask=full_inputs["attention_mask"],
                    extra_model_inputs=extra_model_inputs,
                )
                ref_action_log_probs = ray.get(
                    self.initial_model.forward.remote(
                        full_inputs["input_ids"].cpu(),
                        max_actions,
                        full_inputs["attention_mask"].cpu(),
                        {key: value.cpu() for key, value in extra_model_inputs.items()},
                    )
                ).to(device)

                for row_idx, train_sequence in enumerate(train_sequences):
                    sequence_rows.append(full_inputs["input_ids"][row_idx].detach())
                    attention_rows.append(full_inputs["attention_mask"][row_idx].detach())
                    action_rows.append(action_mask[row_idx].detach())
                    policy_logp_rows.append(action_log_probs[row_idx].detach())
                    ref_logp_rows.append(ref_action_log_probs[row_idx].detach())
                    reward_rows.append(self._reward_vector(train_sequence).to(device))
                    leaf_rewards.append(float(train_sequence.leaf_reward))
                    for key, value in extra_model_inputs.items():
                        extra_inputs_rows.setdefault(key, []).append(value[row_idx].detach())

        if not sequence_rows:
            raise RuntimeError("No speech rollouts were generated. Check the dataset, model, and generation settings.")

        sequences = zero_pad_sequences(sequence_rows, side="left", value=self.tokenizer.pad_token_id).to(device)
        attention_mask = zero_pad_sequences(attention_rows, side="left", value=0).to(device)
        action_mask = zero_pad_sequences(action_rows, side="right", value=0).to(device).bool()
        action_log_probs = zero_pad_sequences(policy_logp_rows, side="right", value=0).to(device)
        base_action_log_probs = zero_pad_sequences(ref_logp_rows, side="right", value=0).to(device)
        raw_reward = zero_pad_sequences(reward_rows, side="right", value=0).to(device)
        leaf_reward_tensor = torch.tensor(leaf_rewards, dtype=torch.float32, device=device)

        if explicit_tree_kl:
            advantages = (raw_reward * action_mask.float()).float()
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
            )
        else:
            reward_input = raw_reward if process_supervision else leaf_reward_tensor
            advantages, kl = compute_reward_naive(
                reward_input,
                self.kl_ctl.value,
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_as_reward=True,
                process_reward=process_supervision,
            )

        if process_supervision:
            reward_info = masked_mean(raw_reward, raw_reward != 0, dim=-1)
        else:
            reward_info = leaf_reward_tensor

        response_entropy = -(action_log_probs * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1)
        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": leaf_reward_tensor,
            "reward_normalized": reward_info,
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
            "response_overlong_ratio": torch.zeros_like(leaf_reward_tensor),
            "pass_rate": torch.zeros_like(leaf_reward_tensor),
            "response_entropy": response_entropy,
        }

        batched_extra_inputs = {
            key: zero_pad_last_dim(values).to(device)
            for key, values in extra_inputs_rows.items()
        }
        return SpeechExperience(
            sequences=sequences,
            action_log_probs=action_log_probs,
            base_action_log_probs=base_action_log_probs,
            advantages=advantages,
            attention_mask=attention_mask,
            action_mask=action_mask,
            info=info,
            kl=kl,
            model_inputs=batched_extra_inputs,
        )
