from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import torch

from openrlhf.speech_leaf import GRPOConfig, compute_corpus_metrics, load_audio_array, sync_audio_features
from openrlhf.speech_leaf.experience_maker_openrlhf import SpeechRemoteExperienceMaker
from openrlhf.speech_leaf.loss import leaf_policy_loss
from openrlhf.speech_leaf.replay_buffer import SpeechExperience, SpeechReplayBuffer
from openrlhf.speech_leaf.text_utils import normalize_text
from openrlhf.trainer.reinforce_trainer import ReinforceTrainer


class SpeechReinforceTrainer(ReinforceTrainer):
    def _use_explicit_tree_loss(self, experience: SpeechExperience) -> bool:
        return (
            bool(getattr(self.strategy.args, "speech_tree_explicit_kl", False))
            and bool(getattr(self.strategy.args, "use_mcts", False))
            and str(getattr(self.strategy.args, "speech_tree_method", "retro_batch")) in ("retro_batch", "grpo_paper")
            and experience.base_action_log_probs is not None
        )

    def __init__(self, *args, processor, reward_mode: str = "bleu", lowercase_bleu: bool = True, eval_dataset=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.experience_maker = SpeechRemoteExperienceMaker(
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
        self.replay_buffer = SpeechReplayBuffer(self.micro_train_batch_size, kwargs.get("buffer_limit", 0), True)
        self.eval_log_path = Path(self.strategy.args.save_path) / "eval_log.jsonl"

    def training_step(self, experience: SpeechExperience) -> Dict[str, float]:
        return self.training_step_actor(experience)

    def training_step_actor(self, experience: SpeechExperience) -> Dict[str, float]:
        self.actor.train()
        num_actions = experience.action_mask.size(1)
        # Safety: skip degenerate batches with 0-length sequences
        if experience.sequences.shape[-1] == 0:
            import warnings
            warnings.warn("Skipping training step: sequences have 0 length after batching")
            return {
                "policy_loss": 0.0, "kl": 0.0, "entropy": 0.0,
                "reward": 0.0, "reward_normalized": 0.0,
                "response_length": 0.0, "total_length": 0.0,
                "response_overlong_ratio": 0.0, "pass_rate": 0.0,
                "response_entropy": 0.0,
            }
        # Re-sync audio features after replay-buffer round-trip (split → pad → re-batch
        # can misalign audio token count vs input_features_mask).
        if experience.model_inputs and "input_features_mask" in experience.model_inputs:
            synced = sync_audio_features(
                {"input_ids": experience.sequences, **experience.model_inputs},
                self.experience_maker.audio_token_id,
            )
            experience.model_inputs["input_features_mask"] = synced["input_features_mask"]
        action_log_probs, output = self.actor(
            experience.sequences,
            num_actions,
            attention_mask=experience.attention_mask,
            extra_model_inputs=experience.model_inputs,
            return_output=True,
        )

        if self._use_explicit_tree_loss(experience):
            tree_cfg = GRPOConfig(
                clip_eps=float(self.args.eps_clip),
                beta_kl=float(self.kl_ctl.value),
            )
            actor_loss = leaf_policy_loss(
                logp_new=action_log_probs,
                logp_old=experience.action_log_probs,
                mask=experience.action_mask.float(),
                per_token_advantage=experience.advantages,
                cfg=tree_cfg,
                logp_ref=experience.base_action_log_probs,
            )
        else:
            actor_loss = self.actor_loss_fn(
                action_log_probs,
                experience.action_log_probs,
                experience.advantages,
                action_mask=experience.action_mask,
                kl=experience.kl,
                kl_coef=self.strategy.args.init_kl_coef,
            )

        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.actor, self.actor_optim)

        import deepspeed.utils

        grad_norm = 0.0
        for param in self.actor.model.module.parameters():
            grad_data = deepspeed.utils.safe_get_full_grad(param)
            if grad_data is None:
                continue
            grad_norm += grad_data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        status = {
            "policy_loss": actor_loss.item(),
            "grad_norm": grad_norm,
        }
        for key, value in experience.info.items():
            if key == "kl":
                status[key] = (
                    (value * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[key] = value.mean().item()
        return status

    def _unwrap_policy_model(self):
        model = self.actor.model
        return model.module if hasattr(model, "module") else model

    def _format_prompt(self, raw_prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return raw_prompt
        messages = [{"role": "user", "content": raw_prompt}]
        system_prompt = getattr(self.strategy.args, "system_prompt", None)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def evaluate_bleu(self, global_step: int) -> Dict[str, float]:
        # Run eval on every rank so checkpoint/save synchronization does not leave
        # non-zero ranks idling inside distributed collectives while rank 0 is
        # still generating validation outputs.
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
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
        audio_token = getattr(self.processor, "audio_token", "<|audio|>")
        audio_token_id = self.tokenizer.convert_tokens_to_ids(audio_token)

        for idx in range(n_samples):
            example = self.eval_dataset[idx]
            prompt = self._format_prompt(example["prompt"])
            reference = normalize_text(example["reference"])
            audio_array, _ = load_audio_array(example["audio"], target_sampling_rate=16000)
            model_inputs = self.processor(
                text=prompt,
                audio=audio_array,
                return_tensors="pt",
                truncation=True,
                max_length=self.prompt_max_len,
            )
            model_inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in model_inputs.items()
            }
            model_inputs = sync_audio_features(model_inputs, audio_token_id)

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

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        saved_model = False
        if global_step % args.logging_steps == 0:
            step_bar.set_postfix(logs_dict)
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)

        if args.eval_steps > 0 and global_step % args.eval_steps == 0:
            self.evaluate_bleu(global_step)
            # Save checkpoint at every eval point
            tag = f"global_step{global_step}"
            self.strategy.save_model(
                self.actor.model, self.tokenizer, os.path.join(args.ckpt_path, f"_actor_{tag}")
            )
            saved_model = True

        if args.save_steps > 0 and global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_model(
                self.actor.model, self.tokenizer, os.path.join(args.ckpt_path, f"_actor_{tag}")
            )
            saved_model = True

        if self.strategy.args.save_ckpt and saved_model:
            tag = f"global_step{global_step}"
            os.makedirs(os.path.join(args.ckpt_path, f"_actor_ckpt_{tag}"), exist_ok=True)
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, f"_actor_ckpt_{tag}"),
                tag,
                max_num=args.max_ckpt_num,
                max_mem=args.max_ckpt_mem,
            )
