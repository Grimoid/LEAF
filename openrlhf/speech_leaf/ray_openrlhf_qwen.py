"""Ray actor classes for Qwen2-Audio RL training.

Mirrors ray_openrlhf.py but uses:
  - load_processor_qwen  (AutoProcessor without Granite version check)
  - FrozenQwenActor / QwenActor  (AutoModelForConditionalGeneration)
  - QwenReinforceTrainer  (Qwen-specific experience maker + eval)
"""
from __future__ import annotations

import math
import os

import ray
import torch

from openrlhf.speech_leaf.actor_qwen import FrozenQwenActor, QwenActor
from openrlhf.speech_leaf.loaders_qwen import load_processor_qwen
from openrlhf.speech_leaf.prompt_dataset import SpeechPromptDataset
from openrlhf.speech_leaf.trainer_openrlhf_qwen import QwenReinforceTrainer
from openrlhf.trainer.ppo_utils.global_envs import RUNTIME_ENV
from openrlhf.trainer.ray.launcher import BasePPORole
from openrlhf.utils import DeepspeedStrategy
from openrlhf.utils.utils import get_cosine_schedule_with_warmup

# Re-use the scheduler and checkpoint helpers from the Granite module.
from openrlhf.speech_leaf.ray_openrlhf import _build_scheduler, _find_latest_actor_ckpt


@ray.remote(num_gpus=1, runtime_env=RUNTIME_ENV)
class QwenReferenceModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        print(
            f"[QwenReferenceModelRayActor rank={self._rank}] init_model_from_pretrained start: {pretrain}",
            flush=True,
        )
        self._setup_distributed(strategy)
        print(f"[QwenReferenceModelRayActor rank={self._rank}] distributed setup complete", flush=True)
        self.processor = load_processor_qwen(pretrain)
        self.tokenizer = self.processor.tokenizer
        model = FrozenQwenActor(pretrain, bf16=strategy.args.bf16)
        print(f"[QwenReferenceModelRayActor rank={self._rank}] model loaded", flush=True)
        self.model = self.strategy.prepare(model, is_rlhf=True)
        print(f"[QwenReferenceModelRayActor rank={self._rank}] strategy.prepare complete", flush=True)
        self.model.eval()
        print(f"[QwenReferenceModelRayActor rank={self._rank}] init_model_from_pretrained done", flush=True)

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: torch.Tensor | None = None,
        extra_model_inputs: dict[str, torch.Tensor] | None = None,
        return_output: bool = False,
    ):
        device = torch.cuda.current_device()
        extra_model_inputs = {
            key: value.to(device) for key, value in (extra_model_inputs or {}).items()
        }
        with torch.no_grad():
            output = self.model(
                sequences.to(device),
                num_actions,
                attention_mask.to(device) if attention_mask is not None else None,
                extra_model_inputs=extra_model_inputs,
                return_output=return_output,
            )
        if return_output:
            log_probs, model_output = output
            return log_probs.to("cpu"), {key: value.to("cpu") for key, value in model_output.items()}
        return output.to("cpu")


@ray.remote(num_gpus=1, runtime_env=RUNTIME_ENV)
class QwenActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        print(
            f"[QwenActorModelRayActor rank={self._rank}] init_model_from_pretrained start: {pretrain}",
            flush=True,
        )
        self._setup_distributed(strategy)
        print(f"[QwenActorModelRayActor rank={self._rank}] distributed setup complete", flush=True)
        self.processor = load_processor_qwen(pretrain)
        self.tokenizer = self.processor.tokenizer
        args = strategy.args
        target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
        actor = QwenActor(
            pretrain,
            bf16=args.bf16,
            use_lora=bool(args.use_lora),
            continue_lora=bool(args.continue_lora),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        print(f"[QwenActorModelRayActor rank={self._rank}] actor model loaded", flush=True)

        self.prepare_datasets()
        print(f"[QwenActorModelRayActor rank={self._rank}] datasets prepared", flush=True)

        actor_optim = strategy.create_optimizer(
            actor,
            lr=args.actor_learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.l2,
        )
        print(f"[QwenActorModelRayActor rank={self._rank}] optimizer created", flush=True)

        num_update_steps_per_episodes = len(self.prompts_dataloader) * args.max_epochs // strategy.accumulated_gradient
        max_steps = max(1, math.ceil(args.num_episodes * num_update_steps_per_episodes))
        self.max_steps = max_steps
        actor_scheduler = _build_scheduler(
            actor_optim,
            args.lr_scheduler_type,
            warmup_steps=max(1, math.ceil(max_steps * 0.05)),
            training_steps=max_steps,
            min_lr=args.min_actor_learning_rate_lr,
        )

        if args.gradient_checkpointing:
            try:
                actor.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
                )
            except ValueError as exc:
                strategy.print(
                    f"Warning: gradient checkpointing is unavailable for {actor.model.__class__.__name__}: {exc}"
                )
        if hasattr(actor.model, "config") and hasattr(actor.model.config, "use_cache"):
            actor.model.config.use_cache = False

        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )
        print(f"[QwenActorModelRayActor rank={self._rank}] strategy.prepare complete", flush=True)
        self.past_global_step = 0
        if args.load_ckpt:
            past_global_step, target_dir = _find_latest_actor_ckpt(args.ckpt_path)
            if target_dir is not None:
                self.past_global_step = past_global_step
                strategy.load_ckpt(
                    self.actor.model,
                    load_dir=target_dir,
                    load_lr_scheduler_states=True,
                    load_optimizer_states=True,
                    load_module_strict=True,
                )
                print(
                    f"[QwenActorModelRayActor rank={self._rank}] loaded ckpt from {target_dir}",
                    flush=True,
                )
            else:
                print(
                    f"[QwenActorModelRayActor rank={self._rank}] no actor ckpt found under {args.ckpt_path}",
                    flush=True,
                )
        self.strategy.args.past_global_steps = self.past_global_step
        self.ema_model = None
        print(f"[QwenActorModelRayActor rank={self._rank}] init_model_from_pretrained done", flush=True)

    def prepare_datasets(self):
        args = self.strategy.args
        dataset = SpeechPromptDataset(
            args.speech_data_dir,
            split=args.train_split,
            max_samples=args.max_samples,
        )
        self.eval_dataset = None
        if int(getattr(args, "eval_steps", -1)) != 0:
            self.eval_dataset = SpeechPromptDataset(
                args.speech_data_dir,
                split=getattr(args, "eval_split", "validation"),
                max_samples=None,
            )
        self.prompts_dataloader = self.strategy.setup_dataloader(
            dataset,
            args.micro_rollout_batch_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )
        self.pretrain_dataloader = None

    def fit(
        self,
        initial_model,
        reward_model=None,
        remote_rm_url=None,
        reward_fn=None,
        vllm_engines=None,
    ):
        del reward_model, remote_rm_url, reward_fn, vllm_engines
        args = self.strategy.args
        trainer = QwenReinforceTrainer(
            self.strategy,
            self.actor,
            None,
            initial_model,
            ema_model=None,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            tokenizer=self.tokenizer,
            prompt_max_len=args.prompt_max_len,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=0.0,
            max_norm=args.max_norm,
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_mcts=args.use_mcts,
            use_vinevalue=False,
            processor=self.processor,
            reward_mode=args.reward_mode,
            lowercase_bleu=bool(getattr(args, "lowercase_bleu", 1)),
            eval_dataset=self.eval_dataset,
        )
        trainer.fit(self.prompts_dataloader, self.pretrain_dataloader, args)

    def save_model(self):
        args = self.strategy.args
        self.strategy.save_model(self.actor, self.tokenizer, args.save_path)
        if self.strategy.is_rank_0():
            self.processor.save_pretrained(args.save_path)
            if getattr(self.processor, "tokenizer", None) is not None:
                self.processor.tokenizer.save_pretrained(args.save_path)
