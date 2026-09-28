# LEAF: Growing Trees Without Branching for Speech-Aware LLM Post-Training (EMNLP 2026)

**Argyrios Gerogiannis**, Yekaterina Yegorova, Mark Hasegawa-Johnson, Venugopal V. Veeravalli

Paper link: [https://arxiv.org/pdf/2606.07610](https://arxiv.org/pdf/2606.07610)


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{gerogiannis2026leafgrowingtreesbranching,
      title={LEAF: Growing Trees Without Branching for Speech-Aware Large Language Model Post-Training}, 
      author={Argyrios Gerogiannis and Yekaterina Yegorova and Mark Hasegawa-Johnson and Venugopal V. Veeravalli},
      year={2026},
      eprint={2606.07610},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.07610}, 
}
```


---

## Repository layout

```
train_reinforce_ray_speech.py   training entrypoint (--method leaf|grpo, --model_family granite|qwen)
openrlhf/
  speech_leaf/                the speech RL package
    sampler.py                  K i.i.d. rollouts + retrospective prefix-tree construction (LEAF)
    experience_maker_openrlhf.py rollouts -> rewards -> advantages -> KL (leaf / grpo)
    trainer_openrlhf.py         training step (clipped loss + explicit KL), validation, checkpoints
    loss.py, reward.py          leaf_policy_loss, sentence-BLEU reward
    ray_openrlhf.py, actor.py, loaders.py, replay_buffer.py, model_utils.py, metrics.py
    *_qwen.py                   Qwen2-Audio variants
    fork_analysis.py, fork_position_logging.py   fork statistics (analysis only)
  trainer/, models/, utils/, datasets/   RL framework core (Ray actor groups, DeepSpeed strategy, ReinforceTrainer)
scripts/
  speechrl.py, speechrl.sh, configs/   config-driven launcher (methods / models / datasets / defaults)
  data_prep/                    dataset construction
  eval/                         text-metric evaluation
  judge/                        LLM-as-judge (DA-100, Likert, pairwise)
  voicebench/                   VoiceBench open-ended QA evaluation
  fork_stats/                   fork-fire statistics and figures
  slurm/                        Slurm template
```

The `openrlhf/trainer`, `openrlhf/models`, `openrlhf/utils` and `openrlhf/datasets` packages
are the underlying RL-framework core needed by the Ray + DeepSpeed training loop; the speech
code does not use their math/text-reasoning parts (`parallel_mcts`, `evaluation`, …), which are
kept only because the trainer package imports them.

## Author, license and acknowledgements

Code by **Argyrios Gerogiannis**. Licensed under Apache 2.0 (see `LICENSE` and `NOTICE`).
The Ray/DeepSpeed RL infrastructure builds on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
and [TreeRL](https://github.com/THUDM/TreeRL) (Hou et al., ACL 2025), which are gratefully
acknowledged. The GRPO baseline follows Elmakies et al., "Advancing Speech Understanding in
Speech-Aware Language Models with GRPO"; VoiceBench is Chen et al., TACL 2026.
