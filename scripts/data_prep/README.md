# Data preparation (`scripts/data_prep/`)

Builds the HuggingFace `save_to_disk` datasets consumed by training and evaluation. Each
prepared dataset is a directory with a `dataset/` subfolder (`DatasetDict` with `train` /
`validation` / `test`), rows `{"prompt", "reference", "audio"}` (+ `question` where available).
Put them under `$SPEECH_DATA_ROOT` (default `<repo>/data`); `scripts/configs/datasets.yaml`
maps dataset names to these directories.

| Dataset (`datasets.yaml`) | Script |
|---|---|
| `librisqa` | `prepare_librisqa.py` (needs LibriSpeech audio + the LibriSQA Part I JSONs) |
| `dailytalk` | `download_and_prepare_dailytalk_longaudio.sh` → `prepare_dailytalk_longaudio.py` — DailyTalk audio + the LongAudio QA prompts, multiple-choice examples removed, audio capped at 40 s; notes in `dailytalk_longaudio_dataset_creation.txt` |
| `longaudio` | `download_and_prepare_speech_longaudio.sh` — merges the Europarl-ST and VoxPopuli LongAudio splits (built first via `download_and_prepare_europarl_longaudio.sh` / `download_and_prepare_voxpopuli_longaudio.sh`), audio capped at 40 s |
| `covost2` (+ `_single_prompt` for eval) | `download_and_prepare_covost2.sh` → `prepare_covost2_en_de.py`, `prepare_covost2_en_de_single_prompt.py` |

```bash
export SPEECH_DATA_ROOT=/path/to/data
python scripts/data_prep/prepare_librisqa.py --help
bash scripts/data_prep/download_and_prepare_speech_longaudio.sh
```

## Licenses of the source corpora

**This repository redistributes no audio or text from any corpus.** The scripts above
download each source at build time; obtaining and using the data is subject to the source
licenses, which the user must respect:

| Source | License |
|---|---|
| LibriSpeech (audio for LibriSQA) | CC BY 4.0 |
| LibriSQA Part I QA pairs | per its Hugging Face dataset card (`ZihanZhao/LibriSQA`) |
| Common Voice (audio for CoVoST2) | CC0, via Mozilla (account/API key required to download) |
| CoVoST2 translations | CC BY-NC 4.0 (non-commercial) |
| DailyTalk audio | CC BY-SA 4.0 |
| LongAudio QA annotations (DailyTalk / Europarl / VoxPopuli QA pairs) | per its Hugging Face dataset card (`nvidia/LongAudio`) |
| Europarl-ST | its own research license |
| VoxPopuli | CC0 |

Note in particular that the CoVoST2 translations are **non-commercial** (CC BY-NC 4.0):
models trained on them and the prepared dataset inherit that restriction.
