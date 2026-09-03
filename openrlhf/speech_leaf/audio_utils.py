from __future__ import annotations

import io
from typing import Tuple

import soundfile as sf
import torch
import torchaudio


def load_audio_array(audio: dict, target_sampling_rate: int = 16000) -> Tuple[object, int]:
    if not isinstance(audio, dict):
        raise ValueError("Expected an audio dict from the prepared speech dataset.")

    if "array" in audio:
        audio_array = audio["array"]
        sampling_rate = int(audio["sampling_rate"])
    elif audio.get("bytes") is not None:
        audio_array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]), always_2d=False)
    else:
        audio_path = audio.get("path")
        if not audio_path:
            raise ValueError(f"Audio example has no waveform, bytes, or path: {audio}")
        audio_array, sampling_rate = sf.read(audio_path, always_2d=False)

    if getattr(audio_array, "ndim", 1) > 1:
        audio_array = audio_array.mean(axis=-1)

    if int(sampling_rate) != int(target_sampling_rate):
        audio_tensor = torch.as_tensor(audio_array, dtype=torch.float32)
        audio_tensor = torchaudio.functional.resample(audio_tensor, int(sampling_rate), int(target_sampling_rate))
        audio_array = audio_tensor.cpu().numpy()
        sampling_rate = int(target_sampling_rate)

    return audio_array, int(sampling_rate)
