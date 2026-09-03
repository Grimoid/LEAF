from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from datasets import Audio, load_from_disk
from torch.utils.data import Dataset


def resolve_dataset_dir(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    dataset_dir = root / "dataset" if (root / "dataset").exists() else root
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Speech dataset directory not found: {dataset_dir}")
    return dataset_dir


class SpeechPromptDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train", max_samples: int | None = None) -> None:
        super().__init__()
        dataset = load_from_disk(str(resolve_dataset_dir(data_dir)))
        if split not in dataset:
            if split == "validation" and "test" in dataset:
                split = "test"
            else:
                raise KeyError(f"Split '{split}' not found in {data_dir}; available: {list(dataset.keys())}")
        self.dataset = dataset[split]
        if "audio" in self.dataset.column_names:
            # Avoid datasets' librosa-based auto decoding inside worker processes.
            self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
        if max_samples is not None and max_samples > 0:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]

    def collate_fn(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return batch
