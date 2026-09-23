from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class SequenceBatch:
    x: torch.Tensor
    lengths: torch.Tensor
    detected: torch.Tensor
    t_detect: torch.Tensor
    t_onset: torch.Tensor
    entity_id: torch.Tensor


class SequenceDataset(Dataset):
    REQUIRED: ClassVar[set[str]] = {"x", "lengths", "detected", "t_detect", "t_onset", "entity_id"}

    def __init__(self, arrays: dict[str, np.ndarray]):
        missing = self.REQUIRED - arrays.keys()
        if missing:
            raise ValueError(f"Missing arrays: {sorted(missing)}")
        self.arrays = arrays
        n = len(arrays["x"])
        if any(len(v) != n for v in arrays.values()):
            raise ValueError("All arrays must have the same first dimension")
        if arrays["x"].ndim != 3:
            raise ValueError("x must have shape [N,L,D]")
        if np.any(arrays["lengths"] < 1) or np.any(arrays["lengths"] > arrays["x"].shape[1]):
            raise ValueError("lengths outside padded sequence range")

    def __len__(self) -> int:
        return len(self.arrays["x"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.as_tensor(self.arrays["x"][index], dtype=torch.float32),
            "lengths": torch.as_tensor(self.arrays["lengths"][index], dtype=torch.long),
            "detected": torch.as_tensor(self.arrays["detected"][index], dtype=torch.bool),
            "t_detect": torch.as_tensor(self.arrays["t_detect"][index], dtype=torch.long),
            "t_onset": torch.as_tensor(self.arrays["t_onset"][index], dtype=torch.long),
            "entity_id": torch.as_tensor(self.arrays["entity_id"][index], dtype=torch.long),
        }


def load_npz(path: str | Path) -> SequenceDataset:
    with np.load(path, allow_pickle=False) as content:
        return SequenceDataset({key: content[key] for key in content.files})


def _split_indices(n: int, ratios: list[float], seed: int) -> list[np.ndarray]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut1 = int(n * ratios[0])
    cut2 = cut1 + int(n * ratios[1])
    return [order[:cut1], order[cut1:cut2], order[cut2:]]


def make_loaders(dataset: SequenceDataset, batch_size: int, ratios: list[float], seed: int):
    indices = _split_indices(len(dataset), ratios, seed)
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        DataLoader(
            Subset(dataset, idx.tolist()),
            batch_size=batch_size,
            shuffle=(part == 0),
            generator=generator,
        )
        for part, idx in enumerate(indices)
    )
