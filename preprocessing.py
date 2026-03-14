from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


@dataclass
class PreprocessingConfig:
    """
    Minimal preprocessing settings.

    Keep this simple for now. Later you can add your real filtering logic here.
    """

    selected_channel_indices: Optional[Iterable[int]] = None
    center_each_channel: bool = True
    clip_uv: Optional[float] = None


class EEGPreprocessor:
    """
    Very light placeholder preprocessing.

    For now this just:
    1. selects channels
    2. optionally mean-centers each channel
    3. optionally clips extreme values

    Later you can replace this with your real pipeline so training and live use match.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def transform(self, eeg_window: np.ndarray) -> np.ndarray:
        x = np.asarray(eeg_window, dtype=float)

        if self.config.selected_channel_indices is not None:
            idx = list(self.config.selected_channel_indices)
            x = x[idx, :]

        if self.config.center_each_channel:
            x = x - np.mean(x, axis=1, keepdims=True)

        if self.config.clip_uv is not None:
            clip = float(self.config.clip_uv)
            x = np.clip(x, -clip, clip)

        return x
