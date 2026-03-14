from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ClassifierOutput:
    jump_detected: bool
    score: Optional[float] = None


class BaseLiveClassifier:
    """
    Interface for your future live classifier.

    The contract is simple:
        input  -> preprocessed EEG window, shape (channels, samples)
        output -> whether the bird should jump
    """

    def predict_window(self, eeg_window: np.ndarray) -> ClassifierOutput:
        raise NotImplementedError


class PlaceholderClassifier(BaseLiveClassifier):
    """No-op classifier so the live loop can be built before model training."""

    def predict_window(self, eeg_window: np.ndarray) -> ClassifierOutput:
        _ = eeg_window
        return ClassifierOutput(jump_detected=False, score=0.0)
