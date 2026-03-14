from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from classifier_interface import BaseLiveClassifier, PlaceholderClassifier
from cyton_stream import CytonStream, CytonStreamConfig
from preprocessing import EEGPreprocessor, PreprocessingConfig


@dataclass
class BCIControllerConfig:
    """Top-level config for live BCI control."""

    serial_port: str = "COM3"
    use_synthetic_board: bool = False
    window_seconds: float = 0.5
    cooldown_seconds: float = 0.35
    enabled: bool = True


class BCIController:
    """
    Bridge between the live EEG stream and the game.

    Public contract used by flappy.py:
        start()
        stop()
        should_jump() -> bool

    Internally this controller:
        1. pulls the newest EEG window from the Cyton
        2. preprocesses it
        3. passes it to a classifier
        4. applies cooldown logic
        5. returns True only when the bird should flap
    """

    def __init__(
        self,
        config: Optional[BCIControllerConfig] = None,
        classifier: Optional[BaseLiveClassifier] = None,
    ):
        self.config = config or BCIControllerConfig()
        self.last_jump_time = 0.0
        self.enabled = self.config.enabled

        stream_config = CytonStreamConfig(
            serial_port=self.config.serial_port,
            window_seconds=self.config.window_seconds,
            use_synthetic_board=self.config.use_synthetic_board,
        )
        self.stream = CytonStream(stream_config)
        self.preprocessor = EEGPreprocessor(PreprocessingConfig())
        self.classifier = classifier or PlaceholderClassifier()

    def start(self) -> None:
        if not self.enabled:
            return
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()

    def should_jump(self) -> bool:
        if not self.enabled:
            return False

        window = self.stream.get_latest_window()
        if window is None:
            return False

        processed = self.preprocessor.transform(window)
        result = self.classifier.predict_window(processed)

        if result.jump_detected and self._cooldown_ready():
            self.last_jump_time = time.time()
            return True

        return False

    def _cooldown_ready(self) -> bool:
        return (time.time() - self.last_jump_time) >= self.config.cooldown_seconds
