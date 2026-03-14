from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
except ImportError:  # pragma: no cover - optional dependency at runtime
    BoardIds = None
    BoardShim = None
    BrainFlowInputParams = None


@dataclass
class CytonStreamConfig:
    """Configuration for a live OpenBCI Cyton stream."""

    serial_port: str = "COM3"
    board_id: int | None = None
    window_seconds: float = 0.5
    startup_buffer_seconds: float = 2.0
    use_synthetic_board: bool = False

    def resolved_board_id(self) -> int:
        if self.use_synthetic_board:
            if BoardIds is None:
                raise ImportError("brainflow is not installed")
            return BoardIds.SYNTHETIC_BOARD.value
        if self.board_id is not None:
            return self.board_id
        if BoardIds is None:
            raise ImportError("brainflow is not installed")
        return BoardIds.CYTON_BOARD.value


class CytonStream:
    """
    Thin wrapper around BrainFlow for live Cyton data.

    Responsibilities:
    1. Connect to the board
    2. Start the stream
    3. Return the newest EEG window as a NumPy array

    Output shape from get_latest_window():
        (n_channels, n_samples)
    """

    def __init__(self, config: CytonStreamConfig):
        self.config = config
        self.board = None
        self.sampling_rate: Optional[int] = None
        self.eeg_channels: List[int] = []
        self.window_size_samples: Optional[int] = None
        self.started = False

    def start(self) -> None:
        if BoardShim is None or BrainFlowInputParams is None:
            raise ImportError(
                "brainflow is required for live Cyton streaming. Install it with `pip install brainflow`."
            )

        if self.started:
            return

        board_id = self.config.resolved_board_id()
        params = BrainFlowInputParams()

        if not self.config.use_synthetic_board:
            params.serial_port = self.config.serial_port

        BoardShim.enable_dev_board_logger()
        self.board = BoardShim(board_id, params)
        self.board.prepare_session()
        self.board.start_stream()

        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.window_size_samples = max(1, int(self.config.window_seconds * self.sampling_rate))
        self.started = True

        startup_wait = max(self.config.window_seconds, self.config.startup_buffer_seconds)
        time.sleep(startup_wait)

    def stop(self) -> None:
        if not self.board:
            self.started = False
            return

        try:
            self.board.stop_stream()
        except Exception:
            pass

        try:
            self.board.release_session()
        except Exception:
            pass

        self.board = None
        self.started = False

    def get_latest_window(self) -> Optional[np.ndarray]:
        """
        Return the newest EEG window with shape (channels, samples).
        Returns None if there are not enough samples yet.
        """
        if not self.started or self.board is None or self.window_size_samples is None:
            return None

        data = self.board.get_current_board_data(self.window_size_samples)
        if data.size == 0:
            return None

        eeg = data[self.eeg_channels, :]
        if eeg.shape[1] < self.window_size_samples:
            return None

        return eeg
