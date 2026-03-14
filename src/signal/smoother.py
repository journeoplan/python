import numpy as np
from collections import deque
from typing import Optional


class MedianSmoother:
    """중앙값 기반 BPM 평활화."""

    def __init__(self, window_size: int = 30):
        self._window = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self._window.append(value)
        return float(np.median(list(self._window)))

    def reset(self) -> None:
        self._window.clear()
