import numpy as np
from collections import deque


class MedianSmoother:
    """중앙값 기반 BPM 평활화.

    NaN 입력은 무시하여 윈도우 오염을 방지한다.
    """

    def __init__(self, window_size: int = 30):
        self._window: deque = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        """새 값을 추가하고 중앙값 반환.

        Args:
            value: 새 BPM 값. NaN이면 무시.

        Returns:
            현재 윈도우의 중앙값. 윈도우가 비어있으면 NaN.
        """
        if not np.isnan(value):
            self._window.append(value)
        if len(self._window) == 0:
            return float("nan")
        return float(np.median(list(self._window)))

    def reset(self) -> None:
        """윈도우 초기화."""
        self._window.clear()
