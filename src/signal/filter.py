import numpy as np
from scipy.signal import butter, filtfilt


class ButterworthFilter:
    """Butterworth 밴드패스 필터.

    카메라 및 WiFi CSI 신호 모두에서 재사용 가능한 범용 필터.
    """

    def __init__(self, lowcut: float, highcut: float, fs: float, order: int = 4):
        """
        Args:
            lowcut: 하한 주파수 (Hz)
            highcut: 상한 주파수 (Hz)
            fs: 샘플링 레이트 (Hz)
            order: 필터 차수
        """
        self._lowcut = lowcut
        self._highcut = highcut
        self._fs = fs
        self._order = order
        nyq = 0.5 * fs
        self._b, self._a = butter(
            order,
            [lowcut / nyq, highcut / nyq],
            btype="band",
        )

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """DC 제거 후 밴드패스 필터 적용.

        Args:
            signal: 1D 배열

        Returns:
            필터링된 신호 (동일 shape)
        """
        if len(signal) < 4 * self._order:
            return signal
        centered = signal - np.mean(signal)
        return filtfilt(self._b, self._a, centered)
