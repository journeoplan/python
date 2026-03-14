import numpy as np
from typing import Optional, Tuple

from src.signal.filter import ButterworthFilter
from src.signal.fft_analyzer import FFTAnalyzer
from src.signal.smoother import MedianSmoother
from src.config import (
    FPS, BREATH_HZ_MIN, BREATH_HZ_MAX, SMOOTHER_WINDOW
)


class BreathDetector:
    """호흡 BPM 감지기.

    신호 처리 파이프라인:
        raw → DC 제거 → Butterworth 밴드패스 → FFT → 피크 → BPM
    """

    def __init__(self, fs: float = FPS):
        self._filter = ButterworthFilter(BREATH_HZ_MIN, BREATH_HZ_MAX, fs)
        self._fft = FFTAnalyzer(fs)
        self._smoother = MedianSmoother(SMOOTHER_WINDOW)

    def process(
        self, signal: np.ndarray
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """신호에서 호흡 BPM 추출.

        Args:
            signal: 1D 원시 신호 배열

        Returns:
            (smoothed_bpm, filtered_signal) 튜플.
            데이터 부족 시 (None, None)
        """
        clean = signal[~np.isnan(signal)]
        if len(clean) < FPS * 4:
            return None, None

        filtered = self._filter.apply(clean)
        peak_hz = self._fft.peak_frequency(filtered, BREATH_HZ_MIN, BREATH_HZ_MAX)

        if peak_hz is None:
            return None, filtered

        raw_bpm = peak_hz * 60.0
        smoothed = self._smoother.update(raw_bpm)
        return smoothed, filtered
