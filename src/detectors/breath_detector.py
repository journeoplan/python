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

    fs(샘플링 레이트)를 실제 센서 FPS로 설정하면
    저사양 기기에서도 필터/FFT 정확도를 유지한다.
    """

    def __init__(self, fs: float = FPS):
        self._fs = fs
        self._filter = ButterworthFilter(BREATH_HZ_MIN, BREATH_HZ_MAX, fs)
        self._fft = FFTAnalyzer(fs)
        self._smoother = MedianSmoother(SMOOTHER_WINDOW)

    @property
    def fs(self) -> float:
        """현재 샘플링 레이트."""
        return self._fs

    def update_fs(self, new_fs: float) -> None:
        """샘플링 레이트 갱신 — 필터와 FFT를 재생성한다.

        센서의 실제 FPS가 안정화된 후 호출하면
        필터 대역이 정확히 맞춰진다.

        Args:
            new_fs: 새 샘플링 레이트 (Hz). 양수여야 한다.
        """
        if new_fs <= 0 or abs(new_fs - self._fs) < 0.5:
            return  # 변화가 미미하면 재생성 비용 회피
        self._fs = new_fs
        self._filter = ButterworthFilter(BREATH_HZ_MIN, BREATH_HZ_MAX, new_fs)
        self._fft = FFTAnalyzer(new_fs)

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
        min_samples = int(self._fs * 4)
        clean = signal[~np.isnan(signal)]
        if len(clean) < min_samples:
            return None, None

        filtered = self._filter.apply(clean)
        peak_hz = self._fft.peak_frequency(filtered, BREATH_HZ_MIN, BREATH_HZ_MAX)

        if peak_hz is None:
            return None, filtered

        raw_bpm = peak_hz * 60.0
        smoothed = self._smoother.update(raw_bpm)
        return smoothed, filtered
