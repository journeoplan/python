import numpy as np
from typing import Optional, Tuple

from src.config import FFT_PEAK_MIN_MAGNITUDE


class FFTAnalyzer:
    """FFT 기반 주파수 분석기."""

    def __init__(self, fs: float):
        self._fs = fs

    def spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """FFT 스펙트럼 반환.

        Returns:
            (freqs, magnitudes) 튜플
        """
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / self._fs)
        magnitudes = np.abs(np.fft.rfft(signal))
        return freqs, magnitudes

    def peak_frequency(
        self, signal: np.ndarray, freq_min: float, freq_max: float
    ) -> Optional[float]:
        """지정 범위 내 피크 주파수 반환.

        Args:
            signal: 필터링된 신호
            freq_min: 탐색 하한 (Hz)
            freq_max: 탐색 상한 (Hz)

        Returns:
            피크 주파수 (Hz), 신호 부족 시 None
        """
        if len(signal) < self._fs * 4:
            return None

        freqs, mags = self.spectrum(signal)
        mask = (freqs >= freq_min) & (freqs <= freq_max)
        if not np.any(mask):
            return None

        masked_mags = mags[mask]
        masked_freqs = freqs[mask]

        # 피크 magnitude가 임계값 미만이면 유의미한 신호 없음
        if np.max(masked_mags) < FFT_PEAK_MIN_MAGNITUDE:
            return None

        k = int(np.argmax(masked_mags))

        # 파라볼라 보간: 인접 3개 빈으로 서브빈 정밀도 달성
        if 0 < k < len(masked_mags) - 1:
            alpha = masked_mags[k - 1]
            beta = masked_mags[k]
            gamma = masked_mags[k + 1]
            denom = alpha - 2.0 * beta + gamma
            if abs(denom) > 1e-12:
                delta = 0.5 * (alpha - gamma) / denom
                freq_res = masked_freqs[1] - masked_freqs[0]
                return float(masked_freqs[k] + delta * freq_res)

        return float(masked_freqs[k])
