"""공통 테스트 픽스처 및 유틸리티."""
import numpy as np
import pytest

from src.config import FPS, BREATH_HZ_MIN, BREATH_HZ_MAX
from src.signal.filter import ButterworthFilter
from src.signal.fft_analyzer import FFTAnalyzer


def make_sine(hz: float, duration: int = 20, fs: float = FPS) -> np.ndarray:
    """합성 사인파 생성.

    Args:
        hz: 주파수 (Hz)
        duration: 시간 (초)
        fs: 샘플링 레이트 (Hz)

    Returns:
        1D 사인파 배열
    """
    t = np.linspace(0, duration, int(duration * fs))
    return np.sin(2 * np.pi * hz * t)


def make_breath_signal(
    bpm: float = 15.0,
    duration: int = 20,
    fs: float = FPS,
    noise_level: float = 0.0,
) -> np.ndarray:
    """합성 호흡 신호 생성.

    Args:
        bpm: 호흡 횟수 (분당)
        duration: 시간 (초)
        fs: 샘플링 레이트
        noise_level: 가우시안 노이즈 표준편차

    Returns:
        1D 호흡 신호 배열
    """
    hz = bpm / 60.0
    signal = make_sine(hz, duration, fs)
    if noise_level > 0:
        rng = np.random.default_rng(42)
        signal = signal + noise_level * rng.standard_normal(len(signal))
    return signal


@pytest.fixture
def breath_filter():
    return ButterworthFilter(BREATH_HZ_MIN, BREATH_HZ_MAX, FPS)


@pytest.fixture
def fft_analyzer():
    return FFTAnalyzer(FPS)
