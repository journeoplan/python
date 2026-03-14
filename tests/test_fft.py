import numpy as np
from src.config import FPS, BREATH_HZ_MIN, BREATH_HZ_MAX
from src.signal.fft_analyzer import FFTAnalyzer


def make_breath_signal(hz: float = 0.25, duration: int = 20) -> np.ndarray:
    t = np.linspace(0, duration, duration * FPS)
    return np.sin(2 * np.pi * hz * t)


def test_peak_detection_025hz():
    analyzer = FFTAnalyzer(FPS)
    signal = make_breath_signal(0.25)
    freq = analyzer.peak_frequency(signal, BREATH_HZ_MIN, BREATH_HZ_MAX)
    assert freq is not None
    assert abs(freq - 0.25) < 0.02


def test_returns_none_for_short_signal():
    analyzer = FFTAnalyzer(FPS)
    short = np.ones(FPS * 2)   # 2초 — 부족
    freq = analyzer.peak_frequency(short, BREATH_HZ_MIN, BREATH_HZ_MAX)
    assert freq is None


def test_spectrum_shape():
    analyzer = FFTAnalyzer(FPS)
    signal = make_breath_signal()
    freqs, mags = analyzer.spectrum(signal)
    assert freqs.shape == mags.shape
    assert freqs[0] == 0.0


def test_parabolic_interpolation_improves_accuracy():
    """파라볼라 보간이 FFT 빈 간격보다 높은 정밀도를 제공하는지 검증."""
    analyzer = FFTAnalyzer(FPS)
    # 0.17 Hz — FFT 빈 경계에 정확히 떨어지지 않는 주파수
    target_hz = 0.17
    signal = make_breath_signal(target_hz, duration=15)
    freq = analyzer.peak_frequency(signal, BREATH_HZ_MIN, BREATH_HZ_MAX)
    assert freq is not None
    # 파라볼라 보간 없이 단순 argmax면 빈 간격(~0.067Hz)까지만 정확
    # 보간 적용 시 0.01Hz 이내로 정확해야 함
    assert abs(freq - target_hz) < 0.01, f"Expected ~{target_hz}, got {freq}"


def test_parabolic_interpolation_off_bin_center():
    """빈 중심에서 벗어난 주파수도 정확히 감지하는지 검증."""
    analyzer = FFTAnalyzer(FPS)
    for target_hz in [0.13, 0.22, 0.33, 0.45]:
        signal = make_breath_signal(target_hz, duration=20)
        freq = analyzer.peak_frequency(signal, BREATH_HZ_MIN, BREATH_HZ_MAX)
        assert freq is not None
        assert abs(freq - target_hz) < 0.015, (
            f"target={target_hz}, got={freq}"
        )


def test_edge_bin_falls_back_to_argmax():
    """피크가 범위 경계에 있을 때 argmax 폴백이 동작하는지 검증."""
    analyzer = FFTAnalyzer(FPS)
    # BREATH_HZ_MIN(0.1)에 가까운 주파수
    signal = make_breath_signal(0.1, duration=20)
    freq = analyzer.peak_frequency(signal, BREATH_HZ_MIN, BREATH_HZ_MAX)
    assert freq is not None
    assert abs(freq - 0.1) < 0.05
