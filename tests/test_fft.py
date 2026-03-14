import numpy as np
from src.config import FPS, BREATH_HZ_MIN, BREATH_HZ_MAX
from tests.conftest import make_sine


def test_peak_detection_025hz(fft_analyzer):
    signal = make_sine(0.25)
    freq = fft_analyzer.peak_frequency(signal, BREATH_HZ_MIN, BREATH_HZ_MAX)
    assert freq is not None
    assert abs(freq - 0.25) < 0.02


def test_returns_none_for_short_signal(fft_analyzer):
    short = np.ones(FPS * 2)
    freq = fft_analyzer.peak_frequency(short, BREATH_HZ_MIN, BREATH_HZ_MAX)
    assert freq is None


def test_spectrum_shape(fft_analyzer):
    signal = make_sine(0.25)
    freqs, mags = fft_analyzer.spectrum(signal)
    assert freqs.shape == mags.shape
    assert freqs[0] == 0.0


def test_parabolic_interpolation_improves_accuracy(fft_analyzer):
    """파라볼라 보간이 FFT 빈 간격보다 높은 정밀도를 제공하는지 검증."""
    target_hz = 0.17
    signal = make_sine(target_hz, duration=15)
    freq = fft_analyzer.peak_frequency(signal, BREATH_HZ_MIN, BREATH_HZ_MAX)
    assert freq is not None
    assert abs(freq - target_hz) < 0.01, f"Expected ~{target_hz}, got {freq}"


def test_parabolic_interpolation_off_bin_center(fft_analyzer):
    """빈 중심에서 벗어난 주파수도 정확히 감지하는지 검증."""
    for target_hz in [0.13, 0.22, 0.33, 0.45]:
        signal = make_sine(target_hz, duration=20)
        freq = fft_analyzer.peak_frequency(signal, BREATH_HZ_MIN, BREATH_HZ_MAX)
        assert freq is not None
        assert abs(freq - target_hz) < 0.015, (
            f"target={target_hz}, got={freq}"
        )


def test_edge_bin_falls_back_to_argmax(fft_analyzer):
    """피크가 범위 경계에 있을 때 argmax 폴백이 동작하는지 검증."""
    signal = make_sine(0.1, duration=20)
    freq = fft_analyzer.peak_frequency(signal, BREATH_HZ_MIN, BREATH_HZ_MAX)
    assert freq is not None
    assert abs(freq - 0.1) < 0.05


def test_all_zero_signal_returns_none(fft_analyzer):
    """전부 0인 신호에서 None을 반환하는지 검증."""
    signal = np.zeros(FPS * 20)
    freq = fft_analyzer.peak_frequency(signal, BREATH_HZ_MIN, BREATH_HZ_MAX)
    assert freq is None, f"Expected None for all-zero signal, got {freq}"
