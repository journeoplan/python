import numpy as np
import pytest
from src.config import FPS
from tests.conftest import make_sine


def test_passband_preserved(breath_filter):
    signal = make_sine(0.25)
    filtered = breath_filter.apply(signal)
    mid = slice(FPS * 3, -FPS * 3)
    ratio = np.std(filtered[mid]) / np.std(signal[mid])
    assert ratio > 0.7, f"Passband attenuated too much: {ratio:.2f}"


def test_stopband_attenuated(breath_filter):
    signal = make_sine(5.0)
    filtered = breath_filter.apply(signal)
    ratio = np.std(filtered) / np.std(signal)
    assert ratio < 0.1, f"Stopband not attenuated: {ratio:.2f}"


def test_short_signal_passthrough(breath_filter):
    short = np.ones(5)
    result = breath_filter.apply(short)
    assert result is not None


def test_dc_removal(breath_filter):
    """DC 오프셋이 제거되는지 검증."""
    signal = make_sine(0.25) + 100.0  # 큰 DC 오프셋
    filtered = breath_filter.apply(signal)
    assert abs(np.mean(filtered)) < 1.0, "DC not removed"


def test_nan_in_signal(breath_filter):
    """NaN 포함 신호에서 크래시 없는지 검증."""
    signal = make_sine(0.25)
    signal[10] = np.nan
    # filtfilt가 NaN을 전파하지만 크래시하지 않아야 함
    result = breath_filter.apply(signal)
    assert result is not None
