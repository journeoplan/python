import numpy as np
import pytest
from src.config import FPS, BREATH_HZ_MIN, BREATH_HZ_MAX
from src.signal.filter import ButterworthFilter


@pytest.fixture
def breath_filter():
    return ButterworthFilter(BREATH_HZ_MIN, BREATH_HZ_MAX, FPS)


def make_sine(hz: float, duration: int = 20) -> np.ndarray:
    t = np.linspace(0, duration, duration * FPS)
    return np.sin(2 * np.pi * hz * t)


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
