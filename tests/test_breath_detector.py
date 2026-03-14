import numpy as np
import pytest
from src.config import FPS
from src.detectors.breath_detector import BreathDetector
from tests.conftest import make_breath_signal


def test_normal_breathing():
    """15 BPM 합성 신호에서 BPM 추출."""
    detector = BreathDetector()
    signal = make_breath_signal(bpm=15.0, duration=20)
    bpm, filtered = detector.process(signal)
    assert bpm is not None
    assert abs(bpm - 15.0) < 2.0, f"Expected ~15, got {bpm}"
    assert filtered is not None


def test_slow_breathing():
    """8 BPM (느린 호흡) 감지."""
    detector = BreathDetector()
    signal = make_breath_signal(bpm=8.0, duration=30)
    bpm, filtered = detector.process(signal)
    assert bpm is not None
    assert abs(bpm - 8.0) < 2.0, f"Expected ~8, got {bpm}"


def test_fast_breathing():
    """30 BPM (빠른 호흡) 감지."""
    detector = BreathDetector()
    signal = make_breath_signal(bpm=30.0, duration=20)
    bpm, filtered = detector.process(signal)
    assert bpm is not None
    assert abs(bpm - 30.0) < 2.0, f"Expected ~30, got {bpm}"


def test_short_signal_returns_none():
    """4초 미만 신호에서 None 반환."""
    detector = BreathDetector()
    short = make_breath_signal(bpm=15.0, duration=2)
    bpm, filtered = detector.process(short)
    assert bpm is None
    assert filtered is None


def test_all_nan_signal():
    """전부 NaN 신호에서 None 반환."""
    detector = BreathDetector()
    signal = np.full(FPS * 10, np.nan)
    bpm, filtered = detector.process(signal)
    assert bpm is None
    assert filtered is None


def test_partial_nan_signal():
    """부분 NaN 포함 신호에서도 동작."""
    detector = BreathDetector()
    signal = make_breath_signal(bpm=15.0, duration=20)
    # 처음 10% NaN으로 채우기
    n_nan = len(signal) // 10
    signal[:n_nan] = np.nan
    bpm, filtered = detector.process(signal)
    assert bpm is not None
    assert abs(bpm - 15.0) < 3.0  # NaN 제거로 약간의 오차 허용


def test_noisy_signal():
    """노이즈 포함 신호에서도 BPM 추출."""
    detector = BreathDetector()
    signal = make_breath_signal(bpm=15.0, duration=20, noise_level=0.3)
    bpm, filtered = detector.process(signal)
    assert bpm is not None
    assert abs(bpm - 15.0) < 3.0, f"Expected ~15, got {bpm}"


def test_all_zero_signal():
    """전부 0인 신호에서 None 또는 유효 BPM 반환 (크래시 없음)."""
    detector = BreathDetector()
    signal = np.zeros(FPS * 10)
    bpm, filtered = detector.process(signal)
    # all-zero → FFT에서 None → bpm은 None이어야 함
    assert bpm is None


@pytest.mark.parametrize("target_bpm", [6, 10, 12, 15, 18, 24, 30, 36])
def test_accuracy_across_bpm_range(target_bpm):
    """±2 BPM 정확도 검증 — Phase 1 완료 조건."""
    detector = BreathDetector()
    signal = make_breath_signal(bpm=target_bpm, duration=30)
    bpm, _ = detector.process(signal)
    assert bpm is not None, f"Failed to detect {target_bpm} BPM"
    assert abs(bpm - target_bpm) < 2.0, (
        f"Accuracy fail: target={target_bpm}, got={bpm:.2f}"
    )
