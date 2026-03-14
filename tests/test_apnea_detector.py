import pytest
from src.detectors.apnea_detector import ApneaDetector, BreathPattern


def test_normal_breathing():
    """12~20 BPM → 정상."""
    det = ApneaDetector(fps=30.0)
    assert det.update(15.0) == BreathPattern.NORMAL
    assert det.update(12.0) == BreathPattern.NORMAL
    assert det.update(20.0) == BreathPattern.NORMAL


def test_shallow_breathing():
    """6~12 BPM → 얕은 호흡."""
    det = ApneaDetector(fps=30.0)
    assert det.update(8.0) == BreathPattern.SHALLOW
    assert det.update(6.0) == BreathPattern.SHALLOW


def test_rapid_breathing():
    """20+ BPM → 빠른 호흡."""
    det = ApneaDetector(fps=30.0)
    assert det.update(25.0) == BreathPattern.RAPID
    assert det.update(36.0) == BreathPattern.RAPID


def test_very_low_bpm_is_apnea():
    """6 BPM 미만 → 무호흡."""
    det = ApneaDetector(fps=30.0)
    assert det.update(3.0) == BreathPattern.APNEA


def test_no_signal_triggers_apnea():
    """신호 없음이 threshold 이상 지속 → 무호흡."""
    det = ApneaDetector(fps=30.0, apnea_threshold_sec=1.0)
    # 1초 = 30프레임
    for _ in range(29):
        result = det.update(None)
        assert result == BreathPattern.UNKNOWN

    # 30번째 (1초) → APNEA
    result = det.update(None)
    assert result == BreathPattern.APNEA


def test_signal_recovery_resets_counter():
    """신호 복구 시 무호흡 카운터 리셋."""
    det = ApneaDetector(fps=30.0, apnea_threshold_sec=1.0)
    for _ in range(20):
        det.update(None)

    # 신호 복구
    result = det.update(15.0)
    assert result == BreathPattern.NORMAL

    # 다시 None — 카운터 처음부터
    for _ in range(20):
        result = det.update(None)
        assert result == BreathPattern.UNKNOWN


def test_reset():
    """reset() 후 상태 초기화."""
    det = ApneaDetector(fps=30.0, apnea_threshold_sec=1.0)
    for _ in range(30):
        det.update(None)
    assert det.update(None) == BreathPattern.APNEA

    det.reset()
    assert det.update(None) == BreathPattern.UNKNOWN


def test_no_signal_duration():
    """no_signal_duration_sec 속성."""
    det = ApneaDetector(fps=10.0)
    for _ in range(20):
        det.update(None)
    assert abs(det.no_signal_duration_sec - 2.0) < 0.01

    det.update(15.0)
    assert det.no_signal_duration_sec == 0.0
