import numpy as np
import math
from src.signal.smoother import MedianSmoother


def test_basic_smoothing():
    """기본 평활화 동작."""
    s = MedianSmoother(window_size=5)
    results = [s.update(v) for v in [15.0, 16.0, 14.0, 15.0, 17.0]]
    # 윈도우가 차면 중앙값 = 15.0
    assert results[-1] == 15.0


def test_single_value():
    """단일 값 입력 시 그 값 반환."""
    s = MedianSmoother(window_size=5)
    assert s.update(12.0) == 12.0


def test_nan_ignored():
    """NaN 입력은 윈도우에 추가되지 않아야 함."""
    s = MedianSmoother(window_size=5)
    s.update(15.0)
    s.update(16.0)
    result = s.update(float("nan"))
    # NaN이 무시되므로 [15, 16]의 중앙값 = 15.5
    assert result == 15.5


def test_all_nan_returns_nan():
    """모든 입력이 NaN이면 NaN 반환."""
    s = MedianSmoother(window_size=5)
    result = s.update(float("nan"))
    assert math.isnan(result)


def test_nan_then_valid():
    """NaN 후 유효 값이 정상 동작하는지."""
    s = MedianSmoother(window_size=5)
    s.update(float("nan"))
    s.update(float("nan"))
    result = s.update(20.0)
    assert result == 20.0


def test_reset():
    """reset() 후 윈도우가 비어야 함."""
    s = MedianSmoother(window_size=5)
    s.update(15.0)
    s.update(16.0)
    s.reset()
    result = s.update(20.0)
    assert result == 20.0


def test_window_size_limit():
    """윈도우 크기 초과 시 오래된 값 제거."""
    s = MedianSmoother(window_size=3)
    s.update(10.0)
    s.update(20.0)
    s.update(30.0)
    result = s.update(40.0)
    # 윈도우 = [20, 30, 40], 중앙값 = 30
    assert result == 30.0
