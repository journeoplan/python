import numpy as np
from src.sensors.base_sensor import SensorOutput


def test_sensor_output_defaults():
    """SensorOutput 기본값 검증."""
    out = SensorOutput(signal=np.array([0.5]))
    assert out.frame is None
    assert out.metadata == {}
    assert out.signal[0] == 0.5


def test_sensor_output_with_frame():
    """프레임 포함 SensorOutput."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = SensorOutput(
        signal=np.array([0.3]),
        frame=frame,
        metadata={"landmarks": "test"},
    )
    assert out.frame is not None
    assert out.frame.shape == (480, 640, 3)
    assert out.metadata["landmarks"] == "test"


def test_sensor_output_nan_signal():
    """NaN 신호 SensorOutput."""
    out = SensorOutput(signal=np.array([np.nan]))
    assert np.isnan(out.signal[0])
