from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SensorOutput:
    """센서 읽기 결과를 담는 공통 구조체.

    모든 센서(카메라, WiFi CSI 등)가 동일한 형식으로 데이터를 반환한다.

    Attributes:
        signal: 1D 신호 배열, shape (n_channels,).
        frame: 시각화용 프레임 (카메라: BGR ndarray, WiFi: None).
        metadata: 센서별 추가 데이터 (랜드마크, CSI 행렬 등).
    """

    signal: np.ndarray
    frame: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSensor(ABC):
    """모든 센서의 추상 베이스 클래스.

    센서 상태 흐름:
        CREATED ──start()──▶ RUNNING ──read()──▶ RUNNING (반복)
                                │
                            stop()
                                ▼
                            STOPPED
    """

    @abstractmethod
    def start(self) -> None:
        """센서 초기화 및 데이터 수집 시작."""

    @abstractmethod
    def read(self) -> SensorOutput:
        """현재 프레임 신호 반환.

        Returns:
            SensorOutput: signal, frame, metadata를 포함한 구조체.
        """

    @abstractmethod
    def stop(self) -> None:
        """센서 중지 및 리소스 해제."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """실제 샘플링 레이트 (Hz)."""
