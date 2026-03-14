from abc import ABC, abstractmethod
import numpy as np


class BaseSensor(ABC):
    """모든 센서의 추상 베이스 클래스."""

    @abstractmethod
    def start(self) -> None:
        """센서 초기화 및 데이터 수집 시작."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """현재 프레임 신호 반환. shape: (n_channels,)"""

    @abstractmethod
    def stop(self) -> None:
        """센서 중지 및 리소스 해제."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """실제 샘플링 레이트 (Hz)."""
