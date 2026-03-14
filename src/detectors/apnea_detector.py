from collections import deque
from enum import Enum
from typing import Optional


class BreathPattern(Enum):
    """호흡 패턴 분류."""
    NORMAL = "정상"
    SHALLOW = "얕은 호흡"
    RAPID = "빠른 호흡"
    APNEA = "무호흡"
    UNKNOWN = "측정 중"


class ApneaDetector:
    """호흡 패턴 분류기.

    BPM과 감지 실패 지속 시간을 기반으로 호흡 상태를 분류한다.

    분류 규칙:
        - 정상: 12~20 BPM
        - 얕은 호흡: 6~12 BPM
        - 빠른 호흡: 20~36 BPM
        - 무호흡: BPM=None 상태가 apnea_threshold_sec 초 이상 지속
    """

    def __init__(
        self,
        fps: float = 30.0,
        apnea_threshold_sec: float = 15.0,
        history_size: int = 60,
    ):
        self._fps = fps
        self._apnea_threshold_sec = apnea_threshold_sec
        self._bpm_history: deque[Optional[float]] = deque(maxlen=history_size)
        self._no_signal_frames: int = 0

    def update(self, bpm: Optional[float]) -> BreathPattern:
        """BPM 값으로 호흡 패턴 갱신.

        Args:
            bpm: 현재 BPM. None이면 감지 실패.

        Returns:
            현재 호흡 패턴.
        """
        self._bpm_history.append(bpm)

        if bpm is None:
            self._no_signal_frames += 1
            no_signal_sec = self._no_signal_frames / self._fps
            if no_signal_sec >= self._apnea_threshold_sec:
                return BreathPattern.APNEA
            return BreathPattern.UNKNOWN

        self._no_signal_frames = 0

        if bpm < 6:
            return BreathPattern.APNEA
        elif bpm < 12:
            return BreathPattern.SHALLOW
        elif bpm <= 20:
            return BreathPattern.NORMAL
        else:
            return BreathPattern.RAPID

    def reset(self) -> None:
        """상태 초기화."""
        self._bpm_history.clear()
        self._no_signal_frames = 0

    @property
    def no_signal_duration_sec(self) -> float:
        """현재 신호 없음 지속 시간 (초)."""
        return self._no_signal_frames / self._fps
