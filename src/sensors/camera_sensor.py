import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from typing import Optional

from src.sensors.base_sensor import BaseSensor, SensorOutput
from src.config import (
    FPS, BUFFER_SIZE,
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
)


class CameraSensor(BaseSensor):
    """MediaPipe Pose 기반 카메라 센서.

    양쪽 어깨의 Y좌표 평균을 호흡 신호로 반환한다.
    """

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._pose: Optional[mp.solutions.pose.Pose] = None
        self._running = False
        self._timestamps: deque = deque(maxlen=BUFFER_SIZE)
        self._mp_pose = mp.solutions.pose

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"카메라(ID={self._camera_id})를 열 수 없습니다. "
                "macOS: 시스템 설정 → 개인정보 보호 및 보안 → 카메라에서 터미널 접근을 허용하세요."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, FPS)
        self._pose = self._mp_pose.Pose(
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )
        self._running = True

    def read(self) -> SensorOutput:
        """프레임 읽기 및 어깨 Y좌표 반환.

        Returns:
            SensorOutput:
                signal: shape (1,) — 어깨 Y좌표 (픽셀 정규화 0~1), 감지 실패 시 [nan]
                frame: OpenCV BGR 프레임, 프레임 손실 시 None
                metadata: {"landmarks": pose_landmarks} 또는 빈 dict
        """
        if not self._running or self._cap is None:
            raise RuntimeError("Sensor not started.")

        ret, frame = self._cap.read()
        if not ret:
            return SensorOutput(signal=np.array([np.nan]))

        frame = cv2.flip(frame, 1)
        self._timestamps.append(time.time())

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            left_y = lm[self._mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_y = lm[self._mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            shoulder_y = (left_y + right_y) / 2.0
            return SensorOutput(
                signal=np.array([shoulder_y]),
                frame=frame,
                metadata={"landmarks": result.pose_landmarks},
            )

        return SensorOutput(signal=np.array([np.nan]), frame=frame)

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()
        if self._pose:
            self._pose.close()

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return float(FPS)
        return len(self._timestamps) / (self._timestamps[-1] - self._timestamps[0] + 1e-8)
