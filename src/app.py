"""카메라 기반 호흡 모니터링 애플리케이션.

아키텍처:
    Sensor ──▶ Detector ──▶ ApneaDetector ──▶ Renderer
                                              Recorder
"""

import csv
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from src.config import BUFFER_SIZE, FPS
from src.detectors.breath_detector import BreathDetector
from src.detectors.apnea_detector import ApneaDetector, BreathPattern
from src.sensors.base_sensor import BaseSensor, SensorOutput
from src.ui.overlay import (
    draw_bpm, draw_signal_graph, draw_progress, draw_rec, draw_fps, _put_text_kr,
)

logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "recordings"


class BreathMonitorApp:
    """호흡 모니터링 오케스트레이터.

    Sensor, Detector, UI, Recorder를 조합하여 실시간 루프를 실행한다.
    """

    def __init__(
        self,
        sensor: BaseSensor,
        enable_face_mesh: bool = False,
    ):
        self._sensor = sensor
        self._enable_face_mesh = enable_face_mesh

        # 감지기 (실제 FPS는 start 후 갱신)
        self._detector = BreathDetector(fs=FPS)
        self._apnea = ApneaDetector(fps=FPS)

        # 버퍼
        self._raw_buffer: deque[float] = deque(maxlen=BUFFER_SIZE)
        self._full_buffer: deque[float] = deque(maxlen=BUFFER_SIZE)
        self._filtered_signal: Optional[np.ndarray] = None

        # 녹화
        self._recording = False
        self._csv_file = None
        self._csv_writer = None

        # FaceMesh (선택적)
        self._face_mesh = None
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_pose = mp.solutions.pose
        self._mp_face_mesh = mp.solutions.face_mesh

    def run(self) -> None:
        """메인 루프 실행."""
        self._sensor.start()
        logger.info(
            "BreathMonitor 시작 | fps_target=%d | face_mesh=%s",
            FPS, self._enable_face_mesh,
        )

        if self._enable_face_mesh:
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        print("호흡 감지 시작 — 어깨가 화면에 보이도록 앉아주세요")
        print("종료: Q | 녹화: R | FaceMesh 토글: F")

        try:
            self._loop()
        finally:
            self._cleanup()

    def _loop(self) -> None:
        """프레임 처리 루프."""
        while True:
            output = self._sensor.read()

            if output.frame is None:
                continue

            frame = output.frame
            h, w = frame.shape[:2]
            val = output.signal[0]
            landmarks = output.metadata.get("landmarks")

            # FaceMesh 처리 (활성화 시)
            if self._face_mesh is not None:
                self._draw_face_mesh(frame)

            bpm: Optional[float] = None
            pattern = BreathPattern.UNKNOWN

            if not np.isnan(val):
                self._raw_buffer.append(val * h)
                self._full_buffer.append(val * h)

                # Pose 랜드마크 드로잉
                if landmarks:
                    self._mp_draw.draw_landmarks(
                        frame, landmarks, self._mp_pose.POSE_CONNECTIONS,
                        self._mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
                        self._mp_draw.DrawingSpec(color=(0, 180, 70), thickness=2),
                    )

                if len(self._full_buffer) >= FPS * 4:
                    bpm, self._filtered_signal = self._detector.process(
                        np.array(self._full_buffer)
                    )
            else:
                frame = _put_text_kr(frame, "포즈 감지 안됨", (20, 30), 24, (0, 80, 255))

            # 호흡 패턴 분류
            pattern = self._apnea.update(bpm)

            # CSV 녹화
            if self._recording and self._csv_writer is not None:
                self._write_csv_row(bpm, pattern)

            # UI 렌더링
            progress = len(self._full_buffer) / BUFFER_SIZE
            frame = draw_progress(frame, progress)
            frame = draw_signal_graph(frame, self._raw_buffer, self._filtered_signal)
            frame = draw_bpm(frame, bpm)
            frame = draw_fps(frame, self._sensor.fps)

            # 호흡 패턴 표시
            if pattern != BreathPattern.UNKNOWN:
                self._draw_pattern(frame, pattern)

            if self._recording:
                frame = draw_rec(frame)

            cv2.imshow("Breath Detection (Q to quit)", frame)

            # 윈도우 닫힘 감지
            if cv2.getWindowProperty("Breath Detection (Q to quit)", cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self._toggle_recording()
            elif key == ord("f"):
                self._toggle_face_mesh()

    def _draw_face_mesh(self, frame: np.ndarray) -> None:
        """FaceMesh를 프레임에 그린다."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = self._face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            face_spec = self._mp_draw.DrawingSpec(
                color=(0, 255, 200), thickness=1, circle_radius=1,
            )
            for face_lm in face_result.multi_face_landmarks:
                self._mp_draw.draw_landmarks(
                    frame, face_lm,
                    self._mp_face_mesh.FACEMESH_TESSELATION,
                    face_spec,
                    self._mp_draw.DrawingSpec(color=(0, 200, 150), thickness=1),
                )

    def _draw_pattern(self, frame: np.ndarray, pattern: BreathPattern) -> None:
        """호흡 패턴을 화면에 표시."""
        h, w = frame.shape[:2]
        colors = {
            BreathPattern.NORMAL: (0, 255, 100),
            BreathPattern.SHALLOW: (0, 200, 255),
            BreathPattern.RAPID: (0, 100, 255),
            BreathPattern.APNEA: (0, 0, 255),
        }
        color = colors.get(pattern, (150, 150, 150))
        frame = _put_text_kr(frame, f"상태: {pattern.value}", (20, 95), 16, color)

    def _write_csv_row(self, bpm: Optional[float], pattern: BreathPattern) -> None:
        """CSV에 한 행 기록."""
        raw_val = self._raw_buffer[-1] if self._raw_buffer else np.nan
        filt_val = (
            float(self._filtered_signal[-1])
            if self._filtered_signal is not None and len(self._filtered_signal) > 0
            else np.nan
        )
        self._csv_writer.writerow([
            f"{time.time():.6f}",
            f"{raw_val:.4f}",
            f"{filt_val:.4f}",
            f"{bpm:.2f}" if bpm is not None else "",
            pattern.value,
        ])

    def _toggle_recording(self) -> None:
        """녹화 시작/중지 토글."""
        if not self._recording:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = RECORDINGS_DIR / f"{ts}.csv"
            self._csv_file = open(filepath, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "timestamp", "raw_signal", "filtered_signal", "bpm", "pattern",
            ])
            self._recording = True
            logger.info("녹화 시작: %s", filepath)
            print(f"녹화 시작: {filepath}")
        else:
            self._recording = False
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
            logger.info("녹화 중지")
            print("녹화 중지")

    def _toggle_face_mesh(self) -> None:
        """FaceMesh 활성화/비활성화 토글."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
            self._enable_face_mesh = False
            print("FaceMesh OFF")
        else:
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._enable_face_mesh = True
            print("FaceMesh ON")

    def _cleanup(self) -> None:
        """리소스 해제."""
        if self._csv_file:
            self._csv_file.close()
        if self._face_mesh:
            self._face_mesh.close()
        self._sensor.stop()
        cv2.destroyAllWindows()
        logger.info("BreathMonitor 종료")
