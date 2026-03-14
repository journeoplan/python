"""카메라 기반 호흡 감지 실행 진입점."""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

from src.sensors.camera_sensor import CameraSensor
from src.detectors.breath_detector import BreathDetector
from src.ui.overlay import draw_bpm, draw_signal_graph, draw_progress, draw_rec, _put_text_kr
from src.config import BUFFER_SIZE, FPS

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "recordings"


def main():
    sensor = CameraSensor(camera_id=0)
    detector = BreathDetector()
    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_spec = mp_draw.DrawingSpec(color=(0, 255, 200), thickness=1, circle_radius=1)
    face_conn_spec = mp_draw.DrawingSpec(color=(0, 200, 150), thickness=1)

    raw_buffer: deque = deque(maxlen=BUFFER_SIZE)
    full_buffer: deque = deque(maxlen=BUFFER_SIZE)
    filtered_signal = None

    # 녹화 상태
    recording = False
    csv_file = None
    csv_writer = None

    sensor.start()
    print("호흡 감지 시작 — 어깨가 화면에 보이도록 앉아주세요 | 종료: Q | 녹화: R")

    try:
        while True:
            signal, frame, landmarks = sensor.read()

            if frame is None:
                break

            h, w = frame.shape[:2]
            val = signal[0]

            # Face Mesh 처리
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_result = face_mesh.process(rgb)
            if face_result.multi_face_landmarks:
                for face_landmarks in face_result.multi_face_landmarks:
                    mp_draw.draw_landmarks(
                        frame, face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        face_spec, face_conn_spec,
                    )
                    mp_draw.draw_landmarks(
                        frame, face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        face_spec,
                        mp_draw.DrawingSpec(color=(0, 255, 100), thickness=1),
                    )
                    mp_draw.draw_landmarks(
                        frame, face_landmarks,
                        mp_face_mesh.FACEMESH_IRISES,
                        mp_draw.DrawingSpec(color=(255, 200, 0), thickness=1, circle_radius=1),
                        mp_draw.DrawingSpec(color=(255, 200, 0), thickness=1),
                    )

            if not np.isnan(val):
                raw_buffer.append(val * h)
                full_buffer.append(val * h)

                if landmarks:
                    mp_draw.draw_landmarks(
                        frame, landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(0, 180, 70), thickness=2),
                    )

                if len(full_buffer) >= FPS * 4:
                    bpm, filtered_signal = detector.process(np.array(full_buffer))
                else:
                    bpm = None
            else:
                bpm = None
                frame = _put_text_kr(frame, "포즈 감지 안됨", (20, 30), 24, (0, 80, 255))

            # CSV 녹화
            if recording and csv_writer is not None:
                raw_val = raw_buffer[-1] if raw_buffer else np.nan
                filt_val = (
                    float(filtered_signal[-1])
                    if filtered_signal is not None and len(filtered_signal) > 0
                    else np.nan
                )
                csv_writer.writerow([
                    f"{time.time():.6f}",
                    f"{raw_val:.4f}",
                    f"{filt_val:.4f}",
                    f"{bpm:.2f}" if bpm is not None else "",
                ])

            progress = len(full_buffer) / BUFFER_SIZE
            frame = draw_progress(frame, progress)
            frame = draw_signal_graph(frame, raw_buffer, filtered_signal)
            frame = draw_bpm(frame, bpm)

            if recording:
                frame = draw_rec(frame)

            cv2.imshow("Breath Detection — Camera (Q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                if not recording:
                    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = RECORDINGS_DIR / f"{ts}.csv"
                    csv_file = open(filepath, "w", newline="")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["timestamp", "raw_signal", "filtered_signal", "bpm"])
                    recording = True
                    print(f"녹화 시작: {filepath}")
                else:
                    recording = False
                    if csv_file:
                        csv_file.close()
                        csv_file = None
                        csv_writer = None
                    print("녹화 중지")

    finally:
        if csv_file:
            csv_file.close()
        face_mesh.close()
        sensor.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
