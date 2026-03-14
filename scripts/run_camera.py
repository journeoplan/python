"""카메라 기반 호흡 감지 실행 진입점."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sensors.camera_sensor import CameraSensor
from src.app import BreathMonitorApp


def main():
    parser = argparse.ArgumentParser(description="카메라 호흡 감지")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID")
    parser.add_argument("--face-mesh", action="store_true", help="FaceMesh 활성화")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로깅")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sensor = CameraSensor(camera_id=args.camera)
    app = BreathMonitorApp(sensor=sensor, enable_face_mesh=args.face_mesh)
    app.run()


if __name__ == "__main__":
    main()
