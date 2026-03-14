# Breath & Vital Sensing

WiFi CSI 및 카메라 기반 생체신호 감지 시스템.

## 빠른 시작

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/run_camera.py
```

## 현재 지원 센서
- 카메라 (MediaPipe Pose) — Phase 1 완료

## 테스트 실행
```bash
pytest tests/ -v
```
