# 전역 설정 — 이 파일에서만 상수를 수정할 것
from typing import Optional
import os as _os
import platform as _platform

FPS: int = 30

# 호흡 주파수 범위
BREATH_HZ_MIN: float = 0.1   # 6 BPM
BREATH_HZ_MAX: float = 0.6   # 36 BPM

# 심박 주파수 범위 (Phase 3)
HEART_HZ_MIN: float = 0.8    # 48 BPM
HEART_HZ_MAX: float = 2.5    # 150 BPM

# 신호 버퍼
WINDOW_SEC: int = 10
BUFFER_SIZE: int = FPS * WINDOW_SEC

# BPM 평활화
SMOOTHER_WINDOW: int = 30

# FFT 피크 감지
FFT_PEAK_MIN_MAGNITUDE: float = 1e-6  # 이 미만의 피크는 노이즈로 간주

# 폰트 경로 (OS별 폴백)
_FONT_PATHS: dict = {
    "Darwin": [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "Windows": [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
}


def get_font_path() -> Optional[str]:
    """현재 OS에 맞는 폰트 경로를 반환. 없으면 None."""
    system = _platform.system()
    for path in _FONT_PATHS.get(system, []):
        if _os.path.exists(path):
            return path
    return None


FONT_PATH: Optional[str] = get_font_path()

# MediaPipe 설정
MEDIAPIPE_MODEL_COMPLEXITY: int = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.6
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.6
