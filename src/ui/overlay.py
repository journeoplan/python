import cv2
import numpy as np
from typing import Optional, Deque, Tuple

from src.config import FONT_PATH

# PIL 기반 한글 렌더링 (폰트 존재 시에만)
_USE_PIL = False
_font_cache: dict = {}

if FONT_PATH is not None:
    try:
        from PIL import ImageFont, ImageDraw, Image
        _USE_PIL = True
    except ImportError:
        pass


def _get_font(size: int):
    """폰트 캐시에서 가져오거나 새로 로드."""
    if size not in _font_cache and FONT_PATH is not None:
        from PIL import ImageFont
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache.get(size)


def _put_text_kr(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_size: int,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """한글 텍스트를 프레임에 렌더링. PIL 없으면 cv2 폴백."""
    if _USE_PIL:
        from PIL import ImageDraw, Image
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = _get_font(font_size)
        rgb_color = (color[2], color[1], color[0])
        draw.text(pos, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # cv2 폴백 (한글 깨짐 가능, 하지만 크래시 방지)
    scale = font_size / 24.0
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
    return frame


def draw_bpm(frame: np.ndarray, bpm: Optional[float]) -> np.ndarray:
    """BPM 수치를 프레임 좌측 상단에 오버레이."""
    cv2.rectangle(frame, (10, 10), (310, 90), (20, 20, 20), -1)

    if bpm is None:
        frame = _put_text_kr(frame, "데이터 수집 중...", (20, 30), 20, (150, 150, 150))
        return frame

    if bpm < 10:
        color, status = (0, 100, 255), "너무 느림"
    elif bpm < 20:
        color, status = (0, 255, 100), "정상 (평온)"
    elif bpm < 30:
        color, status = (0, 200, 255), "약간 빠름"
    else:
        color, status = (0, 80, 255), "빠른 호흡"

    frame = _put_text_kr(frame, "Breath Rate", (20, 14), 18, (180, 180, 180))
    cv2.putText(frame, f"{bpm:.1f} BPM", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    frame = _put_text_kr(frame, status, (215, 55), 18, color)
    return frame


def draw_signal_graph(
    frame: np.ndarray,
    raw_signal: "Deque[float]",
    filtered_signal: Optional[np.ndarray],
) -> np.ndarray:
    """하단에 신호 그래프 오버레이."""
    h, w = frame.shape[:2]
    graph_h = 100
    graph_y = h - graph_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, graph_y), (w - 10, h - 10), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def _plot_signal_line(sig, color, thickness=1):
        if len(sig) < 2:
            return
        arr = np.array(sig)[-300:]
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-6:
            return
        norm = (arr - mn) / (mx - mn)
        pts = [
            (int(10 + i * (w - 20) / len(norm)),
             int(graph_y + (1 - v) * graph_h))
            for i, v in enumerate(norm)
        ]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

    _plot_signal_line(raw_signal, (80, 200, 80), 1)
    if filtered_signal is not None and len(filtered_signal) > 0:
        _plot_signal_line(filtered_signal, (255, 140, 0), 2)

    return frame


def draw_rec(frame: np.ndarray) -> np.ndarray:
    """녹화 중 REC 표시."""
    cv2.circle(frame, (340, 30), 10, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (358, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


def draw_progress(frame: np.ndarray, ratio: float) -> np.ndarray:
    """데이터 수집 진행률 바."""
    h, w = frame.shape[:2]
    y = h - 118
    bar_w = int((w - 20) * min(ratio, 1.0))
    cv2.rectangle(frame, (10, y), (w - 10, y + 7), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, y), (10 + bar_w, y + 7), (0, 200, 100), -1)
    frame = _put_text_kr(frame, f"수집: {ratio * 100:.0f}%", (10, y - 20), 14, (130, 130, 130))
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """실시간 FPS를 우측 상단에 표시."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (w - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    return frame
