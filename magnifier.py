"""Step 7/8 - the zoom lens.

Geometry from hand_signals.py, driven by gesture_canvas_manipulation.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MIN_ZOOM = 1.2
MAX_ZOOM = 4.5
MIN_RADIUS_PX = 55.0
MAX_RADIUS_PX = 240.0

APERTURE_TIGHT = 0.25
APERTURE_WIDE = 1.10

RING_COLOR = (235, 235, 235)
GLINT_COLOR = (255, 255, 255)


# --- Zoom, inverse to the thumb/index circle -----------------------------
def zoom_for_aperture(aperture: float) -> float:
    t = (aperture - APERTURE_TIGHT) / max(APERTURE_WIDE - APERTURE_TIGHT, 1e-6)
    t = max(0.0, min(1.0, t))
    return MAX_ZOOM + (MIN_ZOOM - MAX_ZOOM) * t


@dataclass
class MagnifierState:
    center: tuple[float, float]
    radius: float
    zoom: float

    @property
    def clamped_radius(self) -> float:
        return max(MIN_RADIUS_PX, min(MAX_RADIUS_PX, self.radius))


# --- Rendering the disc --------------------------------------------------
def draw_magnifier(image: np.ndarray, state: MagnifierState) -> None:
    h, w = image.shape[:2]
    radius = int(round(state.clamped_radius))
    zoom = max(1.01, state.zoom)
    cx, cy = int(round(state.center[0])), int(round(state.center[1]))

    half_src = max(2, int(round(radius / zoom)))
    x0, y0 = cx - half_src, cy - half_src
    x1, y1 = cx + half_src, cy + half_src

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 - sx0 < 2 or sy1 - sy0 < 2:
        return
    patch = image[sy0:sy1, sx0:sx1]
    pad_left, pad_top = sx0 - x0, sy0 - y0
    pad_right, pad_bottom = x1 - sx1, y1 - sy1
    if pad_left or pad_top or pad_right or pad_bottom:
        patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)

    side = 2 * radius
    zoomed = cv2.resize(patch, (side, side), interpolation=cv2.INTER_LINEAR)

    dx0, dy0 = cx - radius, cy - radius
    vx0, vy0 = max(0, dx0), max(0, dy0)
    vx1, vy1 = min(w, dx0 + side), min(h, dy0 + side)
    if vx1 - vx0 < 2 or vy1 - vy0 < 2:
        return

    mask = np.zeros((side, side), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius - 2, 255, -1, cv2.LINE_AA)

    sub_zoom = zoomed[vy0 - dy0 : vy1 - dy0, vx0 - dx0 : vx1 - dx0]
    sub_mask = mask[vy0 - dy0 : vy1 - dy0, vx0 - dx0 : vx1 - dx0]
    target = image[vy0:vy1, vx0:vx1]
    alpha = (sub_mask.astype(np.float32) / 255.0)[..., None]
    image[vy0:vy1, vx0:vx1] = (sub_zoom * alpha + target * (1 - alpha)).astype(np.uint8)

    cv2.circle(image, (cx, cy), radius, (35, 35, 35), 6, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, RING_COLOR, 2, cv2.LINE_AA)
    glint_offset = int(radius * 0.55)
    cv2.ellipse(image, (cx - glint_offset, cy - glint_offset), (int(radius * 0.22), int(radius * 0.10)),
                -40, 0, 360, GLINT_COLOR, -1, cv2.LINE_AA)
    cv2.putText(image, f"x{zoom:.1f}", (cx - 18, cy + radius + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RING_COLOR, 2, cv2.LINE_AA)
