"""Step 1/8 - read the hand from MediaPipe's 21 landmarks.

Poses, orientation signals and the lens circle. Consumed by
gesture_canvas_manipulation.py.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


# --- Landmarks and scale -------------------------------------------------
def get_lm(hand_landmarks, idx: int):
    try:
        return hand_landmarks[idx]
    except (KeyError, IndexError, TypeError):
        return None


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_scale(hand_landmarks) -> float:
    wrist = get_lm(hand_landmarks, WRIST)
    mid = get_lm(hand_landmarks, MIDDLE_MCP)
    if wrist is None or mid is None:
        return 1.0
    return max(_dist(wrist, mid), 1e-6)


# --- Poses ---------------------------------------------------------------
FINGER_EXTENSION_MARGIN = 0.12


def _finger_extended(hand_landmarks, tip: int, pip: int) -> bool:
    wrist = get_lm(hand_landmarks, WRIST)
    tip_lm = get_lm(hand_landmarks, tip)
    pip_lm = get_lm(hand_landmarks, pip)
    if wrist is None or tip_lm is None or pip_lm is None:
        return False
    reach = _dist(tip_lm, wrist) - _dist(pip_lm, wrist)
    return reach > FINGER_EXTENSION_MARGIN * palm_scale(hand_landmarks)


def extended_finger_count(hand_landmarks) -> int:
    pairs = ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))
    return sum(1 for tip, pip in pairs if _finger_extended(hand_landmarks, tip, pip))


def is_pointing(hand_landmarks) -> bool:
    return (
        _finger_extended(hand_landmarks, INDEX_TIP, INDEX_PIP)
        and not _finger_extended(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not _finger_extended(hand_landmarks, RING_TIP, RING_PIP)
        and not _finger_extended(hand_landmarks, PINKY_TIP, PINKY_PIP)
    )


def is_pointing_relaxed(hand_landmarks) -> bool:
    return _finger_extended(hand_landmarks, INDEX_TIP, INDEX_PIP) and extended_finger_count(hand_landmarks) <= 2


def hand_pose_signal(hand_landmarks) -> str:
    n = extended_finger_count(hand_landmarks)
    if n <= 1:
        return "closed"
    if n >= 3:
        return "open"
    return "ambiguous"


def is_closed_fist(hand_landmarks) -> bool:
    return extended_finger_count(hand_landmarks) == 0


def is_open_palm(hand_landmarks) -> bool:
    return extended_finger_count(hand_landmarks) == 4


# --- Points on the hand --------------------------------------------------
def get_index_fingertip(hand_landmarks, frame_w: int, frame_h: int) -> tuple[float, float]:
    tip = hand_landmarks[INDEX_TIP]
    return tip.x * frame_w, tip.y * frame_h


def get_palm_center(hand_landmarks, frame_w: int, frame_h: int) -> tuple[float, float]:
    xs = [hand_landmarks[i].x for i in PALM_LANDMARKS]
    ys = [hand_landmarks[i].y for i in PALM_LANDMARKS]
    return (sum(xs) / len(xs)) * frame_w, (sum(ys) / len(ys)) * frame_h


def three_finger_tips(hand_landmarks, frame_w: int, frame_h: int) -> tuple[float, float] | None:
    tips = [get_lm(hand_landmarks, idx) for idx in (MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    tips = [t for t in tips if t is not None]
    if not tips:
        return None
    return (
        sum(t.x for t in tips) / len(tips) * frame_w,
        sum(t.y for t in tips) / len(tips) * frame_h,
    )


# --- Orientation signals -------------------------------------------------
def wrist_roll_angle(hand_landmarks) -> float:
    wrist = hand_landmarks[WRIST]
    mid = hand_landmarks[MIDDLE_MCP]
    return math.atan2(mid.y - wrist.y, mid.x - wrist.x)


def yaw_signal(hand_landmarks) -> float:
    return hand_landmarks[INDEX_MCP].z - hand_landmarks[PINKY_MCP].z


def pitch_signal(hand_landmarks) -> float:
    wrist = get_lm(hand_landmarks, WRIST)
    mid = get_lm(hand_landmarks, MIDDLE_MCP)
    if wrist is None or mid is None:
        return 0.0
    return getattr(mid, "z", 0.0) - getattr(wrist, "z", 0.0)


# --- Lens sign -----------------------------------------------------------
LENS_MAX_APERTURE = 1.25
LENS_MIN_APERTURE = 0.12


def thumb_index_aperture(hand_landmarks) -> float | None:
    thumb_tip = get_lm(hand_landmarks, THUMB_TIP)
    index_tip = get_lm(hand_landmarks, INDEX_TIP)
    if thumb_tip is None or index_tip is None:
        return None
    return max(_dist(thumb_tip, index_tip) / palm_scale(hand_landmarks), LENS_MIN_APERTURE)


def is_lens_sign(hand_landmarks) -> bool:
    if _finger_extended(hand_landmarks, INDEX_TIP, INDEX_PIP):
        return False
    if not (
        _finger_extended(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and _finger_extended(hand_landmarks, RING_TIP, RING_PIP)
        and _finger_extended(hand_landmarks, PINKY_TIP, PINKY_PIP)
    ):
        return False
    aperture = thumb_index_aperture(hand_landmarks)
    return aperture is not None and aperture <= LENS_MAX_APERTURE


def lens_geometry(hand_landmarks, frame_w: int, frame_h: int) -> tuple[tuple[float, float], float] | None:
    thumb_tip = get_lm(hand_landmarks, THUMB_TIP)
    index_tip = get_lm(hand_landmarks, INDEX_TIP)
    if thumb_tip is None or index_tip is None:
        return None
    tx, ty = thumb_tip.x * frame_w, thumb_tip.y * frame_h
    ix, iy = index_tip.x * frame_w, index_tip.y * frame_h
    center = ((tx + ix) / 2, (ty + iy) / 2)
    radius = math.hypot(ix - tx, iy - ty) / 2
    return center, radius


# --- Finger snap ---------------------------------------------------------
SNAP_CONTACT = 0.50
SNAP_RELEASE = 0.95
SNAP_MIN_TRAVEL = 0.45
SNAP_MIN_SPEED = 6.0
SNAP_WINDOW = 0.30
SNAP_COOLDOWN = 0.45
SNAP_MAX_EXTENDED = 2


@dataclass
class _HandSnapHistory:
    samples: deque = field(default_factory=lambda: deque(maxlen=14))
    last_fire: float = -1e9
    armed_until: float = -1e9


@dataclass
class SnapDetector:
    hands: dict[str, _HandSnapHistory] = field(default_factory=dict)

    def _distance(self, hand_landmarks) -> float | None:
        thumb_tip = get_lm(hand_landmarks, THUMB_TIP)
        middle_tip = get_lm(hand_landmarks, MIDDLE_TIP)
        if thumb_tip is None or middle_tip is None:
            return None
        return _dist(thumb_tip, middle_tip) / palm_scale(hand_landmarks)

    def update(self, hand_landmarks, handedness: str | None, now: float) -> bool:
        key = handedness or "?"
        hist = self.hands.setdefault(key, _HandSnapHistory())

        d = self._distance(hand_landmarks)
        if d is None:
            return False

        n_extended = extended_finger_count(hand_landmarks)
        hist.samples.append((now, d, n_extended))
        if d < SNAP_CONTACT and n_extended <= SNAP_MAX_EXTENDED:
            hist.armed_until = now + SNAP_WINDOW

        if d < SNAP_RELEASE or n_extended > SNAP_MAX_EXTENDED:
            return False
        if now - hist.last_fire < SNAP_COOLDOWN:
            return False

        for t_old, d_old, n_old in reversed(hist.samples):
            if now - t_old > SNAP_WINDOW:
                break
            if d_old >= SNAP_CONTACT or n_old > SNAP_MAX_EXTENDED:
                continue
            travel = d - d_old
            elapsed = now - t_old
            if elapsed <= 1e-6:
                continue
            if travel >= SNAP_MIN_TRAVEL and (travel / elapsed) >= SNAP_MIN_SPEED:
                hist.last_fire = now
                hist.samples.clear()
                hist.armed_until = -1e9
                return True
        return False

    def is_armed(self, handedness: str | None, now: float) -> bool:
        hist = self.hands.get(handedness or "?")
        return hist is not None and now <= hist.armed_until

    def reset(self) -> None:
        self.hands.clear()
