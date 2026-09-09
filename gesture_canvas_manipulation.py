"""Application entry point: gestures, state machines, camera loop.

    camera -> MediaPipe -> hand_signals -> [here] -> canvas -> mesh3d -> hud

Gestures and keys are listed in the README.
Requires models/gesture_recognizer.task.
"""
from __future__ import annotations

import collections
import datetime
import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import async_recognition
import canvas as canvas_module
import filters
import hand_signals as hs
import hud
import magnifier
import sketch_recognition as sketch

WRIST = hs.WRIST
THUMB_TIP = hs.THUMB_TIP
INDEX_MCP, INDEX_PIP, INDEX_TIP = hs.INDEX_MCP, hs.INDEX_PIP, hs.INDEX_TIP
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = hs.MIDDLE_MCP, hs.MIDDLE_PIP, hs.MIDDLE_TIP
RING_MCP, RING_PIP, RING_TIP = hs.RING_MCP, hs.RING_PIP, hs.RING_TIP
PINKY_MCP, PINKY_PIP, PINKY_TIP = hs.PINKY_MCP, hs.PINKY_PIP, hs.PINKY_TIP
PALM_LANDMARKS = hs.PALM_LANDMARKS

is_pointing = hs.is_pointing
extended_finger_count = hs.extended_finger_count
hand_pose_signal = hs.hand_pose_signal
get_index_fingertip = hs.get_index_fingertip
get_palm_center = hs.get_palm_center
wrist_roll_angle = hs.wrist_roll_angle
yaw_signal = hs.yaw_signal
pitch_signal = hs.pitch_signal

OneEuroFilter = filters.OneEuroFilter
Point2DFilter = filters.Point2DFilter
clamp = filters.clamp
deadzone = filters.deadzone

Stroke = canvas_module.Stroke
DrawnObject = canvas_module.DrawnObject
Canvas = canvas_module.Canvas
COLOR_PALETTE = canvas_module.COLOR_PALETTE
KIND_LABELS = canvas_module.KIND_LABELS
build_object_mesh = canvas_module.build_object_mesh
_mesh_depth = canvas_module._mesh_depth

apply_shadow = hud.apply_shadow
draw_trail = hud.draw_trail
HELP_LINES = hud.HELP_LINES


# --- Tunables ------------------------------------------------------------
YAW_SENSITIVITY = 20.0
PITCH_SENSITIVITY = 16.0
DEPTH_DEADZONE = 0.006
MAX_YAW = math.radians(720)
MAX_PITCH = math.radians(720)
MIN_OBJECT_SCALE = 0.2
MAX_OBJECT_SCALE = 5.0

AUTO_VALIDATE_DELAY = 0.6
NEW_GROUP_DISTANCE = 260.0

DRAW_ONSET_FRAMES = 4
DRAW_GRACE_FRAMES = 3
FAST_DRAW_PX = 9.0
MAX_DRAW_JUMP_PX = 120.0

GRAB_ONSET_FRAMES = 3
GRAB_RELEASE_GRACE = 20
GRAB_LOST_GRACE = 6
OPEN_HAND_FRAMES = 3
GRAB_REACH = 90.0
FIST_CONFIDENCE = 0.55

DESELECT_DISTANCE = 200.0
DESELECT_DELAY = 0.35

SPIN_MIN = 1.5
SPIN_MAX = 14.0
SPIN_DAMPING = 1.9

STACK_MARGIN = 12.0

TRAIL_POINTS = 14
TRAIL_MAX_AGE = 0.35

GESTURE_MIN_PRESS = 0.12
GESTURE_GAP_GRACE = 0.16
UNDO_MIN_PRESS = 0.20
UNDO_CONFIDENCE = 0.65

CLEAR_ARM_SECONDS = 0.6
CLEAR_CONFIRM_SECONDS = 0.5
CLEAR_ARM_GRACE = 0.6

LENS_HOLD_FRAMES = 2
RESIZE_DOUBLING_PX = 180.0

CAPTURE_DIR = "captures"
WINDOW_TITLE = "Main Pinceau"

CAMERA_RESOLUTIONS = ((1088, 612), (1024, 576), (960, 540), (640, 480))
@dataclass
# --- Interaction sessions ------------------------------------------------
class GrabState:
    object_id: str
    handedness: str | None
    mode: str = "move"
    origin_signal_angle: float = 0.0
    origin_signal_depth: float = 0.0
    origin_signal_pitch: float = 0.0
    origin_signal_pos: tuple[float, float] = (0.0, 0.0)
    origin_object_roll: float = 0.0
    origin_object_yaw: float = 0.0
    origin_object_pitch: float = 0.0
    origin_object_offset: tuple[float, float] = (0.0, 0.0)
    origin_children: dict[str, tuple[float, float]] = field(default_factory=dict)
    prev_angle: float | None = None
    prev_time: float = 0.0
    spin_ema: float = 0.0
    roll_filter: OneEuroFilter | None = None
    yaw_filter: OneEuroFilter | None = None
    pitch_filter: OneEuroFilter | None = None
    pos_filter: Point2DFilter | None = None


def _anchor_grab(grab: GrabState, obj, canvas: Canvas, now: float, angle: float, depth: float,
                 pitch: float, pos: tuple[float, float], mode: str) -> None:
    grab.mode = mode
    grab.origin_signal_angle = angle
    grab.origin_signal_depth = depth
    grab.origin_signal_pitch = pitch
    grab.origin_signal_pos = pos
    grab.origin_object_roll = obj.roll
    grab.origin_object_yaw = obj.yaw
    grab.origin_object_pitch = obj.pitch
    grab.origin_object_offset = (obj.offset[0], obj.offset[1])
    grab.origin_children = {c.id: (c.offset[0], c.offset[1]) for c in canvas.descendants(obj.id)}
    grab.prev_angle = angle
    grab.prev_time = now
    grab.spin_ema = 0.0
    grab.roll_filter = OneEuroFilter(now, angle, min_cutoff=0.8, beta=0.3)
    grab.yaw_filter = OneEuroFilter(now, depth, min_cutoff=0.5, beta=0.2)
    grab.pitch_filter = OneEuroFilter(now, pitch, min_cutoff=0.5, beta=0.2)
    grab.pos_filter = Point2DFilter(now, pos[0], pos[1], min_cutoff=1.4, beta=0.8)


@dataclass
class LensSession:
    origin_aperture: float
    origin_scale: float
    origin_tips_y: float
    center_filter: Point2DFilter
    radius_filter: OneEuroFilter
    aperture_filter: OneEuroFilter
    tips_filter: OneEuroFilter


@dataclass
# --- Discrete gestures: one fire per press -------------------------------
class GestureLatch:
    min_press: float = GESTURE_MIN_PRESS
    gap_grace: float = GESTURE_GAP_GRACE
    since: float | None = None
    last_seen: float = -1e9
    fired: bool = False

    def update(self, active: bool, now: float) -> bool:
        if active:
            if self.since is None or (now - self.last_seen) > self.gap_grace:
                self.since = now
                self.fired = False
            self.last_seen = now
            if not self.fired and (now - self.since) >= self.min_press:
                self.fired = True
                return True
            return False
        if self.since is not None and (now - self.last_seen) > self.gap_grace:
            self.reset()
        return False

    def reset(self) -> None:
        self.since = None
        self.fired = False


@dataclass
# --- Clear everything: two hands, two stages -----------------------------
class ClearAllGesture:
    two_hands_since: float | None = None
    two_hands_last_seen: float = -1e9
    armed: bool = False
    gone_since: float | None = None
    fired: bool = False

    def reset(self) -> None:
        self.two_hands_since = None
        self.armed = False
        self.gone_since = None
        self.fired = False

    def update(self, hand_count: int, both_open: bool, now: float) -> tuple[str, float, bool]:
        if hand_count >= 2 and both_open:
            self.gone_since = None
            if self.two_hands_since is None:
                self.two_hands_since = now
                self.fired = False
            self.two_hands_last_seen = now
            held = now - self.two_hands_since
            if held >= CLEAR_ARM_SECONDS:
                self.armed = True
                return "ready", 1.0, False
            return "arm", held / CLEAR_ARM_SECONDS, False

        if hand_count == 0:
            if not self.armed:
                self.two_hands_since = None
                return "idle", 0.0, False
            if self.gone_since is None:
                self.gone_since = now
            waited = now - self.gone_since
            if waited < CLEAR_CONFIRM_SECONDS:
                return "confirm", waited / CLEAR_CONFIRM_SECONDS, False
            self.reset()
            return "idle", 0.0, True

        self.gone_since = None
        if (now - self.two_hands_last_seen) > CLEAR_ARM_GRACE:
            self.armed = False
            self.two_hands_since = None
            return "idle", 0.0, False
        if self.armed:
            return "ready", 1.0, False
        held = 0.0 if self.two_hands_since is None else now - self.two_hands_since
        return ("arm", clamp(held / CLEAR_ARM_SECONDS, 0.0, 1.0), False) if held > 0 else ("idle", 0.0, False)


@dataclass
# --- MediaPipe result -> one observation per hand ------------------------
class HandObservation:
    landmarks: object
    gesture_name: str | None
    confidence: float
    handedness: str | None

    def is_gesture(self, name: str, min_confidence: float = 0.5) -> bool:
        return self.gesture_name == name and self.confidence >= min_confidence


def top_gesture(gestures_for_hand) -> tuple[str | None, float]:
    if not gestures_for_hand:
        return None, 0.0
    top = gestures_for_hand[0]
    return top.category_name, top.score


def observe_hands(result) -> list[HandObservation]:
    n = len(result.hand_landmarks) if result.hand_landmarks else 0
    out = []
    for i in range(n):
        gesture_name, confidence = (
            top_gesture(result.gestures[i]) if result.gestures and i < len(result.gestures) else (None, 0.0)
        )
        handedness = None
        hlist = getattr(result, "handedness", None)
        if hlist and i < len(hlist) and hlist[i]:
            handedness = hlist[i][0].category_name
        out.append(HandObservation(result.hand_landmarks[i], gesture_name, confidence, handedness))
    return out


def pick_primary(state, hands: list[HandObservation], w: int, h: int) -> HandObservation | None:
    if not hands:
        return None
    if len(hands) == 1:
        return hands[0]
    if state.grab is not None and state.grab.handedness is not None:
        same = next((hd for hd in hands if hd.handedness == state.grab.handedness), None)
        if same is not None:
            return same
    if state.last_palm is not None:
        return min(
            hands,
            key=lambda hd: math.dist(hs.get_palm_center(hd.landmarks, w, h), state.last_palm),
        )
    return hands[0]


@dataclass
# --- Application state, owned by the main loop ---------------------------
class AppState:
    canvas: Canvas
    recognizer: async_recognition.RecognitionService | None = None
    point_filter: Point2DFilter | None = None
    grab: GrabState | None = None
    lens: LensSession | None = None
    was_drawing: bool = False
    last_status: str = "IDLE"
    fist_run: int = 0
    point_run: int = 0
    away_since: float | None = None
    open_run: int = 0
    lens_run: int = 0
    miss_run: int = 0
    lost_run: int = 0
    selected_id: str | None = None
    last_touched_id: str | None = None
    pending_since: float | None = None
    last_palm: tuple[float, float] | None = None
    last_index_tip: tuple[float, float] | None = None
    pen_color_idx: int = 0
    draw_miss_run: int = 0
    last_frame_time: float | None = None
    cursor_trail: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=TRAIL_POINTS)
    )
    status_set_this_frame: bool = False
    snap: hs.SnapDetector = field(default_factory=hs.SnapDetector)
    delete_latch: GestureLatch = field(default_factory=GestureLatch)
    color_latch: GestureLatch = field(default_factory=GestureLatch)
    undo_latch: GestureLatch = field(default_factory=lambda: GestureLatch(min_press=UNDO_MIN_PRESS))
    clear_gesture: ClearAllGesture = field(default_factory=ClearAllGesture)

    def set_status(self, message: str) -> None:
        self.last_status = message
        self.status_set_this_frame = True

    @property
    def pen_color(self) -> tuple[int, int, int]:
        return COLOR_PALETTE[self.pen_color_idx % len(COLOR_PALETTE)]

    def next_pen_color(self) -> None:
        self.pen_color_idx = (self.pen_color_idx + 1) % len(COLOR_PALETTE)

    def selected_object(self) -> DrawnObject | None:
        return self.canvas.get_object(self.selected_id)

    def prune_selection(self) -> None:
        self.canvas.detach_orphans()
        alive = {o.id for o in self.canvas.objects}
        if self.selected_id is not None and self.selected_id not in alive:
            self.selected_id = None
        if self.last_touched_id is not None and self.last_touched_id not in alive:
            self.last_touched_id = None

    def deselect(self) -> bool:
        had = self.selected_id is not None
        self.selected_id = None
        return had


@dataclass
class FrameResult:
    triggered: str | None = None
    hovered_object_id: str | None = None
    drawing: bool = False
    cursor: tuple[float, float] | None = None
    lens: magnifier.MagnifierState | None = None
    resizing: bool = False
    snapped: bool = False
    snap_armed: bool = False
    auto_created: bool = False
    color_changed: bool = False
    undone: str | None = None
    recognized: list = field(default_factory=list)
    clear_progress: float = 0.0
    clear_phase: str = "idle"
    trail: list = field(default_factory=list)


# --- Per-frame steps -----------------------------------------------------
def _selected_within_reach(state: AppState, palm: tuple[float, float]) -> DrawnObject | None:
    obj = state.selected_object()
    if obj is None:
        return None
    return obj if obj.contains_point(palm[0], palm[1], margin=int(GRAB_REACH)) else None


def _distance_to_object(obj: DrawnObject, px: float, py: float) -> float:
    x0, y0, x1, y1 = obj.transformed_bbox()
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def _update_auto_deselect(state: AppState, palm: tuple[float, float] | None, now: float,
                          busy: bool) -> bool:
    if state.selected_id is None or busy:
        state.away_since = None
        return False

    obj = state.selected_object()
    if obj is None:
        state.away_since = None
        return False

    away = palm is None or _distance_to_object(obj, palm[0], palm[1]) > DESELECT_DISTANCE
    if not away:
        state.away_since = None
        return False

    if state.away_since is None:
        state.away_since = now
        return False
    if (now - state.away_since) < DESELECT_DELAY:
        return False

    state.away_since = None
    state.deselect()
    return True


def _update_lens(state: AppState, hand: HandObservation, w: int, h: int, now: float, fr: FrameResult) -> None:
    geom = hs.lens_geometry(hand.landmarks, w, h)
    aperture = hs.thumb_index_aperture(hand.landmarks)
    tips = hs.three_finger_tips(hand.landmarks, w, h)
    if geom is None or aperture is None or tips is None:
        return
    center, radius = geom
    tips_y = tips[1]

    if state.lens is None:
        obj = state.selected_object()
        state.lens = LensSession(
            origin_aperture=aperture,
            origin_scale=obj.scale if obj is not None else 1.0,
            origin_tips_y=tips_y,
            center_filter=Point2DFilter(now, center[0], center[1], min_cutoff=1.2, beta=0.6),
            radius_filter=OneEuroFilter(now, radius, min_cutoff=1.0, beta=0.4),
            aperture_filter=OneEuroFilter(now, aperture, min_cutoff=1.0, beta=0.4),
            tips_filter=OneEuroFilter(now, tips_y, min_cutoff=1.0, beta=0.5),
        )

    smooth_cx, smooth_cy = state.lens.center_filter.update(now, center[0], center[1])
    smooth_radius = state.lens.radius_filter(now, radius)
    smooth_aperture = state.lens.aperture_filter(now, aperture)
    smooth_tips_y = state.lens.tips_filter(now, tips_y)

    obj = state.selected_object()
    if obj is not None:
        rise = state.lens.origin_tips_y - smooth_tips_y
        obj.scale = clamp(
            state.lens.origin_scale * (2.0 ** (rise / RESIZE_DOUBLING_PX)),
            MIN_OBJECT_SCALE, MAX_OBJECT_SCALE,
        )
        state.last_touched_id = obj.id
        fr.resizing = True
        arrow = "^" if rise > 2 else ("v" if rise < -2 else "-")
        state.set_status(f"TAILLE {arrow} ({obj.label}, x{obj.scale:.2f})")
    else:
        fr.lens = magnifier.MagnifierState(
            center=(smooth_cx, smooth_cy),
            radius=smooth_radius,
            zoom=magnifier.zoom_for_aperture(smooth_aperture),
        )
        state.set_status(f"LOUPE (x{fr.lens.zoom:.1f})")


def _update_manipulation(state: AppState, hand: HandObservation, palm: tuple[float, float],
                         mode: str, now: float, fr: FrameResult) -> None:
    obj = state.canvas.get_object(state.grab.object_id)
    if obj is None:
        state.grab = None
        return

    angle = hs.wrist_roll_angle(hand.landmarks)
    depth = hs.yaw_signal(hand.landmarks)
    pitch = hs.pitch_signal(hand.landmarks)

    if state.grab.mode != mode:
        _anchor_grab(state.grab, obj, state.canvas, now, angle, depth, pitch, palm, mode)

    smooth_angle = state.grab.roll_filter(now, angle)
    obj.roll = state.grab.origin_object_roll + filters.angle_delta(smooth_angle, state.grab.origin_signal_angle)

    if state.grab.prev_angle is not None:
        dt = now - state.grab.prev_time
        if dt > 1e-4:
            speed = filters.angle_delta(smooth_angle, state.grab.prev_angle) / dt
            state.grab.spin_ema = 0.6 * state.grab.spin_ema + 0.4 * speed
            obj.spin = (
                clamp(state.grab.spin_ema, -SPIN_MAX, SPIN_MAX)
                if abs(state.grab.spin_ema) >= SPIN_MIN else 0.0
            )
    state.grab.prev_angle = smooth_angle
    state.grab.prev_time = now

    if mode == "move":
        smooth_x, smooth_y = state.grab.pos_filter.update(now, palm[0], palm[1])
        dx = smooth_x - state.grab.origin_signal_pos[0]
        dy = smooth_y - state.grab.origin_signal_pos[1]
        obj.offset[0] = state.grab.origin_object_offset[0] + dx
        obj.offset[1] = state.grab.origin_object_offset[1] + dy
        for child in state.canvas.descendants(obj.id):
            origin = state.grab.origin_children.get(child.id)
            if origin is not None:
                child.offset[0] = origin[0] + dx
                child.offset[1] = origin[1] + dy
        stack = len(state.grab.origin_children)
        state.set_status(f"DEPLACE ({obj.label})" + (f" + {stack} dessus" if stack else ""))
    else:
        smooth_depth = state.grab.yaw_filter(now, depth)
        smooth_pitch = state.grab.pitch_filter(now, pitch)
        obj.yaw = clamp(
            state.grab.origin_object_yaw
            + deadzone(smooth_depth - state.grab.origin_signal_depth, DEPTH_DEADZONE) * YAW_SENSITIVITY,
            -MAX_YAW, MAX_YAW,
        )
        obj.pitch = clamp(
            state.grab.origin_object_pitch
            + deadzone(smooth_pitch - state.grab.origin_signal_pitch, DEPTH_DEADZONE) * PITCH_SENSITIVITY,
            -MAX_PITCH, MAX_PITCH,
        )
        state.set_status(f"ROTATION 3D ({obj.label})")

    state.last_touched_id = obj.id


def _release_grab(state: AppState) -> None:
    if state.grab is None:
        return
    obj = state.canvas.get_object(state.grab.object_id)
    state.grab = None
    if obj is None:
        return

    cx, cy = obj.center_screen()
    descendants = {o.id for o in state.canvas.descendants(obj.id)}
    best, best_area = None, math.inf
    for other in state.canvas.objects:
        if other.id == obj.id or other.id in descendants:
            continue
        x0, y0, x1, y1 = other.transformed_bbox()
        if not (x0 - STACK_MARGIN <= cx <= x1 + STACK_MARGIN and y0 - STACK_MARGIN <= cy <= y1 + STACK_MARGIN):
            continue
        area = max(x1 - x0, 1) * max(y1 - y0, 1)
        if area < best_area:
            best, best_area = other, area
    obj.attached_to = best.id if best is not None else None
    if best is not None:
        state.set_status(f"POSE SUR ({best.label})")


def _update_spin(state: AppState, dt: float) -> None:
    if dt <= 0:
        return
    held = state.grab.object_id if state.grab is not None else None
    damping = math.exp(-SPIN_DAMPING * dt)
    for obj in state.canvas.objects:
        if obj.spin == 0.0 or obj.id == held:
            continue
        obj.roll += obj.spin * dt
        obj.spin *= damping
        if abs(obj.spin) < 0.15:
            obj.spin = 0.0


def _finish_stroke(state: AppState, now: float) -> None:
    state.canvas.end_stroke()
    state.pending_since = now
    state.was_drawing = False
    state.draw_miss_run = 0
    state.point_filter = None
    state.cursor_trail.clear()


def _auto_create_element(state: AppState, now: float, fr: FrameResult) -> None:
    if fr.drawing or state.canvas.is_drawing or not state.canvas.pending_strokes:
        return
    if state.pending_since is None or (now - state.pending_since) < AUTO_VALIDATE_DELAY:
        return
    obj = state.canvas.validate(created_at=now, color_idx=state.pen_color_idx, recognizer=state.recognizer)
    state.pending_since = None
    if obj is not None:
        state.last_touched_id = obj.id
        state.set_status(f"ELEMENT CREE ({obj.label})")
        fr.auto_created = True


def _collect_recognitions(state: AppState, fr: FrameResult) -> None:
    if state.recognizer is None:
        return
    for result in state.recognizer.poll():
        obj = state.canvas.apply_recognition(result)
        if obj is not None and result.text:
            fr.recognized.append(result.text)
            state.set_status(f"ECRITURE ({result.text})")


def _update_clear_all(state: AppState, hands: list[HandObservation], now: float, fr: FrameResult) -> None:
    both_open = len(hands) >= 2 and all(hs.is_open_palm(hd.landmarks) for hd in hands[:2])
    phase, progress, fired = state.clear_gesture.update(len(hands), both_open, now)
    fr.clear_phase = phase
    fr.clear_progress = progress
    if fired:
        fr.triggered = "Clear_All"


def _update_discrete_gestures(state: AppState, hand: HandObservation | None, now: float,
                              fr: FrameResult) -> None:
    enabled = state.grab is None and hand is not None

    if state.delete_latch.update(enabled and hand.is_gesture("Thumb_Down"), now):
        fr.triggered = "Thumb_Down"
    if state.color_latch.update(enabled and hand.is_gesture("Thumb_Up"), now):
        fr.color_changed = True
    if state.undo_latch.update(enabled and hand.is_gesture("Victory", UNDO_CONFIDENCE), now):
        fr.undone = state.canvas.undo()


def _apply_color_change(state: AppState) -> None:
    state.next_pen_color()
    target = state.selected_object()
    if target is None and not state.canvas.pending_strokes and not state.canvas.is_drawing:
        target = state.canvas.get_object(state.last_touched_id)
    if target is not None:
        target.color_idx = state.pen_color_idx
        state.set_status(f"COULEUR ({target.label})")
    else:
        state.set_status("COULEUR DU TRAIT")


def _apply_triggers(state: AppState, fr: FrameResult) -> None:
    if fr.triggered == "Clear_All":
        state.canvas.clear()
        state.grab = None
        state.lens = None
        state.selected_id = None
        state.last_touched_id = None
        state.pending_since = None
        state.was_drawing = False
        state.cursor_trail.clear()
        state.set_status("TOUT EFFACE (z pour restaurer)")
    elif fr.triggered == "Thumb_Down":
        removed = state.canvas.delete_object(state.last_touched_id) if state.last_touched_id else None
        if removed is None:
            removed = state.canvas.delete_last_object()
        state.prune_selection()
        state.set_status(f"SUPPRIME ({removed.label})" if removed is not None else "RIEN A SUPPRIMER")

    if fr.undone is not None:
        state.prune_selection()
        state.set_status({"clear": "TOUT RESTAURE", "object": "ANNULE (element)"}.get(fr.undone, "ANNULE (trait)"))


def _update_drawing(state: AppState, raw_x: float, raw_y: float, tip_travel: float,
                    w: int, h: int, now: float, fr: FrameResult) -> None:
    if state.canvas.is_drawing and tip_travel > MAX_DRAW_JUMP_PX:
        _finish_stroke(state, now)
        state.set_status("SAUT DE SUIVI (trait ferme)")
        return

    fr.drawing = True
    if state.point_filter is None or not state.canvas.is_drawing:
        if state.canvas.pending_strokes and state.canvas.distance_to_pending(raw_x, raw_y) > NEW_GROUP_DISTANCE:
            created = state.canvas.validate(
                created_at=now, color_idx=state.pen_color_idx, recognizer=state.recognizer
            )
            if created is not None:
                state.last_touched_id = created.id
                fr.auto_created = True
            state.pending_since = None
        state.point_filter = Point2DFilter(now, raw_x, raw_y)
        smooth_x, smooth_y = raw_x, raw_y
        state.canvas.start_stroke(now)
    else:
        smooth_x, smooth_y = state.point_filter.update(now, raw_x, raw_y)

    smooth_x = clamp(smooth_x, 0.0, w - 1.0)
    smooth_y = clamp(smooth_y, 0.0, h - 1.0)
    state.canvas.add_point(int(smooth_x), int(smooth_y))
    state.cursor_trail.append((smooth_x, smooth_y, now))
    state.set_status("DESSIN")


# --- The frame pipeline: one call per camera frame -----------------------
def process_frame(state: AppState, result, w: int, h: int, now: float) -> FrameResult:
    fr = FrameResult()
    state.status_set_this_frame = False
    state.prune_selection()
    _collect_recognitions(state, fr)

    dt = 0.0 if state.last_frame_time is None else clamp(now - state.last_frame_time, 0.0, 0.1)
    state.last_frame_time = now

    hands = observe_hands(result)
    primary = pick_primary(state, hands, w, h)
    _update_clear_all(state, hands, now, fr)
    palm_now: tuple[float, float] | None = None

    if primary is None:
        state.lost_run += 1
        if state.grab is not None and state.lost_run > GRAB_LOST_GRACE:
            _release_grab(state)
        state.fist_run = state.open_run = state.lens_run = state.point_run = 0
        state.lens = None
        _update_discrete_gestures(state, None, now, fr)
    else:
        state.lost_run = 0
        lm = primary.landmarks
        raw_x, raw_y = hs.get_index_fingertip(lm, w, h)
        palm = hs.get_palm_center(lm, w, h)
        palm_now = palm
        state.last_palm = palm
        fr.cursor = (raw_x, raw_y)

        ml_command = (
            primary.is_gesture("Thumb_Down")
            or primary.is_gesture("Thumb_Up")
            or primary.is_gesture("Victory", UNDO_CONFIDENCE)
        )
        lens_now = hs.is_lens_sign(lm)
        open_now = hs.is_open_palm(lm)
        fist_now = (not ml_command) and (
            hs.is_closed_fist(lm)
            or (primary.gesture_name == "Closed_Fist" and primary.confidence >= FIST_CONFIDENCE)
        )
        point_now = (not ml_command) and hs.is_pointing(lm)
        state.lens_run = state.lens_run + 1 if lens_now else 0
        state.open_run = state.open_run + 1 if open_now else 0
        state.fist_run = state.fist_run + 1 if fist_now else 0
        state.point_run = state.point_run + 1 if point_now else 0
        pose = hs.hand_pose_signal(lm)

        tip_travel = (
            math.dist((raw_x, raw_y), state.last_index_tip)
            if state.last_index_tip is not None else 0.0
        )
        state.last_index_tip = (raw_x, raw_y)

        drawing_pose = state.point_run >= DRAW_ONSET_FRAMES or (
            state.canvas.is_drawing and not ml_command and hs.is_pointing_relaxed(lm)
        )
        moving_fast = FAST_DRAW_PX <= tip_travel <= MAX_DRAW_JUMP_PX
        draw_grace = (
            state.canvas.is_drawing
            and state.draw_miss_run <= DRAW_GRACE_FRAMES
            and moving_fast
            and not open_now
            and not lens_now
            and not ml_command
        )

        _update_discrete_gestures(state, primary, now, fr)

        if state.lens_run >= LENS_HOLD_FRAMES:
            _release_grab(state)
            _update_lens(state, primary, w, h, now, fr)

        elif state.open_run >= OPEN_HAND_FRAMES and _selected_within_reach(state, palm) is not None:
            state.lens = None
            target = _selected_within_reach(state, palm)
            if state.grab is None or state.grab.object_id != target.id:
                _release_grab(state)
                state.grab = GrabState(object_id=target.id, handedness=primary.handedness)
                _anchor_grab(state.grab, target, state.canvas, now, hs.wrist_roll_angle(lm),
                             hs.yaw_signal(lm), hs.pitch_signal(lm), palm, "rotate")
                state.miss_run = 0
            _update_manipulation(state, primary, palm, "rotate", now, fr)

        elif state.open_run >= OPEN_HAND_FRAMES:
            state.lens = None
            _release_grab(state)
            if state.deselect():
                state.set_status("DESELECTION")

        elif state.grab is not None and state.grab.mode == "move":
            state.lens = None
            if pose == "closed":
                state.miss_run = 0
                _update_manipulation(state, primary, palm, "move", now, fr)
            else:
                state.miss_run += 1
                if state.miss_run > GRAB_RELEASE_GRACE:
                    _release_grab(state)
                    state.set_status("RELACHE")

        elif ml_command:
            state.lens = None
            _release_grab(state)

        elif drawing_pose or draw_grace:
            state.lens = None
            _update_drawing(state, raw_x, raw_y, tip_travel, w, h, now, fr)

        elif state.fist_run >= GRAB_ONSET_FRAMES:
            state.lens = None
            if state.grab is None:
                target = state.canvas.find_object_at(palm[0], palm[1], margin=int(GRAB_REACH))
                if target is not None:
                    state.grab = GrabState(object_id=target.id, handedness=primary.handedness)
                    state.selected_id = target.id
                    state.last_touched_id = target.id
                    target.attached_to = None
                    target.spin = 0.0
                    _anchor_grab(state.grab, target, state.canvas, now, hs.wrist_roll_angle(lm),
                                 hs.yaw_signal(lm), hs.pitch_signal(lm), palm, "move")
                    state.miss_run = 0
                    _update_manipulation(state, primary, palm, "move", now, fr)
                else:
                    state.set_status("POING (aucun element ici)")

        else:
            state.lens = None
            state.point_filter = None
            hovered = state.canvas.find_object_at(palm[0], palm[1])
            fr.hovered_object_id = hovered.id if hovered else None

        if not fr.drawing and fr.lens is None and not fr.resizing:
            fr.snap_armed = state.snap.is_armed(primary.handedness, now)
            if state.snap.update(lm, primary.handedness, now):
                fr.snapped = True

    if _update_auto_deselect(state, palm_now, now, busy=(state.grab is not None or fr.resizing)):
        state.set_status("DESELECTION (main eloignee)")

    if fr.drawing:
        state.draw_miss_run = 0
        state.was_drawing = True
    elif state.was_drawing:
        state.draw_miss_run += 1
        if state.draw_miss_run > DRAW_GRACE_FRAMES:
            _finish_stroke(state, now)

    _update_spin(state, dt)
    _auto_create_element(state, now, fr)

    while state.cursor_trail and (now - state.cursor_trail[0][2]) > TRAIL_MAX_AGE:
        state.cursor_trail.popleft()
    if fr.drawing:
        fr.trail = list(state.cursor_trail)

    if fr.snapped or fr.color_changed:
        fr.color_changed = True
        _apply_color_change(state)

    for obj in state.canvas.objects:
        obj.selected = obj.id == state.selected_id

    _apply_triggers(state, fr)

    if not state.status_set_this_frame:
        if fr.drawing:
            state.set_status("DESSIN")
        elif fr.clear_phase in ("arm", "ready", "confirm"):
            state.last_status = "TOUT EFFACER ?"
        elif fr.hovered_object_id is not None:
            state.last_status = "SURVOL"
        elif state.selected_id is not None:
            state.last_status = "SELECTIONNE"
        else:
            state.last_status = "IDLE"

    return fr


# --- MediaPipe setup -----------------------------------------------------
def build_recognizer(model_path: str) -> mp_vision.GestureRecognizer:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            "Télécharge-le (PowerShell) avec :\n"
            '  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/'
            'gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"'
            ' -OutFile "models\\gesture_recognizer.task"'
        )
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.GestureRecognizerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.GestureRecognizer.create_from_options(options)


# --- Camera loop ---------------------------------------------------------
def save_snapshot(image: np.ndarray) -> str:
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    name = datetime.datetime.now().strftime("pinceau_%Y%m%d_%H%M%S.png")
    path = os.path.join(CAPTURE_DIR, name)
    cv2.imwrite(path, image)
    return path


def open_camera() -> cv2.VideoCapture:
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        for width, height in CAMERA_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if got_w >= width - 16:
                break
        return cap
    raise RuntimeError(
        "Cannot open the webcam (tried index 0, 1 and 2).\n"
        "Check that no other application is using it, and that Windows\n"
        "allows camera access (Settings > Privacy > Camera)."
    )


def main() -> None:
    engine_status = sketch.diagnose_text_recognition()

    recognizer = build_recognizer("models/gesture_recognizer.task")
    cap = open_camera()

    state = AppState(canvas=Canvas(), recognizer=async_recognition.RecognitionService())

    fps_history: collections.deque = collections.deque(maxlen=30)
    prev_frame_time = time.time()
    start_time = time.time()
    show_shadows = True
    show_debug = False
    window_sized = False
    toast, toast_until = "", 0.0

    show_help = True
    help_auto_hidden = False

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if not window_sized:
                cv2.resizeWindow(WINDOW_TITLE, w, h)
                window_sized = True

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            now = time.time()
            fr = process_frame(state, result, w, h, now)

            if show_shadows:
                shadow = state.canvas.shadow_mask((h, w))
                if shadow is not None:
                    hud.apply_shadow(frame, shadow)

            if fr.cursor is not None and not fr.drawing:
                radius = max(5, int(8 * h / 540))
                cv2.circle(frame, (int(fr.cursor[0]), int(fr.cursor[1])), radius, (150, 150, 150), -1)

            canvas_img = state.canvas.render((h, w, 3), fr.hovered_object_id, state.pen_color,
                                             show_labels=show_debug)
            output = cv2.addWeighted(frame, 0.6, canvas_img, 1.0, 0)

            if fr.trail:
                hud.draw_trail(output, fr.trail, state.pen_color)

            if fr.lens is not None:
                magnifier.draw_magnifier(output, fr.lens)

            fps_history.append(1.0 / max(now - prev_frame_time, 1e-6))
            prev_frame_time = now
            fps = sum(fps_history) / len(fps_history)

            if fr.auto_created and show_help and not help_auto_hidden:
                show_help, help_auto_hidden = False, True
                toast, toast_until = "help hidden - press h", now + 1.8

            hud.draw_hud(output, state, fr, show_help)
            if show_debug:
                hud.draw_debug(output, state, fr, fps, engine_status)
            if now < toast_until:
                hud.draw_toast(output, toast)

            cv2.imshow(WINDOW_TITLE, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                show_help, help_auto_hidden = not show_help, True
            elif key == ord("o"):
                show_shadows = not show_shadows
            elif key == ord("d"):
                show_debug = not show_debug
                sketch.VERBOSE = show_debug
                toast, toast_until = f"debug {'on' if show_debug else 'off'}", now + 0.8
            elif key == ord("c"):
                _apply_color_change(state)
                toast, toast_until = "next colour", now + 0.8
            elif key == ord("z"):
                undone = state.canvas.undo()
                state.prune_selection()
                toast = {"clear": "everything restored", "object": "object removed",
                         "stroke": "stroke removed"}.get(undone, "nothing to undo")
                toast_until = now + 1.0
            elif key == ord("s"):
                toast, toast_until = f"saved: {save_snapshot(output)}", now + 1.6
    finally:
        if state.recognizer is not None:
            state.recognizer.close()
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
