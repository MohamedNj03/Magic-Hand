"""Step 3/8 - strokes, 3D objects, and the layer they live on.

Knows nothing about hands. Meshes come from mesh3d.py, recognition from
sketch_recognition.py.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

import mesh3d
import sketch_recognition as sketch


# --- Palette and display constants ---------------------------------------
COLOR_PALETTE = [
    (0, 220, 255),
    (255, 170, 0),
    (120, 100, 255),
    (255, 120, 220),
    (80, 230, 255),
    (255, 220, 120),
    (140, 160, 255),
    (200, 255, 160),
]

KIND_LABELS = {
    "circle": "cercle",
    "triangle": "triangle",
    "square": "carre",
    "rectangle": "rectangle",
    "pentagon": "pentagone",
    "hexagon": "hexagone",
    "text": "lettre",
    "freeform": "dessin",
}

SELECTION_COLOR = (60, 255, 120)
HOVER_COLOR = (180, 180, 180)
LINK_COLOR = (110, 110, 110)

SHADOW_STRENGTH = 0.5
SHADOW_BLUR = 13

MAX_STACK_DEPTH = 32


@dataclass
# --- A stroke ------------------------------------------------------------
class Stroke:
    id: str
    points: list[tuple[int, int]]
    timestamp: float
    width: int = 4
    type: str = "draw"


def new_stroke(points, timestamp: float, width: int = 4) -> Stroke:
    return Stroke(id=str(uuid.uuid4()), points=list(points), timestamp=timestamp, width=width)


# --- From strokes to a 3D mesh -------------------------------------------
def _mesh_depth(bbox_span: float, kind: str) -> float:
    if kind in sketch.SHAPE_KINDS:
        return max(18.0, bbox_span * 0.35)
    return max(8.0, bbox_span * 0.08)


def build_object_mesh(kind: str, strokes: list[Stroke], centroid: tuple[float, float]) -> mesh3d.Mesh3D:
    cx, cy = centroid
    all_pts = [p for s in strokes for p in s.points]
    xs = [p[0] for p in all_pts] or [cx]
    ys = [p[1] for p in all_pts] or [cy]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    depth = _mesh_depth(span, kind)

    if kind in sketch.SHAPE_KINDS and len(strokes) == 1:
        pts = strokes[0].points
        local_poly = [(x - cx, y - cy) for x, y in pts[:-1]]
        if len(local_poly) >= 3:
            return mesh3d.make_prism_mesh(local_poly, depth)

    parts = []
    for s in strokes:
        local_pts = [(x - cx, y - cy) for x, y in s.points]
        parts.append(mesh3d.make_ribbon_mesh(local_pts, depth))
    return mesh3d.combine_meshes(parts)


@dataclass
# --- A validated object --------------------------------------------------
class DrawnObject:
    id: str
    strokes: list[Stroke]
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0])
    roll: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    scale: float = 1.0
    selected: bool = False
    kind: str = "freeform"
    mesh: mesh3d.Mesh3D | None = None
    index: int = 0
    text: str | None = None
    color_idx: int = 0
    spin: float = 0.0
    attached_to: str | None = None
    created_at: float = 0.0
    awaiting_recognition: bool = False
    _bbox_cache: tuple[int, int, int, int] | None = field(default=None, repr=False, compare=False)
    _bbox_key: tuple | None = field(default=None, repr=False, compare=False)

    @property
    def color(self) -> tuple[int, int, int]:
        return COLOR_PALETTE[self.color_idx % len(COLOR_PALETTE)]

    @property
    def label(self) -> str:
        base = KIND_LABELS.get(self.kind, self.kind)
        if self.awaiting_recognition:
            return f"#{self.index} {base} ..."
        if self.kind == "text" and self.text:
            return f"#{self.index} {base} {self.text}"
        return f"#{self.index} {base}"

    def local_points(self) -> list[tuple[int, int]]:
        pts: list[tuple[int, int]] = []
        for s in self.strokes:
            pts.extend(s.points)
        return pts

    def centroid(self) -> tuple[float, float]:
        pts = self.local_points()
        if not pts:
            return (0.0, 0.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def ensure_mesh(self) -> mesh3d.Mesh3D:
        if self.mesh is None:
            self.mesh = build_object_mesh(self.kind, self.strokes, self.centroid())
        return self.mesh

    def invalidate_mesh(self) -> None:
        self.mesh = None
        self._bbox_cache = None
        self._bbox_key = None

    def replace_strokes(self, strokes: list[Stroke]) -> None:
        cx, cy = self.center_screen()
        self.strokes = strokes
        self.invalidate_mesh()
        new_cx, new_cy = self.centroid()
        self.offset[0] = cx - new_cx
        self.offset[1] = cy - new_cy

    def center_screen(self) -> tuple[float, float]:
        cx, cy = self.centroid()
        return cx + self.offset[0], cy + self.offset[1]

    def transformed_bbox(self) -> tuple[int, int, int, int]:
        key = (self.roll, self.yaw, self.pitch, self.scale, self.offset[0], self.offset[1], id(self.mesh))
        if self._bbox_key != key or self._bbox_cache is None:
            self._bbox_cache = mesh3d.screen_bounds(
                self.ensure_mesh(), self.roll, self.yaw, self.center_screen(), self.scale, self.pitch
            )
            self._bbox_key = key
        return self._bbox_cache

    def contains_point(self, px: float, py: float, margin: int = 15) -> bool:
        x0, y0, x1, y1 = self.transformed_bbox()
        return (x0 - margin) <= px <= (x1 + margin) and (y0 - margin) <= py <= (y1 + margin)


# --- The layer -----------------------------------------------------------
class Canvas:
    def __init__(self) -> None:
        self.pending_strokes: list[Stroke] = []
        self.objects: list[DrawnObject] = []
        self._action_log: list[tuple[str, object]] = []
        self._redo_log: list[tuple[str, object]] = []
        self._active: Stroke | None = None
        self._counter: int = 0

    @property
    def active_stroke(self) -> Stroke | None:
        return self._active

    @property
    def is_drawing(self) -> bool:
        return self._active is not None

    def start_stroke(self, t: float, width: int = 4) -> None:
        self._active = new_stroke([], t, width)

    def add_point(self, x: int, y: int) -> None:
        if self._active is None:
            return
        point = (int(x), int(y))
        if self._active.points and self._active.points[-1] == point:
            return
        self._active.points.append(point)

    def end_stroke(self) -> None:
        if self._active is not None and len(self._active.points) >= 2:
            self.pending_strokes.append(self._active)
            self._action_log.append(("stroke", self._active))
            self._redo_log.clear()
        self._active = None

    def pending_bbox(self) -> tuple[float, float, float, float] | None:
        pts = [p for s in self.pending_strokes for p in s.points]
        if self._active is not None:
            pts.extend(self._active.points)
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def distance_to_pending(self, x: float, y: float) -> float:
        box = self.pending_bbox()
        if box is None:
            return math.inf
        x0, y0, x1, y1 = box
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        return math.hypot(dx, dy)

    def validate(self, created_at: float = 0.0, color_idx: int = 0, recognizer=None) -> DrawnObject | None:
        if not self.pending_strokes:
            return None

        kept = sketch.clean_strokes(self.pending_strokes)
        if not kept:
            self.pending_strokes = []
            return None

        widths = [orig.width for orig, _ in kept]
        points_lists = [pts for _, pts in kept]
        base_ts = kept[0][0].timestamp
        base_width = kept[0][0].width

        text: str | None = None
        awaiting = False
        shape = sketch.classify_shape_only(points_lists)
        if shape is not None:
            kind, display_points = shape
        elif recognizer is not None:
            kind, display_points = "freeform", points_lists
            awaiting = True
        else:
            kind, display_points, text = sketch.classify_ex(points_lists, widths)

        self._counter += 1
        obj = DrawnObject(
            id=str(uuid.uuid4()),
            strokes=[new_stroke(pts, base_ts, base_width) for pts in display_points],
            kind=kind,
            index=self._counter,
            text=text,
            color_idx=color_idx,
            created_at=created_at,
            awaiting_recognition=awaiting,
        )
        self.objects.append(obj)
        self.pending_strokes = []
        while self._action_log and self._action_log[-1][0] == "stroke":
            self._action_log.pop()
        self._action_log.append(("object", obj))
        self._redo_log.clear()

        if awaiting:
            recognizer.submit(obj.id, points_lists, widths)
        return obj

    def apply_recognition(self, result) -> DrawnObject | None:
        obj = self.get_object(result.object_id)
        if obj is None:
            return None
        obj.awaiting_recognition = False
        if not result.text or not result.strokes:
            return obj
        base = obj.strokes[0] if obj.strokes else None
        timestamp = base.timestamp if base is not None else obj.created_at
        width = base.width if base is not None else 4
        obj.replace_strokes([new_stroke(pts, timestamp, width) for pts in result.strokes])
        obj.kind = "text"
        obj.text = result.text
        return obj

    def undo(self) -> str | None:
        while self._action_log:
            kind, item = self._action_log.pop()
            if kind == "object" and item in self.objects:
                self.objects.remove(item)
                self._redo_log.append((kind, item))
                self.detach_orphans()
                return "object"
            if kind == "stroke" and item in self.pending_strokes:
                self.pending_strokes.remove(item)
                self._redo_log.append((kind, item))
                return "stroke"
            if kind == "clear":
                objects, pending = item
                self.objects = list(objects)
                self.pending_strokes = list(pending)
                self._redo_log.clear()
                return "clear"
        return None

    def redo(self) -> None:
        if not self._redo_log:
            return
        kind, item = self._redo_log.pop()
        if kind == "object":
            self.objects.append(item)
        elif kind == "stroke":
            self.pending_strokes.append(item)
        else:
            return
        self._action_log.append((kind, item))

    def delete_object(self, object_id: str) -> DrawnObject | None:
        for i, obj in enumerate(self.objects):
            if obj.id == object_id:
                self.objects.pop(i)
                self._action_log = [entry for entry in self._action_log if entry[1] is not obj]
                self._redo_log.append(("object", obj))
                self.detach_orphans()
                return obj
        return None

    def delete_last_object(self) -> DrawnObject | None:
        if not self.objects:
            return None
        return self.delete_object(self.objects[-1].id)

    def clear(self) -> None:
        if self.objects or self.pending_strokes:
            self._action_log.append(("clear", (list(self.objects), list(self.pending_strokes))))
        self._active = None
        self.objects = []
        self.pending_strokes = []
        self._redo_log.clear()

    def get_object(self, object_id: str | None) -> DrawnObject | None:
        if object_id is None:
            return None
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    def stack_depth(self, obj: DrawnObject) -> int:
        depth, current = 0, obj
        while current.attached_to is not None and depth < MAX_STACK_DEPTH:
            current = self.get_object(current.attached_to)
            if current is None:
                break
            depth += 1
        return depth

    def find_object_at(self, px: float, py: float, margin: int = 15) -> DrawnObject | None:
        candidates = [o for o in self.objects if o.contains_point(px, py, margin)]
        if not candidates:
            return None

        def rank(o: DrawnObject) -> tuple[int, float]:
            ox, oy = o.center_screen()
            return (-self.stack_depth(o), (ox - px) ** 2 + (oy - py) ** 2)

        return min(candidates, key=rank)

    def descendants(self, object_id: str) -> list[DrawnObject]:
        found: list[DrawnObject] = []
        frontier = [object_id]
        seen = {object_id}
        while frontier:
            current = frontier.pop()
            for obj in self.objects:
                if obj.attached_to == current and obj.id not in seen:
                    seen.add(obj.id)
                    found.append(obj)
                    frontier.append(obj.id)
        return found

    def detach_orphans(self) -> None:
        alive = {o.id for o in self.objects}
        for obj in self.objects:
            if obj.attached_to is not None and obj.attached_to not in alive:
                obj.attached_to = None

    def draw_order(self) -> list[DrawnObject]:
        return sorted(self.objects, key=self.stack_depth)

    def shadow_mask(self, shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        if not self.objects:
            return None
        h, w = shape
        half = np.zeros((h // 2, w // 2), dtype=np.uint8)
        x0 = y0 = math.inf
        x1 = y1 = -math.inf
        for obj in self.objects:
            hull = mesh3d.shadow_hull(
                obj.ensure_mesh(), obj.roll, obj.yaw, obj.center_screen(), obj.scale, obj.pitch
            )
            if len(hull) < 3:
                continue
            cv2.fillConvexPoly(half, (hull // 2).astype(np.int32), 255, cv2.LINE_AA)
            x0, y0 = min(x0, hull[:, 0].min()), min(y0, hull[:, 1].min())
            x1, y1 = max(x1, hull[:, 0].max()), max(y1, hull[:, 1].max())
        if x1 < x0:
            return None

        half = cv2.GaussianBlur(half, (SHADOW_BLUR, SHADOW_BLUR), 0)
        mask = cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR)
        pad = SHADOW_BLUR * 2
        box = (
            max(0, int(x0) - pad), max(0, int(y0) - pad),
            min(w, int(x1) + pad), min(h, int(y1) + pad),
        )
        return mask, box

    def render(
        self,
        shape: tuple[int, int, int],
        hovered_id: str | None,
        pending_color: tuple[int, int, int] = (0, 220, 255),
        show_labels: bool = True,
    ) -> np.ndarray:
        img = np.zeros(shape, dtype=np.uint8)
        for obj in self.draw_order():
            render_object(img, obj, hovered=(obj.id == hovered_id), show_label=show_labels)

        for obj in self.objects:
            support = self.get_object(obj.attached_to)
            if support is not None:
                p1 = tuple(int(v) for v in obj.center_screen())
                p2 = tuple(int(v) for v in support.center_screen())
                cv2.line(img, p1, p2, LINK_COLOR, 1, cv2.LINE_AA)
                cv2.circle(img, p2, 3, LINK_COLOR, -1, cv2.LINE_AA)

        pending = self.pending_strokes + ([self._active] if self._active else [])
        for stroke in pending:
            for p1, p2 in zip(stroke.points, stroke.points[1:]):
                cv2.line(img, p1, p2, pending_color, stroke.width, cv2.LINE_AA)
        return img


# --- Rendering -----------------------------------------------------------
def render_object(img: np.ndarray, obj: DrawnObject, hovered: bool, show_label: bool = True) -> None:
    mesh3d.render_mesh(
        img, obj.ensure_mesh(), obj.roll, obj.yaw, obj.center_screen(), obj.color, obj.scale, obj.pitch
    )

    x0, y0, x1, y1 = obj.transformed_bbox()
    if obj.selected:
        cv2.rectangle(img, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), SELECTION_COLOR, 2, cv2.LINE_AA)
        for cx, cy in ((x0 - 8, y0 - 8), (x1 + 8, y0 - 8), (x0 - 8, y1 + 8), (x1 + 8, y1 + 8)):
            cv2.circle(img, (cx, cy), 4, SELECTION_COLOR, -1, cv2.LINE_AA)
    elif hovered:
        cv2.rectangle(img, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), HOVER_COLOR, 1, cv2.LINE_AA)

    if show_label:
        label_color = SELECTION_COLOR if obj.selected else (170, 170, 170)
        cv2.putText(img, obj.label, (x0 - 6, max(y0 - 14, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
