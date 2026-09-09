"""Step 4/8 - a finished drawing becomes a shape, a letter, or stays a drawing.

Shapes are read inline; handwriting is slow and runs off-thread. Called by
canvas.Canvas.validate.
"""
from __future__ import annotations

import math
import os
import shutil

import cv2
import numpy as np


# --- Tesseract lookup ----------------------------------------------------
_TESSERACT_COMMON_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR", "tesseract.exe"),
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
]


def _autoconfigure_tesseract() -> None:
    try:
        import pytesseract
    except ImportError:
        return
    if shutil.which("tesseract"):
        return
    for path in _TESSERACT_COMMON_PATHS:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


VERBOSE = False


def log(message: str) -> None:
    if VERBOSE:
        print(message)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


_autoconfigure_tesseract()


# --- Stage 1: clean the raw stroke ---------------------------------------
MIN_STROKE_LENGTH = 10.0
RDP_EPSILON = 2.5


def stroke_path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:]))


def _point_segment_distance(p, a, b) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def rdp_simplify(points: list[tuple[float, float]], epsilon: float = RDP_EPSILON) -> list[tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    start, end = points[0], points[-1]
    max_dist, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = _point_segment_distance(points[i], start, end)
        if d > max_dist:
            max_dist, index = d, i
    if max_dist > epsilon:
        left = rdp_simplify(points[: index + 1], epsilon)
        right = rdp_simplify(points[index:], epsilon)
        return left[:-1] + right
    return [start, end]


def clean_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    simplified = rdp_simplify([(float(x), float(y)) for x, y in points])
    return [(int(round(x)), int(round(y))) for x, y in simplified]


def clean_strokes(strokes: list) -> list:
    kept = []
    for s in strokes:
        if stroke_path_length(s.points) < MIN_STROKE_LENGTH:
            continue
        new_points = clean_points(s.points)
        if len(new_points) >= 2:
            kept.append((s, new_points))
    return kept


# --- Stage 2: shapes -----------------------------------------------------
SHAPE_KINDS = {"circle", "triangle", "square", "rectangle", "pentagon", "hexagon"}

MIN_SHAPE_POINTS = 4
SHAPE_CLOSURE = 0.45
MIN_SHAPE_AREA = 400.0
MIN_SHAPE_PERIMETER = 40.0
MIN_SHAPE_SOLIDITY = 0.80
CIRCLE_FILL = 0.86
CIRCLE_ROUNDNESS = 0.13
CORNER_ANGLE = 26.0
CORNER_RELATIVE = 1.7
CORNER_WINDOW = 0.055
CORNER_SEPARATION = 0.09
SIDE_STRAIGHTNESS = 0.14
MIN_POLYGON_SIDE = 0.07
MAX_POLYGON_SIDES = 6
OVERSHOOT_RETURN = 0.30
OVERSHOOT_MIN_TAIL = 0.02
OVERSHOOT_MAX_TAIL = 0.40
RESAMPLE_POINTS = 96


def _resample_path(points, count: int = RESAMPLE_POINTS) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts
    steps = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    total = float(steps.sum())
    if total <= 1e-6:
        return pts
    walked = np.concatenate([[0.0], np.cumsum(steps)])
    targets = np.linspace(0.0, total, count)
    return np.stack([np.interp(targets, walked, pts[:, 0]),
                     np.interp(targets, walked, pts[:, 1])], axis=1)


def _trim_overshoot(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    if n < 12:
        return pts
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    steps = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    walked = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(walked[-1]) or 1.0

    tail_start = int(n * 0.45)
    tail = pts[tail_start:]
    distances = np.hypot(tail[:, 0] - pts[0, 0], tail[:, 1] - pts[0, 1])
    closest = int(np.argmin(distances))
    index = tail_start + closest
    if index < 3 or float(distances[closest]) > OVERSHOOT_RETURN * span:
        return pts
    tail = total - float(walked[index])
    if not (OVERSHOOT_MIN_TAIL * total <= tail <= OVERSHOOT_MAX_TAIL * total):
        return pts
    return pts[:index + 1]


def _closed_outline(pts: np.ndarray) -> np.ndarray:
    path = _resample_path(np.vstack([pts, pts[:1]]), RESAMPLE_POINTS)
    k = 3
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    wrapped = np.vstack([path[-k:], path, path[:k]])
    return np.stack([np.convolve(wrapped[:, 0], kernel, mode="valid"),
                     np.convolve(wrapped[:, 1], kernel, mode="valid")], axis=1)


def _turn_angles(path: np.ndarray, window: int) -> np.ndarray:
    arriving = path - np.roll(path, window, axis=0)
    leaving = np.roll(path, -window, axis=0) - path
    cross = arriving[:, 0] * leaving[:, 1] - arriving[:, 1] * leaving[:, 0]
    dot = (arriving * leaving).sum(axis=1)
    return np.abs(np.degrees(np.arctan2(cross, dot)))


def _corner_indices(path: np.ndarray) -> list[int]:
    count = len(path)
    window = max(2, int(count * CORNER_WINDOW))
    separation = max(3, int(count * CORNER_SEPARATION))
    turns = _turn_angles(path, window)
    threshold = max(CORNER_ANGLE, CORNER_RELATIVE * float(np.median(turns)))
    chosen: list[int] = []
    for index in np.argsort(-turns):
        if turns[index] < threshold:
            break
        gaps = [min(abs(int(index) - other), count - abs(int(index) - other)) for other in chosen]
        if all(gap >= separation for gap in gaps):
            chosen.append(int(index))
    return sorted(chosen)


def _sides_are_straight(path: np.ndarray, corners: list[int], perimeter: float) -> bool:
    count = len(path)
    for start, end in zip(corners, corners[1:] + corners[:1]):
        indices = list(range(start, end + 1)) if end > start \
            else list(range(start, count)) + list(range(0, end + 1))
        if len(indices) < 3:
            return False
        side = path[indices]
        chord = side[-1] - side[0]
        length = float(np.hypot(*chord))
        if length < MIN_POLYGON_SIDE * perimeter:
            return False
        normal = np.array([-chord[1], chord[0]]) / length
        bow = float(np.abs((side - side[0]) @ normal).max())
        if bow > SIDE_STRAIGHTNESS * length:
            return False
    return True


def _is_round(path: np.ndarray) -> tuple[tuple[float, float], float] | None:
    (cx, cy), radius = cv2.minEnclosingCircle(path.astype(np.float32))
    if radius <= 1e-6:
        return None
    radii = np.hypot(path[:, 0] - cx, path[:, 1] - cy)
    mean_radius = float(radii.mean())
    if mean_radius / radius < CIRCLE_FILL:
        return None
    if float(radii.std()) / max(mean_radius, 1e-6) > CIRCLE_ROUNDNESS:
        return None
    return (cx, cy), radius


def encloses_a_shape(points: list[tuple[int, int]]) -> bool:
    if len(points) < MIN_SHAPE_POINTS:
        return False
    pts = _trim_overshoot(np.array(points, dtype=np.float64))
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    if float(np.hypot(*(pts[0] - pts[-1]))) > SHAPE_CLOSURE * span:
        return False
    contour = pts.astype(np.int32)
    area = abs(float(cv2.contourArea(contour)))
    if area < MIN_SHAPE_AREA:
        return False
    hull_area = abs(float(cv2.contourArea(cv2.convexHull(contour))))
    return hull_area > 0.0 and area / hull_area >= MIN_SHAPE_SOLIDITY


def classify_shape(points: list[tuple[int, int]]) -> tuple[str, list[tuple[float, float]]] | None:
    if len(points) < MIN_SHAPE_POINTS:
        return None
    pts = _trim_overshoot(np.array(points, dtype=np.float64))

    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    if float(np.hypot(*(pts[0] - pts[-1]))) > SHAPE_CLOSURE * span:
        return None

    contour = pts.astype(np.int32)
    area = abs(float(cv2.contourArea(contour)))
    if area < MIN_SHAPE_AREA:
        return None

    hull = cv2.convexHull(contour)
    hull_area = abs(float(cv2.contourArea(hull)))
    perimeter = float(cv2.arcLength(hull, True))
    if hull_area <= 0.0 or perimeter < MIN_SHAPE_PERIMETER:
        return None
    if area / hull_area < MIN_SHAPE_SOLIDITY:
        return None

    outline = _closed_outline(pts)
    corners = _corner_indices(outline)
    if 3 <= len(corners) <= MAX_POLYGON_SIDES and _sides_are_straight(outline, corners, perimeter):
        polygon = [(float(x), float(y)) for x, y in outline[corners]]
        if len(corners) == 3:
            return "triangle", polygon
        if len(corners) == 4:
            return _classify_quad(pts.astype(np.float32))
        return ("pentagon" if len(corners) == 5 else "hexagon"), polygon

    round_fit = _is_round(outline)
    if round_fit is not None:
        (cx, cy), radius = round_fit
        return "circle", _circle_polygon(cx, cy, radius)
    return None


def _classify_quad(pts: np.ndarray) -> tuple[str, list[tuple[float, float]]] | None:
    rect = cv2.minAreaRect(pts)
    (_, _), (w, h), _ = rect
    if w < 1 or h < 1:
        return None
    ratio = max(w, h) / max(min(w, h), 1e-3)
    box = [tuple(p) for p in cv2.boxPoints(rect)]
    return ("square" if ratio <= 1.15 else "rectangle"), box


def _circle_polygon(cx: float, cy: float, r: float, n: int = 32) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


# --- Stage 3a: handwriting through Tesseract -----------------------------
_tesseract_ok: bool | None = None


def tesseract_available() -> bool:
    global _tesseract_ok
    if _tesseract_ok is None:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            _tesseract_ok = True
        except Exception:
            _tesseract_ok = False
    return _tesseract_ok


def diagnose_text_recognition() -> str:
    try:
        import pytesseract
    except ImportError:
        return "ECRITURE: desactivee (pip install pytesseract manquant)"
    try:
        version = pytesseract.get_tesseract_version()
        return f"ECRITURE: active (Tesseract {version})"
    except Exception:
        return "ECRITURE: desactivee (pytesseract est installe mais le binaire Tesseract est introuvable sur le PATH)"


MIN_WRITING_SPAN = 25.0
MAX_WRITING_STROKES = 12
MIN_TEXT_CONFIDENCE = 70.0
CLOSED_LOOP_CONFIDENCE = 82.0


def looks_like_writing(strokes_points: list[list[tuple[int, int]]]) -> bool:
    if not (1 <= len(strokes_points) <= MAX_WRITING_STROKES):
        return False
    all_pts = [p for pts in strokes_points for p in pts]
    if len(all_pts) < 4:
        return False
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if max(w, h) < MIN_WRITING_SPAN:
        return False
    ratio = max(w, 1) / max(h, 1)
    if not (0.10 <= ratio <= 8.0):
        return False
    if len(strokes_points) >= 2:
        return True
    diagonal = max(math.hypot(w, h), 1.0)
    return stroke_path_length(strokes_points[0]) >= 1.35 * diagonal


def try_recognize_text(strokes_points: list[list[tuple[int, int]]], widths: list[int],
                       min_confidence: float = MIN_TEXT_CONFIDENCE) -> str | None:
    if not tesseract_available():
        return None
    try:
        import pytesseract
    except ImportError:
        return None

    all_pts = [p for pts in strokes_points for p in pts]
    if not all_pts:
        return None
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w, h = max(int(x1 - x0), 1), max(int(y1 - y0), 1)
    pad = 28
    img = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)

    ocr_thickness = int(clamp(h * 0.15, 3, 30))
    for pts, _ in zip(strokes_points, widths):
        shifted = [(int(px - x0 + pad), int(py - y0 + pad)) for px, py in pts]
        for p1, p2 in zip(shifted, shifted[1:]):
            cv2.line(img, p1, p2, 255, ocr_thickness, cv2.LINE_AA)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = 255 - img

    best_text, best_conf = None, -1.0
    try:
        for config in ("--psm 7", "--psm 8", "--psm 6"):
            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            words = [(t, float(c)) for t, c in zip(data["text"], data["conf"]) if t.strip() and float(c) >= 0]
            if not words:
                continue
            text = "".join(t for t, _ in words).strip()
            conf = sum(c for _, c in words) / len(words)
            if conf > best_conf:
                best_text, best_conf = text, conf
    except Exception as exc:
        log(f"ECRITURE: erreur Tesseract pendant la reconnaissance ({exc})")
        return None

    if best_text is None:
        log("ECRITURE: aucun texte détecté dans le tracé (Tesseract n'a rien lu)")
        return None
    if best_conf < min_confidence:
        log(f"ECRITURE: '{best_text}' détecté mais confiance trop basse ({best_conf:.0f} < {min_confidence:.0f}) — rejeté")
        return None

    text = "".join(ch for ch in best_text if ch.isalnum() or ch in " -'")
    if not text or len(text) > 14:
        log(f"ECRITURE: '{best_text}' détecté (confiance {best_conf:.0f}) mais rejeté après nettoyage (vide ou trop long)")
        return None
    log(f"ECRITURE: '{text}' reconnu (confiance {best_conf:.0f})")
    return text


# --- Stage 3b: offline template fallback ---------------------------------
LETTER_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_NORM_BOX = 64
_NORM_INK = 52
_NORM_THICKNESS = 3
_TEMPLATE_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
)

_LETTER_MAX_CHAMFER = 4.2
_LETTER_MIN_MARGIN = 0.28

_template_cache: dict | None = None


def _normalize_ink(bitmap: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(bitmap)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    crop = bitmap[y0 : y1 + 1, x0 : x1 + 1]
    h, w = crop.shape
    ratio = _NORM_INK / max(h, w)
    new_w, new_h = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((_NORM_BOX, _NORM_BOX), dtype=np.uint8)
    ox, oy = (_NORM_BOX - new_w) // 2, (_NORM_BOX - new_h) // 2
    out[oy : oy + new_h, ox : ox + new_w] = resized
    return (out > 60).astype(np.uint8) * 255


def _skeleton(bitmap: np.ndarray) -> np.ndarray:
    try:
        return cv2.ximgproc.thinning(bitmap)
    except Exception:
        eroded = cv2.erode(bitmap, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.subtract(bitmap, eroded)


def _thicken(bitmap: np.ndarray, thickness: int = _NORM_THICKNESS) -> np.ndarray:
    k = max(1, thickness)
    return cv2.dilate(bitmap, np.ones((k, k), np.uint8), iterations=1)


def _build_template_bank() -> dict:
    bank: dict = {}
    for ch in LETTER_ALPHABET:
        entries = []
        for font in _TEMPLATE_FONTS:
            scale = 4.0
            (tw, th), baseline = cv2.getTextSize(ch, font, scale, 6)
            canvas = np.zeros((th + baseline + 40, tw + 40), dtype=np.uint8)
            cv2.putText(canvas, ch, (20, th + 20), font, scale, 255, 6, cv2.LINE_AA)
            norm = _normalize_ink(_thicken(_skeleton(canvas)))
            if norm is None:
                continue
            dist = cv2.distanceTransform(255 - norm, cv2.DIST_L2, 3)
            entries.append((norm, dist))
        if entries:
            bank[ch] = entries
    return bank


def _template_bank() -> dict:
    global _template_cache
    if _template_cache is None:
        _template_cache = _build_template_bank()
    return _template_cache


def _chamfer_symmetric(a_bitmap, a_dist, b_bitmap, b_dist) -> float:
    a_ink = a_bitmap > 0
    b_ink = b_bitmap > 0
    if not a_ink.any() or not b_ink.any():
        return 1e9
    return 0.5 * (float(b_dist[a_ink].mean()) + float(a_dist[b_ink].mean()))


def strokes_to_bitmap(strokes_points: list[list[tuple[int, int]]], thickness: int = _NORM_THICKNESS) -> np.ndarray | None:
    all_pts = [p for pts in strokes_points for p in pts]
    if len(all_pts) < 2:
        return None
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x0, y0 = min(xs), min(ys)
    w = max(int(max(xs) - x0), 1) + 2 * thickness + 4
    h = max(int(max(ys) - y0), 1) + 2 * thickness + 4
    img = np.zeros((h, w), dtype=np.uint8)
    off = thickness + 2
    for pts in strokes_points:
        shifted = [(int(px - x0 + off), int(py - y0 + off)) for px, py in pts]
        for p1, p2 in zip(shifted, shifted[1:]):
            cv2.line(img, p1, p2, 255, thickness, cv2.LINE_AA)
    return img


def looks_like_a_letter(strokes_points: list[list[tuple[int, int]]]) -> bool:
    if not (1 <= len(strokes_points) <= 4):
        return False
    all_pts = [p for pts in strokes_points for p in pts]
    if len(all_pts) < 6:
        return False
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if max(w, h) < 25:
        return False
    ratio = max(w, 1) / max(h, 1)
    return 0.12 <= ratio <= 2.2


def _deskew(bitmap: np.ndarray) -> np.ndarray:
    m = cv2.moments(bitmap, binaryImage=True)
    if abs(m["mu02"]) < 1e-2:
        return bitmap
    skew = m["mu11"] / m["mu02"]
    if abs(skew) < 0.02 or abs(skew) > 1.0:
        return bitmap
    h, w = bitmap.shape
    pad = int(abs(skew) * h) + 2
    padded = cv2.copyMakeBorder(bitmap, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    matrix = np.float32([[1, -skew, skew * h * 0.5 + pad * 0], [0, 1, 0]])
    return cv2.warpAffine(padded, matrix, (padded.shape[1], h), flags=cv2.INTER_LINEAR)


def recognize_letter_offline(
    strokes_points: list[list[tuple[int, int]]],
    max_chamfer: float = _LETTER_MAX_CHAMFER,
    min_margin: float = _LETTER_MIN_MARGIN,
) -> tuple[str, float] | None:
    if not looks_like_a_letter(strokes_points):
        return None
    raw = strokes_to_bitmap(strokes_points)
    if raw is None:
        return None

    variants = []
    for candidate in (raw, _deskew(raw)):
        norm = _normalize_ink(candidate)
        if norm is not None:
            variants.append((norm, cv2.distanceTransform(255 - norm, cv2.DIST_L2, 3)))
    if not variants:
        return None

    scored: list[tuple[float, str]] = []
    for ch, entries in _template_bank().items():
        best = min(
            _chamfer_symmetric(drawn, drawn_dist, tpl, tpl_dist)
            for drawn, drawn_dist in variants
            for tpl, tpl_dist in entries
        )
        scored.append((best, ch))
    if not scored:
        return None
    scored.sort()

    best_score, best_char = scored[0]
    if best_score > max_chamfer:
        return None
    if len(scored) > 1 and (scored[1][0] - best_score) < min_margin:
        return None
    return best_char, best_score


OFFLINE_MAX_LETTERS = 6
OFFLINE_LETTER_GAP = 0.18


def split_into_letters(strokes_points: list[list[tuple[int, int]]]) -> list[list[list[tuple[int, int]]]]:
    boxes = []
    for pts in strokes_points:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if not xs:
            continue
        boxes.append((min(xs), max(xs), min(ys), max(ys), pts))
    if not boxes:
        return []
    boxes.sort(key=lambda b: b[0])

    ink_height = max(b[3] for b in boxes) - min(b[2] for b in boxes)
    gap = max(4.0, OFFLINE_LETTER_GAP * ink_height)

    clusters: list[list[list[tuple[int, int]]]] = [[boxes[0][4]]]
    reach = boxes[0][1]
    for x0, x1, _, _, pts in boxes[1:]:
        if x0 - reach > gap:
            clusters.append([pts])
        else:
            clusters[-1].append(pts)
        reach = max(reach, x1)
    return clusters


def recognize_word_offline(strokes_points: list[list[tuple[int, int]]]) -> str | None:
    clusters = split_into_letters(strokes_points)
    if not clusters:
        return None
    if len(clusters) == 1:
        guess = recognize_letter_offline(strokes_points)
        return guess[0] if guess is not None else None
    if len(clusters) > OFFLINE_MAX_LETTERS:
        return None

    letters = []
    for cluster in clusters:
        guess = recognize_letter_offline(cluster)
        if guess is None:
            return None
        letters.append(guess[0])
    return "".join(letters)


# --- Recognised text, drawn back as strokes ------------------------------
def text_to_strokes(text: str, target_width: float, target_height: float) -> list[list[tuple[int, int]]]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 3.0, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    if tw <= 0 or th <= 0:
        return []
    pad = 20
    canvas = np.zeros((th + baseline + 2 * pad, tw + 2 * pad), dtype=np.uint8)
    cv2.putText(canvas, text, (pad, th + pad), font, scale, 255, thickness, cv2.LINE_AA)
    contours, _ = cv2.findContours(canvas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    raw = [[(float(p[0][0]), float(p[0][1])) for p in cnt] for cnt in contours if len(cnt) >= 5]
    if not raw:
        return []
    xs = [x for pts in raw for x, _ in pts]
    ys = [y for pts in raw for _, y in pts]
    ox, oy = min(xs), min(ys)
    sx = target_width / max(max(xs) - ox, 1e-6)
    sy = target_height / max(max(ys) - oy, 1e-6)

    strokes = []
    for pts in raw:
        scaled = [((x - ox) * sx, (y - oy) * sy) for x, y in pts]
        scaled.append(scaled[0])
        strokes.append([(int(round(x)), int(round(y))) for x, y in scaled])
    return strokes


# --- Entry points used by canvas.Canvas.validate -------------------------
def classify_shape_only(
    strokes_points: list[list[tuple[int, int]]],
) -> tuple[str, list[list[tuple[int, int]]]] | None:
    if len(strokes_points) != 1:
        return None
    shape = classify_shape(strokes_points[0])
    if shape is None:
        return None
    kind, polygon = shape
    poly_pts = [(int(round(x)), int(round(y))) for x, y in polygon]
    poly_pts.append(poly_pts[0])
    return kind, [poly_pts]


def recognize_text(
    strokes_points: list[list[tuple[int, int]]],
    widths: list[int],
    allow_tesseract: bool = True,
) -> tuple[str, list[list[tuple[int, int]]]] | None:
    if not looks_like_writing(strokes_points):
        return None

    closed_loop = len(strokes_points) == 1 and encloses_a_shape(strokes_points[0])
    confidence = CLOSED_LOOP_CONFIDENCE if closed_loop else MIN_TEXT_CONFIDENCE

    text = try_recognize_text(strokes_points, widths, confidence) if allow_tesseract else None

    if not text and not closed_loop:
        guess = recognize_word_offline(strokes_points)
        if guess is not None:
            text = guess
            log(f"ECRITURE (hors-ligne): '{text}' reconnu")

    if not text:
        return None

    all_pts = [p for pts in strokes_points for p in pts]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    text_strokes = text_to_strokes(text, max(x1 - x0, 20), max(y1 - y0, 20))
    if not text_strokes:
        return None
    positioned = [[(int(x + x0), int(y + y0)) for x, y in pts] for pts in text_strokes]
    return text, positioned


def classify_ex(
    strokes_points: list[list[tuple[int, int]]],
    widths: list[int],
    allow_tesseract: bool = True,
) -> tuple[str, list[list[tuple[int, int]]], str | None]:
    shape = classify_shape_only(strokes_points)
    if shape is not None:
        return shape[0], shape[1], None

    written = recognize_text(strokes_points, widths, allow_tesseract)
    if written is not None:
        return "text", written[1], written[0]

    return "freeform", strokes_points, None


def classify(strokes_points: list[list[tuple[int, int]]], widths: list[int]) -> tuple[str, list[list[tuple[int, int]]]]:
    kind, strokes, _text = classify_ex(strokes_points, widths)
    return kind, strokes
