"""
sketch_recognition.py — Ce que fait un objet une fois VALIDATE : nettoyer
le trait, puis essayer de le reconnaître (forme géométrique, fleur,
écriture), et sinon le garder tel quel en version nettoyée.

Ordre volontaire : formes géométriques d'abord (fiable, bon marché,
critère net), puis fleur (motif spécifique : plusieurs petites boucles
groupées), puis écriture (le plus flou des trois, essayé en dernier pour
ne pas risquer de faire passer un carré propre pour une lettre).

Honnêteté de portée : la reconnaissance de FORMES est fiable (géométrie
classique, cv2.approxPolyDP). La détection de FLEUR est une heuristique
maison ciblée sur un seul motif (boucles groupées) — pas un classifieur
entraîné, donc pas fiable sur n'importe quel gribouillage. L'écriture
passe par Tesseract OCR (optionnel, dégrade proprement s'il n'est pas
installé) — bonne précision sur des lettres capitales bien séparées,
moins bonne sur du cursif rapide.
"""
from __future__ import annotations

import math
import os
import shutil

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Auto-détection du binaire Tesseract — sur Windows, l'installeur ne coche
# pas toujours "Add to PATH", ce qui fait que `pytesseract` (le paquet pip)
# est présent mais ne trouve pas le VRAI moteur OCR (le binaire système).
# Plutôt que de demander de bricoler les variables d'environnement Windows,
# on cherche aux emplacements d'installation standards et on configure
# pytesseract directement si on trouve quelque chose — aucune action de
# l'utilisateur requise si le binaire est là mais juste pas sur le PATH.
# --------------------------------------------------------------------------
_TESSERACT_COMMON_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR", "tesseract.exe"),
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",  # macOS Apple Silicon (Homebrew)
]


def _autoconfigure_tesseract() -> None:
    try:
        import pytesseract
    except ImportError:
        return
    if shutil.which("tesseract"):
        return  # déjà trouvable normalement, rien à faire
    for path in _TESSERACT_COMMON_PATHS:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


# Journal de la reconnaissance. Éteint par défaut : ces lignes ("ECRITURE:
# 'T' reconnu (confiance 93)") sont un outil de mise au point, pas quelque
# chose à imposer à quelqu'un qui dessine. La touche `d` de l'application
# les rallume quand on cherche pourquoi une lettre n'est pas passée.
VERBOSE = False


def log(message: str) -> None:
    if VERBOSE:
        print(message)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


_autoconfigure_tesseract()  # exécuté une fois à l'import, avant tout usage ci-dessous


# --------------------------------------------------------------------------
# Nettoyage des traits — répond à "mes traits ne sont pas justes... des
# vibrations". Deux passes : simplification (Ramer-Douglas-Peucker, retire
# le zigzag tout en gardant la forme) + suppression des micro-traits
# accidentels (trop courts pour être un vrai geste de dessin).
# --------------------------------------------------------------------------
MIN_STROKE_LENGTH = 10.0  # pixels — en-dessous, considéré comme un trait accidentel
RDP_EPSILON = 2.5  # pixels — tolérance de simplification (plus haut = plus lissé)


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
    """Ramer-Douglas-Peucker : retire les points qui ne s'écartent pas de
    plus de `epsilon` de la ligne droite entre les deux extrémités —
    élimine le tremblement tout en gardant la forme du trait."""
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
    """`strokes` : liste de Stroke (voir gesture_canvas_manipulation.py).
    Retire les micro-traits accidentels, simplifie le tracé de chacun des
    autres. Ne modifie pas les objets d'entrée (renvoie de nouveaux Stroke
    via `replace_points`, une fonction fournie par l'appelant pour éviter
    une dépendance circulaire sur la classe Stroke)."""
    kept = []
    for s in strokes:
        if stroke_path_length(s.points) < MIN_STROKE_LENGTH:
            continue  # trait accidentel (mini-tremblement, faux positif de détection) : ignoré
        new_points = clean_points(s.points)
        if len(new_points) >= 2:
            kept.append((s, new_points))
    return kept


# --------------------------------------------------------------------------
# Reconnaissance de formes géométriques — cv2.approxPolyDP classique,
# fiable. Renvoie (kind, polygone_propre) ou None. `polygone_propre` a des
# coordonnées EXACTES (cercle/rectangle ajustés), pas les points bruts.
# --------------------------------------------------------------------------
SHAPE_KINDS = {"circle", "triangle", "square", "rectangle", "pentagon", "hexagon"}

# --- Les quatre garde-fous, chacun contre un faux positif constaté ---------
#
# 1. MIN_SHAPE_POINTS. Il en fallait 8 — mais ce test tourne APRÈS le
#    nettoyage, qui ramène justement un carré propre à ses 5 points. Les
#    formes les mieux tracées étaient donc les plus souvent refusées.
#
# 2. MIN_SHAPE_SOLIDITY. Tout partait de l'ENVELOPPE CONVEXE du tracé, ce
#    qui efface exactement l'information qui distingue une forme simple
#    d'une lettre : ses creux. Un gribouillage en vagues revenait en
#    "hexagone", un B en cercle. On compare maintenant l'aire réelle à
#    celle de l'enveloppe : une forme franchement creuse n'est pas une
#    primitive, c'est un dessin, et elle le reste.
#
# 3. CIRCLE_FILL / CIRCLE_ROUNDNESS. Le test du cercle était la
#    "circularité" de l'enveloppe (4*pi*aire/perimetre^2), qui vaut plus de
#    0,82 pour à peu près tout ce qui est rond ET pour beaucoup de choses
#    qui ne le sont pas. On vérifie maintenant ce qu'un cercle est
#    vraiment : TOUS les points à la MÊME distance du centre. Un carré, un
#    B, une ellipse allongée échouent, un cercle un peu tremblé passe.
MIN_SHAPE_POINTS = 4
SHAPE_CLOSURE = 0.35        # écart départ/arrivée toléré, en fraction de la taille
MIN_SHAPE_AREA = 400.0      # px² — en-deçà, c'est un gribouillis
MIN_SHAPE_PERIMETER = 40.0
MIN_SHAPE_SOLIDITY = 0.86   # aire / aire de l'enveloppe convexe
CIRCLE_FILL = 0.88          # rayon moyen / rayon du cercle circonscrit
CIRCLE_ROUNDNESS = 0.10     # dispersion des rayons, rapportée au rayon moyen
POLYGON_EPSILON = 0.02      # tolérance d'approximation, en fraction du périmètre
MIN_POLYGON_SIDE = 0.07     # côté le plus court accepté, en fraction du périmètre
MAX_POLYGON_SIDES = 6       # au-delà, ce n'est plus un polygone : c'est un rond


def _shape_polygon(approx) -> list[tuple[float, float]]:
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def _resample_path(points, count: int = 96) -> np.ndarray:
    """Rééchantillonne un tracé en `count` points RÉGULIÈREMENT ESPACÉS le
    long du chemin.

    Indispensable, et pour une raison qui ne saute pas aux yeux : le
    nettoyage réduit un carré propre à ses quatre coins. Or les quatre coins
    d'un carré sont tous à la même distance du centre — exactement comme un
    cercle. Mesurer les rayons sur les points bruts faisait donc passer tous
    les polygones réguliers pour des cercles. En repassant par le CHEMIN, on
    retrouve les milieux de côtés, qui eux sont bien plus près du centre."""
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


def _is_round(path: np.ndarray) -> tuple[tuple[float, float], float] | None:
    """(centre, rayon) si le tracé est un cercle, sinon None. La définition
    même d'un cercle : tous les points à la même distance du centre."""
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


def _polygon_sides_are_real(approx, peri: float) -> bool:
    """Refuse les polygones dont un côté est minuscule : ce sont des faux
    sommets créés par le bruit, pas des angles voulus. Sans ce filtre, un
    rond un peu cabossé décrochait un sommet de plus et devenait un
    pentagone."""
    pts = np.array([p[0] for p in approx], dtype=np.float64)
    sides = np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)
    return bool(sides.min() >= MIN_POLYGON_SIDE * peri)


def classify_shape(points: list[tuple[int, int]]) -> tuple[str, list[tuple[float, float]]] | None:
    """(nom, polygone propre) ou None. None est une réponse NORMALE : mieux
    vaut garder le dessin de l'utilisateur que lui imposer un cercle."""
    if len(points) < MIN_SHAPE_POINTS:
        return None
    pts = np.array(points, dtype=np.float32)

    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    if float(np.linalg.norm(pts[0] - pts[-1])) > SHAPE_CLOSURE * span:
        return None  # pas refermé sur lui-même -> pas une primitive géométrique

    contour = pts.astype(np.int32)
    area = abs(float(cv2.contourArea(contour)))
    if area < MIN_SHAPE_AREA:
        return None

    hull = cv2.convexHull(contour)
    hull_area = abs(float(cv2.contourArea(hull)))
    peri = float(cv2.arcLength(hull, True))
    if hull_area <= 0.0 or peri < MIN_SHAPE_PERIMETER:
        return None
    if area / hull_area < MIN_SHAPE_SOLIDITY:
        return None  # forme creuse (une lettre, un gribouillage) : on n'y touche pas

    # LE POLYGONE D'ABORD, LE CERCLE ENSUITE. C'est l'ordre qui compte :
    # un polygone régulier a, comme un cercle, tous ses SOMMETS à la même
    # distance du centre — les tester dans l'autre sens rendait tout rond.
    # Un cercle, lui, ne s'approxime jamais en six côtés ou moins.
    approx = cv2.approxPolyDP(hull, POLYGON_EPSILON * peri, True)
    n = len(approx)
    if 3 <= n <= MAX_POLYGON_SIDES and _polygon_sides_are_real(approx, peri):
        if n == 3:
            return "triangle", _shape_polygon(approx)
        if n == 4:
            return _classify_quad(pts)
        if n == 5:
            return "pentagon", _shape_polygon(approx)
        return "hexagon", _shape_polygon(approx)

    round_fit = _is_round(_resample_path(points))
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


# --------------------------------------------------------------------------
# Écriture — Tesseract OCR (optionnel : `pip install pytesseract` + binaire
# système `tesseract-ocr`). S'il n'est pas installé, renvoie toujours None
# (dégrade proprement) — mais voir `diagnose_text_recognition()` plus bas :
# ça devenait un échec SILENCIEUX, sans aucun indice pour savoir pourquoi.
#
# Alternative testée et ABANDONNÉE : une comparaison de formes 100%
# hors-ligne (cv2.matchShapes, puis comparaison de bitmaps) pour éviter la
# dépendance à Tesseract. Résultat mesuré sur les 36 caractères A-Z/0-9,
# même avec un tracé propre : 8/36 corrects avec matchShapes, 28/36 avec
# une comparaison de bitmaps — pas assez fiable pour être honnête à livrer
# (une lettre reconnue à tort, avec confiance, est pire qu'une lettre non
# reconnue). Tesseract, quand il est disponible, est nettement plus robuste
# (95%+ de confiance testé, même avec de vrais trous dans les traits) — la
# vraie faiblesse n'est pas l'algorithme, c'est sa disponibilité.
# --------------------------------------------------------------------------
_tesseract_ok: bool | None = None


def tesseract_available() -> bool:
    """Vrai si le VRAI moteur OCR est joignable. Le résultat est mis en
    cache : sans ça, chaque validation d'élément tentait de lancer le
    binaire, ce qui coûte un processus (et une exception) à chaque fois
    sur une machine où il n'est pas installé."""
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
    """Message humain à afficher au démarrage ET dans le HUD — pour ne
    plus jamais avoir un échec silencieux sur cette fonctionnalité."""
    try:
        import pytesseract
    except ImportError:
        return "ECRITURE: desactivee (pip install pytesseract manquant)"
    try:
        version = pytesseract.get_tesseract_version()
        return f"ECRITURE: active (Tesseract {version})"
    except Exception:
        return "ECRITURE: desactivee (pytesseract est installe mais le binaire Tesseract est introuvable sur le PATH)"


# Un tracé doit RESSEMBLER à de l'écriture avant qu'on demande son avis à
# un moteur d'OCR. Sans ce garde-fou, Tesseract répond quelque chose à peu
# près toujours : mesuré, un simple trait diagonal de 20 px revenait en
# "N" avec 67 % de confiance, et le dessin était remplacé par la lettre.
# Un faux positif est bien pire qu'une abstention — le trait de
# l'utilisateur DISPARAÎT au profit d'une lettre qu'il n'a jamais écrite.
MIN_WRITING_SPAN = 25.0     # px : en-deçà, c'est un gribouillis, pas une lettre
MAX_WRITING_STROKES = 12    # au-delà, c'est un dessin, pas un mot
MIN_TEXT_CONFIDENCE = 70.0  # un vrai texte net sort à 85-95 ; le bruit, à 30-65


def looks_like_writing(strokes_points: list[list[tuple[int, int]]]) -> bool:
    """Filtre d'entrée de la reconnaissance d'écriture. Quatre critères,
    tous vérifiables à l'œil sur un contre-exemple :

      * un nombre de traits plausible (un mot, pas un gribouillage) ;
      * une taille minimale (un trait de 20 px n'est pas une lettre) ;
      * une silhouette pas absurde (un fil horizontal de 300x4 px non plus) ;
      * du RELIEF : un trait unique et parfaitement droit est une ligne,
        pas une écriture. On le mesure en comparant la longueur parcourue à
        la diagonale de la boîte englobante — un segment droit donne 1,0,
        une lettre monte facilement au-dessus de 1,35."""
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
    img = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)  # encre blanche sur fond noir d'abord...

    # Épaisseur du trait PROPORTIONNELLE à la hauteur du dessin, pas fixe.
    # Bug réel mesuré : avec une épaisseur fixe (l'ancien `max(width, 4)`,
    # 4px = la largeur d'affichage à l'écran), la précision Tesseract
    # s'effondrait de 32/36 caractères corrects à ~60px de hauteur jusqu'à
    # 8/36 à ~220px — un trait de 4px devient un fil trop fin par rapport à
    # une lettre dessinée en grand à la main. Testé sur A-Z/0-9 de 40px à
    # 400px de haut : un ratio de 0.15 (épaisseur = 15% de la hauteur) donne
    # une précision constante (28-34/36) sur toute la plage, au lieu de
    # s'effondrer avec la taille. La largeur d'affichage à l'écran (`width`,
    # dans `widths`) n'est PAS utilisée ici : elle sert au rendu en direct,
    # pas à ce que Tesseract voit.
    ocr_thickness = int(clamp(h * 0.15, 3, 30))
    for pts, _ in zip(strokes_points, widths):
        shifted = [(int(px - x0 + pad), int(py - y0 + pad)) for px, py in pts]
        for p1, p2 in zip(shifted, shifted[1:]):
            cv2.line(img, p1, p2, 255, ocr_thickness, cv2.LINE_AA)

    # ...puis une dilatation légère : comble les petits trous qu'un frame
    # raté pendant le tracé peut laisser dans un trait, sans épaissir au
    # point de faire fusionner des lettres proches (testé : une fermeture
    # morphologique complète était trop agressive et cassait la lecture).
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = 255 - img  # ...enfin encre noire sur fond blanc, ce que Tesseract attend

    # image_to_data (pas image_to_string) pour avoir un score de confiance :
    # un gribouillage géométrique peut ressembler à du texte pour Tesseract
    # (testé : un simple zigzag sinusoïdal se lit "VAVAVAV"...) mais avec
    # une confiance nettement plus basse (~30) qu'un vrai texte net (~85-95)
    # — le seuil ci-dessous sert de garde-fou contre ces faux positifs. Seuil
    # abaissé (30, était 45) : un tracé au doigt est nettement moins net
    # qu'un rendu propre, mieux vaut accepter plus large et se fier au
    # diagnostic ci-dessous pour ajuster si des faux positifs apparaissent.
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

    # Diagnostic imprimé à CHAQUE tentative, acceptée ou non — pour savoir
    # exactement quoi ajuster (le seuil ? le prétraitement ?) sans deviner.
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


# --------------------------------------------------------------------------
# Reconnaissance de LETTRE hors-ligne (demande n°9) — filet de sécurité quand
# Tesseract n'est pas installé.
#
# Pourquoi c'était nécessaire : l'écriture ne marchait QUE si l'utilisateur
# avait installé un binaire système en plus du paquet pip. Sur une machine
# sans Tesseract, la fonctionnalité était simplement absente. Ici, plus de
# dépendance externe : on compare le tracé à des gabarits A-Z/0-9 rendus
# avec les polices intégrées d'OpenCV.
#
# Méthode : distance de CHAMFER symétrique entre le tracé et chaque gabarit,
# après normalisation (recadrage, mise à l'échelle en conservant le rapport
# largeur/hauteur, centrage). Le chamfer compare des LIGNES à des LIGNES
# via une carte de distance, donc il tolère qu'un trait à main levée passe
# à quelques pixels du gabarit — là où une comparaison pixel à pixel (XOR)
# punit le moindre décalage.
#
# Honnêteté de portée, mesurée (voir test_new_features.py) : ~80-90% sur des
# CAPITALES isolées tracées proprement, contre 95%+ pour Tesseract. Donc
# Tesseract reste prioritaire quand il est là ; ceci prend le relais sinon.
# Une seule lettre à la fois — c'est le cas d'usage demandé ("j'écris une
# lettre"), et un mot entier passe par Tesseract.
# --------------------------------------------------------------------------
LETTER_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_NORM_BOX = 64          # côté du canevas normalisé
_NORM_INK = 52          # côté de l'encre à l'intérieur (marge = tolérance de centrage)
_NORM_THICKNESS = 3     # épaisseur commune tracé/gabarit, pour comparer des lignes comparables
_TEMPLATE_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
)

_LETTER_MAX_CHAMFER = 4.2   # au-delà : le tracé ne ressemble à AUCUN caractère
_LETTER_MIN_MARGIN = 0.28   # écart minimal avec le 2e meilleur candidat (sinon c'est un pile ou face)

_template_cache: dict | None = None


def _normalize_ink(bitmap: np.ndarray) -> np.ndarray | None:
    """Recadre sur l'encre, met à l'échelle en CONSERVANT le rapport
    largeur/hauteur, puis centre dans un canevas carré. Conserver le rapport
    est essentiel : c'est ce qui distingue un "I" (haut et fin) d'un "O"
    (carré) — les étirer tous les deux en carré les rendrait identiques."""
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
    """Réduit une forme pleine à sa ligne centrale — pour comparer un
    gabarit de police (lettre PLEINE) à un tracé au doigt (une LIGNE). Sans
    ça, on comparerait le contour d'une lettre épaisse à un trait fin, ce
    qui fausse tout. `ximgproc` fait partie d'opencv-contrib ; s'il manque,
    on retombe sur une érosion morphologique (moins net mais utilisable)."""
    try:
        return cv2.ximgproc.thinning(bitmap)
    except Exception:
        eroded = cv2.erode(bitmap, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.subtract(bitmap, eroded)


def _thicken(bitmap: np.ndarray, thickness: int = _NORM_THICKNESS) -> np.ndarray:
    k = max(1, thickness)
    return cv2.dilate(bitmap, np.ones((k, k), np.uint8), iterations=1)


def _build_template_bank() -> dict:
    """Un gabarit normalisé (+ sa carte de distance) par caractère et par
    police. Plusieurs polices parce qu'un "A" à empattements et un "A"
    bâton n'ont pas la même silhouette : on garde le meilleur des deux."""
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
    """Distance moyenne des pixels d'encre de A au trait le plus proche de
    B, et réciproquement. Symétrique pour qu'un tracé incomplet et un tracé
    débordant soient pénalisés de la même façon."""
    a_ink = a_bitmap > 0
    b_ink = b_bitmap > 0
    if not a_ink.any() or not b_ink.any():
        return 1e9
    return 0.5 * (float(b_dist[a_ink].mean()) + float(a_dist[b_ink].mean()))


def strokes_to_bitmap(strokes_points: list[list[tuple[int, int]]], thickness: int = _NORM_THICKNESS) -> np.ndarray | None:
    """Rend les traits (coordonnées écran) dans une image binaire serrée."""
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
    """Garde-fou AVANT de tenter la reconnaissance : sans lui, tout
    gribouillage finirait transformé en une lettre au hasard, ce qui est
    pire que ne rien reconnaître. Trois critères simples et vérifiables :
    peu de traits, taille suffisante, silhouette pas absurdement allongée
    (un trait horizontal de 300x8 px n'est pas une lettre)."""
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
    """Redresse l'inclinaison du tracé à partir de ses moments d'ordre 2.

    Pourquoi : mesuré sur des lettres penchées (une écriture manuscrite
    l'est presque toujours un peu), la précision tombait de ~88% à ~75% —
    les gabarits de police, eux, sont parfaitement droits. Redresser avant
    de comparer récupère l'essentiel de cet écart, pour 5 lignes de code et
    aucun modèle à entraîner."""
    m = cv2.moments(bitmap, binaryImage=True)
    if abs(m["mu02"]) < 1e-2:
        return bitmap
    skew = m["mu11"] / m["mu02"]
    if abs(skew) < 0.02 or abs(skew) > 1.0:
        return bitmap  # rien à corriger, ou mesure aberrante qu'on ne veut pas amplifier
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
    """Renvoie (caractère, distance) ou None. Plus la distance est PETITE,
    meilleure est la correspondance."""
    if not looks_like_a_letter(strokes_points):
        return None
    raw = strokes_to_bitmap(strokes_points)
    if raw is None:
        return None

    # Deux variantes du même tracé : tel quel, et redressé. On garde le
    # meilleur score des deux — un tracé déjà droit n'est pas pénalisé, un
    # tracé penché est rattrapé.
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
        # Deux candidats trop proches (O/0, I/1...) : on préfère ne rien
        # affirmer plutôt que d'imposer une lettre au hasard.
        return None
    return best_char, best_score


OFFLINE_MAX_LETTERS = 6      # au-delà, on n'essaie pas : trop de lettres, trop d'occasions de se tromper
OFFLINE_LETTER_GAP = 0.18    # trou horizontal, en fraction de la hauteur d'encre, qui sépare deux lettres


def split_into_letters(strokes_points: list[list[tuple[int, int]]]) -> list[list[list[tuple[int, int]]]]:
    """Découpe un tracé en groupes de traits — un groupe par lettre — sur
    les TROUS HORIZONTAUX.

    C'est ce qui permet d'écrire un MOT sans Tesseract : les gabarits
    hors-ligne ne savent lire qu'un caractère à la fois. Le découpage se
    fait sur l'axe X parce que c'est ainsi qu'on écrit — les traits d'une
    même lettre se chevauchent horizontalement (la barre du T couvre sa
    hampe), ceux de deux lettres voisines ne se chevauchent pas."""
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
    """Un mot court, sans aucune dépendance externe : on découpe en
    lettres, on reconnaît chacune, on recolle.

    Règle d'honnêteté : si UNE seule lettre n'est pas reconnue avec
    certitude, on renvoie None pour le mot entier. Un mot à moitié deviné
    ("SXLUT") serait pire qu'un dessin gardé tel quel — l'utilisateur
    verrait son écriture remplacée par du charabia."""
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


def text_to_strokes(text: str, target_width: float, target_height: float) -> list[list[tuple[int, int]]]:
    """Rend `text` avec une police cv2 FINE et lisible, puis extrait ses
    contours comme des traits — le même pipeline que pour n'importe quel
    dessin, ce qui fait que le texte reconnu profite de la 3D et du GRAB
    comme le reste. FONT_HERSHEY_SIMPLEX + épaisseur réduite (2, au lieu
    de DUPLEX/4) pour un alias fin plutôt que gras."""
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

    # Les contours sont RECADRÉS sur leur propre encre avant d'être mis à
    # l'échelle. Bug réel : en partant des coordonnées brutes du canevas,
    # la marge de rendu (20 px) et la ligne de base de la police étaient
    # comprises dans la mise à l'échelle — la lettre reconnue apparaissait
    # décalée de plusieurs pixels en bas à droite de l'endroit où elle
    # avait été écrite. Recadrer d'abord la fait tomber pile dessus.
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


# --------------------------------------------------------------------------
# Dispatcher — priorité : forme (fiable) -> écriture -> gardé tel quel
# nettoyé. (Une détection de "fleur" existait ici mais a été retirée : son
# critère "boucles groupées" confondait un MOT à plusieurs lettres fermées
# avec une fleur, empêchant l'écriture d'être reconnue — plus simple et
# plus sûr sans.)
# --------------------------------------------------------------------------
def classify_shape_only(
    strokes_points: list[list[tuple[int, int]]],
) -> tuple[str, list[list[tuple[int, int]]]] | None:
    """La moitié INSTANTANÉE de la reconnaissance : une forme géométrique,
    ou rien. Quelques dizaines de microsecondes, donc utilisable dans la
    boucle vidéo sans y faire le moindre trou.

    Séparée de l'écriture parce que l'écriture, elle, coûte 300 ms
    (Tesseract lance un processus) — le temps de dix images. L'application
    fait donc la forme tout de suite et confie l'écriture à un fil de
    fond ; voir `async_recognition.py`."""
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
    """La moitié LENTE : lire un mot ou une lettre, et le redessiner à
    l'endroit EXACT où il a été tracé. Renvoie (texte, traits) ou None.

    Ordre : Tesseract s'il est là (meilleur, gère les mots) -> gabarits
    hors-ligne (aucune installation requise) -> abstention. L'abstention
    est un résultat normal, pas un échec : mieux vaut garder le dessin de
    l'utilisateur que lui imposer une lettre qu'il n'a pas écrite."""
    if not looks_like_writing(strokes_points):
        return None

    text = try_recognize_text(strokes_points, widths) if allow_tesseract else None

    if not text:
        # Filet de sécurité sans dépendance externe : une lettre, ou un mot
        # court découpé lettre par lettre.
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
    """Reconnaissance COMPLÈTE et synchrone : forme -> écriture -> tel quel.
    Renvoie (kind, traits à afficher, texte reconnu ou None).

    L'application, elle, appelle les deux moitiés séparément pour ne pas
    bloquer la vidéo pendant l'OCR ; cette fonction reste le chemin simple
    (et la référence des tests)."""
    shape = classify_shape_only(strokes_points)
    if shape is not None:
        return shape[0], shape[1], None

    written = recognize_text(strokes_points, widths, allow_tesseract)
    if written is not None:
        return "text", written[1], written[0]

    return "freeform", strokes_points, None


def classify(strokes_points: list[list[tuple[int, int]]], widths: list[int]) -> tuple[str, list[list[tuple[int, int]]]]:
    """Renvoie (kind, traits_a_afficher). `kind` in SHAPE_KINDS | "text" |
    "freeform". `traits_a_afficher` remplace les traits nettoyés d'origine
    si une reconnaissance a eu lieu, sinon les renvoie tels quels."""
    kind, strokes, _text = classify_ex(strokes_points, widths)
    return kind, strokes
