"""
hud.py — Tout ce qui s'affiche PAR-DESSUS l'image.

Pourquoi c'est un fichier à part : ces fonctions ne décident de rien. Elles
lisent l'état et le peignent. Les mélanger à la logique de gestes rendait
impossible de répondre à « pourquoi ce geste n'a pas marché ? » sans relire
mille lignes.

CE QU'ON MONTRE, ET SURTOUT CE QU'ON NE MONTRE PAS
==================================================
L'écran affichait le nombre d'images par seconde, l'état interne, le nombre
de traits en attente, le moteur d'OCR et sa version, et une étiquette
numérotée sur chaque élément. C'est un tableau de bord de développeur : ça
raconte la machinerie au lieu de laisser voir le dessin.

Il ne reste que trois choses :

    la PALETTE ......... la couleur courante, et les sept autres
    un MOT d'action .... seulement quand il se passe quelque chose
    les JAUGES ......... avant un geste destructeur, pour pouvoir l'annuler

Tout le reste est visuel : le cadre vert dit la sélection, la comète dit le
tracé, la lettre qui apparaît dit que l'écriture a été lue. Un dessin n'a
pas besoin d'être commenté.

Les diagnostics n'ont pas disparu pour autant — ils sont derrière la touche
`d`. On ne perd rien, on arrête juste de l'imposer.

TOUT EST À L'ÉCHELLE
====================
Les positions et les tailles de police sont multipliées par `h / 540`. Sans
ça, passer la caméra en 1280x720 laissait un HUD minuscule dans un coin.

LES TEXTES SONT EN ANGLAIS
==========================
Le code et ses commentaires restent en français — c'est la langue dans
laquelle ce projet se pense. L'écran, lui, parle anglais : c'est ce qui se
montre.
"""
from __future__ import annotations

import cv2
import numpy as np

import canvas as canvas_module

FONT = cv2.FONT_HERSHEY_SIMPLEX
REFERENCE_HEIGHT = 540.0   # hauteur pour laquelle les tailles ci-dessous sont écrites

HELP_LINES = [
    "Index finger, held ......... draw   (pause -> becomes an object)",
    "Closed fist ................ move the object",
    "Open hand OVER it .......... rotate it in 3D",
    "Move your hand AWAY ........ deselect",
    "Circle sign, 3 fingers ..... zoom  --  with an object selected:",
    "      raise the 3 fingers = BIGGER, lower them = SMALLER",
    "Thumb UP ................... next colour",
    "Thumb DOWN ................. delete the last drawing",
    "V sign ..................... undo",
    "BOTH HANDS open, then REMOVE them ... clear everything",
    "q quit    h help    o shadows    c colour    z undo    s save",
]

ACTION_DRAWING = "drawing"
ACTION_READING = "reading..."

WHITE = (255, 255, 255)
DIM = (200, 200, 200)
ACCENT = (80, 255, 255)
WARN = (80, 80, 255)


def _scale(image: np.ndarray) -> float:
    return max(image.shape[0] / REFERENCE_HEIGHT, 0.6)


def apply_shadow(frame: np.ndarray, shadow: tuple[np.ndarray, tuple[int, int, int, int]]) -> None:
    """Assombrit la vidéo là où tombent les ombres, SUR PLACE et seulement
    dans la zone concernée.

    L'ombre doit s'appliquer à la vidéo AVANT que les éléments soient
    ajoutés : le canevas est composé par addition (`addWeighted`), et une
    addition ne peut pas assombrir. D'où ces deux étapes séparées."""
    mask, (x0, y0, x1, y1) = shadow
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    alpha = (mask[y0:y1, x0:x1].astype(np.float32) / 255.0 * canvas_module.SHADOW_STRENGTH)[..., None]
    frame[y0:y1, x0:x1] = (region.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def draw_trail(image: np.ndarray, trail: list, color: tuple[int, int, int]) -> None:
    """Comète derrière le doigt : les segments récents sont épais et
    lumineux, les anciens fins et sombres.

    Le dégradé est obtenu en assombrissant la COULEUR, pas avec de la
    transparence : mélanger un calque entier coûterait un passage sur toute
    l'image à chaque frame, pour un effet qui ne touche qu'une centaine de
    pixels."""
    if len(trail) < 2:
        return
    s = _scale(image)
    n = len(trail)
    for i in range(n - 1):
        t = (i + 1) / n              # 0 = le plus ancien, 1 = le plus récent
        faded = tuple(int(c * (0.15 + 0.85 * t)) for c in color)
        thickness = max(1, int(round((1 + 6 * t) * s)))
        p1 = (int(trail[i][0]), int(trail[i][1]))
        p2 = (int(trail[i + 1][0]), int(trail[i + 1][1]))
        cv2.line(image, p1, p2, faded, thickness, cv2.LINE_AA)
    head = (int(trail[-1][0]), int(trail[-1][1]))
    cv2.circle(image, head, max(3, int(7 * s)), color, -1, cv2.LINE_AA)
    cv2.circle(image, head, max(5, int(10 * s)), color, max(1, int(s)), cv2.LINE_AA)


def _panel(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.55) -> None:
    """Fond sombre translucide. Un texte posé directement sur la vidéo
    devient illisible dès que la scène est claire ; ce fond garantit le
    contraste quelle que soit la pièce."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    region = image[y0:y1, x0:x1]
    image[y0:y1, x0:x1] = (region.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def draw_palette(output: np.ndarray, current_idx: int) -> None:
    """La palette entière, la couleur courante mise en avant.

    Avant, une seule pastille montrait la couleur du moment : on voyait bien
    qu'elle changeait, mais jamais COMBIEN de gestes séparaient du bleu.
    Avec les huit cases visibles, choisir une couleur devient un calcul
    évident au lieu d'une loterie."""
    palette = canvas_module.COLOR_PALETTE
    s = _scale(output)
    box, gap, margin = int(26 * s), int(5 * s), int(22 * s)
    total = len(palette) * box + (len(palette) - 1) * gap
    x, y = output.shape[1] - margin - total, margin

    for i, color in enumerate(palette):
        x0 = x + i * (box + gap)
        cv2.rectangle(output, (x0, y), (x0 + box, y + box), color, -1, cv2.LINE_AA)
        if i == current_idx % len(palette):
            pad = max(2, int(3 * s))
            cv2.rectangle(output, (x0 - pad, y - pad), (x0 + box + pad, y + box + pad),
                          WHITE, max(2, int(2 * s)), cv2.LINE_AA)
        else:
            cv2.rectangle(output, (x0, y), (x0 + box, y + box), (70, 70, 70), 1, cv2.LINE_AA)


def current_action(state, fr) -> str:
    """Le mot à afficher, ou "" quand il ne se passe rien.

    Déduit de l'ÉTAT et non d'un message : les messages internes sont longs,
    détaillés, et faits pour le débogage. Ici on ne veut qu'un mot, et
    seulement quand il a lieu d'être."""
    if fr.drawing:
        return ACTION_DRAWING
    if fr.resizing:
        return "resizing"
    if state.grab is not None:
        return "rotating" if state.grab.mode == "rotate" else "moving"
    if fr.lens is not None:
        return "zoom"
    recognizer = getattr(state, "recognizer", None)
    if recognizer is not None and recognizer.pending:
        return ACTION_READING
    return ""


def draw_action(output: np.ndarray, action: str, color: tuple[int, int, int]) -> None:
    """Un mot, en haut à gauche, seulement quand il se passe quelque chose.
    Un point de la couleur courante l'accompagne : c'est le lien visuel
    entre ce qu'on fait et ce avec quoi on le fait."""
    if not action:
        return
    s = _scale(output)
    margin = int(22 * s)
    font_scale, thickness = 0.62 * s, max(1, int(round(1.6 * s)))
    (tw, th), _ = cv2.getTextSize(action, FONT, font_scale, thickness)
    dot, pad = int(6 * s), int(12 * s)
    x0, y0 = margin, margin
    x1, y1 = margin + pad * 2 + dot * 3 + tw, margin + th + pad

    _panel(output, x0, y0, x1, y1, alpha=0.45)
    cv2.circle(output, (x0 + pad + dot, (y0 + y1) // 2), dot, color, -1, cv2.LINE_AA)
    cv2.putText(output, action, (x0 + pad + dot * 3, y1 - pad // 2), FONT, font_scale, WHITE,
                thickness, cv2.LINE_AA)


CLEAR_LABELS = {
    "arm": ("BOTH HANDS DETECTED", DIM, "keep them up to arm"),
    "ready": ("NOW REMOVE BOTH HANDS", ACCENT, "take your hands out of view"),
    "confirm": ("CLEARING EVERYTHING", WARN, "show a hand to cancel"),
}


def draw_clear_gauge(output: np.ndarray, phase: str, progress: float) -> None:
    """Le geste « tout effacer », au CENTRE de l'écran.

    Il était en haut à gauche, mêlé au reste. Un geste irréversible mérite
    d'être là où le regard est déjà : au milieu de ce qu'on est sur le point
    de perdre. Et il annonce à chaque étape ce qu'il attend, pour que
    personne n'ait à deviner s'il faut continuer ou s'arrêter."""
    if phase not in CLEAR_LABELS:
        return
    label, color, hint = CLEAR_LABELS[phase]
    s = _scale(output)
    h, w = output.shape[:2]
    bar_w, bar_h = int(340 * s), int(14 * s)
    cx, cy = w // 2, int(h * 0.42)

    weight = max(1, int(round(2 * s)))
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.72 * s, weight)
    half = max(bar_w, tw) // 2 + int(28 * s)   # le fond suit le plus large des deux
    _panel(output, cx - half, cy - int(58 * s), cx + half, cy + int(48 * s), alpha=0.62)
    cv2.putText(output, label, (cx - tw // 2, cy - int(24 * s)), FONT, 0.72 * s, color, weight, cv2.LINE_AA)

    x0, y0 = cx - bar_w // 2, cy
    cv2.rectangle(output, (x0, y0), (x0 + bar_w, y0 + bar_h), (55, 55, 55), -1)
    filled = int(bar_w * max(0.0, min(1.0, progress)))
    if filled > 0:
        cv2.rectangle(output, (x0, y0), (x0 + filled, y0 + bar_h), color, -1)
    cv2.rectangle(output, (x0, y0), (x0 + bar_w, y0 + bar_h), (210, 210, 210), 1)

    (hw, _), _ = cv2.getTextSize(hint, FONT, 0.5 * s, 1)
    cv2.putText(output, hint, (cx - hw // 2, y0 + int(34 * s)), FONT, 0.5 * s, DIM, 1, cv2.LINE_AA)


def draw_help(output: np.ndarray) -> None:
    """Le panneau est dimensionné sur le TEXTE MESURÉ, pas sur une largeur
    écrite à la main : celle-ci finit toujours par être fausse dès qu'on
    ajoute un geste ou qu'on change de langue, et les lignes débordent du
    fond sombre — donc deviennent illisibles sur une scène claire."""
    s = _scale(output)
    h = output.shape[0]
    font_scale = 0.45 * s
    line_h, margin = int(21 * s), int(18 * s)
    width = max(cv2.getTextSize(line, FONT, font_scale, 1)[0][0] for line in HELP_LINES)
    top = h - margin - line_h * len(HELP_LINES)
    _panel(output, margin - int(10 * s), top - line_h,
           margin + width + int(14 * s), h - margin + int(6 * s), alpha=0.62)
    y = top
    for line in HELP_LINES:
        cv2.putText(output, line, (margin, y), FONT, font_scale, (225, 225, 225), 1, cv2.LINE_AA)
        y += line_h


def draw_toast(output: np.ndarray, message: str) -> None:
    """Message court et centré, pour les événements ponctuels (sauvegarde,
    annulation). Rares, mais on ne doit jamais se demander s'ils ont eu
    lieu."""
    s = _scale(output)
    h, w = output.shape[:2]
    font_scale, thickness = 0.66 * s, max(1, int(round(1.6 * s)))
    (tw, th), _ = cv2.getTextSize(message, FONT, font_scale, thickness)
    x, y = (w - tw) // 2, h - int(52 * s)
    pad = int(14 * s)
    _panel(output, x - pad, y - th - pad, x + tw + pad, y + pad, alpha=0.6)
    cv2.putText(output, message, (x, y), FONT, font_scale, WHITE, thickness, cv2.LINE_AA)


def draw_hud(output: np.ndarray, state, fr, show_help: bool) -> None:
    """L'affichage normal : la palette, un mot s'il se passe quelque chose,
    la jauge d'effacement s'il y a lieu, l'aide si elle est demandée."""
    draw_palette(output, state.pen_color_idx)
    draw_action(output, current_action(state, fr), state.pen_color)
    draw_clear_gauge(output, fr.clear_phase, fr.clear_progress)
    if show_help:
        draw_help(output)


def draw_debug(output: np.ndarray, state, fr, fps: float, engine_status: str) -> None:
    """La vue de développeur, derrière la touche `d`. Tout ce qui a été
    retiré de l'affichage normal est ici, et rien n'a été perdu : quand un
    geste se comporte mal, c'est cette vue qui dit pourquoi."""
    s = _scale(output)
    lines = [
        f"fps {fps:.1f}",
        f"state {state.last_status}",
        f"objects {len(state.canvas.objects)}   pending {len(state.canvas.pending_strokes)}",
        engine_status,
        f"snap {'armed' if fr.snap_armed else '-'}   clear {fr.clear_phase} {fr.clear_progress:.2f}",
    ]
    line_h = int(20 * s)
    top = output.shape[0] // 2
    _panel(output, int(12 * s), top - line_h, int(580 * s), top + line_h * len(lines), alpha=0.55)
    y = top
    for line in lines:
        cv2.putText(output, line, (int(20 * s), y), FONT, 0.48 * s, (150, 230, 150), 1, cv2.LINE_AA)
        y += line_h
