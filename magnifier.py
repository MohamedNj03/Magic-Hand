"""
magnifier.py — La LOUPE : le nouveau zoom, à une main.

CE QUI REMPLACE QUOI
====================
L'ancien zoom (deux poings qui s'écartent) déformait TOUTE l'image. Deux
problèmes de fond, en plus d'exiger une deuxième main :

  * il fallait garder en tête que l'écran ne montrait plus le monde à
    l'échelle 1:1, donc dessiner ou saisir en zoomant devenait un exercice
    de correspondance mentale ;
  * une fois le geste fini, la vue restait déformée, et il fallait un autre
    geste pour revenir.

La loupe fait l'inverse : le monde reste à l'échelle 1:1 en permanence, et
seul un DISQUE suit la main pour grossir ce qu'il y a dessous. Le geste dit
tout, tout seul : on forme un cercle avec le pouce et l'index, on le
promène là où on veut regarder, et on le défait quand on a fini. Rien à
annuler, rien à retenir, et aucun risque de désaccord entre ce qu'on voit
et ce que le suivi de main calcule — puisqu'il n'y a plus de déformation.

LE DÉTAIL QUI REND ÇA NATUREL
=============================
Le grossissement est piloté par l'OUVERTURE du cercle, mais À L'ENVERS de
ce à quoi on pense d'abord : un petit cercle grossit BEAUCOUP, un grand
cercle grossit peu. C'est le comportement d'une vraie lentille (plus elle
est petite et bombée, plus elle grossit), et c'est aussi ce qui marche le
mieux à l'usage : on écarte les doigts pour balayer une large zone, on les
resserre pour inspecter un détail.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MIN_ZOOM = 1.2
MAX_ZOOM = 4.5
MIN_RADIUS_PX = 55.0     # en-dessous, le disque est trop petit pour qu'on y voie quoi que ce soit
MAX_RADIUS_PX = 240.0    # au-dessus, il mange l'écran et on perd le contexte autour

# Bornes d'ouverture (en unités "taille de main") entre lesquelles le
# grossissement varie. En-dehors, il sature — pas de discontinuité.
APERTURE_TIGHT = 0.25    # doigts presque joints -> grossissement maximal
APERTURE_WIDE = 1.10     # cercle grand ouvert -> grossissement minimal

RING_COLOR = (235, 235, 235)
GLINT_COLOR = (255, 255, 255)


def zoom_for_aperture(aperture: float) -> float:
    """Ouverture du cercle -> facteur de grossissement, en INVERSE (petit
    cercle = fort grossissement). Interpolation linéaire entre les deux
    bornes, saturée au-delà : la sensation reste continue même quand on
    dépasse largement d'un côté ou de l'autre."""
    t = (aperture - APERTURE_TIGHT) / max(APERTURE_WIDE - APERTURE_TIGHT, 1e-6)
    t = max(0.0, min(1.0, t))
    return MAX_ZOOM + (MIN_ZOOM - MAX_ZOOM) * t


@dataclass
class MagnifierState:
    """Ce que la loupe affiche à cet instant. Les valeurs sont LISSÉES par
    l'appelant avant d'arriver ici : sans lissage, le disque tremble avec
    les landmarks et donne mal au cœur."""
    center: tuple[float, float]
    radius: float
    zoom: float

    @property
    def clamped_radius(self) -> float:
        return max(MIN_RADIUS_PX, min(MAX_RADIUS_PX, self.radius))


def draw_magnifier(image: np.ndarray, state: MagnifierState) -> None:
    """Dessine le disque grossissant DANS `image`, sur place.

    Fonctionnement : on prélève autour du centre un carré `zoom` fois plus
    petit que le disque, on l'agrandit à la taille du disque, et on ne
    recolle que la partie circulaire. Le reste de l'image n'est pas touché —
    c'est ce qui garde le monde à l'échelle 1:1 en dehors de la loupe."""
    h, w = image.shape[:2]
    radius = int(round(state.clamped_radius))
    zoom = max(1.01, state.zoom)
    cx, cy = int(round(state.center[0])), int(round(state.center[1]))

    half_src = max(2, int(round(radius / zoom)))
    x0, y0 = cx - half_src, cy - half_src
    x1, y1 = cx + half_src, cy + half_src

    # Prélèvement borné à l'image, puis re-bordé : sans ça, une loupe
    # promenée au bord de l'écran planterait ou afficherait n'importe quoi.
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

    # La monture : un anneau net, un halo plus sombre à l'extérieur pour
    # décoller le disque du fond, et un petit reflet en haut à gauche — les
    # trois détails qui font lire l'objet comme une loupe et pas comme un
    # bug d'affichage.
    cv2.circle(image, (cx, cy), radius, (35, 35, 35), 6, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, RING_COLOR, 2, cv2.LINE_AA)
    glint_offset = int(radius * 0.55)
    cv2.ellipse(image, (cx - glint_offset, cy - glint_offset), (int(radius * 0.22), int(radius * 0.10)),
                -40, 0, 360, GLINT_COLOR, -1, cv2.LINE_AA)
    cv2.putText(image, f"x{zoom:.1f}", (cx - 18, cy + radius + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RING_COLOR, 2, cv2.LINE_AA)
