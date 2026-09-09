"""
filters.py — Lissage et petites fonctions numériques partagées.

Tout ce qui est ici répond au même problème : les landmarks d'une caméra
RGB tremblent en permanence, même main parfaitement immobile. Transmis
bruts, ce tremblement devient un trait qui vibre, un élément qui frémit,
une taille qui clignote. Ces quelques fonctions sont ce qui sépare une
démo qui « marche » d'une application agréable à utiliser.
"""
from __future__ import annotations

import math


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def deadzone(value: float, threshold: float) -> float:
    """Annule les petites variations, et RETIRE le seuil des grandes.

    Sans ça, les signaux de profondeur (yaw, pitch) feraient vibrer
    l'orientation d'un élément en permanence. Retirer le seuil, plutôt que
    de simplement filtrer, évite le saut qu'on aurait au moment précis où
    on franchit la zone morte."""
    if abs(value) <= threshold:
        return 0.0
    return value - math.copysign(threshold, value)


def angle_delta(a: float, b: float) -> float:
    """Écart angulaire le plus court entre deux angles. Sans ça, le passage
    de +pi à -pi serait lu comme un tour complet à vitesse folle, et l'élan
    partirait en vrille au moment précis où la main croise la verticale."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


# --------------------------------------------------------------------------
# One Euro Filter — suit vite quand ça bouge vite, lisse fort quand ça bouge
# peu. C'est ce qui donne un trait propre sans le rendre mou.
# --------------------------------------------------------------------------
def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t: float, x: float) -> float:
        t_e = max(t - self.t_prev, 1e-6)
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class Point2DFilter:
    """Deux One Euro, un par axe. Filtrer x et y séparément est correct ici
    parce que le bruit des landmarks n'est pas corrélé entre les deux."""

    def __init__(self, t0, x0, y0, min_cutoff=0.8, beta=0.4, d_cutoff=1.0):
        self.fx = OneEuroFilter(t0, x0, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.fy = OneEuroFilter(t0, y0, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def update(self, t: float, x: float, y: float) -> tuple[float, float]:
        return self.fx(t, x), self.fy(t, y)
