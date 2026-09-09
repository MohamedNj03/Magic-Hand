"""Step 2/8 - One Euro smoothing for the noisy landmark signals.

Used by gesture_canvas_manipulation.py.
"""
from __future__ import annotations

import math


# --- Small numeric helpers -----------------------------------------------
def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def deadzone(value: float, threshold: float) -> float:
    if abs(value) <= threshold:
        return 0.0
    return value - math.copysign(threshold, value)


def angle_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


# --- One Euro filter -----------------------------------------------------
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
    def __init__(self, t0, x0, y0, min_cutoff=0.8, beta=0.4, d_cutoff=1.0):
        self.fx = OneEuroFilter(t0, x0, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.fy = OneEuroFilter(t0, y0, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def update(self, t: float, x: float, y: float) -> tuple[float, float]:
        return self.fx(t, x), self.fy(t, y)
