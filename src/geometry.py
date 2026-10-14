"""Parametric 2D bracket: a plate clamped on the left edge, loaded on the right edge, with holes."""
from dataclasses import dataclass, field

import numpy as np
import triangle

from config import HEIGHT, HOLE_RADIUS, LENGTH, LOAD, LOAD_ANGLE, MAX_ELEMENT_AREA


@dataclass
class Design:
    length: float
    height: float
    holes: list = field(default_factory=list)     # (cx, cy, r)
    load: float = 5_000.0                          # N, total force on the right edge
    angle: float = 0.0                             # degrees


def random_design(rng, n_holes):
    length, height = rng.uniform(*LENGTH), rng.uniform(*HEIGHT)
    holes = []
    for _ in range(500):
        if len(holes) == n_holes:
            break
        r = rng.uniform(*HOLE_RADIUS)
        cx = rng.uniform(0.15 * length + r, 0.9 * length - r)
        cy = rng.uniform(r + 0.2 * height, 0.8 * height - r)
        margin = 3.0
        if all(np.hypot(cx - x, cy - y) > r + q + margin for x, y, q in holes) and r < 0.3 * height:
            holes.append((cx, cy, r))
    return Design(length, height, holes, rng.uniform(*LOAD), rng.uniform(*LOAD_ANGLE))


def _circle(cx, cy, r, h):
    n = max(12, int(2 * np.pi * r / h))
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def mesh(design, max_area=MAX_ELEMENT_AREA):
    """Quality triangular mesh (Shewchuk's Triangle). Returns nodes (n, 2) and triangles (m, 3)."""
    h = np.sqrt(2 * max_area)
    L, H = design.length, design.height
    nx, ny = max(2, int(L / h)), max(2, int(H / h))
    bottom = np.c_[np.linspace(0, L, nx, endpoint=False), np.zeros(nx)]
    right = np.c_[np.full(ny, L), np.linspace(0, H, ny, endpoint=False)]
    top = np.c_[np.linspace(L, 0, nx, endpoint=False), np.full(nx, H)]
    left = np.c_[np.zeros(ny), np.linspace(H, 0, ny, endpoint=False)]
    loops = [np.vstack([bottom, right, top, left])] + [_circle(cx, cy, r, h * 0.3) for cx, cy, r in design.holes]
    vertices, segments, start = [], [], 0
    for loop in loops:
        n = len(loop)
        vertices.append(loop)
        segments += [[start + i, start + (i + 1) % n] for i in range(n)]
        start += n
    data = {"vertices": np.vstack(vertices), "segments": np.array(segments)}
    if design.holes:
        data["holes"] = np.array([[cx, cy] for cx, cy, _ in design.holes])
    out = triangle.triangulate(data, f"pq30a{max_area}")
    return out["vertices"], out["triangles"]
