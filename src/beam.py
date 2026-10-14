"""Hand-calculation baseline: the bracket as a cantilever beam (no holes, no stress concentrations)."""
import numpy as np

from config import THICKNESS


def beam_von_mises(design, nodes):
    """Axial + bending stress (Euler-Bernoulli) and parabolic shear at every node, in MPa."""
    a = np.radians(design.angle)
    N, V = design.load * np.cos(a), design.load * np.sin(a)
    H, t = design.height, THICKNESS
    area, inertia = H * t, t * H**3 / 12
    y = nodes[:, 1] - H / 2
    moment = V * (design.length - nodes[:, 0])
    sigma = N / area - moment * y / inertia
    tau = 1.5 * V / area * (1 - (2 * y / H) ** 2)
    return np.sqrt(sigma**2 + 3 * tau**2)
