"""Linear-elastic plane-stress finite elements (constant-strain triangles), written from scratch."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from config import POISSON, THICKNESS, YOUNG


def elasticity_matrix(E=YOUNG, nu=POISSON):
    return E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


def strain_displacement(nodes, tris):
    """B matrices (m, 3, 6) and areas (m,) of all triangles."""
    x, y = nodes[tris, 0], nodes[tris, 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    area = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    B = np.zeros((len(tris), 3, 6))
    B[:, 0, 0::2], B[:, 1, 1::2] = b, c
    B[:, 2, 0::2], B[:, 2, 1::2] = c, b
    return B / (2 * area[:, None, None]), np.abs(area)


def solve(nodes, tris, fixed, forces):
    """Displacements (n, 2). fixed: node indices clamped in x and y; forces: (n, 2) nodal forces in N."""
    D = elasticity_matrix()
    B, area = strain_displacement(nodes, tris)
    Ke = THICKNESS * area[:, None, None] * np.einsum("mji,jk,mkl->mil", B, D, B)
    dofs = np.stack([2 * tris, 2 * tris + 1], axis=2).reshape(len(tris), 6)
    rows = np.repeat(dofs, 6, axis=1).ravel()
    cols = np.tile(dofs, (1, 6)).ravel()
    n = 2 * len(nodes)
    K = coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    blocked = np.zeros(n, dtype=bool)
    blocked[2 * fixed], blocked[2 * fixed + 1] = True, True
    free = ~blocked
    u = np.zeros(n)
    u[free] = spsolve(K[free][:, free], forces.ravel()[free])
    return u.reshape(-1, 2)


def von_mises(nodes, tris, u):
    """Element stresses (MPa) averaged to the nodes, weighted by element area."""
    B, area = strain_displacement(nodes, tris)
    disp = u[tris].reshape(len(tris), 6)
    s = np.einsum("ij,mjk,mk->mi", elasticity_matrix(), B, disp)
    vm = np.sqrt(s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2 + 3 * s[:, 2] ** 2)
    num = np.bincount(tris.ravel(), weights=np.repeat(vm * area, 3), minlength=len(nodes))
    den = np.bincount(tris.ravel(), weights=np.repeat(area, 3), minlength=len(nodes))
    return num / den, s


def edge_traction(nodes, edge_nodes, total_force, angle_deg):
    """Uniform traction on a straight edge, lumped to nodal forces (exact for linear elements)."""
    order = edge_nodes[np.argsort(nodes[edge_nodes, 1])]
    seg = np.diff(nodes[order, 1])
    weights = np.zeros(len(order))
    weights[:-1] += seg / 2
    weights[1:] += seg / 2
    direction = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    forces = np.zeros((len(nodes), 2))
    forces[order] = total_force * (weights / weights.sum())[:, None] * direction
    return forces
