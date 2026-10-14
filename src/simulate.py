"""Run one FE analysis per design and package the result as a graph."""
import numpy as np

from config import THICKNESS
from src.beam import beam_von_mises
from src.fem import edge_traction, solve, von_mises
from src.geometry import mesh


def nominal_stress(design):
    return design.load / (design.height * THICKNESS)


def analyze(design, max_area=None):
    nodes, tris = mesh(design) if max_area is None else mesh(design, max_area)
    tol = 1e-6
    fixed = np.flatnonzero(nodes[:, 0] < tol)
    loaded = np.flatnonzero(nodes[:, 0] > design.length - tol)
    forces = edge_traction(nodes, loaded, design.load, design.angle)
    u = solve(nodes, tris, fixed, forces)
    vm, _ = von_mises(nodes, tris, u)
    return {"nodes": nodes, "tris": tris, "u": u, "von_mises": vm, "fixed": fixed, "loaded": loaded}


def edges_from_triangles(tris):
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.vstack([e, e[:, ::-1]])     # both directions for message passing


def to_graph(design, result, engineered=False):
    """Dimensionless node/edge features, so one model covers every size and load.

    By default the nodes only know their position, boundary type and the load: the geometry of the holes
    must be inferred from the mesh. engineered=True adds hand-crafted hole descriptors (distance to the
    nearest hole and its radius), which only exist for circular holes.
    """
    nodes, tris = result["nodes"], result["tris"]
    L, H = design.length, design.height
    n = len(nodes)
    boundary = np.zeros(n)
    seg = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    key, counts = np.unique(np.sort(seg, axis=1), axis=0, return_counts=True)
    boundary[np.unique(key[counts == 1])] = 1.0          # edges used by a single triangle lie on a boundary
    fixed, loaded = np.zeros(n), np.zeros(n)
    fixed[result["fixed"]], loaded[result["loaded"]] = 1.0, 1.0
    if design.holes:
        h = np.array(design.holes)
        d = np.hypot(nodes[:, None, 0] - h[None, :, 0], nodes[:, None, 1] - h[None, :, 1]) - h[None, :, 2]
        nearest = d.argmin(axis=1)
        hole_dist = d.min(axis=1) / H
        hole_r = h[nearest, 2] / H
    else:
        hole_dist, hole_r = np.ones(n), np.zeros(n)
    a = np.radians(design.angle)
    beam = beam_von_mises(design, nodes) / nominal_stress(design)
    columns = [beam, nodes[:, 0] / H, nodes[:, 1] / H, (L - nodes[:, 0]) / H, np.minimum(nodes[:, 1], H - nodes[:, 1]) / H,
               boundary, fixed, loaded, np.full(n, np.cos(a)), np.full(n, np.sin(a)), np.full(n, L / H)]
    if engineered:
        columns += [np.exp(-hole_dist * 5), hole_r]
    node_feat = np.column_stack(columns).astype(np.float32)
    edges = edges_from_triangles(tris)
    rel = (nodes[edges[:, 1]] - nodes[edges[:, 0]]) / H
    edge_feat = np.column_stack([rel, np.linalg.norm(rel, axis=1)]).astype(np.float32)
    # The networks learn the correction to beam theory: what the holes and the clamp add to the hand calculation
    target = (result["von_mises"] / nominal_stress(design) - beam).astype(np.float32)
    return {"x": node_feat, "e": edge_feat, "senders": edges[:, 0].astype(np.int32),
            "receivers": edges[:, 1].astype(np.int32), "y": target, "beam": beam.astype(np.float32)}
