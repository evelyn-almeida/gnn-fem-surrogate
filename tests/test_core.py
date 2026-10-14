import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import THICKNESS
from src.fem import edge_traction
from src.geometry import Design, random_design
from src.gnn import MeshGNN, batch
from src.simulate import analyze, to_graph


def test_uniform_tension_gives_nominal_stress():
    d = Design(200.0, 40.0, [], load=8_000.0, angle=0.0)
    res = analyze(d, max_area=8.0)
    middle = np.abs(res["nodes"][:, 0] - 100) < 30
    assert np.allclose(res["von_mises"][middle], d.load / (d.height * THICKNESS), rtol=0.02)


def test_traction_resultant_and_direction():
    nodes = np.c_[np.full(5, 10.0), np.linspace(0, 20, 5)]
    f = edge_traction(nodes, np.arange(5), 1_000.0, 90.0)
    assert np.allclose(f.sum(axis=0), [0, 1_000.0], atol=1e-6)


def test_random_designs_have_separate_holes():
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = random_design(rng, 3)
        for i, (x1, y1, r1) in enumerate(d.holes):
            assert r1 < x1 < d.length - r1 and r1 < y1 < d.height - r1
            for x2, y2, r2 in d.holes[i + 1:]:
                assert np.hypot(x1 - x2, y1 - y2) > r1 + r2


def test_graph_is_symmetric_and_target_positive():
    d = random_design(np.random.default_rng(1), 2)
    g = to_graph(d, analyze(d))
    pairs = set(zip(g["senders"], g["receivers"]))
    assert all((r, s) in pairs for s, r in pairs)
    assert g["x"].shape[0] == g["y"].shape[0] == g["beam"].shape[0]


def test_beam_theory_matches_fem_far_from_holes_and_clamp():
    from src.beam import beam_von_mises
    d = Design(200.0, 40.0, [], load=3_000.0, angle=-90.0)
    res = analyze(d, max_area=4.0)
    beam = beam_von_mises(d, res["nodes"])
    region = (np.abs(res["nodes"][:, 0] - 80) < 15) & (np.abs(res["nodes"][:, 1] - 20) > 12)
    assert np.allclose(res["von_mises"][region], beam[region], rtol=0.08)


def test_gnn_is_permutation_equivariant():
    d = random_design(np.random.default_rng(2), 1)
    g = to_graph(d, analyze(d))
    model = MeshGNN(hidden=16, steps=2)
    out = model(*batch([g])[:4]).numpy()
    perm = np.random.default_rng(0).permutation(len(g["x"]))
    inv = np.argsort(perm)
    g2 = {**g, "x": g["x"][perm], "y": g["y"][perm], "senders": inv[g["senders"]].astype(np.int32),
          "receivers": inv[g["receivers"]].astype(np.int32)}
    out2 = model(*batch([g2])[:4]).numpy()
    assert np.allclose(out[perm], out2, atol=1e-4)
