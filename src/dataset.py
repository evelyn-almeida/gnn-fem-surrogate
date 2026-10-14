import pickle

import numpy as np

from config import DATA_DIR, N_OOD, N_TEST, N_TRAIN, N_VAL, OOD_HOLES, SEED, TRAIN_HOLES
from src.geometry import random_design
from src.simulate import analyze, to_graph


def generate(n, holes, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        design = random_design(rng, int(rng.integers(holes[0], holes[1] + 1)))
        result = analyze(design)
        out.append({"design": design, "result": result, "graph": to_graph(design, result)})
    return out


def build():
    """FE simulations for every split; the out-of-distribution set has one more hole than training ever saw."""
    splits = {"train": generate(N_TRAIN, TRAIN_HOLES, SEED), "val": generate(N_VAL, TRAIN_HOLES, SEED + 1),
              "test": generate(N_TEST, TRAIN_HOLES, SEED + 2), "ood": generate(N_OOD, (OOD_HOLES, OOD_HOLES), SEED + 3)}
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "dataset.pkl", "wb") as f:
        pickle.dump(splits, f)
    return splits


def load():
    path = DATA_DIR / "dataset.pkl"
    if not path.exists():
        return build()
    with open(path, "rb") as f:
        return pickle.load(f)
