"""Evaluate the trained models and draw every figure: python evaluate.py"""
import json
import os
import pickle
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import ENGINEERED_FEATURES, HIDDEN, MESSAGE_STEPS, MODEL_DIR, RESULTS_DIR
from src import plots
from src.dataset import load
from src.fem import von_mises
from src.geometry import Design, mesh
from src.gnn import MeshGNN, batch, predict
from src.simulate import analyze, nominal_stress, to_graph

NAMES = {"gnn": "GNN (message passing)", "mlp": "Node-wise MLP"}
SPLITS = {"test": "Test (1–2 holes)", "ood": "Out of distribution (3 holes)"}


def load_models(norm, example):
    models = {}
    for key, name in NAMES.items():
        m = MeshGNN(HIDDEN, MESSAGE_STEPS, messages=key == "gnn")
        x, e, s, r, _ = batch([norm(example)])
        m(x, e, s, r)
        m.load_weights(MODEL_DIR / f"{key}.weights.h5")
        models[name] = m
    return models


def stress(model, norm, design, graph):
    return predict(model, norm(graph)) * nominal_stress(design)


def validation():
    W, L, r, F = 50.0, 200.0, 5.0, 10_000.0
    rows, example = [], None
    for area in [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]:
        res = analyze(Design(L, W, [(L / 2, W / 2, r)], F, 0.0), max_area=area)
        _, s = von_mises(res["nodes"], res["tris"], res["u"])
        net = F / ((W - 2 * r) * 5.0)
        near = np.hypot(res["nodes"][:, 0] - L / 2, res["nodes"][:, 1] - W / 2) < r + 0.5
        rows.append({"nodes": len(res["nodes"]), "kt_element": s[:, 0].max() / net,
                     "kt_nodal": res["von_mises"][near].max() / net})
        window = np.abs(res["nodes"][:, 0] - L / 2) < 20
        example = res
    keep = np.all(np.abs(example["nodes"][example["tris"]][..., 0] - L / 2) < 20, axis=1)
    return pd.DataFrame(rows), 2 + (1 - 2 * r / W) ** 3, (example["nodes"], example["tris"][keep], example["von_mises"])


def screening(models, norm):
    base = dict(length=120.0, height=50.0, load=6_000.0, angle=-60.0)
    radius = 7.0
    xs, ys = np.arange(14, 110, 4.0), np.arange(11, 40, 3.0)
    fem, gnn = np.full((len(ys), len(xs)), np.nan), np.full((len(ys), len(xs)), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            d = Design(holes=[(x, y, radius)], **base)
            res = analyze(d)
            fem[i, j] = res["von_mises"].max()
            gnn[i, j] = stress(models[NAMES["gnn"]], norm, d, to_graph(d, res, ENGINEERED_FEATURES)).max()
    ok = ~np.isnan(fem)
    bf, bg = np.unravel_index(np.nanargmin(fem), fem.shape), np.unravel_index(np.nanargmin(gnn), gnn.shape)
    rho = spearmanr(fem[ok], gnn[ok])[0]
    summary = {"designs": int(ok.sum()), "spearman": float(rho), "fem_best_peak": float(fem[bf]),
               "fem_peak_at_gnn_choice": float(fem[bg]), "worst_peak": float(np.nanmax(fem))}
    return xs, ys, fem, gnn, (xs[bf[1]], ys[bf[0]]), (xs[bg[1]], ys[bg[0]]), Design(**base), summary


def timing(models, norm, designs):
    t_mesh, t_fem, t_gnn = [], [], []
    model = models[NAMES["gnn"]]
    stress(model, norm, designs[0]["design"], designs[0]["graph"])  # warm-up
    for d in designs:
        t0 = time.perf_counter()
        mesh(d["design"])
        t1 = time.perf_counter()
        analyze(d["design"])
        t2 = time.perf_counter()
        stress(model, norm, d["design"], to_graph(d["design"], d["result"], ENGINEERED_FEATURES))
        t3 = time.perf_counter()
        t_mesh.append(t1 - t0)
        t_fem.append((t2 - t1) - (t1 - t0))
        t_gnn.append(t3 - t2)
    return {"Meshing": np.median(t_mesh), "FEM assembly + solve": np.median(t_fem), "GNN inference": np.median(t_gnn)}


def main():
    splits = load()
    for data in splits.values():
        for d in data:
            d["graph"] = to_graph(d["design"], d["result"], ENGINEERED_FEATURES)
    with open(MODEL_DIR / "normalizer.pkl", "rb") as f:
        norm = pickle.load(f)
    models = load_models(norm, splits["train"][0]["graph"])

    rows, preds = [], {}
    for key, label in SPLITS.items():
        for i, d in enumerate(splits[key]):
            fem = d["result"]["von_mises"]
            for name, model in [("Beam theory", None), *models.items()]:
                pred = (d["graph"]["beam"] * nominal_stress(d["design"]) if model is None
                        else stress(model, norm, d["design"], d["graph"]))
                preds[(key, i, name)] = pred
                rows.append({"split": label, "model": name, "design": i,
                             "field_error": np.linalg.norm(pred - fem) / np.linalg.norm(fem),
                             "peak_true": fem.max(), "peak_pred": pred.max(),
                             "peak_error": (pred.max() - fem.max()) / fem.max()})
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(["split", "model"], sort=False).agg(
        field_error=("field_error", "median"), peak_abs_error=("peak_error", lambda s: s.abs().median()),
        peak_within_10pct=("peak_error", lambda s: (s.abs() < 0.10).mean()))

    val_rows, kt_ref, val_example = validation()
    xs, ys, fem_map, gnn_map, best_fem, best_gnn, screen_design, screen = screening(models, norm)
    times = timing(models, norm, splits["test"][:40])

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "per_design_metrics.csv", index=False, float_format="%.5f")
    (RESULTS_DIR / "summary.md").write_text(summary.round(4).to_markdown())
    (RESULTS_DIR / "summary.json").write_text(json.dumps({
        "fem_validation": {"heywood_kt": kt_ref, **val_rows.iloc[-1].to_dict()},
        "screening": screen, "timing_ms": {k: v * 1e3 for k, v in times.items()}}, indent=2, default=float))

    plots.setup_style()
    ex = splits["train"][3]
    plots.problem_setup(ex["design"], ex["result"])
    plots.fem_validation(val_rows, kt_ref, val_example)
    plots.dataset_examples(splits["train"][:6])
    hist = json.loads((RESULTS_DIR / "training_history.json").read_text())
    plots.training_curves({NAMES[k]: v for k, v in hist.items()})

    def case(key, i):
        d = splits[key][i]
        fem = d["result"]["von_mises"]
        g, m, b = preds[(key, i, NAMES["gnn"])], preds[(key, i, NAMES["mlp"])], preds[(key, i, "Beam theory")]
        return {"nodes": d["result"]["nodes"], "tris": d["result"]["tris"], "fem": fem, "gnn": g, "mlp": m, "beam": b,
                "beam_err": np.linalg.norm(b - fem) / np.linalg.norm(fem),
                "gnn_err": np.linalg.norm(g - fem) / np.linalg.norm(fem),
                "mlp_err": np.linalg.norm(m - fem) / np.linalg.norm(fem)}

    gnn_test = metrics[(metrics["split"] == SPLITS["test"]) & (metrics["model"] == NAMES["gnn"])]
    order = gnn_test.sort_values("field_error")["design"].to_numpy()
    picks = [order[len(order) // 4], order[len(order) // 2], order[int(len(order) * 0.9)]]
    plots.predictions([case("test", i) for i in picks])
    plots.accuracy(metrics)
    gnn_ood = metrics[(metrics["split"] == SPLITS["ood"]) & (metrics["model"] == NAMES["gnn"])]
    plots.out_of_distribution(summary, case("ood", int(gnn_ood.sort_values("field_error")["design"].iloc[len(gnn_ood) // 2])))
    plots.screening(xs, ys, fem_map, gnn_map, best_fem, best_gnn, screen_design, screen["spearman"])
    plots.speed(times)

    print(summary.round(4).to_string())
    print(json.dumps({"validation": val_rows.round(3).to_dict(orient="records"), "heywood": kt_ref,
                      "screening": screen, "timing_ms": {k: round(v * 1e3, 2) for k, v in times.items()}}, indent=2))


if __name__ == "__main__":
    main()
