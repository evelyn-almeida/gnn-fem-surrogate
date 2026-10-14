import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from config import FIG_DIR

BLUE, ORANGE, GREEN, RED, GRAY = "#1f4e79", "#e07a1f", "#2a9d8f", "#c0392b", "#8a8a8a"
MODEL_COLORS = {"Beam theory": GRAY, "Node-wise MLP": ORANGE, "GNN (message passing)": BLUE}
CMAP = "turbo"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _field(ax, nodes, tris, values, vmin=None, vmax=None, title=None, cmap=CMAP):
    tpc = ax.tripcolor(nodes[:, 0], nodes[:, 1], tris, values, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(aspect="equal", xticks=[], yticks=[])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    return tpc


def problem_setup(design, result):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    nodes, tris = result["nodes"], result["tris"]
    axes[0].triplot(nodes[:, 0], nodes[:, 1], tris, color=GRAY, lw=0.4)
    axes[0].plot([0, 0], [0, design.height], color="black", lw=4)
    for y in np.linspace(0, design.height, 8):
        axes[0].plot([-4, 0], [y - 3, y], color="black", lw=1)
    a = np.radians(design.angle)
    x0, y0 = design.length, design.height / 2
    axes[0].add_patch(FancyArrowPatch((x0, y0), (x0 + 25 * np.cos(a), y0 + 25 * np.sin(a)), color=RED,
                                      arrowstyle="-|>", mutation_scale=25, lw=3))
    axes[0].text(x0 + 6, y0 + 4, f"F = {design.load / 1e3:.1f} kN\nθ = {design.angle:.0f}°", color=RED)
    axes[0].text(3, design.height + 3, "clamped (bolted to the chassis)", fontsize=9)
    axes[0].set(aspect="equal", xlim=(-8, design.length + 35), title=f"Mesh: {len(nodes)} nodes, {len(tris)} triangles",
                xlabel="mm", ylabel="mm")
    axes[0].grid(False)
    tpc = _field(axes[1], nodes, tris, result["von_mises"], title="Von Mises stress (finite elements)")
    fig.colorbar(tpc, ax=axes[1], label="MPa", shrink=0.8)
    fig.suptitle("A parametric bracket: steel plate, clamped on the left, loaded on the right",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "01_problem_setup")


def fem_validation(rows, reference, example):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].plot(rows["nodes"], rows["kt_element"], color=BLUE, marker="o", lw=2.5, label="FEM, peak element stress")
    axes[0].plot(rows["nodes"], rows["kt_nodal"], color=ORANGE, marker="s", lw=2.5, label="FEM, nodal average")
    axes[0].axhline(reference, color="black", ls="--", label=f"Heywood formula ({reference:.2f})")
    axes[0].set(xscale="log", xlabel="Mesh nodes", ylabel="Stress concentration factor Kt (net section)",
                title="Mesh convergence: plate with a hole under tension")
    axes[0].legend(loc="lower right")
    nodes, tris, vm = example
    _field(axes[1], nodes, tris, vm, title="σ_vm around the hole (finest mesh)")
    fig.tight_layout()
    save(fig, "02_fem_validation")


def dataset_examples(samples):
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    for ax, s in zip(axes.ravel(), samples):
        d, r = s["design"], s["result"]
        _field(ax, r["nodes"], r["tris"], r["von_mises"],
               title=f"{d.length:.0f}×{d.height:.0f} mm, {len(d.holes)} hole(s), F = {d.load / 1e3:.1f} kN at {d.angle:.0f}°, "
                     f"max {r['von_mises'].max():.0f} MPa")
    fig.suptitle("Training data: 400 random designs solved with finite elements", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_dataset_examples")


def training_curves(histories):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, h in histories.items():
        if name not in MODEL_COLORS:
            continue
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train"] for r in h], color=MODEL_COLORS[name], ls="--", alpha=0.6)
        ax.plot(ep, [r["val"] for r in h], color=MODEL_COLORS[name], lw=2.5, label=f"{name} (validation)")
    ax.set(yscale="log", xlabel="Epoch", ylabel="MSE of the correction to beam theory (dashed: training)", title="Training")
    ax.legend()
    save(fig, "04_training_curves")


def predictions(cases):
    fig, axes = plt.subplots(len(cases), 5, figsize=(22, 3.0 * len(cases)))
    for row, c in zip(np.atleast_2d(axes), cases):
        n, t = c["nodes"], c["tris"]
        vmax = c["fem"].max()
        _field(row[0], n, t, c["fem"], 0, vmax, "Finite elements")
        _field(row[1], n, t, c["beam"], 0, vmax, f"Beam theory (error {c['beam_err']:.1%})")
        _field(row[2], n, t, c["mlp"], 0, vmax, f"Beam + MLP correction ({c['mlp_err']:.1%})")
        _field(row[3], n, t, c["gnn"], 0, vmax, f"Beam + GNN correction ({c['gnn_err']:.1%})")
        tpc = _field(row[4], n, t, np.abs(c["gnn"] - c["fem"]) / vmax * 100, 0, 20, "|GNN − FEM| (% of peak)", "Reds")
        plt.colorbar(tpc, ax=row[4], shrink=0.8)
    fig.suptitle("Unseen test designs: von Mises stress (relative L2 error of the field in brackets)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_predictions")


def accuracy(metrics):
    test = metrics[metrics["split"] == "Test (1–2 holes)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(MODEL_COLORS)
    box = axes[0].boxplot([test.loc[test["model"] == n, "field_error"] * 100 for n in names], patch_artist=True,
                          tick_labels=names, showfliers=False, widths=0.5)
    for patch, n in zip(box["boxes"], names):
        patch.set(facecolor=MODEL_COLORS[n], alpha=0.6)
    axes[0].set(ylabel="Relative L2 error of the stress field (%)", title="Field accuracy on 100 test designs")
    for n in names[1:]:
        m = test[test["model"] == n]
        axes[1].scatter(m["peak_true"], m["peak_pred"], s=22, alpha=0.7, color=MODEL_COLORS[n], label=n)
    lim = [0, test["peak_true"].max() * 1.05]
    axes[1].plot(lim, lim, color="black", ls="--", lw=1)
    axes[1].set(xlim=lim, ylim=lim, xlabel="Peak von Mises, FEM (MPa)", ylabel="Peak von Mises, predicted (MPa)",
                title="Peak stress: the number a designer checks")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "06_accuracy")


def out_of_distribution(summary, case):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    splits = list(summary.index.get_level_values(0).unique())
    x = np.arange(len(splits))
    for i, n in enumerate(MODEL_COLORS):
        vals = [summary.loc[(s, n), "field_error"] * 100 for s in splits]
        bars = axes[0].bar(x + (i - 1) * 0.27, vals, 0.27, color=MODEL_COLORS[n], label=n)
        axes[0].bar_label(bars, fmt="%.1f%%", fontsize=8)
    axes[0].set_xticks(x, splits)
    axes[0].set(ylabel="Median field error (%)", title="Designs with more holes than ever seen in training")
    axes[0].legend()
    vmax = case["fem"].max()
    _field(axes[1], case["nodes"], case["tris"], case["fem"], 0, vmax, "FEM, 3-hole design")
    _field(axes[2], case["nodes"], case["tris"], case["gnn"], 0, vmax, f"Beam + GNN (field error {case['gnn_err']:.1%})")
    fig.tight_layout()
    save(fig, "07_out_of_distribution")


def screening(xs, ys, fem_peak, gnn_peak, best_fem, best_gnn, design, rho):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
    vmin, vmax = np.nanmin(fem_peak), np.nanpercentile(fem_peak, 95)
    for ax, z, title in [(axes[0], fem_peak, "Finite elements"), (axes[1], gnn_peak, "GNN")]:
        im = ax.pcolormesh(xs, ys, z, cmap="viridis_r", vmin=vmin, vmax=vmax, shading="auto")
        ax.scatter(*best_fem, marker="*", s=300, color="white", edgecolor="black", label="best (FEM)", zorder=5)
        ax.scatter(*best_gnn, marker="X", s=150, color=RED, edgecolor="black", label="best (GNN)", zorder=5)
        ax.set(aspect="equal", xlim=(0, design.length), ylim=(0, design.height), xlabel="Hole centre x (mm)",
               title=title)
        ax.grid(False)
    axes[0].set_ylabel("Hole centre y (mm)")
    axes[1].legend(loc="upper right", fontsize=8, frameon=True)
    fig.colorbar(im, ax=axes, label="Peak von Mises (MPa)", shrink=0.85)
    fig.suptitle(f"Where to drill a lightening hole? Peak stress for every position (rank correlation ρ = {rho:.2f})",
                 fontsize=13, fontweight="bold")
    save(fig, "08_design_screening")


def speed(times):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bars = ax.barh(list(times), [t * 1e3 for t in times.values()], color=[GRAY, BLUE, GREEN][:len(times)])
    ax.bar_label(bars, fmt="%.1f ms", padding=3)
    ax.set(xlabel="Time per design (ms, single CPU core)", title="Cost per design in this 2D study")
    ax.invert_yaxis()
    save(fig, "09_speed")
