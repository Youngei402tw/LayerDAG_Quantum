"""Regenerate every QAI manuscript figure from code and machine-readable results.

Run from the repository root:
    PYTHONPATH=. python scripts/make_figures.py
Outputs vector PDFs (and PNG previews) into paper/figures/.
Inputs (read-only):
    experiments/qai_invariant_final_20261102/final_result.json        (primary, frozen)
    experiments/qai_transformer_control_20261106/control_result.json  (post-confirmatory control)
    experiments/qai_connectivity_controls_20261113/controls_result.json (post-confirmatory connectivity controls)
Figure 1 is computed with the repository's own graph/serialization code on a toy circuit.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qdag.core import Op  # noqa: E402
from qdag.invariant_benchmark import role_edges  # noqa: E402
from qdag.schedule_benchmark import CircuitExample, canonical_order, critical_labels  # noqa: E402

OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colour-blind-safe hues; Transformer is deliberately de-emphasised (grey + hatch)
C_DAG, C_GRID, C_TF, C_PTR, INK, MUTED = "#0072B2", "#D55E00", "#7F7F7F", "#009E73", "#222222", "#666666"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "mathtext.fontset": "stix", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 1: one circuit, two logical-qubit labelings
# --------------------------------------------------------------------------------------
def figure_representation():
    ops = [Op("h", (0,)), Op("ry", (2,), (0.7,)), Op("cx", (0, 1)), Op("rz", (1,), (0.3,)),
           Op("cx", (2, 1)), Op("h", (0,))]
    names = ["H", "RY", "CX", "RZ", "CX", "H"]
    tags = ["a", "b", "c", "d", "e", "f"]  # node identities (stable under relabeling)
    pi = {0: 2, 1: 1, 2: 0}  # swap logical labels q0 <-> q2
    ex = CircuitExample(3, ops, critical_labels(3, ops))
    ops2 = [Op(o.gate, tuple(pi[q] for q in o.wires), o.params) for o in ops]
    ex2 = CircuitExample(3, ops2, critical_labels(3, ops2))
    e1, a1 = role_edges(ex)
    e2, a2 = role_edges(ex2)
    assert np.array_equal(e1, e2) and np.array_equal(a1, a2), "role DAG must be identical"
    assert np.array_equal(ex.labels, ex2.labels)
    o1, o2 = canonical_order(ex), canonical_order(ex2)
    assert o1 != o2, "toy example should show a serialization change"

    fig = plt.figure(figsize=(7.1, 1.95))
    # ---- circuits (left) ----
    col_x = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 3}  # visual columns
    col_x = {0: 0.0, 1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 2.0}

    def draw_circuit(ax, circuit_ops, title):
        ax.set_xlim(-0.8, 3.6); ax.set_ylim(-0.6, 2.75); ax.axis("off")
        ax.set_title(title, fontsize=7.5, pad=1)
        for q in range(3):
            y = 2 - q
            ax.plot([-0.35, 3.5], [y, y], color=MUTED, lw=0.7, zorder=0)
            ax.text(-0.45, y, f"$q_{q}$", ha="right", va="center", fontsize=7.5)
        for i, op in enumerate(circuit_ops):
            x = col_x[i]
            crit = ex.labels[i] == 1
            if op.gate == "cx":
                c, t = op.wires
                ax.plot([x, x], [2 - c, 2 - t], color=INK, lw=0.8, zorder=1)
                ax.add_patch(plt.Circle((x, 2 - c), 0.07, color=INK, zorder=2))
                ax.add_patch(plt.Circle((x, 2 - t), 0.16, fill=False, color=INK, lw=0.8, zorder=2))
                ax.plot([x - 0.16, x + 0.16], [2 - t, 2 - t], color=INK, lw=0.8, zorder=2)
                ax.plot([x, x], [2 - t - 0.16, 2 - t + 0.16], color=INK, lw=0.8, zorder=2)
                ax.text(x + 0.22, 2 - t - 0.32, tags[i], fontsize=6.3, color=MUTED, style="italic")
            else:
                (q,) = op.wires
                ax.add_patch(FancyBboxPatch((x - 0.24, 2 - q - 0.2), 0.48, 0.4, boxstyle="round,pad=0.01,rounding_size=0.06",
                                            fc="white", ec=INK, lw=0.8, zorder=2))
                ax.text(x, 2 - q, names[i], ha="center", va="center", fontsize=6.3, zorder=3)
                ax.text(x + 0.27, 2 - q - 0.36, tags[i], fontsize=6.3, color=MUTED, style="italic")
            _ = crit

    ax1 = fig.add_axes([0.0, 0.08, 0.2, 0.86]); draw_circuit(ax1, ops, r"(a) circuit $C$")
    ax2 = fig.add_axes([0.215, 0.08, 0.2, 0.86]); draw_circuit(ax2, ops2, r"(b) relabeled $\pi(C)$: $q_0\leftrightarrow q_2$")

    # ---- role DAG (middle) ----
    ax3 = fig.add_axes([0.43, 0.04, 0.29, 0.9]); ax3.axis("off")
    ax3.set_title(r"(c) role DAG, identical for $C$ and $\pi(C)$", fontsize=7.5, pad=1)
    pos = {0: (0.0, 1.0), 1: (1.0, -0.05), 2: (1.0, 1.0), 3: (2.0, 1.0), 4: (3.0, 0.5), 5: (2.0, 1.8)}
    ax3.set_xlim(-0.45, 3.45); ax3.set_ylim(-0.5, 2.35)
    role_name = ["s", "c", "t"]
    for (s, d), attr in zip(e1.T, a1):
        rs, rd = role_name[int(np.argmax(attr[:3]))], role_name[int(np.argmax(attr[3:]))]
        (x0, y0), (x1, y1) = pos[int(s)], pos[int(d)]
        arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=INK,
                              shrinkA=9, shrinkB=9, connectionstyle="arc3,rad=0.0", zorder=1)
        ax3.add_patch(arr)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax3.text(mx, my + 0.13, f"{rs}→{rd}", ha="center", va="bottom", fontsize=6.2, color=C_DAG,
                 bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none"), zorder=3)
    for i, (x, y) in pos.items():
        crit = ex.labels[i] == 1
        ax3.add_patch(plt.Circle((x, y), 0.27, fc="#DCEBF5" if crit else "white", ec=C_DAG if crit else MUTED,
                                 lw=1.1 if crit else 0.7, zorder=2))
        ax3.text(x, y + 0.03, names[i], ha="center", va="center", fontsize=6.3, zorder=3)
        ax3.text(x, y - 0.42, tags[i], ha="center", va="center", fontsize=6.3, color=MUTED, style="italic", zorder=3)
    ax3.text(-0.42, 1.75, "roles: s single, c control,\nt target. Shaded: on a\nmaximum path (label 1)",
             fontsize=5.9, color=MUTED, va="bottom", linespacing=1.1)

    # ---- canonical serializations (right) ----
    ax4 = fig.add_axes([0.745, 0.08, 0.255, 0.86]); ax4.axis("off")
    ax4.set_title("(d) canonical token sequences", fontsize=7.5, pad=1)
    ax4.set_xlim(-0.2, 6.2); ax4.set_ylim(-0.3, 2.6)
    for row, (order, label) in enumerate(((o1, r"$C$"), (o2, r"$\pi(C)$"))):
        y = 1.85 - row * 1.15
        ax4.text(-0.15, y + 0.45, label + ":", fontsize=7, ha="left")
        for p, node in enumerate(order):
            moved = o1[p] != o2[p]
            ax4.add_patch(FancyBboxPatch((p + 0.08, y - 0.25), 0.84, 0.5, boxstyle="round,pad=0.01,rounding_size=0.08",
                                         fc="#FBE3D6" if moved else "white", ec=C_GRID if moved else MUTED,
                                         lw=1.0 if moved else 0.6))
            ax4.text(p + 0.5, y + 0.02, names[node], ha="center", va="center", fontsize=6.2)
            ax4.text(p + 0.5, y - 0.4, tags[node], ha="center", va="center", fontsize=6.0, color=MUTED, style="italic")
    ax4.text(-0.15, -0.28, "highlighted: tokens whose position changes;\ntokens carry 7 features and no wire IDs",
             fontsize=5.9, color=MUTED, va="bottom")
    save(fig, "fig_representation")
    return {"order_C": o1, "order_piC": o2}


# --------------------------------------------------------------------------------------
# Figure 2: primary frozen results + post-confirmatory control
# --------------------------------------------------------------------------------------
def figure_results():
    P = json.loads((ROOT / "experiments/qai_invariant_final_20261102/final_result.json").read_text())
    T = json.loads((ROOT / "experiments/qai_transformer_control_20261106/control_result.json").read_text())
    K = json.loads((ROOT / "experiments/qai_connectivity_controls_20261113/controls_result.json").read_text())
    splits = ["same", "near_width", "far_width", "renamed"]
    xlabels = ["same size\n4–7 q", "near width\n8–12 q", "far width\n13–20 q", "relabeled\n4–7 q"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw={"width_ratios": [2.25, 1], "wspace": 0.55})
    x = np.arange(len(splits)); w = 0.19
    series = [("Role DAG", C_DAG, [P["aggregate"]["dag"][s] for s in splits], None),
              ("Grid GRU", C_GRID, [P["aggregate"]["grid"][s] for s in splits], None),
              ("Pointer GRU*", C_PTR, [K["aggregate"]["pointer_grid"][s] for s in splits], "////"),
              ("Transformer*", C_TF, [T["aggregate"][s] for s in splits], "////")]
    markers = ["o", "s", "D", "^"]
    for k, (name, color, rows, hatch) in enumerate(series):
        means = np.array([r["auc_mean"] for r in rows])
        xs = x + (k - 1.5) * w
        for i, r in enumerate(rows):  # individual seeds
            ax.scatter(np.full(len(r["auc_values"]), xs[i]), r["auc_values"], s=3, color=color, alpha=0.55,
                       zorder=2, linewidths=0)
        ax.plot(xs, means, ls="none", marker=markers[k], ms=4.2, mfc=color if hatch is None else "white",
                mec=color, mew=1.0, label=name, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=6.2)
    ax.set_ylim(0.72, 1.0); ax.set_ylabel("node AUROC")
    ax.yaxis.grid(True, lw=0.4, color="#dddddd"); ax.set_axisbelow(True)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, frameon=False, handlelength=1.2,
               columnspacing=0.7, fontsize=5.9)
    ax.set_title("(a) mean AUROC; small dots are seeds", fontsize=6.8, loc="left", pad=3)
    # (b) paired far-width differences per seed
    d = np.array(P["far_width_auc_paired_differences"]); seeds = P["seeds"]
    dp = np.array(K["comparisons"]["pointer_grid"]["recorded_dag_minus_model_far_auc"])
    bx.axhline(0, color=INK, lw=0.6)
    bx.axhline(0.03, color=MUTED, lw=0.6, ls="--")
    bx.text(0.35, 0.0312, "frozen +0.03\nmean threshold", fontsize=5.5, color=MUTED, va="bottom")
    bx.scatter(np.arange(5), d, s=12, color=C_DAG, zorder=3, label="vs grid")
    bx.axhline(d.mean(), color=C_DAG, lw=0.8)
    bx.scatter(np.arange(5), dp, s=12, marker="D", facecolor="white", edgecolor=C_PTR, lw=0.9, zorder=3, label="vs pointer GRU*")
    bx.axhline(dp.mean(), color=C_PTR, lw=0.8, ls=":")
    bx.legend(loc="center right", fontsize=5.2, frameon=False, handletextpad=0.2, borderaxespad=0.1)
    bx.set_xticks(np.arange(5)); bx.set_xticklabels([str(s) for s in seeds], fontsize=5.8, rotation=45)
    bx.set_xlabel("paired seed", fontsize=6.5, labelpad=1)
    bx.set_ylim(-0.012, 0.055); bx.set_ylabel("DAG − model AUROC\n(far width)", fontsize=6.5)
    bx.yaxis.grid(True, lw=0.4, color="#dddddd"); bx.set_axisbelow(True)
    bx.set_title("(b) paired, per seed", fontsize=6.8, loc="left", pad=3)
    save(fig, "fig_results")


if __name__ == "__main__":
    info = figure_representation()
    figure_results()
    print("canonical orders:", info)
    print("wrote", sorted(p.name for p in OUT.glob("fig_*")))
