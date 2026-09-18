"""
Render the README figure from the package itself.

Three panels, each one a claim the README makes:

  A  the direction preferences are lopsided, which is why the decoder has an
     opinion before it sees anything and why that offset has to be subtracted
  B  what training actually achieved, four seeds, against the frozen control
  C  dopamine moves the decision within a step, before any synapse moves

A and C are computed live from a Brain. B plots the measured run recorded in
the README; those numbers live in RESULT below so the figure and the text
cannot drift apart.

    pip install -e ".[figures]"
    python tools/figure.py
    python tools/figure.py --measured      # same figure from build/graph.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INK = "#eff2df"
DIM = "#78826c"
PAPER = "#0b0d09"
PANEL = "#10130e"
GRID = "#26301c"
ACCENT = "#ccff00"
MUTED = "#6d7763"

# Four seeds, cue task, 600 episodes, 100 frozen evaluation episodes each side.
RESULT = [("seed 0", 29.0, 47.0), ("seed 1", 32.0, 50.0),
          ("seed 2", 3.0, 8.0), ("seed 3", 16.0, 21.0)]
CONTROL_DRIFT, CONTROL_SPREAD, CHANCE = 1.0, 3.0, 25.0


def moving_bar(size: int, step: int, steps: int) -> np.ndarray:
    """A bright bar sweeping left to right: the oldest stimulus in vision."""
    field = np.zeros((size, size), dtype=float)
    x = int((step / max(steps - 1, 1)) * (size - 10))
    field[:, x:x + 8] = 1.0
    field[size // 3:2 * size // 3, :] += 0.15
    return np.clip(field, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured", action="store_true")
    parser.add_argument("--out", default="docs/hero.png")
    parser.add_argument("--neurons", type=int, default=4000)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cortex import Brain

    brain = Brain.measured(quiet=True) if args.measured else Brain.demo(n=args.neurons, quiet=True)
    graph = brain.graph
    label = ("MEASURED — MICrONS mouse V1" if graph.measured
             else "SYNTHETIC — generated with the cube's statistics, not measured")

    fig = plt.figure(figsize=(16.5, 5.6), facecolor=PAPER)
    grid = fig.add_gridspec(1, 3, left=0.05, right=0.955, top=0.68, bottom=0.12, wspace=0.30)
    ax_a = fig.add_subplot(grid[0], projection="polar")
    ax_b = fig.add_subplot(grid[1])
    ax_c = fig.add_subplot(grid[2])

    for ax in (ax_a, ax_b, ax_c):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=DIM, labelsize=7.5, length=2)

    def title(ax, index, text, note):
        pos = ax.get_position()
        fig.text(pos.x0, 0.845, index, color=ACCENT, fontsize=10, family="monospace")
        fig.text(pos.x0 + 0.017, 0.842, text, color=INK, fontsize=11)
        fig.text(pos.x0 + 0.017, 0.760, note, color=DIM, fontsize=8, family="monospace",
                 linespacing=1.6)

    # ---- A. the preferences are not uniform --------------------------------
    pref = np.asarray(graph.pref_dir, dtype=float) % (2 * np.pi)
    bins = 24
    counts, edges = np.histogram(pref, bins=bins, range=(0, 2 * np.pi))
    centres = (edges[:-1] + edges[1:]) / 2
    even = pref.size / bins
    bars = ax_a.bar(centres, counts, width=2 * np.pi / bins * 0.9,
                    color=ACCENT, edgecolor=PAPER, linewidth=0.6)
    for bar, count in zip(bars, counts):
        bar.set_alpha(0.75 if count > even else 0.3)
    ax_a.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, even),
              color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax_a.set_theta_zero_location("E")
    ax_a.set_yticklabels([])
    ax_a.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax_a.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    ax_a.grid(color=GRID, lw=0.6)
    ratio = counts.max() / max(counts.min(), 1)
    title(ax_a, "A", "What the population prefers, before it sees anything",
          f"{pref.size:,} cells · busiest bin {int(counts.max())} against quietest {int(counts.min())}, {ratio:.1f}x apart" "\n"
          "dashed ring = a perfectly even population")

    # ---- B. what training achieved, against its own control ----------------
    ys = np.arange(len(RESULT))[::-1]
    ax_b.axvline(CHANCE, color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax_b.text(CHANCE + 0.8, len(RESULT) - 0.45, "chance", color=DIM, fontsize=7.5,
              family="monospace")
    for y, (name, before, after) in zip(ys, RESULT):
        real = (after - before) > CONTROL_DRIFT + 2 * CONTROL_SPREAD
        colour = ACCENT if real else MUTED
        ax_b.plot([before, after], [y, y], color=colour, lw=2.0, alpha=0.85,
                  solid_capstyle="round")
        ax_b.scatter([before], [y], s=34, color=PANEL, edgecolor=MUTED, lw=1.2, zorder=3)
        ax_b.scatter([after], [y], s=46, color=colour, lw=0, zorder=3)
        ax_b.text(after + 1.8, y, f"+{after - before:.0f}", color=colour,
                  fontsize=8.5, va="center", family="monospace")
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels([name for name, _, _ in RESULT], color=DIM, fontsize=8)
    ax_b.set_xlabel("accuracy on the four-option cue task, %", color=DIM, fontsize=8.5)
    ax_b.set_xlim(0, 60)
    ax_b.set_ylim(-0.7, len(RESULT) - 0.15)
    ax_b.grid(axis="x", color=GRID, lw=0.6)
    ax_b.set_axisbelow(True)
    title(ax_b, "B", "Every seed improved, two of them for real",
          "600 episodes · 100 frozen evaluation episodes each side\n"
          f"lime = clear of the frozen control, which drifts +{CONTROL_DRIFT:.0f} with spread {CONTROL_SPREAD:.0f}")

    # ---- C. what a payment does to the decision ----------------------------
    field = moving_bar(48, 3, 6)
    before_actions = [brain.look(field) for _ in range(14)]
    for _ in range(6):
        brain.look(field)
        brain.reward(+1.0)
    after_actions = [brain.look(field) for _ in range(14)]

    def cloud(actions, colour, text):
        xs = np.array([float(getattr(a, "dx", 0.0)) for a in actions])
        ys = np.array([float(getattr(a, "dy", 0.0)) for a in actions])
        ax_c.scatter(xs, ys, s=30, color=colour, lw=0, alpha=0.7, label=text)
        ax_c.scatter([xs.mean()], [ys.mean()], s=190, facecolor="none",
                     edgecolor=colour, lw=1.5, zorder=4)
        return float(xs.mean()), float(ys.mean())

    mean_before = cloud(before_actions, MUTED, "before reward")
    mean_after = cloud(after_actions, ACCENT, "after reward")
    ax_c.annotate("", xy=mean_after, xytext=mean_before,
                  arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2, alpha=0.85))
    ax_c.axhline(0, color=GRID, lw=0.8)
    ax_c.axvline(0, color=GRID, lw=0.8)
    ax_c.set_xlabel("dx", color=DIM, fontsize=8.5)
    ax_c.set_ylabel("dy", color=DIM, fontsize=8.5)
    ax_c.grid(color=GRID, lw=0.6)
    ax_c.set_axisbelow(True)
    legend = ax_c.legend(frameon=False, fontsize=8, loc="best")
    for text in legend.get_texts():
        text.set_color(DIM)
    shift = float(np.hypot(mean_after[0] - mean_before[0], mean_after[1] - mean_before[1]))
    title(ax_c, "C", "A payment moves the decision now",
          f"same input, 14 decisions each side · centre moved {shift:.2f}\n"
          "this step is the gain change, not the learning")

    fig.text(0.05, 0.935, "mousecortex", color=ACCENT, fontsize=16, family="monospace")
    fig.text(0.05, 0.900, label, color=DIM, fontsize=8.5, family="monospace")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # No creator, no timestamp, no machine: the file says nothing about who made it.
    fig.savefig(out_path, dpi=150, facecolor=PAPER, metadata={"Software": None})
    print(f"wrote {out_path} — {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
