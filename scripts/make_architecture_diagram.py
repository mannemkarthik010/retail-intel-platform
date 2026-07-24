"""Simple architecture diagram for the polished report (boxes + arrows,
matplotlib patches -- no external diagramming tool needed)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(exist_ok=True, parents=True)

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#898781"

fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis("off")
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")


def box(x, y, w, h, text, color, textcolor="white", fontsize=10.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=0, facecolor=color, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=textcolor,
            fontsize=fontsize, weight="bold", zorder=3, linespacing=1.4)
    return (x, y, w, h)


def arrow(b1, b2, **kw):
    x1 = b1[0] + b1[2] / 2
    y1 = b1[1]
    x2 = b2[0] + b2[2] / 2
    y2 = b2[1] + b2[3]
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.6, color=MUTED, zorder=1, **kw)
    ax.add_patch(a)


# OFFLINE (batch) row
data = box(0.3, 5.0, 2.0, 0.9, "data/\ngenerate_data.py\n(synthetic dataset)", BLUE)
pipe = box(2.7, 5.0, 2.6, 0.9, "scripts/run_pipeline.py\nbacktest + anomaly scan\n+ forward forecast", ORANGE)
mon = box(5.7, 5.0, 2.6, 0.9, "scripts/\nrun_monitoring_sim.py\ndrift detection", AQUA)

arrow(data, pipe, connectionstyle="arc3,rad=0")
ax.annotate("", xy=(pipe[0]+pipe[2]+0.05, pipe[1]+pipe[3]/2), xytext=(mon[0]-0.05, mon[1]+mon[3]/2),
            arrowprops=dict(arrowstyle="<|-", color=MUTED, lw=1.6))

# reports layer
rep = box(2.7, 3.55, 4.6, 0.75, "reports/*.csv  (series_summary, anomaly_flags,\ncurrent_forecasts, monitoring logs)", "#52514e", fontsize=9.5)
arrow(pipe, rep)
arrow(mon, rep)

# online layer
agent = box(1.0, 2.1, 2.4, 0.85, "src/agent.py\ntool-calling agent\n+ audit trail", BLUE)
api = box(3.8, 2.1, 2.4, 0.85, "app/server.py\nFlask API", ORANGE)
arrow(rep, agent)
arrow(rep, api)

dash = box(2.4, 0.7, 2.4, 0.85, "app/templates/\ndashboard.html", AQUA)
arrow(api, dash)
arrow(agent, api)

# labels for the two zones
ax.text(9.6, 5.45, "OFFLINE\n(batch)", ha="right", va="center", fontsize=10, color=MUTED, style="italic", weight="bold")
ax.text(9.6, 1.4, "ONLINE\n(serving)", ha="right", va="center", fontsize=10, color=MUTED, style="italic", weight="bold")
ax.axhline(3.3, color="#c3c2b7", linewidth=1, linestyle=(0, (4, 3)), xmin=0.02, xmax=0.98)

fig.tight_layout()
fig.savefig(FIG / "architecture_diagram.png", dpi=160)
print("Wrote", FIG / "architecture_diagram.png")
