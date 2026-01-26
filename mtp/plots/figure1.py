import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as pe

from adjustText import adjust_text

import matplotlib.font_manager as fm

# Add the msttcorefonts directory
fm.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/times.ttf')

sns.set_theme(style='white', rc={"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 14})

df = pd.read_csv('mtpc2.csv')

# Example: df has columns 'acceptance_rate' (x) and 'latency' (y)
ax = sns.scatterplot(data=df, x='acceptance_rate', y='latency', hue='Model', style='LoRA Layers',
    s=200,            # increase dot size (area)
    alpha=0.8,        # optional: transparency helps with overlap
    # edgecolor="black",
    linewidth=0.5
)

ax.text(1.0, 1.02, 'Throughput',
        transform=ax.transAxes,
        fontsize=18, fontfamily="Times New Roman",
        ha='right', va='bottom')

ax.set_xlim(5.0, 7.7)
ax.set_ylim(0.028, 0.044)

# Freeze current limits so adding the line doesn't rescale the plot
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()


cs = range(156, 202, 3)
x = np.linspace(*ax.get_xlim(), 200)
for c in cs:
    ax.plot(x, x/c, ls='--', lw=0.5, c='black', alpha=0.5)

# Labels between adjacent lines, placed to the RIGHT of the Axes
trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

for c_lo, c_hi in zip(cs[:-1], cs[1:]):
    y_lo = x1 / c_lo
    y_hi = x1 / c_hi
    y_mid = np.sqrt(y_lo * y_hi) if ax.get_yscale() == "log" else 0.5 * (y_lo + y_hi)

    if y0 <= y_mid <= y1:
        ax.annotate(
            f"{c_hi}",
            xy=(1.0, y_mid), xycoords=trans,   # right edge in Axes coords
            xytext=(6, 0), textcoords="offset points",  # move outside to the right
            ha="left", va="center",
            fontsize=13, fontfamily="Times New Roman",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            clip_on=False  # allow drawing outside the Axes
        )

# Label each point (slight offset to avoid covering the marker)
# Create text objects first
texts = []
for _, r in df.iterrows():
    if not(r["Model"] == "HMM" and r['n'] == 16) and not (r["Model"] == "BTree" and r['n'] == 16):
        t = ax.text(
            r["acceptance_rate"],
            r["latency"],
            r["Model"] + " " + str(r['n']),
            fontsize=14,
            fontfamily="Times New Roman"
        )
    if (r["Model"] == "BTree" and r['n'] == 16):
        t = ax.text(
            r["acceptance_rate"],
            r["latency"],
            r["Model"] + " " + str(r['n']),
            fontsize=14,
            fontfamily="Times New Roman"
        )
    if (r["Model"] == "HMM" and r['n'] == 16):
        t = ax.text(
            r["acceptance_rate"],
            r["latency"],
            r["Model"] + " " + str(r['n']),
            fontsize=14,
            fontfamily="Times New Roman"
        )
    texts.append(t)

# Adjust to reduce collisions; add faint leader lines
adjust_text(
    texts, ax=ax,
    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
    only_move={"texts":"xy"}  # optional: constrain movement
)

# 1) Build a diagonal gradient in Axes coordinates (0..1 by 0..1)
nx = ny = 512
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
G = (X + (1 - Y)) / 2.0  # 0 at top-left, 1 at bottom-right

# # 2) Colormap from light gray → white
# cmap = LinearSegmentedColormap.from_list("lightgray_to_white", ["#e6e6e6", "#ffffff"])

# # 3) Draw behind the plot, locked to the Axes box
# ax.imshow(
#     G,
#     extent=(0, 1, 0, 1),      # cover the whole Axes
#     origin="lower",           # matches our Y definition above
#     transform=ax.transAxes,   # in Axes coords, independent of data limits
#     cmap=cmap,
#     interpolation="bilinear", # smooth gradient
#     zorder=0,
#     aspect="auto",
#     alpha=1.0                 # lower if you want it subtler (e.g., 0.8)
# )

# Restore limits and show legend
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect('auto')
ax.legend()

ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)

# After creating your scatter plot:
ax.set_xlabel("Accepted tokens →", fontfamily="Times New Roman", labelpad=8, fontsize=24)
ax.set_ylabel("← Latency (ms)", fontfamily="Times New Roman", labelpad=8, fontsize=24)

plt.savefig('mtpc.pdf', bbox_inches="tight")
