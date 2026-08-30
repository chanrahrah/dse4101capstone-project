import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUTPUT_DIR = "dataset"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean")

CHANNELS = [
    "rates_yield_curve",
    "credit_spreads",
    "inflation",
    "volatility_stress",
    "real_activity_labour",
    "dollar_commodity",
]

for channel in CHANNELS:
    path = os.path.join(CLEAN_DIR, f"{channel}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    corr = df.corr()
    print(f"\n=== {channel} ===")
    print(corr.round(2))

    avg_corr = corr.abs().mean().sort_values(ascending=False)
    print("\nAverage |correlation| with other series in this channel (higher = more redundant with the rest):")
    print(avg_corr.round(2))

PLOTS_DIR = os.path.join("dataset", "clean", "correlation_plots")

CHANNELS = [
    "rates_yield_curve",
    "credit_spreads",
    "inflation",
    "volatility_stress",
    "real_activity_labour",
    "dollar_commodity",
]

os.makedirs(PLOTS_DIR, exist_ok=True)

# diverging colormap: red (negative) -> light gray (zero) -> blue (positive)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "corr_diverging", ["#e34948", "#f0efec", "#2a78d6"]
)


fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

im = None
for ax, channel in zip(axes, CHANNELS):
    path = os.path.join(CLEAN_DIR, f"{channel}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    corr = df.corr()

    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                     color=text_color, fontsize=8)

    ax.set_title(channel, fontsize=11)

fig.suptitle("Within-channel correlation matrices", fontsize=14, y=1.02)
fig.tight_layout(rect=[0, 0, 0.92, 1])

cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="correlation")

out_path = os.path.join(PLOTS_DIR, "all_channels_correlation.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")