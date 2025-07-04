import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import warnings

from parser import (
    parse,
    extract_avg_jcr,
    extract_avg_jct,
    extract_avg_util,
)

warnings.filterwarnings("ignore")

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
LINESTYLES = [":", "--", "-.", "-"]
HATCHES = [None, "/", ".", "\\"]
FIG_W, FIG_H = 2.6, 1.5
FONTSIZE = 10


def plot_jct_bar(jct_bars, legend_map=None):
    plt.rcParams["font.size"] = FONTSIZE
    bar_width = 0.15
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H), layout="constrained")

    colors = iter(COLORS)
    hatches = iter(HATCHES)
    percentiles = ["p50", "p90", "p99"]
    x = np.arange(len(percentiles))
    policies = list(jct_bars.keys())
    for i, p in enumerate(policies):
        color = next(colors)
        hatch = next(hatches)
        offset = (i - (len(policies) - 1) / 2) * bar_width
        bars = [jct_bars[p][ptile] for ptile in [0.5, 0.9, 0.99]]
        bars = ax.bar(
            x + offset,
            bars,
            width=bar_width,
            label=legend_map[p] if legend_map else p,
            color=color,
            hatch=hatch,
        )

    ax.set_ylabel("Avg JCT (hours)")
    ax.set_xticks(x, percentiles)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim([-0.6, 2.5])
    # ax.set_ylim([0, 250])
    ax.set_yscale("log")
    ax.legend(
        handlelength=0.8,
        labelspacing=-0.2,
        handletextpad=0.2,
        borderaxespad=0.3,
        borderpad=0.1,
        frameon=False,
        fontsize=FONTSIZE,
        # ncol=2,
        bbox_to_anchor=(0, 1.025),
        loc="upper left",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # plt.show()
    plot_file_name = "jct.pdf"
    plt.savefig(plot_file_name, bbox_inches="tight")


def plot_util(util_cdf, legend_map=None):
    plt.rcParams["font.size"] = FONTSIZE
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H), layout="constrained")

    colors = iter(COLORS)
    linestyles = iter(LINESTYLES)
    policies = list(util_cdf.keys())
    for p in policies:
        color = next(colors)
        linestyle = next(linestyles)
        ax.ecdf(
            util_cdf[p],
            label=f"{legend_map[p] if legend_map else p}",
            color=color,
            linestyle=linestyle,
        )

    ax.set_xlabel("Avg cluster utilization (%)")
    ax.set_ylabel("CDF")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim([-5, 105])
    ax.set_xticks([x for x in range(0, 101, 20)])
    ax.set_ylim([-0.02, 1.02])
    ax.set_yticks([y for y in np.arange(0, 1.02, 0.2)])
    ax.legend(
        handlelength=0.9,
        labelspacing=-0.1,
        handletextpad=0.2,
        borderaxespad=0.2,
        frameon=True,
        fontsize=FONTSIZE,
        borderpad=0.1,
    )
    ax.grid(linestyle="--", alpha=0.7)

    # plt.show()
    plot_file_name = "util.pdf"
    plt.savefig(plot_file_name, bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python <path-to>/plot.py <stats_outdir>")
        sys.exit(1)

    exp_filters = [
        "exp26",
        # "exp28"
    ]
    policy_filter = ["firstfit", "folding", "reconfig", "rfold"]
    stats_outdir = sys.argv[1]
    # Results format: (exp, run) -> {policy: (cluster_stats_df, job_stats_df)}
    results = parse(stats_outdir, exp_filters, policy_filter, runs=100)
    avg_jcr = extract_avg_jcr(results, exp_filters, policy_filter)
    print(f"Average JCR: {avg_jcr}")

    # JCT percentiles
    # ptiles = [0.5, 0.9, 0.99]
    # avg_jct = extract_avg_jct(results, exp_filters, policy_filter, ptiles=ptiles)
    # plot_jct_bar(avg_jct)

    # Plot custom JCT bars
    bars = {
        r"RFold ($2^3$)": {
            0.5: 0.3896,
            0.9: 9.29,
            0.99: 224.267,
        },
        r"Reconfig ($2^3$)": {
            0.5: 0.446,
            0.9: 12.39,
            0.99: 231.897,
        },
        r"RFold ($4^3$)": {
            0.5: 0.47,
            0.9: 16.95,
            0.99: 241.6975,
        },
        r"Reconfig ($4^3$)": {
            0.5: 5.52,
            0.9: 101.27,
            0.99: 526.015,
        },
    }
    plot_jct_bar(bars)

    # Plot utilization CDF
    avg_util = extract_avg_util(
        results, exp_filters, policy_filter, ptiles=np.linspace(0, 1, num=200)
    )
    util_legend_map = {
        "firstfit": r"FirstFit ($16^3$)",
        "folding": r"Folding ($16^3$)",
        "reconfig": r"Reconfig ($4^3$)",
        "rfold": r"RFold ($4^3$)",
    }
    plot_util(avg_util, legend_map=util_legend_map)
