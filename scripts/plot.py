import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
LINESTYLES = ["-", "--", "-.", ":"]

policy = ["reconfig", "folding"]

# from 10hrs to 5000 hrs
T_TRACE_START = 10 * 3600
# T_TRACE_END = 1600 * 3600
T_TRACE_END = 5000 * 3600

labels = {
    "exp9": "1D [1, 64]",
    "exp10": "1D [64, 128]",
    "exp11": "1D [128, 256]",
    "exp12": "2D [2, 256]",
    "exp13": "2D [256, 512]",
    "exp14": "2D [512, 1024]",
    "exp15": "3D [2, 512]",
    "exp16": "3D [512, 1024]",
    "exp17": "3D [1024, 2048]",
    "exp18": "1D+2D+3D (uniform)",
    "exp19": "1D+2D+3D (exp)",
    "exp20": "3D (ISCA paper)",
    "exp21": "3D (ISCA, no sub-cube)",
    "exp22": "3D (ISCA, only sub-cube)",
    "exp23": "uniform (only sub-cube)",
}


def parse(file_path):
    results = {}
    for exp in labels.keys():
        for p in policy:
            cluster_stats_df = pd.read_csv(
                os.path.join(file_path, exp, f"{p}/cluster_stats.csv")
            )
            job_stats_df = pd.read_csv(os.path.join(file_path, exp, f"{p}/job_stats.csv"))
            results.setdefault(exp, {})[p] = (cluster_stats_df, job_stats_df)
    return results


def plot_jct(results, exps, show_queueing=True):
    upper_bound = 0
    cols = 2 if show_queueing else 1
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 3), layout="constrained")

    # Plot CDF of JCT.
    ax = axes[0] if show_queueing else axes
    colors = iter(COLORS)
    for exp_id in exps:
        color = next(colors)
        for i, p in enumerate(policy):
            df = results[exp_id][p][1]
            series = (
                df[
                    (df["arrival time (sec)"] >= T_TRACE_START)
                    & (df["arrival time (sec)"] <= T_TRACE_END)
                ]["jct (sec)"]
                / 3600
            )
            upper_bound = max(upper_bound, series.quantile(0.95))
            print(
                f"{exp_id} {p}\np50: {series.quantile(0.5):.2f}, "
                f"p90: {series.quantile(0.9):.2f}, "
                f"p99: {series.quantile(0.99):.2f}, "
                f"max: {series.max():.2f}"
            )
            ax.ecdf(
                series,
                label=f"{labels[exp_id]} {p}",
                color=color,
                linestyle=LINESTYLES[i],
            )
    ax.set_xlabel("Job completion time (hr)")
    ax.set_ylabel("CDF")
    ax.set_xlim([-0.05 * upper_bound, upper_bound])
    ax.set_ylim([-0.02, 1.02])
    ax.legend()
    ax.grid()

    # Plot CDF of queueing.
    if show_queueing:
        colors = iter(COLORS)
        ax = axes[1]
        for exp_id in exps:
            color = next(colors)
            for i, p in enumerate(policy):
                df = results[exp_id][p][1]
                clip = df[
                    (df["arrival time (sec)"] >= T_TRACE_START)
                    & (df["arrival time (sec)"] <= T_TRACE_END)
                ]
                series = clip["queueing (sec)"] / clip["jct (sec)"] * 100
                ax.ecdf(
                    series,
                    label=f"{labels[exp_id]} {p}",
                    color=color,
                    linestyle=LINESTYLES[i],
                )
        ax.set_xlabel("Queueing time / job completion time (%)")
        ax.set_ylabel("CDF")
        ax.set_xlim([-2, 102])
        ax.set_ylim([-0.02, 1.02])
        ax.legend()
        ax.grid()

    plt.show()


def plot_util(results, exps):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), layout="constrained")
    colors = iter(COLORS)
    for exp_id in exps:
        color = next(colors)
        for i, p in enumerate(policy):
            df = results[exp_id][p][0]
            # Plot the timeseries of utilization.
            # ax.plot(
            #     df["util"].rolling(window=200).mean(),
            #     label=f"{labels[exp_id]} {p}",
            # )
            ax.ecdf(
                df["util"] * 100,
                label=f"{labels[exp_id]} {p}",
                color=color,
                linestyle=LINESTYLES[i],
            )
    ax.set_xlabel("Cluster utilization (%)")
    ax.set_ylabel("CDF")
    ax.set_xlim([-2, 102])
    ax.set_ylim([-0.02, 1.02])
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot.py <stats_outdir>")
        sys.exit(1)

    stats_outdir = sys.argv[1]
    results = parse(stats_outdir)
    exps = ["exp18", "exp19"]
    # exps = ["exp21", "exp22", "exp23"]
    plot_jct(results, exps, show_queueing=True)
    plot_util(results, exps)
