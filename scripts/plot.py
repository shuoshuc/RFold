import matplotlib.pyplot as plt
import pandas as pd
import sys
import os


def run_config(run_id: int) -> str:
    configs = {
        0: "sim=50hr, dim=16x16x16, job=philly",
        16: "sim=100hr, dim=16x16x16, job=philly",
        17: "sim=100hr, dim=16x16x16, job=ali20",
        19: "sim=100hr, dim=16x16x16, job=acme",
        28: "sim=100hr, dim=32x32x32, job=philly",
    }
    return configs[run_id]


def load(stats_outdir: str, run_id: list[int]) -> tuple[dict, dict]:
    job_stats, cluster_stats = {}, {}
    for run in run_id:
        job_stats[run] = pd.read_csv(
            os.path.join(stats_outdir, f"run{run}", "job_stats.csv")
        )
        cluster_stats[run] = pd.read_csv(
            os.path.join(stats_outdir, f"run{run}", "cluster_stats.csv")
        )
    return job_stats, cluster_stats


def plot_job(job_stats: dict):
    # Disable all pandas warnings.
    pd.options.mode.chained_assignment = None
    fig, axs = plt.subplots(
        len(job_stats), 3, figsize=(12, 3 * len(job_stats)), layout="constrained"
    )
    for i, (run, df) in enumerate(job_stats.items()):
        df = df[df["queueing (sec)"].notna() & (df["queueing (sec)"] != 0)]

        ax = axs[i][0]
        ax.ecdf(df["wait on shape (sec)"] / 3600, label="wait on shape")
        ax.ecdf(df["wait on resource (sec)"] / 3600, label="wait on resource")
        ax.set_xlabel("time (hr)")
        ax.set_ylabel("CDF")
        ax.grid()
        ax.legend(loc="lower right")

        ax = axs[i][1]
        df["shape frac"] = df["wait on shape (sec)"] / df["queueing (sec)"] * 100
        df["resource frac"] = df["wait on resource (sec)"] / df["queueing (sec)"] * 100
        df = df.dropna(subset=["shape frac", "resource frac"])
        ax.ecdf(df["shape frac"], label="wait on shape")
        ax.ecdf(df["resource frac"], label="wait on resource")
        ax.set_xlabel("wait time / total queueing time (%)")
        ax.set_ylabel("CDF")
        ax.set_title(run_config(run))
        ax.grid()
        ax.legend(loc="lower right")

        ax = axs[i][2]
        df = df[df["jct (sec)"].notna() & (df["jct (sec)"] != 0)]
        df["shape / jct"] = df["wait on shape (sec)"] / df["jct (sec)"] * 100
        df["resource / jct"] = df["wait on resource (sec)"] / df["jct (sec)"] * 100
        ax.ecdf(df["shape / jct"], label="wait on shape")
        ax.ecdf(df["resource / jct"], label="wait on resource")
        ax.set_xlabel("wait time / jct (%)")
        ax.set_ylabel("CDF")
        ax.grid()
        ax.legend(loc="lower right")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot.py <stats_outdir>")
        sys.exit(1)

    stats_outdir = sys.argv[1]
    run_id = [0, 16, 17, 19, 28]
    job_stats, cluster_stats = load(stats_outdir, run_id)
    plot_job(job_stats)
