import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# from 10hrs to 5000 hrs
T_TRACE_START = 10 * 3600
T_TRACE_END = 5000 * 3600


def parse(file_path, exp_filters, policy_filters, runs=100):
    results = {}
    # Only parse the experiments of interest in the given file path.
    for exp in exp_filters:
        # Only parse the policies of interest for each experiment.
        for p in policy_filters:
            for run in range(1, runs + 1):
                common_path = os.path.join(file_path, exp, f"run{run}/{p}")
                cluster_stats_df = pd.read_csv(
                    os.path.join(common_path, f"cluster_stats.csv")
                )
                job_stats_df = pd.read_csv(os.path.join(common_path, f"job_stats.csv"))
                results.setdefault((exp, run), {})[p] = (cluster_stats_df, job_stats_df)
    return results


def extract_avg_jcr(results, exp_filters, policy_filters):
    """
    Extract average Job Completion Rate (JCR) from the specified experiment and policy.
    """
    jcr = {}
    for (exp, _), policy_items in results.items():
        if exp not in exp_filters:
            continue
        for p, (_, job_stats_df) in policy_items.items():
            if p not in policy_filters:
                continue
            prev_rows = len(job_stats_df)
            df = job_stats_df[np.isfinite(job_stats_df["slowdown"])]
            after_rows = len(df)
            jcr.setdefault(p, []).append(after_rows / prev_rows * 100)
    return {p: np.mean(jcr[p]) for p in jcr.keys()}


def extract_avg_jct(results, exp_filters, policy_filters, ptiles):
    """
    Extract average Job Completion Time (JCT) from the specified experiment and policy.
    """
    jct = {}
    for (exp, _), policy_items in results.items():
        if exp not in exp_filters:
            continue
        for p, (_, job_stats_df) in policy_items.items():
            if p not in policy_filters:
                continue
            df_jct = job_stats_df[np.isfinite(job_stats_df["slowdown"])]
            series = (
                df_jct[
                    (df_jct["arrival time (sec)"] >= T_TRACE_START)
                    & (df_jct["arrival time (sec)"] <= T_TRACE_END)
                ]["jct (sec)"]
                / 3600
            )
            for ptile in ptiles:
                jct.setdefault(p, {}).setdefault(ptile, []).append(series.quantile(ptile))
    avg_jct = {}
    for p, item in jct.items():
        avg_jct[p] = {ptile: np.mean(item[ptile]) for ptile in item.keys()}
    return avg_jct


def extract_avg_util(results, exp_filters, policy_filters, ptiles):
    """
    Extract average cluster utilization from the specified experiment and policy.
    """
    util_cdf = {}
    for (exp, _), policy_items in results.items():
        if exp not in exp_filters:
            continue
        for p, (cluster_stats_df, _) in policy_items.items():
            if p not in policy_filters:
                continue
            series = (
                cluster_stats_df[
                    (cluster_stats_df["#time (sec)"] >= T_TRACE_START)
                    & (cluster_stats_df["#time (sec)"] <= T_TRACE_END)
                ]["util"]
                * 100
            )
            for ptile in ptiles:
                util_cdf.setdefault(p, {}).setdefault(ptile, []).append(
                    series.quantile(ptile)
                )
    avg_util = {}
    for p, item in util_cdf.items():
        avg_util[p] = {ptile: np.mean(item[ptile]) for ptile in item.keys()}
    return avg_util


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python <path-to>/parser.py <stats_outdir>")
        sys.exit(1)

    exp_filters = [
        # "exp26",
        "exp28"
    ]
    policy_filter = ["firstfit", "folding", "reconfig", "rfold"]
    stats_outdir = sys.argv[1]
    # Results format: (exp, run) -> {policy: (cluster_stats_df, job_stats_df)}
    results = parse(stats_outdir, exp_filters, policy_filter, runs=100)
    avg_jcr = extract_avg_jcr(results, exp_filters, policy_filter)
    print(f"Average JCR: {avg_jcr}")
    avg_jct = extract_avg_jct(
        results, exp_filters, policy_filter, ptiles=[0.5, 0.9, 0.99]
    )
    print(f"Average JCT: {avg_jct}")
    avg_util = extract_avg_util(
        results, exp_filters, policy_filter, ptiles=np.linspace(0, 1, num=200)
    )
    print(f"Average Utilization: {avg_util}")
