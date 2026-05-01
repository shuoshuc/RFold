import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot RFold contention data.")
    parser.add_argument('--input', type=str, default='rfold-contention.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='contention.pdf', help='Output image file')
    args = parser.parse_args()

    # Read the csv file
    df = pd.read_csv(args.input)

    # Columns to plot
    columns = [
        ('no contention solo (msec)', 'Baseline'),
        ('firstfit + 4 nodes bg (msec)', 'Firstfit'),
        ('space filling curve + 4 nodes bg (msec)', 'Space Filling Curve'),
        ('L1 clustering + 4 nodes bg (msec)', 'L1 Clustering'),
        ('random + 4 nodes bg (msec)', 'Random'),
        ('TopoMatch + 4 nodes bg (msec)', 'TopoMatch')
    ]

    # Set font size
    plt.rcParams.update({'font.size': 10})

    # Create the plot
    plt.figure(figsize=(3.3, 3.3 * 0.5))

    for col, label in columns:
        plt.plot(df['scale'], df[col] / 60000, marker='o', label=label)

    # Formatting the plot
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())
    
    xticks = [4, 16, 64, 256, 1024, 4096]
    plt.xticks(xticks, xticks)
    
    yticks = [1, 5, 10, 20]
    plt.yticks(yticks, yticks)

    plt.xlabel('Nodes', labelpad=0)
    plt.ylabel('Time (minutes)', labelpad=0)
    plt.legend(ncol=2, handlelength=1.4, handletextpad=0.4, labelspacing=0.0, columnspacing=0.5,
               loc='lower right', bbox_to_anchor=(1.02, 1.01))
    plt.grid(True, which="both", axis="x", ls="--", alpha=0.5)
    plt.grid(True, which="major", axis="y", ls="--", alpha=0.5)
    
    plt.tick_params(axis='x', length=0)
    plt.tick_params(axis='y', length=1.75)

    # Save the plot
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
