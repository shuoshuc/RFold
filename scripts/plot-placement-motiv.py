import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot RFold contention data.")
    parser.add_argument('--input', type=str, default='rfold-contention.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='static-placement-contention.pdf', help='Output image file')
    args = parser.parse_args()

    # Read the csv file
    df = pd.read_csv(args.input)

    # Columns to plot
    columns = [
        ('no contention solo (msec)', 'Ideal'),
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

    markers = ['.', 's', '^', 'v', 'o', 'P']
    baseline_col = 'no contention solo (msec)'
    for i, (col, label) in enumerate(columns):
        plt.plot(df['scale'], df[col] / df[baseline_col], marker=markers[i], markersize=5, label=label)

    # Formatting the plot
    plt.xscale('log', base=2)
    
    xticks = [4, 16, 64, 256, 1024, 4096]
    plt.xticks(xticks, xticks)
    
    plt.ylim(0.9, 2.8)
    
    yticks = [1.0, 1.5, 2.0, 2.5]
    yticklabels = ['1.0x', '1.5x', '2.0x', '2.5x']
    plt.yticks(yticks, yticklabels)
    
    plt.xlabel('Nodes', labelpad=0)
    plt.ylabel('JCT slowdown', labelpad=1)
    plt.legend(ncol=2, handlelength=1.4, handletextpad=0.4, labelspacing=0.0, columnspacing=0.5,
               loc='lower right', bbox_to_anchor=(1.02, 1.01))
    plt.grid(True, which="both", axis="x", ls="--", alpha=0.5)
    plt.grid(True, which="major", axis="y", ls="--", alpha=0.5)
    
    plt.tick_params(axis='x', length=0)
    plt.tick_params(axis='y', length=0)

    # Save the plot
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
