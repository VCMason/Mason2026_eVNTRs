#!/usr/bin/env python3
import argparse
import os
import gzip
import pandas as pd
import matplotlib.pyplot as plt

mm = 1 / 25.4  # mm to inches conversion

# -------------------------------
# File reading
# -------------------------------
def read_file(file_path):
    """Read a TSV or TSV.GZ file into a pandas DataFrame."""
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as f:
            df = pd.read_csv(f, sep="\t")
    else:
        df = pd.read_csv(file_path, sep="\t")
    return df

# -------------------------------
# Processing functions
# -------------------------------
def subtract_two_columns(df):
    """Prompt for two column names, subtract col2 from col1, store in 'result'."""
    print("Current df shape:", df.shape)
    col1 = input("Enter the first column name (col1): ").strip()
    col2 = input("Enter the second column name (col2): ").strip()

    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"One or both columns '{col1}', '{col2}' not found in DataFrame.")

    df["result"] = df[col1] - df[col2]
    print(f"Created 'result' column as {col1} - {col2}")
    return df


def exclude_rows_with_value(df):
    """Exclude all rows where a column has a specific value."""
    print("Current df shape:", df.shape)
    col = input("Enter the column name to filter: ").strip()
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    value_str = input(f"Enter the value in '{col}' to exclude: ").strip()

    # Try to convert value to numeric if possible
    try:
        value = float(value_str)
    except ValueError:
        value = value_str  # keep as string

    before_count = len(df)
    df = df[df[col] != value]
    after_count = len(df)

    print(f"Excluded {before_count - after_count} rows where {col} == {value}")
    print("Current df shape (post exclusion):", df.shape)
    return df

# -------------------------------
# Plotting functions
# -------------------------------
def plot_hist(df, output_dir, prefix, fig_width_mm, fig_height_mm):
    """Plot histogram of the 'result' column with optional x-axis limit."""
    if "result" not in df.columns:
        raise ValueError("'result' column not found. Run a processing step first.")

    bins_str = input("Enter number of bins for histogram: ").strip()
    try:
        bins = int(bins_str)
    except ValueError:
        bins = 10  # default

    x_limit_str = input("Enter maximum x-axis value (blank for no limit): ").strip()
    try:
        x_limit = float(x_limit_str) if x_limit_str else None
    except ValueError:
        x_limit = None

    # plt.rcParams.update({
    #     "axes.titlesize": 12,
    #     "axes.labelsize": 6,
    #     "xtick.labelsize": 5,
    #     "ytick.labelsize": 5,
    # })

    font_size = 10
    font_family = "Arial"
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2
    })

    plt.figure(figsize=(fig_width_mm*mm, fig_height_mm*mm))
    df["result"].hist(bins=bins)
    if x_limit is not None:
        plt.xlim(df["result"].min(), x_limit)

    plt.xlabel("Result")
    plt.ylabel("Frequency")
    plt.title("Histogram of Result")
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{prefix}_hist.pdf")
    plt.savefig(output_path)
    # output_path = os.path.join(output_dir, "result_hist.png")
    # plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Histogram saved to {output_path}")
    return output_path


def plot_bar(df, output_dir, prefix, fig_width_mm, fig_height_mm):
    """Plot histogram of the 'result' column with optional x-axis limit."""
    if "result" not in df.columns:
        raise ValueError("'result' column not found. Run a processing step first.")

    print("Making bar plot of 'result' column")

    x_limit_str = input("Enter maximum x-axis value (blank for no limit): ").strip()
    try:
        x_limit = float(x_limit_str) if x_limit_str else None
    except ValueError:
        x_limit = None

    font_size = 10
    font_family = "Arial"
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2
    })

    plt.figure(figsize=(fig_width_mm*mm, fig_height_mm*mm))
    if pd.api.types.is_integer_dtype(df["result"]) or all(df["result"].dropna() % 1 == 0):
        counts = df["result"].value_counts().sort_index()
        if x_limit is not None:
            counts = counts[counts.index <= x_limit]
    plt.bar(counts.index, counts.values)
    
    plt.xlabel("Result")
    plt.ylabel("Number of Occurrences")
    plt.title("Barplot of Result")
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{prefix}_bar.pdf")
    plt.savefig(output_path)
    # output_path = os.path.join(output_dir, "result_hist.png")
    # plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Histogram saved to {output_path}")
    return output_path


def plot_hist_break_x_axis(df, output_dir, prefix, fig_width_mm, fig_height_mm):
    """Plot two histograms side-by-side with a break in the x-axis."""
    if "result" not in df.columns:
        raise ValueError("'result' column not found. Run a processing step first.")

    break_point_str = input("Enter the x-axis break point value: ").strip()
    try:
        break_point = float(break_point_str)
    except ValueError:
        raise ValueError("Invalid break point value. Must be numeric.")

    bins_str = input("Enter number of bins for histogram: ").strip()
    try:
        bins = int(bins_str)
    except ValueError:
        bins = 10  # default

    font_size = 10
    font_family = "Arial"
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2
    })
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(fig_width_mm*mm, fig_height_mm*mm),
        sharey=True
    )

    ax1.hist(df.loc[df["result"] <= break_point, "result"].dropna(), bins=bins)
    ax1.set_xlim(df["result"].min(), break_point)
    ax1.set_title(f"<= {break_point}")
    ax1.set_xlabel("Result")
    ax1.set_ylabel("Frequency")

    ax2.hist(df.loc[df["result"] > break_point, "result"].dropna(), bins=bins)
    ax2.set_xlim(break_point, df["result"].max())
    ax2.set_title(f"> {break_point}")
    ax2.set_xlabel("Result")

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_hist_break.pdf")
    plt.savefig(output_path)
    # output_path = os.path.join(output_dir, "result_hist_break.png")
    # plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Broken-axis histogram saved to {output_path}")
    return output_path

# -------------------------------
# Summary statistics
# -------------------------------
def summary_statistics(df, plot_path):
    """Print mean, median, min, max, std for 'result' column and save to .log file."""
    if "result" not in df.columns:
        raise ValueError("'result' column not found. Run a processing step first.")

    stats = {
        "mean": df["result"].mean(),
        "median": df["result"].median(),
        "min": df["result"].min(),
        "max": df["result"].max(),
        "std": df["result"].std()
    }

    print("\nSummary Statistics for 'result':")
    for k, v in stats.items():
        print(f"{k.capitalize():<8}: {v:.4f}")

    log_path = os.path.splitext(plot_path)[0] + ".log"
    with open(log_path, "w") as f:
        f.write("Summary Statistics for 'result':\n")
        for k, v in stats.items():
            f.write(f"{k.capitalize():<8}: {v:.4f}\n")

    print(f"Summary statistics saved to {log_path}")

# -------------------------------
# Function registries
# -------------------------------
PROCESS_FUNCTIONS = {
    "subtract_two_columns": subtract_two_columns,
    "exclude_rows_with_value": exclude_rows_with_value
}

PLOT_FUNCTIONS = {
    "plot_hist": plot_hist,
    "plot_bar": plot_bar,
    "plot_hist_break_x_axis": plot_hist_break_x_axis
}

# -------------------------------
# Main
# -------------------------------
def main():
    # usage example:
    # python script.py \
    #   --infile /path/to/data.tsv \
    #   --process exclude_rows_with_value subtract_two_columns \
    #   --plot plot_hist \
    #   --fig-width-mm 150 \
    #   --fig-height-mm 80

    parser = argparse.ArgumentParser(description="Process and plot TSV data.")
    parser.add_argument("--infile", required=True, help="Full path to the .tsv or .tsv.gz file")
    parser.add_argument("--process", nargs="+", choices=PROCESS_FUNCTIONS.keys(), required=True, help="Processing functions to run (space-separated in order)")
    parser.add_argument("--prefix", default="ProcessAndPlotResult", help="Prefix for output files (default: empty)")
    parser.add_argument("--plot", choices=PLOT_FUNCTIONS.keys(), required=True, help="Plotting function to run")
    parser.add_argument("--fig_width_mm", type=float, default=120, help="Figure width in mm (default: 120)")
    parser.add_argument("--fig_height_mm", type=float, default=80, help="Figure height in mm (default: 80)")
    args = parser.parse_args()

    df = read_file(args.infile)

    # Apply process functions in sequence
    for proc_name in args.process:
        df = PROCESS_FUNCTIONS[proc_name](df)

    output_dir = os.path.dirname(os.path.abspath(args.infile))
    plot_path = PLOT_FUNCTIONS[args.plot](df, output_dir, args.prefix, args.fig_width_mm, args.fig_height_mm)

    summary_statistics(df, plot_path)


if __name__ == "__main__":
    main()
