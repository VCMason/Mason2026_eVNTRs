"""
samtools_get_cram_stats_postprocess.py

This script combines individual per-sample cram stats files into a single TSV,
then adds a new column (AverageSharedVNTRAllelePercentage) calculated from
a pairwise matrix CSV.

Usage:
    python samtools_get_cram_stats_postprocess.py \
        --results_dir results \
        --pairwise_matrix pairwise_matrix.csv \
        --metadata_tsv sample_metadata.tsv \
        --output final_results.tsv \
        --plots_prefix final_results


example:
    python samtools_get_cram_stats_postprocess.py \
        --results_dir results \
        --pairwise_matrix /cluster/work/pausch/vmason/analyses/trgt/v2.1.0/120Hifi94Calves18Ass/output/cluster/TRdatabase/SplitByChromosome/Autosomes/SimilarityMatrixRaw_SharedAlleleProportions_LociCount752930.csv \
        --metadata_tsv /cluster/work/pausch/vmason/analyses/trgt/v2.1.0/120Hifi94Calves18Ass/output/cluster/TRdatabase/SplitByChromosome/SamplesGenotypedForVNTRs.tsv \
        --output final_results.tsv \
        --plots_prefix FinalResults
"""

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# --------------------------
# Step 1: Concatenate results/*.tsv
# --------------------------
def concatenate_results(results_dir: str, output_file: str):
    """
    Concatenate all per-sample TSV files into a single file with header:
    Sample, AverageDepth, Divergence
    """
    import glob
    files = glob.glob(f"{results_dir}/*.tsv")
    dfs = [pd.read_csv(f, sep="\t", header=None, names=["Sample", "AverageDepth", "Divergence"]) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_file, sep="\t", index=False)
    print(f"Concatenated {len(files)} files into {output_file}")
    return output_file

# need a function to normalize sample IDs
def normalize_id(s: str) -> str:
    """Normalize sample IDs by taking only the part before the first period '.'."""
    return str(s).strip().split('.')[0]

# --------------------------
# Step 2: Add average shared VNTR allele percentage
# --------------------------
def add_shared_allele_percentage(final_tsv: str, pairwise_matrix: str):
    """
    Reads a pairwise square matrix CSV (samples in first row/column).
    For each sample, compute average value per row (excluding diagonal 1.0).
    Adds a column AverageSharedVNTRAllelePercentage to final_tsv.
    """
    """Add AverageSharedVNTRAllelePercentage column from a pairwise matrix to the final results file."""
    df = pd.read_csv(final_tsv, sep="\t")
    matrix = pd.read_csv(pairwise_matrix, index_col=0)

    # Normalize all IDs
    df['Sample'] = df['Sample'].apply(normalize_id)
    matrix.index = matrix.index.map(normalize_id)
    matrix.columns = matrix.columns.map(normalize_id)

    # Compute per-sample average excluding diagonal
    averages = {}
    for sample in matrix.index:
        row_values = matrix.loc[sample].drop(sample, errors='ignore')
        averages[sample] = row_values.mean() if not row_values.empty else np.nan

    # Map to dataframe
    df["AverageSharedVNTRAllelePercentage"] = df["Sample"].map(averages)

    # Warn if unmatched
    missing = df[df['AverageSharedVNTRAllelePercentage'].isna()]['Sample'].tolist()
    if missing:
        print(f"WARNING: {len(missing)} samples had no match in pairwise matrix: {missing[:5]}{'...' if len(missing)>5 else ''}")

    df.to_csv(final_tsv, sep="\t", index=False)
    print(f"Added AverageSharedVNTRAllelePercentage to {final_tsv}")

# --------------------------
# Step 3: Add BreedGroup from metadata
# --------------------------
def add_breed_group(final_tsv: str, metadata_tsv: str):
    """
    Adds the BreedGroup column from metadata_tsv to final_tsv in place.
    Metadata file must have columns: ID and BreedGroup.
    """
    """Add BreedGroup column from metadata TSV to the final results file."""
    df = pd.read_csv(final_tsv, sep="\t")
    meta = pd.read_csv(metadata_tsv, sep="\t")

    if "ID" not in meta.columns or "BreedGroup" not in meta.columns:
        raise ValueError("Metadata file must contain columns: ID and BreedGroup")

    # Normalize IDs
    df['Sample'] = df['Sample'].apply(normalize_id)
    meta['ID'] = meta['ID'].apply(normalize_id)

    # Merge on normalized IDs
    merged = df.merge(meta[["ID", "BreedGroup"]], left_on="Sample", right_on="ID", how="left")
    merged.drop(columns=["ID"], inplace=True)

    # Warn if unmatched
    missing = merged[merged['BreedGroup'].isna()]['Sample'].tolist()
    if missing:
        print(f"WARNING: {len(missing)} samples had no BreedGroup match: {missing[:5]}{'...' if len(missing)>5 else ''}")

    merged.to_csv(final_tsv, sep="\t", index=False)
    print(f"Updated {final_tsv} with BreedGroup column.")


# --------------------------
# Step 3: Add BreedGroup from metadata
# --------------------------
def add_breed_group(final_tsv: str, metadata_tsv: str):
    """
    Adds the BreedGroup column from metadata_tsv to final_tsv in place.
    Metadata file must have columns: ID and BreedGroup.
    """
    df = pd.read_csv(final_tsv, sep="\t")
    meta = pd.read_csv(metadata_tsv, sep="\t")

    if "ID" not in meta.columns or "BreedGroup" not in meta.columns:
        raise ValueError("Metadata file must contain columns: ID and BreedGroup")

    merged = df.merge(meta[["ID", "BreedGroup"]], left_on="Sample", right_on="ID", how="left")
    merged.drop(columns=["ID"], inplace=True)
    merged.to_csv(final_tsv, sep="\t", index=False)
    print(f"Updated {final_tsv} with BreedGroup column.")


# --------------------------
# Step 4: Plot relationships
# --------------------------
def plot_relationships(
    final_tsv: str,
    output_prefix: str,
    low_depth_threshold: float = 10,
    figsize: tuple = (7, 3.5),
    font_size: int = 10,
    font_family: str = "Arial"
    ):
    """
    Creates two scatterplots:
    1. AverageDepth vs AverageSharedVNTRAllelePercentage (colored by BreedGroup)
    2. AverageDepth vs proportional change between shared allele percent and divergence

    Parameters
    ----------
    figsize : tuple
        Figure size in inches, e.g. (16, 6)
    font_size : int
        Base font size for plot text
    font_family : str
        Font family name, e.g. 'Arial'
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    # --------------------------
    # Load data
    # --------------------------
    df = pd.read_csv(final_tsv, sep="\t")

    # Ensure numeric types
    df['AverageDepth'] = pd.to_numeric(df['AverageDepth'], errors='coerce')
    df['AverageSharedVNTRAllelePercentage'] = pd.to_numeric(
        df['AverageSharedVNTRAllelePercentage'],
        errors='coerce'
    )
    df['Divergence'] = pd.to_numeric(df['Divergence'], errors='coerce')

    df['ProportionalChange'] = (
        (1 - df['AverageSharedVNTRAllelePercentage']) / df['Divergence']
    )

    # --------------------------
    # Correlations
    # --------------------------
    pearson1, _ = pearsonr(
        df['AverageDepth'],
        df['AverageSharedVNTRAllelePercentage']
    )
    spearman1, _ = spearmanr(
        df['AverageDepth'],
        df['AverageSharedVNTRAllelePercentage']
    )

    pearson2, _ = pearsonr(
        df['AverageDepth'],
        df['ProportionalChange']
    )
    spearman2, _ = spearmanr(
        df['AverageDepth'],
        df['ProportionalChange']
    )

    print(
        f"Depth vs Shared Allele %: "
        f"Pearson={pearson1:.3f}, Spearman={spearman1:.3f}"
    )

    print(
        f"Depth vs Proportional Change: "
        f"Pearson={pearson2:.3f}, Spearman={spearman2:.3f}"
    )

    # --------------------------
    # Low-depth flag
    # --------------------------
    df['LowDepth'] = df['AverageDepth'] < low_depth_threshold

    # --------------------------
    # Global style settings
    # --------------------------
    sns.set_style("whitegrid")

    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2
    })

    # --------------------------
    # Create figure
    # --------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1 ---
    sns.scatterplot(
        data=df,
        x='AverageDepth',
        y='AverageSharedVNTRAllelePercentage',
        hue='BreedGroup',
        style='LowDepth',
        palette='tab10',
        ax=axes[0],
        s=80  # marker size
    )

    # sns.regplot(
    #     data=df,
    #     x='AverageDepth',
    #     y='AverageSharedVNTRAllelePercentage',
    #     scatter=False,
    #     ax=axes[0],
    #     color='black',
    #     line_kws={'linewidth': 1.5}
    # )

    # axes[0].set_title('Depth vs Shared Allele %')
    axes[0].set_xlabel('Average Depth')
    axes[0].set_ylabel('Average Shared Allele %')

    # --- Plot 2 ---
    sns.scatterplot(
        data=df,
        x='AverageDepth',
        y='ProportionalChange',
        hue='BreedGroup',
        style='LowDepth',
        palette='tab10',
        ax=axes[1],
        legend=False,
        s=80
    )

    sns.regplot(
        data=df,
        x='AverageDepth',
        y='ProportionalChange',
        scatter=False,
        ax=axes[1],
        color='black',
        line_kws={'linewidth': 1.5}
    )

    # axes[1].set_title(
    #     'Depth vs Proportional Change (1 - Shared) / Divergence'
    # )
    axes[1].set_xlabel('Average Depth')
    axes[1].set_ylabel('Avearge Unique Allele % / Average Divergence')  # (1 - mean(Shared Allele %)) / mean(Divergence)

    # --------------------------
    # Save figure
    # --------------------------
    plt.tight_layout()

    output_path = f"{output_prefix}_depth_vs_shared_and_change.png"
    output_path_pdf = f"{output_prefix}_depth_vs_shared_and_change.pdf"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight'
    )
    plt.savefig(output_path_pdf)

    plt.close()

    print(f"Plots saved to {output_path}")

    return


### def plot_relationships(final_tsv: str, output_prefix: str, low_depth_threshold: float = 10):
###     """
###     Creates two scatterplots:
###     1. AverageDepth vs AverageSharedVNTRAllelePercentage (colored by BreedGroup)
###     2. AverageDepth vs proportional change between shared allele percent and divergence
###     Adds correlation coefficients to console output.
###     """
###     df = pd.read_csv(final_tsv, sep="\t")
###     # Ensure numeric types
###     df['AverageDepth'] = pd.to_numeric(df['AverageDepth'], errors='coerce')
###     df['AverageSharedVNTRAllelePercentage'] = pd.to_numeric(df['AverageSharedVNTRAllelePercentage'], errors='coerce')
###     df['Divergence'] = pd.to_numeric(df['Divergence'], errors='coerce')
###     df['ProportionalChange'] = (1 - df['AverageSharedVNTRAllelePercentage']) / df['Divergence']
### 
###     # Correlation coefficients
###     pearson1, _ = pearsonr(df['AverageDepth'], df['AverageSharedVNTRAllelePercentage'])
###     spearman1, _ = spearmanr(df['AverageDepth'], df['AverageSharedVNTRAllelePercentage'])
###     pearson2, _ = pearsonr(df['AverageDepth'], df['ProportionalChange'])
###     spearman2, _ = spearmanr(df['AverageDepth'], df['ProportionalChange'])
###     print(f"Depth vs Shared Allele %: Pearson={pearson1:.3f}, Spearman={spearman1:.3f}")
###     print(f"Depth vs Proportional Change: Pearson={pearson2:.3f}, Spearman={spearman2:.3f}")
### 
###     # Identify low-depth samples for special marker
###     df['LowDepth'] = df['AverageDepth'] < low_depth_threshold
### 
###     sns.set(style="whitegrid")
###     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
### 
###     # --- Plot 1: Depth vs Shared allele percent ---
###     sns.scatterplot(
###         data=df, x='AverageDepth', y='AverageSharedVNTRAllelePercentage',
###         hue='BreedGroup', style='LowDepth', palette='tab10', ax=axes[0]
###     )
###     sns.regplot(data=df, x='AverageDepth', y='AverageSharedVNTRAllelePercentage',
###                 scatter=False, ax=axes[0], color='black', line_kws={'linewidth':1})
###     axes[0].set_title('Depth vs Shared VNTR Allele %')
###     axes[0].set_xlabel('Average Depth')
###     axes[0].set_ylabel('Average Shared VNTR Allele %')
### 
###     # --- Plot 2: Depth vs Proportional Change ---
###     sns.scatterplot(
###         data=df, x='AverageDepth', y='ProportionalChange',
###         hue='BreedGroup', style='LowDepth', palette='tab10', ax=axes[1], legend=False
###     )
###     sns.regplot(data=df, x='AverageDepth', y='ProportionalChange',
###                 scatter=False, ax=axes[1], color='black', line_kws={'linewidth':1})
###     axes[1].set_title('Depth vs Proportional Change (1 - Shared) / Divergence')
###     axes[1].set_xlabel('Average Depth')
###     axes[1].set_ylabel('Proportional Change')
### 
###     plt.tight_layout()
###     output_path = f"{output_prefix}_depth_vs_shared_and_change.png"
###     plt.savefig(output_path, dpi=300)
###     plt.close()
###     print(f"Plots saved to {output_path}")
### 
###     return

# --------------------------
# Step 5: Plot relationships with sample labels
# --------------------------
def plot_relationships_with_labels(final_tsv: str, output_prefix: str, low_depth_threshold: float = 10):
    """
    Same as plot_relationships(), but also plots sample names (small font)
    next to each point in both scatterplots.
    """
    df = pd.read_csv(final_tsv, sep="\t")
    # Ensure numeric types
    df['AverageDepth'] = pd.to_numeric(df['AverageDepth'], errors='coerce')
    df['AverageSharedVNTRAllelePercentage'] = pd.to_numeric(df['AverageSharedVNTRAllelePercentage'], errors='coerce')
    df['Divergence'] = pd.to_numeric(df['Divergence'], errors='coerce')
    df['ProportionalChange'] = (1 - df['AverageSharedVNTRAllelePercentage']) / df['Divergence']

    # Identify low-depth samples for special marker
    df['LowDepth'] = df['AverageDepth'] < low_depth_threshold

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Depth vs Shared allele percent ---
    sns.scatterplot(
        data=df, x='AverageDepth', y='AverageSharedVNTRAllelePercentage',
        hue='BreedGroup', style='LowDepth', palette='tab10', ax=axes[0]
    )
    sns.regplot(data=df, x='AverageDepth', y='AverageSharedVNTRAllelePercentage',
                scatter=False, ax=axes[0], color='black', line_kws={'linewidth':1})
    axes[0].set_title('Depth vs Shared VNTR Allele %')
    axes[0].set_xlabel('Average Depth')
    axes[0].set_ylabel('Average Shared VNTR Allele %')

    # Add sample labels
    for _, row in df.iterrows():
        axes[0].text(row['AverageDepth'], row['AverageSharedVNTRAllelePercentage'],
                     str(row['Sample']), fontsize=6, alpha=0.7)

    # --- Plot 2: Depth vs Proportional Change ---
    sns.scatterplot(
        data=df, x='AverageDepth', y='ProportionalChange',
        hue='BreedGroup', style='LowDepth', palette='tab10', ax=axes[1], legend=False
    )
    sns.regplot(data=df, x='AverageDepth', y='ProportionalChange',
                scatter=False, ax=axes[1], color='black', line_kws={'linewidth':1})
    axes[1].set_title('Depth vs Proportional Change (1 - Shared) / Divergence')
    axes[1].set_xlabel('Average Depth')
    axes[1].set_ylabel('Proportional Change')

    # Add sample labels
    for _, row in df.iterrows():
        axes[1].text(row['AverageDepth'], row['ProportionalChange'],
                     str(row['Sample']), fontsize=6, alpha=0.7)

    plt.tight_layout()
    output_path = f"{output_prefix}_depth_vs_shared_and_change_with_labels.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plots with labels saved to {output_path}")

    return


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Merge CRAM stats, add VNTR shared allele %, BreedGroup, and plot relationships.")
    parser.add_argument("--results_dir", default=None, help="Directory with per-sample .tsv files.")
    parser.add_argument("--pairwise_matrix", default=None, help="Pairwise shared allele matrix (.csv).")
    parser.add_argument("--metadata_tsv", help="Metadata TSV with columns [ID, BreedGroup].")
    parser.add_argument("--output", required=True, help="Output final results TSV.")
    parser.add_argument("--outdir", required=False, help="Output results to outdir TSV.")
    parser.add_argument("--plots_prefix", help="Prefix for output plots.")
    args = parser.parse_args()

    print(f"Collecting results from {args.results_dir}...")
    print(f"Pairwise matrix: {args.pairwise_matrix}")
    print(f"Metadata TSV: {args.metadata_tsv if args.metadata_tsv else 'None'}")
    print(f"Output file: {args.output}")
    print(f"Plots prefix: {args.plots_prefix if args.plots_prefix else 'None'}")
    
    # Build final results
    if args.results_dir:
        concatenate_results(args.results_dir, args.output)
    if args.pairwise_matrix:
        add_shared_allele_percentage(args.output, args.pairwise_matrix)
    if args.metadata_tsv:
        add_breed_group(args.output, args.metadata_tsv)

    # Generate plots
    if args.plots_prefix:
        plot_relationships(args.output, args.plots_prefix)
        plot_relationships_with_labels(args.output, args.plots_prefix)
    else:
        print("No plots generated since --plots_prefix not provided.")

if __name__ == "__main__":
    main()

