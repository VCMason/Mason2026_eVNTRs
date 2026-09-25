import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest


def binomial_enrichment_test(
    distances,
    total_window=1_000_000,
    target_window=1_000
    ):
    """
    Test whether observations are enriched near 0.

    Parameters
    ----------
    distances : array-like
        Distance values (e.g. df["tss_distance"]).

    total_window : int or float
        Assumed null support is [-total_window, +total_window].

    target_window : int or float
        Enrichment region is [-target_window, +target_window].

    Returns
    -------
    dict
        Summary statistics and p-value.
    """

    s = pd.to_numeric(distances, errors="coerce").dropna()

    # total observations
    n = len(s)

    # observations in target region
    k = ((s >= -target_window) & (s <= target_window)).sum()

    # expected probability under uniform null
    p0 = target_window / total_window

    # expected count
    expected = n * p0

    # exact one-sided enrichment test
    result = binomtest(
        k=k,
        n=n,
        p=p0,
        alternative="greater"
    )

    fold_enrichment = k / expected if expected > 0 else float("nan")
    
    dbintest = {
        "n_observations": n,
        "observed_in_window": k,
        "expected_in_window": expected,
        "expected_fraction": p0,
        "observed_fraction": k / n,
        "fold_enrichment": fold_enrichment,
        "p_value": result.pvalue,
    }

    print(f"Number of observations: {dbintest['n_observations']}\n"
          f"Number observations observed in target window: {dbintest['observed_in_window']}\n"
          f"Number observations expected in target window: {dbintest['expected_in_window']}\n"
          f"Fold enrichment (observed / expected) in target region: {dbintest['fold_enrichment']}\n"
          f"Binomial enrichment test pvalue: {dbintest['p_value']}")
    
    return dbintest


def plot_tss_distance_histogram(df, label, outstem, dir, bins=100):
    """
    Plot histogram of TSS_distance column.
    Saves .png and .pdf.

    df: pandas DataFrame
    label: title label (e.g. "File1")
    outstem: Path stem (no extension)
    dir: Output directory
    """

    result = {"n": 0, "mean": None, "median": None}

    if "tss_distance" not in df.columns:
        print(f"{label}: tss_distance column not found, skipping plot")
        return result

    # convert to numeric safely
    s = pd.to_numeric(df["tss_distance"], errors="coerce").dropna()

    if s.empty:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.text(0.5, 0.5, "no tss_distance values", ha="center", va="center")
        ax.set_title(label)
        fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{outstem}.pdf", bbox_inches="tight")
        plt.close(fig)
        return result

    dbintest = binomial_enrichment_test(s) # returns dictionary
    
    result["n"] = int(len(s))
    result["mean"] = float(s.mean())
    result["median"] = float(s.median())

    fig, ax = plt.subplots(figsize=(3.0,1.5))  # 3, 1.75 # (6,4) is default
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

    #bin_edges = np.linspace(-1, 1, bins + 1)
    ax.hist(s, bins=bins, alpha=0.8)
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("TSS_distance")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} (n={len(s):,})")

    # mark mean and median, pval
    # ax.axvline(s.mean(), linestyle="--", linewidth=1, label=f"mean: {s.mean():.1f}")
    # ax.axvline(s.median(), linestyle=":", linewidth=1, label="median")
    # p_text = f"p < 1e-300" if pval == 0 else f"p = {pval:.2e}"
    p_text = f"p < 1e-300" if dbintest['p_value'] == 0 else f"p = {dbintest['p_value']:.2e}"
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
        label=f"mean: {s.mean():.0f}\nmedian: {s.median():.0f}"
    )
    ax.plot([], [], ' ', label=f"{p_text}")
    #\n{p_text}# added pvalue to label as well to get it in legend
    # 2. Hide the line handle by setting its length to 0
    ax.legend(handlelength=0, handletextpad=0)
    # ax.legend()

    outpng = os.path.join(dir, f"{outstem}.png")
    outpdf = os.path.join(dir, f"{outstem}.pdf")
    fig.savefig(outpng, dpi=150, bbox_inches="tight")
    fig.savefig(outpdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # make zoom in plot
    zoomwindow = 50000
    # dbintest = binomial_enrichment_test(s, total_window=zoomwindow) # returns dictionary. # uncomment if pvalue of zoom window desired
    # zoom window is primarily for visualization purposes
    fig, ax = plt.subplots(figsize=(3.0,1.5))  # 3, 1.75 # (6,4) is default
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

    #bin_edges = np.linspace(-1, 1, bins + 1)
    ax.hist(s, bins=bins, range=(-zoomwindow, zoomwindow), alpha=0.8)
    # Set the x-axis limits to +-50,000
    ax.set_xlim(-zoomwindow, zoomwindow)

    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("TSS_distance")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} (n={len(s):,})")

    # mark mean and median, pval
    # ax.axvline(s.mean(), linestyle="--", linewidth=1, label=f"mean: {s.mean():.1f}")
    # ax.axvline(s.median(), linestyle=":", linewidth=1, label="median")
    # p_text = f"p < 1e-300" if pval == 0 else f"p = {pval:.2e}"
    p_text = f"p < 1e-300" if dbintest['p_value'] == 0 else f"p = {dbintest['p_value']:.2e}"
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
        label=f"mean: {s.mean():.0f}\nmedian: {s.median():.0f}"
    )
    ax.plot([], [], ' ', label=f"{p_text}")
    #\n{p_text}# added pvalue to label as well to get it in legend
    # Hide the line handle by setting its length to 0
    ax.legend(handlelength=0, handletextpad=0)
    # ax.legend()

    outpng = os.path.join(dir, f"{outstem}.ZoomIn.png")
    outpdf = os.path.join(dir, f"{outstem}.ZoomIn.pdf")
    fig.savefig(outpng, dpi=150, bbox_inches="tight")
    fig.savefig(outpdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return result


if __name__=="__main__":
    '''
    USAGE:
    python PlotTSSDistance.py \
    --infile gene_SNP_INDEL_association_results.Autosomes.BestVariantperGene.LessEqualTo_1e-07.tsv \
    --dir /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP_XenaSNPINDELs/output_Zraw_NoRareAlleles_ML
    --label Best SNP/INDEL \
    --outprefix TSS_Distance_BestSNPINDELPerGene.1e-07 \
    --bins 30 \
    --dimensions 6,4
    '''

    # argparse read file path
    p=argparse.ArgumentParser()
    p.add_argument("--dir", default=os.getcwd(), help="example: /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP/output_MeanLength_NOT_LMM")
    p.add_argument("--infile", required=True, help="example: gene_vntr_association_results.No0StandBeta.BestVNTRperGene.LessEqualTo_1e-07.Autosomes.tsv")
    p.add_argument("--label", default="Best_Variant")
    p.add_argument("--outprefix", default="TSS_Distance_Histogram")
    p.add_argument("--bins", type=int, default=100)
    # p.add_argument("--dimensions", default="6,4", help="Figure dimensions in inches, e.g. 6,4")
    args = p.parse_args()

    f = os.path.join(args.dir, args.infile)
    print(f"Reading in {f}")
    df = pd.read_csv(f, sep="\t", dtype=str)
    # fix column names to lowercase for easier access
    df.columns = df.columns.str.lower()
    plot_tss_distance_histogram(df, label=args.label, outstem=args.outprefix, dir=args.dir, bins=args.bins)
