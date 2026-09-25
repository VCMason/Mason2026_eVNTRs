#!/usr/bin/env python3
"""
compare_vntr_files.py  — updated

Adds:
 - distribution plots for p-values of unique (non-shared) gene:VNTR pairs per file
 - counts of how many unique pairs are significant (p < alpha)
 - saves both .png and .pdf files for all plots

Usage:
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr

# ---------- utilities ----------

def normalize_columns(df):
    mapping = {c.strip().lower(): c for c in df.columns}
    df = df.rename(columns={v: k for k, v in mapping.items()})
    return df

def to_float_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    def conv(x):
        if pd.isna(x):
            return np.nan
        xs = str(x).strip()
        if xs == "":
            return np.nan
        if xs.startswith("<") or xs.startswith(">"):
            xs = xs[1:]
        xs = xs.replace(",", "")
        if xs.lower() in ("na", "nan", "none"):
            return np.nan
        try:
            return float(xs)
        except Exception:
            return np.nan
    return s.map(conv).astype(float)

def make_scatter_with_density(x, y, xlabel, ylabel, title, outstem, bins=150, figsize=(8,4)):
    """scatter + right-hand 2D histogram, saves .png and .pdf"""
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = np.asarray(x[mask]), np.asarray(y[mask])
    if x.size == 0:
        return
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3,1], wspace=0.25)
    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(x, y, s=6, alpha=0.3)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.set_title(title)

    # Add diagonal line of equality
    minv, maxv = np.nanmin([x, y]), np.nanmax([x, y])
    ax0.plot([minv, maxv], [minv, maxv], '--', color="black", lw=1.0, label="y = x")  # r-- b--
    ax0.legend(loc="upper left", frameon=False)

    ax1 = fig.add_subplot(gs[1])
    try:
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
        Hlog = np.log10(H + 1)
        pcm = ax1.pcolormesh(xedges, yedges, Hlog.T)
        plt.colorbar(pcm, ax=ax1, orientation="vertical", pad=0.02, label="log10(count+1)")
        ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_xlabel("density")
    except Exception as e:
        ax1.text(0.5,0.5,str(e),ha="center")
    # plt.tight_layout()
    fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{outstem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

def print_extremes(s, name):
    s = s.dropna()
    if s.empty:
        return f"{name}: no data"
    return (f"{name}: n={len(s):,}, min={s.min():.3e}, 1%={np.percentile(s,1):.3e}, "
            f"median={np.median(s):.3e}, 99%={np.percentile(s,99):.3e}, max={s.max():.3e}")


def plot_logp_proportion(pv1, pv2, allele1, allele2, label, outstem, bins=100):
    """
    Histogram of proportional log-p contribution:
        -log10(pv2) / ( -log10(pv1) + -log10(pv2) )

    pv1, pv2: pandas Series of raw p-values for shared VNTRs
    n_alleles_1, n_alleles_2: pandas Series of number of alleles for shared VNTRs
    """
    if allele1 is None or allele2 is None:
        df = pd.concat([pv1, pv2], axis=1).dropna().astype(float)
        df.columns = ["pv1", "pv2"]
        print("Warning: allele count data not provided, skipping allele means in plot annotations")
    else:
        df = pd.concat([pv1, pv2, allele1, allele2], axis=1).dropna().astype(float)
        df.columns = ["pv1", "pv2", "n_alleles_1", "n_alleles_2"]

    result = {"n": int(len(df))}
    if df.empty:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.text(0.5, 0.5, "no paired pvalues", ha="center", va="center")
        ax.set_title(label)
        fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{outstem}.pdf", bbox_inches="tight")
        plt.close(fig)
        return result

    eps = 1e-300
    logp1 = -np.log10(np.clip(df["pv1"], eps, None))
    logp2 = -np.log10(np.clip(df["pv2"], eps, None))
    
    denom = logp1 + logp2 + 1e-10
    prop = logp2 / denom
    
    # keep alignment with df
    prop = prop.replace([np.inf, -np.inf], np.nan)

    # logp1 = -np.log10(np.clip(df["pv1"].values, eps, None))
    # logp2 = -np.log10(np.clip(df["pv2"].values, eps, None))
    # 
    # denom = logp1 + logp2 + 1e-10  # avoid div by zero
    # prop = logp2 / denom
    # prop = prop[np.isfinite(prop)]

    # equal_to_zero = (prop <= 0.01).sum()
    # less_than_half = ( (prop < 0.49) & (prop > 0.01) ).sum()
    # more_than_half = ( (prop > 0.51) & (prop < 0.99) ).sum()
    # equal_to_one = (prop >= 0.99).sum()
    # equal_to_half = ( (prop >= 0.49) & (prop <= 0.51) ).sum()
    # 
    # equal_to_zero_allele1 = df.loc[(prop <= 0.01).index, "n_alleles_1"].mean()
    # equal_to_zero_allele2 = df.loc[(prop <= 0.01).index, "n_alleles_2"].mean()
    # less_than_half_allele1 = df.loc[( (prop < 0.49) & (prop > 0.01) ).index, "n_alleles_1"].mean()
    # less_than_half_allele2 = df.loc[( (prop < 0.49) & (prop > 0.01) ).index, "n_alleles_2"].mean()
    # more_than_half_allele1 = df.loc[( (prop > 0.51) & (prop < 0.99) ).index, "n_alleles_1"].mean()
    # more_than_half_allele2 = df.loc[( (prop > 0.51) & (prop < 0.99) ).index, "n_alleles_2"].mean()
    # equal_to_one_allele1 = df.loc[(prop >= 0.99).index, "n_alleles_1"].mean()
    # equal_to_one_allele2 = df.loc[(prop >= 0.99).index, "n_alleles_2"].mean()

    mask_zero = prop <= 0.05
    mask_low  = (prop > 0.05) & (prop < 0.45)
    mask_half = (prop >= 0.45) & (prop <= 0.55)
    mask_high = (prop > 0.55) & (prop < 0.95)
    mask_one  = prop >= 0.95

    equal_to_zero = mask_zero.sum()
    less_than_half = mask_low.sum()
    equal_to_half = mask_half.sum()
    more_than_half = mask_high.sum()
    equal_to_one = mask_one.sum()

    # if allele1 is not None and allele2 is not None:
    #     equal_to_zero_allele1 = df.loc[mask_zero, "n_alleles_1"].mean()
    #     equal_to_zero_allele2 = df.loc[mask_zero, "n_alleles_2"].mean()
    # 
    #     less_than_half_allele1 = df.loc[mask_low, "n_alleles_1"].mean()
    #     less_than_half_allele2 = df.loc[mask_low, "n_alleles_2"].mean()
    # 
    #     equal_to_half_allele1 = df.loc[mask_half, "n_alleles_1"].mean()
    #     equal_to_half_allele2 = df.loc[mask_half, "n_alleles_2"].mean()
    # 
    #     more_than_half_allele1 = df.loc[mask_high, "n_alleles_1"].mean()
    #     more_than_half_allele2 = df.loc[mask_high, "n_alleles_2"].mean()
    # 
    #     equal_to_one_allele1 = df.loc[mask_one, "n_alleles_1"].mean()
    #     equal_to_one_allele2 = df.loc[mask_one, "n_alleles_2"].mean()

    result.update({
        "mean": float(np.mean(prop)),
        "median": float(np.median(prop)),
        "frac_file2_dominant": float(np.mean(prop > 0.5))
    })

    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlim(0, 1)
    ax.hist(prop, bins=bins, alpha=0.8)
    ax.set_xlabel(r"$-\log_{10}(p_2) / [ -\log_{10}(p_1) + -\log_{10}(p_2) ]$")
    ax.set_ylabel("count")
    ax.set_title(f"{label} (n={len(prop):,})")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.text(0.5, ax.get_ylim()[1]*0.9, f"p1=p2\n(n={equal_to_half:,})\n",
            ha="center", va="top", color="black", fontsize=8)
    ax.text(0.25, ax.get_ylim()[1]*0.9, f"p1<p2\n(n={less_than_half:,})\n",
            ha="center", va="top", color="red", fontsize=8)
    ax.text(0.75, ax.get_ylim()[1]*0.9, f"p2<p1\n(n={more_than_half:,})\n",
            ha="center", va="top", color="red", fontsize=8)
    ax.text(0.0, ax.get_ylim()[1]*0.9, f"p1<<p2\n(n={equal_to_zero:,})\n",
            ha="left", va="top", color="red", fontsize=8)
    ax.text(1.00, ax.get_ylim()[1]*0.9, f"p2<<p1\n(n={equal_to_one:,})\n",
            ha="right", va="top", color="red", fontsize=8)
    # if allele1 is None and allele2 is None:
    #     ax.text(0.5, ax.get_ylim()[1]*0.9, f"~equal p-values\n(n={equal_to_half:,})\n",
    #             ha="center", va="top", color="black", fontsize=8)
    #     ax.text(0.25, ax.get_ylim()[1]*0.9, f"file1 more sig.\n(n={less_than_half:,})\n",
    #             ha="center", va="top", color="red", fontsize=8)
    #     ax.text(0.75, ax.get_ylim()[1]*0.9, f"file2 more sig.\n(n={more_than_half:,})\n",
    #             ha="center", va="top", color="red", fontsize=8)
    #     ax.text(0.0, ax.get_ylim()[1]*0.9, f"file1 more sig.\n(n={equal_to_zero:,})\n",
    #             ha="left", va="top", color="red", fontsize=8)
    #     ax.text(1.00, ax.get_ylim()[1]*0.9, f"file2 more sig.\n(n={equal_to_one:,})\n",
    #             ha="right", va="top", color="red", fontsize=8)
    # else:
    #     ax.text(0.5, ax.get_ylim()[1]*0.9, f"~equal p-values\n(n={equal_to_half:,})\n"
    #             f"{equal_to_half_allele1:.1f} alleles (file1)\n"
    #             f"{equal_to_half_allele2:.1f} alleles (file2)",
    #             ha="center", va="top", color="black", fontsize=8)
    #     ax.text(0.25, ax.get_ylim()[1]*0.9, f"file1 more sig.\n(n={less_than_half:,})\n"
    #             f"{less_than_half_allele1:.1f} alleles (file1)\n"
    #             f"{less_than_half_allele2:.1f} alleles (file2)",
    #             ha="center", va="top", color="red", fontsize=8)
    #     ax.text(0.75, ax.get_ylim()[1]*0.9, f"file2 more sig.\n(n={more_than_half:,})\n"
    #             f"{more_than_half_allele1:.1f} alleles (file1)\n"
    #             f"{more_than_half_allele2:.1f} alleles (file2)",
    #             ha="center", va="top", color="red", fontsize=8)
    #     ax.text(0.0, ax.get_ylim()[1]*0.9, f"file1 more sig.\n(n={equal_to_zero:,})\n"
    #             f"{equal_to_zero_allele1:.1f} alleles (file1)\n"
    #             f"{equal_to_zero_allele2:.1f} alleles (file2)",
    #             ha="left", va="top", color="red", fontsize=8)
    #     ax.text(1.00, ax.get_ylim()[1]*0.9, f"file2 more sig.\n(n={equal_to_one:,})\n"
    #             f"{equal_to_one_allele1:.1f} alleles (file1)\n"
    #             f"{equal_to_one_allele2:.1f} alleles (file2)",
    #             ha="right", va="top", color="red", fontsize=8)
    fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{outstem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return result


def plot_logp_difference(pv1, pv2, label, outstem, bins=100):
    """
    Histogram of difference in log-p:
        -log10(pv1) - -log10(pv2)

    pv1, pv2: pandas Series of raw p-values for shared VNTRs
    """
    df = pd.concat([pv1, pv2], axis=1).dropna().astype(float)
    df.columns = ["pv1", "pv2"]

    result = {"n": int(len(df))}
    if df.empty:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.text(0.5, 0.5, "no paired pvalues", ha="center", va="center")
        ax.set_title(label)
        fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{outstem}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return result

    eps = 1e-300
    diff = (
        -np.log10(np.clip(df["pv1"].values, eps, None))
        - (-np.log10(np.clip(df["pv2"].values, eps, None)))
    )

    result.update({
        "mean": float(np.mean(diff)),
        "median": float(np.median(diff)),
        "frac_file1_more_sig": float(np.mean(diff > 0))
    })

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(diff, bins=bins, alpha=0.8)
    ax.set_xlabel(r"$-\log_{10}(p_1) - -\log_{10}(p_2)$")
    ax.set_ylabel("count")
    ax.set_title(f"{label} (n={len(diff):,})")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.text(0, ax.get_ylim()[1]*0.9, "p1 = p2",
            ha="center", va="top", color="red", fontsize=8)

    fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{outstem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return result


def draw_backet(x1, x2, ypos, h, ax):
    """Draw a bracket from (x1, ypos) to (x2, ypos) with height h on ax."""
    xhalf = (x1 + x2) / 2.0
    ax.plot([x1, x1], [ypos, ypos-h], color="black", lw=1)
    ax.plot([x2, x2], [ypos, ypos-h], color="black", lw=1)
    ax.plot([xhalf, xhalf], [ypos, ypos+0.5*h], color="black", lw=1)
    ax.plot([x1, x2], [ypos, ypos], color="black", lw=1)
    return


def plot_logp_normalized_difference(pv1, pv2, allele1, allele2, label, outstem, fixymax=None, bins=30):
    """
    Histogram of normalized log-p difference:
        ( -log10(pv1) - -log10(pv2) ) / ( -log10(pv1) + -log10(pv2) + 1 )


    Range: [-1, +1]
      +1  -> file1 dominates
       0  -> equal significance
      -1  -> file2 dominates
    """
    if allele1 is None or allele2 is None:
        df = pd.concat([pv1, pv2], axis=1).dropna().astype(float)
        df.columns = ["pv1", "pv2"]
        print("Warning: allele count data not provided, skipping allele means in plot annotations")
    else:
        df = pd.concat([pv1, pv2, allele1, allele2], axis=1).dropna().astype(float)
        df.columns = ["pv1", "pv2", "n_alleles_1", "n_alleles_2"]
    
    # to avoid issues when log10(1) = 0, which can cause the normalized difference to be exactly +1 or -1 and mess up the histogram bins
    # df[['pv1', 'pv2']] = df[['pv1', 'pv2']].replace(1.0, 0.999)

    result = {"n": int(len(df))}
    if df.empty:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.text(0.5, 0.5, "no paired pvalues", ha="center", va="center")
        ax.set_title(label)
        fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{outstem}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return result

    eps = 1e-300
    logp1 = -np.log10(np.clip(df["pv1"], eps, None))
    logp2 = -np.log10(np.clip(df["pv2"], eps, None))

    denom = logp1 + logp2
    # this is real normalized difference
    normdiff = (logp1 - logp2) / (denom + 1)  # add 1 to denominator. This prevents solutions = 1 or -1 when both p-values are 1 and reduces effect when both pvalues are close to 1, while still giving a strong signal when one p-value is much smaller than the other.
    # weighted difference, this makes two pvalues close to 1 ~0 and large changes in small pvalues large numbers (not bound by -1 -> 1), hence the need for tanh below
    weighted_diff = (logp1 - logp2) * denom
    # Squash to [-1, 1] using tanh
    # Adjust 'k' to control sensitivity. 
    # A higher k (e.g., 50) requires a more "interesting" p-value to reach 1.
    k = 10
    tanh_diff = np.tanh(weighted_diff / k)  # naming as norm diff, but really weighted differences bounded to -1 -> 1 by tanh transformation
    # df['final_metric'] = tanh_weighted_diff
    # keep index alignment
    tanh_diff = tanh_diff.replace([np.inf, -np.inf], np.nan)

    # define masks once
    mask_neg_one = tanh_diff <= -0.9
    mask_neg = (tanh_diff < -0.05) & (tanh_diff > -0.9)
    mask_zero = (tanh_diff >= -0.05) & (tanh_diff <= 0.05)
    mask_pos = (tanh_diff > 0.05) & (tanh_diff < 0.9)
    mask_pos_one = tanh_diff >= 0.9

    # counts
    equal_to_neg_one = mask_neg_one.sum()
    less_than_zero = mask_neg.sum()
    equal_to_zero = mask_zero.sum()
    more_than_zero = mask_pos.sum()
    equal_to_one = mask_pos_one.sum()

    # allele means
    def mean_or_nan(mask, col):
        return df.loc[mask, col].mean()

    # if allele1 is not None or allele2 is not None:
    #     equal_to_neg_one_allele1 = mean_or_nan(mask_neg_one, "n_alleles_1")
    #     equal_to_neg_one_allele2 = mean_or_nan(mask_neg_one, "n_alleles_2")
    # 
    #     less_than_zero_allele1 = mean_or_nan(mask_neg, "n_alleles_1")
    #     less_than_zero_allele2 = mean_or_nan(mask_neg, "n_alleles_2")
    # 
    #     equal_to_zero_allele1 = mean_or_nan(mask_zero, "n_alleles_1")
    #     equal_to_zero_allele2 = mean_or_nan(mask_zero, "n_alleles_2")
    # 
    #     more_than_zero_allele1 = mean_or_nan(mask_pos, "n_alleles_1")
    #     more_than_zero_allele2 = mean_or_nan(mask_pos, "n_alleles_2")
    # 
    #     equal_to_one_allele1 = mean_or_nan(mask_pos_one, "n_alleles_1")
    #     equal_to_one_allele2 = mean_or_nan(mask_pos_one, "n_alleles_2")

    result.update({
        "mean": float(tanh_diff.mean()),
        "median": float(tanh_diff.median()),
        "frac_file1_dominant": float((tanh_diff > 0).mean()),
        "frac_strong_disagreement": float((tanh_diff.abs() > 0.5).mean())
    })

    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlim(-1, 1)
    ax.hist(tanh_diff.dropna(), bins=bins, alpha=0.8)
    ax.set_xlabel(
        r"$tanh( ( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) ) / k )$"
    )
    # normalized difference
    # r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) + 1 )$"
    # weighted difference
    # r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) * ( -\log_{10}(p_1) + -\log_{10}(p_2) )$"
    # tanh(weighted difference / k)
    # r"$tanh( ( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) ) / k )$"
    ax.set_ylabel("count")
    ax.set_title(f"{label} (n={len(tanh_diff.dropna()):,}, k={k})")

    ymax = ax.get_ylim()[1]
    # draw backet for each region
    ypos = ymax * 0.85
    h = ymax * 0.02   # height of the bracket "arms"
    # draw_backet(-1.0, -0.9, ypos, h, ax)
    draw_backet(-0.9, -0.1, ypos, h, ax)
    # draw_backet(-0.05, 0.05, ypos, h, ax)
    draw_backet(0.1, 0.9, ypos, h, ax)
    # draw_backet(0.9, 1.0, ypos, h, ax)
    # draw vertical lines instead of brackets
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    # ax.axvline(0.5, color="red", linestyle=":", linewidth=1)
    # ax.axvline(-0.5, color="red", linestyle=":", linewidth=1)

    ax.text(0, ymax*0.95,
            f"p1 = p2\n{equal_to_zero}\n",
            ha="center", va="top", fontsize=8, color="red")
    ax.text(0.5, ymax*0.95,
            f"p1 < p2\n{more_than_zero}\n",
            ha="center", va="top", fontsize=8, color="red")
    ax.text(-0.5, ymax*0.95,
            f"p2 < p1\n{less_than_zero}\n",
            ha="center", va="top", fontsize=8, color="red")
    ax.text(1.0, ymax*0.95,
            f"p1 << p2\n{equal_to_one}\n",
            ha="right", va="top", fontsize=8, color="red")
    ax.text(-1.0, ymax*0.95,
            f"p2 << p1\n{equal_to_neg_one}\n",
            ha="left", va="top", fontsize=8, color="red")
    # if allele1 is None or allele2 is None:
    #     ax.text(0, ymax*0.95,
    #             f"p1 = p2\n{equal_to_zero}\n",
    #             ha="center", va="top", fontsize=8, color="red")
    # 
    #     ax.text(0.5, ymax*0.95,
    #             f"p1 < p2\n{more_than_zero}\n",
    #             ha="center", va="top", fontsize=8, color="red")
    #
    #     ax.text(-0.5, ymax*0.95,
    #             f"p2 < p1\n{less_than_zero}\n",
    #             ha="center", va="top", fontsize=8, color="red")
    # 
    #     ax.text(1.0, ymax*0.95,
    #             f"p1 << p2\n{equal_to_one}\n",
    #             ha="right", va="top", fontsize=8, color="red")
    # 
    #     ax.text(-1.0, ymax*0.95,
    #             f"p2 << p1\n{equal_to_neg_one}\n",
    #             ha="left", va="top", fontsize=8, color="red")
    # else:
    #     ax.text(0, ymax*0.95,
    #             f"p1 = p2\n{equal_to_zero}\n"
    #             f"{equal_to_zero_allele1:.1f} alleles (file1)\n"
    #             f"{equal_to_zero_allele2:.1f} alleles (file2)",
    #             ha="center", va="top", fontsize=8)
    # 
    #     ax.text(0.5, ymax*0.95,
    #             f"p1 < p2\n{more_than_zero}\n"
    #             f"{more_than_zero_allele1:.1f} alleles (file1)\n"
    #             f"{more_than_zero_allele2:.1f} alleles (file2)",
    #             ha="center", va="top", fontsize=8, color="red")
    # 
    #     ax.text(-0.5, ymax*0.95,
    #             f"p2 < p1\n{less_than_zero}\n"
    #             f"{less_than_zero_allele1:.1f} alleles (file1)\n"
    #             f"{less_than_zero_allele2:.1f} alleles (file2)",
    #             ha="center", va="top", fontsize=8, color="red")
    # 
    #     ax.text(1.0, ymax*0.95,
    #             f"p1 << p2\n{equal_to_one}\n"
    #             f"{equal_to_one_allele1:.1f} alleles (file1)\n"
    #             f"{equal_to_one_allele2:.1f} alleles (file2)",
    #             ha="right", va="top", fontsize=8, color="red")
    # 
    #     ax.text(-1.0, ymax*0.95,
    #             f"p2 << p1\n{equal_to_neg_one}\n"
    #             f"{equal_to_neg_one_allele1:.1f} alleles (file1)\n"
    #             f"{equal_to_neg_one_allele2:.1f} alleles (file2)",
    #             ha="left", va="top", fontsize=8, color="red")

    fig.savefig(f"{outstem}_tanh_k{k}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{outstem}_tanh_k{k}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # plot weighted difference histogram (values before tanh transformation above)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(weighted_diff.dropna(), bins=bins, alpha=0.8)
    ax.set_xlabel(
        r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) * ( -\log_{10}(p_1) + -\log_{10}(p_2) )$"
    )
    # normalized difference
    # r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) + 1 )$"
    # weighted difference
    # r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) * ( -\log_{10}(p_1) + -\log_{10}(p_2) )$"
    # tanh(weighted difference / k)
    # r"$tanh( ( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) ) / k )$"
    ax.set_ylabel("count")
    ax.set_title(f"{label} (n={len(weighted_diff.dropna()):,})")
    fig.savefig(f"{outstem}_weighted_diff.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{outstem}_weighted_diff.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # plot normalized difference histogram here, +1 in denominator is importance to focus on 'larger' chnages in small pvalues
    # keep index alignment
    normdiff = normdiff.replace([np.inf, -np.inf], np.nan)

    # define masks once
    mask_neg_one = normdiff <= -0.75
    mask_neg = (normdiff < -0.25) & (normdiff > -0.75)
    mask_zero = (normdiff >= -0.25) & (normdiff <= 0.25)
    mask_pos = (normdiff > 0.25) & (normdiff < 0.75)
    mask_pos_one = normdiff >= 0.75

    # counts
    equal_to_neg_one = mask_neg_one.sum()
    less_than_zero = mask_neg.sum()
    equal_to_zero = mask_zero.sum()
    more_than_zero = mask_pos.sum()
    equal_to_one = mask_pos_one.sum()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlim(-1, 1)
    ax.hist(normdiff.dropna(), bins=bins, alpha=0.8)
    ax.set_xlabel(
        r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) + 1 )$"
    )
    # normalized difference
    # r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) + 1 )$"
    # weighted difference
    # r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) * ( -\log_{10}(p_1) + -\log_{10}(p_2) )$"
    # tanh(weighted difference / k)
    #r"$tanh( ( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) ) / k )$"
    ax.set_ylabel("count")
    ax.set_title(f"{label} (n={len(normdiff.dropna()):,})")

    ymax = ax.get_ylim()[1]
    # draw backet for each region
    ypos = ymax * 0.85
    h = ymax * 0.02   # height of the bracket "arms"
    draw_backet(-0.75, -0.25, ypos, h, ax)
    draw_backet(0.25, 0.75, ypos, h, ax)
    # draw vertical lines instead of brackets
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.text(0, ymax*0.95,
            f"p1 = p2\n{equal_to_zero}\n",
            ha="center", va="top", fontsize=8, color="red")
    ax.text(0.5, ymax*0.95,
            f"p1 < p2\n{more_than_zero}\n",
            ha="center", va="top", fontsize=8, color="red")
    ax.text(-0.5, ymax*0.95,
            f"p2 < p1\n{less_than_zero}\n",
            ha="center", va="top", fontsize=8, color="red")
    ax.text(1.0, ymax*0.95,
            f"p1 << p2\n{equal_to_one}\n",
            ha="right", va="top", fontsize=8, color="red")
    ax.text(-1.0, ymax*0.95,
            f"p2 << p1\n{equal_to_neg_one}\n",
            ha="left", va="top", fontsize=8, color="red")

    fig.savefig(f"{outstem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{outstem}.pdf")
    plt.close(fig)

    #####
    # fig for paper
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
    fig, ax = plt.subplots(figsize=(3.5,2.5))
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['-1', '0', 1])
    bin_edges = np.linspace(-1, 1, bins + 1)
    ax.hist(normdiff.dropna(), bins=bin_edges, alpha=0.8)
    ax.set_xlabel("Weighted Normalized Difference")
    # # This is the formula for weighted normalized difference
    # ax.set_xlabel(
    #     r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) + 1 )$"
    # )
    # normalized difference
    # r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) + 1 )$"
    # weighted difference
    # r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) * ( -\log_{10}(p_1) + -\log_{10}(p_2) )$"
    # tanh(weighted difference / k)
    #r"$tanh( ( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) ) / k )$"
    ax.set_ylabel("Count")
    # ax.set_title(f"{label} n={len(normdiff.dropna()):,}")
    # ax.set_title(f"Comparisons = {len(normdiff.dropna()):,}")
    # ax.set_title("")

    ymin, ymax = ax.get_ylim()
    # this line should be used when running the program normally (fixing the ymax value to 2600 below is only for the paper)
    # ymax = ymax + 0.2*ymax
    if fixymax:
        ymax = fixymax
    ax.set_ylim(ymin, ymax)
    # draw backet for each region
    ypos = ymax * 0.85
    h = ymax * 0.02   # height of the bracket "arms"
    draw_backet(-0.75, -0.25, ypos, h, ax)
    draw_backet(0.25, 0.75, ypos, h, ax)
    # draw vertical lines instead of brackets
    ax.axvline(0, color="grey", linestyle="--", linewidth=1, alpha=0.25)

    ax.text(0, ymax*0.98,
            f"p1=p2\n{equal_to_zero}\n",
            ha="center", va="top", fontsize=8, color="black")
    ax.text(0.50, ymax*0.98,
            f"p1<p2\n{more_than_zero}\n",
            ha="center", va="top", fontsize=8, color="black") # ha="right"
    ax.text(-0.50, ymax*0.98,
            f"p2<p1\n{less_than_zero}\n",
            ha="center", va="top", fontsize=8, color="black") # ha="left"
    ax.text(0.99, ymax*0.98,
            f"p1<<p2\n{equal_to_one}\n",
            ha="right", va="top", fontsize=8, color="black")
    ax.text(-0.99, ymax*0.98,
            f"p2<<p1\n{equal_to_neg_one}\n",
            ha="left", va="top", fontsize=8, color="black")

    # plt.tight_layout()
    # # Adjust margins manually (values are percentages of the figure size)
    # fig.subplots_adjust(bottom=0.18, left=0.18, top=0.90, right=0.95)
    fig.savefig(f"{outstem}.ForPaper.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{outstem}.ForPaper.pdf", bbox_inches="tight")
    plt.close(fig)

    return result


# def plot_logp_normalized_difference(pv1, pv2, allele1, allele2, label, outstem, bins=100):
#     """
#     Histogram of normalized log-p difference:
#         ( -log10(pv1) - -log10(pv2) ) / ( -log10(pv1) + -log10(pv2) )
# 
#     pv1, pv2: pandas Series of raw p-values for shared VNTRs
#     Range: [-1, +1]
#       +1  -> file1 dominates
#        0  -> equal significance
#       -1  -> file2 dominates
# 
#     Returns dict with summary stats.
#     """
#     df = pd.concat([pv1, pv2, allele1, allele2], axis=1).dropna().astype(float)
#     df.columns = ["pv1", "pv2", "n_alleles_1", "n_alleles_2"]
# 
#     result = {"n": int(len(df))}
#     if df.empty:
#         fig, ax = plt.subplots(figsize=(5,3))
#         ax.text(0.5, 0.5, "no paired pvalues", ha="center", va="center")
#         ax.set_title(label)
#         fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
#         fig.savefig(f"{outstem}.pdf", dpi=300, bbox_inches="tight")
#         plt.close(fig)
#         return result
# 
#     eps = 1e-300
#     logp1 = -np.log10(np.clip(df["pv1"].values, eps, None))
#     logp2 = -np.log10(np.clip(df["pv2"].values, eps, None))
# 
#     denom = logp1 + logp2
#     normdiff = (logp1 - logp2) / denom
#     normdiff = normdiff[np.isfinite(normdiff)]
# 
#     equal_to_neg_one = (normdiff <= -0.99).sum()
#     less_than_zero = ( (normdiff < -0.01) & (normdiff > -0.99) ).sum()
#     more_than_zero = ( (normdiff > 0.01) & (normdiff < 0.99) ).sum()
#     equal_to_one = (normdiff >= 0.99).sum()
#     equal_to_zero = ( (normdiff >= 0.49) & (normdiff <= 0.51) ).sum()
# 
#     result.update({
#         "mean": float(np.mean(normdiff)),
#         "median": float(np.median(normdiff)),
#         "frac_file1_dominant": float(np.mean(normdiff > 0)),
#         "frac_strong_disagreement": float(np.mean(np.abs(normdiff) > 0.25))
#     })
# 
#     fig, ax = plt.subplots(figsize=(6,4))
#     ax.hist(normdiff, bins=bins, alpha=0.8)
#     ax.set_xlabel(
#         r"$( -\log_{10}(p_1) - -\log_{10}(p_2) ) / ( -\log_{10}(p_1) + -\log_{10}(p_2) )$"
#     )
#     ax.set_ylabel("count")
#     ax.set_title(f"{label} (n={len(normdiff):,})")
# 
#     ax.axvline(0, color="black", linestyle="--", linewidth=1)
#     ax.axvline(0.5, color="red", linestyle=":", linewidth=1)
#     ax.axvline(-0.5, color="red", linestyle=":", linewidth=1)
# 
#     ax.text(0, ax.get_ylim()[1]*0.75, f"Equal p-values\n{equal_to_zero}",
#             ha="center", va="top", fontsize=8)
#     ax.text(0.5, ax.get_ylim()[1]*0.75, f"File1 more sig.\n{more_than_zero}",
#             ha="center", va="top", fontsize=8, color="red")
#     ax.text(-0.5, ax.get_ylim()[1]*0.75, f"File2 more sig.\n{less_than_zero}",
#             ha="center", va="top", fontsize=8, color="red")
#     ax.text(1.0, ax.get_ylim()[1]*0.95, f"File1 much more sig.\n{equal_to_one}",
#             ha="right", va="top", fontsize=8, color="red")
#     ax.text(-1.0, ax.get_ylim()[1]*0.95, f"File2 much more sig.\n{equal_to_neg_one}",
#             ha="left", va="top", fontsize=8, color="red")
# 
#     fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
#     fig.savefig(f"{outstem}.pdf", dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     return result


def plot_unique_pvalue_distributions(pvals_series, label, outstem, alpha=1e-6, bins=100):
    """
    pvals_series: pandas Series of raw p-values (already float)
    label: 'unique_to_file1' / 'unique_to_file2' or similar
    outstem: Path or str stem for saving files (without extension)
    Saves two-panel figure (.png and .pdf):
      - left: histogram of -log10(p)
      - right: ECDF: cumulative fraction <= p (x axis log scale)
    Returns dictionary with counts and fraction significant.
    """
    s = pvals_series.dropna().astype(float)
    result = {"n": int(len(s)), "n_significant": 0, "frac_significant": 0.0}
    if s.empty:
        # make an empty placeholder plot
        fig, axs = plt.subplots(1,2, figsize=(8,3))
        axs[0].text(0.5,0.5,"no pvalues", ha='center', va='center')
        axs[1].text(0.5,0.5,"no pvalues", ha='center', va='center')
        plt.suptitle(label)
        fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{outstem}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return result

    # handle zeros by clipping to eps
    eps = 1e-300
    s_clipped = np.clip(s, a_min=eps, a_max=None)
    # compute -log10
    minuslog = -np.log10(s_clipped)

    # counts
    n = len(s)
    n_sig = (s < alpha).sum()
    result["n"] = int(n)
    result["n_significant"] = int(n_sig)
    result["frac_significant"] = float(n_sig / n) if n>0 else 0.0

    # prepare figure: left histogram of -log10(p), right ECDF vs p (log-x)
    fig = plt.figure(figsize=(9,4))
    gs = gridspec.GridSpec(1,2, width_ratios=[2.5,1.5], wspace=0.3)

    ax0 = fig.add_subplot(gs[0])
    # histogram of -log10(p)
    ax0.hist(minuslog, bins=bins, alpha=0.8)
    ax0.set_xlabel("-log10(p)")
    ax0.set_ylabel("count")
    ax0.set_title(f"{label}: -log10(p) (n={n:,}, sig={n_sig:,})")
    # mark alpha on the -log10 scale
    ax0.axvline(-np.log10(alpha), color='red', linestyle='--', linewidth=1)
    ax0.text(0.95, 0.95, f"alpha={alpha:.0e}", transform=ax0.transAxes,
             ha='right', va='top', fontsize=8, color='red')

    ax1 = fig.add_subplot(gs[1])
    # ECDF: sort s and compute cumulative fraction
    s_sorted = np.sort(s_clipped)
    ecdf = np.arange(1, len(s_sorted)+1) / len(s_sorted)
    ax1.step(s_sorted, ecdf, where='post')
    ax1.set_xscale('log')
    ax1.set_xlabel("p-value (log scale)")
    ax1.set_ylabel("cumulative fraction")
    ax1.set_title("ECDF")
    # mark alpha
    ax1.axvline(alpha, color='red', linestyle='--', linewidth=1)
    ax1.text(alpha, 0.05, f"alpha={alpha:.0e}", color='red', rotation=90, va='bottom', fontsize=8)

    # plt.tight_layout()
    fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{outstem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return result


def plot_tss_distance_histogram(df, label, outstem, bins=100):
    """
    Plot histogram of TSS_distance column.
    Saves .png and .pdf.

    df: pandas DataFrame
    label: title label (e.g. "File1")
    outstem: Path stem (no extension)
    """

    result = {"n": 0, "mean": None, "median": None}

    if "tss_distance" not in df.columns.str.lower():
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

    result["n"] = int(len(s))
    result["mean"] = float(s.mean())
    result["median"] = float(s.median())

    fig, ax = plt.subplots(figsize=(3,1.5))
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
    
    ax.hist(s, bins=bins, alpha=0.8)

    ax.set_xlabel("TSS_distance")
    ax.set_ylabel("Count")
    ax.set_title(f"{label}: TSS_distance (n={len(s):,})")

    # mark mean and median
    ax.axvline(s.mean(), linestyle="--", linewidth=1, label=f"mean: {s.mean():.0f}\nmedian: {s.median():.0f}")
    # ax.axvline(s.median(), linestyle=":", linewidth=1, label="median")

    ax.legend(handlelength=0, handletextpad=0)

    fig.savefig(f"{outstem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{outstem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return result


# ---------- core compare function ----------

def compare_vntr_files(file1, file2, df1, df2, dir1, dir2, outdir, fixymax=None, alpha=1e-6):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nComparing:\n  {df1.shape}\n  {df2.shape}")
    # df1 = pd.read_csv(file1, sep="\t", dtype=str)  # done in filter_f1_f2()
    # df2 = pd.read_csv(file2, sep="\t", dtype=str)  # done in filter_f1_f2()
    # df1 = normalize_columns(df1)  # done in filter_f1_f2()
    # df2 = normalize_columns(df2)  # done in filter_f1_f2()

    # convert all non-key columns to floats where possible
    for df in (df1, df2):
        for c in df.columns:
            if c not in ("geneid", "vntrid"):
                df[c] = to_float_series(df[c])

    # make pvalue distribution plots for each file including all rows
    plot_unique_pvalue_distributions(df1["raw_pvalue"], "File1", outdir / "file1_pvalues", alpha=1e-6, bins=100)
    plot_unique_pvalue_distributions(df2["raw_pvalue"], "File2", outdir / "file2_pvalues", alpha=1e-6, bins=100)
    plot_tss_distance_histogram(df1, "File1", outdir / "file1_TSS_distance_histogram", bins=100)
    plot_tss_distance_histogram(df2, "File2", outdir / "file2_TSS_distance_histogram", bins=100)

    merged = pd.merge(df1, df2, on=["geneid", "vntrid"], suffixes=("_1","_2"), how="outer", indicator=True)
    both = merged[merged["_merge"]=="both"].copy()

    # basic counts
    n1, n2 = len(df1), len(df2)
    n_common = len(both)
    n_only1 = (merged["_merge"]=="left_only").sum()
    n_only2 = (merged["_merge"]=="right_only").sum()

    # unique genes & vntrs counts
    genes1 = set(df1["geneid"].dropna())
    genes2 = set(df2["geneid"].dropna())
    vntrs1 = set(df1["vntrid"].dropna())
    vntrs2 = set(df2["vntrid"].dropna())

    summary = []
    summary.append(f"File1: {file1} ({n1:,} rows)")
    summary.append(f"File2: {file2} ({n2:,} rows)")
    summary.append(f"Common gene:VNTR pairs: {n_common:,}")
    summary.append(f"Unique to File1: {n_only1:,}")
    summary.append(f"Unique to File2: {n_only2:,}")
    summary.append("")
    summary.append(f"Unique genes -> file1: {len(genes1):,}, file2: {len(genes2):,}, shared: {len(genes1 & genes2):,}")
    summary.append(f"Unique VNTRs -> file1: {len(vntrs1):,}, file2: {len(vntrs2):,}, shared: {len(vntrs1 & vntrs2):,}")
    summary.append("")

    # compare columns
    cols = ["raw_pvalue","sigma_g2","sigma_e2","lambda",
            "r2_fixed","r2_genetic","r2_residual","h2","lr_stat",
            "n_indiv","n_alleles","r2","partialr2","tss_distance"]
    summary.append("Column correlations (Pearson / Spearman):")
    for col in cols:
        c1, c2 = f"{col}_1", f"{col}_2"
        if c1 in both and c2 in both:
            a, b = both[c1], both[c2]
            mask = a.notna() & b.notna()
            if mask.sum() > 2:
                try:
                    p = pearsonr(a[mask], b[mask])[0]
                except Exception:
                    p = np.nan
                try:
                    rho = spearmanr(a[mask], b[mask])[0]
                except Exception:
                    rho = np.nan
                summary.append(f"  {col}: n={mask.sum():,}, Pearson={p:.4f}, Spearman={rho:.4f}")
                # produce scatter+density plot
                try:
                    make_scatter_with_density(a, b,
                                              xlabel=f"{col} {dir1}", ylabel=f"{col} {dir2}",
                                              title=f"{col} comparison", outstem=outdir/f"{col}_scatter_density")
                except Exception as e:
                    summary.append(f"    plotting failed for {col}: {e}")
            else:
                summary.append(f"  {col}: insufficient overlapping data")
        else:
            summary.append(f"  {col}: missing in one/both files")

    # raw pvalue similarity & pvalue scatter
    if "raw_pvalue_1" in both and "raw_pvalue_2" in both:
        pv1 = both["raw_pvalue_1"].astype(float)
        pv2 = both["raw_pvalue_2"].astype(float)

        # caluclate normalized difference for each pair of pvalues
        eps = 1e-300
        # use for pvalue scatter and for normalized difference calculation
        logp1 = -np.log10(np.clip(pv1, eps, None))
        logp2 = -np.log10(np.clip(pv2, eps, None))
        denom = logp1 + logp2
        # this is real normalized difference
        normdiff = (logp1 - logp2) / (denom + 1)  # add 1 to denominator. This prevents solutions = 1 or -1 when both p-values are 1 and reduces effect when both pvalues are close to 1, while still giving a strong signal when one p-value is much smaller than the other.
        # add normalized difference to the merged dataframe for ouput in merged_comparison.tsv
        both["normalized_logp_difference"] = normdiff

        allele1 = both["n_alleles_1"].astype(int) if "n_alleles_1" in both.columns else None  # pd.Series(dtype=int)
        allele2 = both["n_alleles_2"].astype(int) if "n_alleles_2" in both.columns else None  # pd.Series(dtype=int)
        if allele1 is not None and allele2 is not None:
            print(f"Mean number of alleles in shared sites: file1={allele1.mean():.2f}, file2={allele2.mean():.2f}")
        else:
            print("Allele count data not available for shared sites; skipping allele-related stats in pvalue comparisons")
        exact = (pv1==pv2).sum()
        same12 = np.isclose(pv1,pv2,atol=1e-12,rtol=0).sum()
        same10 = np.isclose(pv1,pv2,atol=1e-10,rtol=0).sum()
        same8  = np.isclose(pv1,pv2,atol=1e-8,rtol=0).sum()
        same6  = np.isclose(pv1,pv2,atol=1e-6,rtol=0).sum()
        sig1, sig2 = (pv1<alpha).sum(), (pv2<alpha).sum()
        sigboth = ((pv1<alpha)&(pv2<alpha)).sum()
        summary.append("")
        summary.append(f"Raw-pvalue equality: exact={exact:,}, 1e-12={same12:,}, 1e-10={same10:,}, 1e-8={same8:,}, 1e-6={same6:,}")
        summary.append(f"Significant (p<{alpha}): file1={sig1:,}, file2={sig2:,}, both={sigboth:,}")
        renamedirs = {
            "output_MeanLength_NOT_LMM": "MeanAlleleLength",
            "output_Zraw_NoRareAlleles_ML": "NoRareAlleles",
            "output_Zraw_CollapseRareAlleles_ML": "CollapseRareAlleles",
            "output_Zraw_NoRareAlleles_MinusMeanAlleleLength_ML": "ContrastWithMeanAlleleLength",
            "output_Zraw_NoRareAlleles_BestVariant_ML": "ContrastWithBestSNP/INDEL",
            "output_Zraw_ML": "AllAlleles"
            }
        t1 = renamedirs[dir1] if dir1 in renamedirs else dir1
        t2 = renamedirs[dir2] if dir2 in renamedirs else dir2
        make_scatter_with_density(logp1, logp2,
                                  xlabel=f"-log10(pvalue) {t1}", ylabel=f"-log10(pvalue) {t2}",
                                  title="raw_pvalue (-log10) comparison", outstem=outdir/"raw_pvalue_scatter_density")
        prop_stats = plot_logp_proportion(
                                  pv1, pv2, allele1, allele2,
                                  label=f"{t1} vs {t2}:\nproportional logP",
                                  outstem=outdir / "raw_pvalue_logp_proportional")
        diff_stats = plot_logp_difference(
                                  pv1, pv2,
                                  label=f"{t1} vs {t2}:\nlogP difference",
                                  outstem=outdir / "raw_pvalue_logp_difference")
        normdiff_stats = plot_logp_normalized_difference(
                                  pv1, pv2, allele1, allele2,
                                  label=f"{t1} vs {t2}:\nnormalized log(p) difference",
                                  outstem=outdir / "raw_pvalue_logp_normalized_difference",
                                  fixymax=fixymax)
    else:
        summary.append("\nraw_pvalue column missing in one or both files; skipping pvalue-specific comparisons")

    # extremes for diagnostics
    summary.append("")
    summary.append("Extremes for key numeric columns (helps detect truncation/clipping):")
    for c in ["raw_pvalue","sigma_g2","sigma_e2","lambda","r2_fixed","r2_genetic","r2"]:
        if f"{c}_1" in both:
            summary.append("  " + print_extremes(both[f"{c}_1"], c+"_file1"))
        if f"{c}_2" in both:
            summary.append("  " + print_extremes(both[f"{c}_2"], c+"_file2"))

    # ----------------------------
    # distributions for unique (non-shared) pairs in each file
    # ----------------------------
    # unique to file1
    unique1 = merged[merged["_merge"]=="left_only"].copy()
    unique2 = merged[merged["_merge"]=="right_only"].copy()

    # save unique (non-shared) rows to files
    unique1_out = outdir / f"unique_to_{dir1}.tsv"
    unique2_out = outdir / f"unique_to_{dir2}.tsv"
    unique1.to_csv(unique1_out, sep="\t", index=False)
    unique2.to_csv(unique2_out, sep="\t", index=False)
    summary.append("")
    summary.append(f"Saved unique pairs -> file1: {unique1_out.name} ({len(unique1):,} rows), "
                   f"file2: {unique2_out.name} ({len(unique2):,} rows)")

    # prepare p-value series if present
    def extract_pval_series(df, suffix):
        # df is the merged (outer) table with columns potentially from file1/file2.
        # suffix should be "_1" for file1 unique rows, "_2" for file2 unique rows.
        key = f"raw_pvalue{suffix}"
        if key in df.columns:
            return df[key].astype(float)
        else:
            # maybe p-value under different name (e.g. corrected_pvalue?), fallback None
            return pd.Series(dtype=float)

    pv_unique1 = extract_pval_series(unique1, "_1")
    pv_unique2 = extract_pval_series(unique2, "_2")

    # plot distributions and get counts
    dist1 = plot_unique_pvalue_distributions(pv_unique1, label=f"unique_to_{dir1}", outstem=outdir/f"unique_{dir1}_pvalue_dist", alpha=alpha)
    dist2 = plot_unique_pvalue_distributions(pv_unique2, label=f"unique_to_{dir2}", outstem=outdir/f"unique_{dir2}_pvalue_dist", alpha=alpha)

    # add to summary
    summary.append("")
    summary.append("Unique (non-shared) p-value distributions:")
    summary.append(f"  unique_to_file1: n={dist1.get('n',0):,}, n_significant(p<{alpha})={dist1.get('n_significant',0):,}, frac={dist1.get('frac_significant',0):.3f}")
    summary.append(f"  unique_to_file2: n={dist2.get('n',0):,}, n_significant(p<{alpha})={dist2.get('n_significant',0):,}, frac={dist2.get('frac_significant',0):.3f}")

    # save merged table & summary
    both.to_csv(outdir/"merged_comparison.tsv", sep="\t", index=False)
    with open(outdir/"summary.txt", "w") as fh:
        fh.write("\n".join(summary))

    print(f"Done -> {outdir}")
    return merged, summary


def filter_f1_f2(file1, file2, filterfiles, outbase):
    df1 = pd.read_csv(file1, sep="\t", dtype=str)
    df2 = pd.read_csv(file2, sep="\t", dtype=str)
    df1 = normalize_columns(df1)
    df2 = normalize_columns(df2)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    # if filterfiles is not None:
    #     
    #     for f in filterfiles:
    #         df_filter = pd.read_csv(f, sep="\t", dtype=str)
    #         df_filter = normalize_columns(df_filter)
    #         if "geneid" not in df_filter.columns or "vntrid" not in df_filter.columns:
    #             print(f"Warning: filter file {f} missing geneid/vntrid columns, skipping")
    #             continue
    #         pairs = set(zip(df_filter["geneid"], df_filter["vntrid"]))
    #         df1 = df1[df1.apply(lambda row: (row["geneid"], row["vntrid"]) in pairs, axis=1)]
    #         df2 = df2[df2.apply(lambda row: (row["geneid"], row["vntrid"]) in pairs, axis=1)]
    #     out1 = f"{outbase}_filtered_file1.tsv"
    #     out2 = f"{outbase}_filtered_file2.tsv"
    #     df1.to_csv(out1, sep="\t", index=False)
    #     df2.to_csv(out2, sep="\t", index=False)
    #     print(f"Filtered files saved to:\n{out1}\n{out2}")

    if filterfiles is not None:
        all_pairs = set()
        # collect pairs from all filter files (UNION)
        for f in filterfiles:
            df_filter = pd.read_csv(f, sep="\t", dtype=str)
            df_filter = normalize_columns(df_filter)
            if "geneid" not in df_filter.columns or "vntrid" not in df_filter.columns:
                print(f"Warning: filter file {f} missing geneid/vntrid columns, skipping")
                continue
            pairs = set(zip(df_filter["geneid"], df_filter["vntrid"]))
            all_pairs.update(pairs)
        if len(all_pairs) == 0:
            print("Warning: no valid filter pairs found")
        else:
            # filter df1 and df2 independently
            df1 = df1[df1.apply(lambda row: (row["geneid"], row["vntrid"]) in all_pairs, axis=1)]
            df2 = df2[df2.apply(lambda row: (row["geneid"], row["vntrid"]) in all_pairs, axis=1)]
        out1 = f"{outbase}_filtered_file1.tsv"
        out2 = f"{outbase}_filtered_file2.tsv"
        df1.to_csv(out1, sep="\t", index=False)
        df2.to_csv(out2, sep="\t", index=False)
        print(f"Filtered files saved to:\n{out1}\n{out2}")

    return df1, df2


# ---------- directory loop ----------

def compare_directories(base_dir, dir1, dir2, extension=".tsv", f1=None, f2=None, filterfiles=None, fixymax=None, alpha=1e-6):

    if f1 and f2:
        # compare specific files
        file1=Path(base_dir)/dir1/f1
        file2=Path(base_dir)/dir2/f2
        outbase=Path(base_dir)/f"{dir1}VS{dir2}"/f"{f1.replace(extension,'')}_VS_{f2.replace(extension,'')}"
        outbase.parent.mkdir(parents=True, exist_ok=True)
        df1, df2 = filter_f1_f2(file1, file2, filterfiles, outbase/f"{dir1}VS{dir2}")
        compare_vntr_files(file1, file2, df1, df2, dir1, dir2, outbase, fixymax=fixymax, alpha=alpha)
        return
    # compare all matching files in the two directories
    else:
        base_dir=Path(base_dir)
        d1,d2=base_dir/dir1, base_dir/dir2
        outbase=base_dir/f"{dir1}VS{dir2}"
        outbase.mkdir(parents=True, exist_ok=True)
        files1={f.name:f for f in d1.glob(f"*{extension}")}
        files2={f.name:f for f in d2.glob(f"*{extension}")}
        common=sorted(set(files1)&set(files2))
        print(f"Found {len(common)} matching files.")
        for fn in common:
            df1, df2 = filter_f1_f2(files1[fn], files2[fn], filterfiles, outbase)
            compare_vntr_files(files1[fn], files2[fn], df1, df2, dir1, dir2, outbase/fn.replace(extension,""), fixymax=fixymax, alpha=alpha)

# ---------- CLI ----------

if __name__=="__main__":
    '''
    Usage example with --extension:
    python LM_CompareFilesInSubDirectories.py \
        --dir1 output_Zraw_ML \
        --dir2 output_Zraw_MinusMeanAlleleLength_ML \
        --extension Autosomes.tsv \
        --cwd /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP
    Usage example with --f1 and --f2:
    python LM_CompareFilesInSubDirectories.py \
        --dir1 output_Zraw_NoRareAlleles_ML \
        --dir2 output_Zraw_NoRareAlleles_MinusMeanAlleleLength_ML \
        --f1 gene_vntr_association_results.Autosomes.BestVNTRperGene.LessEqualTo_1e-06.tsv \
        --f2 gene_vntr_association_results.Autosomes.tsv \
        --cwd /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP
    Usage with --filterfiles:
    python LM_CompareFilesInSubDirectories.py \
        --dir1 output_MeanLength_NOTBLUP \
        --dir2 output_Zraw_NoRareAlleles_ML \
        --f1 gene_vntr_association_results.tsv \
        --f2 gene_vntr_association_results.Autosomes.tsv \
        --filterfiles /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP/output_MeanLength_NOTBLUP/gene_vntr_association_results.Autosomes.LessEqualTo_1e-06.BestVNTRPerGene.tsv /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP/output_Zraw_NoRareAlleles_ML/gene_vntr_association_results.Autosomes.BestVNTRperGene.LessEqualTo_1e-06.tsv \
        --cwd /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP

    '''
    p=argparse.ArgumentParser()
    p.add_argument("--cwd",default=os.getcwd())
    p.add_argument("--dir1",required=True)
    p.add_argument("--dir2",required=True)
    p.add_argument("--extension",required=False,default=".tsv")
    p.add_argument("--f1",required=False,default=None,help="specific file in dir1 to compare. --extension is ignored if --f1 is specified")
    p.add_argument("--f2",required=False,default=None,help="specific file in dir2 to compare. --extension is ignored if --f2 is specified")
    p.add_argument("--filterfiles", nargs="+", default=None, metavar="FILE", help="Filter analysis to specific geneid/VNTRid pairs in provided filenames, space-separated list. Ex: put file with significant results and --f1 --f2 with all results. Must be full paths")
    p.add_argument("--fixymax", type=float, default=None, help="fix y-axis max for logP proportion plot (for better comparison across files)")
    p.add_argument("--alpha",type=float,default=1e-6)
    a=p.parse_args()
    # --f1 and --f2, take precident over --extension

    compare_directories(a.cwd,a.dir1,a.dir2,extension=a.extension,f1=a.f1,f2=a.f2,filterfiles=a.filterfiles,fixymax=a.fixymax,alpha=a.alpha)
