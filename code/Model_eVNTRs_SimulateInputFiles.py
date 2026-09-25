#!/usr/bin/env python3
"""
simulate_genotype_expression.py

Extended simulation framework.

New functionality when using:
    --simulate INT

If --simulate is provided:
    - Simulates INT genes.
    - For each gene:
        * A different baseline expression distribution (randomized around input mean/std).
        * 1000 independent variant sites near the gene.
        * Each variant has its own independently simulated genotype matrix.
    - Genotypes file contains 1000 * INT rows.
    - TPM file contains INT rows.
    - Plotting is suppressed.
    - Additionally reports per-allele deviation:
        (Observed allele mean − expected allele effect).

If --simulate is None:
    - Original single-gene behavior with plotting.
"""

from __future__ import annotations
import sys
import argparse
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


# -----------------------------------------------------------------------------
# Core genotype simulation utilities
# -----------------------------------------------------------------------------

def compute_integer_allele_counts(freqs: List[float], total_alleles: int) -> np.ndarray:
    freqs = np.array(freqs, dtype=float)
    raw = freqs * total_alleles
    flo = np.floor(raw).astype(int)
    remainder = raw - flo
    remain = total_alleles - flo.sum()

    if remain > 0:
        idx = np.argsort(-remainder)
        flo[idx[:remain]] += 1
    elif remain < 0:
        idx = np.argsort(remainder)
        flo[idx[:(-remain)]] -= 1

    return flo


def generate_allele_pool(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alleles = np.concatenate([[i] * c for i, c in enumerate(counts)])
    rng.shuffle(alleles)
    return alleles


def pair_alleles_into_genotypes(allele_pool: np.ndarray) -> List[Tuple[int, int]]:
    return [tuple(allele_pool[i:i+2]) for i in range(0, len(allele_pool), 2)]


# -----------------------------------------------------------------------------
# Expression simulation
# -----------------------------------------------------------------------------

def simulate_baseline_expression(n: int, mean: float, std: float, rng: np.random.Generator) -> np.ndarray:
    baseline = rng.normal(mean, std, n)
    return np.clip(baseline, 0.0, None)


def apply_allele_effects(baseline: np.ndarray, genotypes: List[Tuple[int, int]], effects: List[float]) -> np.ndarray:
    eff = np.array(effects)
    add = np.array([eff[a] + eff[b] for (a, b) in genotypes])
    return np.where(baseline + add >= 0, baseline + add, 0.0)


# -----------------------------------------------------------------------------
# Single gene functionality with plotting
# -----------------------------------------------------------------------------
def write_genotype_file(path: Path, genotypes: List[Tuple[int, int]], n: int, m: int) -> None:
    header = ["#chr", "start", "end", "pid", "gid", "strand"] + [f"indiv{i+1}" for i in range(n)]
    with open(path, 'w') as f:
        f.write("\t".join(header) + "\n")
        row = ["1", "5000000", "5000050", "1.5000000.5000050", "1.5000000.5000050", "+"]
        f.write("\t".join(row + [f"{a}/{b}" for a, b in genotypes]) + "\n")


def write_tpm_file(path: Path, tpms: np.ndarray, n: int, mean: float, std: float) -> None:
    header = ["#chr", "start", "end", "pid", "gid", "strand"] + [f"indiv{i+1}" for i in range(n)]
    with open(path, 'w') as f:
        f.write("\t".join(header) + "\n")
        row = ["1", "5000000", "5000100", "SIMULATED", "SIMULATED", "+"]
        f.write("\t".join(row + [f"{v:.6f}" for v in tpms]) + "\n")


def write_covariate_file(path: Path, n: int, rng: np.random.Generator) -> None:
    header = ["ID"] + [f"indiv{i+1}" for i in range(n)]
    cov = rng.normal(0, 1, n) + 10
    with open(path, 'w') as f:
        f.write("\t".join(header) + "\n")
        f.write("\t".join(["cov1"] + [f"{v:.6f}" for v in cov]) + "\n")


def write_log(path: Path, args: argparse.Namespace, allele_counts: np.ndarray, baseline: np.ndarray, final_tpms: np.ndarray, genotypes: np.ndarray) -> None:
    with open(path, 'w') as f:
        f.write(f"Run time: {datetime.now().isoformat()}Z\n")
        f.write(f"n={args.n}, m={args.m}, mean={args.mean}, std={args.std}, noise={args.noise}, seed={args.seed}\n")
        f.write(f"Allele frequencies: {args.m_freq}\nAllele effects: {args.m_effect}\n")
        if args.num_genes is not None:
            f.write(f"Simulated {args.num_genes} genes with {args.m} alleles each.\n")
        else:
            f.write("Single gene / single variant simulation.\n")
            f.write("Allele counts (2n total):\n")
            for i, c in enumerate(allele_counts):
                f.write(f"  allele_{i}: {c}\n")
            f.write(f"Baseline mean={baseline.mean():.3f}, std={baseline.std():.3f}\n")
            f.write(f"Final TPM mean={final_tpms.mean():.3f}, std={final_tpms.std():.3f}\n")
            f.write(f"Genotypes (of all individuals):\n")
            for i, (a, b) in enumerate(zip(genotypes, final_tpms)):
                f.write(f"  indiv_{i+1}: alleles=({a[0]},{a[1]}), final_TPM={b:.6f}\n")


def plot_results_pdf(pdf_path: Path, baseline_no_noise: np.ndarray, baseline: np.ndarray, final_tpms: np.ndarray, genotypes: List[Tuple[int, int]], m_freq: List[float], m_effect: List[float]) -> None:
    """Generate a PDF file showing baseline vs final gene expression and allele freq/effect plot."""

    # Calculate per-allele baseline and final TPM means
    allele_baseline_no_noise = {}
    allele_baseline = {}
    allele_final = {}
    for idx, (a, b) in enumerate(genotypes):
        for al in (a, b):
            allele_baseline_no_noise.setdefault(al, []).append(baseline_no_noise[idx])
            allele_baseline.setdefault(al, []).append(baseline[idx])
            allele_final.setdefault(al, []).append(final_tpms[idx])
    allele_baseline_no_noise_means = [np.mean(allele_baseline_no_noise[a]) for a in sorted(allele_baseline_no_noise.keys())]
    allele_baseline_means = [np.mean(allele_baseline[a]) for a in sorted(allele_baseline.keys())]
    allele_final_means = [np.mean(allele_final[a]) for a in sorted(allele_final.keys())]

    alleles = list(range(len(m_freq)))

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 4, figsize=(7, 3.5))  # 18,6

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

        # Left composite plot: overall baseline and per-allele baseline violins
        axes[0].violinplot(baseline_no_noise, positions=[-1], widths=0.3, showmeans=True)
        v1 = axes[0].violinplot([allele_baseline_no_noise[a] for a in alleles], positions=alleles, showmeans=True)
        axes[0].scatter(alleles, allele_baseline_no_noise_means, color='black', zorder=3)  # , label='Mean per allele'
        # axes[0].set_title('Baseline Gene Expression (no noise)')  # 'Baseline Gene Expression (no noise)'
        axes[0].set_title("a", loc="left", fontweight="bold")
        axes[0].set_xlabel('Allele')
        axes[0].set_ylabel('TPM')
        axes[0].set_xticks([-1] + alleles)
        axes[0].set_xticklabels(['Total'] + [str(a) for a in alleles])
        # axes[0].legend(frameon=False)

        # Left center composite plot: overall baseline and per-allele baseline violins
        axes[1].violinplot(baseline, positions=[-1], widths=0.3, showmeans=True)
        v1 = axes[1].violinplot([allele_baseline[a] for a in alleles], positions=alleles, showmeans=True)
        axes[1].scatter(alleles, allele_baseline_means, color='black', zorder=3)  # , label='Mean per allele'
        # axes[1].set_title('Baseline Gene Expression (with noise)')  # 'Baseline Gene Expression (with noise)'
        axes[1].set_title("b", loc="left", fontweight="bold")
        axes[1].set_xlabel('Allele')
        axes[1].set_ylabel('TPM')
        axes[1].set_xticks([-1] + alleles)
        axes[1].set_xticklabels(['Total'] + [str(a) for a in alleles])
        # axes[1].legend(frameon=False)

        # Middle composite plot: overall final and per-allele final violins
        axes[2].violinplot(final_tpms, positions=[-1], widths=0.3, showmeans=True)
        v2 = axes[2].violinplot([allele_final[a] for a in alleles], positions=alleles, showmeans=True)
        axes[2].scatter(alleles, allele_final_means, color='black', zorder=3)  # , label='Mean per allele'
        # axes[2].set_title('Final Gene Expression (with noise & allele effects)')  # Final Gene Expression (with noise & allele effects)'
        axes[2].set_title("c", loc="left", fontweight="bold")
        axes[2].set_xlabel('Allele')
        axes[2].set_ylabel('TPM')
        axes[2].set_xticks([-1] + alleles)
        axes[2].set_xticklabels(['Total'] + [str(a) for a in alleles])
        # axes[2].legend(frameon=False)

        # Right plot: allele frequency vs effect
        print(f"Allele frequencies: {m_freq}")
        print(f"Allele effects: {m_effect}")
        print(f"Allele labels: {alleles}")
        axes[3].scatter(m_effect, m_freq, c='tab:blue', edgecolor='k')
        axes[3].set_xlabel('Allele Effect')
        axes[3].set_ylabel('Allele Frequency')
        axes[3].set_title("d", loc="left", fontweight="bold")

        # axes[3].set_title('Allele Frequency vs Effect')
        for i, (fx, fy) in enumerate(zip(m_effect, m_freq)):
            axes[3].text(fx, fy + axes[3].get_ylim()[1]*(0.02), str(i), fontsize=10, ha='center', va='bottom')
        axes[3].set_ylim(axes[3].get_ylim()[0] - axes[3].get_ylim()[0]*(0.04), axes[3].get_ylim()[1] + axes[3].get_ylim()[1]*(0.04))
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)



# -----------------------------------------------------------------------------
# Multi-gene simulation logic
# -----------------------------------------------------------------------------

def simulate_multiple_genes(args, rng, outdir: Path, timestamp: str):
    n = args.n
    m = args.m
    G = args.num_genes  # number of genes to simulate int()
    variants_per_gene = args.num_variants  # number of variants to simulate per gene int()

    geno_path = outdir / f"Simulated_TRDB_MatrixGenotypes.MULTI.n{n}.m{m}.mean{args.mean}.std{args.std}.noise{args.noise}.{timestamp}.txt"
    # tpm_path = outdir / f"Simulated_TPM_MatrixGeneExpression.MULTI.n{n}.m{m}.{timestamp}.tsv"
    tpm_baseline_path = outdir / f"Simulated_TPM_MatrixGeneExpression.BASELINE.n{n}.m{m}.mean{args.mean}.std{args.std}.noise{args.noise}.{timestamp}.tsv"
    tpm_effect_path   = outdir / f"Simulated_TPM_MatrixGeneExpression.WITH_ALLELE_EFFECTS.n{n}.m{m}.mean{args.mean}.std{args.std}.noise{args.noise}.{timestamp}.tsv"

    deviation_path = outdir / f"Allele_Effect_Deviation.MULTI.n{n}.m{m}.{timestamp}.tsv"
    header = ["#chr", "start", "end", "pid", "gid", "strand"] + [f"indiv{i+1}" for i in range(n)]

    with open(geno_path, "w") as geno_f, \
         open(tpm_baseline_path, "w") as tpm_base_f, \
         open(tpm_effect_path, "w") as tpm_eff_f, \
         open(deviation_path, "w") as dev_f:

        geno_f.write("\t".join(header) + "\n")
        tpm_base_f.write("\t".join(header) + "\n")
        tpm_eff_f.write("\t".join(header) + "\n")
        dev_f.write("gene\tvariant\tallele\tfreq\texpected_effect\texpected_mean\tobserved_mean\tdifference\tnormalized_difference\n")

        chrom = 1
        base_position = 1_000_000

        for g in range(G):  # for each gene simulated
            gene_start = base_position + g * 1_000_000
            gene_end = gene_start + 1000

            # Randomize gene-specific distribution
            gene_mean = rng.normal(args.mean, args.std * 0.25)
            gene_std = abs(rng.normal(args.std, args.std * 0.1))

            baseline = simulate_baseline_expression(n, gene_mean, gene_std, rng)
            if args.noise > 0:
                baseline = baseline + np.random.uniform(-args.noise, args.noise, size=len(baseline))
            # ---------------------------
            # Write BASELINE TPM row
            # ---------------------------
            tpm_row = [
                str(chrom),
                str(gene_start),
                str(gene_end),
                f"SIMULATED_{g+1}",
                f"SIMULATED_{g+1}",
                "+"
            ] + [f"{v:.6f}" for v in baseline]
            tpm_base_f.write("\t".join(tpm_row) + "\n")

            for v in range(variants_per_gene):
                total_alleles = 2 * n
                counts = compute_integer_allele_counts(args.m_freq, total_alleles)
                pool = generate_allele_pool(counts, rng)
                genotypes = pair_alleles_into_genotypes(pool)

                final_tpms = apply_allele_effects(baseline, genotypes, args.m_effect)

                var_pos = gene_start - (variants_per_gene/2) + v + 1 # rng.integers(-100_000, 100_000)
                var_end = var_pos + 1

                if var_pos % 1_000_000 == 0:
                    print(f"Adding allele specific effects for gene {g+1}/{G}. Expected allee effect association at variant {v+1}/{variants_per_gene} at position {var_pos} (If allele effects are provided in --m_effect).")
                    # ---------------------------
                    # Write EFFECT TPM row
                    # ---------------------------
                    effect_row = [
                        str(chrom),
                        str(int(var_pos)),
                        str(int(var_end)),
                        f"SIMULATED_{g+1}",
                        f"SIMULATED_{g+1}",
                        "+"
                    ] + [f"{v:.6f}" for v in final_tpms]

                    tpm_eff_f.write("\t".join(effect_row) + "\n")
                
                # ---------------------------
                # Write genotype row
                # ---------------------------
                geno_row = [
                    str(chrom),
                    str(int(var_pos)),
                    str(int(var_end)),
                    f"{chrom}.{int(var_pos)}.{int(var_end)}",
                    f"{chrom}.{int(var_pos)}.{int(var_end)}",
                    "+"
                ] + [f"{a}/{b}" for a, b in genotypes]
                geno_f.write("\t".join(geno_row) + "\n")

                # Per-allele deviation reporting
                allele_tpms = {}
                allele_counts_obs = {}

                for i, (a, b) in enumerate(genotypes):
                    for al in (a, b):
                        allele_tpms.setdefault(al, []).append(final_tpms[i])
                        allele_counts_obs[al] = allele_counts_obs.get(al, 0) + 1

                for al in range(m):
                    if al in allele_tpms:
                        observed_mean = np.mean(allele_tpms[al])
                        expected = args.m_effect[al] + gene_mean
                        diff = observed_mean - expected
                        normdiff = diff / gene_std if gene_std > 0 else np.nan
                        freq = allele_counts_obs[al] / total_alleles
                    else:
                        observed_mean = np.nan
                        expected = args.m_effect[al]
                        diff = np.nan
                        freq = 0.0

                    dev_f.write(
                        f"{g+1}\t{v+1}\t{al}\t{freq:.6f}\t{expected:.6f}\t{gene_mean:.6f}\t{observed_mean:.6f}\t{diff:.6f}\t{normdiff:.6f}\n"
                    )

    print("Multi-gene simulation complete.")
    print(f"  Genotypes: {geno_path}")
    print(f"  Baseline TPMs: {tpm_baseline_path}")
    print(f"  TPMs with allele effects (Association expected to central variant): {tpm_effect_path}")
    print(f"  Deviations: {deviation_path}")

    return deviation_path


# -----------------------------------------------------------------------------
# Multi-gene deviation plotting
# -----------------------------------------------------------------------------

def plot_normalized_deviation_from_file(dev_file: Path, output_prefix: Path) -> None:
    """
    Scatter plot:
        allele frequency vs normalized_difference

    Saves:
        output_prefix.pdf
        output_prefix.png
    """

    freqs = []
    normdiffs = []

    with open(dev_file, "r") as f:
        header = next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            try:
                freq = float(parts[3])
                normdiff = float(parts[8])
            except ValueError:
                continue

            freqs.append(freq)
            normdiffs.append(normdiff)

    if not freqs:
        print("No valid data found in deviation file.")
        return

    plt.figure(figsize=(7, 3.5))
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
    plt.scatter(freqs, normdiffs, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Allele Frequency")
    plt.ylabel("Normalized Difference")
    plt.title("Allele Frequency vs Normalized Difference")
    plt.tight_layout()

    plt.savefig(str(output_prefix) + ".pdf")
    plt.savefig(str(output_prefix) + ".png", dpi=300)
    plt.close()

    print(f"Saved scatter plots: {output_prefix}.pdf/.png")


def plot_boxplot_by_frequency(dev_file: Path, output_prefix: Path) -> None:
    """
    Boxplots of normalized_difference grouped by allele,
    ordered from lowest to highest allele frequency.

    Saves:
        output_prefix.pdf
        output_prefix.png
    """

    allele_data = {}
    allele_freq = {}

    with open(dev_file, "r") as f:
        header = next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            try:
                allele = parts[2]
                freq = float(parts[3])
                normdiff = float(parts[8])
            except ValueError:
                continue

            allele_data.setdefault(allele, []).append(normdiff)
            allele_freq[allele] = freq

    if not allele_data:
        print("No valid data found in deviation file.")
        return

    # Sort alleles by frequency (low → high)
    sorted_alleles = sorted(allele_data.keys(),
                            key=lambda a: allele_freq.get(a, 0))

    data = [allele_data[a] for a in sorted_alleles]

    plt.figure(figsize=(7, 3.5)) # 10, 6

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

    plt.boxplot(data, showfliers=True)
    plt.xticks(
        range(1, len(sorted_alleles) + 1),
        [f"{a}\n({allele_freq[a]:.3f})" for a in sorted_alleles],
        rotation=45
    )
    plt.xlabel("Allele (frequency)")
    plt.ylabel("Normalized Difference")
    plt.title("Normalized Difference Distribution by Allele Frequency")
    plt.tight_layout()

    plt.savefig(str(output_prefix) + ".pdf")
    plt.savefig(str(output_prefix) + ".png", dpi=300)
    plt.close()

    print(f"Saved boxplots: {output_prefix}.pdf/.png")


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def validate_args(args: argparse.Namespace) -> None:
    if len(args.m_freq) != args.m:
        raise ValueError(f"--m_freq (length={len(args.m_freq)}) length must equal --m (m={args.m})")
    if len(args.m_effect) != args.m:
        raise ValueError(f"--m_effect length must equal --m")
    if not math.isclose(sum(args.m_freq), 1.0, rel_tol=1e-6):
        raise ValueError(f"--m_freq must sum to 1, current sum {sum(args.m_freq)}")
    # place back if desired that first allele must have 0 effect
    # if not math.isclose(args.m_effect[0], 0.0, abs_tol=1e-9):
    #     raise ValueError("First allele effect must be 0.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type=int, default=100, help="Number of individuals to simulate (default 100)")
    parser.add_argument("--m", type=int, default=10, help="Number of alleles to simulate (default 10)")
    parser.add_argument("--m_freq", nargs='+', type=float, required=True, help="Allele frequencies (space-separated, must sum to 1, length must equal --m)")
    parser.add_argument("--m_effect", nargs='+', type=float, required=True, help="Allele effects (space-separated, length must equal --m, first effect must be 0.0)")
    parser.add_argument("--mean", type=float, required=True, help="Mean of the baseline gene expression distribution")
    parser.add_argument("--std", type=float, required=True, help="Standard deviation of the baseline gene expression distribution")
    parser.add_argument("--noise", type=float, default=0.0, help="An integer value representing the amount of noise to be added to the baseline distribution. Integer value specified represents min (-args.noise) to max (+args.noise) from a uniform distribution (e.g. --noise 5.0 --> ±5.0, default 0.0 for no noise)")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default None)")
    parser.add_argument("--num_genes", type=int, default=None, help="Number of genes to simulate (activates multi-gene mode)")
    parser.add_argument("--num_variants", type=int, default=1000, help="Number of variants to simulate per gene in multi-gene mode (default 1000, requires --num_genes)")
    parser.add_argument("--no_plots", action="store_true", default=False, help="If provided, supresses all plotting (default False)")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_args(args)

    ### # Center effects by population average effect
    ### mean_effect = np.sum(np.array(args.m_freq) * np.array(args.m_effect))
    ### args.m_effect = [x - mean_effect for x in args.m_effect]

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outdirplots = outdir / "plots"
    outdirplots.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"timestamp: {timestamp}")
    if args.num_genes is not None:
        print(f"Simulating {args.num_genes} genes with {args.m} alleles each...")
        deviation_path = simulate_multiple_genes(args, rng, outdir, timestamp)
        # make covariates
        cov_file = outdir / f"Simulated_Covariates.MULTI.n{args.n}.{timestamp}.txt"
        log_file = outdir / f"Simulate_run.{timestamp}.log"
        write_covariate_file(cov_file, args.n, rng)
        write_log(log_file, args, None, None, None, None)
        print(f"  Covariates: {cov_file}\n  Log: {log_file}\n")
        # plotting multi-gene deviation curve
        # dev_file = outdir / f"Allele_Deviations.{timestamp}.tsv"
        # plot_file = outdirplots / f"Allele_Deviation_vs_Frequency.{timestamp}.pdf"
        if not args.no_plots:
            plot_normalized_deviation_from_file(deviation_path, outdirplots / f"Allele_Deviation_vs_Frequency.{timestamp}")
            plot_boxplot_by_frequency(deviation_path, outdirplots / f"Deviation_Boxplot_By_Frequency.{timestamp}")
    else:
        # single-gene simulation with plotting
        print("Simulating single gene with plotting...")

        n, m = args.n, args.m
        total_alleles = 2 * n
        counts = compute_integer_allele_counts(args.m_freq, total_alleles)
        pool = generate_allele_pool(counts, rng)
        genotypes = pair_alleles_into_genotypes(pool)  # , n

        baseline_no_noise = simulate_baseline_expression(n, args.mean, args.std, rng)
        if args.noise > 0:
            baseline = baseline_no_noise + np.random.uniform(-args.noise, args.noise, size=len(baseline_no_noise))
        else:
            baseline = baseline_no_noise
        final_tpms = apply_allele_effects(baseline, genotypes, args.m_effect)

        # File paths
        geno_file = outdir / f"Simulated_TRDB_MatrixGenotypes.n{n}.m{m}.{timestamp}.txt"
        tpm_file = outdir / f"Simulated_TPM_MatrixGeneExpression.n{n}.m{m}.mean{args.mean}.std{args.std}.{timestamp}.tsv"
        cov_file = outdir / f"Simulated_Covariates.n{n}.m{m}.mean{args.mean}.std{args.std}.{timestamp}.txt"
        log_file = outdir / f"Simulate_run.{timestamp}.log"
        pdf_file = outdirplots / f"Simulated_Plots.n{n}.m{m}.mean{args.mean}.std{args.std}.{timestamp}.pdf"

        write_genotype_file(geno_file, genotypes, n, m)
        write_tpm_file(tpm_file, final_tpms, n, args.mean, args.std)
        write_covariate_file(cov_file, n, rng)
        write_log(log_file, args, counts, baseline, final_tpms, genotypes)
        if not args.no_plots:
            plot_results_pdf(pdf_file, baseline_no_noise, baseline, final_tpms, genotypes, args.m_freq, args.m_effect)

        print("Simulation complete.")
        print(f"  Genotypes: {geno_file}\n  TPMs: {tpm_file}\n  Covariates: {cov_file}\n  Log: {log_file}\n  Plots: {pdf_file}")


if __name__ == '__main__':
    main()
