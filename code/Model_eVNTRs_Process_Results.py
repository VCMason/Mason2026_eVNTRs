#!/usr/bin/env python3
"""
Process gene-VNTR association results.

Steps:
1. Merge per-chromosome TSV files into a single autosome file.
2. Filter for significant (p <= 1e-6) associations.
3. Find best VNTR per gene (lowest p-value).
4. Find best gene per VNTR (lowest p-value).
5. Generate filtered significant versions of both.
"""

import os
import argparse
import pandas as pd


def merge_chromosome_files(base_path: str, output_file: str) -> None:
    """Merge per-chromosome TSV files into a single autosomes file."""
    dfs = []
    for chrom in range(1, 30):
        file_path = os.path.join(base_path, f"gene_vntr_association_results.{chrom}.tsv")
        if os.path.exists(file_path):
            dfs.append(pd.read_csv(file_path, sep="\t", low_memory=False))
        else:
            print(f"Warning: Missing file for chromosome {chrom}: {file_path}")
    if not dfs:
        raise FileNotFoundError("No chromosome files found to merge.")
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_file, sep="\t", index=False)
    print(f"Combined autosomes file written to {output_file}")


def remove_too_many_alleles(df: pd.DataFrame, max_alleles: int) -> pd.DataFrame:
    """Keeps rows with n_alleles less than or equal to threshold."""
    # return df[df.iloc[:, -1] <= pval_threshold]
    # Exclude rows where p-value is exactly 0.0, this indicates a problem (e.g. only 1 individual with different genotype for example)
    # return df[(df.iloc[:, -1] <= pval_threshold) & (df.iloc[:, -1] != 0.0)]
    return df[(df.loc[:, "n_alleles"] <= max_alleles)]


def filter_to_autosomes(df):
    chrom = df["VNTRid"].str.split('.', expand=True)[0]
    chrom_numeric = pd.to_numeric(chrom, errors="coerce").notnull()
    df = df[chrom_numeric]
    return df

def filter_significant(df: pd.DataFrame, pval_threshold: float) -> pd.DataFrame:
    """Filter rows with p-value less than or equal to threshold."""
    # return df[df.iloc[:, -1] <= pval_threshold]
    # Exclude rows where p-value is exactly 0.0, this indicates a problem (e.g. only 1 individual with different genotype for example)
    # return df[(df.iloc[:, -1] <= pval_threshold) & (df.iloc[:, -1] != 0.0)]
    return df[(df.loc[:, "raw_pvalue"] <= pval_threshold) & (df.loc[:, "raw_pvalue"] != 0.0)]


def best_per_group(df: pd.DataFrame, group_col: str, pval_col: str) -> pd.DataFrame:
    """Find the row with the lowest p-value per group."""
    return df.loc[df.groupby(group_col)[pval_col].idxmin()]


def count_unique_genes(df: pd.DataFrame, gene_col: str, output_file: str) -> None:
    """Count occurrences of each unique gene and save to file."""
    counts = df[gene_col].value_counts().reset_index()
    counts.columns = [gene_col, "count"]
    counts.to_csv(output_file, sep="\t", index=False)
    print(f"Unique gene counts written to {output_file}")


def process_gene_vntr_results(base_dir: str, sub_dir: str, p_threshold: float = 1e-6, merged_file: str = None, max_alleles=None, autosome_only=None) -> None:
    """Main processing pipeline replicating the bash workflow."""
    work_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    if merged_file is None:
        merged_file = "gene_vntr_association_results.Autosomes.tsv"
        merged_path = os.path.join(work_dir, merged_file)

        # Step 1: Merge
        merge_chromosome_files(work_dir, merged_path)
    else:
        merged_path = os.path.join(work_dir, merged_file)
        print(f"Using provided merged file: {merged_path}")
        
    df = pd.read_csv(merged_path, sep="\t")
    if autosome_only is not None:
        df = filter_to_autosomes(df)

    # Step 2: Filter significant
    significant_path = merged_path.replace(".tsv", f".LessEqualTo_{p_threshold}.tsv")
    df_sig = filter_significant(df, p_threshold)
    if max_alleles is not None:
        df_sig = remove_too_many_alleles(df_sig, max_alleles)
        significant_path = significant_path.replace(".tsv", f".MaxAlleles_{max_alleles}.tsv")
        merged_path = merged_path.replace(".tsv", f".MaxAlleles_{max_alleles}.tsv")
    df_sig.to_csv(significant_path, sep="\t", index=False)
    print(f"Filtered significant results written to {significant_path}")
    # Step 2b: Count unique genes
    unique_genes_path = significant_path.replace(".tsv", ".UniqueGenes.tsv")
    # count_unique_genes(df_sig, df_sig.columns[0], unique_genes_path)
    count_unique_genes(df_sig, "geneid", unique_genes_path)

    # Step 3: Best VNTR per Gene
    best_vntr_path = merged_path.replace(".tsv", ".BestVNTRperGene.tsv")
    # df_best_vntr = best_per_group(df, df.columns[0], df.columns[-1])
    df_best_vntr = best_per_group(df, "geneid", "raw_pvalue")
    df_best_vntr.to_csv(best_vntr_path, sep="\t", index=False)
    print(f"Best VNTR per Gene written to {best_vntr_path}")

    # Step 4: Best Gene per VNTR
    best_gene_path = merged_path.replace(".tsv", ".BestGeneperVNTR.tsv")
    # df_best_gene = best_per_group(df, df.columns[1], df.columns[-1])
    df_best_gene = best_per_group(df, "VNTRid", "raw_pvalue")
    df_best_gene.to_csv(best_gene_path, sep="\t", index=False)
    print(f"Best Gene per VNTR written to {best_gene_path}")

    # Step 5: Filter both best lists
    best_vntr_sig_path = best_vntr_path.replace(".tsv", f".LessEqualTo_{p_threshold}.tsv")
    best_gene_sig_path = best_gene_path.replace(".tsv", f".LessEqualTo_{p_threshold}.tsv")

    df_best_vntr_sig = filter_significant(df_best_vntr, p_threshold)
    df_best_vntr_sig.to_csv(best_vntr_sig_path, sep="\t", index=False)

    df_best_gene_sig = filter_significant(df_best_gene, p_threshold)
    df_best_gene_sig.to_csv(best_gene_sig_path, sep="\t", index=False)

    print("\nAll files generated successfully:")
    for f in [
        merged_path,
        significant_path,
        unique_genes_path,
        best_vntr_path,
        best_gene_path,
        best_vntr_sig_path,
        best_gene_sig_path,
    ]:
        try:
            with open(f, "r") as file:
                line_count = sum(1 for _ in file)
            print(f"  - {os.path.basename(f)} ({line_count:,} lines)")
        except Exception as e:
            print(f"  - {os.path.basename(f)} (error reading file: {e})")


def main() -> None:

    """
    # Usage:
    # if it is needed to merge all per-chromosome files into a single autosomes file, run:
    # python Model_eVNTRs_Process_Results.py \
    # --base_dir /cluster/work/pausch/vmason/analyses/modeling/linear/eQTLCohort_BLUP \
    # --sub_dir output_Zraw \
    # --p_threshold 1e-7
    # --autosome_only

    # if the merged file is already created, you can specify it with --merged_file instead of merging per chromosome files:
    # python Model_eVNTRs_Process_Results.py \
    # --base_dir /cluster/work/pausch/vmason/analyses/modeling/linear/eQTLCohort_BLUP \
    # --sub_dir output_Zraw \
    # --p_threshold 1e-7 \
    # --merged_file mypreviouslymergedfile.tsv

    # max alleles can be specified to filter out VNTRs with too many alleles, but not recommened
    # --max_alleles 30
    # 
    # Parse arguments and execute the processing pipeline.
    # """
    parser = argparse.ArgumentParser(description="Process gene-VNTR association results.")
    parser.add_argument(
        "--base_dir", required=True, default=os.getcwd(), help="Base directory containing the chromosome TSV files. (default: current working directory)"
    )
    parser.add_argument(
        "--sub_dir", required=True, help="Subdirectory for this specific run."
    )
    parser.add_argument(
        "--p_threshold", type=float, default=1e-7, help="P-value significance threshold (default 1e-7)."
    )
    parser.add_argument(
        "--merged_file", type=str, required=False, default=None, help="Name of the merged autosomes file (if different from default). If not provided, defaults to 'gene_vntr_association_results.Autosomes.tsv'., if provided, will skip per chromosome merging step and use this file name instead."
    )
    parser.add_argument(
        "--max_alleles", type=int, required=False, default=None, help="Maximum number of alleles allowed per VNTR."
    )
    parser.add_argument(
        "--autosomes_only", action='store_true', required=False, default=None, help="Filter Merged file to autosomes only (assumes autosomes are numeric 1, 5, 16, etc.). Uses first part of VNTRid column (Ex: 7 from 7.1236799.12367999)."
    )
    args = parser.parse_args()
    process_gene_vntr_results(args.base_dir, args.sub_dir, args.p_threshold, args.merged_file, args.max_alleles)

if __name__ == "__main__":
    main()
