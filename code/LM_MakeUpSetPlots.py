#!/usr/bin/env python3
"""
make_upset_plots.py

Generates UpSet plots comparing the presence of `geneid`, `VNTRid`, and combined `geneid_VNTRid` pairs
across multiple subdirectories containing the same-named TSV files with different header formats.

Example:
python make_upset_plots.py --cwd /path/to/base --subdirs dirA dirB dirC --samefilename gene_vntr_association_results.Autosomes.LessEqualTo_0.000001.tsv --outdir upset_output

Dependencies:
  pip install pandas matplotlib upsetplot

Plots_UpSet/
├── upset_genes.png
├── upset_vntrs.png
├── upset_gene_VNTR_pairs.png
├── upset_intersections_genes/
│   ├── shared__A_B_C__not_D.txt
│   ├── shared__A_D.txt
│   └── shared__A_B_C_D.txt
├── upset_intersections_vntrs/
│   └── ...
└── upset_intersections_gene_vntr_pairs/
    └── ...

"""

import argparse
import os
import sys
import glob
import textwrap
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators, plot as plt_upset


def write_upset_intersection_files(df, label, outdir):
    """
    For an indicator DataFrame (rows = items, columns = groups, values = bool),
    write one file per unique membership pattern listing the items.

    Files are written to:
      outdir/upset_intersections_<label>/

    Example filename:
      shared__A_B_C__not_D.txt
    """
    intersection_dir = os.path.join(outdir, f"upset_intersections_{label}")
    os.makedirs(intersection_dir, exist_ok=True)

    groups = list(df.columns)

    # Group rows by their boolean membership pattern
    pattern_to_items = defaultdict(list)

    for item, row in df.iterrows():
        pattern = tuple(row[g] for g in groups)
        pattern_to_items[pattern].append(item)

    for pattern, items in pattern_to_items.items():
        if not items:
            continue

        present = [g for g, p in zip(groups, pattern) if p]
        absent = [g for g, p in zip(groups, pattern) if not p]

        if not present:
            continue  # skip items not present in any group

        present_str = "_".join(present)
        absent_str = "_".join(absent)

        if absent:
            fname = f"SHARED_{present_str}_EXCLUDING_{absent_str}.txt"
        else:
            fname = f"SHARED_{present_str}.txt"

        outpath = os.path.join(intersection_dir, fname)

        with open(outpath, "w") as f:
            for x in sorted(items):
                f.write(f"{x}\n")

    return


def find_files(base_dir, subdir, samefilename):
    path = os.path.join(base_dir, subdir)
    candidate = os.path.join(path, samefilename)
    matches = []
    if os.path.isfile(candidate):
        matches.append(candidate)
    pattern = os.path.join(path, '**', samefilename)
    matches.extend(glob.glob(pattern, recursive=True))
    return sorted(set(matches))


def detect_gene_vntr_columns(df):
    lower_cols = [c.lower() for c in df.columns]
    gene_col, vntr_col = None, None

    for c in df.columns:
        cl = c.lower()
        if 'gene' in cl and 'id' in cl:
            gene_col = c
        if 'vntr' in cl and 'id' in cl:
            vntr_col = c
    if gene_col is None:
        for c in df.columns:
            if 'gene' in c.lower():
                gene_col = c
                break
    if vntr_col is None:
        for c in df.columns:
            if 'vntr' in c.lower():
                vntr_col = c
                break

    if gene_col is None or vntr_col is None:
        raise ValueError(f"Could not detect geneid/VNTRid columns in columns: {df.columns.tolist()}")
    return gene_col, vntr_col


def read_tsv_extract_sets(file_path):
    df = pd.read_csv(file_path, sep='\t', comment='#', dtype=str, na_filter=False)
    df.columns = [c.strip() for c in df.columns]

    gene_col, vntr_col = detect_gene_vntr_columns(df)
    genes = set(df[gene_col].astype(str).str.strip().replace({'': None}).dropna())
    vntrs = set(df[vntr_col].astype(str).str.strip().replace({'': None}).dropna())
    pairs = {f"{g}|{v}" for g, v in zip(df[gene_col], df[vntr_col]) if g.strip() and v.strip()}
    return genes, vntrs, pairs


def build_indicator_df(set_dict, name):
    all_items = set().union(*set_dict.values())
    df = pd.DataFrame({group: [x in s for x in all_items] for group, s in set_dict.items()}, index=list(all_items))
    df.index.name = name
    return df


# def plot_upset(df, title, outprefix):
#     plt.figure(figsize=(5, 2.5))
# 
#     plt.rcParams["font.family"] = "serif"
#     plt.rcParams["font.serif"] = ["Times New Roman"]
#     plt.rcParams["font.size"] = 12
# 
#     # df = df.copy(deep=True)
#     upset_data = from_indicators(df.columns.tolist(), df)
#     UpSet(upset_data, show_counts='%d').plot()  # .plot(fig=plt.gcf())
#     plt.suptitle(title)
# 
#     for ext in ('.png', '.pdf'):
#         plt.savefig(outprefix + ext, bbox_inches='tight', dpi=300)
#     plt.close()
#     print(f"Saved: {outprefix}.png and {outprefix}.pdf")
# 
# def plot_upset(df, title, outprefix):
# 
# 
#     plt.rcParams["font.family"] = "serif"
#     plt.rcParams["font.serif"] = ["Times New Roman"]
#     plt.rcParams["font.size"] = 12
# 
#     upset_data = from_indicators(df.columns.tolist(), df)
#     upset = UpSet(upset_data, show_counts='%d')
# 
#     fig = plt.figure(figsize=(5, 2.5))
#     plt_upset(upset, fig=fig, element_size=None)
#     # plt.show()
#     # upset.plot(fig=fig)
# 
#     fig.suptitle(title)
# 
#     for ext in ('.png', '.pdf'):
#         fig.savefig(outprefix + ext, dpi=300)  # bbox_inches='tight', 
# 
#     plt.close(fig)
#     print(f"Saved: {outprefix}.png and {outprefix}.pdf")

def plot_upset(df, title, outprefix):
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 14

    upset_data = from_indicators(df.columns.tolist(), df)

    #fig = plt.figure(figsize=(5, 2.5))
    fig = plt.figure(figsize=(4, 2))

    upset = UpSet(upset_data, show_counts='%d')
    upset.plot(fig=fig)

    fig.suptitle(title)

    for ext in ('.png', '.pdf'):
        fig.savefig(outprefix + ext, bbox_inches='tight', dpi=300)

    plt.close(fig)
    print(f"Saved: {outprefix}.png and {outprefix}.pdf")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(__doc__))
    parser.add_argument('--cwd', default='.', help='Base working directory (default: current)')
    parser.add_argument('--subdirs', nargs='+', required=True, help='Subdirectory names relative to cwd')
    parser.add_argument('--samefilename', default='gene_vntr_association_results.Autosomes.LessEqualTo_0.000001.tsv',
                        help='Common filename present in each subdir')
    parser.add_argument('--outdir', default='upset_output', help='Output directory for plots')
    parser.add_argument('--verbose', action='store_true', help='Print progress information')
    args = parser.parse_args()

    base_dir = os.path.abspath(args.cwd)
    outdir = os.path.join(base_dir, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    genes_by_group, vntrs_by_group, pairs_by_group = {}, {}, {}  # pairs are gene_vntr pairs

    for subdir in args.subdirs:
        group = 'eVNTRs_MAL' if subdir == 'output_MeanLength_NOT_LMM' else subdir
        group = 'eQTLs_SNP&INDEL' if subdir == 'output_SNPINDELAssociations_MyModel_NoRareAlleles_ML' else group
        group = 'eVNTRs_LS' if subdir == 'output_Zraw_NoRareAlleles_ML' else group
        group = 'eVNTRs_LS' if subdir == 'output_Zraw_NoRareAlleles_Minsamples80_ML' else group
        group = 'eVNTRs_L' if subdir == 'output_Zraw_NoRareAlleles_AllelesGenotypedByLength_ML' else group
        group = 'eVNTRs_L' if subdir == 'output_Zraw_NoRareAlleles_eVNTRL_Minsamples80_ML' else group
        group = 'CollapseRareAlleles' if subdir == 'output_Zraw_CollapseRareAlleles_ML' else group
        group = 'ContrastWithMeanAlleleLength' if subdir == 'output_Zraw_NoRareAlleles_MinusMeanAlleleLength_ML' else group
        group = 'ContrastWithBestSNP/INDEL' if subdir == 'output_Zraw_NoRareAlleles_BestVariant_ML' else group
        group = 'AllAlleles' if subdir == 'output_Zraw_ML' else group
        matches = find_files(base_dir, subdir, args.samefilename)
        if not matches:
            print(f"Warning: No file found in {subdir}", file=sys.stderr)
            genes_by_group[group] = set()
            vntrs_by_group[group] = set()
            pairs_by_group[group] = set()
            continue

        genes, vntrs, pairs = set(), set(), set()
        for m in matches:
            if args.verbose:
                print(f"Reading {m}")
            try:
                g, v, p = read_tsv_extract_sets(m)
                genes |= g
                vntrs |= v
                pairs |= p
            except Exception as e:
                print(f"Error reading {m}: {e}", file=sys.stderr)

        genes_by_group[group] = genes
        vntrs_by_group[group] = vntrs
        pairs_by_group[group] = pairs

    df_genes = build_indicator_df(genes_by_group, 'geneid')
    df_vntrs = build_indicator_df(vntrs_by_group, 'VNTRid')
    df_pairs = build_indicator_df(pairs_by_group, 'geneid_VNTRid')

    plot_upset(df_genes, 'UpSet: geneid membership across groups', os.path.join(outdir, 'upset_genes'))
    plot_upset(df_vntrs, 'UpSet: VNTRid membership across groups', os.path.join(outdir, 'upset_vntrs'))
    plot_upset(df_pairs, 'UpSet: geneid_VNTRid membership across groups', os.path.join(outdir, 'upset_gene_VNTR_pairs'))

    write_upset_intersection_files(df_genes, "genes", outdir)
    write_upset_intersection_files(df_vntrs, "vntrs", outdir)
    write_upset_intersection_files(df_pairs, "gene_vntr_pairs", outdir)

    print('Done.')


if __name__ == '__main__':
    # Silence chained assignment warnings from upsetplot for cleaner output
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='upsetplot')
    main()
