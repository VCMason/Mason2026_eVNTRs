#!/usr/bin/env python3
# Full updated script replacing placeholder
# pip install adjustText
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
from adjustText import adjust_text

matplotlib.use('Agg')

# ---------------------------------------------------------------
# Parse VNTRid "chrom.start.end"
# ---------------------------------------------------------------
def parse_vntrid(v):
    try:
        chrom, start, end = v.split('.')
        return str(chrom), int(start), int(end)
    except:
        try:
            chrom, start, varianttype, ref, alt = v.split('_')
            start = int(start)
            return str(chrom), start, start + len(alt)-1  # use start as end if not provided
        except:
            return None, None, None

# ---------------------------------------------------------------
# Load chromosome name map
# ---------------------------------------------------------------
def load_chrom_map(path):
    df = pd.read_csv(path, sep="\t", header=0)
    mapping = dict(zip(df['RefSeq'].astype(str), df['Chrom'].astype(str)))
    return mapping

# ---------------------------------------------------------------
# Normalize chromosome names
# ---------------------------------------------------------------
def normalize_chrom_name(name, chrom_map):
    if name in chrom_map:
        return str(chrom_map[name])
    clean = name.replace('chr','').replace('CHR','').replace('Chr','')
    return clean

# ---------------------------------------------------------------
# Load results file
# ---------------------------------------------------------------
def load_results_file(path):
    df = pd.read_csv(path, sep="\t", header=0)
    parsed = df['VNTRid'].apply(parse_vntrid)
    df['chrom'] = parsed.apply(lambda x: x[0])
    df['start'] = parsed.apply(lambda x: x[1])
    df['end'] = parsed.apply(lambda x: x[2])
    if 'h2' in df.columns:
        df['h2'] = pd.to_numeric(df['h2'], errors='coerce')
    elif 'r2' in df.columns:
        df['r2'] = pd.to_numeric(df['r2'], errors='coerce')
    df['raw_pvalue'] = pd.to_numeric(df['raw_pvalue'], errors='coerce')
    df = df[df['raw_pvalue'] > 0]
    print("Top 5 variants by pvalue:")
    print(df.nsmallest(5, 'raw_pvalue')[['VNTRid','chrom','start','raw_pvalue']])
    print(f"Keeping {len(df)} variants with p-value > 0 from {os.path.basename(path)}")
    # print(df[df['VNTRid'] == '27_7719744_SNP_A_C'])  # example of a specific variant to check
    # print(f"Keeping only p-values > 0 and <= 0.1 from {os.path.basename(path)}")
    # df = df[(df['raw_pvalue'] > 0) & (df['raw_pvalue'] <= 0.1)]
    df['mlogp'] = -np.log10(df['raw_pvalue'])
    # print(df[df['VNTRid'] == '27_7719744_SNP_A_C'])  # example of a specific variant to check
    return df


# ---------------------------------------------------------------
# Load Qtltools results file
# ---------------------------------------------------------------
def load_filtered_data(path, geneid=None, chrom=None, start=None, end=None):
    column_names = [
        'phenotype_id', 'phe_chrom', 'phe_start', 'phe_end', 'phe_strand',
        'num_variants_tested', 'distance_to_variant', 'variant_id', 'var_chrom',
        'var_start', 'var_end', 'raw_pvalue', 'unknonwn1', 'slope', 'unknown2', 'is_top_variant'
    ]
    dtypes = {'phenotype_id': str, 'variant_id': str, 'var_chrom': str}

    # If geneid is provided, read in chunks to filter rows before joining
    if geneid is not None:
        chunks = pd.read_csv(
            path, 
            sep=" ", 
            header=None, 
            names=column_names, 
            dtype=dtypes, 
            chunksize=100000  # Adjust chunk size based on available RAM
        )
        # Filter each chunk and concatenate the results
        return pd.concat([chunk[chunk['phenotype_id'] == geneid] for chunk in chunks])
    elif chrom is not None and start is not None and end is not None:
        chunks = pd.read_csv(
            path, 
            sep=" ", 
            header=None, 
            names=column_names, 
            dtype=dtypes, 
            chunksize=100000  # Adjust chunk size based on available RAM
        )
        return pd.concat([
            chunk[
                (chunk['var_chrom'] == chrom) & 
                (chunk['var_start'] >= start) & 
                (chunk['var_start'] <= end)
            ] for chunk in chunks
        ])
    
    # Otherwise, read the full file (pandas handles .gz automatically)
    return pd.read_csv(path, sep=" ", header=None, names=column_names, dtype=dtypes, low_memory=False)


def load_qtltools_results_file(path, geneid=None, chrom=None, start=None, end=None):
    '''
    No header in Qtltools nominal pvalue output, so we assign columns based on Qtltools documentation:

    Nominal pvalue Qtltools results:
    1. The phenotype ID
    2. The chromosome ID of the phenotype
    3. The start position of the phenotype
    4. The end position of the phenotype
    5. The strand orientation of the phenotype
    6. The total number of variants tested in cis
    7. The distance between the phenotype and the tested variant (accounting for strand orientation)
    8. The ID of the tested variant
    9. The chromosome ID of the variant
    10. The start position of the variant
    11. The end position of the variant
    12. The nominal P-value of association between the variant and the phenotype
    13. The corresponding regression slope
    14. A binary flag equal to 1 is the variant is the top variant in cis

    # in the file from Mapel et al. there are 16 columns not 14
    # i hard coded the column names into load_filtered_data() to account for the column difference, this can be changed if needed to follow: https://qtltools.github.io/qtltools/
        1. The phenotype ID
        2. The chromosome ID of the phenotype
        3. The start position of the phenotype
        4. The end position of the phenotype
        5. The strand orientation of the phenotype
        6. The total number of variants tested in cis
        7. The distance between the phenotype and the tested variant (accounting for strand orientation)
        8. The ID of the tested variant
        9. The chromosome ID of the variant
        10. The start position of the variant
        11. The end position of the variant
        12. The nominal P-value of association between the variant and the phenotype
        13. ???
        14. ??? The corresponding regression slope. there are negative values, so assume this is slope
        15. ???
        14. A binary flag equal to 1 is the variant is the top variant in cis
    '''

    # df = pd.read_csv(path, sep=" ", header=None, names=[
    #     'phenotype_id', 'phe_chrom', 'phe_start', 'phe_end', 'phe_strand',
    #     'num_variants_tested', 'distance_to_variant', 'variant_id', 'var_chrom',
    #     'var_start', 'var_end', 'raw_pvalue', 'unknown1', 'slope', 'unknown2', 'is_top_variant'
    #     ], dtype={'phenotype_id': str, 'variant_id': str, 'var_chrom': str}, low_memory=False)
    # df = df[df['phenotype_id'] == geneid] if geneid is not None else df  # filter the df to only the target gene if geneid is provided

    df = load_filtered_data(path, geneid)

    # parsed = df['variant_id'].apply(parse_vntrid)
    df['VNTRid'] = df['variant_id']  # keep original variant_id as VNTRid for labeling (during plotting)
    df['chrom'] = df['var_chrom']  # use var_chrom as chrom
    df['start'] = df['var_start']  # use var_start as start
    df['end'] = df['var_end']  # use var_end as end
    df['raw_pvalue'] = pd.to_numeric(df['raw_pvalue'], errors='coerce')
    df = df[df['raw_pvalue'] > 0]
    print("Top 5 variants by pvalue:")
    print(df.nsmallest(5, 'raw_pvalue')[['VNTRid','chrom','start','raw_pvalue']])
    print(f"Keeping {len(df)} variants with p-value > 0 from {os.path.basename(path)}")
    # print(f"Keeping only p-values > 0 and <= 0.1 from {os.path.basename(path)}")
    # df = df[(df['raw_pvalue'] > 0) & (df['raw_pvalue'] <= 0.1)]
    df['mlogp'] = -np.log10(df['raw_pvalue'])

    return df


# ---------------------------------------------------------------
# Load GFF
# ---------------------------------------------------------------
def load_gff(path):
    cols = ['seqid','source','type','start','end','score','strand','phase','attributes']
    df = pd.read_csv(path, sep="\t", comment='#', header=None, names=cols)
    return df

# ---------------------------------------------------------------
# Extract gene name
# ---------------------------------------------------------------
def gff_get_gene_name(attr):
    for field in attr.split(';'):
        if field.startswith('Name='):
            return field.split('=')[1]
        if field.startswith('gene='):
            return field.split('=')[1]
    return None

# ---------------------------------------------------------------
# Extract genes in region
# ---------------------------------------------------------------
def extract_genes_in_region(gff, chrom, start, end, chrom_map, target_gene=False):
    gff['chrom_norm'] = gff['seqid'].apply(lambda x: normalize_chrom_name(str(x), chrom_map))
    sub = gff[(gff['type']=='gene') & (gff['chrom_norm']==str(chrom)) & (gff['end']>=start) & (gff['start']<=end)].copy()
    sub['gene_name'] = sub['attributes'].apply(gff_get_gene_name)
    if target_gene:
        sub = sub[sub['gene_name'] == target_gene]
    return sub

# ---------------------------------------------------------------
# Determine region
# ---------------------------------------------------------------
def determine_region(args, gff, chrom_map):
    if args.region:
        print("Using --region; prioritizing over geneid/window")
        c, rest = args.region.split(':')
        s, e = rest.split('-')
        chrom = normalize_chrom_name(c, chrom_map)
        return chrom, int(s), int(e)

    if args.geneid:
        print("Using --geneid and --window to define region")
        gff['chrom_norm'] = gff['seqid'].apply(lambda x: normalize_chrom_name(str(x), chrom_map))
        genes = gff[gff['type']=='gene'].copy()
        genes['gene_name'] = genes['attributes'].apply(gff_get_gene_name)
        tg = genes[genes['gene_name']==args.geneid]
        if tg.empty:
            raise ValueError(f"Gene {args.geneid} not found in GFF")
        row = tg.iloc[0]
        chrom = row['chrom_norm']
        start = int(row['start']) - args.window
        end = int(row['start']) + args.window
        return chrom, start, end

    raise ValueError("You must specify either --region or --geneid + --window")

# ---------------------------------------------------------------
# Single plot
# ---------------------------------------------------------------
def plot_single(df, chrom, start, end, genes, highlight_gene, outpath, marker):
    # Create figure and subset region
    df_region = df[(df['chrom']==str(chrom)) & (df['start']>=start) & (df['start']<=end)]
    # print(df_region[df_region['VNTRid'] == '27_7719744_SNP_A_C'])
    fig, ax = plt.subplots(figsize=(12,6))
    if len(df_region)>0:
        if 'h2' in df_region.columns:
            sc = ax.scatter(df_region['start'], df_region['mlogp'], c=df_region['h2'], marker=marker, s=25)
            plt.colorbar(sc, ax=ax, label='h2')
        elif 'Partialr2' in df_region.columns:
            sc = ax.scatter(df_region['start'], df_region['mlogp'], c=df_region['Partialr2'], marker=marker, s=25)
            plt.colorbar(sc, ax=ax, label='Partial r2')
        elif 'r2' in df_region.columns:
            sc = ax.scatter(df_region['start'], df_region['mlogp'], c=df_region['r2'], marker=marker, s=25)
            plt.colorbar(sc, ax=ax, label='r2')
        else:
            ax.scatter(df_region['start'], df_region['mlogp'], marker=marker, s=25)
        # Label the top variant (highest -log10 p) in this region
        try:
            top_idx = df_region['mlogp'].idxmax()
            top_row = df_region.loc[top_idx]
            top_label = str(top_row.get('VNTRid', top_row.get('id', 'top')))
            x_top = float(top_row['start'])
            y_top = float(top_row['mlogp'])
            x_off = 0  # x_off = (end - start) * 0.01 if end is not None and start is not None and end > start else 1.0
            ymin_tmp, ymax_tmp = ax.get_ylim()
            y_off = (ymax_tmp - ymin_tmp) * 0.01
            ax.text(x_top + x_off, y_top + y_off, top_label, fontsize=8, color='black', ha='left', va='bottom')
        except Exception:
            pass

    # Improved gene annotation with offsets and arrows
    ymin, ymax = ax.get_ylim()
    height = (ymax - ymin)
    # Place annotation area just below the plotted points (closer to 0 on y-axis)
    annotation_top = ymin - height * 0.02  # small gap below data
    step = height * 0.03  # vertical spacing between stacked gene labels
    arrow_gap = height * 0.008  # small gap between label and arrow

    # Compute lowest needed y to ensure all labels/arrows are visible, then extend axis if required
    max_label_rows = min(len(genes), 6)
    lowest_needed = annotation_top - (max_label_rows - 1) * step - height * 0.02
    if lowest_needed < ymin:
        ax.set_ylim(bottom=lowest_needed)

    for j, (_, row) in enumerate(genes.iterrows()):
        # Stack labels downward from annotation_top
        text_y = annotation_top - (j % 6) * step
        arrow_y = text_y - arrow_gap

        # Gene label: slightly above the arrow (closer to the plotted area)
        ax.text((row['start'] + row['end']) / 2, text_y, row['gene_name'],
                ha='center', fontsize=7,
                color='red' if highlight_gene and row['gene_name'] == highlight_gene else 'black')

        # Draw arrow a bit below the gene label
        arrow_color = 'red' if highlight_gene and row['gene_name'] == highlight_gene else 'black'
        try:
            if str(row['strand']) == '+':
                # arrow from start -> end at arrow_y
                ax.annotate('', xy=(row['end'], arrow_y), xytext=(row['start'], arrow_y),
                            arrowprops=dict(arrowstyle='-|>', lw=1.2, color=arrow_color))
            else:
                # reverse direction for '-' strand
                ax.annotate('', xy=(row['start'], arrow_y), xytext=(row['end'], arrow_y),
                            arrowprops=dict(arrowstyle='-|>', lw=1.2, color=arrow_color))
        except Exception:
            pass
    
    # ax.set_xlim(start, end)
    ax.set_xlabel(f"Position on chr{chrom} in Mb")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"{highlight_gene}: chr{chrom}:{start}-{end}")
    plt.tight_layout()
    fig.savefig(outpath + '.png')
    fig.savefig(outpath + '.pdf')
    print(f"Saved single plot to:\n{outpath}.png\nand\n{outpath}.pdf")
    plt.close(fig)

# ---------------------------------------------------------------
# Combined plot
# ---------------------------------------------------------------
def plot_combined(dfs, chrom, start, end, genes, highlight_gene, outpath, markers, bufferymax):
    # fig, ax = plt.subplots(figsize=(12,7))
    # Determine global max h2 across all dfs
    max_r2 = 0
    for df in dfs:
        if 'h2' in df.columns:
            m = df['h2'].max()
            if pd.notnull(m) and m > max_r2:
                max_r2 = m
            colortype = 'h2'
        elif 'Partialr2' in df.columns:
            m = df['Partialr2'].max()
            if pd.notnull(m) and m > max_r2:
                max_r2 = m
            colortype = 'Partialr2'
        else:
            pass
            # colortype = None

    norm = plt.Normalize(0, max_r2)
    fig, ax = plt.subplots(figsize=(7,3.5))  # 12, 6
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

    custom_lines = []
    labels = []
    texts = []
    sc = None

    for i, df in enumerate(dfs):
        df_region = df[(df['chrom']==str(chrom)) & (df['start']>=start) & (df['start']<=end)]
        # print('###')
        # print("file", i+1)
        # print("min start:", df_region['start'].min())
        # print("max start:", df_region['start'].max())
        # print(df_region[df_region['VNTRid'] == '27_7719744_SNP_A_C'])
        # print(df_region.loc[df_region['VNTRid']=='27_7719744_SNP_A_C', ['start','mlogp','h2']])
        if len(df_region)>0:
            if 'h2' in df.columns:
                sc = ax.scatter(df_region['start'], df_region['mlogp'], c=df_region['h2'], marker=markers[i], s=25, cmap='viridis', norm=norm, alpha=0.7)  #norm=norm,
                # ax.scatter(7719744, 33.604832)
            elif 'Partialr2' in df.columns:
                sc = ax.scatter(df_region['start'], df_region['mlogp'], c=df_region['Partialr2'], marker=markers[i], s=25, cmap='viridis', norm=norm, alpha=0.7)  #  norm=norm,
            else:
                ax.scatter(df_region['start'], df_region['mlogp'], marker=markers[i], s=25, alpha=0.7)
            # Label the top variant for this file (stagger vertical offsets to reduce overlap)
            try:
                top_idx = df_region['mlogp'].idxmax()
                top_row = df_region.loc[top_idx]
                top_label = f'F{i+1}: '+ str(top_row.get('VNTRid', top_row.get('id', f'top{i+1}')))
                x_top = float(top_row['start'])
                y_top = float(top_row['mlogp'])
                x_off = 0  # x_off = (end - start) * 0.01 if end is not None and start is not None and end > start else 1.0
                ymin_tmp, ymax_tmp = ax.get_ylim()
                stagger = (0.005 + 0.01 * (i % 4))
                y_off = (ymax_tmp - ymin_tmp) * stagger
                # ax.text(x_top + x_off, y_top + y_off, top_label, fontsize=7, color='black', ha='left', va='bottom')
                t = ax.text(x_top + x_off, y_top, top_label, fontsize=7, color='black', ha='left', va='bottom')
                texts.append(t)
                # ax.set_ylim(bottom=ymin_tmp, top=ymax_tmp + (ymax_tmp - ymin_tmp) * 0.01)  # add extra space on top for labels
                ax.set_ylim(bottom=ymin_tmp, top=max(ymax_tmp + (ymax_tmp - ymin_tmp) * bufferymax, y_top + y_off + (ymax_tmp - ymin_tmp) * bufferymax))  # adjust decimals 0.01 -> 0.05 if more space needed on ymax placement
            except Exception:
                pass
        custom_lines.append(Line2D([0],[0], marker=markers[i], color='black', linestyle=''))
        labels.append(f"File {i+1}")

    # Call adjust_text outside and after the loop completes
    if texts:
        adjust_text(texts, va='center', ha='left')
        # adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.75), \
        #                                     va='center', \
        #                                     ha='left')
        #                                     # connectionstyle="angle,angleA=180,angleB=90,rad=0"), \
        #                                     # va = # Forces vertical center alignment for the label anchor
        #                                     # ha = # Forces the label anchor to lock to the left side

    if sc is not None:
        if colortype == 'h2':
            plt.colorbar(sc, ax=ax, label='h2')
        elif colortype == 'Partialr2':
            plt.colorbar(sc, ax=ax, label='Partial r2')

    # Improved gene annotation (labels close to plotted area, arrows just below labels)
    ymin, ymax = ax.get_ylim()
    height = (ymax - ymin)
    annotation_top = ymin - height * 0.02
    step = height * 0.03
    arrow_gap = height * 0.008

    max_label_rows = min(len(genes), 6)
    lowest_needed = annotation_top - (max_label_rows - 1) * step - height * 0.02
    if lowest_needed < ymin:
        ax.set_ylim(bottom=lowest_needed)

    for j, (_, row) in enumerate(genes.iterrows()):
        text_y = annotation_top - (j % 6) * step
        arrow_y = text_y - arrow_gap
        arrow_color = 'red' if highlight_gene and row['gene_name'] == highlight_gene else 'black'
        try:
            if str(row['strand']) == '+':
                ax.annotate('', xy=(row['end'], arrow_y), xytext=(row['start'], arrow_y),
                            arrowprops=dict(arrowstyle='-|>', lw=1.2, color=arrow_color))
            else:
                ax.annotate('', xy=(row['start'], arrow_y), xytext=(row['end'], arrow_y),
                            arrowprops=dict(arrowstyle='-|>', lw=1.2, color=arrow_color))
        except Exception:
            pass
        # Gene label centered above the arrow
        ax.text((row['start'] + row['end']) / 2, text_y, row['gene_name'],
                ha='center', fontsize=7,
                color=arrow_color)

    # ax.set_xlim(start, end)
    ax.set_xlabel(f"Position on chr{chrom} in Mb")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"{highlight_gene}: chr{chrom}:{start}-{end}")
    ax.legend(custom_lines, labels, loc='upper right')
    plt.tight_layout()
    fig.savefig(outpath + '.png')
    fig.savefig(outpath + '.pdf')
    print(f"Saved combined plot to:\n{outpath}.png\n{outpath}.pdf")
    plt.close(fig)

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, nargs='+')
    ap.add_argument('--qtltools_results', required=False, default=None, nargs='+')
    ap.add_argument('--gff', required=False, help=".gff file for gene annotations, requires --ren")
    ap.add_argument('--renamechrom', required=False)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--region', default=None, help="Region to plot in format chrom:start-end (e.g. 27:7500000-8000000). Overrides --geneid and --window if provided.")
    ap.add_argument('--geneid', default=None)
    ap.add_argument('--targetgeneonly', action='store_true', required=False, help="If set, only the target gene specified by --geneid will be plotted (no other genes in the region will be shown), and the region will be defined as gene start +/- --window. This is useful for zooming in on a specific gene without showing nearby genes. Expects arguments")
    ap.add_argument('--window', type=int, default=1000000, help="Window size around the target gene to plot (default: 1000000). ")
    ap.add_argument('--bufferymax', type=float, default=0.01, help="If labels or plot points are getting cut off at top of plot increase this scaling value (ex: 0.01->0.05) to raise ymax value (default: 0.01)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    args.window = args.window / 2  # we will use the window as +/- around the gene start, so we divide by 2 here to get the correct total window size

    print("Using --targetgeneonly: will only plot the target gene and not other genes in the region") if args.targetgeneonly else print("Not using --targetgeneonly: will plot all genes in the region")
    if args.gff and args.renamechrom and args.region is None:
        print("Loading GFF and chromosome map to automatically determine region and genes to plot...")
        chrom_map = load_chrom_map(args.renamechrom)
        gff = load_gff(args.gff)
        chrom, start, end = determine_region(args, gff, chrom_map)
        # if args.targetgeneonly is set, we reduce genes to target gene only, but we still need to extract it from the gff to get coordinates and strand for plotting
        genes = extract_genes_in_region(gff, chrom, start, end, chrom_map, target_gene=args.geneid)
    elif args.region and args.gff and args.renamechrom:
            print("Using --region; prioritizing over geneid/window, specified region:", args.region)
            c, rest = args.region.split(':')
            s, e = rest.split('-')
            chrom, start, end = c, int(s), int(e)
            # if args.targetgeneonly is set, we reduce genes to target gene only, but we still need to extract it from the gff to get coordinates and strand for plotting
            genes = extract_genes_in_region(gff, chrom, start, end, chrom_map, target_gene=args.geneid)
    elif args.region:
        print("Using --region; prioritizing over geneid/window, specified region:", args.region)
        c, rest = args.region.split(':')
        s, e = rest.split('-')
        chrom, start, end = c, int(s), int(e)
        # make empty dataframe for genes, so no genes will be plotted
        genes = pd.DataFrame()
    else:
        raise ValueError("You must specify either --region or --geneid + --window to determine the region to plot")
    
    print(f"Final region: chr{chrom}:{start}-{end}")
    print(f"Targeted gene: {args.geneid}") if args.geneid else "No specific target gene"

    dfs = [load_results_file(p) for p in args.results]
    if args.qtltools_results is not None:
        dfs_qtltools = [load_qtltools_results_file(p, args.geneid, chrom, start, end) for p in args.qtltools_results]
        dfs.extend(dfs_qtltools)

    markers = ['o','x','^','*','v','D','P','s','H','X']  # cycle through different marker styles for each file

    for i, df in enumerate(dfs):
        out = os.path.join(args.outdir, f"zoom_file{i+1}.{args.geneid}.{chrom}.{start}-{end}")
        plot_single(df, chrom, start, end, genes, args.geneid, out, markers[i % len(markers)])

    if len(dfs) > 1:
        out = os.path.join(args.outdir, f"zoom_combined.{args.geneid}.{chrom}.{start}-{end}")
        plot_combined(dfs, chrom, start, end, genes, args.geneid, out, markers, args.bufferymax)

if __name__=='__main__':
    # USAGE:

    '''
    (py3.10)

    # Simple usage:
    GENE="SPATA4"
    python PlotZoom.py \
    --results \
    /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP/output_${GENE}/gene_vntr_association_results.${GENE}.tsv \
    /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_BLUP_SNPsINDELs/output_${GENE}/gene_vntr_association_results.${GENE}.tsv \
    --outdir /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_BLUP_SNPsINDELs/output_${GENE}/ZoomPlots_ML_NoRareAlleles \
    --region 8:24638014-25638014

    # With gene annotation and automatic region determination based on gene coordinates in GFF (using --geneid and --window):
    GENE="SPATA4"
    python PlotZoom.py \
    --results \
    /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_BLUP/output_${GENE}/gene_vntr_association_results.${GENE}.tsv \
    /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_BLUP_SNPsINDELs/output_${GENE}/gene_vntr_association_results.${GENE}.tsv \
    --geneid $GENE \
    --gff /Users/vmason/Documents/Genomes/Bos_taurus_ARS-UCD2.0/Annotation_Feb2025/GCF_002263795.3_ARS-UCD2.0_genomic.gff \
    --renamechrom /Users/vmason/Documents/Genomes/Bos_taurus_ARS-UCD2.0/Annotation_Feb2025/RenameChromMap.txt \
    --outdir /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_BLUP_SNPsINDELs/output_${GENE}/ZoomPlots \
    --window 500000
    '''
    

    main()
