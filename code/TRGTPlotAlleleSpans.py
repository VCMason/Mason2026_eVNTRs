import os
import argparse
import gzip
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from collections import defaultdict

def parse_allele_motif_spans(allele_motif_spans):
    """Parses the AlleleMotifSpans column and returns a list of tuples (motif, start, end) for each allele."""
    alleles = allele_motif_spans.split(',')
    parsed_alleles = []
    for count, allele in enumerate(alleles):
        motifs = []
        if allele == '.':
            print(f"No motifs found for allele: {count}.")
            motifs.append((0, 0, 1))
        else:
            for motif_data in allele.split('_'):
                motif_id, span = motif_data.split('(')
                start, end = map(int, span.strip(')').split('-'))
                motifs.append((int(motif_id), start, end))
            parsed_alleles.append(motifs)
    return parsed_alleles

def load_data(filename, locus_ids=None):
    """Reads the input file and extracts relevant data."""
    data = []
    with gzip.open(filename, 'rt') as FILE:
        header = FILE.readline().strip().split('\t')
        index_map = {col: i for i, col in enumerate(header)}
        
        for line in FILE:
            cols = line.strip().split('\t')
            locus_id = cols[index_map['LocusID']]
            if locus_ids and locus_id not in locus_ids:
                continue
            
            alleleids = list(map(int, cols[index_map['AlleleIDs']].split(',')))
            counts = list(map(int, cols[index_map['CountsPerAllele']].split(',')))
            print(f"counts: {counts}")
            allele_spans = cols[index_map['AlleleMotifSpans']]
            motif_spans = parse_allele_motif_spans(allele_spans)
            print(f"counts: {len(counts)}, alleles: {len(motif_spans)}")
            # print(f"parsed_alleles: {parsed_alleles}")
            
            data.append((locus_id, alleleids, counts, motif_spans))
    
    return data

def plot_alleles(locus_id, alleleids, counts, alleles, motif_colors, outfile):
    """
    alleleids = list of allele IDs
    counts = list of counts for each allele
    alleles = motif spans
    Generates a waterfall-like plot for the given alleles."""
    
    # sorted_alleles = sorted([(idx, count, motifs) for (idx, count, motifs) in zip(alleleids, counts, alleles)], key=lambda x: (-x[2], sum(m[3] - m[2] for m in x[3])))
    sorted_alleles = sorted([(allele_id, count, motifs) for allele_id, count, motifs in zip(alleleids, counts, alleles)], key=lambda x: (-x[1], sum(m[2] - m[1] for m in x[2])))
    for idx, count, motifs in zip(alleleids, counts, alleles):
        print(f"idx={idx}, count={count}, motifs={motifs}, sum={[m[2] - m[1] for m in motifs]}\n")


    # sorted_alleles = sorted(zip(counts, alleles), key=lambda x: (-x[0], sum(m[2] - m[1] for m in x[1])))
    ### allele_data = list(zip(counts, alleles))
    ### allele_data.sort(key=lambda x: (-x[0]))
    ### counts, alleles = zip(*allele_data)
    ### print(f'{counts}, {allele_data}\n')
    ### print(f'{alleles}\n')
    ### print(f'{[f"{i}, {count}, {motifs}" for i, (count, motifs) in enumerate(zip(counts, alleles))]}\n')
    
    # fig, ax = plt.subplots(figsize=(7, len(alleles) * 0.4))
    fig, ax = plt.subplots(figsize=(7, 9))  # for paper
    # ax.yaxis.set_tick_params(pad=10)  # Adjust padding for better alignment

    y_ticks = []
    y_labels = []
    unique_motifs = set()
    
    ### for i, (count, motifs) in enumerate(zip(counts, alleles)):
    for row, (allele_id, count, motifs) in enumerate(sorted_alleles):
        y = len(alleles) - row - 1  # Reverse order
        y_ticks.append(y)
        y_labels.append(f'Allele {allele_id} (Count: {count})')
        for motif_id, start, end in motifs:
            unique_motifs.add(motif_id)
            ax.add_patch(
                patches.Rectangle(
                    (start, y), end - start, 0.9, 
                    facecolor=motif_colors[motif_id], edgecolor='black'
                )
            )
    # for i, (count, motifs) in enumerate(sorted_alleles):
    #     y = len(alleles) - i - 1  # Reverse order
    #     y_ticks.append(y)
    #     y_labels.append(f'Allele {i} (Count: {count})')
    #     
    #     for motif_id, start, end in motifs:
    #         unique_motifs.add(motif_id)
    #         ax.add_patch(
    #             patches.Rectangle(
    #                 (start, y), end - start, 0.9, 
    #                 facecolor=motif_colors[motif_id], edgecolor='black'
    #             )
    #         )
            ###DEBUG print(f'{y}: {i}, {motif_id}, {start}, {end}')
    
    # ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_yticks(y_ticks)          # keep ticks/grid at y
    ax.set_yticklabels([])          # hide default labels
    for y, label in zip(y_ticks, y_labels):
        ax.text(
            -0.02,                  # slightly left of the axis
            y + 0.45,               # center of the row
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Allele Length", fontsize=12)
    ax.set_ylabel(f"Number of Alleles: {len(alleles)}", fontsize=12, labelpad=85)  # increase labelpad to move the label further left from the y axis this is only needed because i hide the y tick labels and then replot them
    ax.set_title(f'Locus: {locus_id}', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, max(max(m[2] for m in motifs) for motifs in alleles))
    ax.set_ylim(0, len(alleles))  # Ensure all alleles are visible
    
    # Create legend for motifs
    unique_motifs = sorted(unique_motifs)  # [:20]
    handles = [patches.Patch(color=motif_colors[m], label=f'Motif {m}') for m in unique_motifs]
    legend = ax.legend(handles=handles, title="Motif Legend", loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=6, fontsize=8, frameon=True)  #  
    
    plt.savefig(outfile, format="pdf", bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {outfile}")


def main():
    # USAGE:
    # Example: python TRGTPlotAlleleSpans.py --ids 11.104306042.104306314 --infile /Users/vmason/Documents/Analyses/trgt/v1.4.1/120Hifi94Calves18Ass/output/size/TRdatabase/SplitByChromosome/Autosomes/Autosomes_TRDB_LocusStats.txt.gz
    # Example: python TRGTPlotAlleleSpans.py --ids X.397808.397842 --infile /Users/vmason/Documents/Analyses/trgt/v1.4.1/120Hifi94Calves18Ass/output/size/TRdatabase/SplitByChromosome/X/TRDB_LocusStats.txt.gz
    # Example: python TRGTPlotAlleleSpans.py --ids 1.157546047.157546607 --infile /Users/vmason/Documents/Analyses/trgt/v1.4.1/120Hifi94Calves18Ass/output/size/TRdatabase/SplitByChromosome/Autosomes/Autosomes_TRDB_LocusStats.txt.gz

    parser = argparse.ArgumentParser(description="Plot allele motifs from LocusStats data.")
    parser.add_argument("--ids", nargs='*', help="List of LocusIDs to plot (optional). If empty, plots all.")
    parser.add_argument("--infile", help="Gzipped LocusStats file.")
    parser.add_argument("--outdir", default=None, help="Output sub-directory for plots (infile path + this subdirectory). If not specified, uses the input file's directory.")
    args = parser.parse_args()
    
    data = load_data(args.infile, set(args.ids) if args.ids else None)
    outdir, infilename = os.path.split(args.infile)
    if args.outdir:
        outdir = os.path.join(outdir, args.outdir)

    motif_palette = sns.color_palette("tab20", n_colors=20)
    motif_colors = defaultdict(lambda: motif_palette[len(motif_colors) % 20])
    
    for locus_id, alleleids, counts, motif_spans in data:
        outfile = os.path.join(outdir, f'{locus_id}.allelspans.pdf')
        plot_alleles(locus_id, alleleids, counts, motif_spans, motif_colors, outfile)

if __name__ == "__main__":
    main()
