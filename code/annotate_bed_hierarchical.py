#!/usr/bin/env python3
"""
annotate_bed.py

Annotate BED intervals against a GFF3/GTF annotation.

Features:
- Uses fast interval joins (ruranges if available, otherwise PyRanges)
- Annotates every BED interval
- Derives introns, promoters, upstream, downstream
- Reports all overlapping features
- Reports a single priority annotation per peak
- Reports nearest gene and distance to TSS
- Produces summary TSVs and plots

Requirements:
    pip install pandas numpy matplotlib pyranges
    pip install biopython bcbio-gff
Optional (faster):
    pip install ruranges
"""

import os
import random
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators


from BCBio import GFF

try:
    import ruranges as rr
    HAVE_RURANGES = True
except Exception:
    HAVE_RURANGES = False

try:
    import pyranges as pr
except Exception:
    raise SystemExit("Please install pyranges: pip install pyranges")


PRIORITY = [
    "CDS",
    "exon",
    "intron",
    "upstream",
    "downstream",
    "gene",
    "pseudogene",
    "intergenic"
]
# excluded to combine with upstream
#    "promoter",
# these are absent from .gff
    # "five_prime_UTR",
    # "three_prime_UTR",
# these are always exon in priority
    # "lnc_RNA",
    # "tRNA",
    # "miRNA",
    # "snoRNA",
    # "snRNA",
    # "rRNA",
    # "ncRNA",

TRANSCRIPT_TYPES = {
    "mRNA",
    "transcript",
    "lnc_RNA",
    "lncRNA",
    "miRNA",
    "snRNA",
    "snoRNA",
    "rRNA",
    "tRNA",
    "ncRNA",
    "C_gene_segment",
    "V_gene_segment",
    "pseudogene"
}

TOP_LEVEL_IGNORE = {
    "region",
    "chromosome",
    "contig",
    "supercontig",
    "cDNA_match",
    "match"
}


def parse_vntrid(v):
    try:
        chrom, start, end = v.split('.')
        return str(chrom), int(start), int(end)
    except:
        try: # if SNP id like: 6_36837726_SNP_T_G or INDEL id like: 26_13072299_INDEL_T_TG
            chrom, start, varianttype, ref, alt = v.split('_')
            start = int(start)
            return str(chrom), start, start + len(alt)  #-1  # use start as end if not provided
        except:
            return None, None, None


def read_bed(path):
    base, ext = os.path.splitext(path)
    if ext == '.tsv':
        print('Loading .tsv not .bed')
        rows = []
        with open(path, 'r') as FILE:
            header = FILE.readline().strip().split('\t')
            # Check if 'VNTRid' is the second column (index 1)
            if len(header) > 1 and header[1] == 'VNTRid':
                print('Good, .tsv has VNTRid as second column')
                for line in FILE:
                    if not line.strip():
                        continue
                    parts = line.strip().split('\t')
                    # Parse out the location block (e.g., 'chr1.1000.5000')
                    vntr_id = parts[1]
                    chrom, start, end = parse_vntrid(vntr_id)  # vntr_id.split('.')
                    # Use the full VNTRid string as the 'Name' column to mimic the BED format
                    rows.append({
                        "Chromosome": chrom,
                        "Start": int(start),
                        "End": int(end),
                        "Name": vntr_id,
                        "GeneID": parts[0]
                    })
                    # geneID is the gene the VNTR is associated with
                    # use geneID to filter the gff
        # Convert the collected rows into the exact same 4-column layout
        df = pd.DataFrame(rows, columns=["Chromosome", "Start", "End", "Name", "GeneID"])
        if df.empty:
            raise ValueError("TSV file contained no valid VNTR entries")
    else:
        cols = ["Chromosome", "Start", "End", "Name"]
        df = pd.read_csv(path, sep="\t", header=None, comment="#", low_memory=False)
        if df.shape[1] < 3:
            raise ValueError("BED must have at least 3 columns")
        while df.shape[1] < 4:
            df[df.shape[1]] = [f"peak_{i}" for i in range(len(df))]
        df = df.iloc[:, :4]
        df.columns = cols

    return df

        
def parse_gff(gff_file, promoter=2000, upstream=10000, downstream=10000):

    from BCBio import GFF

    features = []
    skippedtoplevel = []
    chosentoplevel = []
    strand_map = {1: "+", "1": "+", "+": "+", -1: "-", "-1": "-", "-": "-"}  # convert all strands to + or -
    with open(gff_file) as handle:
        parser = GFF.GFFParser()
        for rec in parser.parse(handle):
            chrom = rec.id
            for gene in rec.features:
                if gene.type in TOP_LEVEL_IGNORE:  #  not in {"gene", "pseudogene"}:
                    skippedtoplevel.append(gene.type)
                    continue
                else:
                    chosentoplevel.append(gene.type)
                # if gene.type != "gene":
                #     continue
                gene_name = gene.qualifiers.get(
                    "gene",
                    gene.qualifiers.get("Name", ["unknown"])
                )[0]
                # gene_biotype = gene.qualifiers.get(
                #     "gene_biotype",
                #     ["unknown"]
                # )[0]
                gene_biotype = (
                    gene.qualifiers.get("gene_biotype")
                    or gene.qualifiers.get("gene_biotype=")
                    or [gene.type]
                )[0]

                gstart = int(gene.location.start)
                gend = int(gene.location.end)
                strand = gene.location.strand
                strand = strand_map[strand]
                features.append({
                    "Chromosome": chrom,
                    "Start": gstart,
                    "End": gend,
                    "Strand": strand,
                    "Feature": gene.type,
                    "TranscriptType": "gene",
                    "GeneBiotype": gene_biotype,
                    "GeneName": gene_name
                })
                # if gene_name == 'AARS1':
                #     print(f"start: {gstart} end: {gend} strand: {strand}")

                ###############################################
                # promoter / upstream / downstream
                ###############################################

                if strand == 1 or strand == "+":
                    # promoter_start = max(0, gstart-promoter)
                    # promoter_end = gstart

                    upstream_start = max(0, gstart-upstream)
                    upstream_end = gstart
                    # upstream_end = promoter_start

                    downstream_start = gend
                    downstream_end = gend+downstream
                else:
                    # promoter_start = gend
                    # promoter_end = gend+promoter

                    # upstream_start = promoter_end
                    upstream_start = gend
                    upstream_end = gend+upstream

                    downstream_start = max(0, gstart-downstream)
                    downstream_end = gstart

                for feature_name, s, e in [
                    # ("promoter", promoter_start, promoter_end),
                    ("upstream", upstream_start, upstream_end),
                    ("downstream", downstream_start, downstream_end)
                ]:
                    features.append({
                        "Chromosome": chrom,
                        "Start": s,
                        "End": e,
                        "Strand": strand,
                        "Feature": feature_name,
                        "TranscriptType": feature_name,
                        "GeneBiotype": gene_biotype,
                        "GeneName": gene_name
                    })

                ###############################################
                # recursive walk
                ###############################################

                def recurse(feature,
                            transcript_type=None):

                    if feature.type in TRANSCRIPT_TYPES:
                        transcript_type = feature.type

                    start = int(feature.location.start)
                    end = int(feature.location.end)

                    features.append({
                        "Chromosome": chrom,
                        "Start": start,
                        "End": end,
                        "Strand": strand,
                        "Feature": feature.type,
                        "TranscriptType": transcript_type,
                        "GeneBiotype": gene_biotype,
                        "GeneName": gene_name
                    })

                    # exons = []
                    # for child in feature.sub_features:
                    #     if child.type == "exon":
                    #         exons.append(child)
                    #     
                    #     recurse(child, transcript_type)
                    
                    # ONLY harvest exons if the current feature is an actual transcript container
                    exons = []
                    for child in feature.sub_features:
                        if child.type == "exon" and feature.type in TRANSCRIPT_TYPES:
                            exons.append(child)

                        # Pass down the verified transcript type string to the child
                        recurse(child, transcript_type)

                    ##########################################
                    # infer introns
                    ##########################################

                    if len(exons) > 1:
                        exons = sorted(
                            exons,
                            key=lambda x: int(x.location.start)
                        )
                        for i in range(len(exons)-1):
                            s = int(exons[i].location.end)
                            e = int(exons[i+1].location.start)

                            if e > s:
                                features.append({
                                    "Chromosome": chrom,
                                    "Start": s,
                                    "End": e,
                                    "Strand": strand,
                                    "Feature": "intron",
                                    "TranscriptType": transcript_type,
                                    "GeneBiotype": gene_biotype,
                                    "GeneName": gene_name
                                })
                for child in gene.sub_features:
                    # If the child type isn't in TRANSCRIPT_TYPES, use its own type as the starter
                    recurse(child, transcript_type=child.type)
                    # t_type = child.type if child.type in TRANSCRIPT_TYPES else child.type
                    # recurse(child, transcript_type=t_type)
                    # recurse(child)
                # recurse(gene)
    print(f"Skipped top level features: {set(skippedtoplevel)}")
    print(f"Top level features chosen: {set(chosentoplevel)}")
    return pd.DataFrame(features)


# def summarize_feature_space(features):
#     """
#     Summarize the genomic space occupied by each feature.
# 
#     Returns
#     -------
#     feature_space : DataFrame
#         Columns:
#             Feature
#             NIntervals
#             TotalBases
#     """
# 
#     df = features.copy()
#     df["Length"] = df["End"] - df["Start"] + 1
# 
#     feature_space = (
#         df.groupby("Feature")
#           .agg(
#               NIntervals=("Feature", "size"),
#               TotalBases=("Length", "sum")
#           )
#           .reset_index()
#     )
# 
#     return feature_space
# 
# 
# 
# def merge_intervals(intervals):
#     """Efficiently merges overlapping genomic intervals."""
#     if not intervals:
#         return 0
#     
#     # Sort intervals by start position
#     sorted_inv = sorted(intervals, key=lambda x: x[0])
#     merged = [sorted_inv[0]]
#     
#     for current in sorted_inv[1:]:
#         prev_start, prev_end = merged[-1]
#         curr_start, curr_end = current
#         
#         if curr_start <= prev_end + 1:  # +1 handles perfectly adjacent regions
#             # Overlap found, merge by updating the end point
#             merged[-1] = (prev_start, max(prev_end, curr_end))
#         else:
#             # No overlap, add as a new interval
#             merged.append(current)
#             
#     # Return the sum of lengths of all merged intervals
#     return sum(end - start + 1 for start, end in merged)


def merge_intervals(intervals):
    """Efficiently merges overlapping genomic intervals."""
    if not intervals:
        return 0, 0
    sorted_inv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_inv[0]]
    for current in sorted_inv[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        if curr_start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)
    total_bases = sum(end - start + 1 for start, end in merged)
    return total_bases, len(merged)



def summarize_feature_space(features):
    """Summarize the genomic space occupied by each feature type, flattening overlaps."""
    summary_data = []
    chrom_col = "Chromosome"

    for feature_type, group in features.groupby("Feature"):
        total_bases = 0
        original_count = len(group)
        
        if chrom_col in group.columns:
            for chrom, chrom_group in group.groupby(chrom_col):
                chrom_intervals = list(zip(chrom_group["Start"], chrom_group["End"]))
                bases, _ = merge_intervals(chrom_intervals)
                total_bases += bases
        else:
            intervals = list(zip(group["Start"], group["End"]))
            total_bases, _ = merge_intervals(intervals)

        summary_data.append({
            "Feature": feature_type,
            "NIntervals": original_count,
            "TotalBases": total_bases
        })

    return pd.DataFrame(summary_data)



def add_intergenic_space(feature_space, features, args, fix_genome_size=None):
    """Add an intergenic row using interval-aware merging."""
    upstream = args.upstream
    downstream = args.downstream
    chrom_col = "Chromosome"

    # Extract genome size from GFF regions if not fixed
    genome_size = 0
    if fix_genome_size is not None:
        genome_size = fix_genome_size
    else:
        with open(args.gff) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip(): continue
                cols = line.rstrip().split("\t")
                if len(cols) < 9 or cols[2] != "region": continue
                genome_size += int(cols[4]) - int(cols[3]) + 1

    # Drop duplicates of the same gene created by the variant merge
    genes = features.loc[features.Feature == "gene"].drop_duplicates(subset=[chrom_col, "Start", "End"]).copy()
    
    # Calculate unique occupied space by genes + flank expansions
    expanded_intervals = []
    for _, gene in genes.iterrows():
        # Fallback to broad boundaries if chromosome limits aren't explicitly parsed
        gene_start = max(1, gene["Start"] - upstream)
        gene_end = gene["End"] + downstream
        expanded_intervals.append((gene_start, gene_end))

    # Merge to find unique footprint of occupied space
    occupied_space, _ = merge_intervals(expanded_intervals) if expanded_intervals else (0, 0)
    intergenic_space = max(0, genome_size - occupied_space)

    feature_space = pd.concat([
        feature_space,
        pd.DataFrame({
            "Feature": ["intergenic"],
            "NIntervals": [np.nan],
            "TotalBases": [intergenic_space]
        })
    ], ignore_index=True)

    return feature_space



# def add_intergenic_space(feature_space, features, args, fix_genome_size=None):
#     """
#     Add an intergenic row to feature_space using chromosome lengths
#     accounting for overlapping upstream/downstream gene boundaries.
#     """
#     gff_file = args.gff
#     upstream = args.upstream
#     downstream = args.downstream
#     
#     # 1. Parse chromosome/region sizes to build a boundary map
#     chrom_sizes = {}
#     with open(gff_file) as fh:
#         for line in fh:
#             if line.startswith("#") or not line.strip():
#                 continue
#             cols = line.rstrip().split("\t")
#             if len(cols) < 9 or cols[2] != "region":
#                 continue
#             chrom = cols[0]
#             start = int(cols[3])
#             end = int(cols[4])
#             chrom_sizes[chrom] = end
# 
#     if fix_genome_size is not None:
#         genome_size = fix_genome_size
#     else:
#         genome_size = sum(chrom_sizes.values())
# 
#     ###########################################################
#     # Calculate Occupied Space (Genes + Flanking Regions)
#     ###########################################################
#     genes = features.loc[features.Feature == "gene"].copy()
#     
#     # Create a list of expanded intervals [Start, End]
#     expanded_intervals = []
#     
#     for _, gene in genes.iterrows():
#         chrom = gene["Chromosome"]  # DataFrame has a chromosome column
#         
#         # Fallback boundary if chromosome isn't explicitly defined in the region lines
#         # max_limit = chrom_sizes.get(chrom, float('inf')) 
#         
#         # Handle strandedness if your data uses it, otherwise treat uniformly
#         if gene["Strand"] == "-":  # if "Strand" in gene and
#             gene_start = gene["Start"] + upstream
#             gene_end = gene["End"] - downstream  # min(max_limit, gene["End"] + upstream)
#         else:
#             gene_start = gene["Start"] - upstream
#             gene_end = gene["End"] + downstream  # min(max_limit, gene["End"] + downstream)
#             
#         expanded_intervals.append((gene_start, gene_end))
# 
#     # Merge all overlapping regions to get exact total base coverage
#     occupied_space = merge_intervals(expanded_intervals)
# 
#     ###########################################################
#     # Intergenic Calculation
#     ###########################################################
#     intergenic_space = genome_size - occupied_space
#     
#     # Prevent negative values due to rounding errors or edge cases
#     intergenic_space = max(0, intergenic_space)
# 
#     feature_space = pd.concat([
#         feature_space,
#         pd.DataFrame({
#             "Feature": ["intergenic"],
#             "NIntervals": [np.nan],
#             "TotalBases": [intergenic_space]
#         })
#     ], ignore_index=True)
# 
#     return feature_space


# def add_intergenic_space(feature_space, features, args, fix_genome_size=None):
#     """
#     Add an intergenic row to feature_space using chromosome lengths
#     stored in GFF 'region' features.
# 
#     Parameters
#     ----------
#     feature_space : DataFrame
#         Output of summarize_feature_space()
# 
#     features : DataFrame
#         Output of parse_gff()
# 
#     args : parsed args from argparse.parser included are args.gff args.upstream args.downstream
# 
#     Returns
#     -------
#     feature_space : DataFrame
#     """
# 
#     gff_file = args.gff
#     upstream = args.upstream
#     downstream = args.downstream
#     genome_size = 0
# 
#     with open(gff_file) as fh:
#         for line in fh:
#             if line.startswith("#"):
#                 continue
#             cols = line.rstrip().split("\t")
#             if len(cols) < 9:
#                 continue
#             if cols[2] != "region":
#                 continue
#             start = int(cols[3])
#             end = int(cols[4])
#             genome_size += end - start + 1
# 
#     ###########################################################
#     # Gene space
#     ###########################################################
# 
#     genes = features.loc[features.Feature == "gene"].copy()
#     genes["Length"] = genes.End - genes.Start + 1
#     gene_space = genes["Length"].sum()
#     updown_space = (upstream * len(genes)) + (downstream * len(genes))  # add upstream and downstream annotations
# 
#     ###########################################################
#     # Intergenic
#     ###########################################################
#     if fix_genome_size is not None:
#         # fix genome size should be int() = to desired genome (or reduced genome) length
#         genome_size = fix_genome_size
# 
#     intergenic_space = genome_size - gene_space - updown_space
#     feature_space = pd.concat([
#         feature_space,
#         pd.DataFrame({
#             "Feature": ["intergenic"],
#             "NIntervals": [np.nan],
#             "TotalBases": [intergenic_space]
#         })
#     ], ignore_index=True)
# 
#     return feature_space


def overlaps_ruranges(bed_df, feat_df):
    # returns DataFrame with peak_idx, feature
    out = []

    for chrom in sorted(set(bed_df.Chromosome) & set(feat_df.Chromosome)):
        q = bed_df[bed_df.Chromosome == chrom]
        t = feat_df[feat_df.Chromosome == chrom]

        if q.empty or t.empty:
            continue

        qi, ti = rr.overlaps(
            q.Start.to_numpy(np.int64),
            q.End.to_numpy(np.int64),
            t.Start.to_numpy(np.int64),
            t.End.to_numpy(np.int64),
        )

        if len(qi):
            tmp = pd.DataFrame({
                "peak_idx":
                    q.index.to_numpy()[qi],
                "Feature":
                    t.Feature.to_numpy()[ti],
                "TranscriptType":
                    t.TranscriptType.to_numpy()[ti],
                "GeneBiotype":
                    t.GeneBiotype.to_numpy()[ti],
                "GeneName":
                    t.GeneName.to_numpy()[ti]
            })
            out.append(tmp)

    if out:
        return pd.concat(out, ignore_index=True)
    return pd.DataFrame(columns=["peak_idx", "feature"])


def overlaps_pyranges(bed_df, feat_df):
    """
    Find overlaps between BED intervals and annotated GFF features.

    Returns
    -------
    DataFrame
        One row per overlap containing:
            peak_idx
            Feature
            TranscriptType
            GeneBiotype
            GeneName
    """

    bed_df = bed_df.copy()
    bed_df["peak_idx"] = bed_df.index

    a = pr.PyRanges(bed_df)
    b = pr.PyRanges(feat_df)
    
    # j = a.join(b, apply_strand_suffix=False)
    # Perform the join and report overlap length in bases
    # This creates a combined PyRanges object with 'Overlap' column
    # convert to dataframe .df
    j_df = a.join(b, report_overlap=True, apply_strand_suffix=False).df

    if j_df.empty:
        return pd.DataFrame(
            columns=[
                "peak_idx",
                "Feature",
                "TranscriptType",
                "GeneBiotype",
                "GeneName"])
    
    # Calculate the exact percentage of 'a' overlapped by 'b'
    # 'End' and 'Start' belong to 'a' because 'b' columns get a suffix (like '_b') by default
    j_df["overlap_fraction_a"] = j_df["Overlap"] / (j_df["End"] - j_df["Start"])

    # Filter for rows where overlap is >= 50% of interval 'a'
    j = j_df[j_df["overlap_fraction_a"] >= 0.90]

    res = (j[[
            "peak_idx",
            "Feature",
            "TranscriptType",
            "GeneBiotype",
            "GeneName"]]
    .drop_duplicates()
    .sort_values(["peak_idx", "Feature"])
    .reset_index(drop=True))
    # .drop_duplicates()
    # .sort_values(["peak_idx", "Feature"])
    # .reset_index(drop=True))
    
    # Fill missing values with strings so sorted() never encounters NoneType
    res["Feature"] = res["Feature"].fillna(".").astype(str)
    res["TranscriptType"] = res["TranscriptType"].fillna(".").astype(str)
    res["GeneBiotype"] = res["GeneBiotype"].fillna(".").astype(str)
    res["GeneName"] = res["GeneName"].fillna(".").astype(str)
    
    return res

    # return (j[[
    #         "peak_idx",
    #         "Feature",
    #         "TranscriptType",
    #         "GeneBiotype",
    #         "GeneName"]]
    # .drop_duplicates()
    # .sort_values(["peak_idx", "Feature"])
    # .reset_index(drop=True)
    # )


def build_feature_lookup(features):
    """
    Build a dictionary:
        GeneName -> feature dataframe
    """
    from collections import defaultdict
    return {
        gene: df.reset_index(drop=True)
        for gene, df in features.groupby("GeneName")
    }

def build_feature_space_lookup(feature_lookup):
    """
    Build a lookup:
        GeneName -> feature_space dataframe
    """

    return {
        gene: summarize_feature_space(df)
        for gene, df in feature_lookup.items()
    }

def annotate_gene_specific(bed_df, feature_lookup):
    assignments = []
    overlap_tables = []
    for idx, peak in bed_df.iterrows():
        gene_features = feature_lookup.get(peak.GeneID)
        if gene_features is None:
            assignments.append({
                "Chromosome": peak.Chromosome,
                "Start": peak.Start,
                "End": peak.End,
                "Name": peak.Name,
                "GeneID": peak.GeneID,
                "Features": "missing_gene",
                "TranscriptTypes": "missing_gene",
                "GeneBiotypes": "missing_gene",
                "Genes": peak.GeneID
            })
            continue

        this_peak = pd.DataFrame([{
            "Chromosome": peak.Chromosome,
            "Start": peak.Start,
            "End": peak.End,
            "Name": peak.Name
        }])

        if HAVE_RURANGES:
            overlaps = overlaps_ruranges(this_peak, gene_features)
        else:
            overlaps = overlaps_pyranges(this_peak, gene_features)

        if overlaps.empty:
            # print(f"overlap: {overlaps}")
            # make dataframe manually b/c no intergenic annotation when grabing gene specific annotation
            overlaps = pd.DataFrame({
                "peak_idx": this_peak.index,
                "Feature": "intergenic",
                "TranscriptType": "intergenic",
                "GeneBiotype": "intergenic",
                "GeneName": peak.GeneID
            })
            overlap_tables.append(overlaps)
            ann = [{
                "Feature": "intergenic",
                "TranscriptType": "intergenic",
                "GeneBiotype": "intergenic",
                "GeneName": peak.GeneID
            }]
        else:
            overlap_tables.append(overlaps)
            ann = []
            for row in overlaps.itertuples(index=False):
                ann.append({
                    "Feature": row.Feature,
                    "TranscriptType": row.TranscriptType,
                    "GeneBiotype": row.GeneBiotype,
                    "GeneName": row.GeneName
                })

        assignments.append({
            "Chromosome": peak.Chromosome,
            "Start": peak.Start,
            "End": peak.End,
            "Name": peak.Name,
            "GeneID": peak.GeneID,
            "Features":
                ",".join(sorted({x["Feature"] for x in ann})),
            "TranscriptTypes":
                ",".join(sorted({x["TranscriptType"] for x in ann})),
            "GeneBiotypes":
                ",".join(sorted({x["GeneBiotype"] for x in ann})),
            "Genes":
                ",".join(sorted({x["GeneName"] for x in ann}))
        })
        # print(f"overlaps: {overlaps}")
        # print(f"gene: {peak.GeneID}\nThis VNTR: {this_peak}\nGene features: {gene_features}\nFeature assignments: {ann}\n")
    if overlap_tables:
        overlap_df = pd.concat(overlap_tables, ignore_index=True)
        print(overlap_df.head())
        print(overlap_df.tail())
    else:
        overlap_df = pd.DataFrame()

    return pd.DataFrame(assignments), overlap_df


def add_missing_intergenic(overlaps, bed_df):

    observed = set(overlaps.peak_idx)

    missing = []

    for idx in bed_df.index:

        if idx not in observed:

            missing.append({

                "peak_idx": idx,
                "Feature": "intergenic",
                "TranscriptType": "intergenic",
                "GeneBiotype": "intergenic",
                "GeneName": "."

            })

    if missing:
        overlaps = pd.concat(
            [overlaps, pd.DataFrame(missing)],
            ignore_index=True
        )

    return overlaps


def annotate_bed(bed_df, features):
    if HAVE_RURANGES:
        overlaps = overlaps_ruranges(bed_df, features)
    else:
        overlaps = overlaps_pyranges(bed_df, features)

    overlaps = add_missing_intergenic(overlaps, bed_df)

    grouped = defaultdict(list)
    for row in overlaps.itertuples(index=False):
        grouped[row.peak_idx].append({
            "Feature": row.Feature,
            "TranscriptType": row.TranscriptType,
            "GeneBiotype": row.GeneBiotype,
            "GeneName": row.GeneName
        })
        # print(f'###DEBUG### {row}')

    assignments = []
    for idx, peak in bed_df.iterrows():
        ann = grouped.get(idx, [])
        ### DEBUG ### print(f'{ann}')
        if not ann:
            ann = [{
                "Feature": "intergenic",
                "TranscriptType": "intergenic",
                "GeneBiotype": "intergenic",
                "GeneName": "."
            }]

        assignments.append({
            "Chromosome": peak.Chromosome,
            "Start": peak.Start,
            "End": peak.End,
            "Name": peak.Name,
            "Features":
                ",".join(sorted(
                    {x["Feature"] for x in ann})),
            "TranscriptTypes":
                ",".join(sorted(
                    {x["TranscriptType"] for x in ann})),
            "GeneBiotypes":
                ",".join(sorted(
                    {x["GeneBiotype"] for x in ann})),
            "Genes":
                ",".join(sorted(
                    {x["GeneName"] for x in ann}))
        })

    return pd.DataFrame(assignments), overlaps


def priority_annotation(assign_df):
    out = []
    for row in assign_df.itertuples(index=False):
        feats = set(row.Features.split(","))
        chosen = "intergenic"
        for p in PRIORITY:
            if p in feats:
                chosen = p
                break

        out.append([
            row.Chromosome,
            row.Start,
            row.End,
            row.Name,
            chosen
        ])

    return pd.DataFrame(
        out,
        columns=["Chromosome", "Start", "End", "Name", "PriorityFeature"]
    )


def normalize_gene_specific(overlaps, feature_space_lookup):
    '''
    Returns normalized counts based upon each genes distribution of features in count/Mb
    '''
    normalized = []
    for gene, gene_overlaps in overlaps.groupby("GeneName"):
        feature_space = feature_space_lookup.get(gene)

        if feature_space is None:
            continue

        counts = (
            gene_overlaps
            .groupby("Feature")
            .size()
            .reset_index(name="Count")
        )
        merged = counts.merge(
            feature_space,
            on="Feature",
            how="left"
        )
        merged["Count_per_Mb"] = (
            merged["Count"] /
            (merged["TotalBases"] / 1e6)
        )
        merged["GeneName"] = gene
        normalized.append(merged)

    if normalized:
        return pd.concat(normalized,
                         ignore_index=True)

    return pd.DataFrame()


def normalize_feature_counts(summary_df, feature_space):
    """
    Normalize observed overlaps by available genomic space.
    """

    norm = summary_df.merge(
        feature_space,
        on="Feature",
        how="left"
    )

    norm["Count_per_interval"] = (
        norm["Count"] /
        norm["NIntervals"]
    )

    norm["Count_per_Mb"] = (
        norm["Count"] /
        (norm["TotalBases"] / 1e6)
    )

    return norm


def feature_summary(overlaps):

    feature_counts = (
        overlaps
        .groupby("Feature")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    feature_biotype_counts = (
        overlaps
        .groupby(["Feature", "GeneBiotype"])
        .size()
        .reset_index(name="Count")
        .sort_values(["Feature", "Count"], ascending=[False, False])
    )
    feature_transcript_counts = (
        overlaps
        .groupby(["Feature", "TranscriptType"])
        .size()
        .reset_index(name="Count")
        .sort_values(["Feature", "Count"], ascending=[False, False])
    )
    return (
        feature_counts,
        feature_biotype_counts,
        feature_transcript_counts
    )


def diagnose_merge(bed_df, genes_df):
    # 1. Perform a left merge to keep all VNTR rows, tracking where the match fails
    diagnostic_merge = pd.merge(
        bed_df, 
        genes_df, 
        left_on="GeneID", 
        right_on="GeneName", 
        how="left", 
        indicator=True
    )

    # 2. Filter for rows that only existed in bed_df (meaning they failed to merge)
    failed_rows = diagnostic_merge[diagnostic_merge["_merge"] == "left_only"]

    # 3. Extract the unique gene names that were excluded
    excluded_genes = failed_rows["GeneID"].unique().tolist()

    # 4. Print the diagnostic summary and the list
    print(f"--- MERGE DIAGNOSTIC REPORT ---")
    print(f"Total rows in bed_df: {len(bed_df)}")
    print(f"Number of rows failed to merge: {len(failed_rows)}")
    print(f"Number of unique excluded gene names ({len(excluded_genes)}):")
    print(excluded_genes)
    print(f"-------------------------------")


def distance_to_gene_from_tsv(bed_df, genes_df, center=False):
    """
    Calculates the distance from a VNTR center to the TSS of a specified gene.
    
    bed_df: Must contain ["Chromosome", "start", "end", "GeneID"]
    genes_df: Must contain ["Chromosome", "Start", "End", "Strand", "GeneName"]
    """

    # 1. Merge VNTRs with their specific genes based on the ID columns
    merged = pd.merge(
        bed_df, 
        genes_df, 
        left_on="GeneID", 
        right_on="GeneName", 
        suffixes=('', '_gene')
    )

    diagnose_merge(bed_df, genes_df)

    # 2. Determine the Transcription Start Site (TSS) based on gene strand
    # If strand is '+', TSS is the Start. If '-', TSS is the End.
    # tss = np.where(merged["Strand"] == "+", merged["Start_gene"], merged["End_gene"])
    # Support Strand format as 1 / -1 (both integer and string versions) or +/-
    is_positive_strand = merged["Strand"].isin([1, "1", "+"])
    tss = np.where(is_positive_strand, merged["Start_gene"], merged["End_gene"])

    if center:
        # 3. Calculate the center of the VNTR
        center = ((merged["Start"] + merged["End"]) / 2).astype(int)
        # 4. Compute the relative distance to the TSS
        merged["DistanceToTSS"] = center - tss
    else:
        # pick the end of the VNTR closest to the TSS
        # closest_end = np.where(merged["Strand"] == "+", merged["Start"], merged["End"])
        # merged["DistanceToTSS"] = closest_end - tss
        # 5. Vectorized logic evaluating boundary overlaps and edge distances
        vntr_start = merged["Start"]
        vntr_end = merged["End"]
        conditions = [
            (vntr_start <= tss) & (tss <= vntr_end),  # 1. TSS overlaps inside VNTR
            (tss > vntr_end),                         # 2. VNTR is completely upstream of TSS
            (tss < vntr_start)                        # 3. VNTR is completely downstream of TSS
        ]
        choices = [
            0,                 # Overlap equals zero
            vntr_end - tss,    # Upstream distance (yields a negative value)
            vntr_start - tss   # Downstream distance (yields a positive value)
        ]
        merged["DistanceToTSS"] = np.select(conditions, choices, default=np.nan)

    # 5. Extract and rename columns to match your original schema requirements
    # Adjust "Name" below if your bed_df used a different column for the VNTR name
    return merged[[
        "Chromosome", "Start", "End", "Name", "GeneID", "GeneName", "DistanceToTSS", "Start_gene", "End_gene", "Strand"
    ]]  # .rename(columns={"GeneID": "Name"})


def distance_to_nearest_gene(bed_df, genes_df, center=False):
    # bed = pr.PyRanges(bed_df.assign(peak_idx=bed_df.index))
    # genes = pr.PyRanges(genes_df)
    # n = bed.nearest(genes).df

    # 1. Create a 1-bp coordinate anchor at the TSS of each gene based on +/- strand orientation
    tss_df = genes_df.copy()
    tss_coords = np.where(tss_df["Strand"] == "+", tss_df["Start"], tss_df["End"])

    tss_df["Start"] = tss_coords
    tss_df["End"] = tss_coords + 1  # Standard 1-bp interval for PyRanges

    # 2. Convert to PyRanges objects
    genes_tss_pr = pr.PyRanges(tss_df)
    bed_pr = pr.PyRanges(bed_df)

    # 3. Find the mathematically nearest TSS
    nearest_tss_df = bed_pr.nearest(genes_tss_pr).df
    n = nearest_tss_df

    # TSS distance
    # tss = np.where(n.Strand == "+", n.Start_b, n.End_b)
    # Support Strand format as 1 / -1 (both integer and string versions) or +/-
    is_positive_strand = n.Strand.isin([1, "1", "+"])
    tss = np.where(is_positive_strand, n.Start_b, n.End_b)

    if center:
        # 3. Calculate the center of the VNTR
        center = ((n.Start + n.End) / 2).astype(int)
        # 4. Compute the relative distance to the TSS
        n["DistanceToTSS"] = center - tss
    else:
        # # pick the end of the VNTR closest to the TSS
        # closest_end = np.where(n.Strand == "+", n.Start, n.End)
        # n["DistanceToTSS"] = closest_end - tss
        # 5. Vectorized logic evaluating boundary overlaps and edge distances
        vntr_start = n.Start
        vntr_end = n.End
        conditions = [
            (vntr_start <= tss) & (tss <= vntr_end),  # 1. TSS overlaps inside VNTR
            (tss > vntr_end),                         # 2. VNTR is completely upstream of TSS
            (tss < vntr_start)                        # 3. VNTR is completely downstream of TSS
        ]
        choices = [
            0,                 # Overlap equals zero
            vntr_end - tss,    # Upstream distance (yields a negative value)
            vntr_start - tss   # Downstream distance (yields a positive value)
        ]
        n["DistanceToTSS"] = np.select(conditions, choices, default=np.nan)

    try:
        n[["GeneID"]]
    except:
        return n[[
        "Chromosome", "Start", "End", "Name", "GeneName",
        "DistanceToTSS", "Start_b", "End_b", "Strand"
        ]]
    else:
        return n[[
            "Chromosome", "Start", "End", "Name", "GeneID", "GeneName",
            "DistanceToTSS", "Start_b", "End_b", "Strand"
        ]]


def make_plots(
        args,
        summary_df,
        norm_counts_df,
        nearest_df,
        priority_summary_df,
        feature_biotype_df,
        feature_transcript_df,
        prefix,
        eGene_distance_df=None,
        tsv_summary_df=None,
        tsv_feature_biotype_df=None,
        tsv_norm_counts_df=None
    ):

    # make histogram of tss distances
    zoomwindow=1000000
    plt.figure(figsize=(3.0,1.5))
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
    plt.hist(
        nearest_df.DistanceToTSS, bins=30, range=(-zoomwindow, zoomwindow), alpha=0.8
    )
    s = nearest_df.DistanceToTSS
    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
        label=f"mean: {s.mean():.0f}\nmedian: {s.median():.0f}"
    )
    plt.xticks(rotation=60, ha="right")
    plt.legend(handlelength=0, handletextpad=0)
    plt.tight_layout()
    plt.savefig(f"{prefix}.TSS_Distances.upstream{args.upstream}.pdf")
    plt.close()

    if eGene_distance_df is not None:
        # make histogram of tss distances
        zoomwindow=1000000
        plt.figure(figsize=(3.0,1.5))
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
        plt.hist(
            eGene_distance_df.DistanceToTSS, bins=30, range=(-zoomwindow, zoomwindow), alpha=0.8
        )
        s = eGene_distance_df.DistanceToTSS
        plt.axvline(
            0,
            linestyle="--",
            linewidth=1,
            label=f"mean: {s.mean():.0f}\nmedian: {s.median():.0f}"
        )
        plt.xticks(rotation=60, ha="right")
        plt.legend(handlelength=0, handletextpad=0)
        plt.tight_layout()
        plt.savefig(f"{prefix}.TSS_Distances_eVNTR_to_eGene.upstream{args.upstream}.pdf")
        plt.close()

    if tsv_summary_df is not None:
        # raw counts
        plt.figure(figsize=(4,3)) # 10,6
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
        plt.bar(
            tsv_summary_df.Feature,
            tsv_summary_df.Count
        )
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        # Use bbox_inches="tight" in savefig to prevent the external legend from being clipped
        plt.savefig(f"{prefix}.feature_counts_eVNTRs.upstream{args.upstream}.pdf")
        plt.close()
    
    if tsv_norm_counts_df is not None:
        # normalized raw counts for only the genes found to be associated with eVNTRs
        plt.figure(figsize=(4,3)) # 10,6
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
        # FILTER: Keep only rows where the '.Count' column is 20 or greater to avoid stochastic results
        filtered_counts_df = tsv_norm_counts_df[tsv_norm_counts_df['Count'] >= 20]
        filtered_counts_df = tsv_norm_counts_df[tsv_norm_counts_df['Count'] >= 20]
        # PLOT: Use the new filtered DataFrame instead of the original one
        plt.bar(
            filtered_counts_df.Feature,
            filtered_counts_df.Count_per_Mb
        )
        # plt.bar(
        #     tsv_norm_counts_df.Feature,
        #     tsv_norm_counts_df.Count_per_Mb
        # )
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        # Use bbox_inches="tight" in savefig to prevent the external legend from being clipped
        plt.savefig(f"{prefix}.feature_counts_eVNTRs.NormalizedPerMb.upstream{args.upstream}.pdf")
        plt.close()

    if tsv_feature_biotype_df is not None:
        pivot = (
            tsv_feature_biotype_df.pivot(
                index="Feature",
                columns="GeneBiotype",
                values="Count"
            ).fillna(0)
        )
        # Sort columns by their total count across all features (highest to lowest)
        # This ensures your most abundant data gets the primary, high-contrast colors
        sorted_columns = pivot.sum().sort_values(ascending=False).index
        pivot = pivot[sorted_columns]
        # Pass cmap="tab20" to handle up to 20 color categories
        ax = pivot.plot(kind="bar", stacked=True, figsize=(4,3), cmap="tab20")  # 12,6
        ax.set_ylabel("Count")
        # Place the legend box outside the plot on the right side
        # loc="upper left" positions the upper-left corner of the legend box 
        # at coordinates (x=1.05, y=1.00) relative to the plot axes bounds.
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", title="Gene Biotype")
        plt.xticks(rotation=60, ha="right")
        #plt.tight_layout()
        # Use bbox_inches="tight" in savefig to prevent the external legend from being clipped
        plt.savefig(f"{prefix}.feature_gene_biotypes_eVNTRs.upstream{args.upstream}.pdf", bbox_inches="tight")   #, bbox_inches="tight"
        plt.close()

    # raw counts
    plt.figure(figsize=(4,3)) # 10,6
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
    plt.bar(
        summary_df.Feature,
        summary_df.Count
    )
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    # Use bbox_inches="tight" in savefig to prevent the external legend from being clipped
    plt.savefig(f"{prefix}.feature_counts.upstream{args.upstream}.pdf")
    plt.close()

    # Normalized raw counts per mb
    plt.figure(figsize=(4,3)) # 10,6
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
    # FILTER: Keep only rows where the '.Count' column is 20 or greater to avoid stochastic results
    filtered_counts_df = norm_counts_df[norm_counts_df['Count'] >= 20]
    # PLOT: Use the new filtered DataFrame instead of the original one
    plt.bar(
        filtered_counts_df.Feature,
        filtered_counts_df.Count_per_Mb
    )
    # plt.bar(
    #     norm_counts_df.Feature,
    #     norm_counts_df.Count_per_Mb
    # )
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    # Use bbox_inches="tight" in savefig to prevent the external legend from being clipped
    plt.savefig(f"{prefix}.feature_counts.NormalizedPerMb.upstream{args.upstream}.pdf")
    plt.close()

    # raw counts priority df
    plt.figure(figsize=(4,3))
    plt.bar(
        priority_summary_df.Feature,
        priority_summary_df.Count
    )
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{prefix}.priority_feature_counts.upstream{args.upstream}.pdf")
    plt.close()

    # percentages priority df
    plt.figure(figsize=(4,3))
    plt.bar(
        priority_summary_df.Feature,
        priority_summary_df.Percentage
    )
    plt.ylabel("Percent")
    plt.xticks(rotation=60, ha="right")
    # plt.tight_layout()
    plt.savefig(f"{prefix}.priority_feature_percentages.upstream{args.upstream}.pdf", bbox_inches="tight")
    plt.close()

    pivot = (
        feature_biotype_df.pivot(
            index="Feature",
            columns="GeneBiotype",
            values="Count"
        ).fillna(0)
    )
    # Sort columns by their total count across all features (highest to lowest)
    # This ensures your most abundant data gets the primary, high-contrast colors
    sorted_columns = pivot.sum().sort_values(ascending=False).index
    pivot = pivot[sorted_columns]
    # Pass cmap="tab20" to handle up to 20 color categories
    ax = pivot.plot(kind="bar", stacked=True, figsize=(4,3), cmap="tab20")  # 12,6
    ax.set_ylabel("Count")
    # Place the legend box outside the plot on the right side
    # loc="upper left" positions the upper-left corner of the legend box 
    # at coordinates (x=1.05, y=1.00) relative to the plot axes bounds.
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", title="Gene Biotype")
    plt.xticks(rotation=60, ha="right")
    #plt.tight_layout()
    # Use bbox_inches="tight" in savefig to prevent the external legend from being clipped
    plt.savefig(f"{prefix}.feature_gene_biotypes.upstream{args.upstream}.pdf", bbox_inches="tight")   #, bbox_inches="tight"
    plt.close()

    pivot = (
        feature_transcript_df.pivot(
            index="Feature",
            columns="TranscriptType",
            values="Count"
        )
        .fillna(0)
    )
    # Sort columns by their total count across all features (highest to lowest)
    # This ensures your most abundant data gets the primary, high-contrast colors
    sorted_columns = pivot.sum().sort_values(ascending=False).index
    pivot = pivot[sorted_columns]
    # Pass cmap="tab20" to handle up to 20 color categories
    pivot.plot(kind="bar", stacked=True, figsize=(4,3), cmap="tab20")
    plt.ylabel("Count")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", title="Transcript Type")
    plt.xticks(rotation=60, ha="right")
    #plt.tight_layout()
    plt.savefig(f"{prefix}.feature_transcript_types.upstream{args.upstream}.pdf", bbox_inches="tight")
    plt.close()


def plot_feature_upset(args, prefix, assign_df):

    '''
    from upsetplot import UpSet, from_indicators
    '''

    # Define the words that mean a row should be skipped
    rows_to_exclude = ["transcript", "missing_gene", "pseudogene"] 

    # Join them with a pipe character '|' to create an "OR" regex pattern
    # Result: 'transcript|missing_gene|pseudogene'
    exclude_pattern = "|".join(rows_to_exclude)

    # Filter the DataFrame
    # We look for rows that do NOT contain the pattern (using the ~ invert operator)
    # na=False ensures we keep rows where Features might be completely blank/NaN for now
    assign_df = assign_df[~assign_df["Features"].str.contains(exclude_pattern, na=False, case=False)]

    rows_to_exclude_exact = "gene"
    assign_df = assign_df[assign_df["Features"] != rows_to_exclude_exact]
    rows_to_exclude_exact = "gene,mRNA"
    assign_df = assign_df[assign_df["Features"] != rows_to_exclude_exact]

    # Get all features
    all_features = sorted(
        {
            feature.strip()
            for feature_list in assign_df["Features"].dropna()
            for feature in feature_list.split(",")
        }
    )
    # Build boolean membership matrix using the original DF index (every row assumed to be eVNTR-eGene pair, if I use column Name (VNTRid) some features get lumped together because same evntr is present twice for two different eGenes)
    membership_df = pd.DataFrame(
        False,
        index=assign_df.index,  # Use index to guarantee uniqueness
        columns=all_features,
    )

    # Use zip with tsv_assign_df.index instead of Name
    for idx, features in zip(
        assign_df.index,
        assign_df["Features"],
    ):
        if pd.isna(features):
            continue
        for feature in map(str.strip, features.split(",")):
            membership_df.loc[idx, feature] = True
    
    # 1. Custom sort/reorder your DataFrame columns manually
    custom_row_order = ["downstream",  "CDS", "exon", "intron", "mRNA", "gene", "upstream", "intergenic"]
    try:
        membership_df = membership_df[custom_row_order]
    except KeyError:
        # likely upstream and downstream were excluded from features because --upstream 0 and --downstream 0
        # or bed coordinates do not overlap a feature so just skip it
        pass
    else:
        # all custom features present, enforce order
        membership_df = membership_df[custom_row_order]

    # # Build boolean membership matrix
    # membership_df = pd.DataFrame(
    #     False,
    #     index=tsv_assign_df["Name"],
    #     columns=all_features,
    # )
    # 
    # for name, features in zip(
    #     tsv_assign_df["Name"],
    #     tsv_assign_df["Features"],
    # ):
    #     for feature in map(str.strip, features.split(",")):
    #         membership_df.loc[name, feature] = True

    print("Total intervals:", len(membership_df))

    # # List the exact row/category names you want to remove
    # rows_to_exclude = ["transcript", "missing_gene", "pseudogene"]  
    # # Drop them from the columns of your membership matrix
    # membership_df = membership_df.drop(columns=rows_to_exclude, errors="ignore")

    upset_data = from_indicators(membership_df)

    # Manually re-sort the underlying multiindex levels of upset_data.
    # This guarantees that the matrix combinations match the index sequence exactly.
    try:
        membership_df = membership_df[custom_row_order]
    except KeyError:
        # likely upstream and downstream were excluded from features because --upstream 0 and --downstream 0
        # or bed coordinates do not overlap a feature so just skip it
        pass
    else:
        # enforce custom order if all features present
        upset_data = upset_data.reorder_levels(custom_row_order).sort_index(ascending=False)

    # print(f"upsetdata {upset_data}")

    plt.figure(figsize=(5, 3))  # 14, 8
    font_size = 12
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
    minrequirement = 0

    upset = UpSet(
        upset_data,
        subset_size="count",
        show_counts=True,
        sort_by="-input",
        sort_categories_by="input",
        min_subset_size=minrequirement,
    )
    #   sort_categories_by="input",
    #   sort_categories_by="-cardinality",

    #   sort_by="cardinality",
    #   sort_categories_by="-cardinality",
    upset.plot()

    plt.suptitle("eVNTR position relative to eGene (Refseq only)")
    plt.savefig(f"{prefix}.Upset.upstream{args.upstream}.pdf")
    plt.close()

# def plot_feature_upset(
#     args,
#     prefix: str,
#     assign_df: pd.DataFrame,
#     feature_col: str = "Features",
#     id_col: str = "Name",
#     min_subset_size: int = 1,
#     figsize: tuple = (12, 8),
#     ):
#     """
#     Create an UpSet plot showing feature overlaps.
# 
#     Parameters
#     ----------
#     tsv_assign_df : pd.DataFrame
#         Input dataframe containing a comma-delimited Features column.
#     feature_col : str
#         Column containing comma-separated feature assignments.
#     id_col : str
#         Unique identifier column (e.g. Name).
#     min_subset_size : int
#         Minimum subset size displayed by UpSet.
#     figsize : tuple
#         Figure size.
# 
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from upsetplot import UpSet
#     """
# 
#     # Build membership table
#     memberships = []
# 
#     for _, row in assign_df[[id_col, feature_col]].dropna().iterrows():
#         features = {
#             feature.strip()
#             for feature in str(row[feature_col]).split(",")
#             if feature.strip()
#         }
# 
#         membership = {feature: True for feature in features}
#         membership[id_col] = row[id_col]
#         memberships.append(membership)
# 
#     print(f"memberships: {memberships}")
# 
#     membership_df = pd.DataFrame(memberships).fillna(False)
#     print(f'membership_df: {membership_df}')
#     # Boolean matrix indexed by Name
#     membership_df = membership_df.set_index(id_col).astype(bool)
#     print(f'membership_df (bool): {membership_df}')
# 
#     # Convert to UpSet-compatible MultiIndex format
#     upset_data = membership_df.groupby(list(membership_df.columns)).size()
#     print(f'upset_data: {upset_data}')
# 
#     # Plot
#     plt.figure(figsize=figsize)
#     upset = UpSet(
#         upset_data,
#         subset_size="count",
#         show_counts=True,
#         min_subset_size=min_subset_size,
#         sort_by="cardinality",
#     )
#     upset.plot()
# 
#     plt.suptitle("Feature Overlap UpSet Plot")
#     plt.tight_layout()
#     plt.savefig(f"{prefix}.Upset.upstream{args.upstream}.pdf")
#     plt.close()
#     # plt.show()
# 
# 
def main():
    '''
    pip install pandas numpy matplotlib pyranges
    #### pip install ruranges (not required)

    USAGE:
    python annotate_bed_hierarchical.py --gff /Users/vmason/Documents/Genomes/Bos_taurus_ARS-UCD2.0/Annotation_Feb2025/GCF_002263795.3_ARS-UCD2.0_genomic.ChromRenamed.gff \
                          --bed /Users/vmason/Documents/Analyses/trgt/v2.1.0/120Hifi94Calves18Ass/inputs/effMotifFinal.all.sort.filter.10.2.TRGT.SimpleComplex.bed
    
    python annotate_bed_hierarchical.py --gff /Users/vmason/Documents/Genomes/Bos_taurus_ARS-UCD2.0/Annotation_Feb2025/GCF_002263795.3_ARS-UCD2.0_genomic.ChromRenamed.gff \
                          --bed /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP/output_Zraw_NoRareAlleles_Minsamples80_ML/gene_vntr_association_results.Autosomes.BestVNTRperGene.LessEqualTo_1e-07.tsv
    
    '''
      
    ap = argparse.ArgumentParser(description="Annotate BED intervals with GFF features")
    ap.add_argument("--gff", type=str, required=True, help="GFF3 annotation")
    ap.add_argument("--bed", type=str, required=True, help="BED file")
    ap.add_argument("--promoter", type=int, default=2000, required=False, help="Functionality disabled. promoter distance upstream of gene")
    ap.add_argument("--upstream", type=int, default=10000, required=False, help="upstream distance upstream of gene")
    ap.add_argument("--downstream", type=int, default=10000, required=False, help="downstream distance downstream of gene")
    # ap.add_argument("--prefix", default="annotation", help="Output prefix")
    args = ap.parse_args()
    base, ext = os.path.splitext(args.bed)

    ##########################################################
    # Read files
    ##########################################################
    print("Loading BED...")
    bed = read_bed(args.bed)

    print("Loading GFF...")
    features = parse_gff(args.gff, promoter=args.promoter, upstream=args.upstream, downstream=args.downstream)

    print(f"Calculating feature space (amount bases covered by each feature)...")
    feature_space = summarize_feature_space(features)
    print(f"Calculating intergenic space...")
    feature_space = add_intergenic_space(feature_space, features, args)

    ##########################################################
    # Annotate
    ##########################################################
    print("Annotating BED...")
    # find all features that the bed entry overlaps (including multiple features per bed entry, multiple genes, multiple trasncrips, etc.)
    assign_df, overlaps = annotate_bed(bed, features)

    ##########################################################
    # Priority annotation
    ##########################################################
    print("Prioritizing Annotations...")
    priority_df = priority_annotation(assign_df)

    ##########################################################
    # Nearest gene
    ##########################################################
    print("Finding nearest genes...")
    nearest_df = distance_to_nearest_gene(bed, features[features.Feature == "gene"], center=False)
    print(f'Median distance between VNTR and the NEAREST gene TSS: {nearest_df.DistanceToTSS.median()}')
    print(f'Number of TSS distance calculations: {len(nearest_df.DistanceToTSS)}')

    ##########################################################
    # eVNTR specific functions
    ##########################################################
    if ext == '.tsv':
        feature_lookup = build_feature_lookup(features)
        # report only the position relative to the gene it regulates
        tsv_assign_df, tsv_overlaps = annotate_gene_specific(bed, feature_lookup)
        print(f"tsv_assign_df shape: {tsv_assign_df.shape}, number of tsv overlaps: {len(tsv_overlaps)}, number of unique gene names in overlaps {tsv_overlaps['GeneName'].unique().shape[0]}")
        # features_filtered = pd.merge(
        #     bed, 
        #     features, 
        #     left_on="GeneID", 
        #     right_on="GeneName", 
        #     suffixes=('', '_gene')
        # )
        # tsv_assign_df, tsv_overlaps = annotate_bed(bed, features_filtered)
        tsv_summary_df, tsv_feature_biotype_df, tsv_feature_transcript_df = feature_summary(tsv_overlaps)
       
        ##########################################################
        # eVNTR specific normalization
        ##########################################################

        #### Normalize counts ###
        print(f'Normlizing per gene counts by feature space...')
        # feature_space_lookup = build_feature_space_lookup(feature_lookup)
        # print(f"feature lookup:\n{feature_lookup}\nfeature space lookup:\n{feature_space_lookup}")
        # tsv_norm_counts_df = normalize_gene_specific(tsv_overlaps, feature_space_lookup)
        # tsv_norm_counts_df = normalize_gene_specific(tsv_overlaps, feature_space_lookup)
        # Merge datasets
        features_filtered = pd.merge(
            bed, 
            features, 
            left_on="GeneID", 
            right_on="GeneName", 
            suffixes=('_VNTR', '')
        )

        # Build non-overlapping cis-window baseline (2 Mb centered on TSS)
        cis_window_half = 1000000 
        true_window_genome_size = 0

        # Deduplicate genes to avoid processing identical windows multiple times
        chrom_col = "Chromosome"
        unique_genes = features_filtered.drop_duplicates(subset=[chrom_col, "Start", "End", "Strand"])

        # Group by chromosome to prevent coordinates from mixing/overlapping across chromosomes
        for chrom, chrom_group in unique_genes.groupby(chrom_col):
            chrom_intervals = []

            for _, row in chrom_group.iterrows():
                # Handle TSS anchoring based on strand orientation
                if row['Strand'] == '+':
                    tss = row['Start']
                elif row['Strand'] == '-':
                    tss = row['End']
                else:
                    tss = row['Start']  # Fallback default if strand is missing or '.'

                w_start = max(1, tss - cis_window_half)
                w_end = tss + cis_window_half
                chrom_intervals.append((w_start, w_end))

            # Merge intervals strictly for THIS chromosome and add to the running total
            if chrom_intervals:
                # merge_intervals returns (total_bases, len_merged); we only need total_bases here
                chrom_bases, _ = merge_intervals(chrom_intervals)
                true_window_genome_size += chrom_bases

        # Process space summaries using the corrected baseline genome size
        tsv_feature_space = summarize_feature_space(features_filtered)
        tsv_feature_space = add_intergenic_space(
            tsv_feature_space, 
            features_filtered, 
            args, 
            fix_genome_size=true_window_genome_size
        )

        tsv_norm_counts_df = normalize_feature_counts(tsv_summary_df, tsv_feature_space)
        ### end normalize counts ###

        print('Finding distance to gene in .tsv')
        eGene_distance_df = distance_to_gene_from_tsv(bed, features[features.Feature == "gene"], center=False)
        print(f'Median eVNTR to eGene Distance: {eGene_distance_df.DistanceToTSS.median()}')
        print(f'length {len(eGene_distance_df.DistanceToTSS)}')
    else:
        tsv_summary_df=None
        tsv_feature_biotype_df=None
        tsv_norm_counts_df=None
        eGene_distance_df=None

    ##########################################################
    # Summaries
    ##########################################################

    print(f"Calculate feature counts...")
    summary_df, feature_biotype_df, feature_transcript_df = feature_summary(overlaps)
    print(f"Normalize counts by feature space...")
    norm_counts_df = normalize_feature_counts(summary_df, feature_space)

    ##########################################################
    # Priority summaries
    ##########################################################

    priority_counts_df = (
        priority_df["PriorityFeature"]
        .value_counts()
        .reindex(PRIORITY, fill_value=0)
        .rename_axis("Feature")
        .reset_index(name="Count")
    )

    priority_percentages_df = (
        priority_df["PriorityFeature"]
        .value_counts(normalize=True)
        .mul(100)
        .reindex(PRIORITY, fill_value=0)
        .rename_axis("Feature")
        .reset_index(name="Percentage")
    )

    priority_summary_df = priority_counts_df.merge(priority_percentages_df, on="Feature")

    #################################
    # output data
    #################################
    assign_df.to_csv(
        f"{base}.feature_assignments.upstream{args.upstream}.tsv", sep="\t", index=False)

    priority_df.to_csv(
        f"{base}.priority_annotations.upstream{args.upstream}.tsv", sep="\t", index=False)

    nearest_df.to_csv(
        f"{base}.nearest_gene.upstream{args.upstream}.tsv", sep="\t", index=False)
    
    if ext == '.tsv':
        eGene_distance_df.to_csv(
            f"{base}.eVNTR_distance_to_eGene.upstream{args.upstream}.tsv", sep="\t", index=False)
        tsv_assign_df.to_csv(
            f"{base}.feature_assignments_eVNTRs.upstream{args.upstream}.tsv", sep="\t", index=False)
        tsv_feature_biotype_df.to_csv(
            f"{base}.feature_gene_biotypes_eVNTRs.upstream{args.upstream}.tsv", sep="\t", index=False)
        tsv_summary_df.to_csv(
            f"{base}.feature_counts_eVNTRs.upstream{args.upstream}.tsv", sep="\t", index=False)
        tsv_norm_counts_df.to_csv(
            f"{base}.feature_counts_eVNTRs.NormalizedPerMb.upstream{args.upstream}.tsv", sep="\t", index=False)
    else:
        eGene_distance_df=None
        tsv_assign_df=None
        tsv_feature_biotype_df=None
        tsv_summary_df=None

    summary_df.to_csv(
        f"{base}.feature_counts.upstream{args.upstream}.tsv", sep="\t", index=False)
    
    norm_counts_df.to_csv(
        f"{base}.feature_counts.NormalizedPerMb.upstream{args.upstream}.tsv", sep="\t", index=False)

    feature_biotype_df.to_csv(
        f"{base}.feature_gene_biotypes.upstream{args.upstream}.tsv", sep="\t", index=False)

    feature_transcript_df.to_csv(
        f"{base}.feature_transcript_types.upstream{args.upstream}.tsv", sep="\t", index=False)

    priority_summary_df.to_csv(
        f"{base}.priority_summar.upstream{args.upstream}.tsv", sep="\t", index=False)

    #################################
    # make plots
    #################################
    print("Making plots...")
    make_plots(
        args,
        summary_df,
        norm_counts_df,
        nearest_df,
        priority_summary_df,
        feature_biotype_df,
        feature_transcript_df,
        base,
        eGene_distance_df=eGene_distance_df,
        tsv_summary_df=tsv_summary_df,
        tsv_feature_biotype_df=tsv_feature_biotype_df,
        tsv_norm_counts_df=tsv_norm_counts_df,
    )

    #################################
    # make UpSet plots with assign dfs
    #################################
    # potential load
    # tsv_assign_df = pd.read_csv(
    #     "annotated_regions.tsv",
    #     sep="\t"
    # )

    plot_feature_upset(args, base+'.feature_counts', assign_df)  # , feature_col="Features", id_col="Name"
    plot_feature_upset(args, base+'.feature_counts_eVNTRs', tsv_assign_df)  # , feature_col="Features", id_col="Name"

    print("Done.")
    print(f"Used ruranges: {HAVE_RURANGES}")
    if HAVE_RURANGES == False:
        print(f"Used pyranges for overlap")


if __name__ == "__main__":
    main()
