import argparse
import gzip
import sys
import os


############################################
# File handling
############################################

def open_input_file(path):
    """Open gzipped or plain text file appropriately."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def read_header(fp):
    """Read header line and return list of columns."""
    header_line = fp.readline().rstrip("\n")
    return header_line.split("\t")


def find_line(fp, search_term):
    """Return the first line containing the search term, or None."""
    for line in fp:
        line = line.rstrip("\n")
        if search_term in line:
            return line
    return None


def write_out(path, content):
    with open(path, "w") as f:
        f.write(content)


############################################
# Formatting functions
############################################

def format_motifs(value):
    """Split motifs by comma, remove MOTIFS= prefix, and number them."""
    motifs = value.split(",")

    # Clean prefix if present
    motifs[0] = motifs[0].lstrip("MOTIFS=")

    output = ["Motifs:"]
    for i, motif in enumerate(motifs):
        output.append(f"  {i}" + " " * (5-len(str(i))) + f"{motif}")
    return "\n".join(output)


def format_alleles(alleles_value, allele_ids_value):
    """Pair Alleles with AlleleIDs and format."""
    alleles = alleles_value.split(",")
    allele_ids = allele_ids_value.split(",")

    output = ["Alleles:"]
    for aid, allele in zip(allele_ids, alleles):
        output.append(f"  {aid}" + " " * (5-len(str(aid))) + f"{allele}")
    return "\n".join(output)


def format_allele_based_field(field_name, field_value, allele_ids):
    """
    Generic helper for:
        CountsPerAllele
        CpGDensityPerAllele
        AvgAlleleMethylation
        AvgAllelePurity
        Allelelength
        NumMotifsUsedPerAllele
        AlleleMotifCounts
        AlleleMotifSpans
    """
    arr = field_value.split(",")
    output = [f"{field_name}:"]
    for aid, val in zip(allele_ids, arr):
        output.append(f"  {aid}\t{val}")
    return "\n".join(output)


def compute_allele_frequencies(counts, allele_ids):
    """Compute allele frequencies from CountsPerAllele."""
    counts = [int(x) for x in counts.split(",")]
    total = sum(counts)

    output = ["AlleleFrequencies (%):"]
    for aid, c in zip(allele_ids, counts):
        freq = (c / total) * 100 if total > 0 else 0
        output.append(f"  {aid}" + " " * (5-len(str(aid))) + f"{freq:.4f}")
    return "\n".join(output)


def format_field(key, value):
    """Default formatting for fields."""
    return f"{key}:\n    {value}"


############################################
# Core processing
############################################

def process_line(header, line):
    """Format the entire VNTR record according to all rules."""
    cols = line.split("\t")
    colmap = dict(zip(header, cols))

    allele_ids = colmap.get("AlleleIDs", "").split(",")

    # Keys that need allele-level expansion
    allele_expansion_keys = [
        "CountsPerAllele",
        "CpGDensityPerAllele",
        "AvgAlleleMethylation",
        "AvgAllelePurity",
        "Allelelength",
        "NumMotifsUsedPerAllele",
        "AlleleMotifCounts",
        "AlleleMotifSpans",
    ]

    output_lines = [
        "====================",
        "Formatted VNTR Record",
        "====================",
        ""
    ]

    for key in header:
        value = colmap.get(key, "")

        # ---------- Special handlers ----------
        if key == "Motifs":
            output_lines.append(format_motifs(value))
            output_lines.append("")
            continue

        if key == "Alleles":
            output_lines.append(format_alleles(value, colmap["AlleleIDs"]))
            output_lines.append("")
            continue

        # Allele-level expansions
        if key in allele_expansion_keys:
            output_lines.append(format_allele_based_field(key, value, allele_ids))
            output_lines.append("")
            continue

        # ---------- Default handler ----------
        output_lines.append(format_field(key, value))
        output_lines.append("")

    # Add allele frequencies
    if "CountsPerAllele" in colmap:
        output_lines.append(
            compute_allele_frequencies(colmap["CountsPerAllele"], allele_ids)
        )
        output_lines.append("")

    return "\n".join(output_lines)


############################################
# Main
############################################
import argparse
import os
import sys

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Format VNTR TSV record.")
    parser.add_argument("--infile", default="Autosomes_TRDB_LocusStats.txt.gz", help="TSV or TSV.GZ file")
    parser.add_argument("--search", required=True, nargs="+", help="Strings to search for in file")
    # Clarified help message to reflect relative creation
    parser.add_argument("--outdir", default="extracted_vntrs", help="Subfolder name to create relative to the infile path")

    args = parser.parse_args()

    # Convert search list to a set for O(1) lookups
    search_terms = set(args.search)
    found_terms = set()

    # 1. Extract the directory path where the infile lives
    infile_dir = os.path.dirname(os.path.abspath(args.infile))
    
    # 2. Build the absolute path for outdir relative to the infile directory
    target_outdir = os.path.join(infile_dir, args.outdir)

    # 3. Automatically create the relative subfolder safely
    try:
        os.makedirs(target_outdir, exist_ok=True)
    except Exception as e:
        sys.exit(f"Error: Could not create directory {target_outdir}. Reason: {e}")

    # Read file sequentially (Single Pass Optimization)
    try:
        with open_input_file(args.infile) as fp:
            header = read_header(fp)
            
            # Read line by line to find matches across the stream
            for line in fp:
                matched_term = None
                for term in search_terms:
                    if term in line:
                        matched_term = term
                        break
                
                if matched_term:
                    found_terms.add(matched_term)
                    
                    # Process formatted output for this match
                    formatted = process_line(header, line)
                    print(f"\nFound match for: {matched_term}")
                    print(formatted)

                    # 4. Save files directly inside the new subfolder
                    outfile = os.path.join(target_outdir, f"VNTR_Locus.{matched_term}.txt")
                    write_out(outfile, formatted)
                    print(f"Output written to: {outfile}")

    except FileNotFoundError:
        sys.exit(f"Error: cannot open {args.infile}")

    # Report any terms that were missing from the file
    missing_terms = search_terms - found_terms
    if missing_terms:
        print(f"\nWarning: No matching lines found for: {', '.join(missing_terms)}")

    if not found_terms:
        sys.exit("No matching lines found at all.")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    # USAGE:
    # python ExtractVNTRInformation.py --infile /path/to/file.txt.gz --search term1 term2 term3
    main()

# def main():
#     parser = argparse.ArgumentParser(description="Format VNTR TSV record.")
#     parser.add_argument("--infile", default="Autosomes_TRDB_LocusStats.txt.gz", help="TSV or TSV.GZ file")
#     parser.add_argument("--search", required=True, nargs="+", help="String to search for in file")
# 
#     args = parser.parse_args()
# 
#     # Read file
#     try:
#         with open_input_file(args.infile) as fp:
#             header = read_header(fp)
#             line = find_line(fp, args.search)
#     except FileNotFoundError:
#         sys.exit(f"Error: cannot open {args.infile}")
# 
#     if line is None:
#         sys.exit("No matching line found.")
# 
#     # Process formatted output
#     formatted = process_line(header, line)
#     print(formatted)
# 
#     # Write to file
#     path, _ = os.path.split(args.infile)
#     outfile = os.path.join(path, f"VNTR_Locus.{args.search}.txt")
#     write_out(outfile, formatted)
#     print(f"\nOutput written to: {outfile}")
# 
# 
# if __name__ == "__main__":
#     # USAGE:
#     # python ExtractVNTRInformation.py --infile /path/to/Autosomes_TRDB_LocusStats.txt.gz --search 18.57579391.57579424
#     
#     main()


