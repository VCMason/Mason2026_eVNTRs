import os
import sys
import gzip
import argparse


def read_tsv(bestvar2gene_path, genotype_matrix_path):
    """
    read in the bestvar2gene space separated file (this file has a header with phe_id, phe_strd, var_id)
    extract values from columns:
        phe_id for gene id --> gid
        phe_strd for gene strand --> strand
        var_id for the variant (associated with gene's id) --pid (even though pid isn't the best column we will store it here)
    replace these vlaue in the currently output genotype_matrix_path file
    """
    dvar = {}
    with open(bestvar2gene_path, 'r') as FILE:
        header = FILE.readline().strip().split()
        required = ['var_id', 'phe_id', 'phe_strd', 'var_chr', 'var_from', 'var_to']
        required2 = ['variant_id', 'phenotype_id', 'phe_strand', 'var_chrom', 'var_start', 'var_end']
        missing = [c for c in required if c not in header]
        if missing:
            missing2 = [c for c in required2 if c not in header]
            if missing2:
                raise ValueError(f"bestvar2gene file is missing required columns: {', '.join(missing)}\nOR\n{', '.join(missing2)}.\nFound: {header}")
            else:
                print(f"Found required columns in bestvar2gene file: {', '.join(required2)}")
                # remap required to required2 for easier downstream processing
                required = required2
        else:
            print(f"Found required columns in bestvar2gene file: {', '.join(required)}")
        header_map = {}
        for count, col in enumerate(header):
            if col == required[0]:  # 'var_id'
                header_map[required[0]] = count
            elif col == required[1]:  # 'phe_id'
                header_map[required[1]] = count
            elif col == required[2]:  # 'phe_strd'
                header_map[required[2]] = count
            elif col == required[3]:  # 'var_chr'
                header_map[required[3]] = count
            elif col == required[4]:  # 'var_from'
                header_map[required[4]] = count
            elif col == required[5]:  # 'var_to'
                header_map[required[5]] = count
        
        for line in FILE:
            if line.strip() == '':
                continue
            parts = line.strip().split()
            pid = parts[header_map[required[0]]]  # 'var_id'
            gid = parts[header_map[required[1]]]  # 'phe_id'
            strand = parts[header_map[required[2]]]  # 'phe_strd'

            # var_chr.var_from.var_to, this is the format of the pid column in the genotype matrix output file, so we can use it to match and replace with the bestvar2gene info.
            # Note that var_from and var_to are 1-based coordinates, which matches the pid column in the genotype matrix output file. If they were 0-based, we would need to add 1 to var_from to match the pid format.
            var_loc = f"{parts[header_map[required[3]]]}.{parts[header_map[required[4]]]}.{parts[header_map[required[5]]]}"
            
            # there can be multiple genes associated with one variant, so store as list
            # dvar.get(...).append(...) returns None, so use setdefault to get-or-create list and then append
            dvar.setdefault(var_loc, []).append([pid, gid, strand])

    outlines = []
    with open(genotype_matrix_path, 'r') as FILE:
        header = FILE.readline().strip().split('\t')
        header_map = {}
        for count, col in enumerate(header):
            if col == 'pid':
                header_map['pid'] = count
            elif col == 'gid':
                header_map['gid'] = count
            elif col == 'strand':
                header_map['strand'] = count
        
        with open(genotype_matrix_path.replace('.txt', '.modified.txt'), 'w') as OUTFILE:
            OUTFILE.write('\t'.join(header) + '\n')

        with open(genotype_matrix_path.replace('.txt', '.modified.txt'), 'a') as OUTFILE:
            for line in FILE:
                parts = line.strip().split('\t')
                var_loc_current = f"{parts[0]}.{parts[1]}.{parts[2]}"

                if var_loc_current in dvar:
                    # replace pid, gid, strand with bestvar2gene values
                    if len(dvar[var_loc_current]) > 1:
                        # then multiple genes associated with one variant, add all gene ids but separate by comman in gid column
                        for result in dvar[var_loc_current]:
                            # making the output long format, one line per gene, meaning variant ids could be repeated. (unique gene id though)
                            parts[header_map['pid']] = result[0]  # var_id
                            parts[header_map['gid']] = result[1]  # gene id
                            parts[header_map['strand']] = result[2]  # gene strand
                            # collect outlines
                            outlines.append('\t'.join(parts) + '\n')
                        # genes = [result[1] for result in dvar[var_loc_current]]  # list of genes associated to variant
                        # strands = [result[2] for result in dvar[var_loc_current]]  # list of gene strands
                    else:
                        parts[header_map['pid']] = dvar[var_loc_current][0][0]
                        parts[header_map['gid']] = dvar[var_loc_current][0][1]
                        parts[header_map['strand']] = dvar[var_loc_current][0][2]
                        # collect outlines
                        outlines.append('\t'.join(parts) + '\n')

                # Write chunks inside the loop to save RAM
                if len(outlines) >= 1000:
                    print(f"Processed {len(outlines)} lines...")
                    OUTFILE.write(''.join(outlines))
                    outlines = []  # reset the list for the next chunk

            # OUTSIDE THE LOOP: Write any leftover lines (e.g., last 450 lines that were not written out as chunks of 1000)
            if outlines:
                OUTFILE.write(''.join(outlines))

    return


def open_vcf_auto(vcf_path):
    """
    Open either a .vcf or .vcf.gz file automatically.
    Returns a file handle in text mode.
    """
    if not os.path.exists(vcf_path):
        sys.exit(f"Error: file not found → {vcf_path}")

    if vcf_path.endswith(".gz"):
        try:
            return gzip.open(vcf_path, "rt")
        except Exception as e:
            sys.exit(f"Error: failed to open compressed VCF {vcf_path}: {e}")
    else:
        try:
            return open(vcf_path, "r")
        except Exception as e:
            sys.exit(f"Error: failed to open VCF {vcf_path}: {e}")


def extract_header(vcf_handle):
    """
    Extract sample IDs (individuals) from the VCF header.
    Returns: list of individual IDs and the file handle positioned after the header.
    """
    samples = []
    for line in vcf_handle:
        if line.startswith("##"):
            continue
        elif line.startswith("#CHROM"):
            parts = line.strip().split("\t")
            samples = parts[9:]  # individual IDs start from column 10
            break
    return samples


def reformat_vcf(input_path, output_path, use_id=None):
    """Read a VCF and write reformatted output."""
    with open_vcf_auto(input_path) as vcf_in, open(output_path, "w") as out:
        # Extract sample IDs
        samples = extract_header(vcf_in)

        # Write header for new format
        out.write("#chr\tstart\tend\tpid\tgid\tstrand\t" + "\t".join(samples) + "\n")

        # Process each variant line
        vcount = 0
        outlines = []
        for line in vcf_in:
            if line.startswith("#"):
                continue

            cols = line.strip().split("\t")
            chrom, pos, pid = cols[0], cols[1], cols[2]
            if use_id is not None:
                metadata = cols[7].split(';')
                for m in metadata:
                    if m.startswith(f"TRID=") or m.startswith(f"ID="):
                        pid = m.split('=')[1]
                        break
                for m in metadata:
                    if m.startswith(f"END="):
                        end = m.split('=')[1]
                        break
                gid = pid
                start = pos
            else:
                start = end = pos
                gid = f"{chrom}.{pos}.{pos}"
            genotypes = [x.split(":")[0] for x in cols[9:]]  # extract only GT from FORMAT fields
            strand = "+"

            outlines.append("\t".join([chrom, start, end, pid, gid, strand] + genotypes) + "\n")
            vcount += 1

            if len(outlines) % 100000 == 0:
                out.write(''.join(outlines))
                outlines = []
                print(f"Processed {vcount} variants...")

        if outlines:
            # write remaining lines
            out.write(''.join(outlines))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reformat a .vcf.gz file into a tab-delimited matrix-like file.")
    parser.add_argument("-i", "--input", required=True, help="Input VCF file (compressed with gzip, e.g., input.vcf.gz).")
    parser.add_argument("-bv2g", "--bestvar2gene", default=None, help="Input .tsv file. File should contain two columns phe_id for gene id, phe_strd for gene strand, and var_id for the variant (associated with gene's id) ")
    parser.add_argument("-o", "--output", required=True, help="Output tab-delimited text file.")
    parser.add_argument("-use_id", "--use_id", action='store_true', default=None, help="Use end position instead of start position in output file for pid.")
    return parser.parse_args()


def main():
    """
    # Usage:
    python VCFToGenotypeMatrix.py \
      -i my_variants.vcf.gz \
      -o my_variants_matrix.txt

    # It will also work with an uncompressed file:
    python VCFToGenotypeMatrix.py \
      -i my_variants.vcf \
      -o my_variants_matrix.txt

    python VCFToGenotypeMatrix.py \
        -i /cluster/work/pausch/vmason/analyses/modeling/linear/Xena_eQTL_Results/bash_workflow_nominal_pvalues/eqtl_nominals.BestVariantPerGene.vcf.gz \
        -o /cluster/work/pausch/vmason/analyses/modeling/linear/Xena_eQTL_Results/bash_workflow_nominal_pvalues/BestVariant.MatrixGenotypes.txt \
        -bv2g /cluster/work/pausch/vmason/analyses/modeling/linear/Xena_eQTL_Results/bash_workflow_nominal_pvalues/eqtl_nominals.BestVariantPerGene.txt
    
    Reformat a .vcf.gz file into a tab-delimited matrix format suitable for downstream analysis.

    Example input (VCF):
        #CHROM  POS  ID  REF  ALT  QUAL  FILTER  INFO  FORMAT  indiv1  indiv2
        1  664472  1_664472_SNP_C_A  C  A  67  .  AF=0.741667;AQ=67;AC=173;AN=234  GT:DP:AD:GQ:PL:RNC  0/1:18:6,12:40:46,0,41:..  0/1:21:6,15:31:41,0,31:..

    Example output:
        #chr  start  end  pid  gid  strand  indiv1  indiv2
        1  664472  664472  1.664472.664472  1.664472.664472  +  0/1  0/1

    Example output with bestvar2gene:
    phenotype_id phe_chrom phe_start phe_end phe_strand num_variants_tested distance_to_variant variant_id var_chrom var_start var_end raw_pvalue slope is_top_variant
    A1BG 18 65645263 65645263 - 11931 760708 18_64884555_SNP_C_T 18 64884555 64884555 0.000720788 0.0950388 -2.72865 0.785169 1
    A2M 5 100819129 100819129 + 16679 138794 5_100957923_INDEL_AC_A 5 100957923 100957924 9.9016e-08 0.219601 0.365482 0.0642478 1
    A2ML1 5 101090736 101090736 - 16234 33746 5_101056990_SNP_A_G 5 101056990 101056990 1.30613e-14 0.404538 -0.673044 0.0761451 1
    ...
    
    """

    args = parse_args()

    reformat_vcf(args.input, args.output, use_id=args.use_id)
    print('Finished reformatting vcf to genotype matrix output format')

    if args.bestvar2gene:
        print('Replacing pid, gid, and strand based on Best variant ID for pid and the gene(s) it is associated with as gid')
        read_tsv(args.bestvar2gene, args.output)


if __name__ == "__main__":

    
    main()
