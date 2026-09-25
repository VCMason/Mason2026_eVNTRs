
import argparse


def write_out(output, outfile):
    '''
    output is list of lines to be written to output file
    '''
    with open(outfile, 'w') as OUT:
        OUT.write('\n'.join(output))
    print(f"Wrote to output file: {outfile}")
    return


def parse_var_id(var):
    """
    Supported formats:
      - 1_295_INDEL_AT_A
      - 1_295_INDEL_A_ATTTTT
      - 1_295_SNP_T_A
      - 24.12937.1230702  (chr.start.end)

    Returns:
      (chr, start, end, type)
    """

    # Format: chr.start.end
    if '.' in var and '_' not in var:
        chr_, start, end = var.split('.')
        return chr_, start, int(end), None

    # Format: chr_start_type_ref_alt
    v = var.split('_')
    if len(v) != 5:
        raise ValueError(f"Unsupported variant format: {var}")

    chr_, start, var_type, ref, alt = v

    start_int = int(start)

    if len(alt) > len(ref):
        end = start_int + len(alt) - len(ref)
    else:
        end = start_int

    return chr_, start, end, var_type

# def parse_var_id(var):
#     # variant id can be in couple different formats. for now handle 1_295_INDEL_AT_A, 1_295_INDEL_A_ATTTTT, 1_295_SNP_T_A
#     v = var.split('_')
#     chr, start, type, ref, alt = v[0], v[1], v[2], v[3], v[4]
#     end = int(start) + len(alt) - len(ref) if len(alt) > len(ref) else start 
#     return chr, start, end, type


def filter_and_reformat_all_variants(dvargenemap, vars, allvar):
    '''
    variant -> chr star end pid
    gene -> gid strand
    # strand info for gene missing from input just putting +
    out format:
    #chr	start	end	pid	gid	strand
    1	247010	247010	1_247010_SNP_G_A	LOC100138661  -
    '''
    with open(allvar, 'r') as FILE:
        output = []
        log = []
        countselected, csnp, cindel = 0, 0, 0
        for c, line in enumerate(FILE):
            if c == 0:
                header = line.strip()
                output.append(header)
                idx = {head: i for i, head in enumerate(line.strip().split('\t'))}
            else:
                varid = line.strip().split('\t')[idx['pid']]
                try:
                    dvargenemap[varid]
                except KeyError:
                    # not the variant we want
                    pass
                else:
                    countselected += 1. # unique variants
                    vargenepairs = dvargenemap[varid]
                    for v in vargenepairs:
                        temp = f"variant: {varid}, number: {countselected}, Number of genes associated with this variant: {len(vargenepairs)}"
                        print(temp)
                        log.append(temp)
                        chr, start, end, type = parse_var_id(v[0])
                        genotypes = '\t'.join(line.strip().split('\t')[6:])
                        outline = f"{chr}\t{start}\t{end}\t{varid}\t{v[1]}\t+\t{genotypes}"
                        output.append(outline)
                        if type == 'SNP':
                            csnp += 1
                        elif type == 'INDEL':
                            cindel += 1
                        else:
                            pass

    print('Selected variants and reformatted to best variant per gene file format')
    print(f"Number SNP: {csnp}, Number INDEL: {cindel}")
    log.append('Selected variants and reformatted to best variant per gene file format')
    log.append(f"Number SNP: {csnp}, Number INDEL: {cindel}")
    return output, log


def read_sig_var(f):
    '''
    strand info missing right now
    '''
    d = {}
    vars = []
    with open(f, 'r') as FILE:
        for c, line in enumerate(FILE):
            if c == 0:
                idx = {head: i for i, head in enumerate(line.strip().split('\t'))}
            else:
                gene = line.strip().split('\t')[idx['geneid']]
                var = line.strip().split('\t')[idx['VNTRid']]
                geneposition = line.strip().split('\t')[idx['genepos']]
                # d[var] = [var, gene, geneposition]
                d.setdefault(var, []).append([var, gene, geneposition])
                vars.append(var)
    print('made variant / gene map')
    return d, vars


def main():
    '''
    # input
    allvariants=""
    sigvariants=""
    # output
    bestvarfile=""

    # output format
    #chr	start	end	pid	gid	strand	
    variant -> chr star end pid
    gene -> gid strand

    # usage:
    python FilterVariantsToBestVariantPerGene_mymodeling.py \
    --allvarmatrix /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP/inputs/final_filtered.all.MatrixGenotypes.txt \
    # --sigvar /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP_XenaSNPINDELs/output_Zraw_NoRareAlleles_ML/gene_vntr_association_results.Autosomes.BestVNTRperGene.LessEqualTo_1e-07.tsv \
    --sigvar /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP_XenaSNPINDELs/output_Zraw_NoRareAlleles_ML/gene_vntr_association_results.Autosomes.BestVNTRperGene.tsv \
    --bestvarmatrix /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort_LMM_BLUP_XenaSNPINDELs/inputs/BestVariantperGene.MyModel.txt
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--allvarmatrix", type=str, required=True, help="input file. full path to the variant matrix file of all variants to be filtered")
    parser.add_argument("--sigvar", type=str, required=True, help="input file. full path to the best variant per gene so that --allvar can be filtered to create a new best varaint per gene file")
    parser.add_argument("--bestvarmatrix", type=str, required=True, help="output file. full path to the best variant per gene file that is to be created (requires --allvar and --sigvar)")
    # example args.
    # parser.add_argument("--reml", default=False, action='store_true', help="If used the value will be set to True, if True the model will be fit using REML, If False model will be fit using ML (default: False)")
    # parser.add_argument("--permutevntrlengths", default=False, action='store_true', help="If used the value will be set to True, if True the mean vntr allele lengths will be permuted across individuals before being added as a fixed effect covariate to the null and full model (default: False)")
    # parser.add_argument("--threads", type=int, default=4, help="Number of threads to use and = the number of genes analyzed simultaneously for multiprocessing (default: 4) (cis-windows or genome wide analysis only, not used for gene-specific analysis when plotting)")

    args = parser.parse_args()

    dvargenemap, vars = read_sig_var(args.sigvar)
    output, log = filter_and_reformat_all_variants(dvargenemap, vars, args.allvarmatrix)
    write_out(output, args.bestvarmatrix)
    write_out(log, args.bestvarmatrix + '.log')


if __name__ == "__main__":
    #USAGE: conda activate py3.10

    main()
