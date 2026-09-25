
import os
import argparse
from Bio.Seq import Seq


def write_out(outlines, outpath):
    print('Writing to file: %s' % outpath)
    with open(outpath, 'w') as OUT:
        OUT.write('\n'.join(outlines))


def translate_dna_to_aa(args, sequences):
    '''
    from Bio.Seq import Seq

    sequences is a dict of dna sequcnes as value and names as key
    sequences, frame=1

    return dict of amino acid sequences as value and names as key
    '''

    AAseqs = {}
    for name, dnaseq in sequences.items():
        seq = Seq(dnaseq)
        if args.revcomp:
            dnaseq = seq.reverse_complement()  # revcompdnaseq
        else:
            dnaseq = seq
        AAseq = dnaseq[args.frame-1:].translate(table="Standard", )
        AAseqs[name] = AAseq

    return AAseqs


def read_fasta_as_dict(f):
    '''
    Reads a FASTA file and returns a dictionary of sequences and a list of names.
    # dnasequences is dict name as key and sequence as value. names is list of names in order of appearance in fasta file
    
    '''
    d = {}  # fasta names are key, values are sequences as string
    namelist = []
    with open(f, 'r') as FILE:
        for line in FILE:
            if line[0] == '>':
                name = line.strip()[1:]
                namelist.append(name)
                d[name] = []
            elif line.strip() != '':  # else: # trying to prevent some crap happening on the last line
                d[name].append(line.strip())
    for name in namelist:
        d[name] = ''.join(d[name])  # join list of partial sequences. Useful if interleaved fasta

    return d, namelist


def read_lines_from_infile(infile):
    seqs = []
    with open(infile, 'r') as FILE:
        for line in FILE:
            if line.strip() != '':
                seqs.append(line.strip())
    return seqs


def main():
    '''
    Main function to demonstrate the translation (or revcomp then translation) of DNA sequences.
    
    python TranslateDNAToAA.py --infile /path/to/dna/sequence/file.fa --frame 1 --revcomp --outfile outname.txt

    NOTE: The input file should be in FASTA format, with each sequence preceded by a header line starting with '>'.
    NOTE: The frame argument specifies which base to start translation from on the DNA sequence, if 1 starts at first base (base 0 for python...), if 2 starts at second base (base 1), if 3 starts at third base (base 2). Default is 1.
    '''

    parser = argparse.ArgumentParser(description="translate DNA sequences to amino acid sequences. Or reverese complement prior to translation")
    # parser.add_argument("--ids", nargs='*', help="List of LocusIDs to plot (optional). If empty, plots all.")
    parser.add_argument("--infile", help="a fasta file containing DNA sequences, >seqname\ndnasequence")
    parser.add_argument("--frame", type=int, default=1, help="Reading frame for translation (1, 2, or 3). Default is 1.")
    parser.add_argument("--outfile", default="AAseqs.txt", help="Output file to save the amino acid sequences, one per line.")
    parser.add_argument("--revcomp", action='store_true', default=None, help="Output file to save the amino acid sequences, one per line.")
    args = parser.parse_args()

    if args.frame not in (1, 2, 3):
        raise ValueError("frame must be 1, 2, or 3")

    path, infilename = os.path.split(args.infile)

    # dnasequences = read_lines_from_infile(args.infile)
    dnasequences, names = read_fasta_as_dict(args.infile)  # dnasequences is dict name as key and sequence as value. names is list of names in order of appearance in fasta file

    AAseqs = translate_dna_to_aa(args, dnasequences)

    outlines = [f">{name}\n{AAseqs[name]}" for name in names]  # maintain order of names from fasta input file
    outname = f"{args.outfile}.frame{args.frame}.fa" if not args.revcomp else f"{args.outfile}.revcomp.frame{args.frame}.fa"
    write_out(outlines, os.path.join(path, outname))

    
if __name__ == "__main__":
    main()