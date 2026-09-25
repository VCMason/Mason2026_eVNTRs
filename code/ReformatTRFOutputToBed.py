# infile = '/Users/vmason/Documents/Analyses/trf/Bta_UCD2.0/AutoAndSex.fa.trf'  # TRF .dat output
# outfile = '/Users/vmason/Documents/Analyses/trf/Bta_UCD2.0/AutoAndSex.fa.tsv'  # .tsv output, add chrom name to each line

def write_out(outlines, outfile):
    with open(outfile, 'w') as OUT:
        OUT.write('\n'.join(outlines))


def reformat_trf_output_to_tsv(infile):
    outlines = []
    with open(infile, 'r') as FILE:
        for line in FILE:
            if line[0] == '@':
                chrom = line.strip()[1:]
            elif line[0] != '@':
                line.strip().split()[0]
                newline = ' '.join([chrom] + line.strip().split()[0:2] + [line.strip().split()[-4]])  # start and end [line.strip().split()[0:2]], consensus motif line.strip().split()[-4]
                outlines.append(newline)

    return outlines

if __name__ == "__main__":
    import os
    import sys
    import glob
    from pathlib import Path

    if len(sys.argv) != 2:
        # python ReformatTRFOutputToBed.py /path/to/trf/output/dir
        print("Usage: python ReformatTRFOutputToBed.py </path/to/trf/output/dir>")

    trfdir =  sys.argv[1]  # path to directory that should contain file.trf
    # alignfile = sys.argv[2]

    globpattern = os.path.join(trfdir, '*.trf')
    filelist = glob.glob(globpattern)

    for infile in filelist:
        path, filename = os.path.split(infile)
        outfilename = filename[:-len('.trf')] + '.bed'
        outdir = os.path.join(path, 'bed') 
        Path(outdir).mkdir(parents=True, exist_ok=True)  # make directory if it does not exist
        outfile = os.path.join(outdir, outfilename)
        
        outlines = reformat_trf_output_to_tsv(infile)
        write_out(outlines, outfile)