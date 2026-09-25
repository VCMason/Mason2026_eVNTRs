import os
import sys
import gzip
import glob
import statistics as st
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list, dendrogram


@dataclass
class locus:  # a variant from TRGT output file .vcf.gz
    id: str
    chrom: str
    start: int
    end: int
    # alleles: List[str] = field(default_factory=list)   # list of strings [ref, allele 1, (maybe) allele2] # call relevent seq by genotype number 0, 1, or 2... etc
    # motifs: List[str] = field(default_factory=list)  # list of strings, [motifs, used, to, genotype, locus]
    cpgdensities: List[str] = field(default_factory=list)  # list of strings (but actually float values), ['CG' density in sequence] ordered from allele 0,1,2,3,...)
    avgallelemeth: List[str] = field(default_factory=list)  # list of strings (but actually float values), average methylation per allele ordered from allele 0,1,2,3,...)
    avgallelepur: List[str] = field(default_factory=list)   # list of strings (but actually float values), average allele purity per allele (ordered from allele 0,1,2,3,...)
    allelemc: List[str] = field(default_factory=list)  # list of strings, each string is underscore delimited motif counts per allele (ordered from allele 0,1,2,3,...)
    allelems: List[str] = field(default_factory=list)  # list of string, each string is a motif span, motif span is motif#(allelestart-alleleend)_motif#(start-end)... there is one span per allele each span is comma delimited # ex: ['0(0-51)_1(57-84)','0(0-72)_1(78-105)']
    samplegenotypes: Dict[str, "variant"] = field(default_factory=dict)  # dictionary, key is sampleID, value is dataclass variant (for that sample)


@dataclass
class variant:
    # always specified
    genotype: str  # 0/0, 0/1, 1/1, 1/2 for diploid; 0, 1, 2 for haploid
    # ploidy: str  # 'haploid' or 'diploid'
    al1: str                   # 0, 1, 2, etc.
    al1len: int                   # AL, 84
    al1depth: int                   # SD, 28
    al1pure: str                   # AP, 0.5  # has to be string cuz sometimes .,.
    al1meth: str                   # AM, 0.4  # has to be string cuz sometimes it's .,. so just .
    al1lenrange: List[int] = field(default_factory=list)  # ALLR, [80, 85]
    al1motifcount: List[int] = field(default_factory=list)  # MC, [17, 9]
    al1motifspan: List[str] = field(default_factory=list)  # MS, [0(0-51), 1(57-84)]

    # Optional entries (for diploid)
    al2: Optional[str] = None  # Only for diploid # 0, 1, 2, etc.
    al2len: Optional[int] = None  # AL, 105 # Only for diploid
    al2depth: Optional[int] = None  # SD, 18  # Only for diploid
    al2pure: Optional[str] = None  # AP, 0.9  # Only for diploid
    al2meth: Optional[str] = None  # AM, 0.5  # Only for diploid

    al2lenrange: Optional[List[int]] = None               # ALLR, [102, 114]  # Only for diploid
    al2motifcount: Optional[List[int]] = None               # MC, [24, 9] # Only for diploid
    al2motifspan: Optional[List[str]] = None               # MS, [0(0-72), 1(78-105)] # Only for diploid

    # Ploidy classification (computed automatically)
    ploidy: str = field(init=False)  # 'haploid' or 'diploid' # Computed in __post_init__

    def __post_init__(self):
        # Determine ploidy based on genotype format
        if "/" in self.genotype:  
            self.ploidy = "diploid"
        else:
            self.ploidy = "haploid"
            self.al2 = None
            self.al2len = None
            self.al2lenrange = None
            self.al2depth = None
            self.al2motifcount = None
            self.al2motifspan = None
            self.al2pure = None
            self.al2meth = None

    def allele_length_diff(self) -> int:
        if self.ploidy == "diploid":
            return abs(self.al1len - self.al2len)
        else:
            return 0


def append_out(outlines, outfile, gate=0, gz=None):
    if gz:
        import gzip
        outfile = outfile if outfile[-3:] == '.gz' else f'{outfile}.gz'
        if gate == 1:
            with gzip.open(outfile, 'at') as OUT:
                OUT.write('\n' + '\n'.join(outlines))
        else:    
            with gzip.open(outfile, 'at') as OUT:
                OUT.write('\n'.join(outlines))
    else:
        if gate == 1:
            with open(outfile, 'a') as OUT:
                OUT.write('\n' + '\n'.join(outlines))
        else:
            with open(outfile, 'a') as OUT:
                OUT.write('\n'.join(outlines))
    print(f'Output file written to: {outfile}')


def CpG_density(seq):
    
    cgcount = seq.upper().count('CG')
    seqlen = len(seq)
    cgdensity = cgcount / seqlen

    return cgdensity


def make_matrix_locus_x_individual(dvariants, ids, sampleIDs, outputdir, chrom, chromlist, dfeatures=None, featureheader=None):   # , **kwargs
    # dvariants is dictionary, key is locusID, value is locus dataclass defined above
    # dvariants[locusID] = locus

    locuslengths, locusallelelengths, locusallelemeth, locusalleledepth, locusallelepurity, locusnumreps, locusalllendiff, locusallgenotypes = [], [], [], [], [], [], [], []
    
    for count, locusID in enumerate(ids):
        # numeffmotifs = len(locus.motifs)
        # numalleles = len(locus.alleles)
        loc = dvariants[locusID]  # locus dataclass

        countambiguous = 0
        outmeanreps, outallelelength, outallelemeth, outalleledepth, outallelepurity, outmeanlength, outalllendiff, outallgenotypes = [], [], [], [], [], [], [], []

        for sampleID in sampleIDs:
            var = loc.samplegenotypes[sampleID]  # variant dataclass per sample for the locus
            if var == '.':
                countambiguous += 1
                outallgenotypes.append('.')
                outmeanlength.append('.')
                outallelelength.append('.')
                outmeanreps.append('.')
                outallelemeth.append('.')
                outalleledepth.append('.')
                outallelepurity.append('.')
                outalllendiff.append('.')
            else:
                if var.ploidy == 'diploid':
                    al1numreps = sum(var.al1motifcount)  # number of motifs in al1
                    al2numreps = sum(var.al2motifcount)  # number of motifs in al2
                    meannumreps = st.mean([al1numreps, al2numreps])  # number of total motifs in both alleles, i.e. in the individual
                    meanlength = st.mean([var.al1len, var.al2len])
                    allelelengths = f'{var.al1len}/{var.al2len}'
                    allelemeth = f'{var.al1meth}/{var.al2meth}'
                    alleledepth = f'{var.al1depth}/{var.al2depth}'
                    allelepurity = f'{var.al1pure}/{var.al2pure}'
                    alllendiff = var.allele_length_diff()
                elif var.ploidy == 'haploid':
                    al1numreps = sum(var.al1motifcount)
                    meannumreps = al1numreps
                    meanlength = var.al1len
                    allelelengths = f'{var.al1len}'
                    allelemeth = f'{var.al1meth}'
                    alleledepth = f'{var.al1depth}'
                    allelepurity = f'{var.al1pure}'
                    alllendiff = 0
                outmeanlength.append(str(meanlength))  # append one entry per sample  # resets to [] at every locus
                outallelelength.append(allelelengths)
                outmeanreps.append(str(meannumreps))  # append one entry per sample  # resets to [] at every locus
                outallelemeth.append(allelemeth)
                outalleledepth.append(alleledepth)
                outallelepurity.append(allelepurity)
                outalllendiff.append(str(alllendiff))   # append one entry per sample   # resets to [] at every locus
                outallgenotypes.append(var.genotype)               
        locuslengths.append([loc.chrom, str(loc.start), str(loc.end), loc.id, loc.id, '+'] + outmeanlength)
        locusallelelengths.append([loc.chrom, str(loc.start), str(loc.end), loc.id, loc.id, '+'] + outallelelength)
        locusallelemeth.append([loc.chrom, str(loc.start), str(loc.end), loc.id, loc.id, '+'] + outallelemeth)
        locusalleledepth.append([loc.chrom, str(loc.start), str(loc.end), loc.id, loc.id, '+'] + outalleledepth)
        locusallelepurity.append([loc.chrom, str(loc.start), str(loc.end), loc.id, loc.id, '+'] + outallelepurity)
        locusnumreps.append([loc.chrom, str(loc.start), str(loc.end), loc.id, loc.id, '+'] + outmeanreps)
        locusalllendiff.append([loc.chrom, str(loc.start), str(loc.end), loc.id, loc.id, '+'] + outalllendiff)
        locusallgenotypes.append([loc.chrom, str(loc.start), str(loc.end), loc.id, loc.id, '+'] + outallgenotypes)
        if count % 10000 == 0:
            print(f'Matrix info on chromosome: {chrom}, collected up to line: {count}')

    header = [['#chr', 'start', 'end', 'pid', 'gid', 'strand'] + sampleIDs]

    outfile = os.path.join(outputdir, f'TRDB_MatrixGenotypes.txt')
    if chrom == chromlist[0]:
        outdata = ['\t'.join(i) for i in header + locusallgenotypes]
        gate = 0
    else:
        outdata = ['\t'.join(i) for i in locusallgenotypes]
        gate = 1
    append_out(outdata, outfile, gate, gz=False)

    outfile = os.path.join(outputdir, f'TRDB_MatrixMeanAlleleLength.txt')
    if chrom == chromlist[0]:
        outdata = ['\t'.join(i) for i in header + locuslengths]
        gate = 0
    else:
        outdata = ['\t'.join(i) for i in locuslengths]
        gate = 1
    append_out(outdata, outfile, gate, gz=False)

    outfile = os.path.join(outputdir, f'TRDB_MatrixAlleleLengths.txt')
    if chrom == chromlist[0]:
        outdata = ['\t'.join(i) for i in header + locusallelelengths]
        gate = 0
    else:
        outdata = ['\t'.join(i) for i in locusallelelengths]
        gate = 1
    append_out(outdata, outfile, gate, gz=False)

    outfile = os.path.join(outputdir, f'TRDB_MatrixAlleleMethylation.txt')
    if chrom == chromlist[0]:
        outdata = ['\t'.join(i) for i in header + locusallelemeth]
        gate = 0
    else:
        outdata = ['\t'.join(i) for i in locusallelemeth]
        gate = 1
    append_out(outdata, outfile, gate, gz=False)

    outfile = os.path.join(outputdir, f'TRDB_MatrixAlleleDepth.txt')
    if chrom == chromlist[0]:
        outdata = ['\t'.join(i) for i in header + locusalleledepth]
        gate = 0
    else:
        outdata = ['\t'.join(i) for i in locusalleledepth]
        gate = 1
    append_out(outdata, outfile, gate, gz=False)

    outfile = os.path.join(outputdir, f'TRDB_MatrixAllelePurity.txt')
    if chrom == chromlist[0]:
        outdata = ['\t'.join(i) for i in header + locusallelepurity]
        gate = 0
    else:
        outdata = ['\t'.join(i) for i in locusallelepurity]
        gate = 1
    append_out(outdata, outfile, gate, gz=False)

    outfile = os.path.join(outputdir, f'TRDB_MatrixMeanNumberMotifs.txt')
    if chrom == chromlist[0]:
        outdata = ['\t'.join(i) for i in header + locusnumreps]
        gate = 0
    else:
        outdata = ['\t'.join(i) for i in locusnumreps]
        gate = 1
    append_out(outdata, outfile, gate, gz=False)

    outfile = os.path.join(outputdir, f'TRDB_MatrixAlleleLengthDifference.txt')
    if chrom == chromlist[0]:
        outdata = ['\t'.join(i) for i in header + locusalllendiff]
        gate = 0
    else:
        outdata = ['\t'.join(i) for i in locusalllendiff]
        gate = 1
    append_out(outdata, outfile, gate, gz=False)
    
    return


def average_of_dict_values(dallelemeth):
    import statistics as st
    from natsort import natsorted

    avgallmeth = []
    for i in natsorted(list(dallelemeth.keys())):  # keys are strings, but integer-'like', '1', '4', '6', '2', '3'...
        meth = []
        for j in dallelemeth[i]:
            if j != '.': meth.append(float(j))  # exclude '.' from average allele methylation per allele calculation
        if meth == []: avgallmeth.append('.')
        else: avgallmeth.append(str(st.mean(meth)))
    # if you want to exclude the entire mean calculation per allele with missing data then
    # avgallmeth = [str(sum([float(j) for j in allelemeth[i]]) / len(allelemeth[i])) if '.' not in allelemeth[i] else '.' for i in natsorted(list(allelemeth.keys()))]  # 
    # avgallpur = [str(sum([float(j) for j in allelepurity[i]]) / len(allelepurity[i])) if '.' not in allelepurity[i] else '.' for i in natsorted(list(allelepurity.keys()))]

    return avgallmeth


def parse_trgt_vcf(infile, chromosome):
    ''' By default will output '''
    import gzip
    from natsort import natsorted

    dvariants = {}  # could collect all parsed info, but memory intensive
    ids = []
    with gzip.open(infile, 'rt') as FILE:
        for linecount, line in enumerate(FILE):
            # line = line.decode('utf-8')ß
            # print(line)
            if line[:len('#CHROM')] == '#CHROM':
                sampleIDs = line.strip().split('\t')[9:]
            if line[0] != '#':
                dsamples = {}  # will contain genoype info for all samples separated by sampleIDs # key sampleID, value var (dataclass variant)
                dallelemeth = {}
                dallelepurity = {}
                dallelemc = {}
                dallelems = {}
                l = line.strip().split('\t')
                if l[0] == chromosome:   # ADDED LINE TO MANAGE MEMORY AND GO CHROM BY CHROM
                    info = l[7].split(';')
                    id = info[0][5:]
                    ids.append(id)
                    format = l[8].split(':')
                    for i, sampleID in enumerate(sampleIDs):
                        # print(f'{i}, {sampleID}, {l[9:][i]}')
                        gen = l[9:][i].split(':') # each column l[9:] corresponds to the genotyp info for each sampleID
                        d = {}
                        for label, value in zip(format, gen): # needs to be updated to apply labels per sample
                            d[label] = value
                        if d['GT'] == '.':
                            dsamples[sampleID] = '.'
                        else:
                            # initialize variant # specify haploid/first allele info
                            var = variant(
                                genotype=d["GT"],
                                al1=d["GT"].split("/")[0],  #  if "/" in d["GT"] else d["GT"]
                                al1len=int(d["AL"].split(",")[0]),
                                al1lenrange=[int(x) for x in d["ALLR"].split(",")[0].split("-")],
                                al1depth=int(d["SD"].split(",")[0]),
                                al1motifcount=[int(x) for x in d["MC"].split(",")[0].split("_")],
                                al1motifspan=d["MS"].split(",")[0].split("_"),
                                al1pure=d["AP"].split(",")[0],
                                al1meth=d["AM"].split(",")[0],
                            )
                            # collect additional stats
                            dallelemeth.setdefault(var.al1, []).append(var.al1meth)  # sometimes al1meth is '.'
                            dallelepurity.setdefault(var.al1, []).append(var.al1pure)  # sometimes al1pur is '.'
                            if var.al1 not in dallelemc:  # Record the first occurrence of the list for the key # allele count lists for each allele should always be the same
                                dallelemc[var.al1] = [str(i) for i in var.al1motifcount]
                                dallelems[var.al1] = var.al1motifspan

                            # If diploid, assign additional diploid specific attributes to variant.
                            if var.ploidy == 'diploid':  # if "/" in d["GT"]:
                                var.al2 = d["GT"].split("/")[1]
                                var.al2len = int(d["AL"].split(",")[1])
                                var.al2lenrange = [int(x) for x in d["ALLR"].split(",")[1].split("-")]
                                var.al2depth = int(d["SD"].split(",")[1])
                                var.al2motifcount = [int(x) for x in d["MC"].split(",")[1].split("_")]
                                var.al2motifspan = d["MS"].split(",")[1].split("_")
                                var.al2pure = d["AP"].split(",")[1]
                                var.al2meth = d["AM"].split(",")[1]

                                # collect additional stats
                                dallelemeth.setdefault(var.al2, []).append(var.al2meth)
                                dallelepurity.setdefault(var.al2, []).append(var.al2pure)
                                if var.al2 not in dallelemc:  # Record the first occurrence of the list for the key # allele count lists for each allele should always be the same
                                    dallelemc[var.al2] = [str(i) for i in var.al2motifcount]
                                    dallelems[var.al2] = var.al2motifspan

                            # collect all variants per sample for each locus
                            dsamples[sampleID] = var

                    alleles = [l[3]] + l[4].split(',') if l[4] != '.' else [l[3]]  # [ref, plus, alt, alleles] or just [ref] if alt is '.'
                    cpgdensities = [str(CpG_density(seq)) for seq in alleles]
                    avgallmeth = average_of_dict_values(dallelemeth)
                    avgallpur = average_of_dict_values(dallelepurity)
                    allelemc = ['_'.join(dallelemc[i]) for i in natsorted(list(dallelemc.keys()))]  # sort list by increasing allele count, join list of motif counts (MC) by underscore, later join allelemc by commas ',' when outputing per locus, so commas will seperate allelles, underscore seperate motifcounts
                    allelems = ['_'.join(dallelems[i]) for i in natsorted(list(dallelems.keys()))]  # sort list by increasing allele count, join list of motif spans (MS) by underscore, later join allelems by commas ',', when outputing per locus, commas will seperate alleles, underscores seperate motifs

                    # locusID, Chrom, Start, End, allelesequences, motifsequences, cpgdensityperallele, avgallelemethylationperallele, allelemotifcounts, allelemotifspans, dict[sampleID]=variantdataclass(one of each individual)
                    # loc = locus(id, l[0], int(l[1]), int(info[1][4:]), alleles, info[2].split(','), cpgdensities, avgallmeth, avgallpur, allelemc, allelems, dsamples)  #END= in info field should be int(info[1][4:]) but END is not present anymore so use end from id as int(id.split('.')[-1])
                    loc = locus(id, l[0], int(l[1]), int(info[1][4:]), cpgdensities, avgallmeth, avgallpur, allelemc, allelems, dsamples)  # I removed allele and motif sequences because takes lots mem
                    dvariants[id] = loc
                    if linecount % 10000 == 0:
                        print(f'Parsing locus number: {linecount+1}, locus ID: {id}')
    
    return dvariants, ids, sampleIDs


def get_chrom_values(infile):
    chromlist = []  
    with gzip.open(infile, 'rt') as FILE:
        for linecount, line in enumerate(FILE):
            if line[0] != '#':
                chrom = line.strip().split('\t')[0]
                if chrom not in chromlist:
                    chromlist.append(chrom)

    print(f'Chromosomes present in infile: {chromlist}')
    return chromlist


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and process a VCF file to calculate stats per individual."
    )
    
    # Required arguments
    parser.add_argument("--infile", type=str, required=True, help="Path to the input VCF.gz file")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory, will be added to CWD")

    return parser.parse_args()


if __name__ == "__main__":
    import os
    import sys
    import argparse

    # program is memory inefficient, recommended to breakup .merged.vcf.gz by chromosome.
    # ex cmd: python TRGTMakeMatrixAlleleLength.py --outdir Matrices --infile /path/to/merged.vcf.gz

    args = parse_arguments()

    # Extracting arguments
    infile = args.infile
    outdir = args.outdir

    # Display the input arguments
    print(f"Input file: {infile}")

    chromlist = get_chrom_values(infile)

    # Process the input merged .VCF file
    # ids is list of locus IDs

    path, infilename = os.path.split(infile)
    outputdir = os.path.join(path, outdir)# infilename.split('.')[1])  # make subdirectories for each chromosome (assuming the merged .vcf was split with ParseTRGTMergedVCF_byChrom.py)
    # ensure output directory exists
    os.makedirs(outputdir, exist_ok=True)
    outfiles = [os.path.join(outputdir, f'TRDB_MatrixGenotypes.txt'), \
                os.path.join(outputdir, f'TRDB_MatrixMeanAlleleLength.txt'), \
                os.path.join(outputdir, f'TRDB_MatrixAlleleLengths.txt'), \
                os.path.join(outputdir, f'TRDB_MatrixAlleleMethylation.txt'), \
                os.path.join(outputdir, f'TRDB_MatrixAlleleDepth.txt'), \
                os.path.join(outputdir, f'TRDB_MatrixAllelePurity.txt'), \
                os.path.join(outputdir, f'TRDB_MatrixMeanNumberMotifs.txt'), \
                os.path.join(outputdir, f'TRDB_MatrixAlleleLengthDifference.txt')
                ]

    for outfile in outfiles:
        with open(outfile, 'w') as OUT:
            OUT.write('')  # create but wipe file contents if exists # will append to later by chrom

    for chrom in chromlist:
        print(f'Starting chromsome: {chrom}')
        dvariants, ids, sampleIDs = parse_trgt_vcf(infile, chrom)
        print(f'Finished parsing merged compressed VCF file: {infile}, for chromosome: {chrom}')

        print(f'Starting to make matrix, for chromosome {chrom}')
        make_matrix_locus_x_individual(dvariants, ids, sampleIDs, outputdir, chrom, chromlist)



# 
# # Description of VCF files generated by TRGT
# 
# The [Variant Calling Format (VCF)](https://samtools.github.io/hts-specs/VCFv4.3.pdf)
# files generated by TRGT describe the lengths, sequence composition, CpG
# methylation, and other properties of tandem repeats (TRs). Each entry
# consists of the fields described in the table below.
# 
# | VCF field | Description                                                   |
# |-----------|---------------------------------------------------------------|
# | CHROM     | Chromosome containing the repeat region                       |
# | POS       | Starting position of the repeat region                        |
# | ID        | Region identifier (currently set to ".")                      |
# | REF       | The full reference sequence of the region                     |
# | ALT       | Sequences of the repeat alleles                               |
# | QUAL      | Currently always set to 0                                     |
# | FILTER    | Currently always set to "."                                   |
# | INFO      | Fields describing the overall properties of the TR region     |
# | FORMAT    | Names of genotype fields describing the region in the sample  |
# | SAMPLE    | Values of genotype fields describing the region in the sample |
# 
# ## Information fields (INFO)
# 
# Information fields describe the overall structure of the repeat region,
# independent of any particular sample. They consist of the following fields:
# 
# | INFO field | Description                          | Example            |
# |------------|--------------------------------------|--------------------|
# | TRID       | Identifier of the repeat region      | HTT                |
# | END        | Ending position of the repeat region | 3074966            |
# | MOTIFS     | Comma separated list of TR motifs    | CAG,CCG            |
# | STRUC      | Structure of the repeat region       | (CAG)nCAACAG(CCG)n |
# 
# ## Genotype fields (FORMAT)
# 
# The names and values of the genotype fields are given in the two last columns of
# the VCF file.
# 
# | Genotype field | Description                                  | Example       |
# |----------------|----------------------------------------------|---------------|
# | GT             | Genotype                                     | 1/2           |
# | AL             | Allele length in bps                         | 84,105        |
# | ALLR           | Length range for AL                          | 80-85,102-114 |
# | SD             | Number of reads spanning each allele         | 28,18         |
# | MC             | Count of motifs of each TR on each allele    | 17_9,24_9     |
# | MS             | Span of each TR on each allele               | 0(0-51)_1(57-84),0(0-72)_1(78-105) |
# | AP             | Purity score for each allele                 | 0.5,0.9       |
# | AM             | Mean methylation level for each allele       | 0.4,0.5       |
# 
# ## MC and MS fields
# 
# The `MC` field contains the count of motifs of each TR on each allele. For
# example, if `MC` field is `17_9,24_9` and `MOTIFS` field is `CAG,CCG`, then one
# allele contains 17 CAGs and 9 CCGs while the second allele contains 25 CAGs and
# 9 CCGs (see panels A & B of the figure below).
# 
# The `MS` field contains spans of each TR on each allele. For example, if `MS`
# field is set to `0(0-51)_1(57-84),0(0-72)_1(78-105)` and `MOTIFS` field is
# `CAG,CCG`. This means that CAG spans bases 0 to 51 on the first allele and
# bases 0 to 72 on the second allele. The CCG repeat spans bases 57 to 84 on the
# first allele and bases 78 to 105 on the second allele (see panels A & C of the
# figure below).
# 
# <img width="600px" src="figures/VCF-overview.png"/>
# 
# ## AP field
# 
# The `AP` field contains *purity score* for each repeat allele. The score is a
# value between 0.0 and 1.0. If it is 1.0, the allele consists of the perfect
# repetition of the consensus motif. The closer this score is to 0.0, the more
# degenerate is the repeat. More formally, the purity score is defined through
# the [edit distance](https://en.wikipedia.org/wiki/Edit_distance): If `E` is the
# edit distance between the allele of length `L` and the perfect repeat of the
# same length, the purity score is defined by `(L - E) / L`.
# 
