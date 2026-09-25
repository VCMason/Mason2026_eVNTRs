
import os
import sys
import numpy as np
import argparse
import subprocess
import pandas as pd

def write_out(outlines, outfile):
    with open(outfile, "w") as OUT:
        OUT.write("\n".join(outlines))


def sort_input_file(infile):
    """Sort buildTRs TSV by chrom,start,end using GNU sort."""
    base, ext = os.path.splitext(infile)
    outfile = base + ".sort" + ext
    ### cmd = ["sort","-k1,1V","-k2,2V","-k3,3V", infile]. # infile has header that needs to be removed before sorting
    # 1. Define your sort flags without the input file name at the end
    cmd = ["sort", "-k1,1V", "-k2,2V", "-k3,3V"]

    # 2. Check if the input file exists before opening files
    if os.path.isfile(infile):
        print("Input file exists and is a file")

        with open(infile, "r") as FILE:
            lines = FILE.readlines()
        # Extract header
        header = lines[0].strip().replace(' ', '\t')  # for some reason there were spaces seperating chr start and end in the header line... everything else was tabs
        # to fix the header issue in the future fix the header.tsv file in the ./lib folder for the snakemake pipeline
        # Read the remaining data rows strictly into Python memory
        remaining_data = '\n'.join([line.strip() for line in lines[1:]]) + '\n'
        # Pass the remaining file content as stdin to the sort command
        #subprocess.run(cmd, stdin=FILE, stdout=OUT, check=True)
        with open(outfile, "w") as OUT:
            subprocess.run(cmd, input=remaining_data, stdout=OUT, text=True, check=True)
        with open(outfile, 'r') as FILE:
            sorted_data = FILE.readlines()
        # write header to first line
        output = [header] + [line.strip() for line in sorted_data]
        with open(outfile, 'w') as OUT:
            OUT.write('\n'.join(output)+'\n')
    else:
        raise FileNotFoundError(f"The input file '{infile}' does not exist.")

    return outfile


def all_motifs_long_enough(effmotifs, min_length):
    if pd.isna(effmotifs):
        return False
    return all(len(m)>=min_length for m in effmotifs.split(","))


def filter_tsv(file_path, min_feature_length, min_effmotif_length):
    df = pd.read_csv(file_path, sep="\t", dtype={0: str}, header=0)
    initial_count=len(df)
    print(df.describe())
    print(df.head)
    df["feature_length"]=df["end"]-df["start"]+1
    filtered=df[df["feature_length"]>=min_feature_length].copy()
    feature_filter_count=initial_count-len(filtered)
    filtered=filtered[filtered["effMotifs"].apply(lambda x: all_motifs_long_enough(x,min_effmotif_length))]
    eff_filter_count=initial_count-feature_filter_count-len(filtered)
    filtered.drop(columns=["feature_length"], inplace=True)
    base=os.path.basename(file_path).replace(".tsv","")
    out=f"{base}.filter.{min_feature_length}.{min_effmotif_length}.tsv"
    out=os.path.join(os.path.dirname(file_path),out)
    filtered.to_csv(out,sep="\t",index=False)
    print(f"Filtered file saved as {out}")
    print(f"Starting number of features: {initial_count}")
    print(f"Rows removed by feature length criterion: {feature_filter_count}")
    print(f"Rows removed by effMotif length criterion: {eff_filter_count}")
    print(f"Filtered number of features: {len(filtered)}")
    return out


def convert_TRF_to_TRGT(ids, dmotifs):
    outlines=[]; outsimple=[]; outcomplex=[]; outnotmonomer=[]
    for id in ids:
        chrom,start,end=id.split(".")
        motifs=dmotifs[id]
        if len(motifs)==1:
            meta=f"ID={id};MOTIFS={motifs[0]};STRUC=({motifs[0]})n"
            newline=f"{chrom}\t{start}\t{end}\t{meta}"
            outsimple.append(newline)
        else:
            eff=",".join(motifs)
            meta=f"ID={id};MOTIFS={eff};STRUC=<{id}>"
            newline=f"{chrom}\t{start}\t{end}\t{meta}"
            outcomplex.append(newline)
        if any(len(m)>1 for m in motifs):
            outnotmonomer.append(newline)
        outlines.append(newline)
    return outlines,outsimple,outcomplex,outnotmonomer


def get_id_motifs_counts_samples(motiffile):
    dmotifs={}; doptimalmotifcounts={}; doptimalmotifratio={}; ids=[]
    with open(motiffile) as f:
        for i,line in enumerate(f):
            if i==0 or not line.strip():
                continue
            id=".".join(line.strip().split("\t")[:3])
            ids.append(id)
            dmotifs[id]=line.strip().split("\t")[3].split(",")
            doptimalmotifcounts[id]=line.strip().split("\t")[6].split('/')  # doptimalmotifratio[id][0] is number of motifs chosen, doptimalmotifratio[id][1] is total number of motifs found  
            doptimalmotifratio[id]= int(line.strip().split("\t")[6].split('/')[0]) / int(line.strip().split("\t")[6].split('/')[1]) 
    return ids, dmotifs, doptimalmotifcounts, doptimalmotifratio


def main():
    '''
    Usage: python buildTRsToTRGT.py --infile buildTRs_effMotifFinal.all.tsv --min-feature-length 10 --min-effmotif-length 2
    This script sorts and filters the buildTRs effMotif TSV file and converts it to TRGT repeat definition file
    '''
    parser=argparse.ArgumentParser(description="Sort, filter buildTRs effMotif TSV, and convert to TRGT repeat definition files.")
    parser.add_argument("--infile",required=True,help="buildTRs effMotifFinal.all.tsv")
    parser.add_argument("--min-feature-length",type=int,default=10)
    parser.add_argument("--min-effmotif-length",type=int,default=2)
    args=parser.parse_args()

    print("Sorting input...")
    sorted_file=sort_input_file(args.infile)
    print(f"Sorted file: {sorted_file}")

    filtered_file=filter_tsv(sorted_file,args.min_feature_length,args.min_effmotif_length)

    ids, dmotifs, doptimalmotifcounts, doptimalmotifratio=get_id_motifs_counts_samples(filtered_file)
    meaneffmotifcount = np.mean([float(counts[0]) for id, counts in doptimalmotifcounts.items()])
    meaneffmotifratio = np.mean([float(ratio) for id, ratio in doptimalmotifratio.items()])
    print(f"Mean number of Effective motifs chosen actoss all loci {meaneffmotifcount}")
    print(f"Mean ratio of effective motifs chosen actoss all loci {meaneffmotifratio}")
    outlines,outsimple,outcomplex,outnotmonomer=convert_TRF_to_TRGT(ids,dmotifs)
    d=os.path.dirname(filtered_file)
    b=os.path.basename(filtered_file)[:-4]
    write_out(outlines, os.path.join(d,b+".TRGT.SimpleComplex.tsv"))
    write_out(outsimple, os.path.join(d,b+".TRGT.Simple.tsv"))
    write_out(outcomplex, os.path.join(d,b+".TRGT.Complex.tsv"))
    print(f"Number of simple repeat definitions: {len(outsimple)}")
    print(f"Number of HMM complex repeat definitions: {len(outcomplex)}")
    print(f"Number of total (simple + complex) repeat definitions: {len(outlines)}")

if __name__=="__main__":
    main()
