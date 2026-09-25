import os
import re
import argparse
import gzip
import glob
import subprocess
import numpy as np
import pandas as pd
from natsort import natsorted


def write_out(outlines, outfile, gz=None):
    if gz:
        import gzip
        outfile = outfile if outfile[-3:] == '.gz' else f'{outfile}.gz'
        with gzip.open(outfile, 'wt') as OUT:
            OUT.write('\n'.join(outlines))
    else:
        with open(outfile, 'w') as OUT:
            OUT.write('\n'.join(outlines))

def get_ordered_subfolders():
    subfolders = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]
    # sorted_subfolders = sorted(subfolders, key=lambda x: (x.isdigit(), x))
    sorted_subfolders = [str(i) for i in natsorted([f for f in subfolders if f.isdigit()])]
    return sorted_subfolders

def create_output_folder(folder_name):
    output_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def get_common_files(subfolders, extensions):
    file_sets = []
    for folder in subfolders:
        files = {f for f in os.listdir(folder) if any(f.endswith(ext) for ext in extensions)}
        file_sets.append(files)
    return set.intersection(*file_sets)

def concatenate_files(subfolders, common_files, output_folder):
    for comfile in common_files:
        output_file = os.path.join(output_folder, f"{os.path.basename(output_folder)}_{comfile}")
        
        # Determine if output should be gzipped
        is_gzipped = output_file.endswith(".gz")

        # Open file in correct mode
        open_func = gzip.open if is_gzipped else open
        write_mode = 'wt' if is_gzipped else 'w'  # Use text mode to avoid binary newline issues
        
        with open_func(output_file, write_mode) as OUT:
            first_file = True  # Track first file
            last_char = None
            
            for folder in subfolders:
                input_file = os.path.join(folder, comfile)
                
                if os.path.exists(input_file):  # Ensure the file exists
                    read_mode = 'rt' if is_gzipped else 'r'  # Read text, handle gzip correctly
                    
                    with open_func(input_file, read_mode) as FILE:
                        if not first_file and last_char and last_char != "\n":
                            OUT.write("\n")  # Add a clean newline before appending

                        # Read file line by line to ensure clean concatenation
                        last_char = None  # Track last character written

                        for line in FILE:
                            OUT.write(line)
                            last_char = line[-1]  # Store last character
                        
                        first_file = False  # First file is now processed
                else:
                    print(f"⚠️ Warning: {input_file} not found, skipping.")



def reformat_specific_files(special_files, output_folder):
    # make special file concatenated files full paths
    special_file_paths = [os.path.join(output_folder, f"{os.path.basename(output_folder)}_{file}") for file in special_files]
    # for each special file reformat
    for infile in special_file_paths:  # output file will be same name as special file, overwrite
        with open(infile, 'r') as FILE:
            if 'ProteinCodingGenes' in infile:
                outlines = natsorted([line.strip() for line in FILE])  # gene names
            elif 'TRFeatureTypes' in infile:
                dfeattypes = {}
                for line in FILE:
                    if ': ' in line:
                        feature, TRoverlapcount = line.strip().split(': ')[0], int(line.strip().split(': ')[1])
                        dfeattypes[feature] = dfeattypes.get(feature, 0) + TRoverlapcount
                outlines = [f'{feattype}: {count}' for feattype, count in natsorted(dfeattypes.items())]

        write_out(outlines, infile)
        print(f"Reformatted special files: {special_files}")

    return

def sum_matrices(subfolders, matrix_pattern, output_folder, matrix_type):
    summed_matrix = None
    total_loci_count = 0
    row_labels, col_labels = None, None

    for folder in subfolders:
        matrix_files = glob.glob(os.path.join(folder, matrix_pattern))
        for matrix_file in matrix_files:
            filename = os.path.basename(matrix_file)

            # Extract count from filename
            match = re.search(r'(LociCount|DenominatorCount)(\d+)', filename)
            if match:
                loci_count = int(match.group(2))
                total_loci_count += loci_count
            else:
                raise ValueError(f"Unexpected filename format: {filename}")

            # Read the matrix using NumPy
            raw_data = np.genfromtxt(matrix_file, delimiter=",", dtype=str)

            # Extract row/column labels (first row and first column)
            if row_labels is None:
                col_labels = raw_data[0, 1:]  # First row, skipping first column
                row_labels = raw_data[1:, 0]  # First column, skipping first row

            # Convert data to a numeric matrix (skip first row/col)
            matrix = raw_data[1:, 1:].astype(float)

            # Sum matrices
            if summed_matrix is None:
                summed_matrix = matrix
            else:
                summed_matrix += matrix  # Element-wise addition

    # Save summed matrix
    output_filename = f"SimilarityMatrixRaw_Total{matrix_type}Count{total_loci_count}.csv"
    output_path = os.path.join(output_folder, output_filename)
    save_labeled_matrix(summed_matrix, row_labels, col_labels, output_path)

    return summed_matrix, total_loci_count, row_labels, col_labels


def save_labeled_matrix(matrix, row_labels, col_labels, filename):
    header = "," + ",".join(map(str, col_labels))
    labeled_matrix = np.column_stack((row_labels, matrix))
    np.savetxt(filename, labeled_matrix, delimiter=",", fmt="%s", header=header, comments="")

def compute_shared_allele_proportions(loci_matrix, denom_matrix, loci_count, row_labels, col_labels, output_folder):
    # Perform element-wise division, avoiding divide-by-zero errors
    proportion_matrix = np.divide(loci_matrix, denom_matrix, out=np.zeros_like(loci_matrix), where=denom_matrix != 0)
    # Save the resulting matrix
    output_filename = f"SimilarityMatrixRaw_SharedAlleleProportions_LociCount{loci_count}.csv"
    output_path = os.path.join(output_folder, output_filename)
    save_labeled_matrix(proportion_matrix, row_labels, col_labels, output_path)

    print(f"Shared Allele Proportions saved to {output_path}")


def main():
    #usage: python TRGTConcatenateChromResults.py --outdir Autosomes --skip2matrix (optional)
    parser = argparse.ArgumentParser(description="Concatenate files and process matrices from multiple subfolders.")
    parser.add_argument("--outdir", type=str, required=True, help="Name of the output folder where concatenated files will be stored in CWD")
    parser.add_argument("--skip2matrix", action="store_true", help="Skip file concatenation and only perform matrix summation / normalization")
    args = parser.parse_args()
    
    subfolders = get_ordered_subfolders()
    output_folder = create_output_folder(args.outdir)
    
    if not args.skip2matrix:
        extensions = ['.txt', '.txt.gz', '.tsv', '.tsv.gz']  # , 'ClusterAssignments.csv'
        common_files = get_common_files(subfolders, extensions)
        concatenate_files(subfolders, common_files, output_folder)
        # special files are not gzipped
        special_files = ['ProteinCodingGenes.txt', 'CountTRFeatureTypes.txt']
        reformat_specific_files(special_files, output_folder)
    
    loci_matrix, loci_count, row_labels, col_labels = sum_matrices(subfolders, "SimilarityMatrixRaw_LociCount*.csv", output_folder, "Loci")
    denom_matrix, _, _, _ = sum_matrices(subfolders, "SimilarityMatrixRaw_DenominatorCount*.csv", output_folder, "Denominator")
    
    compute_shared_allele_proportions(loci_matrix, denom_matrix, loci_count, row_labels, col_labels, output_folder)

if __name__ == "__main__":
    main()
