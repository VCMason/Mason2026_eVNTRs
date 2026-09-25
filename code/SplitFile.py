import argparse
import os

def split_file(input_file, num_parts, output_prefix="splitTRGTLoci"):
    # Read all lines from the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    lines_per_file = total_lines // num_parts
    remainder = total_lines % num_parts

    path, infilename = os.path.split(input_file)

    start = 0
    for i in range(num_parts):
        # Distribute the remainder: one extra line for the first 'remainder' files
        end = start + lines_per_file + (1 if i < remainder else 0)
        split_lines = lines[start:end]

        suffix = f"{i+1:02d}"  # Format: 01, 02, ...
        output_file = os.path.join(path, f"{output_prefix}_{suffix}")

        with open(output_file, 'w') as out:
            out.writelines(split_lines)

        start = end  # Move to the next chunk

    print(f"Successfully split '{input_file}' into {num_parts} files.")

def main():
    parser = argparse.ArgumentParser(description="Split a file into N parts with approximately equal lines.")
    parser.add_argument('--infile', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--n', type=int, required=True, help="Number of output files to create.")
    parser.add_argument('--prefix', type=str, required=False, default="splitTRGTLoci", help="Prefix for output files.")
    args = parser.parse_args()

    split_file(args.infile, args.n)

if __name__ == "__main__":
    main()
