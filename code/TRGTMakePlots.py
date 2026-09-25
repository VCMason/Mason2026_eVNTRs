# cluster and make heatmap, dendro, and pac
import os
import sys
import gzip
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list, dendrogram


def write_out(outlines, outfile, gz=None):
    if gz:
        import gzip
        outfile = outfile if outfile[-3:] == '.gz' else f'{outfile}.gz'
        with gzip.open(outfile, 'wt') as OUT:
            OUT.write('\n'.join(outlines))
    else:
        with open(outfile, 'w') as OUT:
            OUT.write('\n'.join(outlines))


def plot_heatmap(matrix, labels, outfile):
    """
    Create and save a heatmap of the clustered similarity matrix.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 20))
    sns.heatmap(matrix, 
                cmap="viridis", 
                cbar=True, 
                square=True, 
                xticklabels=labels, 
                yticklabels=labels, 
                annot=False)
    plt.xticks(rotation=45, ha="right", fontsize=4)
    plt.yticks(rotation=0, fontsize=4)
    plt.title("Clustered Heatmap")
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {outfile}")

    return


def plot_colored_dendrogram_breeds(linkage_matrix, labels, breeds, threshold, outfile):
    """
    Plot a dendrogram with leaf nodes colored by breed classification.

    Parameters:
    - linkage_matrix: ndarray
        The linkage matrix from hierarchical clustering.
    - labels: list
        The labels for each row/column of the matrix.
    - breeds: list
        List of known breed information corresponding to labels.
    - outfile: str
        File path to save the dendrogram as a .pdf.
    - threshold: float
        The distance threshold used for clustering.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    # Assign colors to breeds
    unique_breeds = np.unique(breeds)
    color_map = plt.get_cmap("tab20", len(unique_breeds))  # Support up to 20 distinct colors or more
    colors = color_map(np.linspace(0, 1, len(unique_breeds)))
    breed_colors = {breed: f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for breed, c in zip(unique_breeds, colors)}
    ### # Assign colors to breeds
    ### colors = plt.cm.tab10(np.linspace(0, 1, len(unique_breeds)))  # Generate distinct colors
    ### breed_colors = {breed: f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for breed, c in zip(unique_breeds, colors)}

    # Map leaves to breeds and colors
    leaf_to_breed = {i: breed for i, breed in enumerate(breeds)}
    leaf_to_color = {i: breed_colors[breed] for i, breed in leaf_to_breed.items()}

    def color_func(link_id):
        """
        Assign colors to branches based on breed classification.
        """
        if link_id < len(labels):
            # Leaf node: return its assigned breed color
            return leaf_to_color[link_id]
        else:
            # Internal node: determine branch color based on threshold
            dist = linkage_matrix[link_id - len(labels), 2]  # Get the distance
            if dist <= threshold:
                # Use one of its child breed colors
                left, right = map(int, linkage_matrix[link_id - len(labels), :2])
                breed = leaf_to_breed.get(left, leaf_to_breed.get(right, None))
                return breed_colors.get(breed, "black")
            else:
                # Above threshold: return black
                return "black"

    # Plot the dendrogram
    plt.figure(figsize=(20, 20))
    dendro = dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=4,
        above_threshold_color="black",  # Black for branches above threshold
        link_color_func=color_func,     # Custom coloring function
    )

    # Color leaf labels by breed
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl_text = lbl.get_text()
        leaf_index = labels.index(lbl_text)
        lbl.set_color(leaf_to_color[leaf_index])

    # Add a legend for breeds
    legend_handles = [plt.Line2D([0], [0], color=color, lw=4, label=breed) for breed, color in breed_colors.items()]
    plt.legend(handles=legend_handles, loc="upper right", title="Breeds", fontsize=8)

    # Plot the horizontal line at the threshold
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold: {threshold}')

    # Add title and labels
    plt.title(f"Dendrogram: Leaf Labels Colored by Breed (Threshold: {threshold})")
    plt.xlabel("Individuals")
    plt.ylabel("Cophenetic Distance")

    # Save the plot
    plt.tight_layout()
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Dendrogram saved to {outfile}")

    return


def plot_colored_dendrogram(linkage_matrix, labels, cluster_assignments, threshold, outfile):
    """
    Plot a dendrogram with branches colored by fcluster assignments below the threshold.

    Parameters:
    - linkage_matrix: ndarray
        The linkage matrix from hierarchical clustering.
    - labels: list
        The labels for each row/column of the matrix.
    - cluster_assignments: ndarray
        Cluster assignments from fcluster().
    - outfile: str
        File path to save the dendrogram as a .pdf.
    - threshold: float
        The distance threshold used for clustering.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    # Assign colors to clusters
    unique_clusters = np.unique(cluster_assignments)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))  # Generate distinct colors
    cluster_colors = {cluster: f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for cluster, c in zip(unique_clusters, colors)}

    # Map leaves to clusters and colors
    leaf_to_cluster = {i: cluster for i, cluster in enumerate(cluster_assignments)}
    leaf_to_color = {i: cluster_colors[cluster] for i, cluster in leaf_to_cluster.items()}

    def color_func(link_id):
        """
        Assign colors to branches based on cluster assignments and threshold.
        """
        if link_id < len(labels):
            # Leaf node: return its assigned cluster color
            return leaf_to_color[link_id]
        else:
            # Internal node: determine branch color based on threshold
            dist = linkage_matrix[link_id - len(labels), 2]  # Get the distance
            if dist <= threshold:
                # Use one of its child cluster colors
                left, right = map(int, linkage_matrix[link_id - len(labels), :2])
                cluster = leaf_to_cluster.get(left, leaf_to_cluster.get(right, None))
                return cluster_colors.get(cluster, "black")
            else:
                # Above threshold: return black
                return "black"

    # Plot the dendrogram
    plt.figure(figsize=(20, 20))
    dendro = dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=4,
        above_threshold_color="black",  # Black for branches above threshold
        link_color_func=color_func,     # Custom coloring function
    )

    # Plot the horizontal line at the threshold
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold: {threshold}')

    # Add title, labels, and legend
    plt.title(f"Dendrogram: Colored by Clusters (Threshold: {threshold})")
    plt.xlabel("Individuals")
    plt.ylabel("Cophenetic Distance")
    plt.legend(loc="upper right")

    # Save the plot
    plt.tight_layout()
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Dendrogram saved to {outfile}")

    return


def plot_clusters_pca(matrix, labels, cluster_assignments, output_basefile):
    """
    Create and save 2D scatter plots of PCA components with proper coloring.
    Generates two plots: PC1 vs. PC2 and PC2 vs. PC3.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Step 1: Perform PCA
    pca = PCA(n_components=4)  # Calculate the first four principal components
    coords = pca.fit_transform(matrix)  # PCA coordinates
    explained_variance = pca.explained_variance_ratio_ * 100  # Percentage of variance explained

    # Step 2: Assign colors to clusters
    unique_clusters = np.unique(cluster_assignments)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

    # Helper function to plot components
    def plot_components(x, y, xlabel, ylabel, filename):
        plt.figure(figsize=(20, 20))
        for cluster in unique_clusters:
            cluster_indices = np.where(cluster_assignments == cluster)[0]
            cluster_points = coords[cluster_indices]

            plt.scatter(
                cluster_points[:, x],
                cluster_points[:, y],
                color=cluster_colors[cluster],
                label=f"Cluster {cluster}",
                s=100,
                alpha=0.8,
            )

        # Annotate points with labels
        for i, (x_coord, y_coord) in enumerate(coords[:, [x, y]]):
            plt.text(x_coord, y_coord, labels[i], fontsize=4, ha="right", va="bottom")

        # Add labels and title
        plt.title(f"PCA: {xlabel} vs. {ylabel}")
        plt.xlabel(f"{xlabel} ({explained_variance[x]:.2f}% variance)")
        plt.ylabel(f"{ylabel} ({explained_variance[y]:.2f}% variance)")
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), fontsize=8)
        plt.tight_layout()

        # Save the plot
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {filename}")

    # Step 3: Plot PC1 vs. PC2
    pc1_pc2_file = f"{output_basefile}_PC1_vs_PC2.pdf"
    plot_components(0, 1, "PC1", "PC2", pc1_pc2_file)

    # Step 4: Plot PC2 vs. PC3
    pc2_pc3_file = f"{output_basefile}_PC2_vs_PC3.pdf"
    plot_components(1, 2, "PC2", "PC3", pc2_pc3_file)

    # Step 5: Plot PC2 vs. PC4
    pc2_pc4_file = f"{output_basefile}_PC2_vs_PC4.pdf"
    plot_components(1, 3, "PC2", "PC4", pc2_pc4_file)

    return


def evaluate_cluster_predictions(sampleIDs, clusters, breeds, output_summary_file, output_misclassified_file):
    """
    Evaluates how well the breed predicts the assigned cluster and identifies misclassified samples.

    Parameters:
        sampleIDs (list): List of sample identifiers.
        clusters (list): List of cluster assignments corresponding to sampleIDs.
        breeds (list): List of known breed information corresponding to sampleIDs.
        output_summary_file (str): Path to the file to write the summary output.
        output_misclassified_file (str): Path to the file to write the misclassified samples.

    Returns:
        None
    """
    from collections import Counter

    # Create a mapping of breed to majority cluster
    breed_cluster_map = {}
    misclassified_samples = []

    # Group samples by breed and determine the majority cluster for each breed
    breed_groups = {}
    for sample, cluster, breed in zip(sampleIDs, clusters, breeds):
        breed_groups.setdefault(breed, []).append(cluster)

    for breed, cluster_list in breed_groups.items():
        cluster_counts = Counter(cluster_list)
        majority_cluster = cluster_counts.most_common(1)[0][0]
        breed_cluster_map[breed] = majority_cluster

    # Identify misclassified samples
    for sample, cluster, breed in zip(sampleIDs, clusters, breeds):
        expected_cluster = breed_cluster_map[breed]
        if cluster != expected_cluster:
            misclassified_samples.append({
                'Sample': sample,
                'Breed': breed,
                'AssignedCluster': cluster,
                'ExpectedCluster': expected_cluster
            })

    # Calculate accuracy
    total_samples = len(sampleIDs)
    correct_assignments = sum(
        cluster == breed_cluster_map[breed]
        for cluster, breed in zip(clusters, breeds)
    )
    accuracy = correct_assignments / total_samples

    # Write summary to file
    with open(output_summary_file, 'w') as summary_file:
        summary_file.write("Breed to Cluster Majority Mapping:\n")
        for breed, majority_cluster in breed_cluster_map.items():
            summary_file.write(f"{breed}: {majority_cluster}\n")
        summary_file.write(f"\nOverall Accuracy: {accuracy:.2%}\n")

    # Write misclassified samples to file
    with open(output_misclassified_file, 'w') as misclassified_file:
        misclassified_file.write("Sample\tBreed\tAssignedCluster\tExpectedCluster\n")
        for misclassified in misclassified_samples:
            misclassified_file.write(
                f"{misclassified['Sample']}\t{misclassified['Breed']}\t{misclassified['AssignedCluster']}\t{misclassified['ExpectedCluster']}\n")

    return


def sort_sampleinfo(sorted_labels, metadatafile):
    """
    # actually doesn't have to accept sorted_labels (just labels you want to sort the metada IDs:breed_index by)
    Sorts the 'BreedGroup' column of the metadata file based on the order of sorted_labels.
    
    Parameters:
        sorted_labels (list): Ordered list of sample IDs.
        sorted_clusters (list): Corresponding list of cluster assignments (not used for sorting).
        metadatafile (str): Path to the metadata file (tab-separated).
    
    Returns:
        list: Sorted list of 'BreedGroup' values.
    """
    # sort the infile column 'BreedGroup' by local variable label. the column with the sample names in the padas df is named 'ID'.
    # Sort column 'ID' from infile by the values in sorted_labels, then apply that sorted order to column 'BreedGroup'.

    # Read the metadata file into a dictionary
    with open(metadatafile, 'r') as FILE:
        header = FILE.readline().strip().split('\t')  # Read and parse the header
        id_index = header.index('ID')  # Find the column index for 'ID'        
        breed_index = header.index('BreedGroup')  # Find the column index for 'BreedGroup'
        data = [line.strip().split('\t') for line in FILE if line.strip() != '']  # Read the rest of the file as rows
    
    for row in data:
        if len(row) <= max(id_index, breed_index):
            print(f"Skipping malformed row: {row}")
        else:
            # Create a mapping of ID to BreedGroup from the metadata. {[sampleID]: breedname}
            id_to_breed = {row[id_index]: row[breed_index] for row in data}
    
    # Generate the sorted list of 'BreedGroup' values based on sorted_labels
    sorted_breed_groups = [id_to_breed[sampleID] for sampleID in sorted_labels if sampleID in id_to_breed]
    
    return sorted_breed_groups, id_to_breed


def get_threshold_for_num_clusters(linkage_matrix, num_clusters):
    """
    determine the dissimilarity threshold 
    to identtify the desired/specified number of clusters.
    
    :param dists: A squareform distance matrix (condensed or full)
    :param num_clusters: The desired number of clusters
    :return: The dissimilarity threshold and cluster labels
    """
    # Extract the dissimilarities (i.e., step.dissimilarity in Rust)
    dissimilarities = linkage_matrix[:, 2]  # Third column contains distances

    # Determine the threshold that results in `num_clusters`
    threshold = None
    for i, d in enumerate(dissimilarities[::-1]):  # Iterate in reverse
        clusters = fcluster(linkage_matrix, d, criterion='distance')
        unique_clusters = len(set(clusters))
        
        if unique_clusters == num_clusters:
            threshold = d
            break

    # If only 1 cluster should be found (high similarity), return that threshold
    if threshold is None:
        print('No threshold found to define ')
        threshold = max(dissimilarities)  # Return max distance if all points are too similar

    return threshold


def cluster_and_assign_group(matrix, labels, threshold, outfile, linkage_outfile, num_clusters=None):
    """
    Perform clustering on a symmetric similarity matrix, assign groups, and save results.
    
    Parameters:
    - matrix: ndarray
        The symmetric similarity matrix.
    - labels: list
        The labels corresponding to rows/columns of the matrix.
    - threshold: float
        The distance threshold for clustering.
    - outfile: str
        File path to save cluster assignments.
    - linkage_outfile: str
        File path to save the linkage matrix with sample ID labels.
    
    Returns:
    - linkage_matrix: ndarray
        The linkage matrix.
    - cluster_assignments: ndarray
        Cluster assignments for each individual.
    - sorted_matrix: ndarray
        The reordered similarity matrix.
    - sorted_labels: list
        The reordered labels.
    - sorted_clusters: list
        The reordered cluster assignments.
    """
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list

    # Step 1: Ensure the matrix is symmetric
    print(f'matrix shape before clustering: {matrix.shape}')
    assert np.allclose(matrix, matrix.T), "Matrix must be symmetric."
    
    # Step 2: Perform hierarchical clustering
    linkage_matrix = linkage(matrix, method="ward")
    
    if num_clusters:
        print(f'Number of clusters specified: {num_clusters}')
        print(f'Original cophenetic distance threshold before override: {threshold}')
        threshold = get_threshold_for_num_clusters(linkage_matrix, num_clusters)
        print(f'New threshold={threshold} after override, which represents cophenetic distance to define {num_clusters} clusters')

    # Step 3: Assign clusters based on the threshold
    cluster_assignments = fcluster(linkage_matrix, t=threshold, criterion="distance")
    
    # Step 4: Get dendrogram-based ordering
    order = leaves_list(linkage_matrix)
    
    # Step 5: Reorder the matrix, labels, and clusters
    sorted_matrix = matrix[order, :][:, order]
    sorted_labels = [labels[i] for i in order]
    sorted_clusters = [cluster_assignments[i] for i in order]
    
    # Step 6: Save cluster assignments with labels to a file
    sampleID_to_cluster = {}
    with open(outfile, "w") as f:
        f.write("Individual,Cluster\n")
        for label, cluster in zip(sorted_labels, sorted_clusters):
            name = label if label[-len('.mm2'):] != '.mm2' else label[:-len('.mm2')]  # fixing fucking ID issue
            sampleID_to_cluster[str(name)] = str(cluster)
            f.write(f"{label},{cluster}\n")
    print(f"Cluster assignments saved to: {outfile}")
    
    # Step 7: Save the linkage matrix to a file
    with open(linkage_outfile, "w") as f:
        f.write("Index1,Index2,Distance,Sample1,Sample2\n")
        for row in linkage_matrix:
            idx1, idx2, dist, count = int(row[0]), int(row[1]), row[2], int(row[3])
            sample1 = labels[idx1] if idx1 < len(labels) else f"Cluster-{idx1}"
            sample2 = labels[idx2] if idx2 < len(labels) else f"Cluster-{idx2}"
            f.write(f"{idx1},{idx2},{dist:.6f},{sample1},{sample2}\n")
    print(f"Linkage matrix saved to: {linkage_outfile}")

    return linkage_matrix, cluster_assignments, sorted_matrix, sorted_labels, sorted_clusters, sampleID_to_cluster, threshold


def read_in_numpy_matrix(matrix_file):
    # Read the matrix using NumPy
    raw_data = np.genfromtxt(matrix_file, delimiter=",", dtype=str)
                             
    # Extract row/column labels (first row and first column)
    col_labels = raw_data[0, 1:]  # First row, skipping first column
    row_labels = raw_data[1:, 0]  # First column, skipping first ro
        
    # Convert data to a numeric matrix (skip first row/col)
    matrix = raw_data[1:, 1:].astype(float)

    return matrix, col_labels, row_labels


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and process a VCF file to calculate stats per individual."
    )
    
    # Required arguments
    parser.add_argument(
        "--infile",
        type=str,
        required=True, 
        help="Path to the input VCF.gz file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True, 
        help="Ex: 1.5, distance threshold value to assign individual samples to clusters, based upon linkage_matrix. Requires --sample arguement"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True, 
        help="Output directory, will be added to CWD"
    )
    #optional args
    parser.add_argument(
        "--numclusters", 
        type=int, 
        required=False, 
        help="If provided this argument changes the function of threshold. With --numclusters, the threshold must be an integer \
            value and it defines the number of clusters to be found after clustering. When no --numclusters is provided the threshold \
            value can be a float and refers to the cophenetic distance to define clusters."
    )

    # optional string arguements
    parser.add_argument(
        "--metadata", 
        type=str, 
        required=False, 
        help="full path to tab delimited .tsv file with sample information. rows sample ID, columns metadata information."
    )

    return parser.parse_args()


def main():
    #usage: python TRGTMakePlots.py --infile /full/path/to/matrix.w.row.col.labels.csv --outdir plots --threshold 2 --numclusters 7 --metadata /full/path/to/sampledata.tsv
    args = parse_arguments()

    # Extracting arguments
    infile = args.infile
    outdir = args.outdir
    threshold = args.threshold
    num_clusters = args.numclusters
    metadata = args.metadata

    path, infilename = os.path.split(infile)
    outputdir = os.path.join(path, outdir)# infilename.split('.')[1])  # make subdirectories for each chromosome (assuming the merged .vcf was split with ParseTRGTMergedVCF_byChrom.py)
    # ensure output directory exists
    os.makedirs(outputdir, exist_ok=True)

    
    print("Will make plots of samples")

    matrix, col_labels, row_labels = read_in_numpy_matrix(infile)  # first row/column of matrix are sample labels. The matrix must be symmetrical
    sampleIDs = col_labels
    fixedIDs = [name if name[-len('.mm2'):] != '.mm2' else name[:-len('.mm2')] for name in sampleIDs]  # fix some ID issues # from messy input naming

    outfile = os.path.join(outputdir, 'ClusterAssignments.csv')
    linkage_outfile = os.path.join(outputdir, 'LinkageMatrix.csv')
    linkage_matrix, cluster_assignments, sorted_matrix, sorted_labels, sorted_clusters, sampleID_to_cluster, threshold = cluster_and_assign_group(matrix, fixedIDs, threshold, outfile, linkage_outfile, num_clusters)


    if metadata:
        outfile_summary = os.path.join(outputdir, 'BreedVSClusterAssignment_Summary.tsv')
        outfile_misclassified = os.path.join(outputdir, 'BreedVSClusterAssignment_Misclassified.tsv')
        breedgroups, sampleID_to_breed  = sort_sampleinfo(fixedIDs, metadata)
        sorted_breedgroups, _ = sort_sampleinfo(sorted_labels, metadata)  # could be relevant in some scenarios... instead of line above
        evaluate_cluster_predictions(fixedIDs, cluster_assignments, breedgroups, outfile_summary, outfile_misclassified)

    print("Starting to make plots...")
    outfilebasename = os.path.join(outputdir, 'SampleClusters')
    plot_clusters_pca(sorted_matrix, sorted_labels, sorted_clusters, outfilebasename)  # make dendrogram and scatter plot of samples colored by assigned cluster group

    outfile = os.path.join(outputdir, 'Dendrogram.pdf')
    plot_colored_dendrogram(linkage_matrix, fixedIDs, cluster_assignments, float(threshold), outfile)

    if metadata:
        outfile = os.path.join(outputdir, 'Dendrogram_ColoredByBreed.pdf')
        plot_colored_dendrogram_breeds(linkage_matrix, fixedIDs, breedgroups, float(threshold), outfile)

        outfilebasename = os.path.join(outputdir, 'SampleBreeds')
        plot_clusters_pca(sorted_matrix, sorted_labels, np.asarray(sorted_breedgroups), outfilebasename)  # make dendrogram and scatter plot of samples colored by assigned cluster group

        outfile = os.path.join(outputdir, 'MatrixHeatmap.pdf')
        plot_heatmap(sorted_matrix, sorted_labels, outfile)  # make heatmap of the rotated, and sorted upper triangle of symmetric similarity matrix
    print("Finished per-sample analysis.")


if __name__ == "__main__":
    main()