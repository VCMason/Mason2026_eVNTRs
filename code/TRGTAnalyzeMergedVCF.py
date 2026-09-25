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


# infile = '/Users/vmason/Documents/Analyses/trgt/120Hifi_AlingedToUCD2.0/SimpleRepeatDefinitionsFromUCD2.0/output/BSWCHEM120143117594.vcf.gz'  # trgt.output.vcf.gz
# outfile = '/Users/vmason/Documents/Analyses/trgt/120Hifi_AlingedToUCD2.0/SimpleRepeatDefinitionsFromUCD2.0/output/BSWCHEM120143117594.stats.tsv'  # make table of relevant values

@dataclass
class locus:  # a variant from TRGT output file .vcf.gz
    id: str
    chrom: str
    start: int
    end: int
    alleles: List[str] = field(default_factory=list)   # list of strings [ref, allele 1, (maybe) allele2] # call relevent seq by genotype number 0, 1, or 2... etc
    motifs: List[str] = field(default_factory=list)  # list of strings, [motifs, used, to, genotype, locus]
    cpgdensities: List[str] = field(default_factory=list)  # list of strings (but actually float values), ['CG' density in sequence] ordered from allele 0,1,2,3,...)
    avgallelemeth: List[str] = field(default_factory=list)  # list of strings (but actually float values), average methylation per allele ordered from allele 0,1,2,3,...)
    avgallelepur: List[str] = field(default_factory=list)   # list of strings (but actually float values), average allele purity per allele (ordered from allele 0,1,2,3,...)
    allelemc: List[str] = field(default_factory=list)  # list of strings, each string is underscore delimited motif counts per allele (ordered from allele 0,1,2,3,...)
    allelems: List[str] = field(default_factory=list)  # list of string, each string is a motif span, motif span is motif#(allelestart-alleleend)_motif#(start-end)... there is one span per allele each span is comma delimited # ex: ['0(0-51)_1(57-84)','0(0-72)_1(78-105)']
    samplegenotypes: Dict[str, "variant"] = field(default_factory=dict)  # dictionary, key is sampleID, value is dataclass variant (for that sample)

    # def CpG_density(self) -> list:
    #     cgdensities = []
    #     for seq in self.alleles:
    #         if seq == '.':  # then the alt allele matches the ref allele
    #             seq = self.alleles[0]  # then run ref allele again
    #         cgcount = seq.upper().count('CG')
    #         seqlen = len(seq)
    #         cgdensity = cgcount / seqlen
    #         cgdensities.append(cgdensity)
    #     return cgdensities


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


# def gc_percent(seq):  # d, scaff, start, end
#     GCcount = 0
#     # seq = d[scaff][start:end+1]
#     for base in seq:
#         if (base == 'G') or (base == 'C'):
#             GCcount += 1
#     GCdensity = round(GCcount / len(seq), 2)
# 
#     return GCdensity, seq


def write_out(outlines, outfile, gz=None):
    if gz:
        import gzip
        outfile = outfile if outfile[-3:] == '.gz' else f'{outfile}.gz'
        with gzip.open(outfile, 'wt') as OUT:
            OUT.write('\n'.join(outlines))
    else:
        with open(outfile, 'w') as OUT:
            OUT.write('\n'.join(outlines))


def CpG_density(seq):
    
    cgcount = seq.upper().count('CG')
    seqlen = len(seq)
    cgdensity = cgcount / seqlen

    return cgdensity


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
    color_map = plt.cm.get_cmap("tab20", len(unique_breeds))  # Support up to 20 distinct colors or more
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
    pca = PCA(n_components=3)  # Calculate the first three principal components
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

    return


def unique_set_elements_per_group(dictofsets):
    # dictofsets is a dictionary, key is group string, value is a set of unique values
    # function finds unique values in each group when compared to all others
    # dict key is breed or cluster group, value is set of unique alleles / motif indeces
    unique_alleles_per_breed = {}
    for breed, alleles in dictofsets.items():
        # added , set()) to union to give null state when no other alleles are present, to avoid error: TypeError: unbound method set.union() needs an argument
        other_alleles = set.union(*(dictofsets[b] for b in dictofsets if b != breed), set()) # union is == | i.e. set1 | set2
        unique_alleles_per_breed[breed] = list(alleles - other_alleles)

    return unique_alleles_per_breed

def identify_breed_specific_motifs_and_alleles(dvariants, ids, sampleIDs, sampleID_to_breed, sampleID_to_cluster, outputdir):
    breedsharedout, breeduniqueout = [], []
    clustersharedout, clusteruniqueout = [], []
    
    for locusID in ids:
        loc = dvariants[locusID]  # locus dataclass

        countambiguous = 0
        # collect info per locus
        breedallele, breedmotif = {}, {}  # key is breed, value is list of allele numbers (which will be made into set) when summarized
        clusterallele, clustermotif = {}, {}

        for sampleID in sampleIDs:
            # per sample collect allele number and motif numbers, find shared/unique numbers, then per locus call the relevant motifs/alleles
            var = loc.samplegenotypes[sampleID]  # variant dataclass per sample for the locus
            if var == '.':
                countambiguous += 1
            else:
                # get breed and assigned cluster information for the sample
                name = sampleID if sampleID[-len('.mm2'):] != '.mm2' else sampleID[:-len('.mm2')]  # fixing fucking ID issue
                breed, cluster = sampleID_to_breed[name], sampleID_to_cluster[name]  # name should be sampleID

                # collect all allele indexes per group
                breedallele.setdefault(breed, []).extend(var.genotype.split('/'))
                clusterallele.setdefault(cluster, []).extend(var.genotype.split('/'))

                # collect all motif indexes per group
                if var.ploidy == 'haploid':
                    # collect all motif indexes, i.e. the number of effmotifs actually utilized by the genotyper (per variant)
                    motifsused = [str(index) for index, motifcount in enumerate(var.al1motifcount) if motifcount > 0]
                elif var.ploidy == 'diploid':
                    al1motifsused = [str(index) for index, motifcount in enumerate(var.al1motifcount) if motifcount > 0]
                    al2motifsused = [str(index) for index, motifcount in enumerate(var.al2motifcount) if motifcount > 0]
                    motifsused = al1motifsused + al2motifsused
                breedmotif.setdefault(breed, []).extend(motifsused)
                clustermotif.setdefault(cluster, []).extend(motifsused)

        # summarize allele/motif usage per locus and decide to output
        # if unique allele, what is unique about it? A unique motif? unique length? Unique motif order?
        breedallelesets = {breed: set(alleles) for breed, alleles in breedallele.items()}
        clusterallelesets = {cluster: set(alleles) for cluster, alleles in clusterallele.items()}
        breedmotifsets = {breed: set(motifs) for breed, motifs in breedmotif.items()}
        clustermotifsets = {cluster: set(motifs) for cluster, motifs in clustermotif.items()}

        # unique alleles/motifs for each breed/cluster group compared to all other groups
        uniquebreedalleles = unique_set_elements_per_group(breedallelesets)
        uniqueclusteralleles = unique_set_elements_per_group(clusterallelesets)
        uniquebreedmotifs = unique_set_elements_per_group(breedmotifsets)
        uniqueclustermotifs = unique_set_elements_per_group(clustermotifsets)

        # shared alleles across all breeds or clusters
        breedsharedalleles = list(set.intersection(*breedallelesets.values())) if breedallelesets else []
        clustersharedalleles = list(set.intersection(*clusterallelesets.values())) if clusterallelesets else []
        # shared motifs across all breeds or clusters
        breedsharedmotifs = list(set.intersection(*breedmotifsets.values())) if breedmotifsets else []
        clustersharedmotifs = list(set.intersection(*clustermotifsets.values())) if clustermotifsets else []

        nummotifsused = len(set([i for almotifcount in loc.allelemc for i, mc in enumerate(almotifcount.split('_')) if int(mc) > 0]))
        # collect data for output
        # format shared motifs & alleles shared across all groups
        breeds = list(breedallelesets.keys())
        clusters = list(clusterallelesets.keys())
        breedsharedout.append([locusID, 'VNTR' if len(loc.alleles) > 1 else 'TR', str(len(loc.motifs)), str(nummotifsused), str(len(loc.alleles)), ','.join(breeds) if breeds else 'NA', ','.join(breedsharedmotifs) if breedsharedmotifs else 'NA', ','.join(breedsharedalleles) if breedsharedalleles else 'NA'])
        clustersharedout.append([locusID, 'VNTR' if len(loc.alleles) > 1 else 'TR', str(len(loc.motifs)), str(nummotifsused), str(len(loc.alleles)), ','.join(clusters) if clusters else 'NA', ','.join(clustersharedmotifs) if clustersharedmotifs else 'NA', ','.join(clustersharedalleles) if clustersharedalleles else 'NA'])
        # format each unique motif/allele per group (1 result per breed per locus)
        for breed, uniquemotifs, in uniquebreedmotifs.items():
            uniquealleles = uniquebreedalleles[breed]
            breeduniqueout.append([locusID, 'VNTR' if len(loc.alleles) > 1 else 'TR', str(len(loc.motifs)), str(nummotifsused), str(len(loc.alleles)), breed, ','.join(uniquemotifs) if uniquemotifs else 'NA', ','.join(uniquealleles) if uniquealleles else 'NA'])  # ','.join([loc.motifs[motifindex] for motifindex in uniquemotifs]) if uniquemotifs else 'NA', ','.join([loc.alleles[alleleindex] for alleleindex in uniquealleles]) if uniquealleles else 'NA'
        for cluster, uniquemotifs, in uniqueclustermotifs.items():
            uniquealleles = uniqueclusteralleles[cluster]
            clusteruniqueout.append([locusID, 'VNTR' if len(loc.alleles) > 1 else 'TR', str(len(loc.motifs)), str(nummotifsused), str(len(loc.alleles)), cluster, ','.join(uniquemotifs) if uniquemotifs else 'NA', ','.join(uniquealleles) if uniquealleles else 'NA'])  # # ','.join([loc.motifs[motifindex] for motifindex in uniquemotifs]) if uniquemotifs else 'NA', ','.join([loc.alleles[alleleindex] for alleleindex in uniquealleles]) if uniquealleles else 'NA'
    
    header= [['LocusID', 'TRorVNTR', 'NumMotifs', 'NumMotifsUsed', 'NumAlleles', 'Breeds', 'SharedMotifsIndices', 'SharedAlleleIndices', 'SharedMotifSeqs', 'SharedAlleleSeqs']]
    outfile = os.path.join(outputdir, 'SharedMotifsAndAlleles_ByBreed.tsv')
    breedsharedout = ['\t'.join(i) for i in header + breedsharedout]
    write_out(breedsharedout, outfile, gz=True)

    header= [['LocusID', 'TRorVNTR', 'NumMotifs', 'NumMotifsUsed', 'NumAlleles', 'Clusters', 'SharedMotifsIndices', 'SharedAlleleIndices', 'SharedMotifSeqs', 'SharedAlleleSeqs']]
    outfile = os.path.join(outputdir, 'SharedMotifsAndAlleles_ByCluster.tsv')
    clustersharedout = ['\t'.join(i) for i in header + clustersharedout]
    write_out(clustersharedout, outfile, gz=True)

    headerbreed = [['LocusID', 'TRorVNTR', 'NumMotifs', 'NumMotifsUsed', 'NumAlleles', 'Breed', 'UniqueMotifsIndices', 'UniqueAlleleIndices', 'UniqueMotifSeqs', 'UniqueAlleleSeqs']]
    outfile = os.path.join(outputdir, 'UniqueMotifsAndAlleles_ByBreed.tsv')
    breeduniqueout = ['\t'.join(i) for i in headerbreed + breeduniqueout]
    write_out(breeduniqueout, outfile, gz=True)

    header = [['LocusID', 'TRorVNTR', 'NumMotifs', 'NumMotifsUsed', 'NumAlleles', 'Cluster', 'UniqueMotifsIndices', 'UniqueAlleleIndices', 'UniqueMotifSeqs', 'UniqueAlleleSeqs']]
    outfile = os.path.join(outputdir, 'UniqueMotifsAndAlleles_ByCluster.tsv')
    clusteruniqueout = ['\t'.join(i) for i in header + clusteruniqueout]
    write_out(clusteruniqueout, outfile, gz=True)

    del breedsharedout, clustersharedout, breeduniqueout, clusteruniqueout

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
        #print(f'id_index={id_index}, breed_index={breed_index}')
    #print(f'data={data}')

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
            # name = label if label[-len('.mm2'):] != '.mm2' else label[:-len('.mm2')]  # fixing fucking ID issue
            sampleID_to_cluster[str(label)] = str(cluster)
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


def save_labeled_matrix(matrix, row_labels, col_labels, filename):
    import numpy as np

    # Ensure row_labels and col_labels have the correct length
    assert len(row_labels) == matrix.shape[0], "Number of row labels must match the number of rows in the matrix."
    assert len(col_labels) == matrix.shape[1], "Number of column labels must match the number of columns in the matrix."
    
    # Add column labels as the first row, preceded by an empty space for row labels
    header = "," + ",".join(map(str, col_labels))  # Convert column labels to a comma-separated string
    
    # Add row labels to each row of the matrix
    labeled_matrix = np.column_stack((row_labels, matrix))  # Stack row labels as the first column
    
    # Save to file with the header
    np.savetxt(filename, labeled_matrix, delimiter=",", fmt="%s", header=header, comments="")

    print(f"Matrix saved to {filename}")


def calculate_similarity(list1, list2):
    """
    Calculate similarity between two lists of two elements efficiently.
    
    Args:
        list1 (list): The first list of two elements.
        list2 (list): The second list of two elements.
    
    Returns:
        float: 0.0 if no shared elements, 0.5 if one shared element, 1.0 if two shared elements.
    """
    # sim_value = calculate_similarity([var1.al1, var1.al2], [var2.al1, var2.al2])  # function call slower

    # Count occurrences of elements in list2
    counter2 = Counter(list2)
    shared_count = 0

    # Count shared elements between list1 and list2
    for elem in list1:
        if counter2[elem] > 0:
            shared_count += 1
            counter2[elem] -= 1  # Decrease count to prevent double counting

    # Normalize to get similarity score (0, 0.5, or 1)
    return shared_count / 2.0


def make_similarity_matrix(dvariants, ids, sampleIDs, outputdir):
    # Initialize similarity and denominator matrices
    num_samples = len(sampleIDs)
    # Initialize the matrix with zeros
    matrix = np.zeros((len(sampleIDs), len(sampleIDs)))
    matrix_denominator = np.zeros((len(sampleIDs), len(sampleIDs)))
    print(f'matrix shape: {matrix.shape}')
    print(f'matrix denominator shape: {matrix_denominator.shape}')
    
    locicount = 0.0

    for locusID in ids:
        loc = dvariants[locusID]
        gate = 0  # Indicates if locus contributes to similarity

        # Extract genotypes once
        genotypes = {sample: loc.samplegenotypes[sample] for sample in sampleIDs}

        for row in range(num_samples):
            sample1 = sampleIDs[row]
            var1 = genotypes[sample1]
            if var1 == '.' or var1.ploidy not in ('haploid', 'diploid'):
                continue  # Skip missing or invalid genotypes

            for col in range(row, num_samples):
                sample2 = sampleIDs[col]
                var2 = genotypes[sample2]
                if var2 == '.' or var2.ploidy not in ('haploid', 'diploid'):
                    continue  # Skip missing or invalid genotypes
                
                gate = 1  # if it makes it past missing data filters then count

                if var1.ploidy == 'diploid' and var2.ploidy == 'diploid':
                    # Diploid-Diploid similarity
                    # Count occurrences of elements in list2
                    counter2 = Counter([var2.al1, var2.al2])
                    sim_value = 0
                    # Count shared elements between list1 and list2
                    for elem in [var1.al1, var1.al2]:
                        if counter2[elem] > 0:
                            sim_value += 0.5  # for diploid get similarity score (0, 0.5, 1)
                            counter2[elem] -= 1  # Decrease count to prevent double counting
                    matrix_denominator[row, col] += 1.0
                elif var1.ploidy == 'haploid' and var2.ploidy == 'haploid':
                    # Haploid-Haploid similarity
                    sim_value = 0.5 if var1.al1 == var2.al1 else 0.0
                    matrix_denominator[row, col] += 0.5  # 1.0 or 0.5  # 0.5 allows hap-hap comparisons to normalize to 1(max), 1.0 allows hap-hap to normlize to 0.5(max)
                else:
                    # Haploid-Diploid or Diploid-Haploid similarity
                    haploid_allele = var1.al1 if var1.ploidy == 'haploid' else var2.al1
                    diploid_alleles = {var1.al1, var1.al2} if var1.ploidy == 'diploid' else {var2.al1, var2.al2}
                    sim_value = 0.5 if haploid_allele in diploid_alleles else 0.0
                    matrix_denominator[row, col] += 0.5  # 1, 0.5 or maybe even 0.75
                    # 1.0 dip-hap/hap-dip comparisons will normlize to 0.5(max), 0.5 will normalize to 1.0(max)

                if sim_value > 0:
                    matrix[row, col] += sim_value

        if gate == 1:
            locicount += 1

    # Ensure symmetry by copying upper triangle to lower triangle
    for i in range(num_samples):
       for j in range(i+1, num_samples):  # Only loop over upper triangle
           if i != j: matrix[j, i] = matrix[i, j]  # Mirror similarity values
           if i != j: matrix_denominator[j, i] = matrix_denominator[i, j]  # Mirror denominator values

    matrix_denominator[matrix_denominator == 0] = 1  # Avoid division by zero

    if locicount == 0:
        raise ValueError("Division by zero: No valid loci found in similarity matrix.")

    # Normalize matrix
    normmatrix = matrix / matrix_denominator
    print(f'Normalized matrix shape: {normmatrix.shape}')
    print(f'Maximum number of loci where genotypes were compared: {locicount}')

    # Fix sample IDs (string operations outside loops)
    fixedIDs = [name.replace('.mm2', '') if name.endswith('.mm2') else name for name in sampleIDs]

    # Save matrices
    save_labeled_matrix(matrix, fixedIDs, fixedIDs, os.path.join(outputdir, f'SimilarityMatrixRaw_LociCount{int(locicount)}.csv'))
    save_labeled_matrix(matrix_denominator, fixedIDs, fixedIDs, os.path.join(outputdir, f'SimilarityMatrixRaw_DenominatorCount{int(locicount)}.csv'))
    outfile = os.path.join(outputdir, 'SimilarityMatrix_SharedAlleleProportions.csv')
    save_labeled_matrix(normmatrix, fixedIDs, fixedIDs, outfile)

    # Print diagonal entries for debugging
    for i, sampleID in enumerate(sampleIDs):
        print(f"Diagonal entry for {sampleID}: {normmatrix[i, i]}")

    return normmatrix


###def make_similarity_matrix(dvariants, ids, sampleIDs, outputdir):
###    import os
###    import numpy as np
###    from collections import Counter
###
###    # Initialize the matrix with zeros
###    matrix = np.zeros((len(sampleIDs), len(sampleIDs)))
###    # make a matrix with all maximum loci similarity values per cell
###    matrix_denominator = np.zeros((len(sampleIDs), len(sampleIDs)))
###    
###    # diagonal no longer = 1, because locuscount increases if even one comparison between samplex and sampley succeeds. Therefore 1-diagonal value should equal proportion of loci with ambiguous genotypes
###    # additionally the maximum diagonal value for haploid genotypes is 0.5. this compensates for dosage
###    locicount = 0.0
###    for locusID in ids:
###        loc = dvariants[locusID]  # locus dataclass
###        gate = 0  # opens to 1 for a locus if there is a sapleROW to sampleCOL comparison found, i.e. not all missing data for a locus
###        for row, sampleID1 in enumerate(sampleIDs):
###            var1 = loc.samplegenotypes[sampleID1]
###            for col in range(row, len(sampleIDs)):
###                sampleID2 = sampleIDs[col]
###                var2 = loc.samplegenotypes[sampleID2]
###                if var1 == '.':
###                    # if ambiguous genotype do not add any similarity to anything
###                    # all comparisons for this sample will be amig, so add values to entire row of missing matrix, no need to mirror
###                    pass
###                elif var1.ploidy == 'diploid':  # could have moved the if statement inside inner loop but would slow down script. assuming all genotypes as same locus are diploid or haploid
###                    if var2 == '.':
###                        # if ambiguous genotype do not add any similarity to anything
###                        pass
###                    elif var2.ploidy == 'diploid':
###                        ##### calculate similarity #####
###                        # Count occurrences of elements in list2
###                        counter2 = Counter([var2.al1, var2.al2])
###                        sim_value, gate = 0, 1
###                        matrix_denominator[row, col] += 1.0
###                        if row != col: matrix_denominator[col, row] += 1.0  # Mirror lower triangle to upper triangle for symmetry # but do not add the diagonal, because it is already present
###                        # Count shared elements between list1 and list2
###                        for elem in [var1.al1, var1.al2]:
###                            if counter2[elem] > 0:
###                                sim_value += 0.5  # for diploid get similarity score (0, 0.5, 1)
###                                counter2[elem] -= 1  # Decrease count to prevent double counting
###                        ##### end similarity calculation #####
###                        matrix[row, col] += sim_value
###                        if row != col: matrix[col, row] += sim_value  # Mirror lower triangle to upper triangle for symmetry # but do not add the diagonal, because it is already present
###                    elif var2.ploidy == 'haploid':
###                        ##### calculate similarity #####
###                        sim_value, gate = 0, 1
###                        matrix_denominator[row, col] += 0.5
###                        if row != col: matrix_denominator[col, row] += 0.5  # Mirror lower triangle to upper triangle for symmetry # but do not add the diagonal, because it is already present
###                        # see if haploid var2.al1 matches either var1.al1/2
###                        for elem in [var1.al1, var1.al2]:
###                            if var2.al1 == elem:
###                                sim_value += 0.5  # get similarity score (0, 0.5)
###                                break  # only count 1 match at max because only 1 allele from haploid
###                        ##### end similarity calculation #####
###                        matrix[row, col] += sim_value
###                        if row != col: matrix[col, row] += sim_value  # Mirror lower triangle to upper triangle for symmetry # but do not add the diagonal, because it is already present
###                elif var1.ploidy == 'haploid':
###                    if var2 == '.':
###                        # if ambiguous genotype add to denominator but do not
###                        pass
###                    elif var2.ploidy == 'haploid':
###                        ##### calculate similarity #####
###                        sim_value, gate = 0, 1
###                        matrix_denominator[row, col] += 0.5
###                        if row != col: matrix_denominator[col, row] += 0.5  # Mirror lower triangle to upper triangle for symmetry # but do not add the diagonal, because it is already present
###                        if var1.al1 == var2.al1:  # only 1 allele in haploid genotypes, so see if they are the same
###                            sim_value += 0.5  # for haploid get 0 or 0.5
###                        ##### end similarity calculation #####
###                        matrix[row, col] += sim_value
###                        if row != col: matrix[col, row] += sim_value  # Mirror lower triangle to upper triangle for symmetry # but do not add the diagonal, because it is already present
###                    elif var2.ploidy == 'diploid':
###                        ##### calculate similarity #####
###                        sim_value, gate = 0, 1
###                        matrix_denominator[row, col] += 0.5
###                        if row != col: matrix_denominator[col, row] += 0.5  # Mirror lower triangle to upper triangle for symmetry # but do not add the diagonal, because it is already present
###                        # see if haploid var1.al1 matches either var2.al1/2
###                        for elem in [var2.al1, var2.al2]:
###                            if var1.al1 == elem:
###                                sim_value += 0.5  # get similarity score (0, 0.5)
###                                break  # only count 1 match at max because only 1 allele from haploid
###                        ##### end similarity calculation #####
###                        matrix[row, col] += sim_value
###                        if row != col: matrix[col, row] += sim_value  # Mirror lower triangle to upper triangle for symmetry # but do not add the diagonal, because it is already present
###        if gate == 1:
###            locicount += 1
###
###    # replace all zeros by 1 to avoid division by zero (can happen for females and y chromosome)
###    matrix_denominator[matrix_denominator == 0] = 1
###
###    # divide each cell by total loci count that have no ambiguous genotypes
###    if locicount == 0.0:
###        raise ValueError("Division by zero is not allowed. No loci included in similarity matrix. Possibly all ambiguous genotypes")
###    
###    # output raw matrix of similarity counts # this can be summed with other matrices from other chromosomes, then normalized in the end to get genome wide
###    fixedIDs = [name if name[-len('.mm2'):] != '.mm2' else name[:-len('.mm2')] for name in sampleIDs]  # fix some ID issues # from messy input naming
###    outfile = os.path.join(outputdir, f'SimilarityMatrixRaw_LociCount{int(locicount)}.csv')
###    save_labeled_matrix(matrix, fixedIDs, fixedIDs, outfile)
###    # output raw matrix of similarity counts # this can be summed with other matrices from other chromosomes, then normalized in the end to get genome wide
###    outfile = os.path.join(outputdir, f'SimilarityMatrixRaw_DenominatorCount{int(locicount)}.csv')
###    save_labeled_matrix(matrix_denominator, fixedIDs, fixedIDs, outfile)
###
###    # Perform element-wise division # return matrix with values 0-1
###    # normmatrix = matrix / locicount
###    normmatrix = matrix / matrix_denominator
###
###    # Diagnostic print for self-comparison entries
###    for i, sampleID in enumerate(sampleIDs):
###        print(f"Diagonal entry for {sampleID}: {normmatrix[i, i]}")
###
###    return normmatrix



def show_stats(nogenotypecount, genotypecounts, allelepurity, noallpurecount, allelelendiff):

    # print('\nGenotype Stats:')
    total = sum([v for v in genotypecounts.values()]) + nogenotypecount
    x = [f'{k}: count={v}, percentage={round((v / total) * 100, 2)}' for k, v in natsorted(genotypecounts.items())]  # ': '.join([k, str(v]))
    x = x + [f'./.: count={nogenotypecount}, percentage={round((nogenotypecount / total) * 100, 2)}']

    # print('\nAllele Purity Stats:')
    # print([f'{k}, {v[:10]}' for k, v in allelepurity.items()])
    y = [f'{k}: count={len(v)}, mean={round(st.mean(v), 2)}, median={round(st.median(v), 2)}, stdev={round(st.stdev(v), 2)}, quantiles={[round(q, 2) for q in st.quantiles(v, n=5)]},  max={round(max(v), 2)}, min={round(min(v), 2)}' \
         if len(v) >= 2 else f'{k}: count={len(v)}, mean={round(v[0], 2)}, median="NA", stdev="NA", quantiles="NA",  max="NA", min="NA"' for k, v in natsorted(allelepurity.items())]
    y = y + [f'.,.: count={noallpurecount}']
    # print(f'.,.: count={noallpurecount}')
    # print('\n'.join(y))

    # print('\nAllele Length Difference Stats:')
    # print([f'{k}, {v[:10]}' for k, v in allelepurity.items()])
    z = [f'{k}: count={len(v)}, mean={round(st.mean(v), 2)}, median={round(st.median(v), 2)}, stdev={round(st.stdev(v), 2)}, quantiles={[round(q, 2) for q in st.quantiles(v, n=10)]},  max={round(max(v), 2)}, min={round(min(v), 2)}' \
         if len(v) >= 2 else f'{k}: count={len(v)}, mean={round(v[0], 2)}, median="NA", stdev="NA", quantiles="NA",  max="NA", min="NA"' for k, v in natsorted(allelelendiff.items())]
    # print('\n'.join(z))

    outstats = x + y + z

    return outstats


def calc_stats_per_sample(dvariants, ids, sampleIDs, outputdir):
    import os

    header = [['id', 'genotype', 'sum_al1al2_num_repeats', 'al1num_repeats', 'al2num_repeats', 'sum_al1al2_len', 'al1len', 'al2len', 'allele_length_diff', 'al1depth', 'al2depth', 'al1pure', 'al2pure', 'al1meth', 'al2meth', 'al1CpGDensity', 'al2CpGDensity', 'al1MC', 'al2MC', 'al1seq', 'al2seq']]  # change to match outdata fields
    for sampleID in sampleIDs:  # for each sample
        genotypecounts = {}
        nogenotypecount = 0
        allelepurity = {}
        noallpurecount = 0
        allelelendiff = {}
        outdata = []
        for locusID in ids:  # iterate through all loci
            loc = dvariants[locusID]
            var = loc.samplegenotypes[sampleID]
            if var == '.':
                nogenotypecount+= 1  #ambiguous genotype, i.e. no genotype
            else:  # then var is variant dataclass
                genotypecounts[var.genotype] = genotypecounts.get(var.genotype, 0) + 1
                if var.ploidy == 'diploid':
                    if (var.al1pure == '.') or (var.al2pure == '.'):
                        noallpurecount += 1
                    else:
                        allelepurity.setdefault(var.al1, []).append(float(var.al1pure))
                        allelepurity.setdefault(var.al2, []).append(float(var.al2pure))
                        # allelepurity[var.al1] = allelepurity.get(var.al1, []) + [var.al1pure]  # same shit different hole
                        # allelepurity[var.al2] = allelepurity.get(var.al2, []) + [var.al2pure]
                    allelelendiff.setdefault(var.genotype, []).append(int(var.allele_length_diff()))
                    al1numreps = sum(var.al1motifcount)  # number of motifs in al1
                    al2numreps = sum(var.al2motifcount)  # number of motifs in al2
                    totalnumreps = al1numreps + al2numreps  # number of total motifs in both alleles, i.e. in the individual
                    totallength = var.al1len + var.al2len
                    outdata.append([locusID, var.genotype, str(totalnumreps), str(al1numreps), str(al2numreps), str(totallength), str(var.al1len), str(var.al2len), str(var.allele_length_diff()), str(var.al1depth), str(var.al2depth), var.al1pure, var.al2pure,
                                    var.al1meth, var.al2meth, loc.cpgdensities[int(var.al1)], loc.cpgdensities[int(var.al2)], '_'.join([str(i) for i in var.al1motifcount]), 
                                    '_'.join([str(i) for i in var.al2motifcount]), loc.alleles[int(var.al1)], loc.alleles[int(var.al2)]])
                elif var.ploidy == 'haploid':
                    if var.al1pure == '.':
                        noallpurecount += 1
                    else:
                        allelepurity.setdefault(var.al1, []).append(float(var.al1pure))
                    al1numreps = sum(var.al1motifcount)
                    totalnumreps = al1numreps
                    totallength = var.al1len
                    outdata.append([locusID, var.genotype, str(totalnumreps), str(al1numreps), 'NA', str(totallength), str(var.al1len), 'NA', 'NA', str(var.al1depth), 'NA', var.al1pure, 'NA',
                                    var.al1meth, 'NA', loc.cpgdensities[int(var.al1)], 'NA', '_'.join([str(i) for i in var.al1motifcount]), 
                                    'NA', loc.alleles[int(var.al1)], 'NA'])
        
        outdata = ['\t'.join(i) for i in header + outdata]
        outfile = os.path.join(outputdir, f'TRDB_{sampleID}.data.tsv')
        write_out(outdata, outfile, gz=True)

        # collect basic stats
        outstats = show_stats(nogenotypecount, genotypecounts, allelepurity, noallpurecount, allelelendiff)
        outfile = os.path.join(outputdir, f'TRDB_{sampleID}.stats.txt')
        write_out(outstats, outfile)

    # cleanup
    del outdata, outstats, nogenotypecount, genotypecounts, allelepurity, noallpurecount, allelelendiff

    return


def calc_stats_per_locus(dvariants, ids, sampleIDs, outputdir, dfeatures=None, featureheader=None):   # , **kwargs
    # dvariants is dictionary, key is locusID, value is locus dataclass defined above
    # dvariants[locusID] = locus

    locusstats = []
    
    for locusID in ids:
        # numeffmotifs = len(locus.motifs)
        # numalleles = len(locus.alleles)
        loc = dvariants[locusID]  # locus dataclass

        countambiguous = 0
        dallelecount = {}
        allelecount = []

        ploidy = 'NA'
        for sampleID in sampleIDs:
            var = loc.samplegenotypes[sampleID]  # variant dataclass per sample for the locus
            if var == '.':
                countambiguous += 1
            else:
                dallelecount[var.al1] = dallelecount.get(var.al1, 0) + 1
                if var.ploidy == 'diploid':
                    dallelecount[var.al2] = dallelecount.get(var.al2, 0) + 1

                ploidy = var.ploidy

        sortedallelenumbers = []   
        for allelenumber in sorted([int(i) for i in list(dallelecount.keys())]):  # convert allelnumbers from str to int to sort 'naturally'
            allelecount.append(str(dallelecount[str(allelenumber)]))  # collect the number of times each allele appears in order form allele 0 -> x
            sortedallelenumbers.append(str(allelenumber))  # collect the sorted allele numbers in order from 0 -> x
        refallelepresent = 'RefAlleleAbsent' if '0' not in sortedallelenumbers else 'RefAllelePresent'
        nummotifsused = len(set([i for almotifcount in loc.allelemc for i, mc in enumerate(almotifcount.split('_')) if int(mc) > 0]))  ### loc.allelemc = [[3_0_4_5], [2_0_3_7]] # allelemc already natsorted by allele number
        nummotifsusedperallele = []
        for almotifcount in loc.allelemc:
            nummotifsusedperallele.append(str(len([mc for mc in almotifcount.split('_') if int(mc) > 0])))
        # could make locusstats a dictionary
        # .setdefault(locusID, []).append([loc.id, 'VNTR' if len(loc.alleles) > 1 else 'TR', loc.chrom, str(loc.start), str(loc.end), str(len(loc.motifs)), str(len(loc.alleles)), ','.join(allelecount)])
        # join loc.allelemc by commas because there is a csv string per allele, so commas seperate alleles, and underscores seperate motif counts
        if dfeatures:
            locusstats.append([loc.id, ploidy, 'VNTR' if len(loc.alleles) > 1 else 'TR', loc.chrom, str(loc.start), str(loc.end), str(len(loc.motifs)), str(nummotifsused), str(len(loc.alleles))] + dfeatures[locusID] + [refallelepresent, ','.join(sortedallelenumbers), ','.join(allelecount),
                           ','.join(loc.cpgdensities), ','.join(loc.avgallelemeth), ','.join(loc.avgallelepur), ','.join([str(len(i)) for i in loc.alleles]), ','.join(nummotifsusedperallele), ','.join(loc.allelemc), ','.join(loc.allelems), ','.join(loc.motifs), ','.join(loc.alleles)])
        else:
            locusstats.append([loc.id, ploidy, 'VNTR' if len(loc.alleles) > 1 else 'TR', loc.chrom, str(loc.start), str(loc.end), str(len(loc.motifs)), str(nummotifsused), str(len(loc.alleles)), refallelepresent, ','.join(sortedallelenumbers), ','.join(allelecount),
                           ','.join(loc.cpgdensities), ','.join(loc.avgallelemeth), ','.join(loc.avgallelepur), ','.join([str(len(i)) for i in loc.alleles]), ','.join(nummotifsusedperallele), ','.join(loc.allelemc), ','.join(loc.allelems), ','.join(loc.motifs), ','.join(loc.alleles)])
    
    if dfeatures:
        header = [['LocusID', 'Ploidy', 'TRorVNTR', 'Chrom', 'Start', 'End', 'NumEffMotifs', 'NumEffMotifsUsed', 'NumUniqueAlleles'] + featureheader + ['RefAllelePresentAbsent', 'AlleleIDs', 'CountsPerAllele', 'CpGDensityPerAllele', 'AvgAlleleMethylation', 'AvgAllelePurity', 'Allelelength', 'NumMotifsUsedPerAllele', 'AlleleMotifCounts', 'AlleleMotifSpans', 'Motifs', 'Alleles']]
    else:
        header = [['LocusID', 'Ploidy', 'TRorVNTR', 'Chrom', 'Start', 'End', 'NumEffMotifs', 'NumEffMotifsUsed', 'NumUniqueAlleles', 'RefAllelePresentAbsent', 'AlleleIDs', 'CountsPerAllele', 'CpGDensityPerAllele', 'AvgAlleleMethylation', 'AvgAllelePurity', 'Allelelength', 'NumMotifsUsedPerAllele', 'AlleleMotifCounts', 'AlleleMotifSpans', 'Motifs', 'Alleles']]
    ###DEBUG print(f'{header}\n')
    ###DEBUG for x in locusstats:
    ###DEBUG     try:
    ###DEBUG         '\t'.join(x)
    ###DEBUG     except TypeError:
    ###DEBUG         print(x)
    ###DEBUG     else:
    ###DEBUG         pass
    outdata = ['\t'.join(i) for i in header + locusstats]
    outfile = os.path.join(outputdir, f'TRDB_LocusStats.txt')
    write_out(outdata, outfile, gz=True)


def overlap_stats(start, end, prevstart, prevend):
    ''' calc two-way overlap stats for two ranges: one range start-end, other range prevstart-prevend'''
    overlap = len(range(max(start, prevstart), min(end, prevend) + 1))  # calcs number of overlapping bases
    percentoverlap = (overlap / (end - start + 1)) * 100  # calculate percentage of overlapping bases that overlap current range
    prevpercentoverlap = (overlap / (prevend - prevstart + 1)) * 100  # calculate percentage of overlapping bases that overlap previous range
    # percoverdiff = prevpercentoverlap - percentoverlap  # negative when prevpercover is longer (shorter range) (ex: 20-100) # positive when curent range is longer (100-20)
    # abspercoverdiff = abs(percoverdiff)

    # return overlap, percentoverlap, prevpercentoverlap, percoverdiff, abspercoverdiff
    # percentoverlap = percentage of TR annotation covered by gene feature
    # prevpercentoverlap = percentage of genic feature covered by TR annotation
    return overlap, percentoverlap, prevpercentoverlap


def do_ranges_overlap(start, end, prevstart, prevend):
    if (start <= prevend) and (end >= prevstart):  # if ranges overlap
        return True
    else:
        return False
    

def read_in_feature_table(featuretablefile):
    dannot = {}
    # dannot dicitonary key is chromosome, val is dict, value is list containing lists, inner lists are a gene feature table line
    # dict speeds up searches by chromosome index
    with gzip.open(featuretablefile, 'rt') as FILE:
        for i, line in enumerate(FILE):
            feature = line.strip().split('\t')
            if i == 0: didx = {label: index for index, label in enumerate(feature)}
            if feature[0] == 'gene': dannot.setdefault(feature[didx['chromosome']], []).append(feature)  # {s[didx['symbol']]: s}

    return dannot, didx


def annotate_TRs(locusids, featuretablefile, outputdir):
    import os
    from natsort import natsorted
    # locusids = [1.1567989.1677890, ..., chrom.start.end]
    # dannot dicitonary key is chromosome, val is dict, value is list containing lists, inner lists are a gene feature table line
    # didx is dictionary where key is column name value is corresponding numerical index (didx == dictionary index)
    dannot, didx = read_in_feature_table(featuretablefile)
    # key is id i.e. chrom.start.end
    # # records gene features that overlap TR annotation
    dfeatures = {}  # dict, key is TR ID, val is list of list(s) (if multiple overlaping features) [str(round(TRpercentoverlap, 2)), str(round(Genepercentoverlap, 2))] + feature] of feature table info 
    dfeaturetypecounts = {}  # dict, key is gene class (i.e. lncRNA,pseudogene,protein_coding, etc.), value is count of number of times it is present
    duniquegenes = {}
    for id in locusids:
        gate = 0  # closed
        chrom, start, end = id.split('.')[0], int(id.split('.')[1]), int(id.split('.')[2])
        # find what features overlap TR
        for feature in dannot[chrom]:  # feature is line from feature_table.txt file with basic gene info
            featurechrom, featurestart, featureend = feature[didx['chromosome']], int(feature[didx['start']]), int(feature[didx['end']])
            if chrom == featurechrom and do_ranges_overlap(start, end, featurestart, featureend):
                gate = 1  # opened
                dfeaturetypecounts[feature[didx['class']]] = dfeaturetypecounts.get(feature[didx['class']], 0) + 1  # count number of times each gene class overalps TR features
                duniquegenes.setdefault(f"{feature[didx['class']]}|{feature[didx['symbol']]}", []).append(id) # record which TRs (id) overlap each gene
                # percentoverlap = percentage of TR annotation covered by gene feature
                # prevpercentoverlap/genepercentoverlap = percentage of genic feature covered by TR annotation
                overlap, percentoverlap, genepercentoverlap = overlap_stats(start, end, featurestart, featureend)
                # dfeatures.setdefault(id, []).append([str(round(TRpercentoverlap, 2)), str(round(Genepercentoverlap, 2))] + feature)
                dfeatures.setdefault(id, []).append([feature[didx['class']], feature[didx['symbol']], feature[didx['GeneID']], feature[didx['start']], feature[didx['end']], feature[didx['strand']], feature[didx['product_accession']], feature[didx['related_accession']], str(round(percentoverlap, 2)), str(round(genepercentoverlap, 2)), feature[didx['name']]])
            else:
                if gate == 1:
                    break
        # assume all matching features will overlap continuously, once past, break gene feature loop, and move to next TR locus
        # also handles last matches per chromosome
        if gate == 1:
            if len(dfeatures[id]) > 1:  # if multiple gene features overlap the TR then collapse the features to one list (seperate overlapping results by pipes)
                dfeatures[id] = ['|'.join(i) for i in zip(*dfeatures[id])]  # i = [['4', '5', '6', '7'], ['1', '2', '3', '4']] ['|'.join(n) for n in zip(*i)] -> ['4|1', '5|2', '6|3', '7|4']
                ### print(dfeatures[id])
            elif len(dfeatures[id]) == 1:
                dfeatures[id] = dfeatures[id][0]  # flatten list of list or strings to -> list of strings  # test = [[1, 2, 3, 4]]  test[0] -> [1, 2, 3, 4]
                ### print(dfeatures[id])
    
    featureheader = ['class', 'symbol', 'GeneID', 'GeneStart', 'GeneEnd', 'strand', 'product_accession', 'related_accession', 'TROverlapByGene', 'GeneOverlapByTR', 'name']
    for id in locusids:
        # make empty placeholders for TR annotations that do not overlap gene features.
        if id not in dfeatures: dfeatures.setdefault(id, []).extend(['NA']*len(featureheader))  # {id: ['NA', 'NA', 'NA', ... 'NA']}

    # ouput dfeaturetypecounts to file
    outfile = os.path.join(outputdir, 'CountTRFeatureTypes.txt')
    outlines = [f'{featuretype}: {count}' for featuretype, count in natsorted(dfeaturetypecounts.items())]
    write_out(outlines, outfile)
    # ouput dfeaturetypecounts to file
    outfile = os.path.join(outputdir, 'CountTRsOverlappingGenes.txt')
    outlines = [f"Number of unique gene features: {len(duniquegenes)}"] + ['Gene\tCountTRsOverlappingGene\tTR_IDs'] + [f"{genename}\t{len(TRs)}\t{', '.join(TRs)}" for genename, TRs in natsorted(duniquegenes.items())]
    write_out(outlines, outfile)
    # ouput dfeaturetypecounts to file
    outfile = os.path.join(outputdir, 'ProteinCodingGenes.txt')
    outlines = [f"{genename.split('|')[1]}" for genename, TRs in natsorted(duniquegenes.items()) if genename.split('|')[0] == 'protein_coding']
    write_out(outlines, outfile)

    return dfeatures, dfeaturetypecounts, featureheader


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


def parse_trgt_vcf(infile):
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
                loc = locus(id, l[0], int(l[1]), int(info[1][4:]), alleles, info[2].split(','), cpgdensities, avgallmeth, avgallpur, allelemc, allelems, dsamples)  #END= in info field should be int(info[1][4:]) but END is not present anymore so use end from id as int(id.split('.')[-1])
                dvariants[id] = loc
                if linecount % 10000 == 0:
                    print(f'Parsing locus number: {linecount+1}, locus ID: {id}')
    
    return dvariants, ids, sampleIDs


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
    # Optional boolean arguments to perform statistics
    parser.add_argument(
        "--locus", 
        action="store_true", 
        required=False, 
        help="Perform analysis on a per-locus basis"
    )
    parser.add_argument(
        "--annotate", 
        action="store_true", 
        required=False, 
        help="Annotates TR when they overlap with known genomic features. Features present in _features_table.txt file. Requires --locus"
    )
    parser.add_argument(
        "--sample", 
        action="store_true", 
        required=False, 
        help="Perform analysis on a per-sample basis"
    )
    parser.add_argument(
        "--makeplots", 
        action="store_true", 
        required=False, 
        help="Make scatter plot, dendrogram, and heatmap of assigned clusters to samples"
    )
    # optional string arguements
    parser.add_argument(
        "--metadata", 
        type=str, 
        required=False, 
        help="full path to tab delimited .tsv file with sample information. rows sample ID, columns metadata information."
    )
    #optional int args
    parser.add_argument(
        "--numclusters", 
        type=int, 
        required=False,
        default=None, 
        help="If provided this argument changes the function of threshold. With --numclusters, the threshold must be an integer \
            value and it defines the number of clusters to be found after clustering. When no --numclusters is provided the threshold \
            value can be a float and refers to the cophenetic distance to define clusters."
    )


    # # Optional arguments
    # parser.add_argument(
    #     "--locus", 
    #     type=str, 
    #     help="Locus ID (e.g., '1.158968.162134')"
    # )
    # parser.add_argument(
    #     "--sample", 
    #     type=str, 
    #     help="ID name (e.g., 'BSWCHEF120082542037')"
    # )

    return parser.parse_args()


if __name__ == "__main__":
    import os
    import sys
    import argparse

    # program is memory inefficient, recommended to breakup .merged.vcf.gz by chromosome.
    # ex cmd: python ParseTRGTMergedVCF.py --locus --sample --outdir Y --infile /path/to/merged.vcf.gz

    args = parse_arguments()

    # Extracting arguments
    infile = args.infile
    outdir = args.outdir
    threshold = args.threshold
    num_clusters = args.numclusters
    locusanalysis = args.locus
    annotatefeatures = args.annotate
    sampleanalysis = args.sample
    makeplots = args.makeplots
    metadata = args.metadata

    # Display the input arguments
    print(f"Input file: {infile}")
    if locusanalysis:
        print("Will perform per-locus analysis.")
    if sampleanalysis:
        print("Will perform per-sample analysis.")
        if makeplots:
            print("Will make plots of samples")

    # Process the input merged .VCF file
    # ids is list of locus IDs
    dvariants, ids, sampleIDs = parse_trgt_vcf(infile)
    print(f'Finished parsing merged compressed VCF file: {infile}')
    
    path, infilename = os.path.split(infile)
    outputdir = os.path.join(path, outdir)# infilename.split('.')[1])  # make subdirectories for each chromosome (assuming the merged .vcf was split with ParseTRGTMergedVCF_byChrom.py)
    # ensure output directory exists
    os.makedirs(outputdir, exist_ok=True)

    if locusanalysis:
        if annotatefeatures:
            print("Starting TR locus annotation...")
            featuretablefilelist=glob.glob('./*_feature_table.txt.gz')
            assert len(featuretablefilelist) == 1, f"There are {len(featuretablefilelist)} files with extension _feature_table.txt.gz in the current working directory. There should only be 1."
            featuretablefile = featuretablefilelist[0]
            dfeatures, dfeaturetypecounts, featureheader = annotate_TRs(ids, featuretablefile, outputdir)
            print("Finished TR annotations.")

        print("Starting per-locus analysis...")
        if annotatefeatures:
            calc_stats_per_locus(dvariants, ids, sampleIDs, outputdir, dfeatures, featureheader)
        else:
            calc_stats_per_locus(dvariants, ids, sampleIDs, outputdir)
        print("Finished per-locus analysis.")

    if sampleanalysis:
        print("Starting per-sample analysis...")

        outfile = os.path.join(outputdir, 'SampleStats.txt')
        calc_stats_per_sample(dvariants, ids, sampleIDs, outputdir)

        print("Start similarity matrix analysis...")
        normmatrix = make_similarity_matrix(dvariants, ids, sampleIDs, outputdir)

        # outfile = os.path.join(outputdir, 'SimilarityMatrix.csv')
        fixedIDs = [name if name[-len('.mm2'):] != '.mm2' else name[:-len('.mm2')] for name in sampleIDs]  # fix some ID issues # from messy input naming
        # save_labeled_matrix(normmatrix, fixedIDs, fixedIDs, outfile)

        outfile = os.path.join(outputdir, 'ClusterAssignments.csv')
        linkage_outfile = os.path.join(outputdir, 'LinkageMatrix.csv')
        linkage_matrix, cluster_assignments, sorted_matrix, sorted_labels, sorted_clusters, sampleID_to_cluster, threshold = cluster_and_assign_group(normmatrix, fixedIDs, threshold, outfile, linkage_outfile, num_clusters)


        if metadata:
            outfile_summary = os.path.join(outputdir, 'BreedVSClusterAssignment_Summary.tsv')
            outfile_misclassified = os.path.join(outputdir, 'BreedVSClusterAssignment_Misclassified.tsv')
            breedgroups, sampleID_to_breed  = sort_sampleinfo(fixedIDs, metadata)
            # sorted_breedgroups = sort_sampleinfo(sorted_labels, metadata)  # could be relevant in some scenarios... instead of line above
            evaluate_cluster_predictions(fixedIDs, cluster_assignments, breedgroups, outfile_summary, outfile_misclassified)
            identify_breed_specific_motifs_and_alleles(dvariants, ids, sampleIDs, sampleID_to_breed, sampleID_to_cluster, outputdir)

        if makeplots:
            print("Starting to make plots...")
            outfilebasename = os.path.join(outputdir, 'SampleClusters')
            plot_clusters_pca(sorted_matrix, sorted_labels, sorted_clusters, outfilebasename)  # make dendrogram and scatter plot of samples colored by assigned cluster group

            outfile = os.path.join(outputdir, 'Dendrogram.pdf')
            plot_colored_dendrogram(linkage_matrix, fixedIDs, cluster_assignments, float(threshold), outfile)
            if metadata:
                outfile = os.path.join(outputdir, 'Dendrogram_ColoredByBreed.pdf')
                plot_colored_dendrogram_breeds(linkage_matrix, fixedIDs, breedgroups, float(threshold), outfile)

            outfile = os.path.join(outputdir, 'MatrixHeatmap.pdf')
            plot_heatmap(sorted_matrix, sorted_labels, outfile)  # make heatmap of the rotated, and sorted upper triangle of symmetric similarity matrix
        print("Finished per-sample analysis.")





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
