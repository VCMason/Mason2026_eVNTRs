### # Function to process the file
### def process_file(file_path, bins):
###     import pandas as pd
###     import matplotlib.pyplot as plt
###     import os
### 
###     # Read the TSV file into a DataFrame
###     if file_path.endswith('.gz'):
###         # Extract the base filename without the extension
###         base_name = os.path.basename(file_path).replace('.tsv.gz', '')
###         df = pd.read_csv(file_path, sep="\t", header=0, compression='gzip')
###     else:
###         base_name = os.path.basename(file_path).replace('.tsv', '')
###         df = pd.read_csv(file_path, sep="\t", header=0)
###     
###     # Convert relevant columns to numeric, forcing errors to NaN and then dropping/cleaning
###     for col in ['Start', 'End']:  # , 'NumEffMotifs', 'NumEffMotifsUsed', 'NumUniqueAlleles'
###         df[col] = pd.to_numeric(df[col], errors='coerce')
### 
###     # Calculate lengths
###     df['length'] = df['End'] - df['Start']
###     
###     # Statistics
###     mean_length = df['length'].mean()
###     median_length = df['length'].median()
###     std_length = df['length'].std()
###     max_length = df['length'].max()
###     min_length = df['length'].min()
### 
###     print(f"Mean length: {mean_length}")
###     print(f"Median length: {median_length}")
###     print(f"Standard deviation of length: {std_length}")
###     print(f"Min and Max of length: {min_length}, {max_length}")
### 
###     # Plot histogram
###     plt.figure(figsize=(10, 6))
###     plt.hist(df['length'], bins=bins, color='blue', edgecolor='black')
###     plt.title("Histogram of Lengths")
###     plt.xlabel("Length")
###     plt.ylabel("Frequency")
###     plt.grid(axis='y', alpha=0.75)
###     
###     # Save as PDF with a customized filename
###     pdf_filename = f"{base_name}.LengthHistogram.pdf"
###     plt.savefig(pdf_filename, format='pdf')
###     print(f"Histogram saved as {pdf_filename}")
###     
###     # Show the plot
###     plt.show()


def broken_histogram_figure(
    data,
    break_left=-1.0,
    break_right=1.0,
    bins=50,
    output_file=None,
    color='darkorange',
    edgecolor='black',
    xlabel='',
    ylabel='',
    title='',
    figsize=(9, 4),
    width_ratios=(1.2, 2.5, 1.2)
    ):
    """
    Three-panel histogram:
    left tail | center | right tail
    Independent y-axes.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    data = np.asarray(data)

    # Split data
    left_data   = data[data < break_left]
    center_data = data[(data >= break_left) & (data <= break_right)]
    right_data  = data[data > break_right]

    fig, (ax_left, ax_center, ax_right) = plt.subplots(
        1, 3,
        figsize=figsize,
        gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.05}
    )

    hist_kwargs = dict(bins=bins, color=color, edgecolor=edgecolor)
    hist_kwargs_sides = dict(bins=int(bins/2), color=color, edgecolor=edgecolor)

    # Plot subsets
    if len(left_data) > 0:
        ax_left.hist(left_data, **hist_kwargs_sides)
        ax_left.set_xlim(left_data.min(), break_left)
        ax_left.legend([f"{len(left_data)} < {break_left}"], loc='upper left', fontsize=10)

    if len(center_data) > 0:
        ax_center.hist(center_data, **hist_kwargs)
        ax_center.set_xlim(break_left, break_right)
        ax_center.legend([f"{len(center_data)} >= {break_left} and <={break_right}"], loc='upper left', fontsize=10)

    if len(right_data) > 0:
        ax_right.hist(right_data, **hist_kwargs_sides)
        ax_right.set_xlim(break_right, right_data.max())
        ax_right.legend([f"{len(right_data)} > {break_right}"], loc='upper left', fontsize=10)

    # Labels
    ax_left.set_ylabel(ylabel)
    ax_center.set_xlabel(xlabel)
    ax_center.set_title(title)

    # Remove x-axis labels from tails
    ax_left.set_xlabel("")
    ax_right.set_xlabel("")
    #Remove y-axis labels from tails
    ax_left.set_ylabel("")
    ax_right.set_ylabel("")

    # Remove touching spines
    ax_left.spines['right'].set_visible(False)
    ax_center.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)

    # ---- Center panel formatting ----
    # Keep spine and ticks on left
    ax_center.yaxis.set_ticks_position('left')

    # Keep labels on left, but move them inside
    ax_center.tick_params(axis='y',
                          labelleft=True,
                          labelright=False,
                          pad=-10)   # negative = move inside plot

    # Align numbers so they read nicely inside
    for tick in ax_center.yaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('left')

    # ---- Right panel formatting ----
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")

    # Save or show
    if output_file is not None:
        base, _ = os.path.splitext(output_file)
        fig.savefig(f"{base}.pdf", bbox_inches='tight')
        fig.savefig(f"{base}.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved as {base}.pdf and {base}.png")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax_left, ax_center, ax_right


def process_file(file_path, bins):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    if file_path.endswith('.gz'):
        df = pd.read_csv(file_path, sep="\t", compression='gzip')
    else:
        df = pd.read_csv(file_path, sep="\t")

    # Clean and prepare data
    df['NumUniqueAlleles'] = pd.to_numeric(df['NumUniqueAlleles'], errors='coerce')
    df['Motifs'] = df['Motifs'].astype(str)
    df['MotifLen'] = df['Motifs'].apply(lambda x: np.mean([len(m) for m in x.split(',') if m != '']))

    print(df.head())
    print(df.columns)
    print(df.describe())
    print(df['AllLen_Median'].describe())
    print(df['MajorAlleleFrequency'].describe())
    print(df['MinorAlleleFrequency'].describe())
    print((df['MajorAlleleFrequency'] >= 0.95).sum(), "alleles with major allele frequency >= 0.95")
    print((df['MinorAlleleFrequency'] <= 0.05).sum(), "alleles with minor allele frequency <= 0.05")

    # Compute major allele frequency
    major_freqs = []
    repeat_diffs = []
    for idx, row in df.iterrows():
        counts = [int(x) for x in row['CountsPerAllele'].split(',')]
        lengths = [int(x) for x in row['Allelelength'].split(',')] if df['RefAllelePresentAbsent'][idx] == 'RefAllelePresent' else [int(x) for x in row['Allelelength'].split(',')[1:]]
        if len(counts) != len(lengths):
            print(f"Skipping row {idx} due to mismatched counts and lengths: {counts} vs {lengths}. It means the data are malformed.")
            continue
        total = sum(counts)
        if total == 0:
            continue
        major_idx = np.argmax(counts)
        major_all_length = lengths[major_idx]
        if (counts[major_idx] / total < 1.0):  # exclude TRs were there are no alternative alleles (making the major allele frequency < 1.0)
            major_freqs.append(counts[major_idx] / total)
        # if a site has one allele that is not the same as the major allele, then we calculate the repeat difference, but this does not gaurantee that the allele is a different length
        repeat_diffs.extend([np.log2(l / major_all_length) for l in lengths])  # - (lengths[major_idx] / major_all_length) 
        # below line, includes only alleles with different lengths from the major allele
        # repeat_diffs.extend([np.log2(l / major_all_length) for l in lengths if l != major_all_length])  # - (lengths[major_idx] / major_all_length) 

    print(f"Number of repeat differences: {len(repeat_diffs)}")
    # repeat_diffs = [x for x in repeat_diffs if (x <= 5) and (x >= -5)]  # Filter out extreme values
    # Start figure
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Panel (a): Histogram of VNTR counts by allele number
    axs[0, 0].hist(df['NumUniqueAlleles'].dropna(), bins=range(1, 100), color='teal', edgecolor='black')
    axs[0, 0].set_xlabel('Number of Unique Alleles')
    axs[0, 0].set_ylabel('VNTR count')
    axs[0, 0].set_title('(a)')

    # Panel (b): Boxplot of allele number vs STR motif length
    sns.histplot(data=df, x='MotifLen', y='NumUniqueAlleles', bins=100, thresh=0, pmax= 0.025, cmap='crest', cbar=True, ax=axs[0, 1])  # pthresh=0.0 # 'flare' 'mako'
    axs[0, 1].set_xlabel('VNTR motif length (bp)')
    axs[0, 1].set_ylabel('Number of Unique Alleles')
    axs[0, 1].set_title('(b)')

    # Panel (c): Histogram of major allele frequency
    axs[1, 0].hist(major_freqs, bins=100, color='steelblue', edgecolor='black')
    axs[1, 0].set_xlabel('Major allele frequency')
    axs[1, 0].set_ylabel('VNTR count')
    axs[1, 0].set_title('(c)')

    # Panel (d): Histogram of repeat differences from major allele
    # print(f"Number of repeat differences calculated: {len(repeat_diffs)}")
    axs[1, 1].hist(repeat_diffs, bins=100, color='darkorange', edgecolor='black')
    axs[1, 1].set_xlabel('log2(Allele Length / Major Allele Length)')  # Proportional length change from major allele
    axs[1, 1].set_ylabel('VNTR count')
    axs[1, 1].set_xlim(-1.5, 1.5)
    axs[1, 1].set_title('(d) Proportional length changes of alleles from major allele')

    sns.histplot(data=df, x='AllLen_Median', y='NumUniqueAlleles', bins=100, thresh=0, pmax=0.025, cmap='crest', cbar=True, ax=axs[2, 0])  # pthresh=0.0 # 'flare' 'mako'
    # axs[2, 0].plot(df['AllLen_Median'].dropna(), df['NumUniqueAlleles'].dropna())
    axs[2, 0].set_xlabel('Allele Length (median)')
    axs[2, 0].set_ylabel('Number of Unique Alleles')
    axs[2, 0].set_title('(e)')

    axs[2, 1].scatter(df['AllLen_Median'], df['NumUniqueAlleles'], color='blue', alpha=0.7, label="NumberOfUniqueAlleles")
    slope_r, intercept_r = np.polyfit(df['AllLen_Median'], df['NumUniqueAlleles'], 1)
    axs[2, 1].plot(df['AllLen_Median'], slope_r * df['AllLen_Median'] + intercept_r, color='red', label=f"Fit: y = {slope_r:.2f}x + {intercept_r:.2f}")
    axs[2, 1].set_xlabel('Allele Length (median)')
    axs[2, 1].set_ylabel('Number of Unique Alleles')
    axs[2, 1].set_ylim(0, 450)
    axs[2, 1].set_title('(f)')

    # Save and show
    output_file = file_path.replace('.summary.tsv.gz', '.summary.pdf').replace('.summary.tsv', '.summary.pdf')  # os.path.basename(file_path)
    if output_file is not None:
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Summary figure saved as {output_file}")
    if output_file is None:
        plt.show()
    print(f"Number of repeat differences calculated: {len(repeat_diffs)}")


    output_file = file_path.replace('.summary.tsv.gz', '.summary.AlleleLengthDifferences.pdf').replace('.summary.tsv', '.summary.AlleleLengthDifferences.pdf')  # os.path.basename(file_path)
    fig, ax_left, ax_cent, ax_right = broken_histogram_figure(
        data=repeat_diffs,
        break_left=-1.0,
        break_right=1.0,
        bins=50,
        output_file=output_file,
        color='darkorange',
        edgecolor='black',
        xlabel='log2(Allele Length / Major Allele Length)',
        ylabel='VNTR count',
        title='Proportional length changes of alleles from major allele',
        figsize=(8, 4)
    )

    return


def process_file_plot_repeatdefinition_lengths(file_path, bins, figsize=(12, 8), fontsize=10, title_fontsize=12, left_fraction=2/3):
    '''
    Plot histogram of feature lengths and motif lengths with split x-ranges.

    Left subplot = lower 95%
    Right subplot = longest 5%

    Parameters
    ----------
    file_path : str
    bins : int
    figsize : tuple
        Figure size in inches
    fontsize : int
        Axis/tick font size
    title_fontsize : int
        Panel title font size
    left_fraction : float
        Fraction of horizontal space used by left panels
        e.g. 2/3 means left panels take 2/3, right panels 1/3
    '''

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # ---------------- Read file ----------------
    if file_path.endswith('.gz'):
        df = pd.read_csv(file_path, sep="\t", compression='gzip')
    else:
        df = pd.read_csv(file_path, sep="\t")

    # ---------------- Prepare data ----------------
    df['Motifs'] = df['Motifs'].astype(str)
    df['MotifLen'] = df['Motifs'].apply(
        lambda x: np.mean([len(m) for m in x.split(',') if m != ''])
    )

    df['Start'] = pd.to_numeric(df['Start'], errors='coerce')
    df['End'] = pd.to_numeric(df['End'], errors='coerce')

    # Feature length = End - Start
    df['FeatureLen'] = df['End'] - df['Start']

    feature_lengths = df['FeatureLen'].dropna()
    print(f"Feature length stats:\n{feature_lengths.describe()}")
    # feature_lengths = feature_lengths[feature_lengths <= 10000] # forcing x axis to cooperate for ax2, didn't work
    motif_lengths = df['MotifLen'].dropna()
    print(f"Motif length stats:\n{motif_lengths.describe()}")

    # ---------------- 95% split ----------------
    feature_break = np.percentile(feature_lengths, 99)
    motif_break = np.percentile(motif_lengths, 99)

    # Right-panel x ticks should start at the actual minimum plotted value
    feature_tail_min = int(feature_lengths[feature_lengths > feature_break].min())
    motif_tail_min = int(motif_lengths[motif_lengths > motif_break].min())

    # # ---------------- 95% split ----------------
    # feature_break_99 = np.percentile(feature_lengths, 99)
    # motif_break_99 = np.percentile(motif_lengths, 99)

    # ---------------- Figure layout ----------------
    right_fraction = 1 - left_fraction

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[left_fraction, right_fraction],
        hspace=0.4,
        wspace=0.21
    )

    # ---------------- Top row: Feature lengths ----------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.hist(
        feature_lengths[feature_lengths <= feature_break],
        bins=int(bins*left_fraction)
    )  # color='steelblue', edgecolor='black'

    ax2.hist(
        feature_lengths[feature_lengths > feature_break],  # _99],
        bins=int(bins*right_fraction)
    )  ## color='steelblue', edgecolor='black'
    ax2.set_xlim(feature_tail_min, feature_lengths.max())  # Set x-axis limits for the tail panel
    # Feature tail panel
    xticks = ax2.get_xticks()
    xticks = xticks[xticks >= feature_tail_min]
    xticks = np.insert(xticks, 0, feature_tail_min)
    ax2.set_xticks(np.unique(xticks))

    ax1.set_title(
        "a",
        loc="left",
        fontweight="bold",
        fontfamily="Arial",
        fontsize=title_fontsize
    )
    ax1.set_title(
        "Feature lengths",
        loc="center",
        fontfamily="Arial",
        fontsize=title_fontsize
    )
    ax2.set_title(
        "b",
        loc="left",
        fontweight="bold",
        fontfamily="Arial",
        fontsize=title_fontsize
    )
    ax2.set_title(
        "Feature lengths",
        loc="center",
        fontfamily="Arial",
        fontsize=title_fontsize
    )

    # ax1.set_xlabel("Feature length", fontsize=fontsize, fontfamily="Arial")
    # ax2.set_xlabel("Feature length", fontsize=fontsize, fontfamily="Arial")

    ax1.set_ylabel("Count", fontsize=fontsize, fontfamily="Arial")
    # ax2.set_ylabel("Count", fontsize=fontsize, fontfamily="Arial")

    # ---------------- Bottom row: Motif lengths ----------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax3.hist(
        motif_lengths[motif_lengths <= motif_break],
        bins=int(bins*left_fraction),
        color='teal'
    )   # color='teal', edgecolor='black'

    ax4.hist(
        motif_lengths[motif_lengths > motif_break],  #_99],
        bins=int(bins*right_fraction),
        color='teal'
    )    # color='teal', edgecolor='black'
    ax4.set_xlim(motif_tail_min, motif_lengths.max())  # Set x-axis limits for the tail panel
    # Motif tail panel
    xticks = ax4.get_xticks()
    xticks = xticks[xticks >= motif_tail_min]
    xticks = np.insert(xticks, 0, motif_tail_min)
    ax4.set_xticks(np.unique(xticks))

    ax3.set_title(
        "c",
        loc="left",
        fontweight="bold",
        fontfamily="Arial",
        fontsize=title_fontsize
    )
    ax3.set_title(
        "Motif lengths",
        loc="center",
        fontfamily="Arial",
        fontsize=title_fontsize
    )
    ax4.set_title(
        "d",
        loc="left",
        fontweight="bold",
        fontfamily="Arial",
        fontsize=title_fontsize
    )
    ax4.set_title(
        "Motif lengths",
        loc="center",
        fontfamily="Arial",
        fontsize=title_fontsize
    )

    # ax3.set_xlabel("Motif length", fontsize=fontsize, fontfamily="Arial")
    # ax4.set_xlabel("Motif length", fontsize=fontsize, fontfamily="Arial")

    ax3.set_ylabel("Count", fontsize=fontsize, fontfamily="Arial")
    # ax4.set_ylabel("Count", fontsize=fontsize, fontfamily="Arial")
    # ---------------- Arial ticks everywhere ----------------
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='both', labelsize=fontsize)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("Arial")

    # ---------------- Save outputs ----------------
    pdf_file = file_path.replace(
        '.summary.tsv.gz', '.summary.RepeatDefinitionLengths.pdf'
    ).replace(
        '.summary.tsv', '.summary.RepeatDefinitionLengths.pdf'
    )

    png_file = file_path.replace(
        '.summary.tsv.gz', '.summary.RepeatDefinitionLengths.png'
    ).replace(
        '.summary.tsv', '.summary.RepeatDefinitionLengths.png'
    )

    plt.tight_layout()
    plt.savefig(pdf_file)
    plt.savefig(png_file, dpi=300)

    print(f"Length figure saved as {pdf_file}")
    print(f"Length figure saved as {png_file}")

    return


def process_file_plot_lengths(file_path, bins):
    '''
    Plot histogram of allele lengths, with a broken x-axis to show the distribution
    of allele lengths while also showing the long tail of allele lengths.

    Plot histogram of motif lengths, with a broken x-axis to show the distribution
    of motif lengths while also showing the long tail of motif lengths.

    Save figure as .pdf and .png

    Plot has 4 subplots: 2 rows, 2 columns
    '''

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Read file
    if file_path.endswith('.gz'):
        df = pd.read_csv(file_path, sep="\t", compression='gzip')
    else:
        df = pd.read_csv(file_path, sep="\t")

    # Clean and prepare data
    df['Motifs'] = df['Motifs'].astype(str)
    df['MotifLen'] = df['Motifs'].apply(
        lambda x: np.mean([len(m) for m in x.split(',') if m != ''])
    )

    df['AllLen_Median'] = pd.to_numeric(df['AllLen_Median'], errors='coerce')

    allele_lengths = df['AllLen_Median'].dropna()
    motif_lengths = df['MotifLen'].dropna()

    # Heuristic breakpoints (same spirit as repeat_diffs in process_file)
    allele_break = np.percentile(allele_lengths, 99)
    # allele_break_right = np.percentile(allele_lengths, 95)

    motif_break = np.percentile(motif_lengths, 99)
    # motif_break_right = np.percentile(motif_lengths, 95)

    # Create figure
    fig = plt.figure(figsize=(12, 8))

    # -------- Allele lengths (top row) --------
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(
        allele_lengths[allele_lengths <= allele_break],
        bins=bins,
        color='steelblue',
        edgecolor='black'
    )
    ax1.set_xlabel("Allele length")
    ax1.set_ylabel("Count")
    ax1.set_title("a", loc="left", fontweight="bold", fontfamily="arial")  # "(a) Allele lengths")

    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(
        allele_lengths[allele_lengths >= allele_break],
        bins=bins,
        color='steelblue',
        edgecolor='black'
    )
    ax2.set_xlabel("Allele length (tail)")
    ax2.set_ylabel("Count")
    ax2.set_title("b", loc="left", fontweight="bold", fontfamily="arial")  # "(b) Allele lengths long tail")

    # -------- Motif lengths (bottom row) --------
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(
        motif_lengths[motif_lengths <= motif_break],
        bins=bins,
        color='teal',
        edgecolor='black'
    )
    ax3.set_xlabel("Motif length")
    ax3.set_ylabel("Count")
    ax3.set_title("c", loc="left", fontweight="bold", fontfamily="arial")  # "(c) Motif lengths")

    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(
        motif_lengths[motif_lengths >= motif_break],
        bins=bins,
        color='teal',
        edgecolor='black'
    )
    ax4.set_xlabel("Motif length (tail)")
    ax4.set_ylabel("Count")
    ax4.set_title("d", loc="left", fontweight="bold", fontfamily="arial")  # (d) Motif lengths long tail")

    plt.tight_layout()

    # Save outputs
    pdf_file = file_path.replace(
        '.summary.tsv.gz', '.summary.Lengths.pdf'
    ).replace(
        '.summary.tsv', '.summary.Lengths.pdf'
    )

    png_file = file_path.replace(
        '.summary.tsv.gz', '.summary.Lengths.png'
    ).replace(
        '.summary.tsv', '.summary.Lengths.png'
    )

    plt.savefig(pdf_file)
    plt.savefig(png_file, dpi=300)

    print(f"Length figure saved as {pdf_file}")
    print(f"Length figure saved as {png_file}")

    return


def summary_stats(datalist):
    import numpy as np
    from scipy import stats

    # Convert the data list to a numpy array for easier calculations
    data = np.array(datalist)

    # Calculate statistics
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    sum_val = np.sum(data)
    # mode = stats.mode(data)[0][0]  # Mode returns an object, we need the first element

    d = {'mean': mean,
        'median': median,
        'std_dev': std_dev,
        'min': min_val,
        'max': max_val,
        'sum': sum_val}
        # 'mode': mode}
    
    d = {key: str(value) for key, value in d.items()}  # Convert all values to string for output to TSV

    return d


def make_summary_file(infile):
    import gzip

    outfile = infile.replace('.txt.gz', '.summary.tsv.gz') if infile.endswith('.gz') else infile.replace('.txt', '.summary.tsv')
    with gzip.open(outfile, 'wt') if outfile.endswith('.gz') else open(outfile, 'w') as OUT:
        OUT.write('')
    with gzip.open(outfile, 'at') if outfile.endswith('.gz') else open(outfile, 'a') as OUT:
        outheader = "LocusID\tPloidy\tTRorVNTR\tChrom\tStart\tEnd\tNumEffMotifs\tNumEffMotifsUsed\tNumUniqueAlleles\tclass\tsymbol\tGeneID\tGeneStart\tGeneEnd\tstrand\tproduct_accession\trelated_accession\tTROverlapByGene\tGeneOverlapByTR\tname\tRefAllelePresentAbsent\tAlleleIDs\tMotifs\tCountsPerAllele\tAllelelength"
        outheader += "\tCPA_Mean\tCPA_Median\tCPA_StdDev\tCPA_Min\tCPA_Max\tCPA_Sum\tMajorAlleleFrequency\tMinorAlleleFrequency\tAllLen_Mean\tAllLen_Median\tAllLen_StdDev\tAllLen_Min\tAllLen_Max\tNumMotifs_Mean\tNumMotifs_Median\tNumMotifs_StdDev\tNumMotifs_Min\tNumMotifs_Max\tMotifLen_Mean\tMotifLen_Median\tMotifLen_StdDev\tMotifLen_Min\tMotifLen_Max"
        OUT.write(outheader + '\n')

        with gzip.open(infile, 'rt') if infile.endswith('.gz') else open(infile, 'r') as FILE:
            header = FILE.readline().strip().split("\t")
            didx = {colname: i for i, colname in enumerate(header)}
            outlines = []
            for line in FILE:
                cols = line.strip().split("\t")
                if (line.strip() == '') or (line.startswith('LocusID')) or (cols[didx['CountsPerAllele']] == ''):
                    continue
                ### if len(cols) != len(header):
                ###     print(f"Skipping malformed line: {line.strip()}")
                ###     continue

                num_alleles = (cols[didx['NumUniqueAlleles']] if cols[didx['RefAllelePresentAbsent']] == 'RefAllelePresent' else str(int(cols[didx['NumUniqueAlleles']]) - 1))
                constantout = '\t'.join([cols[didx['LocusID']], cols[didx['Ploidy']], cols[didx['TRorVNTR']], cols[didx['Chrom']], cols[didx['Start']], cols[didx['End']], cols[didx['NumEffMotifs']], cols[didx['NumEffMotifsUsed']], num_alleles, cols[didx['class']], cols[didx['symbol']], cols[didx['GeneID']], cols[didx['GeneStart']], cols[didx['GeneEnd']], cols[didx['strand']], cols[didx['product_accession']], cols[didx['related_accession']], cols[didx['TROverlapByGene']], cols[didx['GeneOverlapByTR']], cols[didx['name']], cols[didx['RefAllelePresentAbsent']], cols[didx['AlleleIDs']], cols[didx['Motifs']], cols[didx['CountsPerAllele']], cols[didx['Allelelength']]])
                # get summary stats for each relevant column from the options below
                # CpGDensityPerAllele	AvgAlleleMethylation	AvgAllelePurity	Allelelength	NumMotifsUsedPerAllele	AlleleMotifCounts	AlleleMotifSpans	Motifs	Alleles]
                # print(cols[didx['LocusID']], cols[didx['CountsPerAllele']])  # locus 1.54549138.54549166 has no allele counts? why?
                cpa_stats = summary_stats([int(i) for i in cols[didx['CountsPerAllele']].split(',')])
                alllen_stats = summary_stats([int(i) for i in cols[didx['Allelelength']].split(',')])
                nummotifs_stats = summary_stats([int(i) for i in cols[didx['NumMotifsUsedPerAllele']].split(',')])
                motiflen_stats = summary_stats([len(i) for i in cols[didx['Motifs']].split(',')])

                outline = f"{constantout}\t{cpa_stats['mean']}\t{cpa_stats['median']}\t{cpa_stats['std_dev']}\t{cpa_stats['min']}\t{cpa_stats['max']}\t{cpa_stats['sum']}\t{str(float(cpa_stats['max'])/float(cpa_stats['sum']))}\t{str(float(cpa_stats['min'])/float(cpa_stats['sum']))}\t{alllen_stats['mean']}\t{alllen_stats['median']}\t{alllen_stats['std_dev']}\t{alllen_stats['min']}\t{alllen_stats['max']}\t{nummotifs_stats['mean']}\t{nummotifs_stats['median']}\t{nummotifs_stats['std_dev']}\t{nummotifs_stats['min']}\t{nummotifs_stats['max']}\t{motiflen_stats['mean']}\t{motiflen_stats['median']}\t{motiflen_stats['std_dev']}\t{motiflen_stats['min']}\t{motiflen_stats['max']}"
                outlines.append(outline)

                if len(outlines) >= 100:
                    # incrementally write output lines to avoid memory issues
                    OUT.write('\n'.join(outlines) + '\n')
                    outlines = []

            OUT.write('\n'.join(outlines) + '\n')  # write remaining output lines
            outlines = []

    return outfile


if __name__ == "__main__":
    '''
    Usage:
    conda activate py3
    python TRGTSummarizeLocusStats.py --infile /path/to/Autosomes_TRDB_LocusStats.txt.gz --bins 100
    python TRGTSummarizeLocusStats.py --infile /Users/vmason/Documents/Analyses/trgt/v2.1.0/120Hifi94Calves18Ass/output/cluster/TRdatabase/SplitByChromosome/Autosomes/Autosomes_TRDB_LocusStats.txt.gz

    # To skip straight to plotting if you already have a .summary.tsv.gz file:
    python TRGTSummarizeLocusStats.py --infile /path/to/Autosomes_TRDB_LocusStats.summary.tsv.gz --bins 100
    '''
    import argparse

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process a TSV file to generate a histogram of lengths.")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for the histogram")
    parser.add_argument("--infile", type=str, help="Full path to the TSV file, optional give path to .summary.tsv.gz file to skip straight to plotting")
    # Parse arguments
    args = parser.parse_args()
    
    if args.infile.endswith('.summary.tsv.gz') or args.infile.endswith('summary.tsv'):
        outfile = args.infile
    else:
        # Process the file
        outfile = make_summary_file(args.infile)
    
    process_file_plot_repeatdefinition_lengths(outfile, args.bins, figsize=(7, 4))  # , fontsize=10, title_fontsize=12, left_fraction=2/3)
    process_file_plot_lengths(outfile, args.bins)
    process_file(outfile, args.bins)
