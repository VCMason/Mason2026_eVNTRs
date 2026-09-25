import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import limix
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from statsmodels.stats.multitest import multipletests


def read_result_table_and_filter(output_file, workingdir, fdr_threshold=0.5):
    # output_file is the resulting table after running_linear modeling associations
    # working dir is the base directory from which all output will be directed
    # Read the file back in as a DataFrame
    outdir, _ = os.path.split(output_file)
    results_df = pd.read_csv(output_file, sep="\t", dtype={"geneid": str, "VNTRid": str})
    print(f'About raw result matrix: {results_df.describe()}')

    # remove loci with standardizedbeta values = 0. standbeta = 0 when standard deviation of VNTR length is 0. (i.e. not variable)
    results_df = results_df[results_df["StandardizedBeta"] > 0]
    print(f'About result matrix after removing non-variable X-axes: {results_df.shape}')

    # Multiple testing corrections
    results_df["corrected_pvalue_FDR"] = multipletests(results_df["raw_pvalue"], method="fdr_bh", alpha=fdr_threshold)[1]
    results_df["corrected_pvalue_Bonferroni"] = multipletests(results_df["raw_pvalue"], method="bonferroni")[1]
    print('Multiple testing correction completed.')

    # Save updated results with corrections
    output_file = os.path.join(workingdir, outdir, "gene_vntr_association_results.No0StandBeta.tsv")
    results_df.to_csv(output_file, sep="\t", index=False)
    print(output_file)

    output_file = os.path.join(workingdir, outdir, "best_gene_vntr_associations.No0StandBeta.tsv")
    # Select the VNTR(s) with the lowest p-value per gene
    best_results = results_df.loc[results_df.groupby("geneid")["raw_pvalue"].idxmin()]
    best_results.to_csv(output_file, sep="\t", index=False)
    print(f'About the best VNTR per gene result matrix (does not require significance) after removing non-variable X-axes: {best_results.shape}')
    print(output_file)

    output_file = os.path.join(workingdir, outdir, "significant_gene_vntr_associations.No0StandBeta.tsv")
    # Select only significant associations based on FDR correction
    significant_results = results_df[results_df["corrected_pvalue_FDR"] < 0.05]
    significant_results.to_csv(output_file, sep="\t", index=False)
    print(f'About significant result matrix after removing non-variable X-axes: {significant_results.shape}')
    print(output_file)

    return


def perform_lmm_fit_null(gene_expression, vntr_values, common_inds, covariates, kinship_matrix):
    
    Y = gene_expression[common_inds]
    X = vntr_values[common_inds]
    covs = covariates[common_inds].T
    covs = sm.add_constant(covs)
    kinship_sub = kinship_matrix.loc[common_inds, common_inds]
    
    # Fit LMM without VNTR to get residuals (removing covariate & kinship effects)
    null_model = limix.qtl.qtl_test_lmm(np.zeros_like(X), Y.values, covs.values, kinship_sub.values)
    # limix.qtl.qtl_test_lmm(np.zeros_like(X), Y, covs, kinship_sub)  # No VNTR (X = zeros)
    residuals = Y - (null_model["beta"][:-1] @ covs.T).T  # Remove covariate effects

    # beta, se, pvalue = model['beta'][0], model['se'][0], model['pval'][0]
    # variance_explained = (beta**2 * np.var(X)) / np.var(Y)
    
    # return np.mean(Y), np.mean(X), variance_explained, beta, se, pvalue, model
    return residuals


def perform_lmm(gene_expression, vntr_values, common_inds, covariates, kinship_matrix, filter_common=True):
    
    if filter_common == True:
        Y = gene_expression.loc[common_inds]
        X = vntr_values.loc[common_inds]
    else:
        Y = gene_expression
        X = vntr_values

    covs = covariates[common_inds].T
    covs = sm.add_constant(covs)
    kinship_sub = kinship_matrix.loc[common_inds, common_inds]
    
    model = limix.qtl.qtl_test_lmm(X.values.reshape(-1,1), Y.values, covs.values, kinship_sub.values)
    beta, se, pvalue = model['beta'][0], model['se'][0], model['pval'][0]
    variance_explained = (beta**2 * np.var(X)) / np.var(Y)
    
    return Y, X, variance_explained, beta, se, pvalue, model


def perform_multilinear_fit_null(gene_expression, common_inds, covariates):
    
    Y = gene_expression[common_inds]
    covs = covariates[common_inds].T
    covs = sm.add_constant(covs)
    
    # Regress gene expression on covariates only (no vntr_values included)
    null_model = sm.OLS(Y.values, covs).fit()
    residuals = null_model.resid  # These are the gene expression residuals with covariate effects removed

    # beta, se, pvalue = model.params.iloc[-1], model.iloc.bse[-1], model.iloc.pvalues[-1]
    # variance_explained = model.rsquared
    
    # return np.mean(Y), np.mean(X), variance_explained, beta, se, pvalue, model
    return residuals, null_model


def perform_multilinear(gene_expression, vntr_values, common_inds, covariates, filter_common=True):
    """
    Performs multiple linear regression including VNTR values and covariates.
    
    Parameters:
        gene_expression (pd.Series): Gene expression values, indexed by individual IDs.
        vntr_values (pd.Series): VNTR values, indexed by individual IDs.
        common_inds (Index): Individuals shared between gene expression and VNTR data.
        covariates (pd.DataFrame): Covariate matrix with row names as covariates and columns as individuals.
        minsamples (int): Minimum number of samples required for regression.

    Returns:
        tuple: (Mean Y, Mean X, Variance Explained, Beta, SE, P-value, Model)
    """

    # Subset to common individuals while preserving structure
    if filter_common == True:
        Y = gene_expression.loc[common_inds]
        X = vntr_values.loc[common_inds]
    else:
        Y = gene_expression
        X = vntr_values

    # Transpose covariates and subset
    covs = covariates.loc[:, common_inds].T  # Ensure individuals are columns
    covs = sm.add_constant(covs)  # Add intercept term

    # Convert VNTR values into a DataFrame to keep its label
    X_df = X.to_frame(name=f"{X.name}")  # Label VNTR variable

    # Concatenate covariates and VNTR into a single DataFrame
    X_full = pd.concat([covs, X_df], axis=1)

    # Fit model using Pandas DataFrame (preserves column names)
    model = sm.OLS(Y, X_full).fit()

    # Extract relevant statistics
    beta, se, pvalue = model.params[f"{X.name}"], model.bse[f"{X.name}"], model.pvalues[f"{X.name}"]
    variance_explained = model.rsquared

    return Y, X, variance_explained, beta, se, pvalue, model


def perform_simplelinear(gene_expression, vntr_values, common_inds, filter_common=True):
    
    if filter_common == True:
        Y = gene_expression.loc[common_inds]
        X = vntr_values.loc[common_inds]
    else:
        Y = gene_expression
        X = vntr_values

    X_full = sm.add_constant(X)
    
    model = sm.OLS(Y.values, X_full).fit()
    beta, se, pvalue = model.params.iloc[-1], model.bse.iloc[-1], model.pvalues.iloc[-1]
    variance_explained = model.rsquared
    
    return Y, X, variance_explained, beta, se, pvalue, model


def remove_sd_outliers(values, threshold=5):
    """
    Remove individuals whose expression values are more than `threshold` SDs from the mean.
    """
    mean = np.mean(values)
    std = np.std(values)
    mask = np.abs(values - mean) <= threshold * std
    return values[mask]

def run_association_analysis(gene_vntr_dict, expression_matrix, vntr_matrix, covariates, kinship_matrix, minsamples, output_file, fdr_threshold=0.05):
    # output_file = os.path.join(workingdir, "gene_vntr_association_results.tsv")

    # Write the header once
    with open(output_file, "w") as OUT:
        OUT.write("\t".join(["geneid", "VNTRid", "genepos", "TSS_distance", "MeanGeneTPM", "MeanVNTRlength", "r2", "Partialr2", "beta", "StandardizedBeta", "SE", "raw_pvalue"]))

    countmodels = 0
    countnotenoughdata = 0
    countinvariantvntr = 0
    countinvariantgeneexp = 0
    with open(output_file, "a") as OUT:
        numgenes = len(gene_vntr_dict)
        for count, (gene_id, vntr_ids) in enumerate(gene_vntr_dict.items(), start=1):
            gene_results = []
            gene_chr = expression_matrix.loc[gene_id, "#chr"]  # if "#chr" in expression_matrix.columns else None
            gene_start = expression_matrix.loc[gene_id, "start"]  # if "start" in  expression_matrix.columns else None
            gene_end = expression_matrix.loc[gene_id, "end"]  # if "end" in  expression_matrix.columns else None
            gene_strand = expression_matrix.loc[gene_id, "strand"]  # if "strand" in  expression_matrix.columns else None
            gene_pos = gene_start if gene_strand == "+" else gene_end
            gene_expression = expression_matrix.loc[gene_id, expression_matrix.columns[6:]].dropna().astype(float)
            gene_expression = remove_sd_outliers(gene_expression, threshold=5)
            for vntr_id in vntr_ids:
                vntr_start = vntr_matrix.loc[vntr_id, "start"]  # if "start" in  expression_matrix.columns else None
                vntr_end = vntr_matrix.loc[vntr_id, "end"]  # if "end" in  expression_matrix.columns else None
                if vntr_start <= gene_pos <= vntr_end:
                    TSS_distance = 0  # negative if VNTR is upstream of gene TSS, positive if VNTR is downstream of gene TSS, 0 if VNTR overlaps gene TSS
                elif vntr_start >= gene_pos >= vntr_end:
                    TSS_distance = 0
                elif (gene_pos > vntr_start) and (gene_pos > vntr_end):
                    TSS_distance = vntr_end - gene_pos # negative distance if VNTR is upstream of gene TSS
                elif (gene_pos < vntr_start) and (gene_pos < vntr_end):
                    TSS_distance = vntr_start - gene_pos # positive distance if VNTR is downstream of gene TSS
                else:
                    TSS_distance = float('nan')
                vntr_values = vntr_matrix.loc[vntr_id, vntr_matrix.columns[6:]].loc[lambda x: (x != '.') & x.notna()].astype(float)
                common_inds = gene_expression.index.intersection(vntr_values.index)

                if len(common_inds) < minsamples:
                    countnotenoughdata += 1
                    continue  # Skip if not enough samples

                # Subset to common individuals
                gene_values = gene_expression.loc[common_inds]
                vntr_values = vntr_values.loc[common_inds]

                if vntr_values.nunique() < 2:
                    ### print(f"VNTR {vntr_id} has no variation in VNTR values (ex: genotypes, mean allele length). Number vntr_vals: {vntr_vals.nunique()}")
                    ### print(f"{vntr_vals.unique()}")
                    countinvariantvntr += 1
                    continue
                if gene_values.std() < 1e-6:
                    ### print(f"Gene {gene_id} has no variation in gene expression.")
                    countinvariantgeneexp += 1
                    continue

                
                if (kinship_matrix is not None) and (covariates is not None):
                    result = perform_lmm(gene_values, vntr_values, common_inds, covariates, kinship_matrix, filter_common=False)
                elif covariates is not None:
                    covariates = covariates.apply(pd.to_numeric, errors="coerce")
                    covariates = covariates.dropna(axis=1, how="all")  # Drop columns with all NaNs
                    if gene_expression.empty or vntr_values.empty or covariates.empty:
                        print(f'Problem prepping data for multulinear regression')
                        continue  # Skip if any required data is missing
                    result = perform_multilinear(gene_values, vntr_values, common_inds, covariates, filter_common=False)
                else:
                    if gene_expression.empty or vntr_values.empty:
                        continue  # Skip if any required data is missing
                    result = perform_simplelinear(gene_values, vntr_values, common_inds, filter_common=False)

                if result:
                    Y, X, var_exp, beta, se, pval, model = result
                    mean_Y, mean_X = float(np.mean(Y)), float(np.mean(X))
                    d = compute_effect_size_and_heritability(Y, X, model)
                    gene_results.append([gene_id, vntr_id, f"{gene_chr}.{gene_start}.{gene_end}", TSS_distance, mean_Y, mean_X, var_exp, d["Partial_R2"], beta, d["Standardized_Beta"], se, pval])
                    countmodels += 1

            # Append results to file after processing all VNTRs for a gene
            if gene_results:
                OUT.write('\n' + '\n'.join(['\t'.join(map(str, row)) for row in gene_results]))

            if count%100 == 0:
                print(f'Modeled {count} genes of total genes {numgenes}')
    print(f'{countmodels} number of linear models were completed.')
    print(f'{countnotenoughdata} number of gene-VNTR pairs were skipped due to not enough data (<{minsamples} samples).')
    print(f'{countinvariantvntr} number of gene-VNTR pairs were skipped due to invariant VNTR values (only 1 mean length).')
    print(f'{countinvariantgeneexp} number of gene-VNTR pairs were skipped due to invariant gene expression values.')
    with open(output_file.replace('.tsv', '.log') , "w") as OUT:
        OUT.write(
        f"For output file {output_file}:\n"
        f"{countmodels} number of linear models were completed.\n"
        f"{countnotenoughdata} number of gene-VNTR pairs were skipped due to not enough data (<{minsamples} samples).\n"
        f"{countinvariantvntr} number of gene-VNTR pairs were skipped due to invariant VNTR values (only 1 mean length).\n"
        f"{countinvariantgeneexp} number of gene-VNTR pairs were skipped due to invariant gene expression values.\n"
        )

    return  # results_df, best_results, significant_results


### def run_association_analysis(gene_vntr_dict, expression_matrix, vntr_matrix, covariates, kinship_matrix, minsamples, fdr_threshold=0.05):
###     results = []
###     
###     for gene_id, vntr_ids in gene_vntr_dict.items():
###         gene_results = []  # Store per-gene results to append only once per gene
###         for vntr_id in vntr_ids:
###             gene_expression = expression_matrix.loc[gene_id, expression_matrix.columns[6:]].dropna()
###             vntr_values = vntr_matrix.loc[vntr_id, vntr_matrix.columns[6:]].dropna()
###             common_inds = gene_expression.index.intersection(vntr_values.index)
### 
###             if len(common_inds) < minsamples:
###                 continue  # Skip if not enough samples
### 
###             if (kinship_matrix is not None) and (covariates is not None):
###                 result = perform_lmm(gene_expression, vntr_values, common_inds, covariates, kinship_matrix)
###             elif covariates is not None:
###                 result = perform_multilinear(gene_expression, vntr_values, common_inds, covariates)
###             else:
###                 result = perform_simplelinear(gene_expression, vntr_values, common_inds)
### 
###             if result:
###                 mean_Y, mean_X = np.mean(gene_expression), np.mean(vntr_values)
###                 gene_results.append([gene_id, vntr_id, mean_Y, mean_X] + list(result))
### 
###         # Append gene results to the main list after looping over VNTRs
###         results.extend(gene_results)
### 
###     # Convert results to a DataFrame
###     results_df = pd.DataFrame(results, columns=["geneid", "VNTRid", "MeanGeneTPM", "MeanVNTRlength", "r2", "beta", "SE", "raw_pvalue"])
### 
###     # Multiple testing corrections
###     results_df["corrected_pvalue_Bonferroni"] = multipletests(results_df["raw_pvalue"], method="bonferroni")[1]
###     results_df["corrected_pvalue_FDR"] = multipletests(results_df["raw_pvalue"], method="fdr_bh", alpha=fdr_threshold)[1]
### 
###     # Save full results
###     results_df.to_csv("gene_vntr_association_results.tsv", sep="\t", index=False)
### 
###     # Select the VNTR(s) with the lowest p-value per gene
###     best_results = results_df.loc[results_df.groupby("geneid")["raw_pvalue"].idxmin()]
### 
###     # Save best associations
###     best_results.to_csv("best_gene_vntr_associations.tsv", sep="\t", index=False)
### 
###         # Select only significant associations based on FDR correction
###     significant_results = results_df[results_df["corrected_pvalue_FDR"] < 0.05]
### 
###     # Save significant associations
###     significant_results.to_csv("significant_gene_vntr_associations.tsv", sep="\t", index=False)
### 
###     return  # results_df, best_results, significant_results


###################################################################
###################################################################
###################################################################


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def compute_effect_size_and_heritability(Y, X, model):
    #unfinished
    """
    Computes:
    - Standardized beta coefficient (effect size)
    - Partial R-squared
    - Simple heritability estimate (h^2)
    
    Parameters:
        Y (pd.Series): Gene expression values
        X (pd.Series): VNTR values
        model (sm.OLSResults): Fitted OLS regression model
        
    Returns:
        dict: Dictionary containing effect size metrics
    """
    # Extract regression parameters
    beta = model.params.iloc[-1]   # Last coefficient (VNTR effect)
    t_value = model.tvalues.iloc[-1]
    df = model.df_resid       # Degrees of freedom

    # Compute Standardized Beta
    sigma_X = X.std()
    sigma_Y = Y.std()
    beta_standardized = beta * (sigma_X / sigma_Y)

    # Compute Partial R²
    partial_r2 = (t_value ** 2) / (t_value ** 2 + df)

    # Compute Simple Heritability (h²)
    # r2_simple = model.rsquared  # R-squared from simple regression as a lower bound for h²

    return {
        "Standardized_Beta": beta_standardized,
        "Partial_R2": partial_r2,
    }  # "Heritability_(h2)": r2_simple


def assess_multicollinearity(covariates, gene_expression, vntr_values, common_inds, gene_id, vntr_id, outdir, allcovar=False):
    """
    Generates correlation plots to assess multicollinearity between covariates and VNTR values.
    
    Parameters:
        covariates (pd.DataFrame): Covariate matrix (indexed by covariate names, columns are individuals).
        vntr_values (pd.Series): VNTR variable (indexed by individuals).
        common_inds (list): List of individuals used in regression.
        vntr_id (str): ID of the VNTR variable.

    Returns:
        None (Displays plots)
    """
    
    # Filter to selected individuals
    covs_filtered = covariates.T.loc[common_inds]
    
    # Add VNTR values as a new column
    covs_filtered[f"{vntr_id}"] = vntr_values.loc[common_inds]
    # Add raw gene expression TPM to the matrix
    covs_filtered[f"{gene_id}"] = gene_expression.loc[common_inds]

    # Compute correlation matrix
    corr_matrix = covs_filtered.corr()

    # Heatmap of Pearson Correlations
    if allcovar:
        outfilename = f'PairwiseCorrelationHeatmap.AllCovariates.Gene:{gene_id}VS{vntr_id}.pdf'
    else:
        outfilename = f'PairwiseCorrelationHeatmap.Gene:{gene_id}VS{vntr_id}.pdf'
    outfile = os.path.join(outdir, outfilename)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 35 / np.sqrt(len(corr_matrix))})
    plt.title(f"Correlation Heatmap: Covariates & VNTR ({vntr_id})")
    plt.savefig(outfile, format="pdf")
    plt.close()
    print(f'Made Plot: {outfilename}')
    #plt.show()

    # Pairwise Scatter Plots
    if allcovar:
        outfilename = f'PairwiseScatterPlots.AllCovariates.Gene:{gene_id}VS{vntr_id}.pdf'
    else:
        outfilename = f'PairwiseScatterPlots.Gene:{gene_id}VS{vntr_id}.pdf'
    outfile = os.path.join(outdir, outfilename)
    sns.pairplot(covs_filtered)
    plt.suptitle(f"Pairwise Scatter Plots: Covariates & VNTR ({vntr_id})", y=1.02)
    plt.savefig(outfile, format="pdf")
    plt.close()
    print(f'Made Plot: {outfilename}')
    #plt.show()
    
    return


def plot_regression(gene_id, vntr_id, expression_matrix, vntr_matrix, covariates, allcovariates, kinship_matrix, workingdir, minsamples, xaxislabel):
    outfilename = f'Regression.Gene:{gene_id}VS{vntr_id}.pdf'
    outdir = os.path.join(workingdir, 'plots')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, outfilename)

    print(f"Gene expression for {gene_id}:")
    print(expression_matrix.loc[gene_id, expression_matrix.columns[6:]])
    print(f"VNTR values for {vntr_id}:")
    print(vntr_matrix.loc[vntr_id, vntr_matrix.columns[6:]])

    gene_expression = expression_matrix.loc[gene_id, expression_matrix.columns[6:]].dropna().astype(float)
    vntr_values = vntr_matrix.loc[vntr_id, vntr_matrix.columns[6:]].loc[lambda x: (x != '.') & x.notna()].astype(float)
    common_inds = gene_expression.index.intersection(vntr_values.index)
    print(f"Common Individuals (without missing data 'nan'): {len(common_inds)}")
    print(common_inds)
    print(type(gene_expression))
    print(gene_expression.describe())
    print(vntr_values.describe())
    
    print('Assessing Multicolinearity of all covariates:')
    assess_multicollinearity(allcovariates, gene_expression, vntr_values, common_inds, gene_id, vntr_id, outdir, allcovar=True)

    if kinship_matrix is not None:
        covariates = covariates.apply(pd.to_numeric, errors="coerce")
        covariates = covariates.dropna(axis=1, how="all")  # Drop columns with all NaNs
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    elif covariates is not None:
        covariates = covariates.apply(pd.to_numeric, errors="coerce")
        covariates = covariates.dropna(axis=1, how="all")  # Drop columns with all NaNs
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))  # always do at least 1 simple linear regression
    
    
    results = {
        "Simple": perform_simplelinear(gene_expression, vntr_values, common_inds),
        "Multi": perform_multilinear(gene_expression, vntr_values, common_inds, covariates) if covariates is not None else None,
        "LMM": perform_lmm(gene_expression, vntr_values, covariates, common_inds, kinship_matrix) if kinship_matrix is not None else None
    }

    for ax, (title, result) in zip(axes, results.items()):
        if result:
            Y, X, var_exp, beta, se, pval, model = result
            print('')
            print(f'Model Summary for: {title}')
            print(f'{model.summary()}')
            print(compute_effect_size_and_heritability(Y, X, model))
            if title == "Simple":
                x_vals = vntr_values.loc[common_inds]  # VNTR values
                y_vals = gene_expression.loc[common_inds]  # Gene expression
                x_range = np.linspace(x_vals.min(), x_vals.max(), 100)  # Generate a smooth range of x-values
                # x_range = np.array(ax.get_xlim())
                ax.plot(x_range, beta * x_range + model.params.iloc[0], color='red', label="Fitted Line")
                ax.set_ylabel("Gene Expression TPM")
            elif title == "Multi":
                assess_multicollinearity(covariates, gene_expression, vntr_values, common_inds, gene_id, vntr_id, outdir)
                residuals, null_model = perform_multilinear_fit_null(gene_expression, common_inds, covariates)
                result = perform_simplelinear(residuals, vntr_values, common_inds)
                Y, X, var_exp, beta, se, pval, model = result
                print('')
                print('Null Model: GeneExpTMP ~ Desired Covariates')
                print(f'{null_model.summary()}')
                # print('Residuals (GeneExpTMP ~ covariates) ~ vntr_value: simple linear regression results')
                # print(f'{model.summary()}')
                y_vals = residuals
                ax.plot(x_range, beta * x_range + model.params.iloc[0], color='red', label="Fitted Line")
                ax.set_ylabel("Residuals (GeneExpTPM ~ covariates)")
            elif title == "LMM":
                residuals = perform_lmm_fit_null(gene_expression, vntr_values, covariates, common_inds, kinship_matrix)
                result = perform_simplelinear(residuals, vntr_values, common_inds)
                Y, X, var_exp, beta, se, pval, model = result
                # print('Residuals for LMM ~ vntr_value: simple linear regression results')
                # print(f'{model.summary()}')
                y_vals = residuals
                ax.plot(x_range, beta * x_range + model.params.iloc[0], color='red', label="Fitted Line")
                ax.set_ylabel("Residuals (GeneExpTPM ~ covariates, randomeff=kinshipmatrix)")
            ax.scatter(x_vals, y_vals, label="Data")
            ax.set_xlabel(xaxislabel)
            ax.set_title(f"{title} Regression\nVar Expl.: {var_exp:.3f}, p: {pval:.3e}, B: {beta:.3}")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(outfile, format="pdf")
    plt.close()
    # plt.show()

###################################################################
###################################################################
###################################################################

### def do_ranges_overlap(start, end, prevstart, prevend):
###     if (start <= prevend) and (end >= prevstart):  # if ranges overlap
###         return True
###     else:
###         return False


def write_cis_vntrs_to_file(gene_vntr_dict, output_dir, filename="cis_vntrs.tsv"):
    """
    Write gene-to-VNTR associations to a tab-separated file.
    
    Parameters:
    - gene_vntr_dict: dict -> {gene_id: [vntr_id1, vntr_id2, ...]}
    - output_dir: str -> Path to the output directory
    - filename: str -> Name of the output file (default: "cis_vntrs.tsv")
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Full output file path
    output_file = os.path.join(output_dir, filename)

    with open(output_file, "w") as f:
        for gene_id, vntr_ids in gene_vntr_dict.items():
            line = f"{gene_id}\t" + "\t".join(vntr_ids) + "\n"
            f.write(line)
    
    print(f"Saved cis-VNTR results to: {output_file}")


def find_cis_vntrs(expression_matrix, vntr_matrix, window=1000000):
    gene_vntr_dict = {}
    
    for _, gene in expression_matrix.iterrows():
        gene_chr, gene_start, gene_end, gene_strand = gene["#chr"], gene["start"], gene["end"], gene["strand"]
        gene_pos = gene_start if gene_strand == "+" else gene_end
        
        winstart = gene_pos - window // 2
        winend = gene_pos + window // 2

        # Vectorized filtering instead of do_ranges_overlap()
        nearby_vntrs = vntr_matrix[
            (vntr_matrix["#chr"] == gene_chr) &
            (vntr_matrix["start"] <= winend) & 
            (vntr_matrix["end"] >= winstart)
        ]

        if not nearby_vntrs.empty:
            gene_vntr_dict[gene.name] = list(nearby_vntrs.index)
    
    return gene_vntr_dict


def read_in_cis_vntrs(filepath):
    """
    Reads a tab-separated file of gene-to-VNTR associations and returns a dictionary.
    
    Parameters:
    - filepath: str -> Path to the cis_vntrs.tsv file
    
    Returns:
    - gene_vntr_dict: dict -> {gene_id: [vntr_id1, vntr_id2, ...]}
    """
    gene_vntr_dict = {}
    
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            gene_id = parts[0]  # First column is the gene ID
            vntr_ids = parts[1:]  # Remaining columns are VNTR IDs
            gene_vntr_dict[gene_id] = vntr_ids  # Store in dictionary
    
    return gene_vntr_dict


def sort_matrices(expression_matrix, vntr_matrix, covariates, allcovariates, kinship_matrix):
    print('Sorting matrices: gene_expression, vntr_values, if/and covariates, & all covariates, if/and kinship')
    from natsort import natsorted

    # Extract individual IDs (columns for gene expression, VNTR, and covariates)
    # Preserve the first 6 columns (metadata columns)
    metadata_cols = expression_matrix.columns[:6]
    # exlude first 6 columns
    gene_inds = expression_matrix.columns[6:]
    vntr_inds = vntr_matrix.columns[6:]
    cov_inds = covariates.columns if covariates is not None else []
    kinship_inds = kinship_matrix.index if kinship_matrix is not None else []

    # Find common individuals across all available datasets
    common_inds = set(gene_inds) & set(vntr_inds)  
    if covariates is not None:
        common_inds &= set(cov_inds)
    if kinship_matrix is not None:
        common_inds &= set(kinship_inds)

    # Convert to sorted list using natural sorting
    sorted_inds = natsorted(common_inds)
    # Concatenate metadata columns with the sorted individual columns
    sorted_cols = metadata_cols.tolist() + sorted_inds

    # Align gene expression and VNTR matrices
    gene_expression = expression_matrix.loc[:, sorted_cols]  # .dropna()
    vntr_values = vntr_matrix.loc[:, sorted_cols]  # .dropna()
    print('head(gene_expression_matrix)')
    print(f'Gene Expression matrix dimensions: {gene_expression.shape}')
    print(gene_expression.iloc[:10,:10].to_markdown())
    print('head(vntr-values_matrix)')
    print(f'vntr_values matrix dimensions: {vntr_values.shape}')
    print(vntr_values.iloc[:10,:10].to_markdown())

    # Align covariates (keeping only relevant rows and sorted columns)
    if covariates is not None:
        allcovariates = allcovariates.loc[:, sorted_inds]
        covariates = covariates.loc[:, sorted_inds]
        print('head(filtered_covariate_matrix)')
        print(f'Filtered Covariates matrix dimensions: {covariates.shape}')
        print(covariates.iloc[:10,:10].to_markdown())
        print('head(all_covariate_matrix)')
        print(f'All Covariates matrix dimensions: {allcovariates.shape}')
        print(allcovariates.iloc[:,:10].to_markdown())

    # Align kinship matrix (both rows and columns)
    if kinship_matrix is not None:
        kinship_matrix = kinship_matrix.loc[sorted_inds, sorted_inds]
        print('head(kinship_matrix)')
        print(f'Kinship matrix dimensions: {kinship_matrix.shape}')
        print(kinship_matrix.iloc[:10,:10].to_markdown())
    print('Finished sorting')

    return gene_expression, vntr_values, covariates, allcovariates, kinship_matrix


def filter_covariates(covariate_matrix, desired_cov):
    """
    Filters the covariate matrix to keep only the rows where the index matches a covariate in desired_cov.

    Parameters:
        covariate_matrix (pd.DataFrame): Covariate matrix with covariate names as row indices.
        desired_cov (list): List of covariate IDs to keep.

    Returns:
        pd.DataFrame: Filtered covariate matrix with only the desired rows.
    """
    # Ensure desired_cov values are strings (in case of type mismatch)
    desired_cov = [str(cov) for cov in desired_cov]
    
    # Filter based on row index
    filtered_cov = covariate_matrix.loc[covariate_matrix.index.intersection(desired_cov)]
    
    return filtered_cov


def load_data(gxpfile, vntrfile, covfile, kinfile, desired_covariates=None):
    # input data assumed to be in relative directory ./inputs/inputfile.bed
    expression_matrix = pd.read_csv(gxpfile, sep="\t", dtype={"#chr": str, "pid": str, "gid": str, "strand":str})
    vntr_matrix = pd.read_csv(vntrfile, sep="\t", dtype={"#chr": str, "pid": str, "gid": str, "strand":str})
    try:
        allcovariates = pd.read_csv(covfile, index_col=0, sep="\t") # , dtype={"ID": str}
        if desired_covariates:
            covariates = filter_covariates(allcovariates, desired_covariates)  # immediately filter the covariates to only the desired ones
    except FileNotFoundError:
        covariates = None
    
    try:
        kinship_matrix = pd.read_csv(kinfile, index_col=0, sep="\t")  # , dtype=str
    except FileNotFoundError:
        kinship_matrix = None
    
    expression_matrix.set_index("gid", drop=False, inplace=True)
    vntr_matrix.set_index("gid", drop=False, inplace=True)

    # Convert numerical columns to float
    # expression_matrix = expression_matrix.apply(pd.to_numeric, errors='coerce')
    # vntr_matrix = vntr_matrix.apply(pd.to_numeric, errors='coerce')
    # if covariates is not None:
    #     allcovariates = allcovariates.apply(pd.to_numeric, errors='coerce')
    #     covariates = covariates.apply(pd.to_numeric, errors='coerce')
    # if kinship_matrix is not None:
    #     kinship_matrix = kinship_matrix.apply(pd.to_numeric, errors='coerce')

    # sort each matrix to ensure columns (individuals) are represented in the same order
    expression_matrix, vntr_matrix, covariates, allcovariates, kinship_matrix = sort_matrices(expression_matrix, vntr_matrix, covariates, allcovariates, kinship_matrix)

    return expression_matrix, vntr_matrix, covariates, allcovariates, kinship_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workingdir", type=str, required=True, help="full path to working directory containing ./inputs which contains input files")
    parser.add_argument("--fdr", type=float, default=0.05, help="FDR threshold for multiple testing correction")
    parser.add_argument("--window", type=int, default=1000000, help="Window size for VNTR search (default: 1Mb)")
    parser.add_argument("--genexpfile", type=str, default='gene_expression.tsv', help="filename of file in ./inputs :Bed like file, format (tab separated): #chr    start   end     pid     gid     strand  indiv1  indiv2  indiv3...")
    parser.add_argument("--vntrfile", type=str, default='vntr_lengths.tsv', help="filename of file in ./inputs :Bed like file, format (tab separated): #chr    start   end     pid     gid     strand  indiv1  indiv2  indiv3...")
    parser.add_argument("--covarfile", type=str, default='covariates.tsv', required=False, help="filename of file in ./inputs : tab separated file: covariate  indiv1  indiv2  indiv3...")
    parser.add_argument("--kinshipfile", type=str, default='kinship_matrix.tsv', required=False, help="filename of file in ./inputs : tab separated file, symmetrical matrix: indiv1  indiv2  indiv3...")
    parser.add_argument("--minsamples", type=int, default=20, help="int value, if the number of common ids between gene expression matrix and vntr length matrix is < args.minsamples then the locus will be skipped")
    parser.add_argument("--geneid", type=str, required=False, help="Gene ID for specific analysis")
    parser.add_argument("--vntrid", type=str, required=False, help="VNTR ID for specific analysis")
    parser.add_argument("--xaxislabel", type=str, default="VNTR Mean Allele Length", required=False, help="Label for x axis if plotting")
    parser.add_argument("--desired_covs", nargs="+", type=str, required=False, help="List of covariate IDs to include (space-separated).")  # Add an argument to accept multiple strings as a list
    args = parser.parse_args()

    outdir = os.path.join(args.workingdir, 'output')
    print(f'Output directory for results: {outdir}')
    # make output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    output_file = os.path.join(args.workingdir, outdir, "gene_vntr_association_results.tsv")

    print(f'Desired covariates: {args.desired_covs}')
    # input files assumed to be in relative directory ./inputs/input_file.bed
    gxpfile, vntrfile, covfile, kinfile = os.path.join(args.workingdir, 'inputs', args.genexpfile), os.path.join(args.workingdir, 'inputs', args.vntrfile), os.path.join(args.workingdir, 'inputs', args.covarfile), os.path.join(args.workingdir, 'inputs', args.kinshipfile)
    print(f'Reading in matrices from: {args.genexpfile} and {args.vntrfile}')
    expression_matrix, vntr_matrix, covariates, allcovariates, kinship_matrix = load_data(gxpfile, vntrfile, covfile, kinfile, desired_covariates=args.desired_covs)
    print(f'Finished reading in matrices')
    
    if args.geneid and args.vntrid:
        print(f'Plotting relevant regressions between gene expression from gene: {args.geneid} and x-axis value: {args.vntrid}')
        plot_regression(args.geneid, args.vntrid, expression_matrix, vntr_matrix, covariates, allcovariates, kinship_matrix, args.workingdir, args.minsamples, args.xaxislabel)
    elif args.geneid and not args.vntrid:
        # make outdir have gene name
        outdir = outdir + f'_{args.geneid}'
        # perform cis association for the specified gene only
        print(f'Performing cis association analysis for gene: {args.geneid}')
        expression_matrix_gene = expression_matrix.loc[[args.geneid]]
        gene_vntr_dict = find_cis_vntrs(expression_matrix_gene, vntr_matrix, args.window)
        print(f'Found {len(gene_vntr_dict)} cis comparisons for gene {args.geneid}')
        if len(gene_vntr_dict) == 0:
            print(f'No cis VNTRs found for gene {args.geneid} within window {args.window}. Exiting.')
            sys.exit(1)
        print(f'Starting to run cis associations for gene {args.geneid}:')

        # only write to file if cisfile was not specified, writes to outdir, but can be moved to inputs dir for later runs of the program
        write_cis_vntrs_to_file(gene_vntr_dict, outdir, filename=f"cis_vntrs.{args.geneid}.tsv")
        output_file = os.path.join(outdir, f"gene_vntr_association_results.{args.geneid}.tsv")
        print(f'Number of genes: {len(gene_vntr_dict)}, average number of cis VNTRs {round(np.mean([len(vntrs) for vntrs in list(gene_vntr_dict.values())]), 2)}. Estimated number of cis-models: {round(len(gene_vntr_dict) * np.mean([len(vntrs) for vntrs in list(gene_vntr_dict.values())]), 2)}')

        run_association_analysis(gene_vntr_dict, expression_matrix, vntr_matrix, covariates, kinship_matrix, args.minsamples, output_file)
    else:
        cisfile = os.path.join(args.workingdir, outdir, 'cis_vntrs.tsv')
        if os.path.isfile(cisfile):
            print(f'Loading all cis gene_VNTR comparisons from: {cisfile}')
            gene_vntr_dict = read_in_cis_vntrs(cisfile)
        else:
            print(f'Finding all cis comparisons between {args.genexpfile} and {args.vntrfile}')
            gene_vntr_dict = find_cis_vntrs(expression_matrix, vntr_matrix, args.window)
        print(f'Number of genes: {len(gene_vntr_dict)}, average number of cis VNTRs {round(np.mean([len(vntrs) for vntrs in list(gene_vntr_dict.values())]), 2)}. Estimated number of cis-models: {round(len(gene_vntr_dict) * np.mean([len(vntrs) for vntrs in list(gene_vntr_dict.values())]), 2)}')
        write_cis_vntrs_to_file(gene_vntr_dict, args.workingdir, filename="cis_vntrs.tsv")
        print(f'Starting to run all cis associations:')
        run_association_analysis(gene_vntr_dict, expression_matrix, vntr_matrix, covariates, kinship_matrix, args.minsamples, output_file)
        # filter and perform multiple tesing correction on output
        read_result_table_and_filter(output_file, args.workingdir, fdr_threshold=args.fdr)

if __name__ == "__main__":
    # USAGE:
    # conda activate py3.10
    # cd /cluster/work/pausch/vmason/analyses/modeling/linear/eQTLCohort_MeanLength
    # python LM.py --window 2000000 --genexpfile TPM_gene_testis_FILTERED_UNNORM.tsv --vntrfile TRDB_MatrixMeanAlleleLength.txt --covarfile RNA_cov_testis.txt --desired_covs PC1 PC2 PC3 PC4 PC5 rin_testis age genotypePC3 genotypePC4 genotypePC5 genotypePC6 genotypePC7 --workingdir /cluster/work/pausch/vmason/analyses/modeling/linear/eQTLCohort_MeanLength

    main()

