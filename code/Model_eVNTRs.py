
from multiprocessing.pool import Pool

# from Memory_Monitor_Deep import DeepMemoryMonitor
# monitor = DeepMemoryMonitor(interval=100)
# monitor.start()

import os
import gc
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import chi2, pearsonr
from natsort import natsorted
from multiprocessing import Pool
# from limix.qtl import scan
from glimix_core.lmm import LMM
from collections import namedtuple
from numpy_sugar.linalg import economic_qs_linear
from scipy.linalg import solve #, LinAlgError, pinv
# from scipy.optimize import minimize
from sklearn.decomposition import PCA
# from numpy.linalg import inv, slogdet
from statsmodels.regression.linear_model import GLS
from statsmodels.stats.multitest import multipletests


LMMResults = namedtuple("LMMResults", [
                "ll0", "ll1", "lmm1_betas", "lr_stat", "sigma_g2", "sigma_e2", "h2", "scale",
                "var_total", "r2_modeled", "r2_fixed", "r2_genetic", "r2_residual", 
                "p_value", "lambda_pen"
                ])


def read_result_table_and_filter(output_file, workingdir, fdr_threshold=0.5):
    # output_file is the resulting table after running_linear modeling associations
    # working dir is the base directory from which all output will be directed
    # Read the file back in as a DataFrame
    results_df = pd.read_csv(output_file, sep="\t", dtype={"geneid": str, "VNTRid": str, "raw_pvalue": float, "AlleleBLUEs": str, "AlleleFrequencies": str})
    print(f'About raw result matrix: {results_df.describe()}')

    # remove loci with standardizedbeta values = 0. standbeta = 0 when standard deviation of VNTR mean length is 0. (i.e. not variable)
    # results_df = results_df[results_df["StandardizedBeta"] > 0]
    # print(f'About result matrix after removing non-variable X-axes: {results_df.shape}')

    # Multiple testing corrections
    results_df["corrected_pvalue_FDR"] = multipletests(results_df["raw_pvalue"], method="fdr_bh", alpha=fdr_threshold)[1]
    results_df["corrected_pvalue_Bonferroni"] = multipletests(results_df["raw_pvalue"], method="bonferroni")[1]
    print('Multiple testing correction completed.')

    # Save updated results with corrections
    output_file = os.path.join(workingdir, "gene_vntr_association_results.MultiTestCorrect.tsv")
    results_df.to_csv(output_file, sep="\t", index=False)
    print(output_file)

    output_file = os.path.join(workingdir, "best_gene_vntr_associations.MultiTestCorrect.tsv")
    # Select the VNTR(s) with the lowest p-value per gene
    best_results = results_df.loc[results_df.groupby("geneid")["raw_pvalue"].idxmin()]
    best_results.to_csv(output_file, sep="\t", index=False)
    print(f'Shape of the best VNTR per gene result matrix (does not require significance): {best_results.shape}')
    print(output_file)

    output_file = os.path.join(workingdir, "significant_gene_vntr_associations.MultiTestCorrect.tsv")
    # Select only significant associations based on FDR correction
    significant_results = results_df[results_df["corrected_pvalue_FDR"] < 0.05]
    significant_results.to_csv(output_file, sep="\t", index=False)
    print(f'Shape of significant result matrix: {significant_results.shape}')
    print(output_file)

    return

def orthogonalize_Z_relative_to_X(Z: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """
    Orthogonalize allele-dosage matrix Z to the fixed-effect design X.

    Goal:
        Construct Z_ortho = (I - P_X) Z such that X^T Z_ortho = 0
        where P_X is the projection onto col(X).

    Numerically stable computation via QR of X:
        X = Q R  (with Q: n×p, R: p×p)
        P_X = Q Q^T
        Z_ortho = (I - Q Q^T) Z = Z - Q (Q^T Z)

    Parameters
    ----------
    Z : pd.DataFrame, shape (n, m)
        Raw allele dosages (0/1/2). Index: individuals, Columns: allele IDs.
    X : pd.DataFrame, shape (n, p)
        Fixed effects (include intercept). Index must match Z’s index.

    Returns
    -------
    Z_ortho : pd.DataFrame, shape (n, m)
        Z residualized against X, preserving the same row/column labels.
    """
    # Align rows
    if not Z.index.equals(X.index):
        raise ValueError("Z and X must have the same individuals and order (matching index).")

    # QR decomposition of X (economy)
    X_np = X.values
    Z_np = Z.values
    Q, _ = np.linalg.qr(X_np, mode='reduced')   # Q: n×p with orthonormal columns

    # Project Z onto col(X) and subtract: Z_ortho = Z - Q(Q^T Z)
    Z_ortho_np = Z_np - Q @ (Q.T @ Z_np)

    return pd.DataFrame(Z_ortho_np, index=Z.index, columns=Z.columns)


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


def perform_multilinear(gene_expression, vntr_values, covariates, label=None, common_inds=None, add_intercept=True):
    """
    Multiple linear regression with covariates + VNTR.
    Returns residualized Y (covariates removed, VNTR preserved).

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

    # Subset shared individuals
    if common_inds is not None:
        Y = gene_expression[common_inds]
        X = vntr_values[common_inds]
    else:
        Y = gene_expression
        X = vntr_values

    ## # Covariates: rows=covariates, columns=individuals → transpose
    ## covs = covariates.loc[:, Y.index].T
    print("covariates.shape =", covariates.shape)
    print("covariates.index[:5] =", covariates.index[:5])
    print("covariates.columns[:5] =", covariates.columns[:5])
    covs = covariates.loc[Y.index, :]    # subset rows, not columns

    # Label VNTR column
    vntr_label = label if label else X.name
    X_df = X.to_frame(name=vntr_label)

    # Build design matrix
    X_full = pd.concat([covs, X_df], axis=1)

    # Add intercept *before* regression
    if add_intercept:
        X_full = sm.add_constant(X_full)

    # Fit model
    model = sm.OLS(Y, X_full).fit()

    # Extract stats specifically for the VNTR term
    beta = model.params[vntr_label]
    se = model.bse[vntr_label]
    pvalue = model.pvalues[vntr_label]
    variance_explained = model.rsquared

    # -----------------------------
    # Compute added-variable plot residuals:
    # y_resid = Y after removing all covariates (including intercept if present)
    # X_resid = X VNTR after removing all covariates (except itself)
    # -----------------------------
    cov_only = X_full.drop(columns=[vntr_label])
    y_resid = sm.OLS(Y, cov_only).fit().resid
    x_resid = sm.OLS(X_df[vntr_label], cov_only).fit().resid

    return Y, X, variance_explained, beta, se, pvalue, model, y_resid, x_resid


def perform_simplelinear(gene_expression, vntr_values, common_inds=None, add_intercept=True):
    
    if common_inds is not None:
        Y = gene_expression[common_inds]
        X = vntr_values[common_inds]
    else:
        Y = gene_expression
        X = vntr_values

    if add_intercept: # shouldn't be used because it's simple linear regression and we want only one covariate
        X_full = sm.add_constant(X)
    else:
        X_full = X
    
    Y_values = Y.values if hasattr(Y, 'values') else Y
    model = sm.OLS(Y_values, X_full).fit()
    beta, se, pvalue = model.params.iloc[-1], model.bse.iloc[-1], model.pvalues.iloc[-1]
    variance_explained = model.rsquared
    
    return Y, X, variance_explained, beta, se, pvalue, model


def diagnose_matrix(matrix, save_path,
                    cond_threshold=1e12,
                    small_sv_rel_tol_factor=10.0,
                    max_small_svs_report=3,
                    show_component_threshold=0.25):
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    Diagnose a matrix (numpy array / pandas DataFrame) and save a detailed report.

    Parameters
    ----------
    matrix : np.ndarray or pd.DataFrame or array-like
        Input matrix to diagnose. 1D arrays are treated as a single-row matrix.
    save_path : str
        File path where the report will be written (text). If extension is '.json'
        the numeric diagnostics will also be saved in JSON format.
    cond_threshold : float, optional
        Condition-number threshold above which the matrix is considered "ill-conditioned"
        (default 1e12).
    small_sv_rel_tol_factor : float, optional
        Multiplier for the SVD-based numeric tolerance used to identify "very small"
        singular values (default 10).
    max_small_svs_report : int, optional
        How many smallest singular values to analyze for column/row involvement.
    show_component_threshold : float in (0,1), optional
        When pointing to columns/rows likely involved in linear dependence,
        report components whose absolute weight is >= this fraction of the max abs weight
        in the singular vector (default 0.25).
    Returns
    -------
    dict
        A dictionary of computed diagnostics (also written to save_path).
    """
    # Normalize matrix into numpy 2D array and gather meta
    if isinstance(matrix, pd.DataFrame):
        df = matrix.copy()
        col_names = [str(c) for c in df.columns]
        row_names = [str(i) for i in df.index]
        numeric_df = df.apply(pd.to_numeric, errors='coerce')  # non-numeric -> NaN
        A = numeric_df.values.astype(float)
        original_type = 'pandas.DataFrame'
    else:
        A = np.asarray(matrix)
        original_type = type(matrix).__name__
        if A.ndim == 1:
            A = A.reshape(1, -1)
        col_names = [f'c{j}' for j in range(A.shape[1])]
        row_names = [f'r{i}' for i in range(A.shape[0])]

    # Prepare output directory
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    m, n = A.shape
    mn = min(m, n)
    results = {}
    results['shape'] = (m, n)
    results['original_type'] = original_type

    # NaN / Inf checks
    n_nan = int(np.isnan(A).sum())
    n_inf = int(np.isinf(A).sum())
    results['n_nan'] = n_nan
    results['n_inf'] = n_inf

    # Basic numeric stats (on finite entries)
    finite_mask = np.isfinite(A)
    if finite_mask.any():
        finite_vals = A[finite_mask]
        results['finite_count'] = int(finite_vals.size)
        results['finite_min'] = float(np.min(finite_vals))
        results['finite_max'] = float(np.max(finite_vals))
        results['finite_mean'] = float(np.mean(finite_vals))
        results['finite_std'] = float(np.std(finite_vals))
    else:
        results['finite_count'] = 0
        results['finite_min'] = results['finite_max'] = results['finite_mean'] = results['finite_std'] = None

    # Quick exits for empty or completely non-finite matrix
    if m == 0 or n == 0:
        text = f"Matrix is empty with shape {m}x{n}. Nothing to diagnose.\n"
        with open(save_path, 'w') as f:
            f.write(text)
        results['verdict'] = 'empty'
        return results

    if results['finite_count'] == 0:
        text = ("Matrix contains no finite entries (all NaN or Inf). "
                "Cannot compute linear algebra diagnostics until you remove/impute non-finite values.\n")
        with open(save_path, 'w') as f:
            f.write(text)
        results['verdict'] = 'non_finite_only'
        return results

    # Build a DataFrame copy for easy duplicate/constant detection
    df_for_checks = pd.DataFrame(A, columns=col_names, index=row_names)

    # Zero/near-zero rows/columns (norm-based)
    row_norms = np.linalg.norm(A, axis=1)
    col_norms = np.linalg.norm(A, axis=0)
    # tolerance for "zero" based on data magnitude and machine eps
    eps = np.finfo(float).eps
    tol_norm = max(np.max(np.abs(A)), 1.0) * max(m, n) * eps * 100.0
    zero_rows = [row_names[i] for i in np.where(row_norms <= tol_norm)[0].tolist()]
    zero_cols = [col_names[j] for j in np.where(col_norms <= tol_norm)[0].tolist()]
    results['zero_rows'] = zero_rows
    results['zero_cols'] = zero_cols

    # Constant columns/rows (zero variance ignoring NaN)
    const_cols = []
    const_rows = []
    # use nanvar to ignore NaN and ddof=0
    col_var = np.nanvar(A, axis=0)
    row_var = np.nanvar(A, axis=1)
    col_const_mask = (col_var <= tol_norm)
    row_const_mask = (row_var <= tol_norm)
    const_cols = [col_names[j] for j in np.where(col_const_mask)[0].tolist()]
    const_rows = [row_names[i] for i in np.where(row_const_mask)[0].tolist()]
    results['constant_columns'] = const_cols
    results['constant_rows'] = const_rows

    # Duplicate rows/columns (exact duplicates on the numeric values; NaNs count as "different")
    dup_cols = df_for_checks.T.duplicated(keep=False)
    duplicate_columns = [col for col, dup in zip(col_names, dup_cols) if dup]
    dup_rows = df_for_checks.duplicated(keep=False)
    duplicate_rows = [row for row, dup in zip(row_names, dup_rows) if dup]
    results['duplicate_columns'] = duplicate_columns
    results['duplicate_rows'] = duplicate_rows

    # Compute SVD and rank/condition if there are no NaN/Inf (SVD won't like NaN/Inf)
    numeric_ok = (n_nan == 0) and (n_inf == 0)
    if not numeric_ok:
        # We can still attempt some diagnostics (rank on rows/cols w/o NaNs), but it's safer to report and stop
        text = []
        text.append("Matrix diagnostics report\n")
        text.append("========================\n\n")
        text.append(f"Shape: {m} x {n}\n")
        text.append(f"Original type: {original_type}\n")
        text.append(f"Numeric issue: matrix contains {n_nan} NaN(s) and {n_inf} Inf(s).\n")
        text.append("Many linear algebra diagnostics cannot be reliably computed while non-finite values exist.\n")
        text.append("Recommendation: remove or impute NaN/Inf values (e.g., df.dropna(), df.fillna(), or pd.to_numeric(..., errors='coerce') + imputation),\n")
        text.append("or compute diagnostics on a finite submatrix.\n\n")
        # Add the detected zero/constant/duplicate info we already have:
        if zero_rows or zero_cols:
            text.append(f"Zero (near-zero) rows: {zero_rows}\n")
            text.append(f"Zero (near-zero) columns: {zero_cols}\n")
        if const_cols or const_rows:
            text.append(f"Constant columns: {const_cols}\n")
            text.append(f"Constant rows: {const_rows}\n")
        if duplicate_columns or duplicate_rows:
            text.append(f"Duplicate columns: {duplicate_columns}\n")
            text.append(f"Duplicate rows: {duplicate_rows}\n")
        text.append("\nVerdict: BAD (contains non-finite entries)\n")
        with open(save_path, 'w') as f:
            f.write(''.join(text))
        results['verdict'] = 'bad_non_finite'
        return results

    # Now numeric computations
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        # fallback - use scipy if available? but for now report failure
        text = ("SVD failed (np.linalg.svd raised LinAlgError). "
               "This can happen for extremely large or pathological matrices.\n")
        with open(save_path, 'w') as f:
            f.write(text)
        results['verdict'] = 'svd_failed'
        return results

    # singular values & condition
    s_list = s.tolist()
    results['singular_values'] = s_list
    # matrix rank (numpy's matrix_rank uses SVD internally with a tolerance)
    rank = int(np.linalg.matrix_rank(A))
    results['rank'] = rank
    results['rank_deficiency'] = int(mn - rank)

    condition_number = np.inf if s[-1] == 0 else float(s[0] / s[-1])
    results['condition_number'] = condition_number

    # Determine thresholds for "small" singular values
    sv_tol = max(A.shape) * s[0] * eps * small_sv_rel_tol_factor
    small_sv_indices = [i for i, sv in enumerate(s[::-1]) if sv <= sv_tol]  # positions from smallest
    # convert to true indices (0..len-1) for smallest values: e.g., last indices of s
    small_sv_true_indices = [(len(s) - 1 - i) for i in range(min(len(s), max_small_svs_report)) if s[len(s)-1-i] <= sv_tol]
    results['sv_threshold'] = float(sv_tol)
    results['small_singular_value_indices'] = small_sv_true_indices
    results['small_singular_values'] = [float(s[i]) for i in small_sv_true_indices]

    # If square, compute determinant (slogdet for stability)
    det_info = {}
    if m == n:
        sign, logabsdet = np.linalg.slogdet(A)
        # attempt to compute determinant in normal range if possible for small matrices
        try:
            det_val = float(np.linalg.det(A))
        except Exception:
            det_val = None
        det_info = {'slogdet_sign': int(sign), 'slogdet_logabs': float(logabsdet), 'det': det_val}
    results['determinant_info'] = det_info

    # Reasoning: attempt to identify columns/rows involved in linear dependencies using smallest singular vectors
    linear_dep_columns = []
    linear_dep_rows = []
    # number of smallest singular vectors to inspect
    K = min(max_small_svs_report, len(s))
    for k in range(1, K+1):
        idx = len(s) - k  # index of k-th smallest
        sv = s[idx]
        vec_cols = Vt[idx, :]  # shape (n,)
        vec_rows = U[:, idx]   # shape (m,)
        abs_cols = np.abs(vec_cols)
        abs_rows = np.abs(vec_rows)
        if abs_cols.max() > 0:
            comp_thresh_cols = show_component_threshold * abs_cols.max()
            implicated_cols = [col_names[j] for j in np.where(abs_cols >= comp_thresh_cols)[0].tolist()]
        else:
            implicated_cols = []
        if abs_rows.max() > 0:
            comp_thresh_rows = show_component_threshold * abs_rows.max()
            implicated_rows = [row_names[i] for i in np.where(abs_rows >= comp_thresh_rows)[0].tolist()]
        else:
            implicated_rows = []
        if implicated_cols:
            linear_dep_columns.append({'sv_index': int(idx), 'sv_value': float(sv),
                                       'implicated_columns': implicated_cols})
        if implicated_rows:
            linear_dep_rows.append({'sv_index': int(idx), 'sv_value': float(sv),
                                    'implicated_rows': implicated_rows})
    results['linear_dependency_columns'] = linear_dep_columns
    results['linear_dependency_rows'] = linear_dep_rows

    # Build human-readable report
    lines = []
    lines.append("Matrix diagnostics report\n")
    lines.append("=========================\n\n")
    lines.append(f"Shape: {m} x {n}\n")
    lines.append(f"Original type: {original_type}\n\n")

    lines.append("---- Basic numeric checks ----\n")
    lines.append(f"Total elements: {m*n}\n")
    lines.append(f"Finite elements: {results['finite_count']}\n")
    lines.append(f"NaNs: {n_nan}, Infs: {n_inf}\n")
    lines.append(f"Finite min / max / mean / std: {results['finite_min']} / {results['finite_max']} / {results['finite_mean']} / {results['finite_std']}\n\n")

    lines.append("---- Structural checks ----\n")
    if zero_rows:
        lines.append(f"Zero (near-zero) rows (norm <= tol): {zero_rows}\n")
    if zero_cols:
        lines.append(f"Zero (near-zero) columns (norm <= tol): {zero_cols}\n")
    if const_cols:
        lines.append(f"Constant columns (zero variance): {const_cols}\n")
    if const_rows:
        lines.append(f"Constant rows (zero variance): {const_rows}\n")
    if duplicate_columns:
        lines.append(f"Duplicate columns (exact on numeric values): {duplicate_columns}\n")
    if duplicate_rows:
        lines.append(f"Duplicate rows (exact on numeric values): {duplicate_rows}\n")
    if not (zero_rows or zero_cols or const_cols or const_rows or duplicate_columns or duplicate_rows):
        lines.append("No zero/constant/duplicate rows or columns detected.\n")
    lines.append("\n")

    lines.append("---- Linear algebra diagnostics ----\n")
    lines.append(f"Numeric rank (np.linalg.matrix_rank): {rank} (expected max rank = {mn})\n")
    if rank == mn:
        lines.append("Rank status: FULL RANK with respect to min(rows,cols).\n")
    else:
        lines.append(f"Rank status: RANK-DEFICIENT (deficiency = {mn - rank}).\n")

    lines.append(f"Singular values (descending, first 10 shown): {s_list[:10]}\n")
    lines.append(f"Computed condition number (s[0]/s[-1]): {condition_number}\n")
    lines.append(f"SVD-based small singular value threshold: {sv_tol}\n")
    if results['small_singular_values']:
        lines.append(f"Very small singular values (<= threshold): {results['small_singular_values']}\n")
    else:
        lines.append("No singular values smaller than the automatic threshold were detected.\n")

    if m == n:
        lines.append("\n---- Determinant (square matrix) ----\n")
        if det_info:
            lines.append(f"slogdet sign: {det_info['slogdet_sign']}, slogdet(log|det|): {det_info['slogdet_logabs']}\n")
            if det_info['det'] is not None:
                lines.append(f"determinant (direct): {det_info['det']}\n")
            else:
                lines.append("determinant exact value is too small/large to reliably show; use slogdet above.\n")

    # Ill-conditioning / near singularity
    lines.append("\n---- Conditioning verdict ----\n")
    verdict_reasons = []
    if condition_number == np.inf:
        verdict_reasons.append("matrix is singular (smallest singular value is exactly zero).")
    elif condition_number > cond_threshold:
        verdict_reasons.append(f"matrix is numerically ill-conditioned (condition number {condition_number:.3e} > threshold {cond_threshold}).")
    if rank < mn:
        if m == n:
            verdict_reasons.append(f"matrix is rank-deficient: rank {rank} < {mn} -> singular (non-invertible).")
        else:
            verdict_reasons.append(f"matrix is rank-deficient: rank {rank} < min(rows,cols)={mn}.")
    if n_nan > 0 or n_inf > 0:
        verdict_reasons.append("matrix contains NaN/Inf which prevents reliable linear-algebra computations.")
    if zero_cols or const_cols:
        verdict_reasons.append("contains zero or constant columns (these create linear dependencies).")
    if duplicate_columns:
        verdict_reasons.append("contains duplicate columns (exact linear dependencies).")
    if linear_dep_columns:
        # pick a short description
        implicated = []
        for entry in linear_dep_columns:
            implicated.extend(entry['implicated_columns'])
        implicated = list(dict.fromkeys(implicated))  # unique preserve order
        verdict_reasons.append(f"smallest singular vectors implicate columns likely involved in linear dependence: {implicated}")
    if not verdict_reasons:
        verdict = "GOOD: matrix appears numerically well-behaved (full rank and well-conditioned)."
        lines.append(verdict + "\n")
        results['verdict'] = 'good'
    else:
        verdict = "BAD: " + " ".join(verdict_reasons)
        lines.append(verdict + "\n")
        results['verdict'] = 'bad'
        results['verdict_reasons'] = verdict_reasons

    lines.append("\n---- Suggested fixes / next steps ----\n")
    suggestions = []
    suggestions.append("1) If there are NaN/Inf: impute or drop those rows/columns (e.g., df.fillna(), df.dropna()).")
    suggestions.append("2) Remove constant or zero columns/rows; these add no information.")
    suggestions.append("3) Remove exact duplicate columns/rows or merge them.")
    suggestions.append("4) If matrix is ill-conditioned (but not exactly singular), consider regularization (e.g., add lambda*I) or use ridge regression instead of plain least squares.")
    suggestions.append("5) Use SVD / pseudo-inverse (np.linalg.pinv) to solve linear systems for rank-deficient or ill-conditioned matrices.")
    suggestions.append("6) Center and scale columns (standardization) to reduce numeric range issues; rescale data if values differ by many orders of magnitude.")
    suggestions.append("7) Use QR with column pivoting to find an independent set of columns (scipy.linalg.qr with pivoting can help).")
    suggestions.append("8) If a small number of columns are implicated in linear dependence, try dropping/combining those and re-check.")
    for s in suggestions:
        lines.append(s + "\n")

    # More detailed info on implicated columns/rows
    if linear_dep_columns:
        lines.append("\n---- Detailed small-singular-vector analysis (columns) ----\n")
        for entry in linear_dep_columns:
            lines.append(f"Singular index {entry['sv_index']} (sv={entry['sv_value']}): implicated columns {entry['implicated_columns']}\n")
    if linear_dep_rows:
        lines.append("\n---- Detailed small-singular-vector analysis (rows) ----\n")
        for entry in linear_dep_rows:
            lines.append(f"Singular index {entry['sv_index']} (sv={entry['sv_value']}): implicated rows {entry['implicated_rows']}\n")

    # Write text file
    report_text = ''.join(lines)
    with open(save_path, 'w') as f:
        f.write(report_text)

    # Also write JSON of numeric diagnostics if user requested a .json path or companion .json
    _, ext = os.path.splitext(save_path)
    if ext.lower() == '.json':
        # numeric results are already in results
        with open(save_path, 'w') as f:
            json.dump({'report_text': report_text, 'diagnostics': results}, f, indent=2)
    else:
        # write companion JSON file next to the text file for programmatic access
        json_path = save_path + '.diagnostics.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def export_matrix(matrix, save_path, fmt='csv'):
    """
    import numpy as np
    import pandas as pd
    import os

    Export a NumPy array or Pandas DataFrame to a file.

    Parameters
    ----------
    matrix : np.ndarray or pd.DataFrame
        The matrix to export.
    save_path : str
        The file path (including name) where the matrix should be saved.
    fmt : str, optional
        File format ('csv', 'txt', 'xlsx', or 'npy'), default is 'csv'.

    Returns
    -------
    str
        The absolute path of the saved file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Handle Pandas DataFrame
    if isinstance(matrix, pd.DataFrame):
        if fmt == 'csv':
            matrix.to_csv(save_path, index=False)
        elif fmt == 'xlsx':
            matrix.to_excel(save_path, index=False)
        elif fmt == 'txt':
            np.savetxt(save_path, matrix.values, fmt='%s')
        else:
            raise ValueError(f"Unsupported format '{fmt}' for DataFrame.")
    
    # Handle NumPy array
    elif isinstance(matrix, np.ndarray):
        if fmt == 'csv':
            np.savetxt(save_path, matrix, delimiter=',', fmt='%s')
        elif fmt == 'txt':
            np.savetxt(save_path, matrix, fmt='%s')
        elif fmt == 'npy':
            np.save(save_path, matrix)
        else:
            raise ValueError(f"Unsupported format '{fmt}' for NumPy array.")
    
    else:
        raise TypeError("Input must be a NumPy array or Pandas DataFrame.")
    
    return os.path.abspath(save_path)


def likelihood_ratio_test(null_ll, full_ll, df=1):
    """
    Perform likelihood ratio test (LRT).
    """
    lr_stat = -2 * (null_ll - full_ll)  # LRT statistic
    # Calculate p-value from chi-squared distribution
    # with degrees of freedom equal to the difference in the number of parameters
    # between the full and null models
    p_value = chi2.sf(lr_stat, df)

    return lr_stat, p_value


def center_and_standardize(Z, standardize=True):
    """Column-center and (optionally) standardize Z."""
    # Convert DataFrame to NumPy for numeric ops
    is_df = isinstance(Z, pd.DataFrame)
    if is_df:
        data = Z.to_numpy(dtype=float)
    else:
        data = np.asarray(Z, dtype=float)

    mean = data.mean(axis=0)  # per column mean dosages, 
    if standardize:
        std = data.std(axis=0, ddof=0)
        std[std == 0] = 1.0  # avoid division by zero
    else:
        std = 1.0

    result = (data - mean) / std

    # Restore DataFrame structure if input was a DataFrame
    if is_df:
        return pd.DataFrame(result, index=Z.index, columns=Z.columns)
    return result


def standardize_matrix(df):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # df: DataFrame Z with shape (n, m) with allele dosages (0,1,2) columns
    # Simple empirical standardization:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_std = scaler.fit_transform(df)          # numpy array n x m
    Z_std = pd.DataFrame(Z_std, columns=df.columns, index=df.index)
    
    return Z_std


# -------------------------
# 1) Center Z and build K
# -------------------------
def center_Z(Z_df, return_K=True, return_K_df=True):
    """
    Center columns of Z (remove column means) and build Klocus = Zc @ Zc.T.
    Parameters
    ----------
    Z_df : pandas.DataFrame (n x m)
        Original allele-dosage matrix (rows individuals, cols alleles). Values typically 0/1/2.
    return_K_df : bool
        If True, return K as pandas.DataFrame (indexed by Z_df.index).
    Returns
    -------
    Zc_df : pandas.DataFrame (n x m)
        Centered Z (columns mean = 0).
    K : pandas.DataFrame or ndarray (n x n)
        Klocus = Zc @ Zc.T
    info : dict
        Metadata including:
          - 'col_means' (pd.Series), 'col_sds' (pd.Series sample SD of centered col),
          - 'n','m', 'mean_diag_K' (trace(K)/n)
    Notes
    -----
    * Always center columns so K reflects covariance of allele deviations (not mean offsets).
    * suggestion no to scale columns to unit variance because allele frequencies vary widely and rare alleles should not be considered equally accurate measurements.
    
    """
    if not isinstance(Z_df, pd.DataFrame):
        raise ValueError("Z_df must be a pandas DataFrame (n x m).")

    n, m = Z_df.shape
    col_means = Z_df.mean(axis=0)             # Series length m
    Zc = Z_df.values - col_means.values[np.newaxis, :]   # centered numpy array (n x m)
    col_sds = np.sqrt(np.mean(Zc**2, axis=0)) # population sd (ddof=0)
    col_sds_series = pd.Series(col_sds, index=Z_df.columns)

    Zc_df = pd.DataFrame(Zc, index=Z_df.index, columns=Z_df.columns)
    
    if return_K:
        K = Zc @ Zc.T
        mean_diag_K = float(np.trace(K) / n)
        K_df = pd.DataFrame(K, index=Z_df.index, columns=Z_df.index) if return_K_df else K

    info = {
        "n": int(n),
        "m": int(m),
        "col_means": col_means,
        "col_sds": col_sds_series,
        "mean_diag_K": mean_diag_K if return_K == True else None,
        "centered": True,
        }
    if return_K == True:
        return Zc_df, K_df, info
    return Zc_df, info


# -------------------------
# 2) Fit LMM and compute R^2s
# -------------------------
_QS_NULL_CACHE = {}  # set as global variable to cache QS objects for null models of different sizes (identity matrix)
def get_QS_null(n):
    '''
    Used with function fit_lmm
    Cache for QS objects corresponding to null models (identity matrix) of different sizes, only compute once per n and reuse for speed.
    '''
    if n not in _QS_NULL_CACHE:
        _QS_NULL_CACHE[n] = economic_qs_linear(np.eye(n))
    return _QS_NULL_CACHE[n]

def fit_lmm(Y, X_df, Z_df, Z_best_var=None, reml=False, return_models=False, high_throughput=False):
    """
    from collections import namedtuple
    from scipy.stats import chi2
    from glimix_core.lmm import LMM
    from numpy_sugar.linalg import economic_qs_linear

    Fit null (X only) and full (X + Klocus) LMMs using Klocus = Zc @ Zc.T
    Zc should be centered to avoid mean-offset effects (Zc should be centered by center_Z())
    But the function will run on any Z matrix you provide.
    Compute LR test (boundary corrected), variance components and per-sample R^2 decomposition
    into fixed / genetic / residual parts that sum to 1.

    Parameters
    ----------
    Y : pandas.Series (n,)
        Outcome values (indexed same as X_df and Z_df).
    X_df : pandas.DataFrame (n x p)
        Covariates (include intercept column or include intercept in model).
    Z_df : pandas.DataFrame (n x m)
        Centered Z matrix returned by center_Z.
    Z_best_var : pandas.DataFrame (n x 1), optional
        If provided, this single-column DataFrame of the best known variant for a gene will be used to build the null model's QS object instead of the identity matrix. This allows testing if the VNTR (Z_df) provides a better fit than the best known variant association to a gene.
    reml : bool
        If True use REML; if False use ML. (Convention: ML for LRT, REML for final estimates.)
    return_models : bool
        If True, return (lmm0, lmm1) objects along with stats.

    Returns
    -------
    lr_stat, p_value, comps[, (lmm0, lmm1)]
    comps contains:
      - ll0, ll1, sigma_g2, sigma_e2, mean_diag_K, var_fixed, var_genetic, var_residual, var_total
      - r2_fixed, r2_genetic, r2_residual (sum to 1)
        - r2_genetic = (np.trace(K)/n * sigma_g2) / (var_fixed + (np.trace(K)/n * sigma_g2) + sigma_e2)
      - r2_locus_param (sigma_g2/(sigma_g2+sigma_e2)) only meaningful if mean_diag_K == 1
      - lambda_pen = sigma_e2 / sigma_g2

    https://github.com/limix/glimix-core/blob/master/doc/lmm.rst
    https://glimix-core.readthedocs.io/en/latest/lmm.html
    https://glimix-core.readthedocs.io/en/latest/_autosummary/glimix_core.lmm.LMM.html
    
    """
    # input checks
    if not (isinstance(Y, pd.Series) and isinstance(X_df, pd.DataFrame) and isinstance(Z_df, pd.DataFrame)):
        raise ValueError("Y must be Series, X_df and Z_df DataFrames with matching index.")

    if not (Y.index.equals(X_df.index) and Y.index.equals(Z_df.index)):
        raise ValueError("Y, X_df and Z_df must share identical index/order.")

    try:
        n = len(Y)
        M = X_df.values
        y = Y.values.reshape(-1, 1)

        # don't need to build K, built in economic_qs_linear from Z directly
        # K = Z_df.values @ Z_df.values.T
        
        if Z_best_var is not None:
            # Make K_best from the best known variant for a gene
            # this will replace the null np.eye(n) matrix.
            # Meaning we can test if the VNTR is a better fit than the best known variant association to a gene.
            QS_null = economic_qs_linear(Z_best_var.values)
            QS_full = economic_qs_linear(Z_df.values)
            # print(f'Qs_full: {QS_full[1]},\nQs_null: {QS_null[1]}')
        else:
            # QS objects for LMM
            # use cached QS object for null model if available, otherwise compute and cache it
            QS_null = get_QS_null(n)  # economic_qs_linear(np.eye(n))
            QS_full = economic_qs_linear(Z_df.values)

        # fit null
        lmm0 = LMM(y, M, QS_null, restricted=reml)  # restricted default is False i.e. ML, for REML use True
        lmm0.fit(verbose=False)
        ll0 = lmm0.lml()

        # fit full
        # lmm1 = LMM(y, M, QS_full, restricted=restricted)  # uses Klocus as matrix for economic decomposition
        lmm1 = LMM(y, M, QS_full, restricted=reml)  # restricted default is False i.e. ML, for REML use True
        lmm1.fit(verbose=False)
        ll1 = lmm1.lml()


        lr_stat = max(0.0, -2.0 * (ll0 - ll1))
        # Uncorrected (assumes χ²₁) : pvalue_uncorrected = chi2.sf(lr_stat, 1)
        # p_value = chi2.sf(lr_stat, 1)
        # Boundary-corrected p-value (½χ²₀ + ½χ²₁ mixture)
        # LR stat (>=0) and boundary-corrected pvalue (1/2 chi2_0 + 1/2 chi2_1)
        p_value = 0.5 * (lr_stat == 0.0) + 0.5 * chi2.sf(lr_stat, 1)

        # 𝐲 ∼ 𝓝(𝚇𝜷, 𝑠((1-𝛿)𝙺 + 𝛿𝙸))

        # calculate total variance in gene expression for gene
        var_total = float(np.var(y, ddof=0))
        # fixed-effect variance (population variance of X beta predictions)
        # beta_full = np.asarray(lmm1.beta).reshape(-1)
        # fixed_pred = (M @ beta_full).reshape(-1)
        fixed_pred = lmm1.mean()  # equivalent to M @ beta_full.reshape(-1)
        var_fixed = float(np.var(fixed_pred, ddof=0))  # variance accounted for by PCs, sex, age, etc. 
        # variance components (excluding fixed effects) from full model (glimix stores them as v0, v1)
        sigma_g2 = lmm1.v0  # float(getattr(lmm1, "v0", 0.0))  # v0 = scale(1-delta)
        sigma_e2 = lmm1.v1  # float(getattr(lmm1, "v1", 0.0))  # v1 = scale(delta)
        delta_I = lmm1.delta  # delta = v1 / (v0 + v1)  # proportion of residual variance (non-fixed effect variation) (between K and I)
        delta_K = 1 - delta_I  # 1 - delta = v0 / (v0 + v1) # site based heritability (h2) (of residual variance (no fixed effects))
        scale = lmm1.scale  # scale = v0 + v1  # total residual variance (which is partitioned to K and I, v0 and v1)

        # per-sample genetic variance = sigma_g2 * mean_diag_K_after
        var_genetic = sigma_g2  # * mean_diag_K_after
        var_residual = sigma_e2
        var_modeled = var_fixed + scale  # total variance modeled, both fixed effects and LMM part with v0 + v1 # == var_fixed + var_genetic + var_residual
        if var_total <= 0:
            raise ValueError("Non-positive total variance encountered. Check inputs/model fit.")

        r2_fixed = var_fixed / var_total
        r2_genetic = var_genetic / var_total
        r2_residual = var_residual / var_total
        r2_modeled = var_modeled / var_total

        if high_throughput == True:
            # use namedtuple to save memory in high-throughput mode
            comps = LMMResults(
                ll0=float(ll0),
                ll1=float(ll1),
                lmm1_betas=lmm1.beta if return_models else None,
                lr_stat=float(lr_stat),
                sigma_g2=sigma_g2,
                sigma_e2=sigma_e2,
                h2=delta_K,
                scale=scale,
                var_total=var_total,
                r2_modeled=float(r2_modeled),
                r2_fixed=float(r2_fixed),
                r2_genetic=float(r2_genetic),
                r2_residual=float(r2_residual),
                p_value=float(p_value),
                lambda_pen=(sigma_e2 / sigma_g2) if sigma_g2 > 0 else float("inf"),
            )
        else:
            comps = {
                "n": Z_df.shape[0],
                "m": Z_df.shape[1],
                "QS_null": QS_null,
                "QS_full": QS_full,
                "ll0": float(ll0),
                "ll1": float(ll1),
                "lmm0": lmm0 if return_models else None,
                "lmm1": lmm1 if return_models else None,
                "lr_stat": float(lr_stat),
                "sigma_g2": sigma_g2,
                "sigma_e2": sigma_e2,
                "h2": delta_K,
                "scale": scale,
                "var_fixed": var_fixed,
                "var_genetic": var_genetic,
                "var_residual": var_residual,
                "var_total": var_total,
                "r2_modeled": float(r2_modeled),
                "r2_fixed": float(r2_fixed),
                "r2_genetic": float(r2_genetic),
                "r2_residual": float(r2_residual),
                "p_value": float(p_value),
                "lambda_pen": (sigma_e2 / sigma_g2) if sigma_g2 > 0 else np.inf
            }

        if high_throughput == False and return_models == True:
            del QS_null, QS_full  # free memory
            # gc.collect()
            return lr_stat, p_value, comps, lmm0, lmm1
        else:
            del lmm0, lmm1, QS_null, QS_full  # free memory
            # gc.collect()
            return lr_stat, p_value, comps
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"[Warning] LMM fitting failed: {e}")
        if return_models:
            return None, None, None, None, None
        else:
            return None, None, None

# -------------------------
# Compute OLS effects for centered Z (if non-penlized allele estimates are desired)
# -------------------------
def compute_ols_effects(Zc_df, lmm_full, Y, X_df, info=None):
    """
    Since Zc is centered, exclude the intercept from X_df, mean of each column is already = 0
    """
    if not (isinstance(Zc_df, pd.DataFrame) and isinstance(Y, pd.Series) and isinstance(X_df, pd.DataFrame)):
        raise ValueError("Zc_df, X_df and Y must be pandas objects with the same index/order.")

    if not (Y.index.equals(X_df.index) and Y.index.equals(Zc_df.index)):
        raise ValueError("Y, X_df and Zc_df must share the same index/order.")
    
    # residuals r = y - X beta (use full model betas)
    beta_full = np.asarray(lmm_full.beta).reshape(-1)
    r = (Y.values.reshape(-1) - (X_df.values @ beta_full).reshape(-1)).reshape(-1)  # length n

    # ridge solve: (Z^T Z + lambda I) u = Z^T r
    Z = Zc_df.values  # n x m
    n, m = Z.shape
    ZtZ = Z.T @ Z + (1e-8 * np.eye(m)) # no lambda ridge penalty (+ (lambda_pen * np.eye(m))) but add tiny jitter (1e-12) to avoid singular matrix errors
    Ztr = Z.T @ r

    try:
        ols = np.linalg.solve(ZtZ, Ztr)
    except np.linalg.LinAlgError:
        # jitter = 1e-8 * np.mean(np.diag(A))
        # A2 = A + np.eye(m) * jitter
        # u_hat = np.linalg.solve(A2, b)
        # if matrix is still singular, use pinv (pseudo-inverse) which gives a least-squares solution for low rank matrices
        # u_hat = np.linalg.pinv(A) @ b
        return None  # just skip if singular
        # singular matrix error can happen if sigma_g2 or sigma_e2 are very small
        # singular matrix error can also happen if there are duplicate/similar rows in Z (e.g. identical individuals)
        # singular matrix detected skipping u_hat estimation for this locus

    ols = ols.reshape(-1)
    ols_series = pd.Series(ols, index=Zc_df.columns, name="ols_effects")

    # # predicted genetic values and variances, fixed, and residuals
    # g_pred = Z @ ols  # length n
    # var_g_emp = float(np.var(g_pred, ddof=0))
    # var_fixed = float(np.var(X_df.values @ beta_full, ddof=0))
    # var_residual = float(np.var(Y.values - g_pred - (X_df.values @ beta_full), ddof=0))
    # var_total = var_fixed + var_g_emp + var_residual

    return ols_series


# -------------------------
# 3) Compute BLUPs for centered Z
# -------------------------
def compute_blups(Z_df, lmm_full, X_df, Y, high_throughput=False, info=None):
    """
    Compute BLUP allele effects u_hat for centered Z used to build the Klocus that lmm_full used.
    Also compute predicted genetic values g = Zc @ u_hat and empirical r^2 decomposition.

    # when high_throughput is True, lmm_full is expected to be a namedtuple with attributes 'comps', this is constructed in fit_lmm when high_throughput=True to save memory by not storing the full LMM objects, but just the necessary components for BLUP calculation.
        comps contains name tuple attributes:
        lmm_full.sigma_g2
        lmm_full.sigma_e2
        lmm_full.lambda_pen
        lmm_full.lmm1_betas
    # when high_throughput is False, lmm_full is expected to be the full LMM object lmm1

    Parameters
    ----------
    Z_df : pandas.DataFrame (n x m)
        Centered Z used to build K (columns mean 0).
    lmm_full : fitted glimix_core.lmm.LMM
        Full-model object returned from fit_lmm_from_centered_Z(..., return_models=True)
    X_df, Y : covariates DataFrame and outcome Series (same index/order)
    info : dict (optional)
        Metadata returned by center_Z (useful for conversions).

    Returns
    -------
    out : dict with keys:
      - 'u_hat' : pd.Series (length m) : BLUP per 1 allele copy (centered)
      - 'g_pred' : pd.Series (length n) : predicted genetic values
      - 'var_g_emp' : float : empirical var(g_pred)
      - 'u_per_sd' : pd.Series (per-SD effect) computed if info provided with col_sds
      - 'lambda'
        Returns:

    Equivalent BLUP formula:
    Mixed model:
        y = Xβ + Zu + e,
        u ~ N(0, G),     e ~ N(0, R)
        R = σ_e^2 I_n
        G = σ_g^2 I_m

    (1) "Compact BLUP":
        û = σ_g^2 Zt V^{-1} r
        where:
            V = σ_g^2 ZZt + σ_e^2 I_n
            r = y - Xβ

    (2) "Ridge BLUP"
        û = (ZtZ + λI_m)^{-1} Zt r
        where:
            λ = σ_e^2 / σ_g^2
            r = y - Xβ
            I_m = identity matrix of dimensions mxm (#alleles x #alleles)

    (3) Mixed Model Equations (MME) / ridge form:
        û = (Zt R^{-1} Z + G^{-1})^{-1} Zt R^{-1} r
           = ( (1/σ_e^2) Zt Z + (1/σ_g^2) I_m )^{-1} (1/σ_e^2) Zt r.
        In this form the **ridge penalty** is the + G^{-1} term (i.e., +(1/σ_g^2) I_m).
        where:
            R = σ_e^2 I_n
            G = σ_g^2 I_m
            r = y - Xβ
            I_m = identity matrix of dimensions mxm (#alleles x #alleles)

    
    # old # Compute BLUPs (u_hat) for alleles given a fitted full LMM (lmm_full) and the Z matrix
    # old # that was used to produce the K (Z_df). Also produce conversions:
    # old #   - 'raw_centered': u estimates in units of Z_centered (per-copy effects relative to col mean)
    # old #   - 'per_sd': u estimates in units of 1 SD of the centered column (so comparable across alleles)
    # old #   - 'used': u estimates in the exact units used to fit the model (Z_used columns)
    # old #

    # old # Parameters:
    # old #   - Z_orig_df: original Z DataFrame (n x m) as you passed into preprocess_Z
    # old #   - Z_used_df: the Z DataFrame returned by preprocess_Z (n x m)
    # old #   - lmm_full: the fitted LMM object for the full model (glimix_core.lmm.LMM)
    # old #   - X_df, Y: covariates DataFrame and Y Series (for residual r)
    # old #   - info: dict returned by preprocess_Z (recommended)
    # old #   - convert_to: "raw_centered" | "per_sd" | "used" or list/tuple of them
    
    # old # dict with keys:
    # old #     - u_used_df (effects in units of Z_used columns),
    # old #     - u_hat_df (effects in per-copy centered units),
    # old #     - u_per_sd_df (effects per 1 SD),
    # old #     - g_pred (predicted genetic value per sample),
    # old #     - var_g_emp (empirical variance of g_pred),
    # old #     - r2s_using_g_emp (r2s computed using empirical var_g)
    # old #     - lambda (sigma_e2 / sigma_g2)

    """

    if not (isinstance(Z_df, pd.DataFrame) and isinstance(Y, pd.Series) and isinstance(X_df, pd.DataFrame)):
        raise ValueError("Z_df, X_df and Y must be pandas objects with the same index/order.")

    if not (Y.index.equals(X_df.index) and Y.index.equals(Z_df.index)):
        raise ValueError("Y, X_df and Z_df must share the same index/order.")

    # extract variance components
    if high_throughput:
        # when high_throughput is True, lmm_full is expected to be a namedtuple with attributes, otherwise it's a full LMM object
        sigma_g2 = lmm_full.sigma_g2
        sigma_e2 = lmm_full.sigma_e2
        lambda_pen = lmm_full.lambda_pen if sigma_g2 > 0 else 1.0e12
        beta_full = np.asarray(lmm_full.lmm1_betas).reshape(-1)
    else:
        sigma_g2 = float(getattr(lmm_full, "v0", 0.0))
        sigma_e2 = float(getattr(lmm_full, "v1", 0.0))
        lambda_pen = (sigma_e2 / sigma_g2) if sigma_g2 > 0 else 1.0e12

        # residuals r = y - X beta (use full model betas)
        beta_full = np.asarray(lmm_full.beta).reshape(-1)

    r = (Y.values.reshape(-1) - (X_df.values @ beta_full).reshape(-1)).reshape(-1)  # length n
    # beta_full = np.asarray(lmm_full.beta).ravel()
    # r = Y.values.ravel() - X_df.values @ beta_full

    # ridge solve: (Z^T Z + lambda I) u = Z^T r
    Z = Z_df.values  # n x m
    n, m = Z.shape
    ZtZ = Z.T @ Z
    A = ZtZ + (lambda_pen * np.eye(m))
    b = Z.T @ r

    try:
        u_hat = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # jitter = 1e-8 * np.mean(np.diag(A))
        # A2 = A + np.eye(m) * jitter
        # u_hat = np.linalg.solve(A2, b)
        # if matrix is still singular, use pinv (pseudo-inverse) which gives a least-squares solution for low rank matrices
        # u_hat = np.linalg.pinv(A) @ b
        return None  # just skip if singular
        # singular matrix error can happen if sigma_g2 or sigma_e2 are very small
        # singular matrix error can also happen if there are duplicate/similar rows in Z (e.g. identical individuals)
        # singular matrix detected skipping u_hat estimation for this locus

    u_hat = u_hat.reshape(-1)
    u_hat_series = pd.Series(u_hat, index=Z_df.columns, name="u_hat")

    # # predicted genetic values and variances
    # g_pred = Z @ u_hat  # length n
    # var_g_emp = float(np.var(g_pred, ddof=0))

    # per-SD conversion (if info is provided)
    u_per_sd = None
    if info is not None and "col_sds" in info:
        sd_arr = np.asarray(info["col_sds"])
        # guard against zero sd
        sd_arr_safe = sd_arr.copy()
        sd_arr_safe[sd_arr_safe == 0] = 1.0
        u_per_sd_arr = sd_arr_safe * u_hat   # effect per 1 SD (sd * per-copy effect)
        u_per_sd = pd.Series(u_per_sd_arr, index=Z_df.columns, name="u_per_sd")

    # out = {
    #     "u_hat": u_hat_series,
    #     "g_pred": pd.Series(g_pred, index=Z_df.index, name="g_pred"),
    #     "var_g_emp": var_g_emp,
    #     "u_per_sd": u_per_sd,
    #     "lambda": lambda_pen
    # }
    out = {
        "u_hat": u_hat_series,
        "u_per_sd": u_per_sd,
    }

    return out


def calc_dosage_matrix(X, n_common_inds, allele_limit=10, allele_freq_threshold=0.99, no_rare_alleles=False, collapse_rare_alleles=False, full_Z=False):
    """
    Calculate allele dosage matrix from VNTR genotypes.

    n = number of individuals
    m = number of alleles
    Z = n x m
    Parameters:
        X (pd.Series): VNTR values, indexed by individual IDs. Each entry is a genotype (e.g., "1/2", "3/4").
        n_common_inds (int): Number of common individuals between all input matrices.
        allele_limit (int): Maximum number of most frequent alleles to consider.
        allele_freq_threshold (float): Cumulative frequency threshold for including alleles.
        no_rare_alleles: True or False. If True only select individuals with BOTH frequent alleles. i.e. > 0.05 frequency (Default=False)
        collapse_unique: True or False. If True collapse all rare alleles (alleles with total counts of 1 or 2) to the same column (allele name 'rare') (Default=False)
    Returns:
        pd.DataFrame: Dosage matrix with individuals as rows and alleles as columns.
        pd.DataFrame: Filtered individuals with at least one selected allele.
        int: Number of individuals in the dosage matrix.
        int: Number of alleles in the dosage matrix.
        dict: Allele frequencies from the dosage matrix.
    """
    # Sanitize genotype strings and drop obvious missing calls before splitting
    # Accept common missing tokens like './.', '.', 'NA', '' (after stripping)
    X_clean = X.dropna().astype(str).str.strip()
    # Catch exact matches OR any string containing a dot
    missing_mask = X_clean.isin(['NA', 'na', '']) | X_clean.str.contains('\.')
    # missing_mask = X_clean.isin(['./.', '.', 'NA', 'na', ''])
    if missing_mask.any():
        # drop individuals with missing genotype calls
        X_clean = X_clean[~missing_mask]

    # Split genotypes into two alleles
    alleles = X_clean.str.split("/", expand=True)
    alleles.columns = ["allele1", "allele2"]

    # print(f"alleles:\n{alleles}")
    # Flatten the alleles and count their frequencies
    all_alleles = pd.concat([alleles["allele1"], alleles["allele2"]])
    allele_counts = all_alleles.value_counts()  # sorts by descending counts high....low

    # Select alleles based on the cumulative frequency threshold and other conditions
    cumulative_freq = 0
    total_alleles = allele_counts.sum()
    selected_alleles = []
    allele_frequencies = {allele: round(count / total_alleles, 4) for allele, count in allele_counts.items()}
    if collapse_rare_alleles == True:  # handle collapse rare alleles as special case
        # 1) Pre-scan to determine 'common' and 'rare' alleles
        # print([f"Allele: {allele}, Count: {count}\n" for allele, count in allele_counts.items() if count >= 3])
        selected_alleles = [allele for allele, count in allele_counts.items() if count >= 3]   
        rare_alleles = [allele for allele, count in allele_counts.items() if count < 3]  # if count 2 or 1 allele is rare, and will be collapsed to 'rare' column
        rare_allele_dosage = sum(allele_counts[allele] for allele in rare_alleles)

        # 2) column order/names and how to map alleles to columns by index
        # to allow rare alleles to have noise, they must have at least 3 counts. if rare dosage < 3 then they must be excluded
        # rare allele column is 'rare'
        if rare_allele_dosage >= 3:
            # print(f"#####\nCollapsing rare alleles {rare_alleles} into single column 'rare' with total dosage {rare_allele_dosage}\n#####")
            columns = selected_alleles + ['rare']
            # no filtration of individuals, all individuals are kept
            individuals_with_selected_alleles = alleles
        else:
            # print(f"#####\nExcluded rare alleles {rare_alleles} b/c they have total dosage {rare_allele_dosage} < 3, so they will be excluded from the dosage matrix\n#####")
            columns = selected_alleles  # no rare alleles will be in dosage matrix, too few rare alleles to combine into column with sum() > 3
            # exclude individuals who have rare allleles
            individuals_with_selected_alleles = alleles[~(alleles["allele1"].isin(rare_alleles) | alleles["allele2"].isin(rare_alleles))]

        col_index = {allele: j for j, allele in enumerate(columns)}

        # 3) Allocate NumPy array (rows in same order as DataFrame index)
        # print(len(X))
        # print(selected_alleles)
        n_ind = len(individuals_with_selected_alleles)  # X is pd.Series
        n_col = len(columns)  # add one extra column for rare allele column that contains all collapsed rare alleles.
        Z_arr = np.zeros((n_ind, n_col), dtype=int)

        # 4) Fill the NumPy array (dosage matrix Z) using indices
        # individuals with rare alleles were already excluded when rare dosage < 3
        for r, (al1, al2) in enumerate(individuals_with_selected_alleles.itertuples(index=False, name=None)):
            if al1 in selected_alleles:  # selected alleles might be smaller list than rare_alleles
                Z_arr[r, col_index[al1]] += 1
            else:
                # individuals with rare alleles when rare_dosage < 3 were already excluded, so will never get here
                Z_arr[r, col_index['rare']] += 1
            if al2 in selected_alleles:
                Z_arr[r, col_index[al2]] += 1
            else:
                # individuals with rare alleles when rare_dosage < 3 were already excluded, so will never get here
                Z_arr[r, col_index['rare']] += 1
        
        # 5) Build the DataFrame with original index and chosen columns
        Z = pd.DataFrame(Z_arr, index=individuals_with_selected_alleles.index, columns=columns, dtype=int)

    elif full_Z == True:
        # when no_rare_alleles == False and collapse_rare_alleles == False attempt match individuals to --collapse_rare_alleles == True, but include alleles as sepearate columns
        # 1) Pre-scan to determine 'common' and 'rare' alleles
        selected_alleles = [allele for allele, count in allele_counts.items() if count >= 3]   
        rare_alleles = [allele for allele, count in allele_counts.items() if count < 3]  # if count 2 or 1 allele is rare, and will be collapsed to 'rare' column
        rare_allele_dosage = sum(allele_counts[allele] for allele in rare_alleles)
        if rare_allele_dosage >= 3:
            print(f"#####\nCollapsing rare alleles {rare_alleles} into single column 'rare' with total dosage {rare_allele_dosage}\n#####")
            # no filtration of individuals, all individuals are kept
            columns = selected_alleles + rare_alleles  # add one extra column for rare allele column that contains all collapsed rare alleles.
            individuals_with_selected_alleles = alleles
        else:
            print(f"#####\nExcluded rare alleles {rare_alleles} b/c they have total dosage {rare_allele_dosage} < 3, so they will be excluded from the dosage matrix\n#####")
            columns = selected_alleles
            individuals_with_selected_alleles = alleles[~(alleles["allele1"].isin(rare_alleles) | alleles["allele2"].isin(rare_alleles))]
        
        col_index = {allele: j for j, allele in enumerate(columns)}

        # 3) Allocate NumPy array (rows in same order as DataFrame index)
        n_ind = len(individuals_with_selected_alleles)  # X is pd.Series
        n_col = len(columns)
        Z_arr = np.zeros((n_ind, n_col), dtype=int)

        # 4) Fill the NumPy array (dosage matrix Z) using indices
        for r, (al1, al2) in enumerate(individuals_with_selected_alleles.itertuples(index=False, name=None)):
            # Both alleles MUST be in the selected_alleles list in this branch
            if al1 in col_index:
                Z_arr[r, col_index[al1]] += 1
            if al2 in col_index:
                Z_arr[r, col_index[al2]] += 1

        # # 4) Fill the NumPy array (dosage matrix Z) using indices
        # for r, (al1, al2) in enumerate(individuals_with_selected_alleles.itertuples(index=False, name=None)):
        #     if (rare_allele_dosage >= 3):  # only include rare alleles if their sum is > 3 like --collapserareallele requirement. but here the alleles will be their own distinct column. NOT TO BE USED FOR GENERATING KLOCUS AND SIGNIFICANCE/SIGMA_G2 ESTIMATION!
        #         Z_arr[r, col_index[al1]] += 1
        #         Z_arr[r, col_index[al2]] += 1
        
        # 5) Build the DataFrame with original index and chosen columns
        Z = pd.DataFrame(Z_arr, index=individuals_with_selected_alleles.index, columns=columns, dtype=int)  # alleles.index

    else:  # when --collapse_rare_alleles == False
        if no_rare_alleles == True:
            # for allele, count in allele_counts.items():
            for allele, frequency in allele_frequencies.items():
                # frequency = count / (n_common_inds * 2)  # x2 because each individual has two alleles  # this is the frequency of the allele divided by the number of common individuals from input matrices. NOT the frequency of the allele in the filtered individuals with selected alleles. This is important because we want to filter out rare alleles based on their frequency in the original population, not just in the filtered set.
                if frequency > 0.05:
                    selected_alleles.append(allele)
                    # print(f"Selected allele: {allele}, Count: {count}, Frequency: {frequency:.4f}")
            # Filter individuals who have both alleles in selected alleles (no rare alleles) only alleles >0.05 frequency
            individuals_with_selected_alleles = alleles[
                (alleles["allele1"].isin(selected_alleles)) & (alleles["allele2"].isin(selected_alleles))]
            # print(f"####HEEERRRREEE\n\n\n\nIndviduals with both alleles in selected alleles (no rare allelels): {individuals_with_selected_alleles})")
        else:
            # keep all alleles and individuals
            selected_alleles = allele_counts.index.tolist()  # all alleles
            individuals_with_selected_alleles = alleles
        # else:
        #     # this block only keeps rare alleles if carried in an individual that has at least one common allele
        #     # when no_rare == False, when collapse_rare == False, when full_Z == False include alleles based on the cumulative frequency threshold and other conditions
        #     # selected individuals will then be allowed to carry rare alleles, but rare/rare allele will not be allowed
        #     cumulative_freq = 0
        #     total_alleles = allele_counts.sum()
        #     selected_alleles = []
        #     for allele, count in allele_counts.items():
        #         cumulative_freq += count / total_alleles
        #         selected_alleles.append(allele)
        #         if (
        #             cumulative_freq >= allele_freq_threshold
        #             or len(selected_alleles) >= allele_limit
        #             or count < 3
        #         ):
        #             break
        #     # Filter individuals who have at least one selected allele
        #     individuals_with_selected_alleles = alleles[
        #         (alleles["allele1"].isin(selected_alleles)) | (alleles["allele2"].isin(selected_alleles))]

        # Start building the dosage matrix Z when --collapse_rare_alleles == False but avoid using DataFrame operations in loops (slow). i.e. Z[rare_allele] = 0
        selected_set = set(selected_alleles)
        # print(f"Selected alleles: {selected_alleles}")
        # print(f"n_common_inds: {n_common_inds}, no_rare_alleles: {no_rare_alleles}, collapse_rare_alleles: {collapse_rare_alleles}, full_Z: {full_Z}")
        # 1) Pre-scan to determine rare columns order
        rare_order = []
        rare_seen = set()
        pairs = list(individuals_with_selected_alleles.itertuples(index=False, name=None))
        for al1, al2 in pairs:
            if (al1 not in selected_set) or (al2 not in selected_set):
                rare = al1 if (al1 not in selected_set) else al2
                if rare not in rare_seen:
                    rare_seen.add(rare)
                    rare_order.append(rare)

        # 2) Column order & mappings
        columns = list(selected_alleles) + rare_order
        col_index = {allele: j for j, allele in enumerate(columns)}

        # 3) Allocate NumPy array (rows in same order as DataFrame index)
        n_ind = len(pairs)
        n_col = len(columns)
        Z_arr = np.zeros((n_ind, n_col), dtype=int)

        # 4) Fill the NumPy array — use indices (fast)
        if no_rare_alleles == True:
            for r, (al1, al2) in enumerate(pairs):
                if (al1 in selected_set) and (al2 in selected_set):
                    Z_arr[r, col_index[al1]] += 1
                    Z_arr[r, col_index[al2]] += 1
        else:
            # when no_rare == False, when collapse_rare == False, when full_Z == False include alleles based on the cumulative frequency threshold and other conditions
            # selected individuals will then be allowed to carry rare alleles, but rare/rare allele will not be allowed
            for r, (al1, al2) in enumerate(pairs):
                if al1 in selected_set:
                    Z_arr[r, col_index[al1]] += 1
                if al2 in selected_set:
                    Z_arr[r, col_index[al2]] += 1
                if (al1 not in selected_set) or (al2 not in selected_set):
                    rare_allele = al1 if (al1 not in selected_set) else al2
                    Z_arr[r, col_index[rare_allele]] += 1

        # 5) Build the DataFrame with original index and chosen columns
        Z = pd.DataFrame(Z_arr, index=individuals_with_selected_alleles.index, columns=columns, dtype=int)       

    # Ensure the dosage matrix has only 0, 1, or 2 values
    if (Z > 2).any().any():
        raise ValueError("Dosage matrix contains values greater than 2, which is invalid.")

    # Ensure each row has exactly two 1s or one 2
    invalid_rows = Z[(Z.sum(axis=1) != 2)].index
    if not invalid_rows.empty:
        print(f"ERROR in Z.sum(axis=1): row does not sum to 2 {Z.sum(axis=1)}")
        for i, (al1, al2) in individuals_with_selected_alleles.iterrows():
            print(f"i: {i}, al1: {al1}, al2: {al2}, Z.loc[i]: {Z.loc[i].to_dict()}")
        raise ValueError(f"Invalid rows in dosage matrix, dosage does not sum to 2: {invalid_rows.tolist()}")

    # Report final statistics
    # num_individuals = Z.shape[0]
    num_alleles = Z.shape[1]
    total_alleles = Z.sum().sum()  # Total dosage across all alleles
    allele_frequencies = (Z.sum(axis=0) / total_alleles).to_dict()

    ### print(f"Final number of individuals: {num_individuals}")
    ### print(f"Final number of alleles included: {num_alleles}")
    ### print(f"Allele frequencies: {allele_frequencies}")

    return Z, individuals_with_selected_alleles, num_alleles, allele_frequencies


def prep_data(gene_expression, vntr_values, common_inds, covariates, n_common_inds, minsamples, Kgrm, vntr_mean_length_values=None, best_variants=None, no_rare_alleles=False, collapse_rare_alleles=False, permute_vntr_mean_lengths=False, full_Z=False):
    """
    Prepares data for LMM analysis.
    
    Parameters:
        gene_expression (pd.Series): Gene expression values, indexed by individual IDs.
        vntr_values (pd.Series): VNTR values, indexed by individual IDs.
        common_inds (Index): Individuals shared between gene expression and VNTR data.
        covariates (pd.DataFrame): Covariate matrix with row names as covariates and columns as individuals.
        n_common_inds (int): Number of common individuals between all input matrices.
        kinship_matrix (Kgrm)(pd.DataFrame): Kinship matrix with individuals as both rows and columns.

    Returns:
        tuple: (Y, X, Z, covs, kinship_sub)
    """
    enough_data = True
    Z, selected_individuals, n_allele, all_freq = calc_dosage_matrix(vntr_values, n_common_inds, no_rare_alleles=no_rare_alleles, collapse_rare_alleles=collapse_rare_alleles, full_Z=full_Z)  # Calculate dosage matrix from VNTR values
    if Z.shape[0] < minsamples or Z.shape[1] < 2:
        # print(f"Too few individuals (<{minsamples}) or alleles (<1) to be analyzed: Indivs={Z.shape[0]}, alleles={Z.shape[1]}")
        enough_data = False
        return None, None, None, Z, None, None, None, None, None, enough_data  # covs, X_vntr
    # intersect selected individuals with common individuals
    common_inds = common_inds.intersection(selected_individuals.index)
    if best_variants is not None:
        Z_best_var, selected_individuals, _, _ = calc_dosage_matrix(best_variants, n_common_inds)
        # do not filter best variants by allele frequency, just take all alleles, and then intersect individuals with common_inds and selected_individuals to get final set of individuals for best variant dosage matrix
        # intersect selected individuals with common individuals
        common_inds = common_inds.intersection(selected_individuals.index)
    else:
        Z_best_var = None
    # Subset data to common individuals
    n_indiv = len(common_inds)
    Z = Z.loc[common_inds]  # Subset dosage matrix to common individuals
    Z_best_var = Z_best_var.loc[common_inds] if Z_best_var is not None else None  # Subset best variant dosage matrix to common individuals
    if Z.shape[0] < minsamples or Z.shape[1] < 2:
        # print(f"Too few individuals (<{minsamples}) or alleles (<1) to be analyzed: Indivs={Z.shape[0]}, alleles={Z.shape[1]}")
        enough_data = False
        return None, None, None, Z, None, None, None, None, None, enough_data  # covs, X_vntr
    Y = gene_expression[common_inds]
    # X_vntr = vntr_values[common_inds]
    X = covariates[common_inds].T  # X covariate matrix (no intercept)
    X_mat = sm.add_constant(X)  # constant already added when performing LMM # X covariate matrix with leading contant (i.e y-intercept)
    if vntr_mean_length_values is not None:
        vntr_mean_length_values = vntr_mean_length_values.loc[common_inds]
        vntr_mean_length_df = vntr_mean_length_values.to_frame(name=f"vntr_mean_allele_length")  # Label VNTR variable
        if permute_vntr_mean_lengths:  # scramble VNTR mean lengths relative to individual IDs to add fixed effect with values but no true association
            vntr_mean_length_df = vntr_mean_length_df.assign(vntr_mean_allele_length=np.random.permutation(vntr_mean_length_df["vntr_mean_allele_length"].values))
        # Only add to matrices if there is variation (>= 2 unique values)
        # if vntr_mean_length_df["vntr_mean_allele_length"].nunique() >= 2:
        #     X = pd.concat([X, vntr_mean_length_df], axis=1)
        #     X_mat = pd.concat([X_mat, vntr_mean_length_df], axis=1)
        # else:
        #     print("Skipping vntr_mean_length: constant value detected.") 
        X = pd.concat([X, vntr_mean_length_df], axis=1)  # Concatenate covariates and VNTR into a single DataFrame # Fixed effect design matrix
        X_mat = pd.concat([X_mat, vntr_mean_length_df], axis=1)  # Concatenate covariates and VNTR into a single DataFrame # Fixed effect design matrix
        # assert X_mat.index.equals(vntr_mean_length_df.index)
        # assert not X['vntr_mean_allele_length'].isna().any()
    sorted_alleles = ','.join([f"{k}:{all_freq[k]:.3f}" for k in natsorted(all_freq.keys())])  # format for output # all_freq is a dict with allele names as keys and their frequencies as values
    
    if Z.shape[0] < minsamples or Z.shape[1] < 2:  # number of individuals and number of alleles
        # print(f"Too few individuals (<{minsamples}) or alleles (<1) to be analyzed: Indivs={n_indiv}, alleles={Z.shape[1]}")
        enough_data = False

    return Y, X, X_mat, Z, n_indiv, n_allele, all_freq, sorted_alleles, Z_best_var, enough_data  # covs, X_vntr


def remove_sd_outliers(values, threshold=5):
    """
    Remove individuals whose expression values are more than `threshold` SDs from the mean.
    """
    mean = np.mean(values)
    std = np.std(values)
    mask = np.abs(values - mean) <= threshold * std
    return values[mask]


#############################
#############################
#############################


# 1. The worker function: Logic for ONE gene
def process_single_gene(task_variables):
    """
    from multiprocessing import Pool

    This function runs in a separate process.
    task_variables is a tuple: (gene_id, list_of_vntr_ids, no_rare_alleles, collapse_rare_alleles, Zraw, remove_most_frequent_allele, remove_reference_allele, reml, permute_vntr_mean_lengths, n_common_inds, kinship_matrix)
    """
    countmodels, countslightsignificant, countinvariantgeneexp, countinvariantvntr, countfailLMM, countfailraneff, countnotenoughdata, countskipfreqremoval, countskiprefremoval = 0, 0, 0, 0, 0, 0, 0, 0, 0
    gene_id, vntr_ids, minsamples, no_rare_alleles, collapse_rare_alleles, Zraw, remove_most_frequent_allele, remove_reference_allele, reml, permute_vntr_mean_lengths, n_common_inds, kinship_matrix = task_variables
    gene_results = []
    # Access the matrices from the 'global' space 
    # (they are inherited from the main process)
    global vntr_matrix
    global expression_matrix
    global covariates
    global best_variant_matrix
    global vntr_mean_length_matrix
    global best_vntr_mean_length_matrix

    gene_chr = expression_matrix.loc[gene_id, "#chr"]  # if "#chr" in expression_matrix.columns else None
    gene_start = expression_matrix.loc[gene_id, "start"]  # if "start" in  expression_matrix.columns else None
    gene_end = expression_matrix.loc[gene_id, "end"]  # if "end" in  expression_matrix.columns else None
    gene_strand = expression_matrix.loc[gene_id, "strand"]  # if "strand" in  expression_matrix.columns else None
    gene_pos = gene_start if gene_strand == "+" else gene_end
    gene_expression = expression_matrix.loc[gene_id, expression_matrix.columns[6:]].dropna().astype(float)
    gene_expression = remove_sd_outliers(gene_expression, threshold=5)
    best_variants = best_variant_matrix.loc[gene_id, best_variant_matrix.columns[6:]].dropna().astype(str) if best_variant_matrix is not None else None
    best_variant_ID = best_variant_matrix.loc[gene_id, 'pid'] if best_variant_matrix is not None else None
    for vntr_id in vntr_ids:
        # vntr_chr = vntr_matrix.loc[vntr_id, "#chr"]  # if "#chr" in expression_matrix.columns else None
        vntr_start = vntr_matrix.loc[vntr_id, "start"]  # if "start" in  expression_matrix.columns else None
        # print(type(vntr_start), vntr_id)
        # print(vntr_matrix.index.is_unique)
        # print(vntr_matrix.index[vntr_matrix.index.duplicated()])
        vntr_end = vntr_matrix.loc[vntr_id, "end"]  # if "end" in  expression_matrix.columns else None
        # vntr_strand = vntr_matrix.loc[vntr_id, "strand"]  # if "strand" in  expression_matrix.columns else None
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
        
        vntr_values = vntr_matrix.loc[vntr_id, vntr_matrix.columns[6:]].dropna().astype(str)
        common_inds = gene_expression.index.intersection(vntr_values.index)
        # print(f"Processing gene {count}/{numgenes}: {gene_id} with VNTR {vntr_id} ({len(common_inds)} common individuals)")
        common_inds = common_inds.intersection(best_variants.index) if best_variants is not None else common_inds  # filter to best variant individuals b/c can have missing genotypes for specific individuals
        
        if best_vntr_mean_length_matrix is not None:
            # below two are None, unless -> if best_vntr_mean_length_matrix is not None:
            vntr_mean_length_values = best_vntr_mean_length_matrix.loc[gene_id, best_vntr_mean_length_matrix.columns[6:]].loc[lambda x: (x != '.') & x.notna()].astype(float) #  if best_vntr_mean_length_matrix is not None else None
            vntr_mean_length_ID = best_vntr_mean_length_matrix.loc[gene_id, 'pid'] #  if best_vntr_mean_length_matrix is not None else None
        elif vntr_mean_length_matrix is not None:
            # store below two as None unless -> if vntr_mean_length_matrix is not None:
            vntr_mean_length_values = vntr_mean_length_matrix.loc[vntr_id, vntr_mean_length_matrix.columns[6:]].loc[lambda x: (x != '.') & x.notna()].astype(float) #  if vntr_mean_length_matrix is not None else None
            vntr_mean_length_ID = vntr_mean_length_matrix.loc[vntr_id, 'pid'] #  if vntr_mean_length_matrix is not None else None
        else:
            # have to set as None so next line can be evaluated
            vntr_mean_length_values = None
            vntr_mean_length_ID = None
        common_inds = common_inds.intersection(vntr_mean_length_values.index) if vntr_mean_length_values is not None else common_inds # ensure common_inds filter has data from individuals in vntr_mean_length_matrix (and/or best_variant_matrix and gene_expression and vntr_values)
        if len(common_inds) < minsamples:
            countnotenoughdata += 1
            del vntr_values, common_inds, vntr_mean_length_values
            continue  # Skip if not enough samples
        # Subset to common individuals
        gene_vals = gene_expression.loc[common_inds]
        vntr_vals = vntr_values.loc[common_inds]
        #vntr_mean_length_values = vntr_mean_length_values.loc[common_inds] if vntr_mean_length_matrix is not None else None
        vntr_mean_length_values = vntr_mean_length_values.loc[common_inds] if vntr_mean_length_values is not None else None
        best_vars = best_variants.loc[common_inds] if best_variants is not None else None
        # Re-query best_variant_matrix per-iteration rather than slicing the possibly-deleted
        # `best_variants` object. This avoids UnboundLocalError if `best_variants` was removed
        # in a previous iteration (some branches delete it to free memory).
        # best_variants = (best_variant_matrix.loc[gene_id, best_variant_matrix.columns[6:]].dropna().astype(str).loc[common_inds]
        #                  if best_variant_matrix is not None else None)
        # Check variation in gene expression
        if gene_vals.std() < 1e-6:
            ### print(f"Gene {gene_id} has no variation in gene expression.")
            countinvariantgeneexp += 1
            del gene_vals, vntr_vals, vntr_mean_length_values, vntr_values, common_inds, best_vars, vntr_start, vntr_end, TSS_distance
            continue  # gene expression is nearly constant
        # Check variation in VNTR values
        if vntr_vals.nunique() < 2:
            ### print(f"VNTR {vntr_id} has no variation in VNTR values (ex: genotypes, mean allele length). Number vntr_vals: {vntr_vals.nunique()}")
            ### print(f"{vntr_vals.unique()}")
            countinvariantvntr += 1
            del gene_vals, vntr_vals, vntr_mean_length_values, vntr_values, common_inds, best_vars, vntr_start, vntr_end, TSS_distance
            continue  # no variation in VNTR values
        # check if required values are not empty
        if gene_expression.empty or vntr_values.empty or covariates.empty:
            # print(f'Problem prepping data for multulinear regression')
            del gene_vals, vntr_vals, vntr_mean_length_values, vntr_values, common_inds, best_vars, vntr_start, vntr_end, TSS_distance
            continue  # Skip if any required data is missing
        Y, X, X_mat, Z, n_indiv, n_allele, allele_freqs, sorted_alleles, Z_best_var, enough_data = prep_data(gene_expression, vntr_values, common_inds, covariates, n_common_inds, minsamples, kinship_matrix, vntr_mean_length_values, best_vars, no_rare_alleles=no_rare_alleles, collapse_rare_alleles=collapse_rare_alleles, permute_vntr_mean_lengths=permute_vntr_mean_lengths)
        # Z_ortho, Kgrm, Klocus, Klocus_ortho, 
        if enough_data == False:
            countnotenoughdata += 1
            del Y, X, X_mat, Z, gene_vals, vntr_vals, vntr_mean_length_values, vntr_values, common_inds, best_vars, vntr_start, vntr_end, TSS_distance
            continue  # Skip if not enough data after filtering
        allele_removed = 'nan'
        if remove_most_frequent_allele == True and remove_reference_allele == False:
            # remove the most frequent allele column from Z to make Z relative to most frequent allele
            # this changes the interpretation of the allele effect predictions to be relative to the most frequent allele
            # only remove if most frequent allele is present with MAF > 0.05
            all_freq_norare = {k: v for k, v in allele_freqs.items() if k != 'rare'}
            allele_removed = max(all_freq_norare, key=all_freq_norare.get)
            if allele_freqs[allele_removed] >= 0.05:
                Z = Z.drop(columns=[allele_removed])
            else:
                countskipfreqremoval += 1
                del Y, X, X_mat, Z, gene_vals, vntr_vals, vntr_mean_length_values, vntr_values, common_inds, allele_freqs, all_freq_norare, best_vars, vntr_start, vntr_end, TSS_distance
                continue  # Skip, most frequent allele is not frequent enough to serve as baseline AF < 0.05                    
        elif remove_reference_allele == True and '0' in Z.columns:
            # remove the reference allele column (column name 0) from Z to make Z relative to reference allele
            # only remove if reference allele is present with MAF > 0.05 else skip
            if allele_freqs['0'] >= 0.05:
                Z = Z.drop(columns=['0'])
                allele_removed = '0'
            else:
                countskiprefremoval += 1
                del Y, X, X_mat, Z, gene_vals, vntr_vals, vntr_mean_length_values, vntr_values, common_inds, allele_freqs, best_vars, vntr_start, vntr_end, TSS_distance
                continue  # skip, reference alleles has AF < 0.5
        else:
            pass  # don't modify Z by removing allele columns
        
        # Center columns of Z, i.e. remove mean of each column (allele) from Z to get Zc (Z centered), return Kc= ZcZcᵀ (centered locus-specific GRM)
        Zc, info = center_Z(Z, return_K=False)
        # Zcs = center_and_standardize(Z, standardize=True)
        if Zraw == True:
            # use raw Z (not centered) for LMM fitting and BLUPs
            Zc = Z
        # Fit LMM using centered Zc compute pval by LRT
        # option to scale K to have the mean diagonal = 1 (trace(K)/n == 1), but here we do not scale K because we want sigma_g2 to be interpretable as variance explained by the locus
        # reml = True means use REML (default for heritability estimation), False means use ML (default for LRT testing)
        # Y: pandas Series of expression values (n samples)
        # X: pandas DataFrame of covariates (n samples × p covariates) covariates include genotypePCs and rnaPCs and intercept
        # Klocus: pandas DataFrame of n×n kinship for the locus
        # lr, pval, comps = test_klocus(Y, X, Klocus)
        # Step 0: Check if locus shows any signal first
        # assume that Kgrm is accounted for in genotype_PCs in X
        # reml = True performs restricted maximum likelihood
        lr, pval, comps = fit_lmm(Y, X_mat, Zc, Z_best_var, reml=reml, return_models=True, high_throughput=True)
        if lr is None:
            # print(f"Skipping comparison between gene: {gene_id} and {vntr_id} due to failed LMM fitting.")
            countfailLMM += 1
            del Y, X, X_mat, Z, Zc, gene_vals, vntr_vals, vntr_mean_length_values, vntr_values, common_inds, lr, pval, comps, allele_freqs, best_vars, vntr_start, vntr_end, TSS_distance
            continue                
        
        # start collecting results for this gene-VNTR pair in result list
        result = [gene_id, vntr_id, f"{gene_chr}.{gene_start}.{gene_end}",TSS_distance, float(np.mean(Y))]
        if best_variant_matrix is not None:
            result.append(best_variant_ID)
        if vntr_mean_length_matrix is not None or best_vntr_mean_length_matrix is not None:
            result.extend([vntr_mean_length_ID, float(np.mean(vntr_mean_length_values))])
        result.extend([comps.r2_fixed, comps.r2_genetic, comps.r2_residual,
            comps.var_total, comps.ll0, comps.ll1, lr,
            n_indiv, n_allele, allele_removed, sorted_alleles,
            comps.sigma_g2, comps.sigma_e2, comps.lambda_pen,
            comps.h2])
        
        if pval < 0.000001: # 1e-6:
            #print(f"\nSignificant locus detected!")
            countslightsignificant += 1
            # extract variance components for covariance matrix .v0 and residuals .v1
            # from: https://glimix-core.readthedocs.io/en/latest/lmm.html
            # 𝐲 ∼ 𝓝(X𝜷, v₀GGᵀ + v₁I).
            # beta_hat = lmm1.beta
            # y_resid = (Y - X_mat @ beta_hat).to_numpy()
            # sigma_g2, sigma_e2 = lmm1.v0, lmm1.v1
            # FixedEffectVariation = np.var(Y - X_mat @ beta_hat)
            # comput BLUPs from centered Zc. Low-frequency alleles will be shrunk more as they are not as reliable as the data from frequent alleles.
            # lambda = sigma_e2 / sigma_g2 (0-1) is the noise-to-signal ratio and controls the amount of allele estimate shrinkage (high lambda = more shrinkage)
            blup = compute_blups(Zc, comps, X_mat, Y, high_throughput=True, info=info)
            # u_hat_locus = blup['u_hat']  # pandas Series of allele effects (BLUps) with allele names as index, using Zc (centered Z), unit is effect per allele copy
            # blup['u_per_sd']  # pandas Series of allele effects (BLUps) with allele names as index, using Zc (centered Z), unit is effect per 1 SD of allele dosage
            if blup is None:
                # print(f"V is a singular matrix. Skipping gene: {gene_id} and {vntr_id} due to failed random effects estimation.\n - pval: {pval}, LR: {lr}, sigma_g2: {comps.sigma_g2}, sigma_e2: {comps.sigma_e2}, n_indiv: {n_indiv}, n_allele: {n_allele}")
                countfailraneff += 1
                del Y, X, X_mat, Z, Zc, gene_vals, vntr_vals, vntr_mean_length_values, vntr_values, common_inds, lr, pval, comps, blup, allele_freqs, best_vars, vntr_start, vntr_end, TSS_distance
                continue
            removed_allele_effect = f"{allele_removed}:0," if allele_removed != 'nan' else ""  # removed allele effect is always 0, it is the baseline. but if not requested by user then don't output anything
            # use named tuple
            # changed comps.r2_genetic to comps.h2

            # extend result with allele effects and pval
            result.extend([
                removed_allele_effect + ','.join([f"{allele}:{blup['u_hat'].loc[allele]:.5f}" for allele in blup['u_hat'].index]),
                removed_allele_effect + ','.join([f"{allele}:{blup['u_per_sd'].loc[allele]:.5f}" for allele in blup['u_per_sd'].index]),
                pval])
            
            # if vntr_mean_length_matrix is None and best_variant_matrix is None:
            #     result = [gene_id, vntr_id, f"{gene_chr}.{gene_start}.{gene_end}", TSS_distance, float(np.mean(Y)), comps.r2_fixed, comps.r2_genetic, comps.r2_residual, comps.var_total, comps.ll0, comps.ll1, lr, n_indiv, n_allele, allele_removed, sorted_alleles, comps.sigma_g2, comps.sigma_e2, comps.lambda_pen, comps.h2, 
            #               removed_allele_effect + ','.join([f"{allele}:{blup['u_hat'].loc[allele]:.5f}" for allele in blup['u_hat'].index]), removed_allele_effect + ','.join([f"{allele}:{blup['u_per_sd'].loc[allele]:.5f}" for allele in blup['u_per_sd'].index]), pval]  # ','.join([f"{k}:{v}" for k, v in zip(allele_labels, u_locus.flatten())])
            # elif vntr_mean_length_matrix is None:
            #     result = [gene_id, vntr_id, f"{gene_chr}.{gene_start}.{gene_end}", TSS_distance, float(np.mean(Y)), best_variant_ID, comps.r2_fixed, comps.r2_genetic, comps.r2_residual, comps.var_total, comps.ll0, comps.ll1, lr, n_indiv, n_allele, allele_removed, sorted_alleles, comps.sigma_g2, comps.sigma_e2,comps.lambda_pen, comps.h2,  
            #               removed_allele_effect + ','.join([f"{allele}:{blup['u_hat'].loc[allele]:.5f}" for allele in blup['u_hat'].index]), removed_allele_effect + ','.join([f"{allele}:{blup['u_per_sd'].loc[allele]:.5f}" for allele in blup['u_per_sd'].index]), pval]  # ','.join([f"{k}:{v}" for k, v in zip(allele_labels, u_locus.flatten())])
            # elif best_variant_matrix is None:
            #     result = [gene_id, vntr_id, f"{gene_chr}.{gene_start}.{gene_end}", TSS_distance, float(np.mean(Y)), float(np.mean(vntr_mean_length_values)), comps.r2_fixed, comps.r2_genetic, comps.r2_residual, comps.var_total, comps.ll0, comps.ll1, lr, n_indiv, n_allele, allele_removed, sorted_alleles, comps.sigma_g2, comps.sigma_e2, comps.lambda_pen, comps.h2,  
            #               removed_allele_effect + ','.join([f"{allele}:{blup['u_hat'].loc[allele]:.5f}" for allele in blup['u_hat'].index]), removed_allele_effect + ','.join([f"{allele}:{blup['u_per_sd'].loc[allele]:.5f}" for allele in blup['u_per_sd'].index]), pval]  # ','.join([f"{k}:{v}" for k, v in zip(allele_labels, u_locus.flatten())])
            # else:
            #     result = [gene_id, vntr_id, f"{gene_chr}.{gene_start}.{gene_end}", TSS_distance, float(np.mean(Y)), best_variant_ID, float(np.mean(vntr_mean_length_values)), comps.r2_fixed, comps.r2_genetic, comps.r2_residual, comps.var_total, comps.ll0, comps.ll1, lr, n_indiv, n_allele, allele_removed, sorted_alleles, comps.sigma_g2, comps.sigma_e2, comps.lambda_pen, comps.h2,  
            #               removed_allele_effect + ','.join([f"{allele}:{blup['u_hat'].loc[allele]:.5f}" for allele in blup['u_hat'].index]), removed_allele_effect + ','.join([f"{allele}:{blup['u_per_sd'].loc[allele]:.5f}" for allele in blup['u_per_sd'].index]), pval]  # ','.join([f"{k}:{v}" for k, v in zip(allele_labels, u_locus.flatten())])
            del blup
        else:
            # if no significant association found, fill in 'nan' for allele effects but still add pval
            result.extend(['nan', 'nan', pval])
            
            # # print(f"No significant association found for {gene_id} and {vntr_id} (p-value: {p_value:.4e})")
            # if vntr_mean_length_matrix is None and best_variant_matrix is None:
            #     result = [gene_id, vntr_id, f"{gene_chr}.{gene_start}.{gene_end}", TSS_distance, float(np.mean(Y)), comps.r2_fixed, comps.r2_genetic, comps.r2_residual, comps.var_total, comps.ll0, comps.ll1, lr, n_indiv, n_allele, allele_removed, sorted_alleles, comps.sigma_g2, comps.sigma_e2, comps.lambda_pen, comps.h2,'nan', 'nan', pval]  # ','.join([f"{k}:{v}" for k, v in zip(allele_labels, u_locus.flatten())])
            # elif vntr_mean_length_matrix is None:
            #     result = [gene_id, vntr_id, f"{gene_chr}.{gene_start}.{gene_end}", TSS_distance, float(np.mean(Y)), best_variant_ID, comps.r2_fixed, comps.r2_genetic, comps.r2_residual, comps.var_total, comps.ll0, comps.ll1, lr, n_indiv, n_allele, allele_removed, sorted_alleles, comps.sigma_g2, comps.sigma_e2, comps.lambda_pen, comps.h2, 'nan', 'nan', pval]  # ','.join([f"{k}:{v}" for k, v in zip(allele_labels, u_locus.flatten())])
            # elif best_variant_matrix is None:
            #     result = [gene_id, vntr_id, f"{gene_chr}.{gene_start}.{gene_end}", TSS_distance, float(np.mean(Y)), float(np.mean(vntr_mean_length_values)), comps.r2_fixed, comps.r2_genetic, comps.r2_residual, comps.var_total, comps.ll0, comps.ll1, lr, n_indiv, n_allele, allele_removed, sorted_alleles, comps.sigma_g2, comps.sigma_e2, comps.lambda_pen, comps.h2, 'nan', 'nan', pval]  # ','.join([f"{k}:{v}" for k, v in zip(allele_labels, u_locus.flatten())])
            # else:
            #     result = [gene_id, vntr_id, f"{gene_chr}.{gene_start}.{gene_end}", TSS_distance, float(np.mean(Y)), best_variant_ID, float(np.mean(vntr_mean_length_values)), comps.r2_fixed, comps.r2_genetic, comps.r2_residual, comps.var_total, comps.ll0, comps.ll1, lr, n_indiv, n_allele, allele_removed, sorted_alleles, comps.sigma_g2, comps.sigma_e2, comps.lambda_pen, comps.h2, 'nan', 'nan', pval]  # ','.join([f"{k}:{v}" for k, v in zip(allele_labels, u_locus.flatten())])
        gene_results.append(result)
        countmodels += 1
        
        del Y, X, X_mat, Z, Zc, result, gene_vals, vntr_vals, vntr_mean_length_values, vntr_values, common_inds, lr, pval, comps, allele_freqs, best_vars, vntr_start, vntr_end, TSS_distance
        
    del gene_expression, best_variants, gene_chr, gene_start, gene_end, gene_strand, gene_pos

    counts = [countmodels, countslightsignificant, countinvariantgeneexp, countinvariantvntr, countfailLMM, countfailraneff, countnotenoughdata, countskipfreqremoval, countskiprefremoval]
    return gene_results, gene_id, len(vntr_ids), counts


def init_worker(v_matrix, e_matrix, c_matrix, bv_matrix=None, vl_matrix=None, bvl_matrix=None):
    '''
    Setup a 'Global' initializer
    This function is called when each worker process starts, and it sets the global variables for the VNTR and expression matrices that will be used in the worker function.
    Avoids the need to pass large matrices as arguments to the worker function for each task, which can be inefficient. Instead, the matrices are loaded once per worker process and stored in global variables that the worker function can access.
    '''
    global vntr_matrix  # global_
    global expression_matrix
    global covariates
    global best_variant_matrix
    global vntr_mean_length_matrix
    global best_vntr_mean_length_matrix

    vntr_matrix = v_matrix
    expression_matrix = e_matrix
    covariates = c_matrix
    best_variant_matrix = bv_matrix
    vntr_mean_length_matrix = vl_matrix
    best_vntr_mean_length_matrix = bvl_matrix  # if best_vntr_mean_length_matrix is not None else None


def run_association_analysis(output_file, gene_vntr_dict, expression_matrix, vntr_matrix, covariates, workingdir, minsamples: int, threads: int, common_inds, kinship_matrix=None, vntr_mean_length_matrix=None, best_vntr_mean_length_matrix=None, best_variant_matrix=None, no_rare_alleles=False, collapse_rare_alleles=False, Zraw=False, remove_most_frequent_allele=False, remove_reference_allele=False, reml=False, permute_vntr_mean_lengths=False, fdr_threshold=0.05):
    # output_file = os.path.join(workingdir, "gene_vntr_association_results.tsv")

    # Write the header once
    with open(output_file, "w") as OUT:
        # if vntr_mean_length_matrix is None and best_vntr_mean_length_matrix is None and best_variant_matrix is None:
        #     header = ["geneid", "VNTRid", "genepos", "TSS_distance", "mean_gene_TPM", "r2_fixed", "r2_genetic", "r2_residual", "total_variance", "null_lml", "full_lml", "lr_stat", "n_indiv", "n_alleles", "removed_allele", "allele_frequencies", "sigma_g2", "sigma_e2", "lambda", "h2", "BLUPs_per_allele", "BLUPs_per_SD", "raw_pvalue" ]
        # elif vntr_mean_length_matrix is None and best_vntr_mean_length_matrix is None:
        #     header = ["geneid", "VNTRid", "genepos", "TSS_distance", "mean_gene_TPM", "best_variant_ID", "r2_fixed", "r2_genetic", "r2_residual", "total_variance", "null_lml", "full_lml", "lr_stat", "n_indiv", "n_alleles", "removed_allele", "allele_frequencies", "sigma_g2", "sigma_e2", "lambda", "h2", "BLUPs_per_allele", "BLUPs_per_SD", "raw_pvalue" ]
        # elif best_variant_matrix is None and best_vntr_mean_length_matrix is None:
        #     header = ["geneid", "VNTRid", "genepos", "TSS_distance", "mean_gene_TPM", "mean_VNTR_length_ID", "mean_VNTR_length", "r2_fixed", "r2_genetic", "r2_residual", "total_variance", "null_lml", "full_lml", "lr_stat", "n_indiv", "n_alleles", "removed_allele", "allele_frequencies", "sigma_g2", "sigma_e2", "lambda", "h2", "BLUPs_per_allele", "BLUPs_per_SD", "raw_pvalue" ]
        # elif best_variant_matrix is None and vntr_mean_length_matrix is None:
        #     header = ["geneid", "VNTRid", "genepos", "TSS_distance", "mean_gene_TPM", "best_mean_VNTR_length_ID", "mean_VNTR_length", "r2_fixed", "r2_genetic", "r2_residual", "total_variance", "null_lml", "full_lml", "lr_stat", "n_indiv", "n_alleles", "removed_allele", "allele_frequencies", "sigma_g2", "sigma_e2", "lambda", "h2", "BLUPs_per_allele", "BLUPs_per_SD", "raw_pvalue" ]
        # elif best_vntr_mean_length_matrix is None:
        #     header = ["geneid", "VNTRid", "genepos", "TSS_distance", "mean_gene_TPM", "best_variant_ID", "mean_VNTR_length_ID", "mean_VNTR_length", "r2_fixed", "r2_genetic", "r2_residual", "total_variance", "null_lml", "full_lml", "lr_stat", "n_indiv", "n_alleles", "removed_allele", "allele_frequencies", "sigma_g2", "sigma_e2", "lambda", "h2", "BLUPs_per_allele", "BLUPs_per_SD", "raw_pvalue" ]
        # elif vntr_mean_length_matrix is None:
        #     header = ["geneid", "VNTRid", "genepos", "TSS_distance", "mean_gene_TPM", "best_variant_ID", "best_mean_VNTR_length_ID", "mean_VNTR_length", "r2_fixed", "r2_genetic", "r2_residual", "total_variance", "null_lml", "full_lml", "lr_stat", "n_indiv", "n_alleles", "removed_allele", "allele_frequencies", "sigma_g2", "sigma_e2", "lambda", "h2", "BLUPs_per_allele", "BLUPs_per_SD", "raw_pvalue" ]
        header = ["geneid", "VNTRid", "genepos", "TSS_distance", "mean_gene_TPM"]
        if best_variant_matrix is not None:
            header.append("best_variant_ID")
        if vntr_mean_length_matrix is not None:
            header.extend(["mean_VNTR_length_ID", "mean_VNTR_length"])
        elif best_vntr_mean_length_matrix is not None:
            header.extend(["best_mean_VNTR_length_ID", "best_mean_VNTR_length"])
        header += ["r2_fixed", "r2_genetic", "r2_residual", "total_variance",
                   "null_lml", "full_lml", "lr_stat", "n_indiv", "n_alleles",
                   "removed_allele", "allele_frequencies", "sigma_g2", "sigma_e2",
                   "lambda", "h2", "BLUPs_per_allele", "BLUPs_per_SD", "raw_pvalue"]
        OUT.write("\t".join(header))

    countmodels = 0
    countslightsignificant = 0
    countinvariantgeneexp = 0
    countinvariantvntr = 0
    countfailLMM = 0
    countfailraneff = 0
    countnotenoughdata = 0
    countskipfreqremoval = 0
    countskiprefremoval = 0

    # number of starting common individuals between all input matrices, use this for allele frequency filtration in prep_data -> make_allele_dosage_matrix
    n_common_inds = len(common_inds)
    covariates = covariates.apply(pd.to_numeric, errors="coerce")
    covariates = covariates.dropna(axis=1, how="all")  # Drop columns with all NaNs

    # Prepare tasks: each task is a tuple of the specific gene info + extra params
    # tasks = list(gene_vntr_dict.items())
    tasks = [(gene_id, vntrs, minsamples, no_rare_alleles, collapse_rare_alleles, Zraw, remove_most_frequent_allele, remove_reference_allele, reml, permute_vntr_mean_lengths, n_common_inds, kinship_matrix) for gene_id, vntrs in gene_vntr_dict.items()]
    with open(output_file, "a") as OUT:
        numgenes = len(gene_vntr_dict)
        all_accumulated_results = []
        # Setup Pool: maxtasksperchild=1 ensures the process is killed after finishing ONE gene
        with Pool(processes=threads, maxtasksperchild=1, initializer=init_worker, initargs=(vntr_matrix, expression_matrix, covariates, best_variant_matrix, vntr_mean_length_matrix, best_vntr_mean_length_matrix)) as pool:
            ### WRONG ### imap allows us to iterate through results as they finish (does 4 VNTRs for a gene at a time due to processes=4)
            # imap should return the results one by one in the same order as the input tasks.
            ### CORRECT ### # processes up to `threads` genes in parallel, each worker handles all VNTRs for one gene
            # We wrap the pool in enumerate to keep track of count
            for count, results in enumerate(pool.imap(process_single_gene, tasks), start=1):
                result_list, gene_id, len_vntr_ids, counts = results
                print(f'Finished gene {gene_id} for {len_vntr_ids} VNTRs')
                variables = [countmodels, countslightsignificant, countinvariantgeneexp, countinvariantvntr, countfailLMM, countfailraneff, countnotenoughdata, countskipfreqremoval, countskiprefremoval]
                variables = [v + c for v, c in zip(variables, counts)]
                countmodels, countslightsignificant, countinvariantgeneexp, countinvariantvntr, countfailLMM, countfailraneff, countnotenoughdata, countskipfreqremoval, countskiprefremoval = variables
                all_accumulated_results.extend(result_list)
                # 3. Periodic write to disk (when list > 1000)
                if len(all_accumulated_results) >= 1000:
                    OUT.write('\n' + '\n'.join(['\t'.join(map(str, row)) for row in all_accumulated_results]))
                    all_accumulated_results = []  # Clear the list after writing
                if count%100 == 0:
                    print(f'Modeled {count} genes of total genes {numgenes}')
            # Append results to file after processing all VNTRs for a gene
            if all_accumulated_results:  # Write any remaining results after the loop
                OUT.write('\n' + '\n'.join(['\t'.join(map(str, row)) for row in all_accumulated_results]))

    print(f'{countmodels} number of linear models were completed.')
    print(f'{countslightsignificant} number of marginally significant (LRT < 0.000001) associations were found.')
    print(f'{countinvariantgeneexp} number of comparisons where invariant gene expression were found.')
    print(f'{countinvariantvntr} number of comparisons where invariant VNTR values were found.')
    print(f'{countskipfreqremoval} number of times most frequent allele was NOT removed and gene/vntr pair was skipped because it had allele freq < 0.05.')
    print(f'{countskiprefremoval} number of times reference allele was NOT removed and gene/vntr pair was skipped because it had allele freq < 0.05.')
    print(f'{countmodels - countslightsignificant} number of non-significant associations were found.')
    print(f'{countfailLMM} number of gene/VNTR comparisons that failed the LMM.')
    print(f'{countfailraneff} number of gene/VNTR comparisons that could fit LMM but failed allele specific random effect estimation.')
    print(f'{countnotenoughdata} number of gene/VNTR pairs skipped due to not enough data (n < 20 individuals, < 2 VNTR alleles)')
    print(f'Finished writing to {output_file}')

    with open(output_file.replace('.tsv', '.log') , "w") as OUT:
        OUT.write(
        f"For output file {output_file}:\n"
        f"{countmodels} number of linear models were completed.\n"
        f"{countslightsignificant} number of marginally significant (LRT < 0.000001) associations were found.\n"
        f"{countinvariantgeneexp} number of comparisons where invariant gene expression were found.\n"
        f"{countinvariantvntr} number of comparisons where invariant VNTR values were found.\n"
        f"{countskipfreqremoval} number of times most frequent allele was NOT removed and gene/vntr pair was skipped because it had allele freq < 0.05."
        f"{countskiprefremoval} number of times reference allele was NOT removed and gene/vntr pair was skipped because it had allele freq < 0.05."
        f"{countmodels - countslightsignificant} number of non-significant associations were found.\n"
        f"{countfailLMM} number of gene/VNTR comparisons that failed the LMM.\n"
        f"{countfailraneff} number of gene/VNTR comparisons that could fit LMM but failed allele specific random effect estimation.\n"
        f"{countnotenoughdata} number of gene/VNTR pairs skipped due to not enough data (n < 20 individuals, < 2 VNTR alleles)\n"
        )

    return  # results_df, best_results, significant_results



################################
################################
################################



def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


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


def compare_uhat_NoRidge_to_uhat_BLUP(
    u_hat_ols: pd.Series,
    u_hat_BLUP: pd.Series,
    save_path: str):
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA

    Compare variance components, BLUPs, and dosage matrices Z vs Z_ortho.
    Save all results in a single multipanel PDF.

    Parameters
    ----------
    sigma_g2 : float
        Genetic variance estimate (raw).
    sigma_e2 : float
        Residual variance estimate (raw).
    sigma_g2_ortho : float
        Genetic variance estimate (orthogonalized).
    sigma_e2_ortho : float
        Residual variance estimate (orthogonalized).
    u_hat : pd.Series
        BLUPs using raw sigma_g2 and sigma_e2.
    u_hat_ortho_sigmas : pd.Series
        BLUPs using sigma_g2_ortho and sigma_e2_ortho.
    Z : pd.DataFrame (n x m)
        Dosage matrix: individuals (rows) × alleles (columns), raw version.
    Z_ortho : pd.DataFrame (n x m)
        Dosage matrix orthogonalized to fixed effects X.
    save_path : str
        Path to save the PDF report.
    """

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    # ---------------- BLUP Magnitude Comparison ----------------
    # Allele-wise bar plot comparing magnitudes

    x = np.arange(len(u_hat_ols))  # numeric positions
    width = 0.35

    ax0 = axes[0]
    ax0.bar(x - width/2, u_hat_ols, width, label="No Ridge, OLS")
    ax0.bar(x + width/2, u_hat_BLUP, width, label="BLUP (with ridge)")

    ax0.set_xticks(x)
    ax0.set_xticklabels(u_hat_ols.index, rotation=90)  # label alleles
    ax0.set_title("u_hat Allele Effect Size Comparison")
    ax0.set_ylabel("u_hat")
    ax0.set_xlabel("Alleles")
    ax0.legend()

    # ---------------- Proportional Change in u_hat ----------------
    ax1 = axes[1]

    # proportional change = change relative to OLS, so negative if u_hat_BLUP is closer to 0 and positive if u_hat_BLUP further from 0
    uhat_prop_change = (np.abs(u_hat_BLUP) - np.abs(u_hat_ols)) / (np.abs(u_hat_ols) + 1e-08)  # small value avoids div by zero

    ax1.bar(x, uhat_prop_change, color="purple", label="Negative if |BLUP| < |OLS|")
    ax1.set_xticks(x)
    ax1.set_xticklabels(u_hat_ols.index, rotation=90)  # label alleles
    ax1.set_title("u_hat Proportional Change (due to ridge penalty)")
    ax1.set_ylabel("(|u_hat_BLUP| - |u_hat_ols|) / (|u_hat_ols| + 1e-08)")
    ax1.set_xlabel("Alleles")
    ax1.legend()

    # ---------------- BLUP Comparison ----------------
    ax2 = axes[2]
    ax2.scatter(u_hat_ols, u_hat_BLUP, alpha=0.5)
    # label each allele (assuming u_hat_ols index has allele IDs)
    for i, allele in enumerate(u_hat_ols.index):
        ax2.text(u_hat_ols.iloc[i], u_hat_BLUP.iloc[i], allele, fontsize=14, alpha=0.7)
    ax2.set_xlabel("u_hat_ols (no ridge)")
    ax2.set_ylabel("u_hat_BLUP (with ridge)")
    ax2.set_title("u_hat Correlation: OLS vs BLUP")

    # Regression line
    slope, intercept = np.polyfit(u_hat_ols, u_hat_BLUP, 1)
    x_line = np.linspace(u_hat_ols.min(), u_hat_ols.max(), 100)
    ax2.plot(x_line, slope * x_line + intercept, color="red", label=f"Fit slope={slope:.3f}")
    ax2.plot(x_line, x_line, color="black", linestyle="--", label="y=x, slope=1, if uhat values were identical")
    ax2.legend()

    corr = np.corrcoef(u_hat_ols, u_hat_BLUP)[0, 1]
    ax2.text(0.05, 0.95, f"Corr={corr:.3f}", transform=ax2.transAxes, ha="left", va="top")

    print(f"Correlation between u_hat_ols and u_hat_BLUP: {corr:.3f}")
    print(f"OLS slope: {slope:.3f}, intercept: {intercept:.3f}")

    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("u_hat OLS: û = (ZtZ)^-1 Zt (y-XB) vs u_hat_BLUP: û = (ZtZ + λI)^-1 Zt (y-XB)", fontsize=14)
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Saved report to {save_path}")


def plot_r2_and_PCA(r2_fixed: float, r2_genetic: float, r2_residual: float, Z: pd.DataFrame, save_path: str):
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA

    Compare variance components, BLUPs, and dosage matrices Z vs Z_ortho.
    Save all results in a single multipanel PDF.

    Parameters
    ----------
    r2_fixed : float
        Fixed effects r2 estimate.
    r2_genetic : float
        Genetic r2 estimate.
    r2_residual : float
        Residual r2 estimate.
    Z : pd.DataFrame (n x m)
        Dosage matrix: individuals (rows) × alleles (columns), centered version.
    save_path : str
        Path to save the PDF report.
    """

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # ---------------- r2 bar plot ----------------
    components = ["fixed_r2", "genetic_r2", "residual_r2"]
    r2_vals = [r2_fixed, r2_genetic, r2_residual]

    x = np.arange(len(components))
    width = 0.5
    axes[0].bar(x, r2_vals, width, label=f"Raw: r2_f={r2_fixed:.3f}, r2_g={r2_genetic:.3f}, r2_r={r2_residual:.3f}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(components)
    axes[0].set_ylabel("R²")
    axes[0].set_title("R²: fixed, genetic, & residual")
    axes[0].legend()

    # ---------------- PCA Allele loadings comparison ----------------
    def plot_pca(ax, M, title):
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(M)

        # scatter of individuals
        ax.scatter(pcs[:, 0], pcs[:, 1], alpha=0.3, label="Individuals")

        # top 10 allele loadings (absolute largest variance contribution)
        loadings = pca.components_.T
        norms = np.linalg.norm(loadings, axis=1)
        top_idx = np.argsort(norms)[-10:]

        for i in top_idx:
            ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                     color='red', alpha=0.7, head_width=0.05)
            ax.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, 
                    M.columns[i], fontsize=12, color='black')

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.legend()

    plot_pca(axes[1], Z, "PCA on Z (centered)")

    # Save combined
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("Comparison of Variance Components, BLUPs, and PCA of Dosage Matrices", fontsize=14)
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Saved report to {save_path}")


def plot_K_mean_eigenstructure(Z_raw, save_path=None, corr_threshold=0.1, eps_eig=1e-10, use_pca=False):
    """
    Visualize how mean allele-frequency structure affects K_raw vs K_centered.
    Includes eigenvalue filtering for numerical stability and signed annotation formatting.
    """
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib import cm

    indiv_labels = list(Z_raw.index)
    n, m = Z_raw.shape

    # (1) Center columns (alleles)
    allele_means = Z_raw.mean(axis=0).to_numpy()
    Z_centered = Z_raw - allele_means

    # (2) Compute Gram matrices
    Zr = Z_raw.to_numpy()
    Zc = Z_centered.to_numpy()
    K_raw = Zr @ Zr.T
    K_centered = Zc @ Zc.T

    # (3) Compute mean-structure scores
    mean_structure_raw = Zr @ allele_means
    mean_structure_centered = Zc @ allele_means
    mean_structure_z = (mean_structure_raw - mean_structure_raw.mean()) / mean_structure_raw.std()
    mean_structure_z_centered = (mean_structure_centered - mean_structure_centered.mean()) / mean_structure_centered.std()
    print("mean_structure_raw mean:", np.mean(mean_structure_raw))
    print("mean_structure_centered mean:", np.mean(mean_structure_centered))  # should be near zero
    print("mean_structure_z mean:", np.mean(mean_structure_z))
    print("mean_structure_z_centered mean:", np.mean(mean_structure_z_centered))


    # (4) Eigendecomposition or PCA
    if not use_pca:
        # (4) Eigen decomposition (more stable than PCA)
        evals_raw, evecs_raw = np.linalg.eigh(K_raw)
        evals_cent, evecs_cent = np.linalg.eigh(K_centered)
        # numeric stability: clip small negatives
        evals_raw = np.where(evals_raw < eps_eig, 0.0, evals_raw)
        evals_cent = np.where(evals_cent < eps_eig, 0.0, evals_cent)
        # sort descending
        idx_raw = np.argsort(evals_raw)[::-1]
        idx_cent = np.argsort(evals_cent)[::-1]
        evals_raw, evecs_raw = evals_raw[idx_raw], evecs_raw[:, idx_raw]
        evals_cent, evecs_cent = evals_cent[idx_cent], evecs_cent[:, idx_cent]
        # project individuals into first 2 PCs
        pcs_raw = evecs_raw[:, :2] * np.sqrt(evals_raw[:2])
        pcs_cent = evecs_cent[:, :2] * np.sqrt(evals_cent[:2])
    else:
        pca_raw = PCA().fit(K_raw)
        pca_cent = PCA().fit(K_centered)
        evals_raw = pca_raw.explained_variance_
        evals_cent = pca_cent.explained_variance_
        pcs_raw = pca_raw.transform(K_raw)[:, :2]
        pcs_cent = pca_cent.transform(K_centered)[:, :2]
        evecs_raw = pca_raw.components_.T
        evecs_cent = pca_cent.components_.T

    # (5) Correlation of eigenvectors with mean-structure (using same mean_structure_z)
    def compute_r_and_r2(evecs, mean_vec):
        n = evecs.shape[1]
        r_vals = np.full(n, np.nan)
        if np.nanstd(mean_vec) > 0:
            for i in range(n):
                v = evecs[:, i]
                if np.std(v) > 0:
                    r_vals[i] = np.corrcoef(v, mean_vec)[0, 1]
        r2_vals = r_vals**2
        return r_vals, r2_vals

    r_values_raw, r2_raw = compute_r_and_r2(evecs_raw, mean_structure_z)
    r_values_cent, r2_cent = compute_r_and_r2(evecs_cent, mean_structure_z)

    best_idx_raw = np.nanargmax(np.abs(r_values_raw))
    best_idx_cent = np.nanargmax(np.abs(r_values_cent))
    r_best_raw = r_values_raw[best_idx_raw]
    r_best_cent = r_values_cent[best_idx_cent]

    print(f"[K_raw] best_idx={best_idx_raw}, r_best={r_best_raw:+.3f}, r2={r_best_raw**2:.3f}")
    print(f"[K_cent] best_idx={best_idx_cent}, r_best={r_best_cent:+.3f}, r2={r_best_cent**2:.3f}")

    # (6) Plotting
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cmap = cm.coolwarm

    # ---------- Top row: PC scatter plots ----------
    sc1 = axes[0, 0].scatter(
        pcs_raw[:, 0], pcs_raw[:, 1],
        c=mean_structure_z, cmap=cmap, s=70, edgecolor="k"
    )
    plt.colorbar(sc1, ax=axes[0, 0], label="Mean-structure z-score (raw Z)")
    for i in range(n):
        axes[0, 0].text(
            pcs_raw[i, 0] + 0.04, pcs_raw[i, 1],
            f"{mean_structure_raw[i]:+.2f}",
            fontsize=8, color="black"
        )
    axes[0, 0].set_title("K_raw eigenmap — colored by mean-structure z-score")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")

    sc2 = axes[0, 1].scatter(
        pcs_cent[:, 0], pcs_cent[:, 1],
        c=mean_structure_z, cmap=cmap, s=70, edgecolor="k"
    )
    plt.colorbar(sc2, ax=axes[0, 1], label="Mean-structure z-score (same scale)")
    for i in range(n):
        axes[0, 1].text(
            pcs_cent[i, 0] + 0.04, pcs_cent[i, 1],
            f"{mean_structure_centered[i]:+.2f}",
            fontsize=8, color="black"
        )
    axes[0, 1].set_title("K_centered eigenmap — same coloring (alignment removed)")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")

    # ---------- Bottom row: Scree plots ----------
    var_expl_raw = evals_raw / evals_raw.sum()
    var_expl_cent = evals_cent / evals_raw.sum()  # keep y-scale comparable
    max_show = min(30, len(var_expl_raw))

    # Scree for K_raw
    axes[1, 0].plot(var_expl_raw[:max_show], "o-", color="blue")
    if not np.isnan(r_best_raw) and abs(r_best_raw) >= corr_threshold:
        axes[1, 0].axvline(best_idx_raw, color="red", linestyle="--",
                           label=(f"Most mean-structure-aligned eigvec: {best_idx_raw}\n"
                                  f"r={r_best_raw:+.2f}, r²={r_best_raw**2:.2f}\n"
                                  f"Var explained: {var_expl_raw[best_idx_raw]:.3f}"))
    else:
        axes[1, 0].text(0.5, 0.8,
                        f"No strong mean structure (r={r_best_raw:+.2f})",
                        transform=axes[1, 0].transAxes,
                        ha="center", fontsize=10, color="gray")
    axes[1, 0].set_title("Eigenvalues (proportion variance): K_raw")
    axes[1, 0].set_xlabel("Component")
    axes[1, 0].set_ylabel("Proportion of variance")
    axes[1, 0].legend()

    # Scree for K_centered
    axes[1, 1].plot(var_expl_cent[:max_show], "o-", color="orange")
    if not np.isnan(r_best_cent) and abs(r_best_cent) >= corr_threshold:
        axes[1, 1].axvline(best_idx_cent, color="red", linestyle="--",
                           label=(f"Most mean-structure-aligned eigvec: {best_idx_cent}\n"
                                  f"r={r_best_cent:+.2f}, r²={r_best_cent**2:.2f}\n"
                                  f"Var explained: {var_expl_cent[best_idx_cent]:.3f}"))
    else:
        axes[1, 1].text(0.5, 0.8,
                        f"No strong mean structure (r={r_best_cent:+.2f})",
                        transform=axes[1, 1].transAxes,
                        ha="center", fontsize=10, color="gray")
    axes[1, 1].set_title("Eigenvalues (proportion variance): K_centered")
    axes[1, 1].set_xlabel("Component")
    axes[1, 1].set_ylabel("Proportion of variance")
    axes[1, 1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Info] Saved plot to {save_path}")
    else:
        plt.show()

    # ---------- Additional figure: r² alignment barplot ----------
    topk = 20
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 3))
    ind = np.arange(1, topk + 1)
    ax2.bar(ind - 0.2, r2_raw[:topk], width=0.4, label="K_raw", color="C0", alpha=0.8)
    ax2.bar(ind + 0.2, r2_cent[:topk], width=0.4, label="K_centered", color="C1", alpha=0.6)
    ax2.set_xlabel("Component")
    ax2.set_ylabel("r² (fraction of mean-structure variance)")
    ax2.legend()
    ax2.set_title("Alignment of eigenvectors with mean-structure (r²)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.pdf', '.r2.Barplot.pdf'), dpi=300)
        print(f"[Info] Saved plot to {save_path}")
    else:
        plt.show()

    print(f"[Info] Correlation of top mean-structure eigenvector: r = {r_best_raw:+.3f} (component {best_idx_raw})")

    return {
        "K_raw": K_raw,
        "K_centered": K_centered,
        "r_best": r_best_raw,
        "best_idx": best_idx_raw,
        "r_values": r_values_raw,
        "evals_raw": evals_raw,
        "evals_centered": evals_cent,
        "mean_structure_raw": mean_structure_raw,
        "mean_structure_centered": mean_structure_centered
    }


def summarize_shrinkage(Z, sigma_g2, sigma_e2, y_resid=None, save_path=None, top_k=10):
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.linalg import solve

    Summarize eigenstructure of allele dosage matrix Z and visualize BLUP shrinkage.

    Parameters
    ----------
    Z : pd.DataFrame
        Allele dosage matrix (n_indiv × m_alleles).
    sigma_g2 : float
        Genetic variance component.
    sigma_e2 : float
        Residual variance component.
    y_resid : np.ndarray, optional
        Residuals after fixed effect regression (n × 1).
    save_path : str or None
        If provided, save figure to this path as PDF.
    top_k : int
        Number of top eigenvectors to display in loadings heatmap.
    """
    print(f"Plotting summary of allele shrinkage: {save_path if save_path else 'showing interactively'}")

    ZM = Z.values
    n, m = ZM.shape

    # --- Step 1. Gram matrix ---
    S = ZM.T @ ZM  # m × m

    # --- Step 2. Eigen-decomposition ---
    mu, U = np.linalg.eigh(S)  # mu = eigenvalues (vector mx1, ecach eigenvalue tells amount of information 'variance' on each eigevector), U = eigenvectors (mxm matrix, columns Uk = eigenvectors, rows Uj = alleles, Ujk is loading of allele j on eigenvector k) # symmetric
    order = np.argsort(mu)[::-1]
    mu, U = mu[order], U[:, order]  # sort descending

    # --- Step 3. Shrinkage factors ---
    lam = sigma_e2 / sigma_g2  # noise-to-signal ratio
    shrink_factors = mu / (mu + lam)  # shrinkage due to ridge penalty (0-1) (based upon signal-to-noise ratio) # PER EIGENVECTOR
    allele_shrink = (U**2) @ shrink_factors  # effective shrinkage PER ALLELE
    # Each allele j is represented across eigenvectors by its squared loadings
    # 𝑈𝑗𝑘^2 measures the proportion of variance of allele j explained by eigenvector k
    # allele_shrinkj = m∑k=1 ​Ujk^2​⋅sk

    # --- Step 4. Raw and penalized scores ---
    # Raw scores and OLS-scaled raw effects
    raw_scores, u_hat_spec, beta_marginal_no_colinearity, beta_ols, u_hat_blup, u_hat_ridge = None, None, None, None, None, None
    if y_resid is not None:
        r = y_resid.reshape(-1, 1)
        b = ZM.T @ r  # raw allele scores (dot products)
        raw_scores = b.flatten()

        # (a) Spectral shrinkage
        u_hat_spec = U @ (np.diag(shrink_factors) @ (U.T @ b))
        u_hat_spec = u_hat_spec.flatten()

        #fine but not scaled# # lamda only correction
        #fine but not scaled# left = lam * np.eye(m)
        #fine but not scaled# lambda_only = solve(left, b, assume_a="pos")
        #fine but not scaled# lambda_only = lambda_only.flatten()

        # (diag(ZᵀZ))-1Zᵀr
        # This is the effect per allele copy if all other alleles are ignored (like a univariate regression of r onto allele j only).
        # It's equivalent to the OLS coefficient when columns are orthogonal to each other or when you regress each allele separately
        # accounts for per-allele scaling but not off-diagonals (i.e. no collinearity correction)
        ZtZ = ZM.T @ ZM
        ZtZ_diag = np.diag(np.diag(ZtZ))  # * np.diag() twice to remove then add diagonal back to matrix # allele "information" (counts × dosage)
        beta_marginal_no_colinearity = np.linalg.pinv(ZtZ_diag) @ b  # OLS-like scaling i.e. per-allele normalization by allele "information" (counts × dosage)
        beta_marginal_no_colinearity = beta_marginal_no_colinearity.flatten()
        # ols_effects = np.divide(raw_scores, ZtZ_diag, out=np.zeros_like(raw_scores), where=info_diag>0)

        # diagonal(ZtZ) plus lambda correction
        left = ZtZ_diag + lam * np.eye(m)
        diag_plus_lambda = solve(left, b).flatten()  # np.linalg.solve

        # (b) OLS beta estimates, no ridge form: (ZᵀZ)⁻¹ Zᵀr # no ridge / sigmas
        # standard least-squares solution for regression of r on all columns of Z jointly.
        # It accounts for off-diagonals (collinearity) and gives the best linear unbiased estimator under homoskedastic errors
        beta_ols = np.linalg.pinv(ZtZ) @ b
        beta_ols = beta_ols.flatten()

        # (b) BLUP form: û = σ_g^2 Zᵀ V⁻¹ r, with V = σ_g^2 Z Zᵀ + σ_e^2 I
        V = sigma_g2 * (ZM @ ZM.T) + sigma_e2 * np.eye(n)
        V_inv_r = solve(V, r, assume_a="pos")
        u_hat_blup = sigma_g2 * (ZM.T @ V_inv_r)
        u_hat_blup = u_hat_blup.flatten()

        # (c) Ridge form: (ZᵀZ + λI)⁻¹ Zᵀr
        left_simple = ZtZ + lam * np.eye(m)
        right_simple = ZM.T @ r
        u_hat_ridge = solve(left_simple, right_simple).flatten()

    # --- Step 5. Visualization ---
    n_plots = 5 if y_resid is None else 7
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5*n_plots))

    # (a) Eigenvalues and shrinkage
    axes[0].plot(range(1, m+1), mu, "o-", label="Eigenvalue μ (raw)")
    axes[0].plot(range(1, m+1), (mu / mu.sum())*100, "s-", label="% variance explained")
    axes[0].plot(range(1, m+1), shrink_factors*100, "x-", label="Shrinkage μ/(μ+λ) * 100")
    axes[0].set_title("Eigenvalues and shrinkage factors per eigenvector")
    axes[0].set_xlabel("Eigenvector index (sorted)")
    axes[0].legend()

    # (b) Eigenvector loadings
    k = min(m, top_k)
    loadings = pd.DataFrame(U[:, :k],
                            index=Z.columns,
                            columns=[f"EV{i+1}" for i in range(k)])
    sns.heatmap(loadings, cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title(f"Allele loadings on top {k} eigenvectors")
    axes[1].set_xlabel("Eigenvectors")

    # (b2) Squared loadings (raw U²)
    sq_loadings = pd.DataFrame(U[:, :k]**2,
                               index=Z.columns,
                               columns=[f"EV{i+1}" for i in range(k)])
    sns.heatmap(sq_loadings, cmap="magma", ax=axes[2])  # viridis
    axes[2].set_title(f"Allele contributions (U², raw)")
    axes[2].set_xlabel("Eigenvectors")

    # (b3) Column-normalized squared loadings
    sq_loadings_norm = sq_loadings.div(sq_loadings.sum(axis=0), axis=1)
    sns.heatmap(sq_loadings_norm, cmap="magma", ax=axes[3])
    axes[3].set_title(f"Allele contributions (U², column-normalized)")
    axes[3].set_xlabel("Eigenvectors")

    # (c) Per-allele shrinkage
    pd.Series(allele_shrink, index=Z.columns).plot.bar(ax=axes[4])
    axes[4].set_title("Effective shrinkage (due to ridge penalty) per allele")
    axes[4].set_ylabel("Shrinkage factor (0–1)")
    axes[4].set_xlabel("Alleles")

    # (d) Raw vs shrunk allele effect values based upon lamda (if y_resid provided)
    if y_resid is not None:
        width = 0.2
        ind = np.arange(m)
        axes[5].bar(ind - 1.5*width, raw_scores, width, label="Raw Zᵀr")
        axes[5].bar(ind - 0.5*width, u_hat_spec, width, label="Spectral shrinkage")
        axes[5].set_xticks(ind)
        axes[5].set_xticklabels(Z.columns, rotation=90)
        axes[5].set_title("Allele effect estimates: raw vs penalized")
        axes[5].set_xlabel("Alleles")
        axes[5].legend()

        # (e) Allele count normaalization, OLS, BLUP, lambda (noise-to-signal ratio) Ridge 
        # axes[6].bar(ind - 1.5*width, lambda_only, width, label="λ only = (λI)⁻¹ Zᵀr")
        axes[6].bar(ind - 2.5*width, beta_marginal_no_colinearity, width, label="β_ScaledByDosage = (diag(ZᵀZ))⁻¹ Zᵀr")
        axes[6].bar(ind - 1.5*width, diag_plus_lambda, width, label="β_ScaledByDosage + λ = (diag(ZᵀZ) + λI)⁻¹ Zᵀr")
        axes[6].bar(ind - 0.5*width, beta_ols, width, label="β_ScaledByDosage + Collinearity: β_ols = (ZᵀZ)⁻¹ Zᵀr")
        axes[6].bar(ind + 0.5*width, u_hat_blup, width, label="BLUP form: û = σ_g^2 Zᵀ V⁻¹ r")
        axes[6].bar(ind + 1.5*width, u_hat_ridge, width, label="Ridge form: û = (ZᵀZ + λI)⁻¹ Zᵀr")
        axes[6].set_xticks(ind)
        axes[6].set_xticklabels(Z.columns, rotation=90)
        axes[6].set_title(f"Allele effect estimates with different penalties:\nSigma_g2={sigma_g2}, Sigma_e2={sigma_e2}, lambda={lam}")  # \nDosage Scaled (no penalty) vs λ only vs Dosage scaling + Colinearity (OLS) vs BLUP vs Ridge with lambda (û) effect estimates")
        axes[6].set_ylabel("Allele specific effect predictions on gene expression residuals")
        axes[6].set_xlabel("Alleles")
        axes[6].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    # --- Step 6. Return results ---
    results = {
        "S": S,
        "eigenvalues": mu,
        "U": U,
        "shrinkage_factors": shrink_factors,
        "allele_effective_shrinkage": allele_shrink,
    }
    if y_resid is not None:
        results["raw_scores"] = raw_scores
        results["u_hat_spec"] = u_hat_spec
        results["beta_marginal_no_colinearity"] = beta_marginal_no_colinearity
        results["beta_ols"] = beta_ols
        results["u_hat_blup"] = u_hat_blup
        results["u_hat_ridge"] = u_hat_ridge
    return results


def plot_heatmaps(Z, save_path=None):
    """
    Plot heatmaps for the dosage matrix and similarity matrix.
    """
    print("Plotting heatmaps for Z, ZZᵀ, and ZᵀZ...")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3))

    font_size = 10
    font_family = "Arial"
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size - 2,
        "axes.labelsize": font_size - 2,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2
    })
    # Compute ZZᵀ
    ZZt = Z @ Z.T
    # Compute ZᵀZ
    ZtZ = Z.T @ Z

    print(f'Boundary values of Z: min={Z.values.min()}, max={Z.values.max()}')
    print(f'Boundary values of ZZᵀ: min={ZZt.values.min()}, max={ZZt.values.max()}')
    print(f'Boundary values of ZᵀZ: min={ZtZ.values.min()}, max={ZtZ.values.max()}')

    sns.heatmap(Z, ax=axes[0], cmap="YlGnBu", cbar=True, linewidths=0.5, linecolor='gray')  # cmap=Oranges
    axes[0].set_title(f"Dosage Matrix\n(Indiv={Z.shape[0]}  × Alleles={Z.shape[1]})")
    axes[0].set_xlabel("Alleles")
    axes[0].set_ylabel("Individuals")
    # Suppress y-axis tick labels on subplot 0
    axes[0].tick_params(labelleft=False)

    sns.heatmap(ZZt, ax=axes[1], cmap="YlGnBu", cbar=True)
    axes[1].set_title(f"Allele Covariance per Indiv\n(Indiv={Z.shape[0]} x Indiv={Z.shape[0]})")  # (Indiv. × Indiv.)
    axes[1].set_xlabel("Individuals")
    axes[1].set_ylabel("Individuals")
    # Suppress both x and y tick labels on subplot 1
    axes[1].tick_params(labelbottom=False, labelleft=False)

    sns.heatmap(ZtZ, ax=axes[2], cmap="YlGnBu", cbar=True)
    axes[2].set_title(f"Individual covariance per allele\n(Alleles={Z.shape[1]} x Alleles={Z.shape[1]})")  # (Alleles × Alleles)
    axes[2].set_xlabel("Alleles")
    axes[2].set_ylabel("Alleles")

    # sns.heatmap(allele_T_df, ax=axes[3], cmap="YlGnBu", cbar=True)
    # axes[3].set_title("(Allele-Allele Cov)^-1 (5x5)")  # (Alleles × Alleles)
    # axes[3].set_xlabel("Alleles")
    # axes[3].set_ylabel("")

    # Tight layout and save/show
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    print("Done plotting heatmaps.")

    return


def plot_BLUPs_VS_best_var(Y, y_resid, Z_raw, Z_used, u_hat_locus, allele_frequencies, lr, pval, n_indiv, n_allele, comps, save_path=None):
    """
    Plot individual-level BLUPs against gene expression and residuals.
    Parameters:
        Y (pd.Series): Raw gene expression values.
        y_resid (np.ndarray): Residuals after fixed effect regression.
        Z (pd.DataFrame): Allele dosage matrix (raw).
        Z_ortho (pd.DataFrame): Allele dosage matrix orthogonalized to fixed effects.
        u_hat_locus (pd.Series): Estimated allele effects (BLUPs).
    """
    Z = Z_raw  # Z_raw has to be raw dosage matrix to correctly label genotypes
    sigma_g2, sigma_e2 = comps['sigma_g2'], comps['sigma_e2']
    var_genetic, var_residual, var_fixed = comps['var_genetic'], comps['var_residual'], comps['var_fixed']
    r2_genetic, r2_residual, r2_fixed = comps['r2_genetic'], comps['r2_residual'], comps['r2_fixed']
    lambda_pen, num_eigenvals = comps['lambda_pen'], len(comps['QS_full'][1])
    allele_removed = comps['allele_removed'] if 'allele_removed' in list(comps.keys()) else None
    # Create a figure with a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Calculate individual-level BLUEs
    print("Calculating individual-level BLUEs using raw Z for allele dosage...")
    individual_blues = Z_used @ u_hat_locus.values

    # print(f'Z:\n{Z}')
    # print(f'Z_ortho:\n{Z_ortho}')
    # print(f'u_hat_locus:\n{u_hat_locus}')
    print(f'Summary stats for individual BLUPs: min={individual_blues.min()}, max={individual_blues.max()}, mean={individual_blues.mean()}, std={individual_blues.std()}')

    # Subplot 1: Raw Gene Expression (Y) vs. Individual BLUEs
    ax1 = axes[0, 0]
    ax1.scatter(individual_blues, Y, color='green', alpha=0.7, label="Data")
    slope, intercept = np.polyfit(individual_blues, Y, 1)
    # get unique genotypes with x and max y positions for annotation
    uniquegenotypes = {}
    ax1.plot(individual_blues, slope * individual_blues + intercept, color='red', label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")
    for i in range(Z.shape[0]):  # for each row
        alleles = Z.columns[np.where(Z.iloc[i] > 0)[0]]  # get column names (allele number) when cell value > 0
        genotype = f"{alleles[0]}/{alleles[1]}" if len(alleles) == 2 else f"{alleles[0]}/{alleles[0]}"  # there are always just two columns in Z_raw, one allele per column
        # ax1.annotate(genotype, (individual_blues.iloc[i], y_resid[i]), fontsize=12, alpha=0.7, ha="center", va="bottom")
        try:
            uniquegenotypes[genotype]
        except KeyError:
            uniquegenotypes[genotype] = (individual_blues.iloc[i], Y.iloc[i])
        else:
            if Y.iloc[i] > uniquegenotypes[genotype][1]:
                uniquegenotypes[genotype] = (individual_blues.iloc[i], Y.iloc[i])
    for genotype, (xpos, ypos) in uniquegenotypes.items():
        ax1.annotate(genotype, (xpos, ypos+(0.1*ypos)), fontsize=12, alpha=1, ha="center", va="bottom")
    ax1.set_xlabel("Individual's BLUP: sum(Z[i,] * û)")  # û is BLUP 'predicted allele effects on gene expression effect size'
    ax1.set_ylabel("Gene Expression (Y)")
    ax1.set_title("Gene Expression vs. Individual BLUPs")
    ax1.legend()

    # Subplot 2: Residuals (y_resid) vs. Individual BLUEs
    ax2 = axes[0, 1]
    ax2.scatter(individual_blues, y_resid, color='blue', alpha=0.7, label="Residuals")
    slope_r, intercept_r = np.polyfit(individual_blues, y_resid, 1)
    ax2.plot(individual_blues, slope_r * individual_blues + intercept_r, color='red', label=f"Fit: y = {slope_r:.2f}x + {intercept_r:.2f}")
    # get unique genotypes with x and max y positions for annotation
    uniquegenotypes = {}
    for i in range(Z.shape[0]):
        alleles = Z.columns[np.where(Z.iloc[i] > 0)[0]]
        genotype = f"{alleles[0]}/{alleles[1]}" if len(alleles) == 2 else f"{alleles[0]}/{alleles[0]}"
        # ax2.annotate(genotype, (individual_blues.iloc[i], y_resid[i]), fontsize=12, alpha=0.7, ha="center", va="bottom")
        try:
            uniquegenotypes[genotype]
        except KeyError:
            uniquegenotypes[genotype] = (individual_blues.iloc[i], Y.iloc[i])
        else:
            if Y.iloc[i] > uniquegenotypes[genotype][1]:
                uniquegenotypes[genotype] = (individual_blues.iloc[i], Y.iloc[i])
    for genotype, (xpos, ypos) in uniquegenotypes.items():
        ax2.annotate(genotype, (xpos, ypos+(0.1*ypos)), fontsize=12, alpha=1, ha="center", va="bottom")
        # ax2.annotate(genotype, (individual_blues.iloc[i], y_resid[i]), fontsize=12, alpha=0.7, ha="center", va="bottom")
    ax2.set_xlabel("Individual's BLUP: sum(Z[i,] * û)")  # û is BLUP 'predicted allele effects on gene expression effect size'
    ax2.set_ylabel("Gene Expression Residuals (y_resid)")
    ax2.set_title("Gene Expression Residuals vs. Individual BLUPs")
    ax2.legend()

    # Subplot 3: Allele BLUEs vs. Allele Frequencies
    # Make allele_frequencies a list or Series, matching u_hat_locus index
    # num_individuals = Z.shape[0]
    # num_alleles = Z.shape[1]
    ### calc allele frequencies from Z
    ## total_alleles = Z.sum().sum()  # Total dosage across all alleles
    ## allele_freqs = pd.Series(Z.sum(axis=0) / total_alleles)
    ## allele_freqs = allele_frequencies.reindex(u_hat_locus.index)
    # Convert the supplied dictionary into a Series for easier indexing and plotting
    allele_freqs = pd.Series(allele_frequencies)
    # Reindex to match u_hat_locus order (ensures alignment)
    allele_freqs = allele_freqs.reindex(u_hat_locus.index)
    # print("Z columns:", Z.columns)
    # print("u_hat_locus index:", u_hat_locus.index)
    # print("Missing in u_hat_locus:", set(Z.columns) - set(u_hat_locus.index))
    # print("Extra in u_hat_locus:", set(u_hat_locus.index) - set(Z.columns))
    ax3 = axes[1, 0]
    for i, allele in enumerate(u_hat_locus.index):
        # freq = allele_frequencies_series[allele]
        # ax3.scatter(u_hat_locus.loc[allele], freq, color='purple', alpha=0.8)
        # ax3.annotate(f"{allele}", (u_hat_locus.loc[allele], freq), fontsize=12, alpha=0.8, ha="center", va="bottom")
        ax3.scatter(u_hat_locus.loc[allele], allele_freqs[allele], color='purple', alpha=0.8)
        ax3.annotate(f"{allele}", (u_hat_locus.loc[allele], allele_freqs[allele] + (0.1*allele_freqs[allele])), fontsize=12, alpha=1, ha="center", va="bottom")
    if allele_removed:
        # allele_removed effect is by definition 0
        ax3.scatter(0, allele_frequencies[allele_removed], color='red', alpha=0.8)
        ax3.annotate(f"Baseline: {allele_removed}", (0, allele_frequencies[allele_removed] + (0.1*allele_freqs[allele])), fontsize=12, alpha=1, ha="center", va="bottom")
    ax3.set_xlabel("Allele-specific BLUPs")
    ax3.set_ylabel("Allele Frequency")
    ax3.set_title("Allele Frequencies vs. Allele BLUP Effect Sizes")

    # Subplot 4: Display metadata as a table
    ax4 = axes[1, 1]
    ax4.axis('off')  # Hide axes
    table_data = [
        ["Number of Individuals", Z_used.shape[0]],
        ["Number of Alleles", Z_used.shape[1]],
        ["Baseline Allele (0 effect)", allele_removed if allele_removed else 'No allele set to 0 effect'],
        ["Fixed Effect Variation", f"{var_fixed:.4f}"],
        ["Variance Explained (sigma_g²)", f"{sigma_g2:.4f}"],
        ["Residual Variance (sigma_e²)", f"{sigma_e2:.4f}"],
        ["Lambda (sigma_e²/sigma_g²)", f"{lambda_pen:.4f}"],
        ["Genetic Variance", f"{var_genetic:.4f}"],
        ["Residual Variance", f"{var_residual:.4f}"],
        ["Number of non-zero Eigenvalues", f"{num_eigenvals}"],
        ["Fixed Effect R²", f"{r2_fixed:.4f}"],
        ["Genetic R²", f"{r2_genetic:.4f}"],
        ["Residual R²", f"{r2_residual:.4f}"],
        ["Likelihood Ratio", f"{lr:.2f}"],
        ["P-Value", f"{pval:.2e}"]
    ]
    table = ax4.table(cellText=table_data, colLabels=["Metric", "Value"], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.2)
    ax4.set_title("Model Summary", pad=20)

    # Tight layout and save/show
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_added_variable(y_resid, x_resid, r2, beta, se, pvalue, save_path=None, title="Added-Variable Plot"):
    
    print(f"Plotting added-variable regression: {save_path if save_path else 'showing interactively'}")
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(x_resid, y_resid, color='blue', alpha=0.7, label='Residuals')

    # Fit simple regression line
    model = sm.OLS(y_resid, sm.add_constant(x_resid)).fit()
    intercept = model.params.iloc[0]
    slope = model.params.iloc[1]

    x_vals = np.linspace(x_resid.min(), x_resid.max(), 200)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='red', label="Regression Line")

    # Annotation
    annotation_text = (
        f'R² = {r2:.4f}\n'
        f'β = {beta:.4f} ± {se:.4f}\n'
        f'P-value = {pvalue:.4e}'
    )
    ax.text(0.05, 0.95, annotation_text,
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_title(title)
    ax.set_xlabel('Residualized Predictor (X_resid)')
    ax.set_ylabel('Residualized Response (Y_resid)')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    return


def plot_simplelinear(Y_simple, X_simple, r2_simple, beta_simple, se_simple, pvalue_simple, model_simple, save_path=None, title='Simple Linear Regression Results'):
    """
    Plot simple linear regression results.
    Parameters:
        Y_simple (pd.Series): Response variable.
        X_simple (pd.Series): Predictor variable.
        r2_simple (float): R-squared value.
        beta_simple (float): Regression coefficient.
        se_simple (float): Standard error of the coefficient.
        pvalue_simple (float): P-value of the regression.
        model_simple: Fitted regression model object.
        save_path (str): Path to save the plot.
        title (str): Title of the plot.
    """
    print(f"Plotting simple linear regression results: {save_path if save_path else 'showing interactively'}")
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot of data points
    ax.scatter(X_simple, Y_simple, color='blue', alpha=0.7, label='Data Points')

    # Regression line

    x_vals = np.linspace(X_simple.min(), X_simple.max(), 100)
    params = model_simple.params
    if len(params) == 1:
        # No intercept
        intercept = 0
        slope = params.iloc[0]  # beta_simple
    else:
        intercept = params.iloc[0]
        slope = params.iloc[1]  # beta_simple

    y_vals = intercept + slope * x_vals
    # y_vals = beta_simple * x_vals
    ax.plot(x_vals, y_vals, color='red', label='Regression Line')

    # Annotations
    ax.set_title(title)
    ax.set_xlabel('Predictor Variable (X)')
    ax.set_ylabel('Response Variable (Y)')
    annotation_text = (f'R² = {r2_simple:.4f}\n'
                       f'β = {beta_simple:.4f} ± {se_simple:.4f}\n'
                       f'P-value = {pvalue_simple:.2e}' if r2_simple is not None else '')
                       # f'Prop. Genetic Variance = {prop_genetic_variance:.4f}' if prop_genetic_variance is not None else '')
    ax.text(0.05, 0.95, annotation_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    return


def make_plots_for_paper(y_resid, Z_raw, Z_used, u_hat_locus, allele_frequencies, comps, save_path=None):
    """
    Plot individual-level BLUPs against gene expression and residuals.
    Parameters:
        Y (pd.Series): Raw gene expression values.
        y_resid (np.ndarray): Residuals after fixed effect regression.
        Z (pd.DataFrame): Allele dosage matrix (raw).
        Z_ortho (pd.DataFrame): Allele dosage matrix orthogonalized to fixed effects.
        u_hat_locus (pd.Series): Estimated allele effects (BLUPs).
    """
    # format for paper

    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"]
    # plt.rcParams["font.size"] = 12
    # paper_rc = {
    #     "font.family": "serif",
    #     "font.serif": ["Times New Roman"],
    #     "font.size": 12,
    #     "axes.spines.top": False,
    #     "axes.spines.right": False
    # }
    font_size = 10
    font_family = "Arial"
    # plt.rcParams.update()
    paper_rc = {
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "axes.spines.top": False,
        "axes.spines.right": False
    }

    with plt.rc_context(paper_rc):
        # plt.figure(figsize=(10, 6))
        # plt.figure(figsize=(6, 4))
        # Create a figure with a 2x2 grid
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'width_ratios':[3,1]})  # figsize6,3 width3,1
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        allele_removed = comps['allele_removed'] if 'allele_removed' in list(comps.keys()) else None
        Z = Z_raw  # Z_raw has to be raw dosage matrix to correctly label genotypes

        # Calculate individual-level BLUEs
        print("Calculating individual-level BLUEs using raw Z for allele dosage...")
        individual_blues = Z_used @ u_hat_locus.values

        # print(f'Z:\n{Z}')
        # print(f'Z_ortho:\n{Z_ortho}')
        # print(f'u_hat_locus:\n{u_hat_locus}')
        print(f'Summary stats for individual BLUPs: min={individual_blues.min()}, max={individual_blues.max()}, mean={individual_blues.mean()}, std={individual_blues.std()}')

        ax2 = axes[0]
        # plot 2: Residuals (y_resid) vs. Individual BLUEs
        ax2.scatter(individual_blues, y_resid, color='blue', alpha=0.7, label="Individual")
        slope_r, intercept_r = np.polyfit(individual_blues, y_resid, 1)
        ax2.plot(individual_blues, slope_r * individual_blues + intercept_r, color='red') # , label=f"Fit: y = {slope_r:.2f}x + {intercept_r:.2f}"
        # ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)  # vertical line at x=0 to show zero individual BLUP and intercept
        # get unique genotypes with x and max y positions for annotation
        uniquegenotypes = {}
        for i in range(Z.shape[0]):
            alleles = Z.columns[np.where(Z.iloc[i] > 0)[0]]
            genotype = f"{alleles[0]}/{alleles[1]}" if len(alleles) == 2 else f"{alleles[0]}/{alleles[0]}"
            # ax2.annotate(genotype, (individual_blues.iloc[i], y_resid[i]), fontsize=12, alpha=0.7, ha="center", va="bottom")
            try:
                uniquegenotypes[genotype]
            except KeyError:
                uniquegenotypes[genotype] = (individual_blues.iloc[i], y_resid[i])
            else:
                if y_resid[i] > uniquegenotypes[genotype][1]:
                    uniquegenotypes[genotype] = (individual_blues.iloc[i], y_resid[i])
        # for genotype, (xpos, ypos) in uniquegenotypes.items():
        #     if ypos+(0.0025*(abs(ypos))) < ax2.get_ylim()[1]:  # only annotate if within y-axis limits
        #         ax2.annotate(genotype, (xpos+((1/xpos)*0.025), ypos+0.3+(0.02*(abs(ax2.get_ylim()[1])))), fontsize=12, alpha=1, ha="center", va="bottom")  # ypos+0.2+(0.0025*(abs(ypos))
        #     else:
        #         ax2.annotate(genotype, (xpos, ypos-(0.0025*(abs(ypos)))), fontsize=12, alpha=1, ha="center", va="top")
        #         newxmin = ax3.get_xlim()[0] + ax3.get_xlim()[0]*0.05 if ax3.get_xlim()[0] < 0 else ax3.get_xlim()[0] - ax3.get_xlim()[0]*0.05
        
        # Get current plot limits to calculate relative offsets
        ymin, ymax = ax2.get_ylim()
        y_range = ymax - ymin
        for genotype, (xpos, ypos) in uniquegenotypes.items():
            # Define an upper threshold (e.g., top 15% of the plot area)
            if ypos > ymax - (0.15 * y_range): 
                # Near the top boundary: place label below the data point
                ax2.annotate(
                    genotype, 
                    (xpos, ypos - (0.05 * y_range)), 
                    fontsize=12, alpha=1, ha="center", va="top"
                )
            else:
                # Default/Low expression: safely place label above the data point
                ax2.annotate(
                    genotype, 
                    (xpos, ypos + (0.05 * y_range)), 
                    fontsize=12, alpha=1, ha="center", va="bottom"
                )
        newxmin = ax2.get_xlim()[0] + ax2.get_xlim()[0]*0.05 if ax2.get_xlim()[0] < 0 else ax2.get_xlim()[0] - ax2.get_xlim()[0]*0.05
        ax2.set_xlim(newxmin, ax2.get_xlim()[1])
        ax2.set_xlabel("Individual BLUPs")  # û is BLUP 'predicted allele effects on gene expression effect size'
        ax2.set_ylabel("Gene Expression Residuals")
        # ax2.set_title("Gene Expression Residuals vs. Individual BLUPs")
        ax2 = plt.gca()
        # ax2.spines["top"].set_visible(False)
        # ax2.spines["right"].set_visible(False)
        # ax2.legend()

        # Subplot 3: Allele BLUEs vs. Allele Frequencies
        allele_freqs = pd.Series(allele_frequencies)
        # Reindex to match u_hat_locus order (ensures alignment)
        allele_freqs = allele_freqs.reindex(u_hat_locus.index)

        ax3 = axes[1]
        for i, allele in enumerate(u_hat_locus.index):
            # freq = allele_frequencies_series[allele]
            # ax3.scatter(u_hat_locus.loc[allele], freq, color='purple', alpha=0.8)
            # ax3.annotate(f"{allele}", (u_hat_locus.loc[allele], freq), fontsize=12, alpha=0.8, ha="center", va="bottom")
            ax3.scatter(u_hat_locus.loc[allele], allele_freqs[allele], color='purple', alpha=0.8)
            if allele_freqs[allele] + 0.0025 < ax3.get_ylim()[1]:  # only annotate if within y-axis limits
                ax3.annotate(f"{allele}", (u_hat_locus.loc[allele], allele_freqs[allele]+ 0.005), fontsize=12, alpha=1.0, ha="center", va="bottom")  # allele_freqs[allele]+ 0.1 + 0.0025
            else:
                ax3.annotate(f"{allele}", (u_hat_locus.loc[allele], allele_freqs[allele] - 0.005), fontsize=12, alpha=1.0, ha="center", va="top")
        if allele_removed:
            # allele_removed effect is by definition 0
            ax3.scatter(0, allele_frequencies[allele_removed], color='red', alpha=0.8)
            ax3.annotate(f"Baseline: {allele_removed}", (0, allele_frequencies[allele_removed] + 0.0025), fontsize=12, alpha=1.0, ha="center", va="bottom")
        newxmin = ax3.get_xlim()[0] + ax3.get_xlim()[0]*0.05 if ax3.get_xlim()[0] < 0 else ax3.get_xlim()[0] - ax3.get_xlim()[0]*0.05
        ax3.set_xlim(newxmin, ax3.get_xlim()[1])
        ax3.set_xlabel("Allele BLUPs")
        ax3.set_ylabel("Allele Frequency")
        # ax3.set_title("Allele Frequencies vs. Allele BLUP")
        # ax3.spines["top"].set_visible(False)
        # ax3.spines["right"].set_visible(False)

            # Tight layout and save/show
        plt.tight_layout()

        if save_path:
            save_path_png = save_path.replace('.pdf', '.png')
            # save_path = save_path.replace('.pdf', '_PaperFig.pdf')
            plt.savefig(save_path)
            plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_BLUPs(Y, y_resid, Z_raw, Z_used, u_hat_locus, allele_frequencies, lr, pval, n_indiv, n_allele, comps, save_path=None):
    """
    Plot individual-level BLUPs against gene expression and residuals.
    Parameters:
        Y (pd.Series): Raw gene expression values.
        y_resid (np.ndarray): Residuals after fixed effect regression.
        Z (pd.DataFrame): Allele dosage matrix (raw).
        Z_ortho (pd.DataFrame): Allele dosage matrix orthogonalized to fixed effects.
        u_hat_locus (pd.Series): Estimated allele effects (BLUPs).
    """
    Z = Z_raw  # Z_raw has to be raw dosage matrix to correctly label genotypes
    sigma_g2, sigma_e2 = comps['sigma_g2'], comps['sigma_e2']
    var_genetic, var_residual, var_fixed = comps['var_genetic'], comps['var_residual'], comps['var_fixed']
    r2_modeled, r2_genetic, r2_residual, r2_fixed = comps['r2_modeled'], comps['r2_genetic'], comps['r2_residual'], comps['r2_fixed']
    lambda_pen, num_eigenvals = comps['lambda_pen'], len(comps['QS_full'][1])

    allele_removed = comps['allele_removed'] if 'allele_removed' in list(comps.keys()) else None
    # Create a figure with a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Calculate individual-level BLUEs
    print("Calculating individual-level BLUEs using raw Z for allele dosage...")
    individual_blues = Z_used @ u_hat_locus.values

    # print(f'Z:\n{Z}')
    # print(f'Z_ortho:\n{Z_ortho}')
    # print(f'u_hat_locus:\n{u_hat_locus}')
    print(f'Summary stats for individual BLUPs: min={individual_blues.min()}, max={individual_blues.max()}, mean={individual_blues.mean()}, std={individual_blues.std()}')

    # Subplot 1: Raw Gene Expression (Y) vs. Individual BLUEs
    ax1 = axes[0, 0]
    ax1.scatter(individual_blues, Y, color='green', alpha=0.7, label="Data")
    slope, intercept = np.polyfit(individual_blues, Y, 1)
    ax1.plot(individual_blues, slope * individual_blues + intercept, color='red', label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")
    # get unique genotypes with x and max y positions for annotation
    uniquegenotypes = {}
    for i in range(Z.shape[0]):  # for each row
        alleles = Z.columns[np.where(Z.iloc[i] > 0)[0]]  # get column names (allele number) when cell value > 0
        genotype = f"{alleles[0]}/{alleles[1]}" if len(alleles) == 2 else f"{alleles[0]}/{alleles[0]}"  # there are always just two columns in Z_raw, one allele per column
        # ax1.annotate(genotype, (individual_blues.iloc[i], Y.iloc[i]), fontsize=12, alpha=0.7, ha="center", va="bottom")
        try:
            uniquegenotypes[genotype]
        except KeyError:
            uniquegenotypes[genotype] = (individual_blues.iloc[i], Y.iloc[i])
        else:
            if Y.iloc[i] > uniquegenotypes[genotype][1]:
                uniquegenotypes[genotype] = (individual_blues.iloc[i], Y.iloc[i])
    for genotype, (xpos, ypos) in uniquegenotypes.items():
        if ypos+(0.0025*(abs(ypos))) < ax1.get_ylim()[1]:  # only annotate if within y-axis limits
            ax1.annotate(genotype, (xpos, ypos+(0.0025*(abs(ypos)))), fontsize=14, alpha=1, ha="center", va="bottom")
        else:
            ax1.annotate(genotype, (xpos, ypos-(0.0025*(abs(ypos)))), fontsize=14, alpha=1, ha="center", va="top")
    ax1.set_xlabel("Individual's BLUP: sum(Z[i,] * û)")  # û is BLUP 'predicted allele effects on gene expression effect size'
    ax1.set_ylabel("Gene Expression (Y)")
    ax1.set_title("Gene Expression vs. Individual BLUPs")
    ax1.legend()

    # Subplot 2: Residuals (y_resid) vs. Individual BLUEs
    ax2 = axes[0, 1]
    ax2.scatter(individual_blues, y_resid, color='blue', alpha=0.7, label="Residuals")
    slope_r, intercept_r = np.polyfit(individual_blues, y_resid, 1)
    ax2.plot(individual_blues, slope_r * individual_blues + intercept_r, color='red', label=f"Fit: y = {slope_r:.2f}x + {intercept_r:.2f}")
    # get unique genotypes with x and max y positions for annotation
    uniquegenotypes = {}
    for i in range(Z.shape[0]):
        alleles = Z.columns[np.where(Z.iloc[i] > 0)[0]]
        genotype = f"{alleles[0]}/{alleles[1]}" if len(alleles) == 2 else f"{alleles[0]}/{alleles[0]}"
        # ax2.annotate(genotype, (individual_blues.iloc[i], y_resid[i]), fontsize=12, alpha=0.7, ha="center", va="bottom")
        try:
            uniquegenotypes[genotype]
        except KeyError:
            uniquegenotypes[genotype] = (individual_blues.iloc[i], y_resid[i])
        else:
            if y_resid[i] > uniquegenotypes[genotype][1]:
                uniquegenotypes[genotype] = (individual_blues.iloc[i], y_resid[i])
    for genotype, (xpos, ypos) in uniquegenotypes.items():
        if ypos+(0.0025*(abs(ypos))) < ax2.get_ylim()[1]:  # only annotate if within y-axis limits
            ax2.annotate(genotype, (xpos, ypos+(0.0025*(abs(ypos)))), fontsize=14, alpha=1, ha="center", va="bottom")
        else:
            ax2.annotate(genotype, (xpos, ypos-(0.0025*(abs(ypos)))), fontsize=14, alpha=1, ha="center", va="top")
    ax2.set_xlabel("Individual's BLUP: sum(Z[i,] * û)")  # û is BLUP 'predicted allele effects on gene expression effect size'
    ax2.set_ylabel("Gene Expression Residuals (y_resid)")
    ax2.set_title("Gene Expression Residuals vs. Individual BLUPs")
    ax2.legend()

    # Subplot 3: Allele BLUEs vs. Allele Frequencies
    # Make allele_frequencies a list or Series, matching u_hat_locus index
    # num_individuals = Z.shape[0]
    # num_alleles = Z.shape[1]
    ### calc allele frequencies from Z
    ## total_alleles = Z.sum().sum()  # Total dosage across all alleles
    ## allele_freqs = pd.Series(Z.sum(axis=0) / total_alleles)
    ## allele_freqs = allele_frequencies.reindex(u_hat_locus.index)
    # Convert the supplied dictionary into a Series for easier indexing and plotting
    allele_freqs = pd.Series(allele_frequencies)
    # Reindex to match u_hat_locus order (ensures alignment)
    allele_freqs = allele_freqs.reindex(u_hat_locus.index)
    # print("Z columns:", Z.columns)
    # print("u_hat_locus index:", u_hat_locus.index)
    # print("Missing in u_hat_locus:", set(Z.columns) - set(u_hat_locus.index))
    # print("Extra in u_hat_locus:", set(u_hat_locus.index) - set(Z.columns))
    ax3 = axes[1, 0]
    for i, allele in enumerate(u_hat_locus.index):
        # freq = allele_frequencies_series[allele]
        # ax3.scatter(u_hat_locus.loc[allele], freq, color='purple', alpha=0.8)
        # ax3.annotate(f"{allele}", (u_hat_locus.loc[allele], freq), fontsize=12, alpha=0.8, ha="center", va="bottom")
        ax3.scatter(u_hat_locus.loc[allele], allele_freqs[allele], color='purple', alpha=0.8)
        if allele_freqs[allele] + 0.0025 < ax3.get_ylim()[1]:  # only annotate if within y-axis limits
            ax3.annotate(f"{allele}", (u_hat_locus.loc[allele], allele_freqs[allele] + 0.0025), fontsize=14, alpha=1.0, ha="center", va="bottom")
        else:
            ax3.annotate(f"{allele}", (u_hat_locus.loc[allele], allele_freqs[allele] - 0.0025), fontsize=14, alpha=1.0, ha="center", va="top")
    if allele_removed:
        # allele_removed effect is by definition 0
        ax3.scatter(0, allele_frequencies[allele_removed], color='red', alpha=0.8)
        ax3.annotate(f"Baseline: {allele_removed}", (0, allele_frequencies[allele_removed] + 0.0025), fontsize=14, alpha=1.0, ha="center", va="bottom")
    ax3.set_xlabel("Allele-specific BLUPs")
    ax3.set_ylabel("Allele Frequency")
    ax3.set_title("Allele Frequencies vs. Allele BLUP Effect Sizes")

    # Subplot 4: Display metadata as a table
    ax4 = axes[1, 1]
    ax4.axis('off')  # Hide axes
    table_data = [
        ["Number of Individuals", Z_used.shape[0]],
        ["Number of Alleles", Z_used.shape[1]],
        ["Number of non-zero Eigenvalues", f"{num_eigenvals}"],
        ["Mean gene expression (Y)", f"{Y.mean():.4f}"],
        ["Baseline Allele (0 effect)", allele_removed if allele_removed else 'No allele set to 0 effect'],
        ["Gen. var. comp. (v0, sigma_g²)", f"{sigma_g2:.4f}"],
        ["Resid. var. comp. (v1, sigma_e²)", f"{sigma_e2:.4f}"],
        ["h2 (heritability v0/(v0+v1))", f"{comps['h2']:.4f}"],
        ["Lambda (sigma_e²/sigma_g²)", f"{lambda_pen:.4f}"],
        ["Scale (Gen+Res variance)", f"{comps['scale']:.4f}"],
        ["Fixed Effect Variation", f"{var_fixed:.4f}"],
        ["Total variance in Y", f"{comps['var_total']:.4f}"],
        ["Genetic R²", f"{r2_genetic:.4f}"],
        ["Residual R²", f"{r2_residual:.4f}"],
        ["Fixed Effect R² (Fix)", f"{r2_fixed:.4f}"],
        ["Scale R² (Gen+Res)", f"{(comps['scale']/comps['var_total']):.4f}"],
        ["Modeled R² (Fix+Gen+Res)", f"{r2_modeled:.4f}"],
        ["Null lml, & Full lml", f"{comps['ll0']:.4f}, {comps['ll1']:.4f}"],
        ["LRT: -2*(null-full)", f"{lr:.2f}"],
        ["P-Value", f"{pval:.2e}"]
    ]
    #         ["Mean(diagonal(K)): K = ZZᵀ)", f"{comps['mean_diag_K_after']:.4f}"],

    table = ax4.table(cellText=table_data, colLabels=["Metric", "Value"], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.1)
    ax4.set_title("Model Summary", pad=20)

    # Tight layout and save/show
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def write_out_dict(outdict, outfile, individuals, pid):
    """
    Write out a dictionary to a file.
    Parameters:
        outdict (dict): Dictionary to write.
        outfile (str): Output file path.
        individuals (list): List of individual IDs.
        pid (str): PID to include in the header.
    """
    # individuals = list(outdict.keys())
    orderedindivs = '\t'.join(individuals)
    header = f"#chr	start	end	pid	gid	strand	{orderedindivs}\n"
    orderedvals = '\t'.join([str(outdict[i]) for i in individuals])
    outvalues = f"#chr	start	end	{pid}	gid	strand	{orderedvals}\n"
    with open(outfile, 'w') as OUT:
        OUT.write(f"{header}{outvalues}\n")
    return


def recreate_genotypes_from_Z(Z):
    """
    Recreate genotype strings from allele dosage matrix Z.
    Parameters:
        Z (pd.DataFrame): Allele dosage matrix (individuals x alleles).
    """
    genotypes = {}
    for i in range(Z.shape[0]):  # for each row
        alleles = Z.columns[np.where(Z.iloc[i] > 0)[0]]  # get column names (allele number) when cell value > 0
        genotype = f"{alleles[0]}/{alleles[1]}" if len(alleles) == 2 else f"{alleles[0]}/{alleles[0]}"  # there are always just two columns in Z_raw, one allele per column
        genotypes[Z.index[i]] = genotype
    return genotypes


# def plot_blup_dumbbell(df, save_path=None):
#     # Color-blind friendly palette (Okabe-Ito)
#     # color_base = '#D55E00'      # Vermillion (Warm/Orange)
#     # color_contrast = '#0072B2'  # Blue
#     # color_bar = '#E0E0E0'      # Light Grey for the bar
#     plt.figure(figsize=(6, 5))
#     
#     # Sort by absolute change to make the plot readable (optional)
#     df = df.sort_values('Base_Model', ascending=True).reset_index(drop=True)
#     
#     # Draw the lines and dots
#     for i in range(len(df)):
#         base_val = df.loc[i, 'Base_Model']
#         contrast_val = df.loc[i, 'Contrast_Model']
#         change = df.loc[i, 'Change']
#         allele = df.loc[i, 'Allele']
#         
#         # 1. Draw the horizontal line (the dumbbell bar)
#         plt.hlines(y=i, xmin=min(base_val, contrast_val), xmax=max(base_val, contrast_val), 
#                    color='grey', alpha=0.5, linewidth=2, zorder=1)
#         
#         # 2. Draw the points
#         plt.scatter(base_val, i, color='firebrick', s=100, label='Base' if i == 0 else "", zorder=2)
#         plt.scatter(contrast_val, i, color='blue', s=100, label='Contrast' if i == 0 else "", zorder=2) # seagreen
#         
#         # 3. Annotate the values
#         # Base value text
#         plt.text(base_val, i + 0.05, f"{base_val:.2f}", color='firebrick', 
#                  ha='center', fontsize=9, fontweight='bold')
#         # Contrast value text
#         plt.text(contrast_val, i + 0.05, f"{contrast_val:.2f}", color='blue', 
#                  ha='center', fontsize=9, fontweight='bold')
#         # Change (Delta) text in the middle of the bar
#         mid_point = (base_val + contrast_val) / 2
#         plt.text(mid_point, i - 0.075, f"Δ: {change:.2f}", 
#                  ha='center', fontsize=10, style='italic')
# 
#     # Formatting
#     plt.yticks(range(len(df)), df['Allele'])
#     plt.axvline(0, color='black', linestyle='--', alpha=0.3) # Reference line at 0
#     plt.title("Shrinkage of Allele BLUPs: Base vs. Contrast Model", loc='left', fontsize=14)
#     plt.xlabel("BLUP Value (u_hat)")
#     plt.ylabel("Allele ID")
#     plt.legend()  # loc='upper right'
#     plt.grid(axis='x', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     return

# def BLUP_vs_BLUP_dumbell(df, save_path=None):
#     plt.figure(figsize=(7, 6))
#     # Sort for readability
#     df = df.sort_values('Base_Model', ascending=True).reset_index(drop=True)
#     # Detect change columns
#     change_cols = [c for c in df.columns if c.startswith('Change')]
#     # Define preferred order
#     preferred_order = ['ChangeMLBV', 'ChangeBV', 'ChangeML']
#     change_cols = [c for c in preferred_order if c in change_cols] + \
#                   [c for c in change_cols if c not in preferred_order]
#     # Map change → corresponding model column
#     model_map = {
#         'ChangeMLBV': 'MeanLength+BestVariant',
#         'ChangeBV': 'BestVariant',
#         'ChangeML': 'MeanLength',
#         'Change': 'Contrast_Model'
#     }
#     # Color palette (light, dark)
#     color_map = {
#         'ChangeMLBV': ('#CAB2D6', '#6A3D9A'),  # light/dark purple
#         'ChangeBV': ('#A6CEE3', '#1F78B4'),    # light/dark blue
#         'ChangeML': ('#B2DF8A', '#33A02C'),    # light/dark green
#         'Change': ('#FDBF6F', '#FF7F00')       # fallback (orange)
#     }
#     # Vertical offsets if multiple contrasts
#     n_changes = len(change_cols)
#     offsets = np.linspace(-0.2, 0.2, n_changes) if n_changes > 1 else [0]
# 
#     for i in range(len(df)):
#         base_val = df.loc[i, 'Base_Model']
#         allele = df.loc[i, 'Allele']
#         for j, change_col in enumerate(change_cols):
#             contrast_col = model_map.get(change_col, 'Contrast_Model')
#             contrast_val = df.loc[i, contrast_col]
#             change = df.loc[i, change_col]
#             y = i + offsets[j]
#             light_color, dark_color = color_map.get(change_col, ('#CCCCCC', '#333333'))
#             # Line
#             plt.hlines(
#                 y=y,
#                 xmin=min(base_val, contrast_val),
#                 xmax=max(base_val, contrast_val),
#                 color='grey',
#                 alpha=0.4,
#                 linewidth=2,
#                 zorder=1
#             )
#             # Points
#             plt.scatter(base_val, y, color=light_color, s=80,
#                         label=f"{change_col} (Base)" if (i == 0) else "", zorder=2)
#             plt.scatter(contrast_val, y, color=dark_color, s=80,
#                         label=f"{change_col} (Contrast)" if (i == 0) else "", zorder=2)
#             # Annotations
#             plt.text(base_val, y + 0.03, f"{base_val:.2f}",
#                      color=light_color, ha='center', fontsize=8)
#             plt.text(contrast_val, y + 0.03, f"{contrast_val:.2f}",
#                      color=dark_color, ha='center', fontsize=8)
#             mid = (base_val + contrast_val) / 2
#             plt.text(mid, y - 0.05, f"Δ {change:.2f}",
#                      ha='center', fontsize=9, style='italic')
#     # Formatting
#     plt.yticks(range(len(df)), df['Allele'])
#     plt.axvline(0, color='black', linestyle='--', alpha=0.3)
#     title = "BLUP Shrinkage: Base vs Multiple Contrast Models" if n_changes > 1 \
#             else "BLUP Shrinkage: Base vs Contrast Model"
#     plt.title(title, loc='left', fontsize=14)
#     plt.xlabel("BLUP Value (u_hat)")
#     plt.ylabel("Allele ID")
#     plt.grid(axis='x', alpha=0.3)
#     plt.legend(fontsize=8)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     plt.close()
# 
#     return

def BLUP_vs_BLUP_dumbell(df, save_path=None):
    plt.figure(figsize=(7, 6))  
    # Sort
    df = df.sort_values('Base_Model', ascending=True).reset_index(drop=True)    
    # Detect change columns
    change_cols = [c for c in df.columns if c.startswith('Change')] 
    preferred_order = ['ChangeMLBV', 'ChangeBV', 'ChangeML']
    change_cols = [c for c in preferred_order if c in change_cols] + \
                  [c for c in change_cols if c not in preferred_order]  
    model_map = {
        'ChangeMLBV': 'MeanLength+BestVariant',
        'ChangeBV': 'BestVariant',
        'ChangeML': 'MeanLength',
        'Change': 'Contrast_Model'
    }   
    # Contrast colors (ONLY dark used now)
    color_map = {
        'ChangeMLBV': '#6A3D9A',  # purple
        'ChangeBV': '#1F78B4',    # blue
        'ChangeML': '#33A02C',    # green
        'Change': '#FF7F00'
    }   
    # Base color (shared)
    base_color = '#D55E00'  # dark orange (Okabe-Ito)   
    # Offsets
    n_changes = len(change_cols)
    offsets = np.linspace(-0.2, 0.2, n_changes) if n_changes > 1 else [0]   
    # ---- Plot ----
    for i in range(len(df)):
        base_val = df.loc[i, 'Base_Model']
        for j, change_col in enumerate(change_cols):
            contrast_col = model_map.get(change_col, 'Contrast_Model')
            contrast_val = df.loc[i, contrast_col]
            change = df.loc[i, change_col]  
            y = i + offsets[j]
            color = color_map.get(change_col, '#333333')    
            # Line
            plt.hlines(
                y=y,
                xmin=min(base_val, contrast_val),
                xmax=max(base_val, contrast_val),
                color='grey',
                alpha=0.4,
                linewidth=2,
                zorder=1
            )
            # Draw j times per allele
            plt.scatter(base_val, y, color=base_color, s=80,
                        label='Base (NoContrast)' if i == 0 else "",
                        zorder=3)   
            plt.text(base_val, y + 0.05, f"{base_val:.2f}",
                     color=base_color, ha='center', fontsize=8)    
            # Contrast point
            label_clean = change_col.replace('Change', '') + " (Contrast)"
            plt.scatter(contrast_val, y, color=color, s=80,
                        label=label_clean if i == 0 else "",
                        zorder=2)   
            # Contrast annotation
            plt.text(contrast_val, y + 0.05, f"{contrast_val:.2f}",
                     color=color, ha='center', fontsize=8, fontweight='bold')  
            # Delta
            mid = (base_val + contrast_val) / 2
            plt.text(mid, y - 0.07, f"Δ {change:.2f}",
                     ha='center', fontsize=9, style='italic')   
    # ---- Formatting ----
    plt.yticks(range(len(df)), df['Allele'])
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)    
    title = "BLUP Shrinkage: Base vs Multiple Contrast Models" if n_changes > 1 \
            else "BLUP Shrinkage: Base vs Contrast Model"   
    plt.title(title, loc='left', fontsize=14)
    plt.xlabel("BLUP Value (u_hat)")
    plt.ylabel("Allele ID") 
    plt.grid(axis='x', alpha=0.3)   
    # Clean legend (removes duplicates automatically)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=9)  
    plt.tight_layout()  
    if save_path:
        plt.savefig(save_path)  
    plt.close()

    return


def BLUP_vs_BLUP_scatter_bar(Z, blup_base, blup_contrast, outfile):
    # Create a figure with two subplots (1 row, 2 columns)
    # width_ratios=[3, 1] makes the first plot 3x wider than the second
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [3, 1]})
    # --- AX1: SCATTER PLOT ---
    ax1.axline((0, 0), slope=1, color='r', linestyle='--', label='Slope 1')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.axvline(0, color='black', linewidth=1)
    ax1.scatter(blup_base['u_hat'], blup_contrast['u_hat'])
    # Annotate points
    for i, (xb, yb) in enumerate(zip(blup_base['u_hat'], blup_contrast['u_hat'])):
        ax1.annotate(Z.columns[i], (xb, yb), weight='bold', fontsize=14)
    # Scaling
    all_vals = np.concatenate([[0], blup_base['u_hat'].values, blup_contrast['u_hat'].values])
    minlim, maxlim = all_vals.min(), all_vals.max()
    pad = 0.1 * (maxlim - minlim)
    ax1.set_xlim(minlim - pad, maxlim + pad)
    ax1.set_ylim(minlim - pad, maxlim + pad)
    ax1.set_xlabel(f'BLUPs from Base Model')
    ax1.set_ylabel(f'BLUPs from Contrast Model')
    ax1.set_title('BLUPs: Base Model vs Contrast Model with MAL/BV in X')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=':')
    # --- AX2: PROPORTIONAL CHANGE BAR PLOT ---
    # Calculate % change: (New - Old) / Old. Note: avoid division by zero if base is 0.
    prop_change = (blup_contrast['u_hat'] - blup_base['u_hat']) / blup_base['u_hat'].replace(0, np.nan)  # Replace 0 with a small number to avoid division by zero
    ax2.barh(Z.columns, prop_change, color='skyblue')
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Proportional Change')
    ax2.set_ylabel('Allele')
    ax2.set_title('$\Delta$ BLUP')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(outfile)

    return


def BLUP_vs_BLUP_scatter(Z, blup_base, blup_contrast, Z_best_var, vntr_mean_length_values, outfile, allele_removed=''):
    # make scatter plot with correlation and p-value of BLUPs from base model vs BLUPs from model with mean allele length and best variant genotypes in X_mat
    plt.figure(figsize=(6, 5))
    # plot line with slope 1 and intercept 0 for reference
    plt.axline((0, 0), slope=1, color='r', linestyle='--', label='Line w/ slope 1 and intercept 0')
    # plt.plot([blup_base['u_hat'].min(), blup_base['u_hat'].max()], [blup_base['u_hat'].min(), blup_base['u_hat'].max()], color='red', linestyle='--')
    # Horizontal line at y=0 (Slope 0)
    plt.axhline(0, color='black', linewidth=1)
    # Vertical line at x=0 (Opposite: Undefined slope)
    # Annotate the 4 quadrants
    plt.annotate('(+,+)', xy=(1, 1), xycoords='axes fraction', fontsize=14, ha='right', va='top')   # Quadrant I
    plt.annotate('(-,+)', xy=(0, 1), xycoords='axes fraction', fontsize=14, ha='left', va='top')  # Quadrant II
    plt.annotate('(-,-)', xy=(0, 0), xycoords='axes fraction', fontsize=14, ha='left', va='bottom') # Quadrant III
    plt.annotate('(+,-)', xy=(1, 0), xycoords='axes fraction', fontsize=14, ha='right', va='bottom')  # Quadrant IV
    plt.axvline(0, color='black', linewidth=1)
    plt.scatter(blup_base['u_hat'], blup_contrast['u_hat'])
    if allele_removed:
        blup_base_vals = np.append(blup_base['u_hat'].values, 0)
        blup_contrast_vals = np.append(blup_contrast['u_hat'].values, 0)
        plt.scatter(0,0, color='red')
        plt.annotate(str(allele_removed), (0,0), color='red', fontsize=14)
    else:
        blup_base_vals = blup_base['u_hat'].values
        blup_contrast_vals = blup_contrast['u_hat'].values
    corr_blups, pval_blups = pearsonr(blup_base_vals, blup_contrast_vals)
    if vntr_mean_length_values is not None and Z_best_var is None:
        plt.title(f'BLUPs w/o MAL in X_mat vs BLUPs with MAL in X_mat')
        plt.xlabel('BLUPs without mean allele length in X_mat')
        plt.ylabel('BLUPs with mean allele length in X_mat')
        plt.annotate(f"r={corr_blups:.4f}\npval={pval_blups:.4e}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
        # annotate allele numbers (column names of Z) for each point in scatter plot
        for i, (x, y) in enumerate(zip(blup_base['u_hat'], blup_contrast['u_hat'])):
            plt.annotate(Z.columns[i], (x, y), fontsize=14)
    elif Z_best_var is not None and vntr_mean_length_values is None:
        plt.title(f'BLUPs w/o BV in X_mat vs BLUPs with BV in X_mat')
        plt.xlabel('BLUPs without best variant genotypes in X_mat')
        plt.ylabel('BLUPs with best variant genotypes in X_mat')
        plt.annotate(f"r={corr_blups:.4f}\npval={pval_blups:.4e}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
        # annotate allele numbers (column names of Z) for each point in scatter plot
        for i, (x, y) in enumerate(zip(blup_base['u_hat'], blup_contrast['u_hat'])):
            plt.annotate(Z.columns[i], (x, y), fontsize=14)
    else:
        plt.title(f'BLUPs w/o MAL or BV in X_mat vs BLUPs with MAL and BV in X_mat')
        plt.xlabel('BLUPs without mean allele length or best variant genotypes in X_mat')
        plt.ylabel('BLUPs with mean allele length and best variant genotypes in X_mat')
        plt.annotate(f"r={corr_blups:.4f}\npval={pval_blups:.4e}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
        # annotate allele numbers (column names of Z) for each point in scatter plot
        for i, (x, y) in enumerate(zip(blup_base['u_hat'], blup_contrast['u_hat'])):
            plt.annotate(Z.columns[i], (x, y), fontsize=14)
    
    # plt.axis('square')
    minlim = min(blup_base_vals.min(), blup_contrast_vals.min())
    maxlim = max(blup_base_vals.max(), blup_contrast_vals.max())
    # minlim = min(blup_base['u_hat'].min(), blup_Zbestvar['u_hat'].min() if Z_best_var is not None else blup_Zlen['u_hat'].min())
    # maxlim = max(blup_base['u_hat'].max(), blup_Zbestvar['u_hat'].max() if Z_best_var is not None else blup_Zlen['u_hat'].max())
    # plt.ylim(blup_base['u_hat'].min() - 0.01*(blup_base['u_hat'].max() - blup_base['u_hat'].min()), blup_base['u_hat'].max() + 0.01*(blup_base['u_hat'].max() - blup_base['u_hat'].min()))
    # plt.xlim(blup_base['u_hat'].min() - 0.01*(blup_base['u_hat'].max() - blup_base['u_hat'].min()), blup_base['u_hat'].max() + 0.01*(blup_base['u_hat'].max() - blup_base['u_hat'].min()))
    # plt.xlim(minlim - 0.01*(maxlim - minlim), maxlim + 0.01*(maxlim - minlim))
    # plt.ylim(minlim - 0.01*(maxlim - minlim), maxlim + 0.01*(maxlim - minlim))
    range_val = maxlim - minlim
    if range_val == 0:
        center = minlim
        plt.xlim(center - 0.1, center + 0.1)
        plt.ylim(center - 0.1, center + 0.1)
    else:
        padding = 0.05 * range_val
        plt.xlim(minlim - padding, maxlim + padding)
        plt.ylim(minlim - padding, maxlim + padding)
    plt.grid(True)
    plt.savefig(outfile)
    plt.close()

    return


def plot_bv_ml_triptych(genotypes, vntr_genotypes, X_mat, Y, outfile, jitter_strength=0.3):
    """
    Creates a 3-panel figure:
    1) BV + ML (violin + scatter)
    2) BV vs Gene Expression
    3) ML vs Gene Expression

    Saves to a single combined PDF.
    """

    # --- Normalize sizes ---
    y_sizes = ((Y - Y.min()) / (Y.max() - Y.min()) * 100).values

    # --- Jitter ---
    jitter_genotype = genotypes + np.random.uniform(
        -jitter_strength, jitter_strength, size=genotypes.shape
    )
    jitter_allele_length = X_mat['vntr_mean_allele_length'] + np.random.uniform(
        -jitter_strength, jitter_strength,
        size=X_mat['vntr_mean_allele_length'].shape
    )

    # --- Create figure ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # =========================
    # 1. BV + ML (Violin + Scatter)
    # =========================
    ax = axes[0]

    sns.violinplot(
        x=genotypes,
        y=X_mat['vntr_mean_allele_length'],
        inner=None,
        color=".8",
        ax=ax
    )

    sns.scatterplot(
        x=jitter_genotype,
        y=jitter_allele_length,
        hue=vntr_genotypes,
        size=y_sizes,
        sizes=(20, 150),
        legend='brief',
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )

    ax.set_title("Best Biallelic Variant VS Mean Allele Length")
    ax.set_xlabel('Genotype of Best Variant')
    ax.set_ylabel('Mean VNTR Allele Length')
    ax.grid(True)

    # =========================
    # 2. BV vs Gene Expression
    # =========================
    ax = axes[1]

    sns.scatterplot(
        x=jitter_genotype,
        y=Y,
        hue=vntr_genotypes,
        size=y_sizes,
        sizes=(20, 150),
        legend='brief',
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )

    ax.set_title("Best Biallelic Variant VS Gene Expression")
    ax.set_xlabel('Genotype of Best Variant')
    ax.set_ylabel('Gene Expression (raw)')
    ax.grid(True)

    # =========================
    # 3. ML vs Gene Expression
    # =========================
    ax = axes[2]

    sns.scatterplot(
        x=jitter_allele_length,
        y=Y,
        hue=vntr_genotypes,
        size=y_sizes,
        sizes=(20, 150),
        legend='brief',
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )

    ax.set_title("VNTR Mean Allele Length VS Gene Expression")
    ax.set_xlabel('Mean Allele Length')
    ax.set_ylabel('Gene expression (raw)')
    ax.grid(True)

    # --- Shared legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="VNTR Genotypes\nPercent GeneExpr",
        fontsize=7,
        bbox_to_anchor=(1.01, 0.9),
        loc='upper left'
    )

    # Remove per-axis legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # --- Layout + save ---
    plt.tight_layout()
    plt.savefig(
        outfile.replace('.pdf', '.BV+ML_BV_ML_BLUPvsExpr.triptych.pdf'),
        bbox_inches='tight'
    )
    plt.close()

    return


def plot_bv_ml_quadriptych(genotypes, vntr_genotypes, X_mat, Y, Z, blups, outfile, jitter_strength: float=0.1, font_size: int=10, figsize: tuple=(7, 3.5), font_family: str="Arial"):
    """
    Creates a 4-panel figure:
    1) BV + ML (violin + scatter)
    2) BV vs Gene Expression
    3) ML vs Gene Expression
    4) BLUPs from base model (no BV/ML in X) vs Gene Expression

    Saves to a single combined PDF.
    """

    # --- Normalize sizes ---
    y_sizes = ((Y - Y.min()) / (Y.max() - Y.min()) * 100).values

    # --- Jitter ---
    jitter_genotype = genotypes + np.random.uniform(
        -jitter_strength, jitter_strength, size=genotypes.shape
    )
    jitter_allele_length = X_mat['vntr_mean_allele_length'] + np.random.uniform(
        -jitter_strength, jitter_strength,
        size=X_mat['vntr_mean_allele_length'].shape
    )

    # --- Create figure ---
    fig, axes = plt.subplots(1, 4, figsize=figsize)  # (16, 5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2
    })

    # =========================
    # 2. BV vs Gene Expression
    # =========================
    ax = axes[0]

    sns.scatterplot(
        x=jitter_genotype,
        y=Y,
        hue=vntr_genotypes,
        size=y_sizes,
        sizes=(20, 150),
        legend='brief',
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )

    sns.regplot(
        x=jitter_genotype,
        y=Y,
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7}
    )
    # ax.set_title("Best Biallelic Variant VS Gene Expression")
    ax.set_xlabel('Genotype of Best Variant')
    ax.set_ylabel('Gene Expression (raw)')
    ax.grid(True)

    # =========================
    # 3. ML vs Gene Expression
    # =========================
    ax = axes[1]

    sns.scatterplot(
        x=jitter_allele_length,
        y=Y,
        hue=vntr_genotypes,
        size=y_sizes,
        sizes=(20, 150),
        legend='brief',
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )

    sns.regplot(
        x=jitter_allele_length,
        y=Y,
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7}
    )

    # ax.set_title("VNTR Mean Allele Length VS Gene Expression")
    ax.set_xlabel('Mean Allele Length')
    ax.set_ylabel('Gene expression (raw)')
    ax.grid(True)

    # =========================
    # 1. BV + ML (Violin + Scatter)
    # =========================
    ax = axes[2]

    sns.violinplot(
        x=genotypes,
        y=X_mat['vntr_mean_allele_length'],
        inner=None,
        color=".8",
        ax=ax
    )

    sns.scatterplot(
        x=jitter_genotype,
        y=jitter_allele_length,
        hue=vntr_genotypes,
        size=y_sizes,
        sizes=(20, 150),
        legend='brief',
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )

    # ax.set_title("Best Biallelic Variant VS Mean Allele Length")
    ax.set_xlabel('Genotype of Best Variant')
    ax.set_ylabel('Mean VNTR Allele Length')
    ax.grid(True)

    # =========================
    # 4. BLUPs from base model (no BV/ML in X)
    # =========================
    ax = axes[3]

    individual_blues = Z @ blups.values
    sns.scatterplot(
        x=individual_blues,
        y=Y,
        hue=vntr_genotypes,
        size=y_sizes,
        sizes=(20, 150),
        legend='brief',
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )

    sns.regplot(
        x=individual_blues,
        y=Y,
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7}
    )

    # ax.set_title("Individual BLUPs VS Gene Expression")
    ax.set_xlabel('Individual BLUPs')
    ax.set_ylabel('Gene expression (raw)')
    ax.grid(True)

    # --- Shared legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:-6], labels[:-6],
        title="VNTR\nGenotypes",
        bbox_to_anchor=(1.01, 0.9),
        loc='upper left'
    )
    # fontsize=8,
    # title="VNTR Genotypes\nPercent GeneExpr"

    # Remove per-axis legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # --- Layout + save ---
    plt.tight_layout()
    plt.savefig(
        outfile.replace('.pdf', '.BV+ML_BV_ML_BLUPvsExpr.quadriptych.pdf'),
        bbox_inches='tight'
    )
    plt.close()

    return

def plot_bv_ml_VS_ge_triptych(genotypes, vntr_genotypes, X_mat, Y, Z, blups, outfile, gene_id, vntr_id, jitter_strength: float=0.1, font_size: int=10, figsize: tuple=(7, 3.5), font_family: str="Arial"):
    """
    Creates a 3-panel figure:
    1) BV vs Gene Expression
    2) ML vs Gene Expression
    3) BLUPs from base model (no BV/ML in X) vs Gene Expression

    Saves to a single combined PDF.
    """

    # --- Normalize sizes ---
    y_sizes = ((Y - Y.min()) / (Y.max() - Y.min()) * 100).values

    # --- Jitter ---
    jitter_genotype = genotypes + np.random.uniform(
        -jitter_strength, jitter_strength, size=genotypes.shape
    )
    jitter_allele_length = X_mat['vntr_mean_allele_length'] + np.random.uniform(
        -jitter_strength, jitter_strength,
        size=X_mat['vntr_mean_allele_length'].shape
    )

    # --- Create figure ---
    fig, axes = plt.subplots(1, 3, figsize=figsize)  # (16, 5)
    fig.suptitle(f"Gene: {gene_id} | VNTR: {vntr_id}")
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2
    })

    # =========================
    # 2. BV vs Gene Expression
    # =========================
    ax = axes[0]

    sns.scatterplot(
        x=jitter_genotype,
        y=Y,
        hue=vntr_genotypes,
        size=y_sizes,
        sizes=(20, 150),
        legend='brief',
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )

    sns.regplot(
        x=jitter_genotype,
        y=Y,
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7}
    )
    # ax.set_title("Best Biallelic Variant VS Gene Expression")
    ax.set_xlabel('Genotype of Best Variant')
    ax.set_ylabel('Gene Expression (raw)')
    ax.set_ylim(Y.min() - 0.05*(Y.max() - Y.min()), Y.max() + 0.05*(Y.max() - Y.min()))
    ax.grid(True)

    # =========================
    # 3. ML vs Gene Expression
    # =========================
    ax = axes[1]

    sns.scatterplot(
        x=jitter_allele_length,
        y=Y,
        hue=vntr_genotypes,
        size=y_sizes,
        sizes=(20, 150),
        legend='brief',
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )

    sns.regplot(
        x=jitter_allele_length,
        y=Y,
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7}
    )

    # ax.set_title("VNTR Mean Allele Length VS Gene Expression")
    ax.set_xlabel('Mean Allele Length')
    # ax.set_ylabel('Gene expression (raw)')
    ax.set_ylabel('')
    ax.set_ylim(Y.min() - 0.05*(Y.max() - Y.min()), Y.max() + 0.05*(Y.max() - Y.min()))
    # ax.get_yaxis().set_visible(False)
    ax.grid(True)


    # =========================
    # 4. BLUPs from base model (no BV/ML in X) vs Gene Expression
    # =========================
    ax = axes[2]

    individual_blues = Z @ blups.values
    sns.scatterplot(
        x=individual_blues,
        y=Y,
        hue=vntr_genotypes,
        size=y_sizes,
        sizes=(20, 150),
        legend='brief',
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )

    sns.regplot(
        x=individual_blues,
        y=Y,
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7}
    )

    # ax.set_title("Individual BLUPs VS Gene Expression")
    ax.set_xlabel('Individual BLUPs')
    # ax.set_ylabel('Gene expression (raw)')
    ax.set_ylabel('')
    ax.set_ylim(Y.min() - 0.05*(Y.max() - Y.min()), Y.max() + 0.05*(Y.max() - Y.min()))
    # ax.get_yaxis().set_visible(False)
    ax.grid(True)

    # --- Shared legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:-6], labels[:-6],
        title="VNTR\nGenotypes",
        bbox_to_anchor=(1.01, 0.9),
        loc='upper left'
    )
    # fontsize=8,
    # title="VNTR Genotypes\nPercent GeneExpr"

    # Remove per-axis legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # --- Layout + save ---
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)  # Adjust top placement of subplots
    plt.savefig(
        outfile.replace('.pdf', '.BV+ML_BV_ML_BLUPvsExpr.triptychVSge.pdf'),
        bbox_inches='tight'
    )
    plt.close()

    return


def plot_bv_ml_blup_bluplength_VS_ge_quadriptych(genotypes, vntr_genotypes, vntr_genotypes_length, X_mat, X_mat_length, Y, Y_length, Z, Z_length, blups, blups_length, outfile, gene_id, vntr_id, best_variant_ID, best_vntr_length_ID, vntr_mean_length_ID, jitter_strength: float=0.1, font_size: int=10, figsize: tuple=(7, 3.5), font_family: str="Arial", LengthModelNullBetas: bool=False):
    """
    Creates a 4-panel figure:
    1) MAL vs Gene Expression
    3) BLUPs from base model (no BV/ML in X) VNTR alleles genotyped only by length vs Gene Expression
    4) BLUPs from base model (no BV/ML in X) VNTR alleles genotyped by length and sequencevs Gene Expression
    4) BV vs Gene Expression

    Saves to a single combined PDF.
    """

    # --- Normalize sizes ---
    y_sizes = ((Y - Y.min()) / (Y.max() - Y.min()) * 100).values

    # --- Jitter ---
    jitter_genotype = genotypes + np.random.uniform(
        -jitter_strength, jitter_strength, size=genotypes.shape
    )
    jitter_allele_length = X_mat['vntr_mean_allele_length'] + np.random.uniform(
        -jitter_strength, jitter_strength,
        size=X_mat['vntr_mean_allele_length'].shape
    )

    # --- Create figure ---
    fig, axes = plt.subplots(1, 4, figsize=figsize)  # (16, 5)
    # fig.suptitle(f"Gene: {gene_id} | VNTR: {vntr_id}")
    fig.suptitle(f"Gene: {gene_id}")
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2
    })




    # =========================
    # 1. ML vs Gene Expression
    # =========================
    ax = axes[0]

    sns.scatterplot(
        x=jitter_allele_length,
        y=Y,
        hue=vntr_genotypes,
        legend='brief',
        size=y_sizes,
        sizes=(20, 150),
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )
    #    size=y_sizes,
    #    sizes=(20, 150),

    sns.regplot(
        x=jitter_allele_length,
        y=Y,
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7}
    )

    ax.set_title(f"{vntr_mean_length_ID}", fontsize=10)# "VNTR Mean Allele Length VS Gene Expression")
    ax.set_xlabel('Mean Allele Length')
    ax.set_ylabel('Gene expression (raw)')
    # ax.set_ylabel('')
    ax.set_ylim(Y.min() - 0.05*(Y.max() - Y.min()), Y.max() + 0.05*(Y.max() - Y.min()))
    # ax.get_yaxis().set_visible(False)
    ax.grid(True)

    # =========================
    # 2. BLUPs from base model (no BV/ML in X) but estimated for vntr allele length NOT length and sequence vs Gene Expression
    # =========================
    ax = axes[1]

    individual_blues = Z_length @ blups_length.values
    temp = '\n'.join(map(str, individual_blues))
    print(f"Individual BLUPs for VNTR allele length only model: {temp}")
    print(f"Lengths of {len(individual_blues)}, {len(Y_length)}, {len(vntr_genotypes_length)}, {len(vntr_genotypes)}")
    sns.scatterplot(
        x=individual_blues,
        y=Y,
        hue=vntr_genotypes,
        legend='brief',
        size=y_sizes,
        sizes=(20, 150),
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )
    # s = 75
    #    size=y_sizes,
    #    sizes=(20, 150),
    # vntr_genotypes_length
    sns.regplot(
        x=individual_blues,
        y=Y,
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7}
    )

    ax.set_title(f"{best_vntr_length_ID}", fontsize=10)
    ax.set_xlabel('BLUPs (Length)')
    # ax.set_ylabel('Gene expression (raw)')
    ax.set_ylabel('')
    ax.set_ylim(Y.min() - 0.05*(Y.max() - Y.min()), Y.max() + 0.05*(Y.max() - Y.min()))
    # ax.get_yaxis().set_visible(False)
    ax.grid(True)

    # =========================
    # 3. BLUPs from base model (no BV/ML in X) blups for vntr allele length and sequence vs Gene Expression
    # =========================
    ax = axes[2]

    individual_blues = Z @ blups.values
    temp = '\n'.join(map(str, individual_blues))
    print(f"Individual BLUPs for VNTR allele length and sequence model: {temp}")
    sns.scatterplot(
        x=individual_blues,
        y=Y,
        hue=vntr_genotypes,
        legend='brief',
        size=y_sizes,
        sizes=(20, 150),
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )
    # s = 75
    #    size=y_sizes,
    #    sizes=(20, 150),

    sns.regplot(
        x=individual_blues,
        y=Y,
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7}
    )

    ax.set_title(f"{vntr_id}", fontsize=10)# "Individual BLUPs VS Gene Expression")
    ax.set_xlabel('BLUPs (Length & Sequence)')
    # ax.set_ylabel('Gene expression (raw)')
    ax.set_ylabel('')
    ax.set_ylim(Y.min() - 0.05*(Y.max() - Y.min()), Y.max() + 0.05*(Y.max() - Y.min()))
    # ax.get_yaxis().set_visible(False)
    ax.grid(True)

    # =========================
    # 4. BV vs Gene Expression
    # =========================
    ax = axes[3]

    sns.scatterplot(
        x=jitter_genotype,
        y=Y,
        hue=vntr_genotypes,
        legend='brief',
        size=y_sizes,
        sizes=(20, 150),
        alpha=0.5,
        edgecolor='black',
        ax=ax
    )
    #    size=y_sizes,
    #    sizes=(20, 150),

    sns.regplot(
        x=jitter_genotype,
        y=Y,
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7}
    )
    ax.set_title(f"{best_variant_ID}", fontsize=10)
    ax.set_xlabel('Genotype of Best Variant')
    # ax.set_ylabel('Gene Expression (raw)')
    ax.set_ylabel('')
    ax.set_ylim(Y.min() - 0.05*(Y.max() - Y.min()), Y.max() + 0.05*(Y.max() - Y.min()))
    ax.grid(True)

    ######

    # --- Shared legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:-6], labels[:-6],
        title="VNTR\nGenotypes",
        bbox_to_anchor=(1.01, 0.9),
        loc='upper left'
    )
    # fontsize=8,
    # title="VNTR Genotypes\nPercent GeneExpr"

    # Remove per-axis legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # --- Layout + save ---
    plt.tight_layout()
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(top=0.87)  # Adjust top placement of subplots
    if LengthModelNullBetas:
        plt.savefig(
            outfile.replace('.pdf', '.BV_ML_BLUPLen_BLUPLenSeqVSExpr.quadriptych.LengthModelNullBetas.pdf'),
            bbox_inches='tight'
        )
    else:
        plt.savefig(
            outfile.replace('.pdf', '.BV_ML_BLUPLen_BLUPLenSeqVSExpr.quadriptych.pdf'),
            bbox_inches='tight'
        )
    plt.close()

    return


def plot_regression(gene_id, vntr_id, expression_matrix, vntr_matrix, covariates, allcovariates, base_common_inds, kinship_matrix, vntr_mean_length_matrix, best_variant_matrix, vntr_length_matrix, best_vntr_length_matrix, best_vntr_mean_length_matrix, bestvarfile, workingdir, minsamples, xaxislabel, no_rare_alleles=False, collapse_rare_alleles=False, reml=False, permute_vntr_mean_lengths=False):
    
    outfilename = f'Regression.Gene.{gene_id}.VS.VNTR.{vntr_id}.pdf' if no_rare_alleles == False else f'Regression.Gene.{gene_id}.VS.VNTR.{vntr_id}.NoRareAlleles.pdf'
    outfilename = outfilename if collapse_rare_alleles == False else f'Regression.Gene.{gene_id}.VS.VNTR.{vntr_id}.CollapseRareAlleles.pdf'
    outdir = os.path.join(workingdir, 'plots')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, outfilename)

    # number of starting common individuals between all input matrices, use this for allele frequency filtration in prep_data -> make_allele_dosage_matrix
    n_common_inds = len(base_common_inds)
    print(f"Gene expression for {gene_id}:")
    print(expression_matrix.loc[gene_id, expression_matrix.columns[6:]])
    print(f"VNTR values for {vntr_id}:")
    print(vntr_matrix.loc[vntr_id, vntr_matrix.columns[6:]])
    if vntr_mean_length_matrix is not None or best_vntr_mean_length_matrix is not None:
        print(f"VNTR mean allele length values for {vntr_id}:")
        if permute_vntr_mean_lengths:
            outfile = f"{outfile[:-len('.pdf')]}.PermuteMeanAlleleLength.pdf"
        elif best_vntr_mean_length_matrix is not None:
            print(best_vntr_mean_length_matrix.loc[gene_id, best_vntr_mean_length_matrix.columns[6:]])
            outfile = f"{outfile[:-len('.pdf')]}.BestMeanAlleleLength.pdf"
        else:
            print(vntr_mean_length_matrix.loc[vntr_id, vntr_mean_length_matrix.columns[6:]])
            outfile = f"{outfile[:-len('.pdf')]}.MeanAlleleLength.pdf"
    if best_variant_matrix is not None:
        print(f"Best SNV/indel genotype values for {gene_id}:")
        print(best_variant_matrix.loc[gene_id, best_variant_matrix.columns[6:]])
        outfile = f"{outfile[:-len('.pdf')]}.{os.path.basename(bestvarfile).split('.')[0]}.pdf"  # BestVariant

    gene_expression = expression_matrix.loc[gene_id, expression_matrix.columns[6:]].dropna().astype(float)
    gene_expression = remove_sd_outliers(gene_expression, threshold=5)  # remove extreme outliers
    vntr_values = vntr_matrix.loc[vntr_id, vntr_matrix.columns[6:]].dropna().astype(str)
    common_inds = gene_expression.index.intersection(vntr_values.index)
    best_variants = best_variant_matrix.loc[gene_id, best_variant_matrix.columns[6:]].dropna().astype(str) if best_variant_matrix is not None else None
    best_variant_ID = best_variant_matrix.loc[gene_id, 'pid'] if best_variant_matrix is not None else None
    print(f"Best variant ID for {gene_id}: {best_variant_ID}") if best_variant_ID is not None else None
    vntr_length = vntr_length_matrix.loc[vntr_id, vntr_length_matrix.columns[6:]].dropna().astype(str) if vntr_length_matrix is not None else None
    vntr_length_ID = vntr_length_matrix.loc[vntr_id, 'pid'] if vntr_length_matrix is not None else None
    best_vntr_length = best_vntr_length_matrix.loc[gene_id, best_vntr_length_matrix.columns[6:]].dropna().astype(str) if best_vntr_length_matrix is not None else None
    best_vntr_length_ID = best_vntr_length_matrix.loc[gene_id, 'pid'] if best_vntr_length_matrix is not None else None
    if best_vntr_length is not None:
        vntr_length = best_vntr_length
        vntr_length_matrix = best_vntr_length_matrix
        vntr_length_ID = best_vntr_length_ID
    print(f"Best VNTR length variant ID for {gene_id}: {best_vntr_length_ID}") if best_vntr_length_ID is not None else None
    # vntr_mean_length_values = vntr_mean_length_matrix.loc[vntr_id, vntr_mean_length_matrix.columns[6:]].dropna().astype(float) if vntr_mean_length_matrix is not None else None
    vntr_mean_length_values = vntr_mean_length_matrix.loc[vntr_id, vntr_mean_length_matrix.columns[6:]].loc[lambda x: (x != '.') & x.notna()].astype(float) if vntr_mean_length_matrix is not None else None
    vntr_mean_length_ID = vntr_mean_length_matrix.loc[vntr_id, 'pid'] if vntr_mean_length_matrix is not None else None
    best_vntr_mean_length_values = best_vntr_mean_length_matrix.loc[gene_id, best_vntr_mean_length_matrix.columns[6:]].loc[lambda x: (x != '.') & x.notna()].astype(float) if best_vntr_mean_length_matrix is not None else None
    best_vntr_mean_length_ID = best_vntr_mean_length_matrix.loc[gene_id, 'pid'] if best_vntr_mean_length_matrix is not None else None
    # since best vntr mean length values and vntr mean length values are mutually exclusive
    # just replace vntr_mean_length values variable with best_vntr_mean_length_values and use that for plotting instead.
    if best_vntr_mean_length_values is not None:
        vntr_mean_length_values = best_vntr_mean_length_values
        vntr_mean_length_matrix = best_vntr_mean_length_matrix
        vntr_mean_length_ID = best_vntr_mean_length_ID
    print(f"Best VNTR mean length variant ID for {gene_id}: {best_vntr_mean_length_ID}") if best_vntr_mean_length_ID is not None else None
    print(f"Best allele length variant ID for {gene_id}: {best_vntr_length_ID}") if best_vntr_length_ID is not None else None
    
    print(f"Common Individuals (without missing data 'nan'): {len(common_inds)}")
    print(common_inds)
    print(f"Gene's gene expression matrix description: {gene_expression.describe()}")
    print(f"VNTR's genotype matrix description: {vntr_values.describe()}")
    print(f"Best known variant's genotype matrix description: {best_variants.describe()}") if best_variant_matrix is not None else print('No best variant genotypes to supplied to --bestvariantfile')
    print(f"VNTR's mean allele length matrix description: {vntr_mean_length_values.describe()}") if vntr_mean_length_matrix is not None else print('No mean VNTR allele lengths to correct for (regress out)')
    print(f"Best known allele length VNTR variant's genotype matrix description: {best_vntr_length.describe()}") if best_vntr_length is not None else print('No best allele length VNTR variants to supplied to --vntrallelelengthfile')
    # print('Assessing Multicolinearity of all covariates:')
    # assess_multicollinearity(allcovariates, gene_expression, vntr_values, common_inds, gene_id, vntr_id, outdir, allcovar=True)

    # if kinship_matrix is not None:
    covariates = covariates.apply(pd.to_numeric, errors="coerce")
    covariates = covariates.dropna(axis=1, how="all")  # Drop columns with all NaNs
    # prep data
    Y, X, X_mat, Z, n_indiv, n_allele, all_freq, sorted_alleles, Z_best_var, enough_data = prep_data(gene_expression, vntr_values, common_inds, covariates, n_common_inds, minsamples, kinship_matrix, vntr_mean_length_values, best_variants, no_rare_alleles=no_rare_alleles, collapse_rare_alleles=collapse_rare_alleles, permute_vntr_mean_lengths=permute_vntr_mean_lengths)
    
    
    # Z_ortho, Kgrm, Klocus, Klocus_ortho,

    if enough_data == False:
        try:
            raise ValueError(f"Not enough data: Too few individuals (<20) or alleles (<1) to be analyzed: Indivs={Z.shape[0]}, alleles={Z.shape[1]}")
        except AttributeError:
            raise ValueError(f"Not enough data: Too few individuals (<20) or alleles (<1) to be analyzed")
    else:
        print(f"Enough data: Number of individuals={Z.shape[0]}, Number of Alleles={Z.shape[1]}")

    # ---------------------------------
    # Center columns of Z, i.e. remove mean of each column (allele) from Z to get Zc (Z centered), return Kc= ZcZcᵀ (centered locus-specific GRM)
    Klocus = Z @ Z.T  # locus-specific GRM (not centered) Klocus = ZZᵀ
    Zc, Kc, info = center_Z(Z, return_K=True)
    print(f"y:{Y}\nX (plus intercept):{X_mat}\nKlocus:{Klocus}\nKc:{Kc}")  # \nZ:{Z}\nZc:{Zc}\nKlocus:{Klocus}\nKc:{Kc}")  # \n Kgrm{Kgrm}
    print(f"Shape of X_mat: {X_mat.shape}")
    print(f"Number of missing '.' mean allele lengths in X_mat:{X_mat['vntr_mean_allele_length'].apply(lambda x: 1 if x == '.' else 0).sum() if vntr_mean_length_values is not None else 'No mean allele lengths'}")
    print(f"raw z:\n{Z}")
    print(f"Sum of raw Z columns:\n{Z.sum(axis=1)}")
    print(f"centered z:\n{Zc}")
    print(f"Sum of Zc columns:\n{Zc.sum(axis=1)}")

    allele_removed = 'nan'
    # remove the most frequent allele column from Z to make Z relative to most frequent allele
    # this changes the interpretation of the allele effect predictions to be relative to the most frequent allele
    # only remove if most frequent allele is present with MAF > 0.05
    # no 'rare' all_freq
    # don't remove 'rare' when collapse_rare_alleles == True, then remove second most frequent allele if MAF > 0.05
    # get most frequent allele that is not 'rare'
    all_freq_norare = {k: v for k, v in all_freq.items() if k != 'rare'}
    allele_removed = max(all_freq_norare, key=all_freq_norare.get)
    if all_freq[allele_removed] > 0.05:
        Znofreq = Z.copy()
        Znofreq = Znofreq.drop(columns=[allele_removed])
        print(f"Z minus most frequent allele {allele_removed} (Znofreq):\n{Znofreq}")
        print(f"Sum of Znofreq columns (should not all add up to 2 like Zraw):\n{Znofreq.sum(axis=1)}")
        # Zcnofreq = Zc.copy()
        # Zcnofreq = Zcnofreq.drop(columns=[allele_removed])
        Znofreq_c = center_and_standardize(Znofreq.copy(), standardize=False)
        Znofreq_c_s = center_and_standardize(Znofreq.copy(), standardize=True)
    else:
        Znofreq = None
        print('Most frequent allele is not above MAF 0.05, most frequent allele not removed, no Znofreq')

    if '0' in Z.columns:
        # remove the reference allele column (column name 0) from Z to make Z relative to reference allele
        # only remove if reference allele is present with MAF > 0.05 else skip
        if all_freq['0'] > 0.05:
            Znoref = Z.copy()
            Znoref = Znoref.drop(columns=['0'])
            allele_removed_ref = '0'
            print(f"Z minus reference allele '0' (Znoref):\n{Znoref}")
            print(f"Sum of Znoref columns (should not all add up to 2 like Zraw):\n{Znoref.sum(axis=1)}")
        else:
            Znoref = None
            print('Reference allele not above MAF 0.05. No Znoref')
    else:
        Znoref = None
        print('No reference allele in Z dosage matrix to remove. No Znoref')

    # Fit LMM using centered Zc compute pval by LRT
    # option to scale K to have the mean diagonal = 1 (trace(K)/n == 1), but here we do not scale K because we want sigma_g2 to be interpretable as variance explained by the locus
    # reml = True means use REML (default for heritability estimation), False means use ML (default for LRT testing)
    lr, pval, comps, lmm0, lmm1 = fit_lmm(Y, X_mat, Zc, Z_best_var, reml=reml, return_models=True, high_throughput=False)
    print(f"Non-zero eigenvalues after economic decomposition of Zc: {comps['QS_full'][1]}")
    print(f"Number of non-zero eigenvalues from economic decomposition of Zc: {len(comps['QS_full'][1])}")
    # print(f"{comps['QS_full'][0]}")
    # comps = {
    #     "ll0": float(ll0),
    #     "ll1": float(ll1),
    #     "sigma_g2": sigma_g2,
    #     "sigma_e2": sigma_e2,
    #     "mean_diag_K_before": mean_diag_K_before,
    #     "mean_diag_K_after": mean_diag_K_after,
    #     "var_fixed": var_fixed,
    #     "var_genetic": var_genetic,
    #     "var_residual": var_residual,
    #     "var_total": var_total,
    #     "r2_fixed": float(r2_fixed),
    #     "r2_genetic": float(r2_genetic),
    #     "r2_residual": float(r2_residual),
    #     "r2_locus_param": (float(r2_locus_param) if r2_locus_param is not None else None),
    #     "lr_stat": float(lr_stat),
    #     "p_value": float(p_value),
    #     "lambda": (sigma_e2 / sigma_g2) if sigma_g2 > 0 else np.inf
    #     }
    # comps['r2_genetic'] is the same per-sample genetic fraction which is comparable across loci (already in blup['r2_genetic'])

    # comput BLUPs from centered Zc. Low-frequency alleles will be shrunk more as they are not as reliable as the data from frequent alleles.
    # lambda = sigma_e2 / sigma_g2 (0-1) is the noise-to-signal ratio and controls the amount of allele estimate shrinkage (high lambda = more shrinkage)
    print("\nEstimating random effects per allele (u_hat)...")
    blup = compute_blups(Zc, lmm1, X_mat, Y, info=info)
    # blup['u_hat'] are per-copy (centered) allele effects
    # blup['r2_genetic'] is per-sample genetic fraction that sums with fixed+resid to 1 (so fixed + genetic + resid = 1)
    # blup = {
    #     "u_hat": u_hat_series,
    #     "u_per_sd": u_per_sd,
    #     }

    # No lambda (sig_e2 / sig_e2) ridge penalty effect estimates (for comparison to blup)
    # still add small ridge 1e-8 for stability
    ols = compute_ols_effects(Zc, lmm1, Y, X_mat, info=None)
    
    # ---------------------------------
    # for comparison of using Z (not centered, and could have mean-offset effects per allele) vs Zc (centered, so no mean-offset effects per allele)
    print("\nFitting LMM using raw Z (not centered)...")
    lr_Zraw, pval_Zraw, comps_Zraw, lmm0_Zraw, lmm1_Zraw = fit_lmm(Y, X_mat, Z, Z_best_var, reml=reml, return_models=True, high_throughput=False)
    print(f"Non-zero eigenvalues after economic decomposition of Zraw: {comps_Zraw['QS_full'][1]}")
    print(f"Number of non-zero eigenvalues from economic decomposition of Zraw: {len(comps_Zraw['QS_full'][1])}")
    # print(f"{comps_Zraw['QS_full'][0]}")
    blup_Zraw = compute_blups(Z, lmm1_Zraw, X_mat, Y, info=info)
    ols_Zraw = compute_ols_effects(Z, lmm1_Zraw, Y, X_mat, info=None)
    # use null betas instead of full betas for plotting and blup calculations
    blup_Zraw_nullbetas = compute_blups(Z, lmm0_Zraw, X_mat, Y, info=info)

    # ---
    # test centered and standardized Z -> Zcs
    Zcs = center_and_standardize(Z, standardize=True)
    lr_Zcs, pval_Zcs, comps_Zcs, lmm0_Zcs, lmm1_Zcs = fit_lmm(Y, X_mat, Zcs, Z_best_var, reml=reml, return_models=True, high_throughput=False)
    print(f"Non-zero eigenvalues after economic decomposition of Zcs: {comps_Zcs['QS_full'][1]}")
    print(f"Number of non-zero eigenvalues from economic decomposition of Zcs: {len(comps_Zcs['QS_full'][1])}")
    # print(f"{comps_Zcs['QS_full'][0]}")
    blup_Zcs = compute_blups(Zcs, lmm1_Zcs, X_mat, Y, info=None)
    # ols_Zcs = compute_ols_effects(Zcs, lmm1_Zcs, Y, X_mat, info=None)
    y_resid_Zcs = (Y - X_mat @ lmm1_Zcs.beta).to_numpy()
    export_matrix(Zcs, outfile.replace('.pdf', '.Matrix.Zcs.txt'))
    # make heatmaps of Z, ZZt, ZtZ
    plot_heatmaps(Zcs, save_path=outfile.replace('.pdf', '.Heatmaps.Zcs.pdf'))
    comps_Zcs['allele_removed'] = allele_removed
    plot_BLUPs(Y, y_resid_Zcs, Z, Zcs, blup_Zcs['u_hat'], all_freq, lr_Zcs, pval_Zcs, n_indiv, n_allele, comps_Zcs, save_path=outfile.replace('.pdf', '.Zcs.ZcsSigmas.pdf'))

    # ---------------------------------
    # for comparison drop most frequent allele from Z
    if isinstance(Znofreq, pd.DataFrame):
        lr_Znofreq, pval_Znofreq, comps_Znofreq, lmm0_Znofreq, lmm1_Znofreq = fit_lmm(Y, X_mat, Znofreq, Z_best_var, reml=reml, return_models=True, high_throughput=False)
        print(f"Non-zero eigenvalues after economic decomposition of Znofreq: {comps_Znofreq['QS_full'][1]}")
        print(f"Number of non-zero eigenvalues from economic decomposition of Znofreq: {len(comps_Znofreq['QS_full'][1])}")
        # print(f"{comps_Znofreq['QS_full'][0]}")
        blup_Znofreq = compute_blups(Znofreq, lmm1_Znofreq, X_mat, Y, info=None)
        ols_Znofreq = compute_ols_effects(Znofreq, lmm1_Znofreq, Y, X_mat, info=None)
        y_resid_Znofreq = (Y - X_mat @ lmm1_Znofreq.beta).to_numpy()
        export_matrix(Znofreq, outfile.replace('.pdf', '.Matrix.Z_NoFreq.txt'))
        # make heatmaps of Z, ZZt, ZtZ
        plot_heatmaps(Znofreq, save_path=outfile.replace('.pdf', '.Heatmaps.Znofreq.pdf'))
        compare_uhat_NoRidge_to_uhat_BLUP(ols_Znofreq, blup_Znofreq['u_hat'], save_path=outfile.replace('.pdf', '.OLSEffectsvsBLUPEffects.Znofreq.pdf'))
        comps_Znofreq['allele_removed'] = allele_removed
        plot_BLUPs(Y, y_resid_Znofreq, Z, Znofreq, blup_Znofreq['u_hat'], all_freq, lr_Znofreq, pval_Znofreq, n_indiv, n_allele, comps_Znofreq, save_path=outfile.replace('.pdf', '.Znofreq.ZnofreqSigmas.pdf'))

        # # ---
        # # test effect of dropping most frequent allele on Zc (Z raw was centered before allele was dropped)
        # lr_Zcnofreq, pval_Zcnofreq, comps_Zcnofreq, lmm0_Zcnofreq, lmm1_Zcnofreq = fit_lmm(Y, X_mat, Zcnofreq, scale_K=False, reml=reml, return_models=True, high_throughput=False)
        # print(f"Non-zero eigenvalues after economic decomposition of Zcnofreq: {comps_Zcnofreq['QS_full'][1]}")
        # print(f"Number of non-zero eigenvalues from economic decomposition of Zcnofreq: {len(comps_Zcnofreq['QS_full'][1])}")
        # # print(f"{comps_Zcnofreq['QS_full'][0]}")
        # blup_Zcnofreq = compute_blups(Zcnofreq, lmm1_Zcnofreq, X_mat, Y, info=None)
        # # ols_Zcnofreq = compute_ols_effects(Zcnofreq, lmm1_Zcnofreq, Y, X_mat, info=None)
        # y_resid_Zcnofreq = (Y - X_mat @ lmm1_Zcnofreq.beta).to_numpy()
        # export_matrix(Zcnofreq, outfile.replace('.pdf', '.Matrix.Zc_NoFreq.txt'))
        # # make heatmaps of Z, ZZt, ZtZ
        # plot_heatmaps(Zcnofreq, save_path=outfile.replace('.pdf', '.Heatmaps.Zcnofreq.pdf'))
        # comps_Zcnofreq['allele_removed'] = allele_removed
        # plot_BLUPs(Y, y_resid_Zcnofreq, Z, Zcnofreq, blup_Zcnofreq['u_hat'], all_freq, lr_Zcnofreq, pval_Zcnofreq, n_indiv, n_allele, comps_Zcnofreq, save_path=outfile.replace('.pdf', '.Zcnofreq.ZcnofreqSigmas.pdf'))

        # ---
        # test effect of centering after removing most frequent allele Znofreq -> Znofreq_c
        lr_Znofreq_c, pval_Znofreq_c, comps_Znofreq_c, lmm0_Znofreq_c, lmm1_Znofreq_c = fit_lmm(Y, X_mat, Znofreq_c, Z_best_var, reml=reml, return_models=True, high_throughput=False)
        print(f"Non-zero eigenvalues after economic decomposition of Znofreq_c: {comps_Znofreq_c['QS_full'][1]}")
        print(f"Number of non-zero eigenvalues from economic decomposition of Znofreq_c: {len(comps_Znofreq_c['QS_full'][1])}")
        # print(f"{comps_Znofreq_c['QS_full'][0]}")
        blup_Znofreq_c = compute_blups(Znofreq_c, lmm1_Znofreq_c, X_mat, Y, info=None)
        # ols_Znofreq = compute_ols_effects(Znofreq, lmm1_Znofreq, Y, X_mat, info=None)
        y_resid_Znofreq_c = (Y - X_mat @ lmm1_Znofreq_c.beta).to_numpy()
        export_matrix(Znofreq_c, outfile.replace('.pdf', '.Matrix.Z_NoFreq_c.txt'))
        # make heatmaps of Z, ZZt, ZtZ
        plot_heatmaps(Znofreq_c, save_path=outfile.replace('.pdf', '.Heatmaps.Znofreq_c.pdf'))
        comps_Znofreq_c['allele_removed'] = allele_removed
        plot_BLUPs(Y, y_resid_Znofreq_c, Z, Znofreq_c, blup_Znofreq_c['u_hat'], all_freq, lr_Znofreq_c, pval_Znofreq_c, n_indiv, n_allele, comps_Znofreq_c, save_path=outfile.replace('.pdf', '.Znofreq_c.Znofreq_cSigmas.pdf'))

        # ---
        # test effect of centering and standardizing variance to 1 std. after removing most frequent allele Znofreq -> Znofreq_c_s
        lr_Znofreq_c_s, pval_Znofreq_c_s, comps_Znofreq_c_s, lmm0_Znofreq_c_s, lmm1_Znofreq_c_s = fit_lmm(Y, X_mat, Znofreq_c_s, Z_best_var, reml=reml, return_models=True, high_throughput=False)
        print(f"Non-zero eigenvalues after economic decomposition of Znofreq_c_s: {comps_Znofreq_c_s['QS_full'][1]}")
        print(f"Number of non-zero eigenvalues from economic decomposition of Znofreq_c_s: {len(comps_Znofreq_c_s['QS_full'][1])}")
        # print(f"{comps_Znofreq_c_s['QS_full'][0]}")
        blup_Znofreq_c_s = compute_blups(Znofreq_c_s, lmm1_Znofreq_c_s, X_mat, Y, info=None)
        # ols_Znofreq = compute_ols_effects(Znofreq, lmm1_Znofreq, Y, X_mat, info=None)
        y_resid_Znofreq_c_s = (Y - X_mat @ lmm1_Znofreq_c_s.beta).to_numpy()
        export_matrix(Znofreq_c_s, outfile.replace('.pdf', '.Matrix.Z_NoFreq_c_s.txt'))
        # make heatmaps of Z, ZZt, ZtZ
        plot_heatmaps(Znofreq_c_s, save_path=outfile.replace('.pdf', '.Heatmaps.Znofreq_c_s.pdf'))
        comps_Znofreq_c_s['allele_removed'] = allele_removed
        plot_BLUPs(Y, y_resid_Znofreq_c_s, Z, Znofreq_c_s, blup_Znofreq_c_s['u_hat'], all_freq, lr_Znofreq_c_s, pval_Znofreq_c_s, n_indiv, n_allele, comps_Znofreq_c_s, save_path=outfile.replace('.pdf', '.Znofreq_c_s.Znofreq_c_sSigmas.pdf'))

    # ---------------------------------
    # for comparison drop most reference allele from Z
    if isinstance(Znoref, pd.DataFrame):
        lr_Znoref, pval_Znoref, comps_Znoref, lmm0_Znoref, lmm1_Znoref = fit_lmm(Y, X_mat, Znoref, Z_best_var, reml=reml, return_models=True, high_throughput=False)
        print(f"Non-zero eigenvalues after economic decomposition of Znoref: {comps_Znoref['QS_full'][1]}")
        print(f"Number of non-zero eigenvalues from economic decomposition of Znoref: {len(comps_Znoref['QS_full'][1])}")
        # print(f"{comps_Znoref['QS_full'][0]}")
        blup_Znoref = compute_blups(Znoref, lmm1_Znoref, X_mat, Y, info=None)
        ols_Znoref = compute_ols_effects(Znoref, lmm1_Znoref, Y, X_mat, info=None)
        y_resid_Znoref= (Y - X_mat @ lmm1_Znoref.beta).to_numpy()
        export_matrix(Znoref, outfile.replace('.pdf', '.Matrix.Z_NoRef.txt'))
        # make heatmaps of Z, ZZt, ZtZ
        plot_heatmaps(Znoref, save_path=outfile.replace('.pdf', '.Heatmaps.Znoref.pdf'))
        compare_uhat_NoRidge_to_uhat_BLUP(ols_Znoref, blup_Znoref['u_hat'], save_path=outfile.replace('.pdf', '.OLSEffectsvsBLUPEffects.Znoref.pdf'))
        comps_Znoref['allele_removed'] = allele_removed_ref
        plot_BLUPs(Y, y_resid_Znoref, Z, Znoref, blup_Znoref['u_hat'], all_freq, lr_Znoref, pval_Znoref, n_indiv, n_allele, comps_Znoref, save_path=outfile.replace('.pdf', '.Znoref.ZnorefSigmas.pdf'))
        
    # ---------------------------------
    # Z_full will have same individuals as --collapserarealleles, but have all alleles seperate, and not collapsed to 'rare' column
    if collapse_rare_alleles == True:
        # full_Z = true will have same individuals as --collapserarealleles, but have all alleles seperate, and not collapsed to 'rare' column
        Y_Zfull, X_Zfull, X_mat_Zfull, Z_Zfull, n_indiv_Zfull, n_allele_Zfull, all_freq_Zfull, sorted_alleles_Zfull, Z_best_var, enough_data_Zfull = prep_data(gene_expression, vntr_values, common_inds, covariates, n_common_inds, minsamples, kinship_matrix, vntr_mean_length_values, best_variants, no_rare_alleles=False, collapse_rare_alleles=False, permute_vntr_mean_lengths=permute_vntr_mean_lengths, full_Z=True)
        Zc_Zfull, info = center_Z(Z_Zfull, return_K=False)
        lr_Zfull, pval_Zfull, comps_Zfull, lmm0_Zfull, lmm1_Zfull = fit_lmm(Y, X_mat, Zc_Zfull, Z_best_var, reml=reml, return_models=True, high_throughput=False)
        blup_Zcfull = compute_blups(Zc_Zfull, lmm1_Zfull, X_mat, Y, info=info)   # use 'raw' Z_full no centering # use sigmas from lmm1_Zfull on Zc_Zfull
        blup_Zcfull_w_ZcSigmas = compute_blups(Zc_Zfull, lmm1, X_mat, Y, info=info)  # use 'raw' Z_full no centering # use sigmas from lmm1 on Zc_Zfull
        ols_Zfull = compute_ols_effects(Zc_Zfull, lmm1, Y, X_mat, info=None)

    export_matrix(Z, outfile.replace('.pdf', '.Matrix.Z_Raw.txt'))
    export_matrix(Klocus, outfile.replace('.pdf', '.Matrix.Klocus_Raw.txt'))
    export_matrix(Zc, outfile.replace('.pdf', '.Matrix.Z_centered.txt'))
    export_matrix(Kc, outfile.replace('.pdf', '.Matrix.Klocus_centered.txt'))

    diagnose_matrix(Klocus, outfile.replace('.pdf', '.Matrix.Klocus_Raw.Diagnose.txt'))
    diagnose_matrix(Kc, outfile.replace('.pdf', '.Matrix.Klocus_centered.Diagnose.txt'))

    w = np.linalg.eigvalsh(Kc)
    print("eigenvalues (desc, of centered Z):", np.sort(w)[::-1][:10])
    print("count > 1e-12:", np.sum(w > 1e-12))

    beta_hat = lmm1.beta
    y_resid = (Y - X_mat @ beta_hat).to_numpy()
    if Z_best_var is not None:
        # if best variant already in X_mat do not do this.
        # JUST FOR VISUALIZATION BELOW
        # get vector of genotypes for best variant and plot against mean allele length, length = number of individuals (rows)
        # sum the column header names instead of summing the values. assume header names are 0, 1, 2 for different alleles. 
        genotypes = np.zeros(Z_best_var.shape[0])
        for col in Z_best_var.columns:
            genotypes += Z_best_var[col].values * int(col)
        # add Z_best_var genotypes to X_mat and fit lmm null and full with this new X_mat, do not add Z_best_var to null random effect matrix
        X_mat_with_bestvar = X_mat.copy()
        X_mat_with_bestvar['best_variant_genotype'] = genotypes
        lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_bestvar, Z, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
        y_resid_Zraw = (Y - X_mat_with_bestvar @ lmm1_temp.beta).to_numpy()  # by adding snp genotypes to the fixed effect X_mat matrix we can visualize the residuals on y axis of top right plot when plot_BLUPs is called
        # JUST FOR VISUALIZATION ABOVE
    else:
        y_resid_Zraw = (Y - X_mat @ lmm1_Zraw.beta).to_numpy()
    y_resid_Zraw_nullbetas = (Y - X_mat @ lmm0_Zraw.beta).to_numpy()
    
    # y_resid_Zfull = (Y - X_mat @ lmm1_Zfull.beta).to_numpy()  # should = y_resid b/c same individuals and X_mat
    # from: https://glimix-core.readthedocs.io/en/latest/lmm.html
    # 𝐲 ∼ 𝓝(X𝜷, v₀GGᵀ + v₁I).
    # sigma_g2, sigma_e2 = lmm1.v0, lmm1.v1


    print(f"Unscaled (raw/absolute) sigmas g and e: {comps['sigma_g2']}, {comps['sigma_e2']}")
    print(f"Variance explained by fixed effects: {comps['var_fixed']:.4f}, locus variance (genetic): {comps['var_genetic']:.4f}, residual variance: {comps['var_residual']:.4f}, Total variance: {comps['var_total']:.4f}")
    print(f"Proportion of variance explained by fixed effects: {comps['r2_fixed']:.4f}, by locus (genetic): {comps['r2_genetic']:.4f}, residuals: {comps['r2_residual']:.4f}")
    print(f"Likelihood ratio: {lr}, p-value: {pval}")

    print("\nEstimated random effects (u_hat_locus):")
    print(blup['u_hat'].to_string(float_format=lambda x: f"{x:.4f}"))

    print("Maximal allele effect (OLS like beta) when no lambda ridge penalty applied:")
    print(ols.to_string(float_format=lambda x: f"{x:.4f}"))

    # somedata = plot_K_mean_eigenstructure(Z, save_path=outfile.replace('.pdf', '.CompareZMeansInEigenstructureOfK.pdf'))

    # make heatmaps of Z, ZZt, ZtZ
    plot_heatmaps(Z, save_path=outfile.replace('.pdf', '.Heatmaps.Zraw.pdf'))
    # make heatmapes of Zc, ZcZc^T, Zc^TZc
    plot_heatmaps(Zc, save_path=outfile.replace('.pdf', '.Heatmaps.Zc.pdf'))

    plot_r2_and_PCA(comps['r2_fixed'], comps['r2_genetic'], comps['r2_residual'], Zc, save_path=outfile.replace('.pdf', '.R2_PCA.pdf'))
    # compare_sigma_u_Z(sigma_g2, sigma_e2, sigma_g2_ortho, sigma_e2_ortho, u_hat_locus, u_hat_locus_ortho_sigmas, Z, Z_ortho, save_path=outfile.replace('.pdf', '.DifferencesBtwnSigmasAndDosageMatrices.pdf'))

    # make BLUE plots
    # Specify u_hat values for top two plots Z * u_hat for individual BLUE. Allele effect vs allele frequency uses u_hat specified. Then specify sigmas to be reported in table.
    plot_BLUPs(Y, y_resid, Z, Zc, blup['u_hat'], all_freq, lr, pval, n_indiv, n_allele, comps, save_path=outfile.replace('.pdf', '.Zc.ZcSigmas.pdf'))
    plot_BLUPs(Y, y_resid_Zraw, Z, Z, blup_Zraw['u_hat'], all_freq, lr_Zraw, pval_Zraw, comps_Zraw['n'], comps_Zraw['m'], comps_Zraw, save_path=outfile.replace('.pdf', '.Zraw.ZrawSigmas.pdf'))
    plot_BLUPs(Y, y_resid_Zraw_nullbetas, Z, Z, blup_Zraw_nullbetas['u_hat'], all_freq, lr_Zraw, pval_Zraw, comps_Zraw['n'], comps_Zraw['m'], comps_Zraw, save_path=outfile.replace('.pdf', '.Zraw.ZrawSigmas.NullBetas.pdf'))
    make_plots_for_paper(y_resid_Zraw, Z, Z, blup_Zraw['u_hat'], all_freq, comps_Zraw, save_path=outfile.replace('.pdf', '.Zraw.ZrawSigmas_PaperFigure.pdf'))
    
    # TEST #
    # TEST #
    # simple linear regression of Y ~ individual level BLUPs to see how well BLUPs predict raw gene expression and get R2
    individual_blues = Z @ blup_Zraw['u_hat'].values
    Y_simple, X_simple, r2_simple, beta_simple, se_simple, pvalue_simple, model_simple = perform_simplelinear(Y, individual_blues, add_intercept=True)
    prop_genetic_variance_null = lmm0_Zraw.v0 / (lmm0_Zraw.v0 + lmm0_Zraw.v1)
    prop_genetic_variance = lmm1_Zraw.v0 / (lmm1_Zraw.v0 + lmm1_Zraw.v1)
    prop_not_fixed_variance = comps_Zraw['scale'] / comps_Zraw['var_total']
    print(f"Proportion of variance that is not fixed effects: {prop_not_fixed_variance}")
    print(f"\nZraw null model values:\nlml: {lmm0_Zraw.lml()}\nDelta: {lmm0_Zraw.delta}\nSacling factor: {lmm0_Zraw.scale}\nGenetic Variance: {lmm0_Zraw.v0}\nResidual Variance: {lmm0_Zraw.v1}\nMean: {lmm0_Zraw.mean()}\nCovariance: {lmm0_Zraw.covariance()}\nProportion genetic variance: {prop_genetic_variance_null}")
    print(f"\nZraw full model values:\nlml: {lmm1_Zraw.lml()}\nDelta: {lmm1_Zraw.delta}\nSacling factor: {lmm1_Zraw.scale}\nGenetic Variance: {lmm1_Zraw.v0}\nResidual Variance: {lmm1_Zraw.v1}\nMean: {lmm1_Zraw.mean()}\nCovariance: {lmm1_Zraw.covariance()}\nProportion genetic variance: {prop_genetic_variance}")
    print(f"length of mean: {len(lmm1_Zraw.mean())}, should be n={lmm1_Zraw.nsamples}")
    K_test = Z.values @ Z.values.T
    mean_diag_K = float(np.trace(K_test) / K_test.shape[0])
    print(f"full model mean diagonal: {mean_diag_K} maybe same as scaling factor")
    mean_diag_cov = float(np.trace(lmm1_Zraw.covariance()) / lmm1_Zraw.nsamples)
    print(f"full model mean diagonal of covariance matrix (full model.covariance()): {mean_diag_cov}")
    plot_simplelinear(Y_simple, X_simple, r2_simple, beta_simple, se_simple, pvalue_simple, model_simple, save_path=outfile.replace('.pdf', '.SimpleLinear_Yraw_vs_IndividualBLUPs_Zraw.pdf'), title='Simple Linear Regression of Gene Expression Residuals vs Individual BLUPs from Zraw')
    
    # TEST #
    # TEST #
    # simple linear regression of y_residuals ~ individual level BLUPs to see how well BLUPs predict gene expression residuals and get R2
    individual_blues = Z @ blup_Zraw['u_hat'].values
    Y_simple, X_simple, r2_simple, beta_simple, se_simple, pvalue_simple, model_simple = perform_simplelinear(y_resid_Zraw, individual_blues, add_intercept=True)
    prop_genetic_variance_null = lmm0_Zraw.v0 / (lmm0_Zraw.v0 + lmm0_Zraw.v1)
    prop_genetic_variance = lmm1_Zraw.v0 / (lmm1_Zraw.v0 + lmm1_Zraw.v1)
    prop_not_fixed_variance = comps_Zraw['scale'] / comps_Zraw['var_total']
    print(f"Proportion of variance that is not fixed effects: {prop_not_fixed_variance}")
    print(f"\nZraw null model values:\nlml: {lmm0_Zraw.lml()}\nDelta: {lmm0_Zraw.delta}\nSacling factor: {lmm0_Zraw.scale}\nGenetic Variance: {lmm0_Zraw.v0}\nResidual Variance: {lmm0_Zraw.v1}\nMean: {lmm0_Zraw.mean()}\nCovariance: {lmm0_Zraw.covariance()}\nProportion genetic variance: {prop_genetic_variance_null}")
    print(f"\nZraw full model values:\nlml: {lmm1_Zraw.lml()}\nDelta: {lmm1_Zraw.delta}\nSacling factor: {lmm1_Zraw.scale}\nGenetic Variance: {lmm1_Zraw.v0}\nResidual Variance: {lmm1_Zraw.v1}\nMean: {lmm1_Zraw.mean()}\nCovariance: {lmm1_Zraw.covariance()}\nProportion genetic variance: {prop_genetic_variance}")
    print(f"length of mean: {len(lmm1_Zraw.mean())}, should be n={lmm1_Zraw.nsamples}")
    K_test = Z.values @ Z.values.T
    mean_diag_K = float(np.trace(K_test) / K_test.shape[0])
    print(f"full model mean diagonal: {mean_diag_K} maybe same as scaling factor")
    mean_diag_cov = float(np.trace(lmm1_Zraw.covariance()) / lmm1_Zraw.nsamples)
    print(f"full model mean diagonal of covariance matrix (full model.covariance()): {mean_diag_cov}")
    plot_simplelinear(Y_simple, X_simple, r2_simple, beta_simple, se_simple, pvalue_simple, model_simple, save_path=outfile.replace('.pdf', '.SimpleLinear_Yresid_vs_IndividualBLUPs_Zraw.pdf'), title='Simple Linear Regression of Gene Expression Residuals vs Individual BLUPs from Zraw')


    Y_multi, X_multi, r2_multi, beta_multi, se_multi, pvalue_multi, model_multi, y_resid_multi, x_resid_multi = perform_multilinear(Y, individual_blues, X_mat, label=vntr_id, common_inds=None, add_intercept=True)
    print(f"Multiple Linear Regression of Y ~ 1 + Covariates + Individual BLUPs from Zraw:\nR2: {r2_multi}\nBeta (Individual BLUPs): {beta_multi}\nSE: {se_multi}\nP-value: {pvalue_multi}\nModel Summary:\n{model_multi.summary()}")
    plot_added_variable(Y_multi, X_multi, r2_multi, beta_multi, se_multi, pvalue_multi, save_path=outfile.replace('.pdf', '.MultipleLinear.rawY_vs_IndividualBLUPs.rawX_Zraw.pdf'), title="Added-Variable Plot: Y_residuals vs Individual predicted effect on gene expression")
    plot_added_variable(Y_simple, x_resid_multi, r2_multi, beta_multi, se_multi, pvalue_multi, save_path=outfile.replace('.pdf', '.MultipleLinear.SIMPLEMIMICresid_vs_IndividualBLUPs.resid_Zraw.pdf'), title="Added-Variable Plot: Y_residuals vs Individual predicted effect on gene expression")
    plot_added_variable(y_resid_multi, x_resid_multi, r2_multi, beta_multi, se_multi, pvalue_multi, save_path=outfile.replace('.pdf', '.MultipleLinear.resid_vs_IndividualBLUPs.resid_Zraw.pdf'), title="Added-Variable Plot: Y_residuals vs Individual predicted effect on gene expression")

    for b, multi_b in zip(lmm1_Zraw.beta, model_multi.params):
        print(f"Beta coef (full Zraw Model): {b:.4f}, Multi Beta coef: {multi_b:.4f}")
    # correlation between LMM residuals and multiple linear model residuals
    corr_resid, pval_resid = pearsonr(y_resid_Zraw, y_resid_multi)
    print(f"Correlation between LMM residuals and Multiple Linear Model residuals: r={corr_resid:.4f}, pval={pval_resid:.4e}")
    plt.figure(figsize=(6, 5))
    plt.scatter(y_resid_Zraw, y_resid_multi)
    plt.grid(True)
    # plt.show()
    # correlation between LMM residuals and multiple linear model residuals
    for b, multi_b in zip(lmm0_Zraw.beta, model_multi.params):
        print(f"Beta coef (null Zraw model): {b:.4f}, Multi Beta coef: {multi_b:.4f}")
    corr_resid, pval_resid = pearsonr((Y - X_mat @ lmm0_Zraw.beta).to_numpy(), y_resid_multi)
    print(f"Correlation between LMM residuals and Multiple Linear Model residuals: r={corr_resid:.4f}, pval={pval_resid:.4e}")
    plt.figure(figsize=(6, 5))
    plt.scatter((Y - X_mat @ lmm0_Zraw.beta).to_numpy(), y_resid_multi)
    plt.grid(True)
    # plt.show()


    print(f'X (plus intercept):{X_mat}\n')
    for b0, b1 in zip(lmm0_Zraw.beta, lmm1_Zraw.beta):
        print(f"lmm0_Zraw.beta: {b0:.4f}, lmm1_Zraw.beta: {b1:.4f}")
    plt.figure(figsize=(6, 5))
    plt.scatter(Y - X_mat @ lmm1_Zraw.beta, Y - X_mat @ lmm0_Zraw.beta)
    plt.title('Comparison of LMM Residuals: Full Model vs Null Model')
    plt.grid(True)
    # plt.show()

    # lmm0_temp and lmm1_temp have the same column labels (the names of the fixed effects) as X_mat_with_len_bestvar
    plot_df = pd.DataFrame({
        'Fixed Effect': X_mat.columns,
        'Null Model': lmm0_Zraw.beta,
        'Full Model': lmm1_Zraw.beta,
    })
    # Reshape from "wide" to "long" format
    df_melted = plot_df.melt(id_vars='Fixed Effect', var_name='Model', value_name='Beta Value')
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_melted,
        x='Fixed Effect', 
        y='Beta Value',
        hue='Model'       # This creates the "side-by-side" bars
    )
    plt.title("Comparison of Beta Estimates: Null vs Full Models")
    plt.xticks(rotation=45, ha='right') # Rotates labels if you have many effects
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(outfile.replace('.pdf', '.FixedEffectBetaComparisonNullVSFullZraw.barchart.pdf'))
    plt.close()

    # lmm0_temp and lmm1_temp have the same column labels (the names of the fixed effects) as X_mat_with_len_bestvar
    plot_df = pd.DataFrame({
        'Fixed Effect': X_mat.columns,
        'Null Model': lmm0_Znofreq.beta,
        'Full Model': lmm1_Znofreq.beta,
    })
    # Reshape from "wide" to "long" format
    df_melted = plot_df.melt(id_vars='Fixed Effect', var_name='Model', value_name='Beta Value')
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_melted,
        x='Fixed Effect', 
        y='Beta Value',
        hue='Model'       # This creates the "side-by-side" bars
    )
    plt.title("Comparison of Beta Estimates: Null vs Full Models")
    plt.xticks(rotation=45, ha='right') # Rotates labels if you have many effects
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(outfile.replace('.pdf', '.FixedEffectBetaComparisonNullVSFullZnofreq.barchart.pdf'))
    plt.close()
    
    if Z_best_var is not None:
        # just running to make 'true' null model with identity matrix instead of 'best variants' in Z_best_var set to None
        lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat, Z, Z_best_var=Z_best_var, reml=reml, return_models=True, high_throughput=False)
        plt.figure(figsize=(6, 5))
        plt.scatter(Y - X_mat @ lmm1_Zraw.beta, Y - X_mat @ lmm0_temp.beta)
        if vntr_id is not None and 'vntr_mean_allele_length' in X_mat.columns:
            for count, (fullresid, nullresid) in enumerate(zip((Y - X_mat @ lmm1_Zraw.beta).to_numpy(), (Y - X_mat @ lmm0_temp.beta).to_numpy())):
                plt.annotate(X_mat.iloc[count]['vntr_mean_allele_length'], (fullresid, nullresid))
        plt.title('Compare LMM Residuals: Full vs Null Model')
        plt.xlabel('Full Model Residuals (w/ Z_VNTR Matrix)')
        plt.ylabel('Null Model Residuals (w/ Z_best_variant Matrix, not Identity Matrix)')
        plt.grid(True)
        plt.savefig(outfile.replace('.pdf', '.LMMFullResiduals_vs_ZBestVarNullResiduals.pdf'))
        
        # just running to make 'true' null model with identity matrix instead of 'best variants' in Z_best_var set to None
        lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat, Z, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
        lr_temp_bv, pval_temp_bv, comps_temp_bv, lmm0_temp_bv, lmm1_temp_bv = fit_lmm(Y, X_mat, Z_best_var, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
        plt.figure(figsize=(6, 5))
        plt.scatter(Y - X_mat @ lmm1_temp_bv.beta, Y - X_mat @ lmm0_temp.beta)
        if vntr_id is not None and 'vntr_mean_allele_length' in X_mat.columns:
            for count, (fullresid, nullresid) in enumerate(zip((Y - X_mat @ lmm1_temp_bv.beta).to_numpy(), (Y - X_mat @ lmm0_temp.beta).to_numpy())):
                plt.annotate(X_mat.iloc[count]['vntr_mean_allele_length'], (fullresid, nullresid))
        plt.title('Compare LMM Residuals: Full vs Null Model')
        plt.xlabel('Full Model Residuals (w/ Z_Best_Variant Matrix)')
        plt.ylabel('Null Model Residuals (w/ Identity Matrix)')
        plt.grid(True)
        plt.savefig(outfile.replace('.pdf', '.LMMFullZBestVarResiduals_vs_IndentityNullResiduals.pdf'))

        # add Z_best_var genotypes to X_mat and fit lmm null and full with this new X_mat, do not add Z_best_var to null random effect matrix
        X_mat_with_bestvar = X_mat.copy()
        X_mat_with_bestvar['best_variant_genotype'] = genotypes

        if vntr_mean_length_values is not None:
            # delete mean allele length from X_mat_with_bestvar to see if best variant genotype is tagging mean allele length
            X_mat_with_bestvar = X_mat_with_bestvar.drop(columns=['vntr_mean_allele_length'])
        lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_bestvar, Z, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
        plt.figure(figsize=(6, 5))
        plt.scatter(Y - X_mat_with_bestvar @ lmm1_temp.beta, Y - X_mat_with_bestvar @ lmm0_temp.beta)
        plt.title('BestVariant Genotype vs VNTR allele effects in LMM')
        plt.xlabel('Full Model Residuals (w/ Z_VNTR Matrix)')
        plt.ylabel('Null Model Residuals (w/ Identity Matrix)')
        plt.grid(True)
        plt.savefig(outfile.replace('.pdf', '.BestVariantGenotype012.InXMat.pdf'))
        y_resid_Zbestvar = (Y - X_mat_with_bestvar @ lmm0_temp.beta).to_numpy()  # use residuals from model with best variant genotype as covariate in X_mat to compute BLUPs with Z_ortho and plot against BLUPs from Zraw full model
        blup_Zbestvar_nullbetas = compute_blups(Z, lmm0_temp, X_mat_with_bestvar, Y, info=None)
        blup_Zbestvar = compute_blups(Z, lmm1_temp, X_mat_with_bestvar, Y, info=None)
        plot_BLUPs(Y, y_resid_Zbestvar, Z, Z, blup_Zbestvar['u_hat'], all_freq, lr_temp, pval_temp, comps_temp['n'], comps_temp['m'], comps_temp, save_path=outfile.replace('.pdf', '.Zraw.ZrawSigmas.BVInFixedEffects.pdf'))

        # with Z_ortho # just running to make 'true' null model with identity matrix instead of 'best variants' in Z_best_var set to None
        Z_ortho = orthogonalize_Z_relative_to_X(Z, X_mat_with_bestvar)
        lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_bestvar, Z_ortho, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
        # y_resid_Zortho = (Y - X_mat @ lmm1_Zortho.beta).to_numpy()
        blup_Zortho_bestvar = compute_blups(Z_ortho, lmm1_temp, X_mat_with_bestvar, Y, info=None)
        # # this is plotting the same as Zraw dosage matrix and Zraw full model BLUPs but with y residuals from lmm0 with identity matrix
        # plot_BLUPs(Y, y_resid_Zortho, Z, Z, blup_Zraw['u_hat'], all_freq, lr_Zortho, pval_Zortho, comps_Zortho['n'], comps_Zortho['m'], comps_Zortho, save_path=outfile.replace('.pdf', '.Zraw.ZorthoSigmas.pdf'))
        
        # lmm0_temp and lmm1_temp have the same column labels (the names of the fixed effects) as X_mat_with_len_bestvar
        plot_df = pd.DataFrame({
            'Fixed Effect': X_mat_with_bestvar.columns,
            'Null Model': lmm0_temp.beta,  # this is Zortho
            'Full Model': lmm1_temp.beta,  # this is Zortho
        })
        # Reshape from "wide" to "long" format
        df_melted = plot_df.melt(id_vars='Fixed Effect', var_name='Model', value_name='Beta Value')
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_melted,
            x='Fixed Effect', 
            y='Beta Value',
            hue='Model'       # This creates the "side-by-side" bars
        )
        plt.title(f"Comparison of Beta Estimates: Null vs Full Models (pval={pval_temp})")
        plt.xticks(rotation=45, ha='right') # Rotates labels if you have many effects
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(outfile.replace('.pdf', '.FixedEffectBetaComparisonNullVSFullZortho.BVinXmat.barchart.pdf'))
        plt.close()
        
        if vntr_mean_length_values is not None:
            # plot vntr mean allele length against best variant genotypes to see if best variant is tagging mean allele length
            plt.figure(figsize=(6, 5))
            # get vector of genotypes for best variant and plot against mean allele length, length = number of individuals (rows)
            # sum the column header names instead of summing the values. assume header names are 0, 1, 2 for different alleles. 
            genotypes = np.zeros(Z_best_var.shape[0])
            for col in Z_best_var.columns:
                genotypes += Z_best_var[col].values * int(col)
            # print(genotypes)
            plt.scatter(genotypes, X_mat['vntr_mean_allele_length']) # Z_best_var.values.flatten()
            plt.title('Mean VNTR Allele Length vs Genotype of Best Variant')
            plt.xlabel('Genotype of Best Variant')
            plt.ylabel('Mean VNTR Allele Length')
            plt.grid(True)
            plt.savefig(outfile.replace('.pdf', '.BestVariantGenotype_VS_MeanAlleleLength.pdf'))
            print(f"###\nCorrelation between mean VNTR allele length and best variant genotype: r={pearsonr(X_mat['vntr_mean_allele_length'], genotypes)[0]:.4f}, pval={pearsonr(X_mat['vntr_mean_allele_length'], genotypes)[1]:.4e}\n###")

            vntr_genotypes = []
            for i in range(Z.shape[0]):  # for each row
                alleles = Z.columns[np.where(Z.iloc[i] > 0)[0]]  # get column names (allele number) when cell value > 0
                vntr_genotype = f"{alleles[0]}/{alleles[1]}" if len(alleles) == 2 else f"{alleles[0]}/{alleles[0]}"  # there are always just two columns in Z_raw, one allele per column
                vntr_genotypes.append(vntr_genotype)
            print(f"genotype length {len(genotypes)}")
            print(f"vntr_genotypes length {len(vntr_genotypes)}")
            print(f"gene expression length {len(Y)}")
            # unique_vntr_genotypes = set(vntr_genotypes)
            # test hue by randomly permuting the vntr genotypes
            # vntr_genotypes_permuted = np.random.permutation(vntr_genotypes)
            plt.figure(figsize=(6, 5))
            # 1. Create the violin plot (set inner=None to keep the background clean)
            sns.violinplot(x=genotypes, y=X_mat['vntr_mean_allele_length'], inner=None, color=".8")
            # 2. Layer the jittered points on top colored by the original vntr genotypes
            # sns.stripplot(x=genotypes, y=X_mat['vntr_mean_allele_length'], hue=vntr_genotypes, jitter=0.3, alpha=0.75)  # , palette="viridis" jitter=True
            # Normalize Y for sizing (so dots aren't too tiny or too huge) # Now dot size reflects the original gene expression and hue represents the vntr_genotypes
            print(Y)
            y_sizes = ((Y - Y.min()) / (Y.max() - Y.min()) * 100).values
            # It handles 'size' and 'hue' together much more reliably
            jitter_strength = 0.3
            jitter_genotype = genotypes + np.random.uniform(-jitter_strength, jitter_strength, size=genotypes.shape)
            jitter_allele_length = X_mat['vntr_mean_allele_length'] + np.random.uniform(-jitter_strength, jitter_strength, size=X_mat['vntr_mean_allele_length'].shape)
            # x=genotypes, 
            # y=X_mat['vntr_mean_allele_length'], 
            sns.scatterplot(
                x=jitter_genotype,
                y=jitter_allele_length,
                hue=vntr_genotypes, 
                size=y_sizes,          # This will now map correctly
                sizes=(20, 150),       # This defines the min/max visual size on the plot
                legend='brief',        # Keeps the legend manageable
                alpha=0.5,
                edgecolor='black'
            )
            plt.title("Distribution of VNTR Mean Allele Length by Genotype")
            plt.xlabel('Genotype of Best Variant')
            plt.ylabel('Mean VNTR Allele Length')
            # Move legend outside if there are many unique VNTR genotypes
            plt.legend(title="VNTR Genotype", fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
            # plt.legend(title="VNTR Genotype", fontsize=7)
            plt.grid(True)
            plt.tight_layout()  # Adjust layout to make room for legend
            plt.savefig(outfile.replace('.pdf', '.BestVariantGenotype_VS_MeanAlleleLength.Violin.pdf'), bbox_inches='tight')
            plt.close()

            #--------------------------
            # make scatter plot from the perspective of the best variant
            plt.figure(figsize=(6, 5))
            # sns.violinplot(x=genotypes, y=X_mat['vntr_mean_allele_length'], inner=None, color=".8")
            jitter_zeros = [0]*len(X_mat['vntr_mean_allele_length']) + np.random.uniform(-jitter_strength, jitter_strength, size=X_mat['vntr_mean_allele_length'].shape)
            sns.scatterplot(
                x=jitter_genotype,
                y=jitter_zeros,
                hue=vntr_genotypes, 
                size=y_sizes,          # This will now map correctly
                sizes=(20, 150),       # This defines the min/max visual size on the plot
                legend='brief',        # Keeps the legend manageable
                alpha=0.5,
                edgecolor='black'
            )
            plt.title("Distribution of VNTR Mean Allele Length by Genotype")
            plt.xlabel('Genotype of Best Variant')
            plt.ylabel('Zero')
            # Move legend outside if there are many unique VNTR genotypes
            plt.legend(title="VNTR Genotype", fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
            # plt.legend(title="VNTR Genotype", fontsize=7)
            plt.xlim(-0.5, 2.5)
            plt.ylim(-1, 1)
            plt.grid(True)
            plt.tight_layout()  # Adjust layout to make room for legend
            plt.savefig(outfile.replace('.pdf', '.BestVariantGenotype.Violin.pdf'), bbox_inches='tight')
            plt.close()

            # make scatter plot from the perspective of the best variant VS raw gene expression
            plt.figure(figsize=(6, 5))
            # sns.violinplot(x=genotypes, y=X_mat['vntr_mean_allele_length'], inner=None, color=".8")
            sns.scatterplot(
                x=jitter_genotype,
                y=Y,
                hue=vntr_genotypes, 
                size=y_sizes,          # This will now map correctly
                sizes=(20, 150),       # This defines the min/max visual size on the plot
                legend='brief',        # Keeps the legend manageable
                alpha=0.5,
                edgecolor='black'
            )
            plt.title("Distribution of VNTR Mean Allele Length by Genotype")
            plt.xlabel('Genotype of Best Variant')
            plt.ylabel('Gene Expression (raw)')
            # Move legend outside if there are many unique VNTR genotypes
            plt.legend(title="VNTR Genotype", fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
            # plt.legend(title="VNTR Genotype", fontsize=7)
            plt.grid(True)
            plt.tight_layout()  # Adjust layout to make room for legend
            plt.savefig(outfile.replace('.pdf', '.BestVariantGenotype_VS_GeneExpressionRaw.Violin.pdf'), bbox_inches='tight')
            plt.close()

            # make scatter plot from the perspective of the mean allele length
            plt.figure(figsize=(6, 5))
            # sns.violinplot(x=genotypes, y=X_mat['vntr_mean_allele_length'], inner=None, color=".8")
            jitter_zeros = [0]*len(genotypes) + np.random.uniform(-jitter_strength, jitter_strength, size=genotypes.shape)
            sns.scatterplot(
                x=jitter_zeros,
                y=jitter_allele_length,
                hue=vntr_genotypes, 
                size=y_sizes,          # This will now map correctly
                sizes=(20, 150),       # This defines the min/max visual size on the plot
                legend='brief',        # Keeps the legend manageable
                alpha=0.5,
                edgecolor='black'
            )
            plt.title("Distribution of VNTR Mean Allele Length by Genotype")
            plt.xlabel('Zeros')
            plt.ylabel('Mean Allele Length')
            # Move legend outside if there are many unique VNTR genotypes
            plt.legend(title="VNTR Genotype", fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
            # plt.legend(title="VNTR Genotype", fontsize=7)
            plt.xlim(-1, 1)
            plt.ylim(jitter_allele_length.min() - 1, jitter_allele_length.max() + 1)
            plt.grid(True)
            plt.tight_layout()  # Adjust layout to make room for legend
            plt.savefig(outfile.replace('.pdf', '.MeanAlleleLength.Violin.pdf'), bbox_inches='tight')
            plt.close()

            # make scatter plot from the perspective of the mean allele length vs raw gene expression
            plt.figure(figsize=(6, 5))
            # sns.violinplot(x=genotypes, y=X_mat['vntr_mean_allele_length'], inner=None, color=".8")
            sns.scatterplot(
                x=jitter_allele_length,
                y=Y,
                hue=vntr_genotypes, 
                size=y_sizes,          # This will now map correctly
                sizes=(20, 150),       # This defines the min/max visual size on the plot
                legend='brief',        # Keeps the legend manageable
                alpha=0.5,
                edgecolor='black'
            )
            plt.title("Distribution of VNTR Mean Allele Length by Genotype")
            plt.xlabel('Mean Allele Length')
            plt.ylabel('Gene expression (raw)')
            # Move legend outside if there are many unique VNTR genotypes
            plt.legend(title="VNTR Genotype", fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
            # plt.legend(title="VNTR Genotype", fontsize=7)
            plt.grid(True)
            plt.tight_layout()  # Adjust layout to make room for legend
            plt.savefig(outfile.replace('.pdf', '.MeanAlleleLength_VS_GeneExpressionRaw.Violin.pdf'), bbox_inches='tight')
            plt.close()

            #--------------------------


            # add Z_best_var genotypes to X_mat and fit lmm null and full with this new X_mat, do not add Z_best_var to null random effect matrix X_mat should already have mean allele length
            X_mat_with_len_bestvar = X_mat.copy()
            X_mat_with_len_bestvar['best_variant_genotype'] = genotypes
            lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_len_bestvar, Z, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
            plt.figure(figsize=(6, 5))
            plt.scatter(Y - X_mat_with_len_bestvar @ lmm1_temp.beta, Y - X_mat_with_len_bestvar @ lmm0_temp.beta)
            plt.title('Mean VNTR Allele Length + BestVariant Genotype vs VNTR allele effects in LMM')
            plt.xlabel('Full Model Residuals (w/ Z_VNTR Matrix)')
            plt.ylabel('Null Model Residuals (w/ Identity Matrix)')
            plt.grid(True)
            plt.savefig(outfile.replace('.pdf', '.MeanAlleleLength+BestVariantGenotype012.InXMat.pdf'))
            y_resid_Zlen_bestvar = (Y - X_mat_with_len_bestvar @ lmm0_temp.beta).to_numpy()  # use residuals from model with best variant genotype as covariate in X_mat to compute BLUPs with Z_ortho and plot against BLUPs from Zraw full model
            blup_Zlen_bestvar_nullbetas = compute_blups(Z, lmm0_temp, X_mat_with_len_bestvar, Y, info=None)
            blup_Zlen_bestvar = compute_blups(Z, lmm1_temp, X_mat_with_len_bestvar, Y, info=None)
            plot_BLUPs(Y, y_resid_Zlen_bestvar, Z, Z, blup_Zlen_bestvar['u_hat'], all_freq, lr_temp, pval_temp, comps_temp['n'], comps_temp['m'], comps_temp, save_path=outfile.replace('.pdf', '.Zraw.ZrawSigmas.ML+BVInFixedEffects.pdf'))

            #--------------------------

            # # make bar chart of lmm0 vs lmm1 betas
            # plt.figure(figsize=(6, 5))
            # # sns.violinplot(x=genotypes, y=X_mat['vntr_mean_allele_length'], inner=None, color=".8")
            # sns.barplot(
            #     x=lmm0_temp.beta,
            #     y=lmm1_temp.beta,
            #     legend='brief',        # Keeps the legend manageable
            # )
            # plt.title("Comparison of Beta estimate values for fixed effects in\ncontrast model w/ ML + BV")
            # plt.xlabel('Fixed Effects from contrast model w/ ML and BV')
            # plt.ylabel('Fixed Effect Betas Values (Null VS Full Models)')
            # # Move legend outside if there are many unique VNTR genotypes
            # # plt.legend(title="VNTR Genotype", fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
            # plt.legend(title="VNTR Genotype", fontsize=7)
            # plt.grid(True)
            # plt.tight_layout()  # Adjust layout to make room for legend
            # plt.savefig(outfile.replace('.pdf', '.FixedEffectBetaComparisonNullVSFull.barchart.pdf'), bbox_inches='tight')
            # plt.close()

            # lmm0_temp and lmm1_temp have the same column labels (the names of the fixed effects) as X_mat_with_len_bestvar
            plot_df = pd.DataFrame({
                'Fixed Effect': X_mat_with_len_bestvar.columns,
                'Null Model (ML+BV in X)': lmm0_temp.beta,
                'Full Model (ML+BV in X)': lmm1_temp.beta,
            })

            # Reshape from "wide" to "long" format
            df_melted = plot_df.melt(id_vars='Fixed Effect', var_name='Model', value_name='Beta Value')

            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=df_melted,
                x='Fixed Effect', 
                y='Beta Value',
                hue='Model'       # This creates the "side-by-side" bars
            )

            plt.title("Comparison of Beta Estimates: Null vs Full Models")
            plt.xticks(rotation=45, ha='right') # Rotates labels if you have many effects
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            plt.savefig(outfile.replace('.pdf', '.FixedEffectBetaComparisonNullVSFullZraw.MLBVinXmat.barchart.pdf'))
            plt.close()

            # with Z_ortho # just running to make 'true' null model with identity matrix instead of 'best variants' in Z_best_var set to None
            Z_ortho = orthogonalize_Z_relative_to_X(Z, X_mat_with_len_bestvar)
            lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_len_bestvar, Z_ortho, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
            # y_resid_Zortho = (Y - X_mat @ lmm1_Zortho.beta).to_numpy()
            blup_Zortho_len_bestvar = compute_blups(Z_ortho, lmm1_temp, X_mat_with_len_bestvar, Y, info=None)
            # # this is plotting the same as Zraw dosage matrix and Zraw full model BLUPs but with y residuals from lmm0 with identity matrix
            # plot_BLUPs(Y, y_resid_Zortho, Z, Z, blup_Zraw['u_hat'], all_freq, lr_Zortho, pval_Zortho, comps_Zortho['n'], comps_Zortho['m'], comps_Zortho, save_path=outfile.replace('.pdf', '.Zraw.ZorthoSigmas.pdf'))
            
            # lmm0_temp and lmm1_temp have the same column labels (the names of the fixed effects) as X_mat_with_len_bestvar
            plot_df = pd.DataFrame({
                'Fixed Effect': X_mat_with_len_bestvar.columns,
                'Null Model': lmm0_temp.beta,  # this is Zortho
                'Full Model': lmm1_temp.beta,  # this is Zortho
            })
            # Reshape from "wide" to "long" format
            df_melted = plot_df.melt(id_vars='Fixed Effect', var_name='Model', value_name='Beta Value')
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=df_melted,
                x='Fixed Effect', 
                y='Beta Value',
                hue='Model'       # This creates the "side-by-side" bars
            )
            plt.title(f"Comparison of Beta Estimates: Null vs Full Models (pval={pval_temp})")
            plt.xticks(rotation=45, ha='right') # Rotates labels if you have many effects
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(outfile.replace('.pdf', '.FixedEffectBetaComparisonNullVSFullZortho.MLBVinXmat.barchart.pdf'))
            plt.close()

            # with Z_ortho # just running to make 'true' null model with identity matrix instead of 'best variants' in Z_best_var set to None
            Z_ortho = orthogonalize_Z_relative_to_X(Z, X_mat)
            lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat, Z_ortho, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
            # y_resid_Zortho = (Y - X_mat @ lmm1_Zortho.beta).to_numpy()
            blup_Zortho_len = compute_blups(Z_ortho, lmm1_temp, X_mat, Y, info=None)
            # # this is plotting the same as Zraw dosage matrix and Zraw full model BLUPs but with y residuals from lmm0 with identity matrix
            # plot_BLUPs(Y, y_resid_Zortho, Z, Z, blup_Zraw['u_hat'], all_freq, lr_Zortho, pval_Zortho, comps_Zortho['n'], comps_Zortho['m'], comps_Zortho, save_path=outfile.replace('.pdf', '.Zraw.ZorthoSigmas.pdf'))
            
            # lmm0_temp and lmm1_temp have the same column labels (the names of the fixed effects) as X_mat_with_len_bestvar
            plot_df = pd.DataFrame({
                'Fixed Effect': X_mat.columns,
                'Null Model': lmm0_temp.beta,  # this is Zortho
                'Full Model': lmm1_temp.beta,  # this is Zortho
            })
            # Reshape from "wide" to "long" format
            df_melted = plot_df.melt(id_vars='Fixed Effect', var_name='Model', value_name='Beta Value')
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=df_melted,
                x='Fixed Effect', 
                y='Beta Value',
                hue='Model'       # This creates the "side-by-side" bars
            )
            plt.title(f"Comparison of Beta Estimates: Null vs Full Models (pval={pval_temp})")
            plt.xticks(rotation=45, ha='right') # Rotates labels if you have many effects
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(outfile.replace('.pdf', '.FixedEffectBetaComparisonNullVSFullZortho.MLinXmat.barchart.pdf'))
            plt.close()

    if vntr_mean_length_values is not None:  #  and Z_best_var is None
        # add mean allele length to X_mat and fit lmm null and full with this new X_mat, do not add Z_best_var to null random effect matrix
        X_mat_with_len = X_mat.copy() # X_mat_with_len
        lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_len, Z, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
        plt.figure(figsize=(6, 5))
        plt.scatter(Y - X_mat_with_len @ lmm1_temp.beta, Y - X_mat_with_len @ lmm0_temp.beta)
        plt.title('Mean Allele Length in X_mat vs VNTR allele effects in LMM')
        plt.xlabel('Full Model Residuals (w/ Z_VNTR Matrix)')
        plt.ylabel('Null Model Residuals (w/ Identity Matrix)')
        plt.grid(True)
        plt.savefig(outfile.replace('.pdf', '.MeanAlleleLengthInXMat.pdf'))
        y_resid_Zlen = (Y - X_mat_with_len @ lmm0_temp.beta).to_numpy()  # use residuals from model with mean allele length as covariate in X_mat to compute BLUPs with Z_ortho and plot against BLUPs from Zraw full model
        blup_Zlen_nullbetas = compute_blups(Z, lmm0_temp, X_mat_with_len, Y, info=None)
        blup_Zlen = compute_blups(Z, lmm1_temp, X_mat_with_len, Y, info=None)
        plot_BLUPs(Y, y_resid_Zlen, Z, Z, blup_Zlen['u_hat'], all_freq, lr_temp, pval_temp, comps_temp['n'], comps_temp['m'], comps_temp, save_path=outfile.replace('.pdf', '.Zraw.ZrawSigmas.MLInFixedEffects.pdf'))
    
    if Z_best_var is not None or vntr_mean_length_values is not None:  # note there is OR in if statement
        X_mat_nolen_nobv = X_mat.copy()
        if vntr_mean_length_values is not None:
            # delete mean allele length from X_mat_with_bestvar to see if best variant genotype is tagging mean allele length
            X_mat_nolen_nobv = X_mat_nolen_nobv.drop(columns=['vntr_mean_allele_length']) if vntr_mean_length_values is not None else X_mat_nolen_nobv
            # X_mat_nolen_nobv = X_mat_nolen_nobv.drop(columns=['best_variant_genotype']) if Z_best_var is not None else X_mat_nolen_nobv
        # Compare BLUPs of base model without mean allele length or best variant genotypes in X_mat to BLUPs from X_mat containing mean allele length and best variant genotypes
        # for comparison of using Z (not centered, and could have mean-offset effects per allele) vs Zc (centered, so no mean-offset effects per allele)
        print("\nFitting LMM using raw Z (not centered)... without mean allele length or best variant genotypes in X_mat")
        lr_base, pval_base, comps_base, lmm0_base, lmm1_base = fit_lmm(Y, X_mat_nolen_nobv, Z, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
        blup_base_nullbetas = compute_blups(Z, lmm0_base, X_mat_nolen_nobv, Y, info=info)  # use betas from null model not full model
        blup_base = compute_blups(Z, lmm1_base, X_mat_nolen_nobv, Y, info=info)  # use betas from full model not null model

        # with Z_ortho base model fit without len or bestvar
        Z_ortho_base = orthogonalize_Z_relative_to_X(Z, X_mat_nolen_nobv)
        lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_nolen_nobv, Z_ortho_base, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
        # y_resid_Zortho = (Y - X_mat @ lmm1_Zortho.beta).to_numpy()
        blup_Zortho_base = compute_blups(Z_ortho_base, lmm1_temp, X_mat_nolen_nobv, Y, info=None)

        if Z_best_var is not None and vntr_mean_length_values is not None:
            # make three scatter plots
            plot_bv_ml_quadriptych(genotypes, vntr_genotypes, X_mat, Y, Z, blup_base['u_hat'], outfile)
            plot_bv_ml_VS_ge_triptych(genotypes, vntr_genotypes, X_mat, Y, Z, blup_base['u_hat'], outfile, gene_id, vntr_id)
            plot_bv_ml_triptych(genotypes, vntr_genotypes, X_mat, Y, outfile)
            if vntr_length_matrix is not None:
                # this plot section is fucking cursed
                common_inds_length = gene_expression.index.intersection(vntr_length.index)
                Y_length, X_length, X_mat_length, Z_length, _, _, _, sorted_alleles, Z_best_length_var, enough_data_length = prep_data(gene_expression, vntr_values, common_inds_length, covariates, n_common_inds, minsamples, kinship_matrix, None, vntr_length, no_rare_alleles=no_rare_alleles, collapse_rare_alleles=collapse_rare_alleles, permute_vntr_mean_lengths=permute_vntr_mean_lengths)
                lr_length, pval_length, comps_length, lmm0_length, lmm1_length = fit_lmm(Y_length, X_mat_length, Z_length, Z_best_var=Z_best_length_var, reml=reml, return_models=True, high_throughput=False)
                blup_length_nullbetas = compute_blups(Z_best_length_var, lmm0_length, X_mat_length, Y_length, info=None)
                blup_length = compute_blups(Z_best_length_var, lmm1_length, X_mat_length, Y_length, info=None)
                common_inds_length = Z.index.intersection(Z_best_length_var.index) # get the common individuals between gene expression and vntr_length after prep_data because prep_data may filter out some individuals due to missing data in gene expression or vntr_length, so take the intersection of gene expression index and vntr_length index to get the common individuals and subset to those for plotting
                print(f"common_inds_length length: {len(common_inds_length)}")
                # subset rows from Z and Z_best_var and Y and X_mat and blup_base to just the individuals in common_inds_length
                Z_plot = Z.loc[common_inds_length]  # it is not always a subset of individuals in gene_expression index because some individuals may have missing values in the VNTR length matrix, so take the intersection of gene_expression index and vntr_length index to get the common individuals and subset to those
                Z_best_var_plot = Z_best_var.loc[common_inds_length]
                Y_plot = Y.loc[common_inds_length]
                X_mat_plot = X_mat.loc[common_inds_length]
                print(list(blup_base["u_hat"].index[:5]))
                #blup_base_plot = blup_base['u_hat'].loc[common_inds_length]
                #blup_base_plot = blup_length["u_hat"][blup_length["u_hat"].index.isin(common_inds_length)]
                Z_length = Z_length.loc[common_inds_length]
                Z_best_length_var = Z_best_length_var.loc[common_inds_length]
                Y_length = Y_length.loc[common_inds_length]
                X_mat_length = X_mat_length.loc[common_inds_length]
                #blup_length_plot = blup_length['u_hat'].loc[common_inds_length]
                #blup_length_plot = blup_length["u_hat"][blup_length["u_hat"].index.isin(common_inds_length)]
                genotypes_plot = np.zeros(Z_best_var.shape[0])
                for col in Z_best_var_plot.columns:
                    genotypes_plot += Z_best_var_plot[col].values * int(col)
                vntr_genotypes_plot = []
                for i in range(Z.shape[0]):  # for each row
                    alleles = Z.columns[np.where(Z.iloc[i] > 0)[0]]  # get column names (allele number) when cell value > 0
                    vntr_genotype = f"{alleles[0]}/{alleles[1]}" if len(alleles) == 2 else f"{alleles[0]}/{alleles[0]}"  # there are always just two columns in Z_raw, one allele per column
                    vntr_genotypes_plot.append(vntr_genotype)
                vntr_genotypes_length = []
                for i in range(Z_best_length_var.shape[0]):  # for each row
                    alleles = Z_best_length_var.columns[np.where(Z_best_length_var.iloc[i] > 0)[0]]  # get column names (allele number) when cell value > 0
                    vntr_genotype_length = f"{alleles[0]}/{alleles[1]}" if len(alleles) == 2 else f"{alleles[0]}/{alleles[0]}"  # there are always just two columns in Z_raw, one allele per column
                    vntr_genotypes_length.append(vntr_genotype_length)
                print(f"{Z}\n\n{Z_length}\n\n{Z_best_length_var}")
                print(f"{vntr_genotypes_length}")

                print(f"BLUP values from length model: {blup_length['u_hat']}")
                plot_bv_ml_blup_bluplength_VS_ge_quadriptych(genotypes_plot, vntr_genotypes_plot, vntr_genotypes_length, X_mat_plot, X_mat_length, Y_plot, Y_length, Z_plot, Z_best_length_var, blup_base['u_hat'], blup_length['u_hat'], outfile, gene_id, vntr_id, best_variant_ID, vntr_length_ID, vntr_mean_length_ID)
                plot_bv_ml_blup_bluplength_VS_ge_quadriptych(genotypes_plot, vntr_genotypes_plot, vntr_genotypes_length, X_mat_plot, X_mat_length, Y_plot, Y_length, Z_plot, Z_best_length_var, blup_base['u_hat'], blup_length_nullbetas['u_hat'], outfile, gene_id, vntr_id, best_variant_ID, vntr_length_ID, vntr_mean_length_ID, LengthModelNullBetas=True)
                print('\n\n\n###MADE IT THROUGH THE BEST VNTR ALLELE Length PLOTTING###\n\n\n')
            elif vntr_length_matrix is None:
                print('\n\n\n###Skipping THE BEST VNTR ALLELE Length PLOTTING###\n\n\n')

        #######################################################
        # Determine which model was used for 'y' values
        if Z_best_var is not None and vntr_mean_length_values is not None:
            blup_contrast_nullbetas = blup_Zlen_bestvar_nullbetas
            blup_contrast = blup_Zlen_bestvar
        elif Z_best_var is not None and vntr_mean_length_values is None:
            blup_contrast_nullbetas = blup_Zbestvar_nullbetas
            blup_contrast = blup_Zbestvar
        elif Z_best_var is None and vntr_mean_length_values is not None:
            blup_contrast_nullbetas = blup_Zlen_nullbetas
            blup_contrast = blup_Zlen
        plotfilename = outfile.replace('.pdf', '.BLUPsWithoutLenBVvsWithLenBV.pdf')
        BLUP_vs_BLUP_scatter(Z, blup_base, blup_contrast, Z_best_var, vntr_mean_length_values, plotfilename)
        plotfilename = outfile.replace('.pdf', '.BLUPsWithoutLenBVvsWithLenBV.pdf')
        BLUP_vs_BLUP_scatter(Z, blup_base, blup_contrast, Z_best_var, vntr_mean_length_values, plotfilename)
        #######################################################
        def prep_data_for_dumbbell_plot(Z, blup_base, blup_Zlen_bestvar, blup_Zbestvar, blup_Zlen, blup_contrast, Z_best_var, vntr_mean_length_values):    
            # prep data for dumbell plot with full model betas
            if Z_best_var is not None and vntr_mean_length_values is not None:
                # plot all three contrast models (base vs mean allele length in X_mat, base vs best variant genotype in X_mat, and base vs both mean allele length and best variant genotype in X_mat) in a single figure with 3 scatter plots side by side with a shared x-axis of BLUPs from base model and y-axes of BLUPs from each contrast model, annotate points with allele numbers, and include correlation coefficient and p-value for each scatter plot
                df_blup = pd.DataFrame({
                    'Allele': Z.columns,
                    'Base_Model': blup_base['u_hat'].values,
                    'MeanLength+BestVariant': blup_Zlen_bestvar['u_hat'].values,
                    'BestVariant': blup_Zbestvar['u_hat'].values,
                    'MeanLength': blup_Zlen['u_hat'].values
                })
                df_blup_single_contrast = df_blup.copy()
                df_blup_single_contrast['ChangeMLBV'] = df_blup['MeanLength+BestVariant'] - df_blup['Base_Model']
                # make dumbell plot to show change in predicted BLUPs after contrast
                BLUP_vs_BLUP_dumbell(df_blup_single_contrast, save_path=outfile.replace('.pdf', '.BLUPs_Dumbbell_MLBVOnly.pdf'))
                df_blup['ChangeMLBV'] = df_blup['MeanLength+BestVariant'] - df_blup['Base_Model']
                df_blup['ChangeBV'] = df_blup['BestVariant'] - df_blup['Base_Model']
                df_blup['ChangeML'] = df_blup['MeanLength'] - df_blup['Base_Model']
            elif Z_best_var is not None and vntr_mean_length_values is None:
                # plot just best variant genotype contrast model vs base model
                df_blup = pd.DataFrame({
                    'Allele': Z.columns,
                    'Base_Model': blup_base['u_hat'].values,
                    'BestVariant': blup_contrast['u_hat'].values
                })
                # Add a 'Change' column for easier annotation
                df_blup['ChangeBV'] = df_blup['BestVariant'] - df_blup['Base_Model']
            elif Z_best_var is None and vntr_mean_length_values is not None:
                # plot just mean allele length contrast model vs base model
                df_blup = pd.DataFrame({
                    'Allele': Z.columns,
                    'Base_Model': blup_base['u_hat'].values,
                    'MeanLength': blup_contrast['u_hat'].values
                })
                # Add a 'Change' column for easier annotation
                df_blup['ChangeML'] = df_blup['MeanLength'] - df_blup['Base_Model']

            return df_blup
        
        # pass None to some variables passed to dumbell functions
        if Z_best_var is not None and vntr_mean_length_values is None:
            blup_Zlen_bestvar, blup_Zlen_bestvar_nullbetas, blup_Zlen, blup_Zlen_nullbetas = None, None, None, None
            blup_Zortho_len_bestvar, blup_Zortho_len_bestvar_nullbetas, blup_Zortho_len, blup_Zortho_len_nullbetas = None, None, None, None
        elif Z_best_var is None and vntr_mean_length_values is not None:
            blup_Zlen_bestvar, blup_Zlen_bestvar_nullbetas, blup_Zbestvar, blup_Zbestvar_nullbetas = None, None, None, None
            blup_Zortho_len_bestvar, blup_Zortho_len_bestvar_nullbetas, blup_Zortho_len, blup_Zortho_len_nullbetas = None, None, None, None

        # with Zortho # blup_contrast is wrong value here...
        df_blup = prep_data_for_dumbbell_plot(Z_ortho_base, blup_Zortho_base, blup_Zortho_len_bestvar, blup_Zortho_bestvar, blup_Zortho_len, blup_contrast, Z_best_var, vntr_mean_length_values)
        BLUP_vs_BLUP_dumbell(df_blup, save_path=outfile.replace('.pdf', '.BLUPs_Dumbbell_Zortho.pdf'))

        df_blup = prep_data_for_dumbbell_plot(Z, blup_base, blup_Zlen_bestvar, blup_Zbestvar, blup_Zlen, blup_contrast, Z_best_var, vntr_mean_length_values)
        df_blup_nullbetas = prep_data_for_dumbbell_plot(Z, blup_base_nullbetas, blup_Zlen_bestvar_nullbetas, blup_Zbestvar_nullbetas, blup_Zlen_nullbetas, blup_contrast_nullbetas, Z_best_var, vntr_mean_length_values)
        
        # make dumbell plot to show change in predicted BLUPs after contrast
        BLUP_vs_BLUP_dumbell(df_blup, save_path=outfile.replace('.pdf', '.BLUPs_Dumbbell.pdf'))
        BLUP_vs_BLUP_dumbell(df_blup_nullbetas, save_path=outfile.replace('.pdf', '.BLUPs_Dumbbell_NullBetas.pdf'))

        #######################################################
        # Create a figure with two subplots (1 row, 2 columns)
        plotfilename = outfile.replace('.pdf', '.BLUPs_with_Bar.pdf')
        BLUP_vs_BLUP_scatter_bar(Z, blup_base, blup_contrast, plotfilename)
        # plotfilename = outfile.replace('.pdf', '.BLUPs_with_Bar.NullBetas.pdf')
        # BLUP_vs_BLUP_scatter_bar(Z, blup_base_nullbetas, blup_contrast, plotfilename)
        #######################################################

        if isinstance(Znofreq, pd.DataFrame):
            # Determine which model was used for 'y' values
            if Z_best_var is not None and vntr_mean_length_values is not None:
                lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_len_bestvar, Znofreq, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
                blup_contrast = compute_blups(Znofreq, lmm0_temp, X_mat_with_len_bestvar, Y, info=None)
            elif Z_best_var is not None and vntr_mean_length_values is None:
                lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_bestvar, Znofreq, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
                blup_contrast = compute_blups(Znofreq, lmm0_temp, X_mat_with_bestvar, Y, info=None)
            elif Z_best_var is None and vntr_mean_length_values is not None:
                lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_len, Znofreq, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
                blup_contrast = compute_blups(Znofreq, lmm0_temp, X_mat_with_len, Y, info=None)
            lr_base, pval_base, comps_base, lmm0_base, lmm1_base = fit_lmm(Y, X_mat_nolen_nobv, Znofreq, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
            blup_base = compute_blups(Znofreq, lmm0_base, X_mat_nolen_nobv, Y, info=None)
            
            # make scatter plot with correlation and p-value of BLUPs from base model vs BLUPs from model with mean allele length and best variant genotypes in X_mat
            plotfilename = outfile.replace('.pdf', '.BLUPsWithoutLenBVvsWithLenBV.ZNoFreq.NullBetas.pdf')
            BLUP_vs_BLUP_scatter(Znofreq, blup_base, blup_contrast, Z_best_var, vntr_mean_length_values, plotfilename, allele_removed=allele_removed)
            #######################################################
            # Create a figure with two subplots (1 row, 2 columns)
            plotfilename = outfile.replace('.pdf', '.BLUPs_with_Bar.Znofreq.NullBetas.pdf')
            # add the removed allele back to blup dataframes for plotting with value 0, add row to u_hat with value 0 and row name allele removed
            blup_base['u_hat'].loc[allele_removed] = 0
            blup_contrast['u_hat'].loc[allele_removed] = 0
            BLUP_vs_BLUP_scatter_bar(Z, blup_base, blup_contrast, plotfilename)
            #######################################################

        if isinstance(Znoref, pd.DataFrame):
            if Z_best_var is not None and vntr_mean_length_values is not None:
                lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_len_bestvar, Znoref, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
                blup_contrast = compute_blups(Znoref, lmm0_temp, X_mat_with_len_bestvar, Y, info=None)
            elif Z_best_var is not None and vntr_mean_length_values is None:
                lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_bestvar, Znoref, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
                blup_contrast = compute_blups(Znoref, lmm0_temp, X_mat_with_bestvar, Y, info=None)
            elif vntr_mean_length_values is not None:
                lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat_with_len, Znoref, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
                blup_contrast = compute_blups(Znoref, lmm0_temp, X_mat_with_len, Y, info=None)
            lr_base, pval_base, comps_base, lmm0_base, lmm1_base = fit_lmm(Y, X_mat_nolen_nobv, Znoref, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
            blup_base = compute_blups(Znoref, lmm0_base, X_mat_nolen_nobv, Y, info=None)
            
            # make scatter plot with correlation and p-value of BLUPs from base model vs BLUPs from model with mean allele length and best variant genotypes in X_mat            plotfilename = outfile.replace('.pdf', '.BLUPsWithoutLenBVvsWithLenBV.ZNoFreq.pdf')
            plotfilename = outfile.replace('.pdf', '.BLUPsWithoutLenBVvsWithLenBV.Znoref.NullBetas.pdf')
            BLUP_vs_BLUP_scatter(Znoref, blup_base, blup_contrast, Z_best_var, vntr_mean_length_values, plotfilename, allele_removed=allele_removed_ref)
            #######################################################
            # Create a figure with two subplots (1 row, 2 columns)
            plotfilename = outfile.replace('.pdf', '.BLUPs_with_Bar.Znoref.NullBetas.pdf')
            blup_base['u_hat'].loc[allele_removed_ref] = 0
            blup_contrast['u_hat'].loc[allele_removed_ref] = 0
            BLUP_vs_BLUP_scatter_bar(Z, blup_base, blup_contrast, plotfilename)
            #######################################################

    # just running to make 'true' null model with identity matrix instead of 'best variants' in Z_best_var set to None
    lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat, Z, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
    plt.figure(figsize=(6, 5))
    plt.scatter(Y - X_mat @ lmm1_Zraw.beta, Y - X_mat @ lmm0_temp.beta)
    if vntr_id is not None and 'vntr_mean_allele_length' in X_mat.columns:
        for count, (fullresid, nullresid) in enumerate(zip((Y - X_mat @ lmm1_Zraw.beta).to_numpy(), (Y - X_mat @ lmm0_temp.beta).to_numpy())):
            plt.annotate(X_mat.iloc[count]['vntr_mean_allele_length'], (fullresid, nullresid))
    plt.title('Compare LMM Residuals: Full Model vs Null Model w/ Identity Matrix')
    plt.xlabel('Full Model Residuals')
    plt.ylabel('Null Model Residuals (w/ Identity Matrix)')
    plt.grid(True)
    plt.savefig(outfile.replace('.pdf', '.LMMFullResiduals_vs_TrueNullResiduals.pdf'))
    # plt.show()

    # with Z_ortho # just running to make 'true' null model with identity matrix instead of 'best variants' in Z_best_var set to None
    Z_ortho = orthogonalize_Z_relative_to_X(Z, X_mat)
    lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y, X_mat, Z_ortho, Z_best_var=None, reml=reml, return_models=True, high_throughput=False)
    lr_Zortho, pval_Zortho, comps_Zortho, lmm0_Zortho, lmm1_Zortho = fit_lmm(Y, X_mat, Z_ortho, Z_best_var=Z_best_var, reml=reml, return_models=True, high_throughput=False)
    plt.figure(figsize=(6, 5))
    plt.scatter(Y - X_mat @ lmm1_Zortho.beta, Y - X_mat @ lmm0_temp.beta)
    if vntr_id is not None and 'vntr_mean_allele_length' in X_mat.columns:
        for count, (fullresid, nullresid) in enumerate(zip((Y - X_mat @ lmm1_Zortho.beta).to_numpy(), (Y - X_mat @ lmm0_temp.beta).to_numpy())):
            plt.annotate(X_mat.iloc[count]['vntr_mean_allele_length'], (fullresid, nullresid))
    plt.title('Compare LMM Residuals: Full Model vs Null Model w/ Identity Matrix')
    plt.xlabel('Full Model Residuals')
    plt.ylabel('Null Model Residuals (w/ Identity Matrix)')
    plt.grid(True)
    plt.savefig(outfile.replace('.pdf', '.LMMFullResiduals.Zortho_vs_TrueNullResiduals.pdf'))
    # plt.show()
    y_resid_Zortho = (Y - X_mat @ lmm1_Zortho.beta).to_numpy()


    blup_Zortho = compute_blups(Z_ortho, lmm1_Zortho, X_mat, Y, info=None)
    # this is plotting the same as Zraw dosage matrix and Zraw full model BLUPs but with y residuals from lmm0 with identity matrix
    plot_BLUPs(Y, y_resid_Zortho, Z, Z, blup_Zraw['u_hat'], all_freq, lr_Zortho, pval_Zortho, comps_Zortho['n'], comps_Zortho['m'], comps_Zortho, save_path=outfile.replace('.pdf', '.Zraw.ZorthoSigmas.pdf'))

        # lmm0_temp and lmm1_temp have the same column labels (the names of the fixed effects) as X_mat_with_len_bestvar
    plot_df = pd.DataFrame({
        'Fixed Effect': X_mat.columns,
        'Null Model': lmm0_Zortho.beta,
        'Full Model': lmm1_Zortho.beta,
    })
    # Reshape from "wide" to "long" format
    df_melted = plot_df.melt(id_vars='Fixed Effect', var_name='Model', value_name='Beta Value')
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_melted,
        x='Fixed Effect', 
        y='Beta Value',
        hue='Model'       # This creates the "side-by-side" bars
    )
    plt.title("Comparison of Beta Estimates: Null vs Full Models")
    plt.xticks(rotation=45, ha='right') # Rotates labels if you have many effects
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(outfile.replace('.pdf', '.FixedEffectBetaComparisonNullVSFullZortho.barchart.pdf'))
    plt.close()

    # ---------------------------------
    if vntr_mean_length_matrix is not None:
        correlation = np.corrcoef((X_mat['vntr_mean_allele_length'], Z.values @ blup_Zraw['u_hat'].values))
        print(f"##################\n\nCorrelation between VNTR mean allele length and individual BLUPs from full LMM: r={correlation[0,1]:.4f}\n\n##################")
        plt.figure(figsize=(6, 5))
        plt.scatter(X_mat['vntr_mean_allele_length'], Z.values @ blup_Zraw['u_hat'].values)
        plt.title('Compare LMM Residuals: Full Model vs Null Model w/ Identity Matrix')
        plt.xlabel('Mean VNTR Allele Length')
        plt.ylabel('Individual BLUPs from Full LMM')
        plt.grid(True)
        plt.savefig(outfile.replace('.pdf', '.MeanAlleleLength_vs_IndividualBLUPs.pdf'))

        print(f"Plot gene expression vs. VNTR mean allele length for {gene_id} and {vntr_id}")
        # this finds association between gene expression and mean allele length of the VNTR
        # in other parts of this program mean allele length is included in X, used as a fixed effect, and therefore corrected for when testing VNTR genotype effects
        # but here we test mean allele length directly against gene expression
        Y_vs_meanlength, X_vs_meanlength, r2_vs_meanlength, beta_vs_meanlength, se_vs_meanlength, pvalue_vs_meanlength, model_vs_meanlength = perform_simplelinear(Y, vntr_mean_length_values.loc[Y.index], add_intercept=True)
        plot_simplelinear(
            Y_simple=Y_vs_meanlength,
            X_simple=X_vs_meanlength,
            r2_simple=r2_vs_meanlength,
            beta_simple=beta_vs_meanlength,
            se_simple=se_vs_meanlength,
            pvalue_simple=pvalue_vs_meanlength,
            model_simple=model_vs_meanlength,
            save_path=outfile.replace('.pdf', '.SimpleLinear.GeneExpr.VS.VNTRMeanAlleleLength.pdf'),
            title=f'Simple Linear Regression: Gene Expression vs. VNTR Mean Allele Length\nGene: {gene_id} VNTR: {vntr_id}'
        )
        # here plot mean allele length vs. individual BLUPs from the main LMM analysis # variable names are not ideal but reuse the same function
        # we have to fit LMM without mean allele length in X to get individual BLUPs that are not corrected for mean allele length
        Y_temp, X_temp, X_mat_temp, Z_temp, _, _, _, _, Z_best_var_temp, enough_data_temp = prep_data(gene_expression, vntr_values, common_inds, covariates, n_common_inds, minsamples, kinship_matrix, None, best_variants, no_rare_alleles=no_rare_alleles, collapse_rare_alleles=collapse_rare_alleles, permute_vntr_mean_lengths=permute_vntr_mean_lengths)  # fixed vntr_mean_length_values=None to exclude from X
        lr_temp, pval_temp, comps_temp, lmm0_temp, lmm1_temp = fit_lmm(Y_temp, X_mat_temp, Z_temp, Z_best_var_temp, reml=reml, return_models=True, high_throughput=False)
        blup_temp = compute_blups(Z_temp, lmm1_temp, X_mat_temp, Y_temp, info=info)
        individual_blues_temp = Z_temp @ blup_temp['u_hat'].values
        Y_vs_meanlength, X_vs_meanlength, r2_vs_meanlength, beta_vs_meanlength, se_vs_meanlength, pvalue_vs_meanlength, model_vs_meanlength = perform_simplelinear(vntr_mean_length_values.loc[Y.index], individual_blues_temp, add_intercept=True)
        plot_simplelinear(
            Y_simple=Y_vs_meanlength,
            X_simple=X_vs_meanlength,
            r2_simple=r2_vs_meanlength,
            beta_simple=beta_vs_meanlength,
            se_simple=se_vs_meanlength,
            pvalue_simple=pvalue_vs_meanlength,
            model_simple=model_vs_meanlength,
            save_path=outfile.replace('.pdf', '.SimpleLinear.VNTRMeanAlleleLength.VS.IndividualBLUPs.pdf'),
            title=f'Simple Linear Regression: VNTR Mean Allele Length VS Individual BLUPs\nGene: {gene_id} VNTR: {vntr_id}'
        )
    # ---------------------------------
    # TEST #
    # TEST #

    # compare uhat_OLS (no ridge penalty) to uhat_BLUP (with ridge penalty)
    compare_uhat_NoRidge_to_uhat_BLUP(ols, blup['u_hat'], save_path=outfile.replace('.pdf', '.OLSEffectsvsBLUPEffects.Zc.pdf'))
    compare_uhat_NoRidge_to_uhat_BLUP(ols_Zraw, blup_Zraw['u_hat'], save_path=outfile.replace('.pdf', '.OLSEffectsvsBLUPEffects.Zraw.pdf'))
    # make allele penalty (shrinkage) summary plots
    summarize_shrinkage(Zc, comps['sigma_g2'], comps['sigma_e2'], y_resid, save_path=outfile.replace('.pdf', '.AllelePenalties.Zc.SigmaZc.pdf'), top_k=10)
    summarize_shrinkage(Z, comps_Zraw['sigma_g2'], comps_Zraw['sigma_e2'], y_resid_Zraw, save_path=outfile.replace('.pdf', '.AllelePenalties.Zraw.SigmaZraw.pdf'), top_k=10)
    
    if Z_best_var is not None:
        lr_best, pval_best, comps_best, lmm0_best, lmm1_best = fit_lmm(Y, X_mat, Z_best_var, None, reml=reml, return_models=True, high_throughput=False)  # fitting best variant as Z alone, to see association of biallelic best variant (NOT VNTR)
        y_resid_Zbest_nullbetas = (Y - X_mat @ lmm0_best.beta).to_numpy()
        y_resid_Zbest = (Y - X_mat @ lmm1_best.beta).to_numpy()
        blup_Zbest_nullbetas = compute_blups(Z_best_var, lmm0_best, X_mat, Y, info=None)
        blup_Zbest = compute_blups(Z_best_var, lmm1_best, X_mat, Y, info=None)
        ols_Zbest = compute_ols_effects(Z_best_var, lmm1_best, Y, X_mat, info=None)

        # n_best = Z_best_var.shape[0]
        # m_best = Z_best_var.shape[1]
        total_alleles = Z_best_var.sum().sum()  # Total dosage across all alleles
        allele_freqs_best = (Z_best_var.sum(axis=0) / total_alleles).to_dict()

        # comps_Zbest = comps_Zraw.copy()
        # comps_Zbest['n'] = n_best
        # comps_Zbest['m'] = m_best
        # comps_Zbest['sigma_g2'] = lmm0_Zraw.v0
        # comps_Zbest['sigma_e2'] = lmm0_Zraw.v1
        plot_heatmaps(Z_best_var, save_path=outfile.replace('.pdf', '.Heatmaps.Zbest.pdf'))
        plot_BLUPs(Y, y_resid_Zbest_nullbetas, Z_best_var, Z_best_var, blup_Zbest_nullbetas['u_hat'], allele_freqs_best, lr_best, pval_best, comps_best['n'], comps_best['m'], comps_best, save_path=outfile.replace('.pdf', f'.Zbest.ZbestSigmas.{best_variant_ID}.NullBetas.pdf'))
        plot_BLUPs(Y, y_resid_Zbest, Z_best_var, Z_best_var, blup_Zbest['u_hat'], allele_freqs_best, lr_best, pval_best, comps_best['n'], comps_best['m'], comps_best, save_path=outfile.replace('.pdf', f'.Zbest.ZbestSigmas.{best_variant_ID}.pdf'))
        compare_uhat_NoRidge_to_uhat_BLUP(ols_Zbest, blup_Zbest['u_hat'], save_path=outfile.replace('.pdf', '.OLSEffectsvsBLUPEffects.Zbest.pdf'))
        validate_genos = recreate_genotypes_from_Z(Z_best_var)
        validate_geneexp = Y.to_dict()
        snp_individuals = Z_best_var.index.tolist()
        y_individuals = Y.index.tolist()
        write_out_dict(validate_genos, outfile.replace('.pdf', f'.Genotypes_FromZbest.{best_variant_ID}.txt'), snp_individuals, best_variant_ID)
        write_out_dict(validate_geneexp, outfile.replace('.pdf', f'.GeneExpression_FromInput.{gene_id}.txt'), y_individuals, gene_id)


    if collapse_rare_alleles == True:
        export_matrix(Z_Zfull, outfile.replace('.pdf', '.Matrix.Zfull_Raw.txt'))
        export_matrix(Zc_Zfull, outfile.replace('.pdf', '.Matrix.Zcfull_centered.txt'))

        plot_heatmaps(Z_Zfull, save_path=outfile.replace('.pdf', '.Heatmaps.Zfull.pdf'))
        plot_heatmaps(Zc_Zfull, save_path=outfile.replace('.pdf', '.Heatmaps.Zcfull.pdf'))
        
        plot_BLUPs(Y, y_resid, Z, Zc_Zfull, blup_Zcfull_w_ZcSigmas['u_hat'], all_freq_Zfull, lr, pval, n_indiv, n_allele, comps, save_path=outfile.replace('.pdf', '.Zfull.ZcSigmas.pdf'))
        plot_BLUPs(Y, y_resid, Z, Zc_Zfull, blup_Zcfull['u_hat'], all_freq_Zfull, lr_Zfull, pval_Zfull, n_indiv_Zfull, n_allele_Zfull, comps_Zfull, save_path=outfile.replace('.pdf', '.Zfull.ZfullSigmas.pdf'))
        
        compare_uhat_NoRidge_to_uhat_BLUP(ols_Zfull, blup_Zcfull_w_ZcSigmas['u_hat'], save_path=outfile.replace('.pdf', '.OLSEffectsvsBLUPEffects.ZfullwZcSigmas.pdf'))
        compare_uhat_NoRidge_to_uhat_BLUP(ols_Zfull, blup_Zcfull['u_hat'], save_path=outfile.replace('.pdf', '.OLSEffectsvsBLUPEffects.Zfull.pdf'))
        
        summarize_shrinkage(Zc_Zfull, comps_Zfull['sigma_g2'], comps_Zfull['sigma_e2'], y_resid, save_path=outfile.replace('.pdf', '.AllelePenalties.Zfull.SigmaZfull.pdf'), top_k=10)
        # this one swaps in sigmas form Zc model, but uses Zfull for allele dosage
        summarize_shrinkage(Zc_Zfull, comps['sigma_g2'], comps['sigma_e2'], y_resid, save_path=outfile.replace('.pdf', '.AllelePenalties.Zfull.SigmaZc.pdf'), top_k=10)
        


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


def find_cis_vntrs(expression_matrix, vntr_matrix, window=1000000, chr=None):
    gene_vntr_dict = {}
    
    for _, gene in expression_matrix.iterrows():
        gene_chr, gene_start, gene_end, gene_strand = gene["#chr"], gene["start"], gene["end"], gene["strand"]
        if chr == None:  # if chr not specified, gather all cis-vntrs
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
        elif (chr is not None) and (gene_chr == chr):  # if chr specified, only gather cis-vntrs on that chromosome
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
        else:
            pass

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


def sort_matrices(expression_matrix, vntr_matrix, covariates, allcovariates, kinship_matrix, vntr_mean_length_matrix, best_variant_matrix, vntr_length_matrix, best_vntr_length_matrix, best_vntr_mean_length_matrix):
    print('Sorting matrices: gene_expression, vntr_values, if/and covariates, & all covariates, if/and kinship')

    # Extract individual IDs (columns for gene expression, VNTR, and covariates)
    # Preserve the first 6 columns (metadata columns)
    # metadata cols '#chr', 'start', 'end', 'pid', 'gid', 'strand
    metadata_cols = expression_matrix.columns[:6]
    print(f'metadata cols {metadata_cols}')
    # exlude first 6 columns
    gene_inds = expression_matrix.columns[6:]
    vntr_inds = vntr_matrix.columns[6:]
    cov_inds = covariates.columns if covariates is not None else []
    kinship_inds = kinship_matrix.columns if kinship_matrix is not None else []
    vntr_mean_length_inds = vntr_mean_length_matrix.columns[6:] if vntr_mean_length_matrix is not None else []
    best_variant_inds = best_variant_matrix.columns[6:] if best_variant_matrix is not None else []
    vntr_length_inds = vntr_length_matrix.columns[6:] if vntr_length_matrix is not None else []
    best_vntr_mean_length_inds = best_vntr_mean_length_matrix.columns[6:] if best_vntr_mean_length_matrix is not None else []
    best_vntr_length_inds = best_vntr_length_matrix.columns[6:] if best_vntr_length_matrix is not None else []

    # Find common individuals across all available datasets
    common_inds = set(gene_inds) & set(vntr_inds)
    if covariates is not None:
        common_inds &= set(cov_inds)
    if kinship_matrix is not None:
        common_inds &= set(kinship_inds)
        # print(f"common individuals: {common_inds}")
    if vntr_mean_length_matrix is not None:
        common_inds &= set(vntr_mean_length_inds)
    if best_variant_matrix is not None:
        common_inds &= set(best_variant_inds)
    if vntr_length_matrix is not None:
        common_inds &= set(vntr_length_inds)
    if best_vntr_mean_length_matrix is not None:
        common_inds &= set(best_vntr_mean_length_inds)
    if best_vntr_length_matrix is not None:
        common_inds &= set(best_vntr_length_inds)
    print(f'\nlen(common individuals){len(common_inds)}')
    print(f"Common individuals: {common_inds}")

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
        print('Kinship matrix dimensions:')
        print(f"{kinship_matrix.shape}")
        kinship_matrix = kinship_matrix.loc[sorted_inds, sorted_inds]
        print('head(kinship_matrix)')
        print(f'Kinship matrix dimensions: {kinship_matrix.shape}')
        print(kinship_matrix.iloc[:10,:10].to_markdown())

    # Align Mean VNTR allele length matrix
    if vntr_mean_length_matrix is not None:
        vntr_mean_length_matrix = vntr_mean_length_matrix.loc[:, sorted_cols]  # .dropna()
        print('head(vntr_mean_length_matrix)')
        print(f'vntr_mean_length_matrix matrix dimensions: {vntr_mean_length_matrix.shape}')
        print(vntr_mean_length_matrix.iloc[:10,:10].to_markdown())

    # Align Best VNTR variant matrix
    if best_variant_matrix is not None:
        best_variant_matrix = best_variant_matrix.loc[:, sorted_cols]  # .dropna()
        print('head(best_variant_matrix)')
        print(f'best_variant_matrix matrix dimensions: {best_variant_matrix.shape}')
        print(best_variant_matrix.iloc[:10,:10].to_markdown())
        # Align Best VNTR variant matrix
    
    # Alignt VNTR individual allele length matrix
    if vntr_length_matrix is not None:
        vntr_length_matrix = vntr_length_matrix.loc[:, sorted_cols]  # .dropna()
        print('head(vntr_length_matrix)')
        print(f'vntr_length_matrix matrix dimensions: {vntr_length_matrix.shape}')
        print(vntr_length_matrix.iloc[:10,:10].to_markdown())

    # Alignt VNTR individual best allele length matrix
    if best_vntr_length_matrix is not None:
        best_vntr_length_matrix = best_vntr_length_matrix.loc[:, sorted_cols]  # .dropna()
        print('head(best_vntr_length_matrix)')
        print(f'best_vntr_length_matrix matrix dimensions: {best_vntr_length_matrix.shape}')
        print(best_vntr_length_matrix.iloc[:10,:10].to_markdown())

    # Align Mean VNTR best allele length matrix
    if best_vntr_mean_length_matrix is not None:
        best_vntr_mean_length_matrix = best_vntr_mean_length_matrix.loc[:, sorted_cols]  # .dropna()
        print('head(best_vntr_mean_length_matrix)')
        print(f'best_vntr_mean_length_matrix matrix dimensions: {best_vntr_mean_length_matrix.shape}')
        print(best_vntr_mean_length_matrix.iloc[:10,:10].to_markdown())

    print('Finished sorting')

    return gene_expression, vntr_values, covariates, allcovariates, common_inds, kinship_matrix, vntr_mean_length_matrix, best_variant_matrix, vntr_length_matrix, best_vntr_length_matrix, best_vntr_mean_length_matrix


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


# def load_data(gxpfile, vntrfile, covfile, kinfile=None, vntrmeanlengthfile=None, bestvarfile=None, desired_covariates=None, geneid=None):
#     # input data assumed to be in relative directory ./inputs/inputfile.bed
#     expression_matrix = pd.read_csv(gxpfile, sep="\t", dtype={"#chr": str, "pid": str, "gid": str, "strand":str})
#     expression_matrix.set_index("gid", drop=False, inplace=True)
# 
#     # Read just the header
#     with open(vntrfile, 'r') as f:
#         header = f.readline().strip().split('\t')
#     # Define dtype for known columns
#     dtype_dict = {"#chr": str, "pid": str, "gid": str, "strand": str}
#     # Add all columns from index 6 onwards as strings
#     for col in header[6:]:
#         dtype_dict[col] = str
#     vntr_matrix = pd.read_csv(vntrfile, sep="\t", dtype=dtype_dict)
#     vntr_matrix.set_index("gid", drop=False, inplace=True)


def load_data(gxpfile, vntrfile, covfile, kinfile=None, vntrmeanlengthfile=None, bestvarfile=None, vntrlengthfile=None, bestvntrlengthfile=None, bestvntrmeanallelelengthfile=None,desired_covariates=None, geneid=None):

    # --- Expression matrix ---
    expr_dtype = {"#chr": str, "pid": str, "gid": str, "strand": str}
    expression_matrix = pd.read_csv(gxpfile, sep="\t", dtype=expr_dtype)
    if geneid is not None:
        expression_matrix = expression_matrix.loc[expression_matrix["gid"] == geneid]
        if expression_matrix.empty: raise ValueError(f"Gene {geneid} not found.")
        expr_chr = expression_matrix["#chr"].iat[0]
    else:
        expr_chr = None
    expression_matrix.set_index("pid", drop=False, inplace=True)

    # --- VNTR dtypes ---
    with open(vntrfile) as f: header = f.readline().strip().split("\t")
    dtype_dict = {c: str for c in ["#chr","pid","gid","strand"] + header[6:]}

    # --- VNTR matrix ---
    if expr_chr is not None:
        chunks = []
        for chunk in pd.read_csv(vntrfile, sep="\t", dtype=dtype_dict, chunksize=50000):
            sub = chunk.loc[chunk["#chr"] == expr_chr]
            if not sub.empty: chunks.append(sub)
        vntr_matrix = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=header)
    else:
        vntr_matrix = pd.read_csv(vntrfile, sep="\t", dtype=dtype_dict)

    vntr_matrix.set_index("pid", drop=False, inplace=True)

    # --- Covariate matrix ---
    if covfile is not None and os.path.exists(covfile):
        print(f"Reading covariate matrix: {covfile}")
    try:
        allcovariates = pd.read_csv(covfile, index_col=0, sep="\t") # , dtype={"ID": str}
        if desired_covariates:
            covariates = filter_covariates(allcovariates, desired_covariates)  # immediately filter the covariates to only the desired ones
    except FileNotFoundError:
        covariates = None
    
    # --- Kinship matrix (NOT USED) ---
    if kinfile is not None and os.path.exists(kinfile):
        print(f"Reading kinship matrix: {kinfile}")
        kinship_matrix = pd.read_csv(kinfile, index_col=0, sep="\t")  # , dtype=str
    else:
        kinship_matrix = None

    # --- VNTR mean allele length matrix (or another continuous variable that has one value per individual that will be added as a fixed effect to null and full model to remove the variation in gene expression) ---
    if vntrmeanlengthfile is not None and os.path.exists(vntrmeanlengthfile):
        print(f"Reading mean VNTR allele lengths: {vntrmeanlengthfile}")
        # Read just the header
        with open(vntrmeanlengthfile, 'r') as f:
            header = f.readline().strip().split('\t')
        # Define dtype for known columns
        dtype_dict = {"#chr": str, "pid": str, "gid": str, "strand": str}
        # Add all columns from index 6 onwards as strings
        for col in header[6:]:
            dtype_dict[col] = str
        vntr_mean_length_matrix = pd.read_csv(vntrmeanlengthfile, sep="\t", dtype=dtype_dict)
        vntr_mean_length_matrix.set_index("pid", drop=False, inplace=True)
    else:
        vntr_mean_length_matrix = None
    
    # gid as index
    # --- Best varaint matrix (contains genotypes of the best known variant associated with a gene. Use this to contrast VNTRs to these genotypes.) ---
    if bestvarfile is not None and os.path.exists(bestvarfile):
        print(f"Reading best variant genotypes: {bestvarfile}")
        # Read just the header
        with open(bestvarfile, 'r') as f:
            header = f.readline().strip().split('\t')
        # Define dtype for known columns
        dtype_dict = {"#chr": str, "pid": str, "gid": str, "strand": str}
        # Add all columns from index 6 onwards as strings
        for col in header[6:]:
            dtype_dict[col] = str
        best_variant_matrix = pd.read_csv(bestvarfile, sep="\t", dtype=dtype_dict)
        best_variant_matrix.set_index("gid", drop=False, inplace=True)  # gene ids are unique
    else:
        best_variant_matrix = None

    # --- vntr length matrix (contains genotypes of allele lengths (ex 156/302). Use this for plotting.) ---
    if vntrlengthfile is not None and os.path.exists(vntrlengthfile):
        print(f"Reading allele length genotypes: {vntrlengthfile}")
        # Read just the header
        with open(vntrlengthfile, 'r') as f:
            header = f.readline().strip().split('\t')
        # Define dtype for known columns
        dtype_dict = {"#chr": str, "pid": str, "gid": str, "strand": str}
        # Add all columns from index 6 onwards as strings
        for col in header[6:]:
            dtype_dict[col] = str
        vntr_length_matrix = pd.read_csv(vntrlengthfile, sep="\t", dtype=dtype_dict)
        vntr_length_matrix.set_index("pid", drop=False, inplace=True)  # gene ids are unique
    else:
        vntr_length_matrix = None

    # gid as index
    # bestvntrlengthfile --- best vntr length matrix (contains genotypes of allele lengths (ex 156/302). Use this for plotting.) ---
    if bestvntrlengthfile is not None and os.path.exists(bestvntrlengthfile):
        print(f"Reading best allele length genotypes: {bestvntrlengthfile}")
        # Read just the header
        with open(bestvntrlengthfile, 'r') as f:
            header = f.readline().strip().split('\t')
        # Define dtype for known columns
        dtype_dict = {"#chr": str, "pid": str, "gid": str, "strand": str}
        # Add all columns from index 6 onwards as strings
        for col in header[6:]:
            dtype_dict[col] = str
        best_vntr_length_matrix = pd.read_csv(bestvntrlengthfile, sep="\t", dtype=dtype_dict)
        best_vntr_length_matrix.set_index("gid", drop=False, inplace=True)  # gene ids are unique
    else:
        best_vntr_length_matrix = None

    # gid as index
    # bestvntrmeanallelelengthfile
    if bestvntrmeanallelelengthfile is not None and os.path.exists(bestvntrmeanallelelengthfile):
        print(f"Reading best mean VNTR allele lengths: {bestvntrmeanallelelengthfile}")
        # Read just the header
        with open(bestvntrmeanallelelengthfile, 'r') as f:
            header = f.readline().strip().split('\t')
        # Define dtype for known columns
        dtype_dict = {"#chr": str, "pid": str, "gid": str, "strand": str}
        # Add all columns from index 6 onwards as strings
        for col in header[6:]:
            dtype_dict[col] = str
        best_vntr_mean_length_matrix = pd.read_csv(bestvntrmeanallelelengthfile, sep="\t", dtype=dtype_dict)
        best_vntr_mean_length_matrix.set_index("gid", drop=False, inplace=True)
    else:
        best_vntr_mean_length_matrix = None

    #print(f"Kinship matrix: {kinship_matrix}")

    # Convert numerical columns to float
    # expression_matrix = expression_matrix.apply(pd.to_numeric, errors='coerce')
    # vntr_matrix = vntr_matrix.apply(pd.to_numeric, errors='coerce')
    # if covariates is not None:
    #     allcovariates = allcovariates.apply(pd.to_numeric, errors='coerce')
    #     covariates = covariates.apply(pd.to_numeric, errors='coerce')
    # if kinship_matrix is not None:
    #     kinship_matrix = kinship_matrix.apply(pd.to_numeric, errors='coerce')

    # sort each matrix to ensure columns (individuals) are represented in the same order
    expression_matrix, vntr_matrix, covariates, allcovariates, common_inds, kinship_matrix, vntr_mean_length_matrix, best_variant_matrix, vntr_length_matrix, best_vntr_length_matrix, best_vntr_mean_length_matrix = sort_matrices(expression_matrix, vntr_matrix, covariates, allcovariates, kinship_matrix, vntr_mean_length_matrix, best_variant_matrix, vntr_length_matrix, best_vntr_length_matrix, best_vntr_mean_length_matrix)

    return expression_matrix, vntr_matrix, covariates, allcovariates, common_inds, kinship_matrix, vntr_mean_length_matrix, best_variant_matrix, vntr_length_matrix, best_vntr_length_matrix, best_vntr_mean_length_matrix


import os

def build_output_dir(args):
    """
    Construct output directory path based on analysis arguments.
    """

    # If user explicitly provides an output directory, respect it
    if args.outdir:
        return os.path.join(args.workingdir, args.outdir)

    # Gene-specific cis-eQTL analysis
    if args.geneid and not args.vntrid:
        return os.path.join(args.workingdir, f"output_{args.geneid}")

    parts = []

    # Base Z matrix
    parts.append("Zraw" if args.Zraw else "Zc")

    # Rare allele handling
    if args.collapserarealleles:
        parts.append("CollapseRareAlleles")
    elif args.norarealleles:
        parts.append("NoRareAlleles")

    # Allele removal strategies
    if args.removereferenceallele:
        parts.append("RemoveReference")
    elif args.removemostfrequentallele:
        parts.append("RemoveMostFrequent")

    # Fixed-effect adjustments
    if args.vntrmeanallelelengthfile:
        parts.append("MinusMeanAlleleLength")

    # Best variant conditioning
    if args.bestvariantfile:
        parts.append(args.bestvariantfile.split('.')[0])  # parts.append("BestVariant")

    # Model fitting method
    if args.reml:
        parts.append("REML")
    else:
        parts.append("ML")

    # Permute VNTR mean lengths
    if args.permutevntrmeanlengths:
        parts.append("PermutedVNTRMeanLengths")

    dirname = "output_" + "_".join(parts)

    return os.path.join(args.workingdir, dirname)


def read_matrix(matrix_path, target_pid):
    with open(matrix_path) as f:
        header = f.readline().strip().split("\t")
        sample_ids = header[6:]  # after chr,start,end,pid,gid,strand

        for line in f:
            parts = line.strip().split("\t")
            pid = parts[3]
            if pid == target_pid:
                gts = parts[6:]
                return dict(zip(sample_ids, gts))
    return None


def validate_args(args):
    if args.norarealleles and args.collapserarealleles:
        raise ValueError("--norarealleles and --collapserarealleles are mutually exclusive")

    if args.removereferenceallele and args.removemostfrequentallele:
        raise ValueError("--removereferenceallele and --removemostfrequentallele are mutually exclusive")
    
    if args.vntrallelelengthfile and args.bestvntrallelelengthfile:
        raise ValueError("--vntrallelelengthfile and --bestvntrallelelengthfile are mutually exclusive")
    
    if args.vntrmeanallelelengthfile and args.bestvntrmeanallelelengthfile:
        raise ValueError("--vntrmeanallelelengthfile and --bestvntrmeanallelelengthfile are mutually exclusive")


def main():
    '''
    Usage:
    # to exclude rare alleles, require a minimum of 80 samples to present a result, use Zraw (un-normalized) matrix for analysis, specify number of threads for parallelization of cis eQTL analyses, specify chromosome name to anlayze one chromosome, specify window size for 2000000 for +- 1Mb window from gene start site.
    # specify desired covariates from the covariate file:
    python Model_eVNTRs.py --window 2000000 \
    --chromosome 3 \
    --geneexpfile TPM_gene_testis_FILTERED_UNNORM.tsv \
    --vntrfile TRDB_MatrixGenotypes.txt \
    --covarfile RNA_cov_testis.txt \
    --Zraw \
    --norarealleles \
    --threads 8 \
    --minsamples 80 \
    --desired_covs PC1 PC2 PC3 PC4 PC5 rin_testis age genotypePC3 genotypePC4 genotypePC5 genotypePC6 genotypePC7 \
    --workingdir /path/to/workingdir \
    --outdir output_Zraw_NoRareAlleles_Minsamples80_ML
    
    # perform cis-eVNTR_LS analysis excluding rare alleles on non-cetered-nomrmalized dosage matrix --Zraw
    python LM_BLUP_byChrom_mp.py --window "${WIN}" --chromosome "${CHROM}" --genexpfile TPM_gene_testis_FILTERED_UNNORM.tsv --vntrfile TRDB_MatrixGenotypes.txt --covarfile RNA_cov_testis.txt --desired_covs "${COVARIATES[@]}" --workingdir "${WORKDIR}" --norarealleles --Zraw --threads "${THREADS}" --minsamples 80 --outdir output_Zraw_NoRareAlleles_Minsamples80_ML
    # perform allele length cis-eVNTR_L association analysis
    python LM_BLUP_byChrom_mp.py --window "${WIN}" --chromosome "${CHROM}" --genexpfile TPM_gene_testis_FILTERED_UNNORM.tsv --vntrfile TRDB_MatrixAlleleLengths.txt --covarfile RNA_cov_testis.txt --desired_covs "${COVARIATES[@]}" --workingdir "${WORKDIR}" --norarealleles --Zraw --threads "${THREADS}" --minsamples 80 --outdir output_Zraw_NoRareAlleles_AlleleLength_Minsamples80_ML
    # perform best SNP/INDEL cis-eQTL_SNPINDEL association analysis
    python LM_BLUP_byChrom_mp.py --window "${WIN}" --chromosome "${CHROM}" --genexpfile TPM_gene_testis_FILTERED_UNNORM.tsv --vntrfile final_filtered.all.MatrixGenotypes.txt --covarfile RNA_cov_testis.txt --desired_covs "${COVARIATES[@]}" --workingdir "${WORKDIR}" --norarealleles --Zraw --threads "${THREADS}" --minsamples 80 --outdir output_Zraw_NoRareAlleles_SNPINDEL_Minsamples80_ML
    # contrast against best mean allele length result
    python LM_BLUP_byChrom_mp.py --window "${WIN}" --chromosome "${CHROM}" --genexpfile TPM_gene_testis_FILTERED_UNNORM.tsv --vntrfile TRDB_MatrixGenotypes.txt --covarfile RNA_cov_testis.txt --bestvntrmeanallelelengthfile BestVariantperGene.VNTRMeanAlleleLength.txt --desired_covs "${COVARIATES[@]}" --workingdir "${WORKDIR}" --norarealleles --Zraw --threads "${THREADS}" --minsamples 80 --outdir output_Zraw_NoRareAlleles_MinusBestMeanAlleleLength_Minsamples80_ML
    # contrast SNPINDEL best variant against mean allele length result
    python LM_BLUP_byChrom_mp.py --window "${WIN}" --chromosome "${CHROM}" --genexpfile TPM_gene_testis_FILTERED_UNNORM.tsv --vntrfile TRDB_MatrixGenotypes.txt --covarfile RNA_cov_testis.txt --bestvariantfile BestVariantperGene.MyModel.txt --desired_covs "${COVARIATES[@]}" --workingdir "${WORKDIR}" --norarealleles --Zraw --threads "${THREADS}" --minsamples 80 --outdir output_Zraw_NoRareAlleles_MinusBestVariantMyModel_Minsamples80_ML

    workingdir should contain ./inputs directory which contains the input files. Output will be written to workingdir/output_Zraw_NoRareAlleles_Minsamples80_ML

    ####### Input File Formats and arguments ########
    # --genexpfile: this file should be in ./inputs directory
    # should be a tab separated file with columns:
    # pid and gid are just the gene ids
    chr start end pid gid strand indiv1 indiv2 indiv3...
    1 283122 283123 LOC101903639 LOC101903639 - 2.60919 2.89257 1.76162 ...

    # --vntrfile: this file should be in ./inputs directory
    # should be a tab separated file with columns:
    # pid and gid are just the VNTR ids
    chr  start  end  pid  gid  strand  indiv1  indiv2
    1  664472  664472  1.664472.664472  1.664472.664472  +  0/1  0/1  3/5 ...

    # --covarfile: this file should be in ./inputs directory
    # should be a tab separated file with columns:
    ID  indiv1  indiv2  indiv3...
    PC1	 -85.2591870	 102.0774854	 -68.6080591
    PC2 ...
    rin_testis ...
    age ...
    genotypePC3 ...

    # --vntrmeanallelelengthfile: this file should be in ./inputs directory
    # this is used to perform a contrast model against the --vntrfile associated to --geneexpfile. Added as a fixed effect covariate (to remove mean allele length effect on dependent variable (gene expression))
    # this should be in the same format as --vntrfile, but instead of genotypes, it contains integer values representing the mean allele length for each individual (columns) for each VNTR (rows)

    # --bestvariantfile: this file should be in ./inputs directory
    # this is used to perform a contrast model against the --vntrfile associated to --geneexpfile. Added as a random effect covariance structure to the null model (to remove best variant effect on dependent variable (gene expression))
    # this should be in the same format as --vntrfile, however pid and gid must be the best variant as pid and gid is the gene the best variant is associated with.
    chr  start  end  pid  gid  strand  indiv1  indiv2
    ...
    27	7699875	7699875	27_7699875_SNP_A_G	SPATA4	+	./.	0/1	0/1	1/1	0/1
    ...
    
    # --vntrallelelengthfile : this file should be in ./inputs directory
    # this is used to plot the VNTR genotypes as allele lengths (ex 156/302) instead of genotypes (ex 0/1, 1/1, 0/2, etc). This argument is only used for plotting and not for analysis.
    # however setting the vntrallelelength file as the --vntrfile will allow the cis-eVNTR analysis to be performed using the allele lengths as different alleles in a genotype matrix (156/302)

    # --bestvntrmeanallelelengthfile: this file should be in ./inputs directory
    # this is used to contrast the best mean allele length VNTR aginst the --vntrfile associated to --geneexpfile. Added as a fixed effect covariate (to remove best mean allele length effect on dependent variable (gene expression))

    # --bestvntrallelelengthfile : this file should be in ./inputs directory
    # this is used to plot the best VNTR genotypes as allele lengths (ex 156/302) instead of genotypes (ex 0/1, 1/1, 0/2, etc). This argument is only used for plotting and not for analysis.

    # --desired_covs: list of covariate IDs to include (space-separated). These should match the IDs in the covarfile. If not specified, all covariates will be used.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--workingdir", type=str, required=True, help="full path to working directory containing ./inputs which contains input files")
    parser.add_argument("--fdr", type=float, default=0.05, help="FDR threshold for multiple testing correction")
    parser.add_argument("--window", type=int, default=1000000, help="Window size for VNTR search (default: 1Mb), 500kb up- and down-stream from gene start (if + strand) or gene end (if - strand)")
    parser.add_argument("--genexpfile", type=str, default='gene_expression.tsv', help="filename of file in ./inputs :Bed like file, format (tab separated): #chr    start   end     pid     gid     strand  indiv1  indiv2  indiv3...")
    parser.add_argument("--vntrfile", type=str, default='TRDB_MatrixGenotypes.txt', help="filename of file in ./inputs :Bed like file, format (tab separated): #chr    start   end     pid     gid     strand  indiv1  indiv2  indiv3...")
    parser.add_argument("--vntrallelelengthfile", type=str, default=None, help="Matrix of allele lengths as genotypes. Only used for plotting. filename of file in ./inputs :Bed like file, format (tab separated): #chr    start   end     pid     gid     strand  indiv1  indiv2  indiv3...")
    parser.add_argument("--vntrmeanallelelengthfile", type=str, default=None, help="Where values are mean allele length. Adds meanallelelength as fixed effect covariate (removes effect before mixed effect estimation). filename of file in ./inputs :Bed like file, format (tab separated): #chr    start   end     pid     gid     strand  indiv1  indiv2  indiv3...")
    parser.add_argument("--bestvntrallelelengthfile", type=str, default=None, help="Matrix of allele lengths as genotypes where each variant is the best associated variant with a gene. Only used for plotting, if you want to use this as a contrast (null model rand effect matrix) input this matrix as --bestvariantfile. filename of file in ./inputs :Bed like file, format (tab separated): #chr    start   end     pid     gid     strand  indiv1  indiv2  indiv3...")
    parser.add_argument("--bestvntrmeanallelelengthfile", type=str, default=None, help="Where values are mean allele length where each variant is the best associated variant with a gene. Adds meanallelelength as fixed effect covariate (removes effect before mixed effect estimation). filename of file in ./inputs :Bed like file, format (tab separated): #chr    start   end     pid     gid     strand  indiv1  indiv2  indiv3...")
    parser.add_argument("--bestvariantfile", type=str, default=None, help="Where values are genotypes of the best variants associated with a gene. These variants will be encoded as a dosage matrix (make a K matrix) and then added as the genetic component in the null model for mixed model testing. This tests if the vntr covariance matrix explains more variation in gene expression than this 'best' variant for the gene. (removes effect before mixed effect estimation). filename should start with 'bestSNP.', 'bestINDEL.', or 'bestVNTR' this prefix is used for naming output file when plotting. file in ./inputs: Bed like file, format (tab separated): #chr    start   end     pid     gid     strand  indiv1  indiv2  indiv3...")
    parser.add_argument("--covarfile", type=str, default='covariates.tsv', required=False, help="filename of file in ./inputs : tab separated file: covariate  indiv1  indiv2  indiv3...")
    parser.add_argument("--kinshipfile", type=str, default=None, required=False, help="filename of file in ./inputs : tab separated file, symmetrical matrix: indiv1  indiv2  indiv3...")
    parser.add_argument("--minsamples", type=int, default=20, help="int value, if the number of common ids between gene expression matrix and vntr length matrix is < args.minsamples then the locus will be skipped")
    parser.add_argument("--geneid", type=str, required=False, default=None, help="Gene ID for specific analysis, if only --geneid specified (no --vntrid) do a gene specific cis-eqtl analysis. (--vntrid must be specified to make plots)")
    parser.add_argument("--vntrid", type=str, required=False, help="VNTR ID for specific analysis, required for plotting (also requires --geneid to make plots)")
    parser.add_argument("--xaxislabel", type=str, default="VNTR Mean Allele Length", required=False, help="Label for x axis if plotting")
    parser.add_argument("--desired_covs", nargs="+", type=str, required=False, help="List of covariate IDs to include (space-separated).")  # Add an argument to accept multiple strings as a list
    parser.add_argument("--cisfile", type=str, default=None, help="Filename for cis VNTR comparisons present in inputs directory (default: None) (example: cis_vntrs.tsv)")
    parser.add_argument("--chromosome", type=str, default=None, help="If specified, reduces analyzed cis_vntrs to only those on the specified chromosome. If --cisfile is specified, --chromosome has no function (default: None)")
    parser.add_argument("--outdir", type=str, default=None, required=False, help="Output directory for results, will be created in args.workingdir (default: output)")
    parser.add_argument("--norarealleles", default=False, action='store_true', help="If used the value will be set to True, only include alleles (per VNTR locus) > 0.05 frequency (default: False, requires --collapserarealleles to be False)")
    parser.add_argument("--collapserarealleles", default=False, action='store_true', help="If used the value will be set to True, collapses all 1 or 2 count alleles to the same allele 'rare' (default: False, requires --norarealleles to be False)")
    parser.add_argument("--Zraw", default=False, action='store_true', help="If used the value will be set to True and will use the raw dosage matrix (Z) for LMM, if not provided default value false is used and the Z matrix will be centered by average allele dosage making Zc where that will be used for LMM (default: False), (only functional in genome wide analysis, Zraw is alwazs done for plotting)")
    parser.add_argument("--removereferenceallele", default=False, action='store_true', help="If used the value will be set to True, if True the reference allele (assumed to be allele '0') will be remove from Z making all BLUP predictions relative to the reference allele. This requires MAF of '0' > 0.05 and --removemostfrequentallele == False (default: False)")
    parser.add_argument("--removemostfrequentallele", default=False, action='store_true', help="If used the value will be set to True, if True the most frequent allele will be removed from Z, this requires the most frequent allele to be MAF > 0.05, additionally this will not remove 'rare' column if --collapserarealleles == True. --removereferenceallele must be == False (default: False)")
    parser.add_argument("--reml", default=False, action='store_true', help="If used the value will be set to True, if True the model will be fit using REML, If False model will be fit using ML (default: False)")
    parser.add_argument("--permutevntrmeanlengths", default=False, action='store_true', help="If used the value will be set to True, if True the mean VNTR allele lengths will be permuted across individuals before being added as a fixed effect covariate to the null and full model (default: False)")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use and = the number of genes analyzed simultaneously for multiprocessing (default: 4) (cis-windows or genome wide analysis only, not used for gene-specific analysis when plotting)")

    args = parser.parse_args()

    print(f'Desired covariates: {args.desired_covs}')
    # input files assumed to be in relative directory ./inputs/input_file.bed
    gxpfile, vntrfile, covfile = os.path.join(args.workingdir, 'inputs', args.genexpfile), os.path.join(args.workingdir, 'inputs', args.vntrfile), os.path.join(args.workingdir, 'inputs', args.covarfile)
    kinfile = os.path.join(args.workingdir, 'inputs', args.kinshipfile) if args.kinshipfile is not None else None

    validate_args(args)

    outdir = build_output_dir(args)
    print(f"Output directory for results: {outdir}")
    os.makedirs(outdir, exist_ok=True)


    if args.vntrmeanallelelengthfile is not None:
        vntrmeanlengthfile = os.path.join(args.workingdir, 'inputs', args.vntrmeanallelelengthfile)
        print(f"VNTR mean allele length file to correct by adding as fixed effect to null and full models: {vntrmeanlengthfile}")
    else:
        vntrmeanlengthfile = None
    if args.bestvariantfile is not None:
        bestvarfile = os.path.join(args.workingdir, 'inputs', args.bestvariantfile)
        print(f"The best variant file to add as genetic covariance matrix to null model: {bestvarfile}")
    else:
        bestvarfile = None
    if args.vntrallelelengthfile is not None:
        vntrlengthfile = os.path.join(args.workingdir, 'inputs', args.vntrallelelengthfile)
        print(f"The allele length VNTR variant file to add as genetic covariance matrix to null model: {vntrlengthfile}")
    else:
        vntrlengthfile = None
    if args.bestvntrallelelengthfile is not None:
        bestvntrlengthfile = os.path.join(args.workingdir, 'inputs', args.bestvntrallelelengthfile)
        print(f"The best allele length VNTR variant file to add as genetic covariance matrix to null model: {bestvntrlengthfile}")
    else:
        bestvntrlengthfile = None
    if args.bestvntrmeanallelelengthfile is not None:
        bestvntrmeanallelelengthfile = os.path.join(args.workingdir, 'inputs', args.bestvntrmeanallelelengthfile)
        print(f"The best mean allele length VNTR variant file to correct for by adding as fixed effect to null and full models: {bestvntrmeanallelelengthfile}")
    else:
        bestvntrmeanallelelengthfile = None

    print(f'Reading in matrices from: {args.genexpfile} and {args.vntrfile}')
    expression_matrix, vntr_matrix, covariates, allcovariates, common_inds, kinship_matrix, vntr_mean_length_matrix, best_variant_matrix, vntr_length_matrix, best_vntr_length_matrix, best_vntr_mean_length_matrix = load_data(gxpfile, vntrfile, covfile, kinfile, vntrmeanlengthfile=vntrmeanlengthfile, bestvarfile=bestvarfile, vntrlengthfile=vntrlengthfile, bestvntrlengthfile=bestvntrlengthfile, bestvntrmeanallelelengthfile=bestvntrmeanallelelengthfile, desired_covariates=args.desired_covs, geneid=args.geneid)
    print(f'Finished reading in matrices')


    if best_variant_matrix is not None:
        if args.geneid and args.vntrid:
            print(f"VNTR genotypes for gene {args.vntrid}: {vntr_matrix.loc[args.vntrid].to_dict()}")
            if vntr_mean_length_matrix is not None:
                print(f"VNTR mean allele lengths for gene {args.vntrid}: {vntr_mean_length_matrix.loc[args.vntrid].to_dict()}")
            # user requested plotting
            in_gene_expr = args.geneid in expression_matrix.index
            in_best = args.geneid in best_variant_matrix.index
            print(f"Gene in gene expression matrix: {in_gene_expr}.\nGene in best variant matrix {in_best}")
            if vntr_length_matrix is not None:
                print(f"VNTR allele length genotypes for VNTR {args.vntrid}, and target gene: {args.geneid}: {vntr_length_matrix.loc[args.vntrid].to_dict()}")
            if vntr_mean_length_matrix is not None:
                print(f"VNTR mean allele lengths for VNTR {args.vntrid}, and target gene: {args.geneid}: {vntr_mean_length_matrix.loc[args.vntrid].to_dict()}")
            if best_vntr_length_matrix is not None:
                print(f"Best VNTR allele length genotypes for gene {args.geneid}: {best_vntr_length_matrix.loc[args.geneid].to_dict()}")
            if best_vntr_mean_length_matrix is not None:
                print(f"Best VNTR mean allele lengths for gene {args.geneid}: {best_vntr_mean_length_matrix.loc[args.geneid].to_dict()}")
            print(f"best_variant_matrix genotypes for gene {args.geneid}: {best_variant_matrix.loc[args.geneid].to_dict()}")
            # check that the genoptypes are in the same order as the initial file:
            # read the bext variant file again to check order
            best_variant_file_check = pd.read_csv(bestvarfile, sep="\t", dtype={"#chr": str, "pid": str, "gid": str, "strand": str})
            print(f"best_variant_file_check genotypes for gene {args.geneid}: {best_variant_file_check.set_index('gid').loc[args.geneid].to_dict()}")
            # report individuals that are different, if any
            for ind in best_variant_matrix.columns[6:]:
                if best_variant_matrix.loc[args.geneid, ind] != best_variant_file_check.set_index('gid').loc[args.geneid, ind]:
                    print(f"\n\n[warning] Individual {ind} has different genotype values between loaded best_variant_matrix and bestvariantfile! {best_variant_matrix.loc[args.geneid, ind]} vs {best_variant_file_check.set_index('gid').loc[args.geneid, ind]}\n\n")

            if not (in_gene_expr and in_best):
                print(f'Gene ID: {args.geneid} not found in expression matrix or best-variant matrix; please check --geneid or input files')
                sys.exit(1)
        else:
            # limit the gene expression matrix to only those genes that also appear in the best-variant matrix
            # this will reduce the number of genes tested to only those with best-variant data
            common_genes = expression_matrix.index.intersection(best_variant_matrix.index)
            print(f"Number of shared genes in both expression matrix and best variant matrix: {len(common_genes)}")
            if len(common_genes) == 0:
                print("[warning] No overlapping genes between expression matrix and best-variant matrix after loading data. Exiting.")
                sys.exit(1)
            else:
                print(f"Expression matrix shape: {expression_matrix.shape}")
                print(f"Best-variant matrix shape: {best_variant_matrix.shape}")
                expression_matrix = expression_matrix.loc[common_genes]
                best_variant_matrix = best_variant_matrix.loc[common_genes]
                print(f"Expression matrix shape (after filtering to common_genes): {expression_matrix.shape}")
                print(f"Best-variant matrix shape (after filtering to common_genes): {best_variant_matrix.shape}")

    if args.geneid and args.vntrid:
        print(f'Plotting relevant regressions between gene expression from gene: {args.geneid} and x-axis value: {args.vntrid}')
        print(f'Expression values for gene {args.geneid}: {expression_matrix.loc[args.geneid].to_dict()}')
        print(f'VNTR values for VNTR {args.vntrid}: {vntr_matrix.loc[args.vntrid].to_dict()}')
        
        # from collections import Counter
        # 
        # 
        # allele_counts = Counter()
        # 
        # for geno in vntr_matrix.loc[args.vntrid].iloc[6:]:
        #     if not isinstance(geno, str):
        #         continue
        #     alleles = geno.split('/')
        #     allele_counts.update(alleles)
        # 
        # for allele, count in sorted(allele_counts.items()):
        #     print(f"{allele}: {count}")

        plot_regression(args.geneid, args.vntrid, expression_matrix, vntr_matrix, covariates, allcovariates, common_inds, kinship_matrix, vntr_mean_length_matrix, best_variant_matrix, vntr_length_matrix, best_vntr_length_matrix, best_vntr_mean_length_matrix, bestvarfile, args.workingdir, args.minsamples, args.xaxislabel, no_rare_alleles=args.norarealleles, collapse_rare_alleles=args.collapserarealleles, reml=args.reml, permute_vntr_mean_lengths=args.permutevntrmeanlengths)
    elif args.geneid and not args.vntrid:
        # perform cis association for the specified gene only
        print(f'Performing cis association analysis for gene: {args.geneid}')
        expression_matrix_gene = expression_matrix.loc[[args.geneid]]
        gene_vntr_dict = find_cis_vntrs(expression_matrix_gene, vntr_matrix, args.window, args.chromosome)
        print(f'Found {len(gene_vntr_dict)} cis comparisons for gene {args.geneid}')
        if len(gene_vntr_dict) == 0:
            print(f'No cis VNTRs found for gene {args.geneid} within window {args.window}. Exiting.')
            sys.exit(1)
        print(f'Starting to run cis associations for gene {args.geneid}:')

        # only write to file if cisfile was not specified, writes to outdir, but can be moved to inputs dir for later runs of the program
        write_cis_vntrs_to_file(gene_vntr_dict, outdir, filename=f"cis_vntrs.{args.geneid}.tsv")
        output_file = os.path.join(outdir, f"gene_vntr_association_results.{args.geneid}.tsv")
        print(f'Number of genes: {len(gene_vntr_dict)}, average number of cis VNTRs {round(np.mean([len(vntrs) for vntrs in list(gene_vntr_dict.values())]), 2)}. Estimated number of cis-models: {round(len(gene_vntr_dict) * np.mean([len(vntrs) for vntrs in list(gene_vntr_dict.values())]), 2)}')

        run_association_analysis(output_file, gene_vntr_dict, expression_matrix_gene, vntr_matrix, covariates, args.workingdir, args.minsamples, args.threads, common_inds, kinship_matrix, vntr_mean_length_matrix, best_vntr_mean_length_matrix, best_variant_matrix, no_rare_alleles=args.norarealleles, collapse_rare_alleles=args.collapserarealleles, Zraw=args.Zraw, remove_reference_allele=args.removereferenceallele, remove_most_frequent_allele=args.removemostfrequentallele, reml=args.reml, permute_vntr_mean_lengths=args.permutevntrmeanlengths)
    else:
        if args.cisfile:
            cisfile = os.path.join(args.workingdir, 'inputs', args.cisfile)
            if os.path.isfile(cisfile):
                print(f'Loading all cis gene_VNTR comparisons from: {cisfile}')
                gene_vntr_dict = read_in_cis_vntrs(cisfile)  # i don't want to filter cis-vntrs by chromosome because I don't want to assume ID strucuture like chr.start.end: 2.156789.167800, could also be VNTR5087
            else:
                print(f'--cisfile {cisfile} does not exist in inputs directory. Exiting')
                sys.exit(1)
        else:  # if --cisfile is not specified, find cis-vntrs
            print(f'Finding all cis comparisons between {args.genexpfile} and {args.vntrfile}')
            if args.chromosome:
                print(f'--chromosome specified, Reducing cis vntr comparisons to only those on chromosome: {args.chromosome}')
            gene_vntr_dict = find_cis_vntrs(expression_matrix, vntr_matrix, args.window, args.chromosome)
            print(f'Found {len(gene_vntr_dict)} cis comparisons')

            # only write to file if cisfile was not specified, writes to outdir, but can be moved to inputs dir for later runs of the program
            write_cis_vntrs_to_file(gene_vntr_dict, outdir, filename=f"cis_vntrs.{args.chromosome}.tsv")

        print(f'Number of genes: {len(gene_vntr_dict)}, average number of cis VNTRs {round(np.mean([len(vntrs) for vntrs in list(gene_vntr_dict.values())]), 2)}. Estimated number of cis-models: {round(len(gene_vntr_dict) * np.mean([len(vntrs) for vntrs in list(gene_vntr_dict.values())]), 2)}')
        
        # print(f'Starting to run all cis associations in chunks:')
        # run_association_analysis_in_chunks(gene_vntr_dict, expression_matrix, vntr_matrix, covariates, args.workingdir, args.minsamples, kinship_matrix, vntr_mean_length_matrix, fdr_threshold=args.fdr, chunk_size=1)

        print(f'Starting to run all cis associations:')
        if args.chromosome:
            output_file = os.path.join(outdir, f"gene_vntr_association_results.{args.chromosome}.tsv")
        else:
            output_file = os.path.join(outdir, f"gene_vntr_association_results.tsv")
        run_association_analysis(output_file, gene_vntr_dict, expression_matrix, vntr_matrix, covariates, args.workingdir, args.minsamples, args.threads, common_inds, kinship_matrix, vntr_mean_length_matrix, best_vntr_mean_length_matrix, best_variant_matrix, no_rare_alleles=args.norarealleles, collapse_rare_alleles=args.collapserarealleles, Zraw=args.Zraw, remove_reference_allele=args.removereferenceallele, remove_most_frequent_allele=args.removemostfrequentallele, reml=args.reml, permute_vntr_mean_lengths=args.permutevntrmeanlengths)

        if args.chromosome:
            print(f"--chromosome was specified, so it is assumed not all cis-vntr comparisons were present in this run")
            print(f"Skipping multiple testing correction on all cis-vntr comparisons, after all cis-vntr comparisons are completed for each chromosome, \
                  run the LM_filter_results.py script to perform multiple testing correction on all cis-vntr comparisons")
        else:
            print(f"--chromosome not specified, so assuming all cis-vntr comparisons were present in --cisfile")
            print(f"Start performing multiple testing correction on all cis-vntr comparisons")
            # filter and perform multiple tesing correction on output
            read_result_table_and_filter(output_file, args.workingdir, fdr_threshold=args.fdr)

    return


if __name__ == "__main__":
    #USAGE: conda activate py3.10
    #USAGE: python script.py --workingdir /path/to/workingdir --fdr 0.05 --window 1000000 --genexpfile gene_expression.tsv --vntrfile vntr_mean_lengths.tsv --covarfile covariates.tsv --kinshipfile kinship_matrix.tsv --minsamples 20
    #USAGE: python LM_BLUE.py --genexpfile TPM_gene_testis_FILTERED_UNNORM.tsv --vntrfile TRDB_MatrixMeanAlelleLength.txt --covarfile RNA_cov_testis.txt --kinshipfile SimilarityMatrixRaw_SharedAlleleProportions_LociCount752950.tsv --desired_covs PC1 PC2 PC3 PC4 PC5 genotypePC3 genotypePC4 genotypePC5 genotypePC6 genotypePC7 --workingdir /Users/vmason/Documents/Analyses/Modeling/Linear/eQTLCohort
    #USAGE: python LM_BLUE_byChrom.py --chromosome "${CHROM}" --genexpfile TPM_gene_testis_FILTERED_UNNORM.tsv --vntrfile TRDB_MatrixGenotypes.txt --covarfile RNA_cov_testis.txt --desired_covs PC1 PC2 PC3 PC4 PC5 genotypePC3 genotypePC4 genotypePC5 genotypePC6 genotypePC7 --workingdir /cluster/work/pausch/vmason/analyses/modeling/linear/eQTLCohort_BLUE
    #USAGE: python LM_BLUE_byChrom.py --chromosome "${CHROM}" --genexpfile TPM_gene_testis_FILTERED_UNNORM.tsv --vntrfile TRDB_MatrixGenotypes.txt --covarfile RNA_cov_testis.txt --vntrmeanallelelengthfile TRDB_MatrixMeanAlelleLength.txt --desired_covs PC1 PC2 PC3 PC4 PC5 genotypePC3 genotypePC4 genotypePC5 genotypePC6 genotypePC7 --workingdir /cluster/work/pausch/vmason/analyses/modeling/linear/eQTLCohort_BLUE_MinusAlleleLength

    main()

    # monitor.stop()  # Stop the monitor when the program finishes