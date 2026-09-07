import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

import warnings


def compute_dci(latents, factors, is_classification=True):
    """
    Compute DCI scores: Disentanglement, Completeness, Informativeness.
    
    Args:
        latents (np.ndarray): shape (n_samples, n_latents)
        factors (np.ndarray): shape (n_samples, n_factors)
        is_classification (bool): True if factors are discrete (classification),
                                  False if factors are continuous (regression).
    Returns:
        dict scores
    """
    n_samples, n_latents = latents.shape
    _, n_factors = factors.shape
    
    importance_matrix = np.zeros((n_latents, n_factors))
    predictions = []
    
    for j in range(n_factors):
        y = factors[:, j]
        
        # Encode categorical targets if classification
        if is_classification:
            y = LabelEncoder().fit_transform(y)
            model = GradientBoostingClassifier(n_estimators=100)
        else:
            model = GradientBoostingRegressor(n_estimators=100)
        
        model.fit(latents, y)
        importance_matrix[:, j] = model.feature_importances_
        
        # Collect predictions for informativeness
        if is_classification:
            y_pred = model.predict(latents)
            score = accuracy_score(y, y_pred)
        else:
            y_pred = model.predict(latents)
            score = r2_score(y, y_pred)
        predictions.append(score)
    
    # ---- Disentanglement ----
    prob_matrix = importance_matrix / (importance_matrix.sum(axis=0, keepdims=True) + 1e-8)
    disentanglement_latents = 1.0 - (-(prob_matrix * np.log(prob_matrix + 1e-8)).sum(axis=1)) / np.log(n_factors)
    latent_importance = importance_matrix.sum(axis=1)
    disentanglement_score = np.sum(latent_importance * disentanglement_latents) / (latent_importance.sum() + 1e-8)
    
    # ---- Completeness ----
    prob_matrix_T = importance_matrix.T / (importance_matrix.T.sum(axis=0, keepdims=True) + 1e-8)
    completeness_factors = 1.0 - (-(prob_matrix_T * np.log(prob_matrix_T + 1e-8)).sum(axis=1)) / np.log(n_latents)
    factor_importance = importance_matrix.sum(axis=0)
    completeness_score = np.sum(factor_importance * completeness_factors) / (factor_importance.sum() + 1e-8)
    
    # ---- Informativeness ----
    informativeness_score = np.mean(predictions)
    
    return {
        "disentanglement": disentanglement_score,
        "completeness": completeness_score,
        "informativeness": informativeness_score,
        "importance_matrix": importance_matrix
    }
    
EPS = 1e-10

def estimate_mi_matrix(latents, factors, factor_discrete):
    """
    Return MI matrix of shape (n_latents, n_factors)
    Uses sklearn mutual_info_regression / mutual_info_classif.
    """
    n_latents = latents.shape[1]
    n_factors = factors.shape[1]
    MI = np.zeros((n_latents, n_factors), dtype=float)

    # standardize latents for stability
    latents_s = StandardScaler().fit_transform(latents)

    for j in range(n_factors):
        y = factors[:, j]
        if factor_discrete[j]:
            # mutual_info_classif expects discrete y (int labels)
            # if y is not integer labels, try to discretize
            if not np.issubdtype(y.dtype, np.integer):
                # simple discretization: unique values -> integers
                uniq, inv = np.unique(y, return_inverse=True)
                y_disc = inv
            else:
                y_disc = y
            # compute MI of each latent dim with factor j
            # mutual_info_classif expects 2D X
            mi_vals = mutual_info_classif(latents_s, y_disc, discrete_features=False, random_state=0)
        else:
            mi_vals = mutual_info_regression(latents_s, y, discrete_features=False, random_state=0)

        MI[:, j] = mi_vals

    return MI


def modularity_from_mi(MI):
    """
    Compute modularity per-latent following a typical information-theoretic modularity proxy:
      For each latent i:
        let k = argmax_j MI[i,j]
        modularity_i = 1 - (sum_{j != k} MI[i,j]^2) / (sum_j MI[i,j]^2)
      If sum_j MI[i,j]^2 == 0 -> set modularity_i = 0
    Then aggregate modularity across latents; weight by total MI per latent to downweight inactive latents.
    Returns scalar modularity in [0,1] and per-latent scores.
    """
    n_latents, n_factors = MI.shape
    mi_sq = MI ** 2
    denom = mi_sq.sum(axis=1)  # per-latent
    idx_max = np.argmax(MI, axis=1)
    modularity_per = np.zeros(n_latents, dtype=float)

    for i in range(n_latents):
        if denom[i] <= EPS:
            modularity_per[i] = 0.0
        else:
            k = idx_max[i]
            numer = mi_sq[i].sum() - mi_sq[i, k]
            modularity_per[i] = 1.0 - numer / denom[i]  # in [0,1] (higher -> more modular)

    # weight by informativeness (sum MI) to reduce effect of dead latents
    mi_sum = MI.sum(axis=1)
    if mi_sum.sum() <= EPS:
        modularity = modularity_per.mean()
    else:
        modularity = np.sum(modularity_per * (mi_sum / (mi_sum.sum())))  # weighted average

    return float(modularity), modularity_per


def compactness_from_mi(MI):
    """
    For each factor j, consider the MI vector across latents MI[:, j] and compute:
      p = MI[:,j] / (sum MI[:,j] + eps)
      entropy = -sum p * log(p)
      normalized_entropy = entropy / log(n_latents)
      compactness_j = 1 - normalized_entropy   # 1 => concentrated in 1 lat, 0 => uniform spread
    Aggregate across factors (simple average).
    Returns scalar compactness in [0,1] and per-factor compactness.
    """
    n_latents, n_factors = MI.shape
    compact_per = np.zeros(n_factors, dtype=float)
    log_n = np.log(n_latents) if n_latents > 1 else 1.0

    for j in range(n_factors):
        v = MI[:, j].astype(float)
        s = v.sum()
        if s <= EPS:
            compact_per[j] = 0.0
        else:
            p = v / (s + EPS)
            # numeric: avoid log(0)
            p_safe = np.where(p > 0, p, 1.0)  # temporary for computing p*log(p) (p=0 contributes 0)
            entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0))
            norm_entropy = entropy / (log_n + EPS)
            compact_per[j] = max(0.0, 1.0 - norm_entropy)

    compactness = float(compact_per.mean())
    return compactness, compact_per


def explicitness_from_latents(latents, factors, factor_discrete, cv_folds=5):
    """
    Explicitness: how well ground-truth factors can be predicted from latents.
    For discrete factors -> logistic regression (accuracy); continuous -> ridge regression (R^2 mapped to [0,1]).
    Use cross-validation (default 5-fold) and return mean predictive score in [0,1].
    """
    n_factors = factors.shape[1]
    scores = np.zeros(n_factors, dtype=float)
    latents_s = StandardScaler().fit_transform(latents)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)

    for j in range(n_factors):
        y = factors[:, j]
        if factor_discrete[j]:
            # ensure labels are integer class indices
            if not np.issubdtype(y.dtype, np.integer):
                _, y = np.unique(y, return_inverse=True)
            clf = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto')
            try:
                cv_scores = cross_val_score(clf, latents_s, y, cv=kf, scoring='accuracy')
                scores[j] = float(np.mean(cv_scores))
            except Exception:
                # fallback: simple train-test split
                warnings.warn(f"LogisticRegression CV failed for factor {j}, returning 0.")
                scores[j] = 0.0
        else:
            # regression (R^2) normalized to [0,1] by sigmoid-like mapping: r2 -> (r2+1)/2 if r2 in [-1,1]
            reg = Ridge(alpha=1.0, fit_intercept=True)
            try:
                cv_scores = cross_val_score(reg, latents_s, y, cv=kf, scoring='r2')
                mean_r2 = float(np.mean(cv_scores))
                # map to [0,1] conservatively (if r2 negative, map <0.5)
                mapped = (mean_r2 + 1.0) / 2.0
                mapped = float(np.clip(mapped, 0.0, 1.0))
                scores[j] = mapped
            except Exception:
                warnings.warn(f"Ridge CV failed for factor {j}, returning 0.")
                scores[j] = 0.0

    explicitness = float(scores.mean())
    return explicitness, scores


def compute_infoMEC(latents, factors, factor_discrete):
    """
    Main wrapper: compute MI matrix, then M, E, C.
    Returns dict with MI matrix and scores.
    """
    assert latents.shape[0] == factors.shape[0], "latents and factors must have same number of samples"
    MI = estimate_mi_matrix(latents, factors, factor_discrete)  # shape (n_latents, n_factors)

    modularity, modularity_per = modularity_from_mi(MI)
    compactness, compact_per = compactness_from_mi(MI)
    explicitness, explicit_per = explicitness_from_latents(latents, factors, factor_discrete)

    info = {
        'MI_matrix': MI,
        'Modularity': modularity,
        'Modularity_per_latent': modularity_per,
        'Compactness': compactness,
        'Compactness_per_factor': compact_per,
        'Explicitness': explicitness,
        'Explicitness_per_factor': explicit_per
    }
    return info