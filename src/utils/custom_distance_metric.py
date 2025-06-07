"""
Computes gower distance matrix for mix data types.
Credits: M. Beckmann  https://github.com/scikit-learn/scikit-learn/pull/9555
"""

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import pairwise
from sklearn.utils import validation

# pylint: skip-file


def check_pairwise_arrays(X, Y, precomputed=False, dtype=None):
    """Validate and format input arrays X and Y for pairwise operations."""
    X, Y, dtype_float = pairwise._return_float_dtype(X, Y)

    warn_on_dtype = dtype is not None
    estimator = "check_pairwise_arrays"

    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = validation.check_array(
            X,
            accept_sparse="csr",
            dtype=dtype,
            force_all_finite="allow-nan",
            estimator=estimator,
        )
    else:
        X = validation.check_array(
            X,
            accept_sparse="csr",
            dtype=dtype,
            force_all_finite="allow-nan",
            estimator=estimator,
        )
        Y = validation.check_array(
            Y,
            accept_sparse="csr",
            dtype=dtype,
            force_all_finite="allow-nan",
            estimator=estimator,
        )

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError(
                "Precomputed metric requires shape "
                "(n_queries, n_indexed). Got (%d, %d) "
                "for %d indexed." % (X.shape[0], X.shape[1], Y.shape[0])
            )
    elif X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Incompatible dimension for X and Y matrices: "
            "X.shape[1] == %d while Y.shape[1] == %d"
            % (X.shape[1], Y.shape[1])
        )

    return X, Y


def _gower_distance_row(
    xi_cat,
    xi_num,
    xj_cat,
    xj_num,
    feature_weight_cat,
    feature_weight_num,
    feature_weight_sum,
    cat_features,
    ranges_of_num,
    max_of_num,
):
    """
    Computes the Gower distance between one row and a set of rows for
    categorical and numerical features.
    """
    # categorical columns
    sij_cat = np.where(
        xi_cat == xj_cat, np.zeros_like(xi_cat), np.ones_like(xi_cat)
    )
    sum_cat = np.multiply(feature_weight_cat, sij_cat).sum(axis=1)

    # numerical columns
    abs_delta = np.absolute(xi_num - xj_num)
    sij_num = np.divide(
        abs_delta,
        ranges_of_num,
        out=np.zeros_like(abs_delta),
        where=ranges_of_num != 0,
    )

    sum_num = np.multiply(feature_weight_num, sij_num).sum(axis=1)
    sums = np.add(sum_cat, sum_num)
    sum_sij = np.divide(sums, feature_weight_sum)
    return sum_sij


def gower_distances(X, Y=None, feature_weight=None, cat_features=None):
    """
    Computes the Gower distances between X and Y.

    Gower is a similarity measure for categorical, boolean, and numerical mixed data.

    :param X: First dataset of shape (n_samples, n_features)
    :type X: array-like or pandas.DataFrame
    :param Y: Second dataset of shape (n_samples, n_features)
    :type Y: array-like or pandas.DataFrame
    :param feature_weight: Attribute weights according to the Gower formula
    :type feature_weight: array-like of shape (n_features)
    :param categorical_features: Indicates whether each column is categorical (True/False) or
                                 list of indices for categorical columns. Ordinal attributes
                                 should be marked as False.
    :type categorical_features: array-like of shape (n_features) or list of int
    :return: Similarity matrix
    :rtype: numpy.ndarray of shape (n_samples, n_samples)

    :note: The non-numeric features and numeric feature ranges are determined from X and not Y.
           Sparse matrices are not supported.
    """

    if issparse(X) or issparse(Y):
        raise TypeError("Sparse matrices are not supported for gower distance")

    y_none = Y is None

    # It is necessary to convert to ndarray in advance to define the dtype
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    # this is necessary as strangelly the validator is rejecting
    # numeric arrays with NaN
    array_type = object

    if np.issubdtype(X.dtype, np.number) and (
        np.isfinite(X.sum()) or np.isfinite(X).all()
    ):
        array_type = type(np.zeros(1, X.dtype).flat[0])

    X, Y = check_pairwise_arrays(X, Y, precomputed=False, dtype=array_type)

    n_rows, n_cols = X.shape

    if cat_features is None:
        cat_features = np.zeros(n_cols, dtype=bool)
        for col in range(n_cols):
            # In numerical columns, None is converted to NaN,
            # and the type of NaN is recognized as a number subtype
            if not np.issubdtype(type(X[0, col]), np.number):
                cat_features[col] = True
    else:
        cat_features = np.array(cat_features)

    if np.issubdtype(cat_features.dtype, np.int64):
        new_cat_features = np.zeros(n_cols, dtype=bool)
        new_cat_features[cat_features] = True
        cat_features = new_cat_features

    # Categorical columns
    X_cat = X[:, cat_features]

    # Numerical columns
    X_num = X[:, np.logical_not(cat_features)].astype(np.float32)
    ranges_of_num = None
    max_of_num = None

    # Calculates the normalized ranges and max values of numeric values
    _, num_cols = X_num.shape
    ranges_of_num = np.zeros(num_cols)
    max_of_num = np.zeros(num_cols)
    for col in range(num_cols):
        col_array = X_num[:, col].astype(np.float32)
        max = np.nanmax(col_array)
        min = np.nanmin(col_array)

        if np.isnan(max):
            max = 0.0
        if np.isnan(min):
            min = 0.0
        max_of_num[col] = max
        ranges_of_num[col] = np.absolute(1 - min / max) if (max != 0) else 0.0

    # This is to normalize the numeric values between 0 and 1.
    X_num = np.divide(
        X_num, max_of_num, out=np.zeros_like(X_num), where=max_of_num != 0
    )

    if feature_weight is None:
        feature_weight = np.ones(n_cols)

    feature_weight_cat = feature_weight[cat_features]
    feature_weight_num = feature_weight[np.logical_not(cat_features)]

    y_n_rows, _ = Y.shape

    dm = np.zeros((n_rows, y_n_rows), dtype=np.float32)

    feature_weight_sum = feature_weight.sum()

    Y_cat = None
    Y_num = None

    if not y_none:
        Y_cat = Y[:, cat_features]
        Y_num = Y[:, np.logical_not(cat_features)].astype(np.float32)
        # This is to normalize the numeric values between 0 and 1.
        Y_num = np.divide(
            Y_num, max_of_num, out=np.zeros_like(Y_num), where=max_of_num != 0
        )
    else:
        Y_cat = X_cat
        Y_num = X_num

    for i in range(n_rows):
        j_start = i

        # for non square results
        if n_rows != y_n_rows:
            j_start = 0

        Y_cat[j_start:n_rows, :]
        Y_num[j_start:n_rows, :]
        result = _gower_distance_row(
            X_cat[i, :],
            X_num[i, :],
            Y_cat[j_start:n_rows, :],
            Y_num[j_start:n_rows, :],
            feature_weight_cat,
            feature_weight_num,
            feature_weight_sum,
            cat_features,
            ranges_of_num,
            max_of_num,
        )
        dm[i, j_start:] = result
        dm[i:, j_start] = result

    return dm
