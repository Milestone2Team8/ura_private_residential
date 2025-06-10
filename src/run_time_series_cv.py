"""
Performs time series cross validation to find the best regressor model.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVR

from src.utils.load_configs import load_configs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RANDOM_STATE = 42
OUTPUT_PATH = Path("./src/data/output/cv_results.json")


def build_regressors(random_state=RANDOM_STATE):
    """Builds a list of regressors and their param grids."""
    param_grids = {
        "Linear": (
            LinearRegression,
            {},
        ),
        "RF": (
            RandomForestRegressor,
            {
                "max_depth": [5, 10, 15],
                "random_state": [random_state],
            },
        ),
        "SVR": (
            SVR,
            {"kernel": ["rbf"], "C": [0.1, 0.5, 1.0]},
        ),
    }

    all_regressors = []
    for name_prefix, (model_class, param_grid) in param_grids.items():
        all_regressors.extend(
            generate_regressor_grid(
                model_class, param_grid, name_prefix=name_prefix
            )
        )

    return all_regressors


def generate_regressor_grid(model_class, param_grid, name_prefix=None):
    """Generate a list of regressor instance for model experimentation."""
    grid = list(ParameterGrid(param_grid))
    regressors = []
    for i, params in enumerate(grid):
        model = model_class(**params)
        name = name_prefix or model_class.__name__
        name_with_params = f"{name}_{i+1}: " + ", ".join(
            f"{k}={v}" for k, v in params.items()
        )
        regressors.append((name_with_params, model))
    return regressors


def evaluate_model(y_true, y_pred):
    """Calculate RMSE, MAE, and R2 metrics for model predictions."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def get_model_feature_importance(pipeline, feature_names):
    """Extract feature importances or coefficients from a fitted model pipeline."""
    model = pipeline.named_steps["model"]

    # Try feature importance
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))

    # Try coefficients
    if hasattr(model, "coef_"):
        coefs = model.coef_
        # Handle multi-output
        if len(coefs.shape) > 1:
            coefs = coefs[0]
        return dict(zip(feature_names, coefs))

    return None


def run_time_series_cv(
    df_train,
    date_column,
    target_column,
    n_splits=5,
    best_metric_name="MAE",
    output_path=OUTPUT_PATH,
):  # pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-positional-arguments
    """
    Perform time series cross-validation for multiple regression models with preprocessing.

    :param df_train: Input dataset containing features, target, and date column
    :type df_train: pd.DataFrame
    :param date_column: Column name representing the time order to split data for time series CV
    :type date_column: str
    :param target_column: Column name of the target variable to predict
    :type target_column: str
    :param n_splits: Number of splits/folds for TimeSeriesSplit cross-validation, defaults to 5
    :type n_splits: int, optional
    :param best_metric_name: Metric name to select best model ('MAE' minimized, 'R2' maximized),
                            defaults to 'MAE'
    :type best_metric_name: str, optional
    :return: Dictionary containing CV results, best model info, and metric used for selection
    :rtype: dict
    """
    logger.info("---Running Supervised Learning Cross-Validation\n")

    configs = load_configs("features.yml")
    cv_features = configs["cv_features"]

    num_features = (
        cv_features["num_primary"]
        + cv_features["num_amenities"]
        + cv_features["num_ecosocial"]
    )

    cat_features = cv_features["cat_primary"]

    num_features = [col for col in num_features if col in df_train.columns]
    cat_features = [col for col in cat_features if col in df_train.columns]

    regressors = build_regressors()

    df_sorted = df_train.sort_values(date_column).copy()
    x_all = df_sorted.drop(columns=[target_column])

    # Convert output from pd.cut into str
    for col in cat_features:
        if pd.api.types.is_categorical_dtype(x_all[col]) or isinstance(
            x_all[col].dropna().iloc[0], pd.Interval
        ):
            x_all[col] = x_all[col].astype(str)

    y_all = df_sorted[target_column]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []

    for model_name, regressor in regressors:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", MinMaxScaler()),
                        ]
                    ),
                    num_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant", fill_value="missing"
                                ),
                            ),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    cat_features,
                ),
            ]
        )
        fold_metrics = []
        importances_all_folds = []

        for _, (train_idx, val_idx) in enumerate(tscv.split(x_all)):
            x_train = x_all.iloc[train_idx][num_features + cat_features]
            x_val = x_all.iloc[val_idx][num_features + cat_features]
            y_train = y_all.iloc[train_idx]
            y_val = y_all.iloc[val_idx]

            pipeline = Pipeline(
                [("preprocessing", preprocessor), ("model", clone(regressor))]
            )

            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_val)

            metrics = evaluate_model(y_val, y_pred)
            fold_metrics.append(metrics)

            # Get feature names after transformation
            feature_names = []
            try:
                preprocessor_fitted = pipeline.named_steps["preprocessing"]
                feature_names = preprocessor_fitted.get_feature_names_out()
            except (AttributeError, KeyError):
                pass

            importance = get_model_feature_importance(pipeline, feature_names)
            if importance:
                importances_all_folds.append(importance)

        # Aggregate metrics
        avg_metrics = {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in fold_metrics[0]
        }
        std_metrics = {
            f"{k}_std": float(np.std([m[k] for m in fold_metrics]))
            for k in fold_metrics[0]
        }
        all_metrics = {**avg_metrics, **std_metrics}

        # Aggregate importances
        if importances_all_folds:
            df_imp = pd.DataFrame(importances_all_folds).fillna(0)
            avg_importance = df_imp.mean().to_dict()
        else:
            avg_importance = None

        cv_results.append(
            {
                "model": model_name,
                "hyperparameters": regressor.get_params(),
                "features": num_features + cat_features,
                "metrics": all_metrics,
                "feature_importance": avg_importance,
            }
        )

        logger.info("Completed Model: %s", model_name)

    for r in cv_results:
        logger.info(
            "\nModel: %s\nHyperparameters: %s\nFeatures: %s\nMetrics: %s",
            r["model"],
            r["hyperparameters"],
            r["features"],
            r["metrics"],
        )

        if r["feature_importance"]:
            top_feats = sorted(
                r["feature_importance"].items(), key=lambda x: -abs(x[1])
            )[:5]
            logger.info("\nTop features: %s\n", top_feats)

    if best_metric_name == "R2":
        best_result = max(
            cv_results, key=lambda r: r["metrics"][best_metric_name]
        )
    else:
        best_result = min(
            cv_results, key=lambda r: r["metrics"][best_metric_name]
        )

    logger.info(
        "\nBest Model based on %s:\nModel: %s\nHyperparameters: %s\nFeatures: %s\nMetrics: %s\n",
        best_metric_name,
        best_result["model"],
        best_result["hyperparameters"],
        best_result["features"],
        best_result["metrics"],
    )

    cv_results = {
        "cv_results": cv_results,
        "best_model": best_result,
        "best_metric_name": best_metric_name,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=4)

    return cv_results
