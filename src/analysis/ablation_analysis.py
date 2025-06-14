"""
Ablation analysis function and identify feature importance
"""

import logging
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from src.utils.load_configs import load_configs
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_new_pipeline_with_addon_features(
    model,
    num_features,
    cat_features
    ):
    """
    Build new pipeline using specified numeric and categorical features.

    :param model: Trained model in the pipeline
    :type model: sklearn.base.BaseEstimator
    :param num_features: List of numeric feature names
    :type num_features: list[str]
    :param cat_features: List of categorical feature names
    :type cat_features: list[str]
    :return: A sklearn Pipeline object with preprocessing and model steps
    :rtype: sklearn.pipeline.Pipeline
    """
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

    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", clone(model))
    ])


def run_ablation_analysis(best_model_pipeline, df_input_train,
        df_input_test, target_column="target_price"):
    """
    Performs ablation analysis on the best model pipeline by removing one feature at a time
    and observing the change in Mean Absolute Error (MAE) on the test set.

    :param best_model_pipeline: Fitted sklearn Pipeline from cross-validation
    :type best_model_pipeline: Pipeline
    :param df_input_train: Training DataFrame
    :type df_input_train: pd.DataFrame
    :param df_input_test: Test DataFrame
    :type df_input_test: pd.DataFrame
    :param target_column: Name of the target column
    :type target_column: str
    """
    
    all_ablation_features = load_configs("features.yml")["ablation_features"]

    with open("./src/data/output/cv_results.json", "r", encoding="utf-8") as f:
        cv_results = json.load(f)
    baseline_mae = cv_results["best_model"]["metrics"]["MAE"]
    logger.info("Loaded baseline MAE: %.4f", baseline_mae)

    num_features = best_model_pipeline.named_steps["preprocessing"].transformers_[0][2]
    cat_features = best_model_pipeline.named_steps["preprocessing"].transformers_[1][2]
    pipeline_features = set(num_features + cat_features)

    addable_features = [f for f in all_ablation_features if f not in pipeline_features]
    logger.info("Features to be added: %s", addable_features)

    model = best_model_pipeline.named_steps["model"]
    ablation_results = {}

    df_train_sorted = df_input_train.sort_values("contract_date_dt").copy()
    df_test_sorted = df_input_test.sort_values("contract_date_dt").copy()

    x_train_all = df_train_sorted.drop(columns=[target_column])
    x_test_all = df_test_sorted.drop(columns=[target_column])

    for col in cat_features + addable_features:
        if col in x_train_all.columns:
            if pd.api.types.is_categorical_dtype(x_train_all[col]) or isinstance(
                x_train_all[col].dropna().iloc[0], pd.Interval
            ):
                x_train_all[col] = x_train_all[col].astype(str)
        if col in x_test_all.columns:
            if pd.api.types.is_categorical_dtype(x_test_all[col]) or isinstance(
                x_test_all[col].dropna().iloc[0], pd.Interval
            ):
                x_test_all[col] = x_test_all[col].astype(str)

    for feature in addable_features:
        updated_num = list(num_features)
        updated_cat = list(cat_features)

        if feature in x_train_all.columns and (
            x_train_all[feature].dtype == "object"
            or str(x_train_all[feature].dtype).startswith("category")
        ):
            updated_cat.append(feature)
        else:
            updated_num.append(feature)

        x_train = x_train_all[updated_num + updated_cat]
        x_test = x_test_all[updated_num + updated_cat]

        model_ablate_pipeline = _build_new_pipeline_with_addon_features(
            model, updated_num, updated_cat
        )
        model_ablate_pipeline.fit(x_train, df_train_sorted[target_column])
        y_pred = model_ablate_pipeline.predict(x_test)
        mae = mean_absolute_error(df_test_sorted[target_column], y_pred)
        ablation_results[feature] = mae

    diffs = {k: baseline_mae - v for k, v in ablation_results.items()}

    plt.figure(figsize=(10, 6))
    plt.barh(list(diffs.keys()), list(diffs.values()))
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("MAE Reduction After Adding Feature")
    plt.title("Feature Additive Ablation Analysis")
    plt.tight_layout()
    plt.savefig("./src/data/plot/ablation_additive_analysis.png")