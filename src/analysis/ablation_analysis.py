"""
Ablation analysis function and identify feature importance
"""

import logging
import json
import seaborn as sns
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.load_configs import load_configs_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """
    Dataset params
    """
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

@dataclass
class FeatureSet:
    """
    feature params
    """
    numeric: list[str]
    categorical: list[str]
    added_feature: str

@dataclass
class AblationInput:
    """
    Ablation params
    """
    pipeline: any
    data: Dataset
    features: FeatureSet
    model: BaseEstimator


def _load_ablation_config_and_metrics(config_path, cv_result_path):
    """
    Load ablation features configuration and baseline MAE from JSON file.

    :param config_path: Path to the feature configuration YAML file
    :type config_path: str
    :param cv_result_path: Path to the cross-validation results JSON file
    :type cv_result_path: str
    :return: Tuple of (list of ablation features, baseline MAE)
    :rtype: tuple[list[str], float]
    """
    all_features = load_configs_file(config_path)["ablation_features"]
    with open(cv_result_path, "r", encoding="utf-8") as f:
        baseline_mae = json.load(f)["best_model"]["metrics"]["MAE"]
    return all_features, baseline_mae


def get_updated_preprocessor(original_pipeline, new_num_features, new_cat_features):
    """Generate preprocessor with new features
    on top of old ones.
    :param original_pipeline: Fitted sklearn Pipeline from cross-validation
    :type original_pipeline: Pipeline
    :param new_num_features: Additional numerical features
    :type new_num_features: list of str
    :param new_cat_features: Additional categorical features
    :type new_num_features: list of str
    """

    old_transformers = original_pipeline.named_steps["preprocessing"].transformers

    num_transformer = [t for t in old_transformers if t[0] == "num"][0][1]
    cat_transformer = [t for t in old_transformers if t[0] == "cat"][0][1]

    new_preprocessor = ColumnTransformer(
        transformers=[
            ("num", clone(num_transformer), new_num_features),
            ("cat", clone(cat_transformer), new_cat_features)
        ]
    )
    return new_preprocessor


def _convert_categoricals_as_str(df, columns):
    """
    Convert categorical or interval columns 
    in the given DataFrame to string type.

    :param df: DataFrame to convert columns
    :type df: pd.DataFrame
    :param columns: List of column names to check and convert
    :type columns: list of str
    :return: Updated DataFrame with converted columns
    :rtype: pd.DataFrame
    """
    for col in columns:
        if col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]) \
            or isinstance(df[col].dropna().iloc[0], pd.Interval):
                df[col] = df[col].astype(str)
    return df

def _evaluate_with_added_feature(inputs: AblationInput) -> float:
    """
    Evaluate the model performance after adding a new feature using updated pipeline.

    :param inputs: Structured input containing all necessary components for evaluation
    :type inputs: AblationInput
    :return: MAE score after adding the feature
    :rtype: float
    """
    num, cat = list(inputs.features.numeric), list(inputs.features.categorical)
    feature = inputs.features.added_feature

    x_train_base = inputs.data.x_train
    x_test_base = inputs.data.x_test

    if feature in x_train_base.columns and (
        x_train_base[feature].dtype == "object"
        or str(x_train_base[feature].dtype).startswith("category")
    ):
        cat.append(feature)
    else:
        num.append(feature)

    x_train = x_train_base[num + cat]
    x_test = x_test_base[num + cat]

    preprocessor = get_updated_preprocessor(inputs.pipeline, num, cat)
    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", clone(inputs.model))
    ])
    model_pipeline.fit(x_train, inputs.data.y_train)
    preds = model_pipeline.predict(x_test)
    return mean_absolute_error(inputs.data.y_test, preds)

def extract_top_features_from_importance(path: str,
    top_k: int = 10) -> tuple[list[str], list[str]]:
    """
    Extract top unique feature names.

    :param path: Path to the JSON file.
    :type path: str
    :param top_k: Number of top unique features based on importance score.
    :type top_k: int
    :return: Tuple containing two lists — top numeric and top categorical 
    :rtype: tuple[list[str], list[str]]
    """
    with open(path, "r", encoding="utf-8") as f:
        feature_importance = json.load(f)["best_model"]["feature_importance"]

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    seen = set()
    top_features = []

    for full_name, _ in sorted_features:
        if full_name.startswith("num__"):
            name = full_name.split("__", 1)[1]
        elif full_name.startswith("cat__"):
            name = full_name.split("__", 1)[1].split("_", 1)[0]
        else:
            continue

        if name not in seen:
            seen.add(name)
            top_features.append((full_name, name))

        if len(top_features) == top_k:
            break

    num_features = [name for full_name, name in top_features if full_name.startswith("num__")]
    cat_features = [name for full_name, name in top_features if full_name.startswith("cat__")]

    return num_features, cat_features


def plot_ablation(diffs: dict, save_path: str = "./src/data/plot/ablation_additive_analysis.png"):
    """
    Plot chart to visualize feature ablation effects.

    :param diffs: Dictionary mapping feature name
    :type diffs: dict[str, float]
    :param save_path: Path to save the generated plot image
    :type save_path: str
    """
    diffs_sorted = dict(sorted(diffs.items(), key=lambda item: item[1], reverse=True))
    df_plot = pd.DataFrame({
    "feature": list(diffs_sorted.keys()),
    "mae_reduction": list(diffs_sorted.values())
    })

    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df_plot, x="mae_reduction", y="feature", jitter=True, palette="coolwarm", size=10)
    plt.axvline(0, linestyle='--', color='black')
    plt.xlabel("MAE Reduction After Adding Feature")
    plt.title("Feature Additive Ablation Analysis")
    plt.tight_layout()
    plt.savefig(save_path)


def perform_ablation_analysis(best_model_pipeline, df_input_train,
        df_input_test, target_column="target_price"):
    """
    Performs ablation analysis on the best model pipeline by adding one feature at a time
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

    all_ablation_features, baseline_mae = _load_ablation_config_and_metrics(
        "features.yml", "./src/data/output/all_models_features_results.json"
    )
    logger.info("Loaded baseline MAE: %.4f", baseline_mae)

    num_features, cat_features = extract_top_features_from_importance(
    "./src/data/output/all_models_features_results.json"
    )

    addable = [f for f in all_ablation_features if f not in set(num_features + cat_features)]
    logger.info("Top 10 features: %s and %s", num_features, cat_features)

    logger.info("Features to be added: %s", addable)

    df_input_train = df_input_train.sort_values("contract_date_dt").copy()
    df_input_test = df_input_test.sort_values("contract_date_dt").copy()

    x_train_base = df_input_train.drop(columns=[target_column])
    x_test_base = df_input_test.drop(columns=[target_column])

    x_train_base = _convert_categoricals_as_str(x_train_base, cat_features + addable)
    x_test_base = _convert_categoricals_as_str(x_test_base, cat_features + addable)

    results = {}
    for feature in addable:
        inputs = AblationInput(
            pipeline=best_model_pipeline,
            data=Dataset(x_train_base, x_test_base,
                df_input_train[target_column], df_input_test[target_column]),
            features=FeatureSet(num_features, cat_features, feature),
            model=best_model_pipeline.named_steps["model"]
        )

        results[feature] = _evaluate_with_added_feature(inputs)

    diffs = {k: baseline_mae - v for k, v in results.items()}
    plot_ablation(diffs)
