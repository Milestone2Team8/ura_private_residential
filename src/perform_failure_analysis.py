"""
Module for training models, predicting, and explaining results with SHAP plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.utils.convert_interval_to_str import convert_interval_to_str
from src.utils.load_configs import load_features

OUTPUT_PATHS = {
    "test_results_raw": Path("./src/data/output/test_results_raw.csv"),
    "test_results_trans": Path("./src/data/output/test_results_trans.csv"),
    "shap_plot_dir": Path("./src/data/plot"),
}

# pylint: disable=too-many-locals


def preprocess_data(df_train, df_test, target_column="target_price"):
    """Prepare training and testing feature matrices and target vector."""
    df_train = df_train.dropna(axis=1, how="all")
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    num_features, cat_features = load_features(df_train, "all_features")
    x_train = df_train[num_features + cat_features]
    y_train = df_train[target_column]
    x_test = df_test[num_features + cat_features]

    x_train = convert_interval_to_str(x_train, cat_features)
    x_test = convert_interval_to_str(x_test, cat_features)

    return df_train, df_test, x_train, y_train, x_test


def train_predict(
    model_pipeline, df_train, df_test, target_column="target_price"
):
    """Train model pipeline, predict on test set, and save results with errors."""
    _, df_test, x_train, y_train, x_test = preprocess_data(df_train, df_test)

    fitted_model_pipeline = model_pipeline.fit(x_train, y_train)
    df_test["pred"] = fitted_model_pipeline.predict(x_test)
    df_test["abs_error"] = abs(df_test[target_column] - df_test["pred"])

    df_test = df_test[
        ["project", "street", target_column, "pred", "abs_error"]
        + list(x_test.columns)
    ]
    df_test.to_csv(OUTPUT_PATHS["test_results_raw"], index=True)

    return df_test, fitted_model_pipeline


def explain_prediction_waterfall(
    fitted_model_pipeline, df_train, df_test, indices, prefix="explanation"
):
    """SHAP waterfall plot explanations for selected test records."""
    _, df_test, x_train, _, x_test = preprocess_data(df_train, df_test)

    preprocessor = fitted_model_pipeline.named_steps["preprocessing"]
    model = fitted_model_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    x_train_processed = preprocessor.transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    df_x_train_processed = pd.DataFrame(
        x_train_processed, columns=feature_names
    )
    df_x_test_processed = pd.DataFrame(x_test_processed, columns=feature_names)

    explainer = shap.Explainer(model, df_x_train_processed)

    shap_values = explainer(
        df_x_test_processed.iloc[indices], check_additivity=False
    )

    df_x_test_processed.to_csv(OUTPUT_PATHS["test_results_trans"], index=True)

    for i, idx in enumerate(indices):
        shap.plots.waterfall(shap_values[i], show=False)
        output_path = OUTPUT_PATHS["shap_plot_dir"] / f"{prefix}_{idx}.png"
        plt.tight_layout()
        plt.gcf().set_size_inches(30, 6)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


def plot_mean_abs_error_by_district_line(
    df_test, district_col="district", error_col="abs_error"
):
    """
    Groups the dataframe by district and plots the mean absolute error as a line plot.
    """
    mean_error_by_district = (
        df_test.groupby(district_col, observed=False)[error_col]
        .mean()
        .sort_values()
    )

    output_path = (
        OUTPUT_PATHS["shap_plot_dir"] / "mean_abs_error_by_district.png"
    )

    plt.figure(figsize=(12, 6))
    mean_error_by_district.plot(kind="line", marker="o")
    plt.title("Mean Absolute Error by District", fontweight="bold")
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("District")
    plt.xticks(
        ticks=range(len(mean_error_by_district)),
        labels=mean_error_by_district.index,
        rotation=45,
        ha="right",
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")


def perform_failure_analysis(model_pipeline, df_train, df_test, indices):
    """
    Runs training, prediction, and SHAP waterfall plot explanations for selected test records.

    This function fits the given model pipeline on the training data, generates predictions
    and absolute errors on the test data, and then produces SHAP waterfall plots for the
    specified test record indices. The SHAP plots are saved as PNG files in the output directory.

    :param model_pipeline: Trained sklearn pipeline or compatible model to fit and explain.
    :type model_pipeline: object
    :param df_train: Training DataFrame including features and target column.
    :type df_train: pandas.DataFrame
    :param df_test: Test DataFrame including features and target column.
    :type df_test: pandas.DataFrame
    :param indices: List of row indices in the test set for which to generate SHAP plots.
    :type indices: list[int]
    """
    df_test, fitted_model_pipeline = train_predict(
        model_pipeline, df_train, df_test
    )

    explain_prediction_waterfall(
        fitted_model_pipeline, df_train, df_test, indices
    )

    plot_mean_abs_error_by_district_line(df_test)
