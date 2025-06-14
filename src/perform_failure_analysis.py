"""
Module for training models, predicting, and explaining results with SHAP plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import shap

from src.utils.convert_interval_to_str import convert_interval_to_str
from src.utils.load_configs import load_features

OUTPUT_PATHS = {
    "test_results": Path("./src/data/output/test_results.csv"),
    "shap_plot_dir": Path("./src/data/plot"),
}


def preprocess_data(df_train, df_test, target_column="target_price"):
    """Prepare training and testing feature matrices and target vector."""
    df_train = df_train.dropna(axis=1, how="all")
    num_features, cat_features = load_features(df_train, "all_features")

    x_train = df_train[num_features + cat_features]
    y_train = df_train[target_column]
    x_test = df_test[num_features + cat_features]

    x_train = convert_interval_to_str(x_train, cat_features)
    x_test = convert_interval_to_str(x_test, cat_features)

    return x_train, y_train, x_test


def train_predict(
    model_pipeline, df_train, df_test, target_column="target_price"
):
    """Train model pipeline, predict on test set, and save results with errors."""
    x_train, y_train, x_test = preprocess_data(df_train, df_test)

    model_pipeline.fit(x_train, y_train)
    df_test["pred"] = model_pipeline.predict(x_test)
    df_test["abs_error"] = abs(df_test[target_column] - df_test["pred"])

    df_test = df_test[
        ["project", "street", target_column, "pred", "abs_error"]
        + list(x_test.columns)
    ]
    df_test.to_csv(OUTPUT_PATHS["test_results"], index=True)

    return df_test, model_pipeline


def explain_prediction_waterfall(
    fitted_model_pipeline, df_train, df_test, indices, prefix="explanation"
):
    """SHAP waterfall plot explanations for selected test records."""
    *_, x_test = preprocess_data(df_train, df_test)

    preprocessor = fitted_model_pipeline.named_steps["preprocessing"]
    model = fitted_model_pipeline.named_steps["model"]
    x_test_processed = preprocessor.transform(x_test)

    explainer = shap.Explainer(model, x_test_processed)
    shap_values = explainer(x_test_processed)

    for i in indices:
        shap.plots.waterfall(shap_values[i], show=False)
        output_path = OUTPUT_PATHS["shap_plot_dir"] / f"{prefix}_{i}.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


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
    _, fitted_model_pipeline = train_predict(model_pipeline, df_train, df_test)
    explain_prediction_waterfall(
        fitted_model_pipeline, df_train, df_test, indices
    )
