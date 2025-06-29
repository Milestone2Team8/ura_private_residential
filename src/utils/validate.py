"""
Validation utilities for merging datasets and checking for missing values.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_missing_data(df_ura, df_name="DataFrame"):
    """Tabulates the number and percentage of missing values in the data."""
    missing_table = df_ura.isnull().sum().to_frame(name="missing_count")
    missing_table["missing_pct"] = (
        missing_table["missing_count"] / len(df_ura)
    ) * 100

    missing_table = missing_table[missing_table["missing_count"] > 0]

    if not missing_table.empty:
        logger.info("Missing Values in %s\n%s\n", df_name, missing_table)
    else:
        logger.info("No Missing Values in %s.\n", df_name)

    return missing_table


def validate_merge(df_ura, df_merged, df_name="DataFrame"):
    """
    Logs the number of rows before and after merging the primary and
    secondary datasets and the number of missing values in the output.
    """
    rows_before = len(df_ura)
    rows_after = len(df_merged)

    logger.info(
        "Completed Merging Datasets\n"
        "Number of rows before merge: %s\n"
        "Number of rows after merge: %s\n",
        rows_before,
        rows_after,
    )

    diagnose_missing_data(df_merged, df_name)

    logger.info(
        "\nColumns with all missing values will be dropped before modeling.\n"
        "Missing values will be imputed with 0, assuming absence of amenities.\n"
    )
