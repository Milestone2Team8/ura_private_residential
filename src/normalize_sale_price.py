"""
This module to normalize sale price for each transaction
by removing cumulative cpi effect
"""

from pathlib import Path
import pandas as pd

OUTPUT_NORMALIZED_PATH = Path("./src/data/output/clean_merged_ura_data.csv")

def normalize_prices(df_input: pd.DataFrame, output_path: Path = OUTPUT_NORMALIZED_PATH):
    """
    Process dataframe and normalize transaction prices using cumulative cpi.

    :param df_input: dataframe to be processed
    :type df_input: pd.Dataframe
    :param output_path: Path to the output normalized csv
    :type output_path: Path
    :return: dataFrame with normalized transaction price
    :rtype: pd.DataFrame
    """

    cpi_each_month = df_input[["month", "cpi"]].drop_duplicates()
    cpi_each_month = cpi_each_month.sort_values("month")
    cpi_each_month["cpi_calc"] = 1 + cpi_each_month["cpi"]
    cpi_each_month["cpi_accum"] = cpi_each_month["cpi_calc"].cumprod()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(cpi_each_month)

    df_input = df_input.merge(cpi_each_month[["month", "cpi_accum"]], on="month", how="left")

    df_input["target_price_cpi_adjusted"] = round(df_input["target_price"] \
        * 1 / df_input["cpi_accum"])

    df_input.to_csv(output_path, index=False)

    return df_input
