"""
Module provides helper or utility functions below are applied to specific 
economic indicator datasets, such as the Private home index, CPI, population 
growth, and marriage rates, to prepare them for further analysis.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


def clean_singstat_ds(df_raw):
    """
    This method will be used to clean Singstat datasets.
    It assigns columns as the first row, drops rows with
    null values, and lastly drops rows after footnotes.

    :param df_raw: DataFrame to be cleaned
    :type df_raw: pd.DataFrame
    :return: Updated and cleaned DataFrame
    :rtype: pd.DataFrame
    :raises: Exception
    """
    try:
        df_clean = df_raw.copy()
        df_clean.columns = df_clean.iloc[0]
        df_clean = df_clean[1:]
        df_clean.dropna(axis=1, how="all", inplace=True)

        df_clean = df_clean[df_clean["Data Series"].notna()]
        df_clean.reset_index(drop=True, inplace=True)
        footnotes_idx = df_clean[
            df_clean["Data Series"]
            .astype(str)
            .str.contains("Footnotes", na=False)
        ].index
        if not footnotes_idx.empty:
            df_clean = df_clean[: footnotes_idx[0]]

        df_clean.reset_index(drop=True, inplace=True)

        return df_clean
    except Exception as e:
        raise e

def parse_quarter(row):
    """
    This method converts quarter to month, used for hdb resale index.

    :param row: Dataframe row to be converted to month
    :param row: DataFrame row to be converted to month
    :type row: str
    :return: Updated row as timestamp in months
    :rtype: pd.Timestamp
    :raises: Exception
    """
    try:
        year, quarter = row.split()
        quarter = int(quarter[0])
        month = (quarter - 1) * 3 + 1
        return pd.Timestamp(f"{year}-{month:02d}-01")
    except Exception as e:
        raise e


def clean_and_prepare_dataset(
    df_data,
    cn_data_series,
    cn_melt,
    cn_date="year",
    options=None
):
    """
    This method cleans and prepares the dataset for analysis by
    melting and converting the date column.

    :param df_data: DataFrame to be cleaned and prepared
    :type df_data: pd.DataFrame
    :param cn_data_series: The data series column name to filter on
    :type cn_data_series: str
    :param cn_melt: The melt column name to filter on
    :type cn_melt: str
    :param cn_date: The name of the date column. Defaulted to 'year'
    :type cn_date: str
    :param options: is_quarterly, is_monthly. Defaulted to False
    :type options: dict of booleans
    :return: Prepared DataFrame
    :rtype: pd.DataFrame
    :raises: Exception
    """
    try:
        options = options or {}
        is_monthly = options.get("is_monthly", False)
        is_quarterly = options.get("is_quarterly", False)
        df_data.columns = df_data.columns.astype(str).str.strip()
        df_data_series = df_data[df_data["Data Series"] == cn_data_series].copy()
        df_data_series = df_data_series.melt(
            id_vars=["Data Series"], var_name=cn_date, value_name=cn_melt
        )
        if is_monthly is False:
            if is_quarterly:
                df_data_series["quarter_index"] = df_data_series[cn_date].apply(
                    parse_quarter
                )
                df_data_series.set_index("quarter_index", inplace=True)
                df_data_series.sort_index(inplace=True)
            else:
                df_data_series[cn_date] = (
                    df_data_series[cn_date].astype(float).astype(int).astype(str)
                )
                df_data_series[cn_date] = pd.to_datetime(
                    df_data_series[cn_date].str.strip(), format="%Y", errors="coerce"
                )
                df_data_series["year_index"] = df_data_series[cn_date].dt.year
                df_data_series.set_index("year_index", inplace=True)
                df_data_series = (
                    df_data_series.groupby("year_index")[cn_melt].mean().reset_index()
                )
                df_data_series.sort_index(inplace=True)
        return df_data_series
    except Exception as e:
        raise e


def predict_missing_year(df_data, column, year):
    """
    This method predicts the missing value for a specific year
    using linear regression.

    :param df_data: DataFrame containing the data
    :type df_data: pd.DataFrame
    :param column: The column to predict
    :type column: str
    :param year: The year to predict
    :type year: int
    :return: DataFrame with the predicted year added
    :rtype: pd.DataFrame
    :raises: Exception
    """
    try:
        x = df_data["year_index"].values.reshape(-1, 1)
        y = df_data[column].values
        model = LinearRegression()
        model.fit(x, y)
        predicted_value = model.predict(np.array([[year]]))[0]
        predicted_row = pd.DataFrame({"year_index": [year], column: [predicted_value]})
        return pd.concat([df_data, predicted_row], ignore_index=True)
    except Exception as e:
        raise e


def distribute_yearly_to_monthly_rate(
    df_data, column, start_year, end_year
):
    """
    This method distributes yearly rates to monthly rates.

    :param df_data: DataFrame containing the yearly data
    :type df_data: pd.DataFrame
    :param column: The column containing the yearly rates
    :type column: str
    :param start_year: The start year for distribution
    :type start_year: int
    :param end_year: The end year for distribution
    :type end_year: int
    :return: DataFrame with monthly rates
    :rtype: pd.DataFrame
    :raises: Exception
    """
    try:
        monthly_rates = []
        for _, row in df_data.iterrows():
            year = int(row["year_index"]) - 1
            monthly_rate = ((1 + row[column]) ** (1 / 12)) - 1
            for month in range(1, 13):
                monthly_rates.append(
                    {
                        "year": year,
                        "month": pd.Timestamp(
                            year=year, month=month, day=1
                        )
                        + pd.offsets.MonthEnd(0),
                        f"monthly_{column}": monthly_rate,
                    }
                )
        df_monthly_rates = pd.DataFrame(monthly_rates)
        df_monthly_rates = df_monthly_rates[
            (df_monthly_rates["month"] >= f"{start_year}-01-01")
            & (df_monthly_rates["month"] <= f"{end_year}-12-31")
        ]
        df_monthly_rates.drop(columns=["year"], inplace=True)
        df_monthly_rates.set_index("month", inplace=True)
        df_monthly_rates.index = df_monthly_rates.index.to_period("M")
        return df_monthly_rates
    except Exception as e:
        raise e


def prepare_housing_cpi(
        df_cpi_clean: pd.DataFrame,
        base_date: str = "2019-11-30",
        start_date: str = "2019-12-01",
        end_date: str = "2025-04-01",
    ) -> pd.DataFrame:
    """
    This method cleans and prepares the Housing & Utilities CPI
    dataset by converting the date column, calculating the adjusted
    CPI, and computing the CPI growth rate.

    :param df_cpi_clean: Cleaned CPI dataframe
    :type df_cpi_clean: pd.DataFrame
    :param base_date: The base date to calculate the adjusted CPI,
                      defaults to '2019-11-30'
    :type base_date: str, optional
    :param start_date: The start date for the dataset, defaults to '2019-12-01'
    :type start_date: str, optional
    :param end_date: The end date for the dataset, defaults to '2025-04-01'
    :type end_date: str, optional
    :return: DataFrame with monthly CPI rates
    :rtype: pd.DataFrame
    :raises: Exception
    """
    try:
        housing_cpi = clean_and_prepare_dataset(
            df_cpi_clean, "  Housing & Utilities", "cpi_housing", "month",
                options={"is_monthly": True}
        )
        housing_cpi["month"] = pd.to_datetime(
            housing_cpi["month"].str.strip(), format="%Y %b", errors="coerce"
        )
        df_cpi = housing_cpi[
            (housing_cpi["month"] >= start_date) &
            (housing_cpi["month"] <= end_date)
        ].reset_index(drop=True)
        df_cpi["month"] = df_cpi["month"] - pd.offsets.MonthEnd(1)

        base_cpi = (
            df_cpi[df_cpi["month"] == base_date]["cpi_housing"].values[0]
        )
        df_cpi["cpi_adjusted"] = df_cpi["cpi_housing"] / base_cpi
        df_cpi.drop(columns=["Data Series"], inplace=True)
        df_cpi.set_index("month", inplace=True)
        df_cpi.sort_index(inplace=True)

        df_cpi = df_cpi.apply(pd.to_numeric)

        df_cpi["cpi"] = df_cpi["cpi_adjusted"].pct_change()
        df_cpi.drop(columns=["cpi_housing"], inplace=True)
        df_cpi.drop(columns=["cpi_adjusted"], inplace=True)
        df_cpi = df_cpi[df_cpi.index >= "2020-01-31"]
        df_cpi.index = df_cpi.index.to_period("M")
        return df_cpi
    except Exception as e:
        raise e
