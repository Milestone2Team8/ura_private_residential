"""
Module provides helper or utility functions below are applied to specific 
economic indicator datasets, such as the Private home index, CPI, population 
growth, and marriage rates, to prepare them for further analysis.
"""

import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



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

def parse_quarter(row, separator = None):
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
        if separator:
            year, quarter = row.split(separator)
        else:
            year, quarter = row.split()
        quarter = int(re.search(r"\d", quarter).group())
        end_month = quarter * 3
        return pd.Timestamp(f"{year}-{end_month:02d}-01")
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


def distribute_quarterly_to_monthly_rate(
    df_data, column, value_column, start_date, end_date
):
    """
    This method distributes quarterly percentage rates to monthly rates using compound growth.

    :param df_data: DataFrame indexed by month (PeriodIndex), containing quarterly data
    :type df_data: pd.DataFrame
    :param column: The column containing quarterly percentage rates (e.g., 'quarterly_growth')
    :type column: str
    :param value_column: The column containing base index values (e.g., 'price_index')
    :type value_column: str
    :param start_date: The start date for filtering (inclusive)
    :type start_date: str
    :param end_date: The end date for filtering (inclusive)
    :type end_date: str
    :return: DataFrame with distributed monthly rates and monthly index values
    :rtype: pd.DataFrame
    :raises: Exception
    """
    try:
        records = []

        def distribute_row(row):
            end = row.Index.to_timestamp()
            rate_pct = getattr(row, column)
            end_val = getattr(row, value_column)

            if pd.isna(rate_pct) or pd.isna(end_val):
                return []

            r = (1 + rate_pct / 100) ** (1 / 3) - 1
            base = end_val / ((1 + r) ** 2)

            return [{
                "month": (end - pd.DateOffset(months=2 - j) + pd.offsets.MonthEnd(0)),
                "monthly_rate": round(r * 100, 6),
                f"monthly_{value_column}": base * ((1 + r) ** j),
                "quarterly_rate": rate_pct
            } for j in range(3)]

        for row in df_data.sort_index().itertuples():
            records.extend(distribute_row(row))

        df_monthly = pd.DataFrame(records)
        df_monthly = df_monthly[
            (df_monthly["month"] >= pd.to_datetime(start_date)) &
            (df_monthly["month"] <= pd.to_datetime(end_date))
        ]
        df_monthly.set_index("month", inplace=True)
        df_monthly.index = df_monthly.index.to_period("M")

        return df_monthly

    except Exception as e:
        raise e


def prepare_housing_cpi(
        df_cpi_housing: pd.DataFrame,
        start_date: str = "2019-12-01",
        end_date: str = "2025-04-01",
    ) -> pd.DataFrame:
    """
    This method cleans and prepares the Housing & Utilities CPI
    dataset by converting the date column, calculating the adjusted
    CPI, and computing the CPI growth rate.

    :param df_cpi_housing: Cleaned CPI dataframe
    :type df_cpi_housing: pd.DataFrame
    :param start_date: The start date for the dataset, defaults to '2019-12-01'
    :type start_date: str, optional
    :param end_date: The end date for the dataset, defaults to '2025-04-01'
    :type end_date: str, optional
    :return: DataFrame with monthly CPI rates
    :rtype: pd.DataFrame
    :raises: Exception
    """
    try:
        start_date_dt = pd.to_datetime(start_date)
        base_date_dt = start_date_dt - pd.offsets.MonthEnd(1)

        df_cpi_housing["month"] = pd.to_datetime(
            df_cpi_housing["month"].str.strip(), format="%Y %b", errors="coerce"
        )
        df_cpi = df_cpi_housing[
            (df_cpi_housing["month"] >= start_date) &
            (df_cpi_housing["month"] <= end_date)
        ].reset_index(drop=True)
        df_cpi["month"] = df_cpi["month"] - pd.offsets.MonthEnd(1)

        base_cpi = (
            df_cpi[df_cpi["month"] == base_date_dt]["cpi_housing"].values[0]
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


def clean_sora(df_sora):
    """
    This method cleans and prepares the SORA dataset by adjusting
    the column headers, converting the date column, and resampling
    the data to monthly end frequency.

    :param df_sora: Raw SORA dataframe
    :type df_sora: pd.DataFrame
    :return: Cleaned and prepared SORA dataframe
    :rtype: pd.DataFrame
    :raises: Exception
    """
    try:
        df_clean = df_sora.copy()
        df_clean.columns = (
            ['Value Date', 'Unnamed 1', 'Index', 'Publication Date', 'SORA']
        )
        df_clean = df_clean.drop(columns=['Unnamed 1', 'Index'])
        df_clean = df_clean.drop(0).reset_index(drop=True)

        df_clean['Publication Date'] = (
            pd.to_datetime(
                df_clean['Publication Date'],
                format='%d-%b-%y',
                errors='coerce')
        )
        df_clean['SORA'] = pd.to_numeric(df_clean['SORA'], errors='coerce')
        df_clean.set_index('Publication Date', inplace=True)
        return df_clean
    except Exception as e:
        raise e
