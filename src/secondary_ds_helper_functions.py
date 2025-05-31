import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

"""
The helper or utility functions below are applied to specific economic indicator datasets, 
such as the HDB resale index, CPI, population growth, and marriage rates, to prepare them 
for further analysis.
"""

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
        raise Exception(f"clean_singstat_ds has an exception{e}")


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
        raise Exception(f"parse_quarter has an exception{e}")


def clean_and_prepare_dataset(
    df_data,
    cn_data_series,
    cn_melt,
    cn_date="year",
    is_monthly=False,
    is_quarterly=False,
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
    :param is_quarterly: Flag indicating if the data is quarterly.
                         Defaulted to False
    :type is_quarterly: bool
    :return: Prepared DataFrame
    :rtype: pd.DataFrame
    :raises: Exception
    """
    try:
        df_data.columns = df_data.columns.astype(str).str.strip()
        df_data_series = df_data[df_data["Data Series"] == cn_data_series].copy()
        df_data_series = df_data_series.melt(
            id_vars=["Data Series"], var_name=cn_date, value_name=cn_melt
        )
        if is_monthly == False:
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
        raise Exception(
            f"clean_and_prepare_dataset has an exception{e}"
        )


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
        raise Exception(f"predict_missing_year has an exception{e}")


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
        raise Exception(
            f"distribute_yearly_to_monthly_rate has an exception{e}"
        )


def distribute_quarterly_to_monthly_rate(
    df_data, column, start_year, end_year
):
    """
    This method distributes quarterly rates to monthly rates.

    :param df_data: DataFrame containing the quarterly data
    :type df_data: pd.DataFrame
    :param column: The column containing the quarterly rates
    :type column: str
    :param start_date: The start date for distribution
    :type start_date: str
    :param end_date: The end date for distribution
    :type end_date: str
    :return: DataFrame with monthly rates
    :rtype: pd.DataFrame
    :raises: Exception
    """
    try:
        monthly_rates = []
        for i in range(len(df_data) - 1):
            start_date = df_data.index[i]
            end_date = df_data.index[i + 1]
            quarterly_index = df_data.iloc[i + 1][column]
            hdb_resale_index = df_data.iloc[i]["hdb_resale_index"]
            if np.isnan(quarterly_index):
                monthly_index = 0
            else:
                monthly_index = (1 + quarterly_index) ** (1 / 3) - 1
            for j, month in enumerate(
                pd.date_range(start=start_date, periods=3, freq="MS")
            ):
                monthly_rates.append(
                    {
                        "month": month + pd.offsets.MonthEnd(0),
                        f"monthly_{column}": monthly_index,
                        "monthly_hdb_resale_index": hdb_resale_index
                        * ((1 + monthly_index) ** j),
                        "quarterly_index": quarterly_index,
                    }
                )

        df_monthly_rates = pd.DataFrame(monthly_rates)
        df_monthly_rates = df_monthly_rates[
            (df_monthly_rates["month"] >= f"{start_year}-01-01")
            & (df_monthly_rates["month"] <= f"{end_year}-12-31")
        ]
        df_monthly_rates.set_index("month", inplace=True)
        df_monthly_rates.index = df_monthly_rates.index.to_period("M")
        return df_monthly_rates
    except Exception as e:
        raise Exception(
            f"distribute_quarterly_to_monthly_rate has an exception{e}"
        )