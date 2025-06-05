"""
This module merges 2 datasets (ura, ecosocial) using date primary key
"""

import pandas as pd

def merge_dataframes_pri_sec(df_ura : pd.DataFrame, df_ecos : pd.DataFrame):
    """
    Merges ura and ecosocial datasets 
    
    :param df_ura: dataframe to be processed
    :type df_ura: pd.Dataframe
    :param df_ecos: dataframe to be processed
    :type df_ecos: pd.Dataframe
    """
    df_ura['contract_month'] = pd.to_datetime(df_ura['contract_date_dt']).dt.to_period('M')
    df_merged = pd.merge(
        df_ura,
        df_ecos,
        left_on='contract_month',
        right_on='month',
        how='left'
    )

    df_merged.drop(columns=['contract_month'], inplace=True)

    return df_merged
