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
    df_ecos['ecosocial_month'] = pd.to_datetime(df_ecos.index).to_period('M')
    df_merged = pd.merge(
        df_ura,
        df_ecos.reset_index().assign(ecosocial_month=pd. \
            to_datetime(df_ecos.index).to_period('M')),
        left_on='contract_month',
        right_on='ecosocial_month',
        how='left'
    )

    df_merged.drop(columns=['contract_month', 'ecosocial_month'], inplace=True)

    return df_merged
