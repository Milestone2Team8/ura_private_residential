"""
Pipeline runner script.

This module executes the data cleaning function and prepares the dataset
for downstream unsupervised and supervised learning tasks.
"""

from src.clean_ura_data import clean_ura_data


def run_all():
    """Cleans raw ura private residential data and prepares it for modeling tasks."""
    df_clean = clean_ura_data()

    # TO-DO: Left join secondary data to df_condo_clean

    return df_clean


if __name__ == "__main__":
    run_all()
