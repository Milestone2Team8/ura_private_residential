"""
Module to perform fuzzy matching between Points of Interest (POI) data and ACRA 
business registry data.

- Normalizes business names using `cleanco`
- Matches POI names to ACRA names using RapidFuzz
- Applies additional fuzzy matching between address components
- Filters matches based on score and incorporation date
- Saves the result to a processed CSV file
"""

import re
from tqdm import tqdm
import pandas as pd
import cleanco
from rapidfuzz import process, fuzz


def normalize_text(text):
    """
    Normalize text by lowercasing, removing non-alpha characters, and stripping whitespace.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def match_acra_poi_data(df_acra, df_poi):
    """
    Match POI names to ACRA business names using fuzzy matching and address proximity scoring.
    Returns a filtered merged DataFrame.
    """
    df_poi['name_clean'] = df_poi['name'].map(cleanco.clean.normalized)
    acra_names = df_acra['name_clean'].tolist()

    def match_poi_name(poi_name):
        match, score, idx = process.extractOne(
            poi_name,
            acra_names,
            scorer=fuzz.token_sort_ratio
        )
        return poi_name, match, score, idx

    poi_names = df_poi['name_clean'].tolist()

    matches = []
    for poi_name in tqdm(poi_names, desc="Matching POIs"):
        result = match_poi_name(poi_name)
        matches.append(result)

    df_matches = pd.DataFrame(matches, columns=[
        "poi_name_clean", "matched_name_clean", "score", "acra_index"
    ])
    df_matches = df_matches.drop_duplicates(subset=['poi_name_clean'])
    df_matches = df_matches[df_matches['score'] >= 80]

    df_poi = pd.merge(
        df_poi,
        df_matches,
        how="left",
        left_on="name_clean",
        right_on="poi_name_clean")

    df_merged = df_poi.merge(
        df_acra,
        how="left",
        left_on="acra_index",
        right_index=True)

    df_merged["vicinity_clean"] = df_merged["vicinity"].apply(normalize_text)
    df_merged["street_name_clean"] = df_merged["street_name"].apply(normalize_text)

    df_merged["fuzzy_score"] = df_merged.apply(
        lambda row: fuzz.token_sort_ratio(
            row["vicinity_clean"], row["street_name_clean"]
        ),
        axis=1)

    df_merged["registration_incorporation_date"] = pd.to_datetime(
        df_merged["registration_incorporation_date"], errors="coerce")

    df_merged = df_merged[df_merged["fuzzy_score"] >= 80]
    df_merged = df_merged[
        df_merged["registration_incorporation_date"].dt.year > 2010]

    df_merged.to_csv("src/data/processed/poi_acra_matched.csv", index=False)

    return df_merged
