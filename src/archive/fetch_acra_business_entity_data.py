"""
Dowloand ACRA Corporate Entity Dataset from data.gov.sg
"""
import glob
import os
import time
import requests
import pandas as pd
import cleanco

# pylint: disable=redefined-outer-name

DATASET_LIST = [
    "d_8575e84912df3c28995b8e6e0e05205a", "d_3a3807c023c61ddfba947dc069eb53f2",
    "d_c0650f23e94c42e7a20921f4c5b75c24", "d_acbc938ec77af18f94cecc4a7c9ec720",
    "d_124a9bd407c7a25f8335b93b86e50fdd", "d_4526d47d6714d3b052eed4a30b8b1ed6",
    "d_b58303c68e9cf0d2ae93b73ffdbfbfa1", "d_fa2ed456cf2b8597bb7e064b08fc3c7c",
    "d_85518d970b8178975850457f60f1e738", "d_478f45a9c541cbe679ca55d1cd2b970b",
    "d_5573b0db0575db32190a2ad27919a7aa", "d_a2141adf93ec2a3c2ec2837b78d6d46e",
    "d_9af9317c646a1c881bb5591c91817cc6", "d_67e99e6eabc4aad9b5d48663b579746a",
    "d_5c4ef48b025fdfbc80056401f06e3df9", "d_300ddc8da4e8f7bdc1bfc62d0d99e2e7",
    "d_181005ca270b45408b4cdfc954980ca2", "d_4130f1d9d365d9f1633536e959f62bb7",
    "d_2b8c54b2a490d2fa36b925289e5d9572", "d_df7d2d661c0c11a7c367c9ee4bf896c1",
    "d_72f37e5c5d192951ddc5513c2b134482", "d_0cc5f52a1f298b916f317800251057f3",
    "d_e97e8e7fc55b85a38babf66b0fa46b73", "d_af2042c77ffaf0db5d75561ce9ef5688",
    "d_1cd970d8351b42be4a308d628a6dd9d3", "d_4e3db8955fdcda6f9944097bef3d2724"
]


def get_acra_dataset(dataset_id):
    """
    Polls and downloads ACRA dataset file by ID.
    """
    while True:
        response = requests.get(
            f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download",
            timeout=10)
        poll = response.json()
        url = poll["data"].get("url")
        if url:
            print("Download URL:", url)
            break
        time.sleep(1)

    resp = requests.get(url, stream=True, timeout=60)
    input_folder = "src/data/input/sg_corporate_entities"
    os.makedirs(input_folder, exist_ok=True)
    file_path = os.path.join(input_folder, f"{dataset_id}.csv")
    with open(file_path, "wb") as file:
        for chunk in resp.iter_content(1024 * 1024):
            file.write(chunk)


def fetch_acra_business_entity_data(dataset_list):
    """
    Downloads and processes ACRA business entity datasets.
    Returns a cleaned DataFrame with normalized entity names.
    """
    for dataset_id in dataset_list:
        get_acra_dataset(dataset_id)

    input_folder = "src/data/input/sg_corporate_entities"
    selected_columns = [
        "uen",
        "entity_name",
        "registration_incorporation_date",
        "entity_type_description",
        "primary_ssic_description",
        "entity_status_description",
        "street_name"]

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    df_acra = pd.concat(
        (pd.read_csv(f, usecols=selected_columns) for f in csv_files),
        ignore_index=True)

    df_acra["name_clean"] = df_acra["entity_name"].apply(
        lambda x: cleanco.clean.normalized(str(x)))

    return df_acra


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Download and process ACRA Singapore business registry datasets.")
    parser.add_argument(
        "--output",
        type=str,
        default="src/data/output/acra_entities.csv",
        help="Path to save the processed ACRA data")
    args = parser.parse_args()

    df_acra = fetch_acra_business_entity_data(DATASET_LIST)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_acra.to_csv(args.output, index=False)
    print(f"Saved processed data to {args.output}")
