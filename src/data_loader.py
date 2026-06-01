# src/data_loader.py

import os
import zipfile
import gdown


def download_dataset(
    file_id: str,
    output_zip: str = "data.zip",
    extract_path: str = ".",
):
    if os.path.exists(output_zip):
        os.remove(output_zip)

    gdown.download(
        id=file_id,
        output=output_zip,
        quiet=False,
    )

    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def ensure_dataset(
    file_id: str,
    data_dir: str = "data",
    output_zip: str = "data.zip",
):
    """
    Download and extract the dataset only if data_dir
    does not already exist.
    """

    if os.path.isdir(data_dir):
        print(f"Dataset already present: {data_dir}")
        return

    print("Dataset not found.")
    print("Downloading from Google Drive...")

    gdown.download(
        id=file_id,
        output=output_zip,
        quiet=False,
    )

    print("Extracting archive...")

    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(".")

    print("Dataset ready.")
