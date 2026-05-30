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
        fuzzy=True,
    )

    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(extract_path)
