import os
import pandas as pd
from datasets import load_dataset

# ================================================================
# ----------------------- LOADING HELPERS ------------------------
# ================================================================

def load_csv_from_github(
    csv_path:str,
    base_url:str="https://media.githubusercontent.com/media/d-noe/NLP_DH_PSL_F26/refs/heads/main/",
):
    """
    Helper function to load a `.csv` file stored in the repository as a pandas.DataFrame.
    Originally implemented to centralize loading process, after the change to Git LFS, base URL switched from
        https://raw.githubusercontent.com/d-noe/NLP_DH_PSL_Fall2025/main/
     to https://media.githubusercontent.com/media/d-noe/NLP_DH_PSL_Fall2025/refs/heads/main/
    Input:
        - csv_path: [str]
            Path of the file within the repository (e.g.; "data/topic_data/ungdc.csv")
        - base_url: [str] | default: "https://media.githubusercontent.com/media/d-noe/NLP_DH_PSL_Fall2025/refs/heads/main/"
            Base URL for the stored files, do not mess with it.
    Returns:
        - pandas.DataFrame
            Loaded csv as a pandas DataFrame object
    """
    csv_url = os.path.join(base_url,csv_path)
    df = pd.read_csv(csv_url, encoding='utf-8')
    return df

def load_dataset_from_github(
    dataset_path:str=None,
    base_url:str="https://raw.githubusercontent.com/d-noe/NLP_DH_PSL_F26/main/",
    data_files:dict=None,
    splits:list=["train", "validation", "test"],
):
    """
    Helper function to load a dataset stored in the repository as a HuggingFace DatasetDict (arrow-formatted).
    Intended to generalize the data loading process for multi-split datasets (train/validation/test)
    hosted remotely (e.g. via GitHub).

    Notes:
        - Either `dataset_path` or `data_files` must be provided.
        - If `data_files` is None, the function will automatically construct split URLs
          assuming Arrow-formatted files are stored under the specified dataset path.
        - The logic under `if data_files is None` might be fragile if repository structure
          differs from the expected format.

    Inputs:
        - dataset_path: [str | None]
            Relative path to the dataset folder within the repository (e.g.; "data/literary_sft/sampled_chunks").
            Required if `data_files` is not explicitly provided.
        - base_url: [str] | default: "https://raw.githubusercontent.com/d-noe/NLP_DH_PSL_Fall2025/main/"
            Base URL for the stored files (e.g., raw GitHub URL). Typically left unchanged.
        - data_files: [dict | None]
            Optional manual mapping of dataset splits to their remote Arrow file URLs.
            Example: {"train": "...", "validation": "...", "test": "..."}
        - splits: [list[str]] | default: ["train", "validation", "test"]
            Expected dataset splits to load. Only used if `data_files` is None.

    Returns:
        - dataset: datasets.DatasetDict
            Loaded dataset object, with keys corresponding to each split.
    """
    assert ( not (dataset_path is None)) or (not (data_files is None))

    if data_files is None:
        dataset_url = os.path.join(base_url,dataset_path)
        data_files = {
            s: os.path.join(dataset_url, f"{s}/data-00000-of-00001.arrow") # data file for each split
            for s in splits
        }
    return load_dataset("arrow", data_files=data_files)
