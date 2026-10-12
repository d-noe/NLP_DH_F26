"""
Data files to be downloaded from:
    https://github.com/impresso/CLEF-HIPE-2020/tree/master/data/v1.4/fr
For this script, the following files:
    HIPE-data-v1.4-train-fr.tsv
    HIPE-data-v1.4-dev-fr.tsv
    HIPE-data-v1.4-test-fr.tsv
need to be stored in the folder:
    `data/data_dev/clef_hipe/`


This script produces an arrow-formatted `datasets.DatasetDict` 
  compliant with the `transformer` library data architecture.

The script produces segments based on the PySDB annotations
  and filters short (< min_len) and long (> max_len) sentences.

The resulting dataset has the following columns for each data split:
    - `tokens`: list of tokens (from the original csv file)
    - `ner`: the associated NE-COARSE-LIT labels from the original data
    - `meto`: the associated NE-COARSE-METO labels from the original data
"""

import os
import csv
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel

##

BASE_PATH = "../data_dev/clef_hipe/"
OUTPUT_FOLDER = "../hipe_ner/"
FILENAME = "prepro_hipe_fr"

##

min_len = 8
max_len = 128


if __name__ == '__main__':
    print("Preprocessing CLEF-HIPE-2020 shared task data.")
    # 1. Read csv files associated with each data split
    data_splits =["train","dev","test"]
    dfs_splits = {}

    print(f"Loading files from {BASE_PATH}")
    for split in data_splits:
        print(os.path.join(BASE_PATH, f"/HIPE-data-v1.4-{split}-fr.tsv"))
        df = pd.read_csv(
            f"{BASE_PATH}HIPE-data-v1.4-{split}-fr.tsv",
            sep="\t",
            quoting=csv.QUOTE_NONE,
            )

        df = df.dropna().reset_index()

        dfs_splits[split] = df

    # print number of samples per split (before filtering!)
    for split in data_splits:
        print(f"[Before filtering] Number of samples in the '{split:<6}': {len(dfs_splits[split]):,}")

    # 2. Segment data and filter by sentence length, store in dictionary
    dict_tokens_labels = {
        split: {"tokens": [], "ner": [], "meto": []}
        for split in data_splits
    }

    for split in data_splits:
        print(f"Preparing split {split}....")

        df = dfs_splits[split]
        n = len(df)

        segment_ids = np.flatnonzero( # extract segment based on PySDB annot
            df["MISC"].map(
                lambda m: isinstance(m, str) and "PySBDSegment" in m
            ).to_numpy()
        )

        # Extract data for each 'segment'
        for i in range(len(segment_ids)):
            if i == 0: # first/initial segment
                start, end = 0, segment_ids[i] + 1
            elif i == len(segment_ids) - 1: # last segment
                start, end = segment_ids[i] + 1, n
            else: # in between 
                start, end = segment_ids[i] + 1, segment_ids[i + 1] + 1

            seg_len = end - start

            if min_len <= seg_len <= max_len:
                seg = df.iloc[start:end]

                dict_tokens_labels[split]["tokens"].append(
                    seg["TOKEN"].astype(str).tolist()
                )
                dict_tokens_labels[split]["ner"].append(
                    seg["NE-COARSE-LIT"].astype(str).tolist()
                )
                dict_tokens_labels[split]["meto"].append(
                    seg["NE-COARSE-METO"].astype(str).tolist()
                )

    # 3. Transform dict into datasets.DatasetDict object and 
    dataset = DatasetDict({
        split: Dataset.from_dict({
            "tokens": dict_tokens_labels[split]["tokens"],
            "ner": dict_tokens_labels[split]["ner"],
            "meto": dict_tokens_labels[split]["meto"],
        })
        for split in data_splits
    })
    # Build label vocabularies across ALL splits (ensures that the same label always has the same ID.)
    def get_label_names(label_key):
        names = dict.fromkeys(
            label
            for split in data_splits
            for sequence in dict_tokens_labels[split][label_key]
            for label in sequence
        )
        return list(names)

    ner_names = get_label_names("ner")
    meto_names = get_label_names("meto")

    # create label-to-integer mappings
    ner2id = {label: i for i, label in enumerate(ner_names)}
    meto2id = {label: i for i, label in enumerate(meto_names)}

    # 3. Define the dataset schema
    features = Features({
        "tokens": Sequence(Value("string")),
        "ner": Sequence(ClassLabel(names=ner_names)),
        "meto": Sequence(ClassLabel(names=meto_names)),
    })

    # convert each split
    dataset = DatasetDict()

    for split in data_splits:
        data = dict_tokens_labels[split]

        dataset[split] = Dataset.from_dict(
            {
                "tokens": data["tokens"],
                "ner": [
                    [ner2id[label] for label in sequence]
                    for sequence in data["ner"]
                ],
                "meto": [
                    [meto2id[label] for label in sequence]
                    for sequence in data["meto"]
                ],
            },
            features=features,
        )

    # 3. Save to disk
    save_path = f"{OUTPUT_FOLDER}{FILENAME}"
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to disk at: {save_path} ✅")