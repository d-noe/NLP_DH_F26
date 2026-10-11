"""
Data files to be downloaded from:
    https://bl.iro.bl.uk/concern/datasets/1a677294-cbd3-4bc0-b714-d3bbfd2a6da1
For this script, the files are to be stored in the folder:
    `data/data_dev/living_with_machines/`

This script produces a .csv file that compiles sources and annotations for the different target words included 
in the Living With Machines annotation campaign.
"""

import os
import pandas as pd
from collections import Counter

# Hard-coded path variables
DATA_PATH = "../data_dev/living_with_machines/"
OUTPUT_FLNM = "../word_sense/lwm_annot.csv"

WORD2FLNM = {
    "car":"combined-23628-car-classifications.csv",
    "bike":"combined-23672-bike-classifications.csv",
    "trolley":"combined-23452-trolley-classifications.csv",
    "coach":"combined-23681-coach-classifications.csv",
}

kept_columns = {
    "!text": "text",
    "T0_task_value": "annot",
    "newspaper_date": "newspaper_date",
    "newspaper_title": "newspaper_title",
    "newspaper_place": "newspaper_place",
    "locations_list": "image_url",
}

# utils

def deduplicate_annotations(df):
    all_columns = df.columns.tolist()
    key_col = "text"
    annot_col = "annot"

    dedup_records = []

    for text, group in df.groupby(key_col):
        annotations = group[annot_col].dropna().astype(str).tolist()
        counts = Counter(a.strip() for a in annotations)
        most_common_label, count = counts.most_common(1)[0]

        # handle ties deterministically (e.g., alphabetical order)
        max_count = count
        tied_labels = [label for label, c in counts.items() if c == max_count]
        if len(tied_labels) > 1:
            most_common_label = sorted(tied_labels)[0]

        # build base record
        record = {key_col: text, annot_col: most_common_label}

        # preserve other columns
        for col in all_columns:
            if col in [key_col, annot_col]:
                continue

            unique_vals = group[col].dropna().unique()
            if len(unique_vals) == 1:
                record[col] = unique_vals[0]
            elif len(unique_vals) == 0:
                record[col] = None
            else:
                # fallback: pick first, but note conflict
                record[col] = unique_vals[0]
                record[f"{col}_conflict_values"] = list(unique_vals)

        dedup_records.append(record)

    return pd.DataFrame(dedup_records)


if __name__ == '__main__':
    print("Preprocessing Living With Machines Data")
    word_dfs = []
    for focus_word in WORD2FLNM.keys(): # for each annotation file
        # read data
        df = pd.read_csv(os.path.join(DATA_PATH, WORD2FLNM[focus_word]))
        # drop N/A values on subset
        df = df.dropna(axis=0, subset=["!text", "locations_list"])

        # select columns of interest
        df_filtered = df[kept_columns.keys()].rename(columns=kept_columns)
        df_filtered["target"] = focus_word
        df_filtered["year"] = df_filtered["newspaper_date"].apply(lambda d: int(d[:4]))

        # keep only one annotation (majority vote) per sample
        df_filtered = deduplicate_annotations(df_filtered)
        # filter by annotations
        not_kept_labels = [
            'The image is too hard to read',
            f"The highlighted word is not '{focus_word}'",
        ]

        df_filtered = df_filtered[[not (l in not_kept_labels) for l in df_filtered["annot"]]]

        print(focus_word, len(df_filtered))

        word_dfs += [df_filtered]

    ov_df = pd.concat(word_dfs)
    ov_df.to_csv(OUTPUT_FLNM, index=False)
    print(f"Pre-processed file saved at: {OUTPUT_FLNM}.")