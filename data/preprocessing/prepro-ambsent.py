"""
Data files to be downloaded from:
    https://osf.io/x687m/files/vjkdx
For this script, the file `AmbiSent_Minimal_n53.csv` are to be stored in the folder:
    `data/data_dev/`

This script produces a .csv file that samples sentences with annotation percentages per category
based on an "annotation entropy" heuristic.
"""

import os
import numpy as np
import pandas as pd

BASE_PATH = "../data_dev/AmbiSent_Minimal_n53.csv"
OUTPUT_FOLDER = "../annotation_data/"

n_top_entropy = 15
n_bottom_entropy = 15

SEED = 100-8

CATEGORIES = ["Positive", "Negative", "Mixed"]

def get_entropy(row, cols=CATEGORIES):
    """
    Compute the annotation entropy for the given 'cols'
    values in 'cols' need to be frequencies/probabilities
    """
    p_i = np.array([row[c] for c in cols if row[c]>0])
    return -np.sum(p_i*np.log(p_i))

if __name__ == '__main__':
    print("Sampling from AmbiSent dataset — based on annotation entropy heuristic.")
    # load the original dataset
    df = pd.read_csv(BASE_PATH)

    # Compute entropy of the annotations
    df["entropy"] = df.apply(get_entropy, axis=1)

    # Sample sentences associated with top and bottom annotation entropy values
    np.random.seed(SEED)
    df_sampled = pd.concat(
        [
            df.sort_values(by="entropy", ascending=False).head(n_top_entropy), # highest entropy
            df.sort_values(by="entropy", ascending=False).tail(n_bottom_entropy) # lowest entropy
        ]
    ).sample(frac=1).reset_index(drop=True) # shuffle and reset index
    df_sampled["Sentence_id"] = np.arange(len(df_sampled)) # re-index sampled data

    # Select only columns of interest
    df_sampled = df_sampled[["Sentence_id", "Sentence", "Positive", "Negative", "Mixed", "Neutral", "Correct", "Sentence_type"]]

    # keep only sentence (& id) for the annotation exercise
    df_to_annot = df_sampled[["Sentence_id", "Sentence"]]

    df_sampled.to_csv(os.path.join(OUTPUT_FOLDER, "sentences_with_annot.csv"), index=False)
    df_to_annot.to_csv(os.path.join(OUTPUT_FOLDER, "sentences_to_annot.csv"), index=False)

    print(f"Pre-processed files saved at: {OUTPUT_FOLDER}.\n\t- 'sentences_with_annot.csv': Original samples from the dataset.\n\t- 'sentences_to_annot.csv': Sentences without annotations.")