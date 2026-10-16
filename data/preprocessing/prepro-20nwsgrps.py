import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
# 20 Newsgroups desc.: http://qwone.com/~jason/20Newsgroups/

OUTPUT_FLNM = "../topic_data/4newsgroups.csv"

if __name__ == '__main__':
    print("Filtering 20 News groups")
    # load 20 NewsGroup (train) from sklearn
    newsgroup_corpus = fetch_20newsgroups(subset='train',  remove=('headers', 'footers', 'quotes'))

    # Get the texts and sources (labels here)
    docs = newsgroup_corpus["data"]
    labels = newsgroup_corpus["target"]

    # Filter down by hand-picked targets
    id_to_target = {
        i: t
        for i, t in enumerate(newsgroup_corpus["target_names"])
    }
    labels = [id_to_target[l] for l in labels]

    selected_targets = [
        "talk.religion.misc",
        "sci.med",
        "talk.politics.misc",
        "rec.motorcycles",
    ]

    target_mask = [l in selected_targets for l in labels]

    filtered_labels = np.array(labels)[target_mask]
    filtered_docs = np.array(docs)[target_mask]

    # store in a dataframe
    df = pd.DataFrame()
    df["text"] = filtered_docs
    df["label"] = filtered_labels

    # some more filtering (don't keep empty or single-word documents)
    df = df[[len(t.strip().split())>1 for t in df["text"]]]

    df.to_csv(OUTPUT_FLNM, index=False)
    print(f"Pre-processed file saved at: {OUTPUT_FLNM}.")


        