"""
Data files to be downloaded from:
    https://github.com/danlou/bert-disambiguation/tree/master/data
For this script, the files are to be stored in the folder:
    `data/data_dev/danlou bert-disambiguation master data/`

This script produces a .csv file that compiles texts, targets and annotations 
for the CoarseWSD-20 data
"""

import os
import json
import pandas as pd

#
BASE_PATH = "../data_dev/danlou bert-disambiguation master data/"
DATA_PATH = "../data_dev/danlou bert-disambiguation master data/CoarseWSD-20/"

OUTPUT_FLNM = "../word_sense/CoarseWSD-20.csv"

wsd_words = [
    'apple',
    'club',
    'bow',
    'bank',
    'crane',
    'square',
    'chair',
    'java',
    'bass',
    'seal',
    'pitcher',
    'arm',
    'mole',
    'pound',
    'deck',
    'trunk',
    'spring',
    'hood',
    'yard',
    'digit'
]
#

def load_coarse_senses(
    base_path:str=BASE_PATH,
):
    senses = defaultdict(list)
    with open(os.path.join(base_path, 'senses.tsv')) as senses_f:
        for line in senses_f:
            amb_word, sense = line.strip().split('\t')
            senses[amb_word].append(sense)
    return dict(senses)

def get_word_classes(
    wordfolder_path:str,
):
    with open(os.path.join(wordfolder_path, "classes_map.txt")) as classes_json_f:
        word_classes = json.load(classes_json_f)
    return word_classes

def get_sentences_labels_list(
    wordfolder_path:str,
    split:str,
):
    sentence_file_path = os.path.join(wordfolder_path, f"{split}.data.txt")
    label_file_path = os.path.join(wordfolder_path, f"{split}.gold.txt")

    with open(sentence_file_path, 'r') as f:
        sentences = f.readlines()
    f.close()

    with open(label_file_path, 'r') as f:
        labels = f.readlines()
    f.close()

    sentences = [s.split("\t")[1].strip() for s in sentences]
    labels = [l.strip() for l in labels]

    return sentences, labels

def df_from_wordfolder(
    wordfolder:str,
    setname="CoarseWSD-20",
    #senses_dict:dict=None,
    base_path = BASE_PATH,
):
    folder_path = os.path.join(base_path, setname, wordfolder)
    word_classes = get_word_classes(folder_path)
    # if senses_dict is None:
    #     senses_dict = load_coarse_senses()

    sentences, labels, splits =  [], [], []
    for split in ["train", "test"]:
        split_sentences, split_labels = get_sentences_labels_list(folder_path, split=split)
        sentences += split_sentences
        labels += split_labels
        splits += [split]*len(split_labels)

    df = pd.DataFrame()

    df["text"] = sentences
    df["label"] = [int(l) for l in labels]
    df["label_sense"] = [word_classes[l] for l in labels]
    df["split"] = splits
    df["target"] = wordfolder

    return df



if __name__ == '__main__':
    print("Preprocessing CoarseWSD-20")

    single_word_dfs = []
    for w in wsd_words:
        single_word_dfs += [df_from_wordfolder(wordfolder=w)]

    ov_df = pd.concat(single_word_dfs)
    ov_df.to_csv(OUTPUT_FLNM, index=None)
    
    print(f"Pre-processed file saved at: {OUTPUT_FLNM}.")