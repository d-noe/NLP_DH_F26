"""
Data files to be downloaded from:
    https://zenodo.org/records/7446728
For this script, the files are to be stored in the folder:
    `data/data_dev/ANRChapitres-2000romans19e20e-ea770e4/`

This script produces an arrow-formated dataset with `N_SENTENCES` long excerpts (at the beginning of the chapter, excludign the first sentence (often chapter title)) 
from random chapters of the ANRChapitres Corpus.
The 'text' chunks are associated with various labels in addition to the `label` tag that states if a book is part of the 'canon' or not.
These metadata include:
- 'author'
- 'author_gender'
- 'book_title'
- 'publication_year'
"""

import os 
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, ClassLabel

import spacy
from tqdm import tqdm

# ============================================================
# Dirty hard-coded variables

SEED = 0
N_CHAPTERS = 10
N_SENTENCES = 5

DATA_PATH = "../data_dev/ANRChapitres-2000romans19e20e-ea770e4"
VERBOSE = False

OUTPUT_DIR = "../canon_challenge/"
DO_NOT_TELL = True

# ============================================================

# ============================================================
# ------------------------- HELPERS --------------------------
# ============================================================

def parse_tei_file(filepath):
    """
    Parse a TEI XML file and extract:
      - 'author': author name
      - 'author_gender': author gender
      - 'title': book title
      - 'publication_year': publication date
      - 'label': canon tag (or 'non-canon')
      - 'chapters': list of chapters (each as a string)
    """
    # Parse XML
    ## Older version: issue with encoding?
    # tree = ET.parse(filepath, parser = ET.XMLParser(encoding = 'iso-8859-9'))
    # root = tree.getroot()
    ## Newer version: trial from strings:
    with open(filepath, "r") as f:
        xml_str = f.read()
    f.close()
    root = ET.fromstring(xml_str)

    # Helper function for safe find
    def find_text(path):
        el = root.find(path)
        return el.text.strip() if el is not None and el.text else None

    # Extract author gender 
    author_el = root.find(".//author")
    author_gender = author_el.get("sex", "").strip() if author_el is not None else None

    ## extract author name directly from filename:
    author_name = filepath.split('/')[-1].split('_')[1].replace('-', ' ')

    #  Extract publication date 
    publication_date = int(filepath.split('/')[-1].split('_')[0])

    # Extract book title 
    ## Extract directly from filename:
    book_title = filepath.split('/')[-1].split('_')[2][:-3].replace('-', ' ')

    # Extract 'canon' or 'non-canon'
    profile_el = root.find(".//editionStmt/profileDesc[@type='genre']")
    genre_tag = profile_el.get("tag", "").strip() if profile_el is not None else None

    # Extract chapters
    chapters = []
    for div in root.findall(".//div[@type='chapter']"):
        # Collect all <p> elements inside the chapter
        paragraphs = []
        for p in div.findall(".//p"):
            if p.text and p.text.strip():
                paragraphs.append(p.text.strip())
        # Join paragraphs into one string for the chapter
        if paragraphs:
            chapters.append(" ".join(paragraphs))

    return {
        "author": author_name,
        "author_gender": author_gender,
        "book_title": book_title,
        "publication_year": publication_date,
        "label": genre_tag,
        "chapters": chapters,
    }

def xml_to_pandas(data_path=DATA_PATH):
    # retrieve files list
    xml_files = []
    for file in os.listdir(data_path):
        if file.endswith('.xml'):
            xml_files += [file]

    # parse the files
    data_dicts = [
        parse_tei_file(os.path.join(data_path, xml_f))
        for xml_f in xml_files
    ]

    # turn into dataframe
    df = pd.DataFrame(data=data_dicts)

    def str_to_year(str_year):
        try:
            return int(str_year)
        except:
            return np.nan
    
    # clean a bit
    df["publication_year"] = df["publication_year"].map(str_to_year)
    df = df.replace('', np.nan).dropna().reset_index()

    return df

def train_val_test_split_books(df, verbose=False):
    df_with_splits = df.copy()

    np.random.seed(SEED)
    rng = np.random.default_rng(seed=SEED)
    rdm_ids = rng.choice(len(df), size=len(df), replace=False)

    train_ids, tmp_ids = train_test_split(rdm_ids, test_size=0.6, random_state=SEED)
    val_ids, test_ids = train_test_split(tmp_ids, test_size=0.5, random_state=SEED)

    if verbose:
        for split_ids in [train_ids, val_ids, test_ids]:
            df_split = df.iloc[split_ids]
            print('-----')
            for gender in ["male", "female"]:
                df_gender = df_split[df_split["author_gender"]==gender]
                df_gender_canon = df_gender[df_gender['label']=='canon']
                print(f"Gender: {gender} -> {len(df_gender)/len(df_split):.3f} ({len(df_gender)} / {len(df_split)})")
                print(f"\tProp canon: {len(df_gender_canon)/len(df_gender):.3f} ({len(df_gender_canon)} / {len(df_gender)})")


    splits = np.zeros(len(df)).astype(str)
    splits[train_ids] = "train"
    splits[val_ids] = "validation"
    splits[test_ids] = "test"

    df_with_splits["split"] = splits

    return df_with_splits

def extract_text_chunks(df_splits, n_chapters=N_CHAPTERS, n_sentences=N_SENTENCES, verbose=False):
    # Load a blank French model (no parser, tagger, etc.)
    nlp = spacy.blank("fr")
    # Add the simple rule-based sentencizer
    nlp.add_pipe("sentencizer")

    real_n_chapters = []

    chapters_data = {
        split: {
            c: []
            for c in list(df_splits.columns)+["text"] if not (c=='split' or c=='index' or c=="chapters")
        }
        for split in ["train", "validation", "test"]
    }

    for i, row in tqdm(df_splits.iterrows(), "Extracting Chapters' Chunks...."):
        curr_split = row['split']

        curr_chapters = row["chapters"]
        curr_n_chapters = min([n_chapters, len(curr_chapters)])

        if curr_n_chapters > 0:
            curr_chapter_ids = np.arange(len(curr_chapters))
            np.random.seed(seed=SEED+i)
            np.random.shuffle(curr_chapter_ids)
            rdm_chapter_ids = curr_chapter_ids[:n_chapters]

            for chap in [curr_chapters[i] for i in rdm_chapter_ids]:
                doc = nlp(chap)
                chapters_data[curr_split]["text"] += [' '.join([s.text for s in list(doc.sents)[1:1+n_sentences]])]

            real_n_chapters += [curr_n_chapters]

            for k in chapters_data[curr_split].keys():
                if not k == "text":
                    chapters_data[curr_split][k] += [row[k]]*curr_n_chapters

    return chapters_data, real_n_chapters

# ============================================================
# --------------------------- MAIN ---------------------------
# ============================================================


def main():
    # 1. Parse the xml files
    df_books = xml_to_pandas(DATA_PATH)
    # 2. Create partitions in the data at the book level
    df_splits = train_val_test_split_books(df_books, verbose=VERBOSE)
    # 3. Extract the chunks (max. n_sentences long excerpts from max. n_chapters per book) 
    chapters_data, _ = extract_text_chunks(df_splits, n_chapters=N_CHAPTERS, n_sentences=N_SENTENCES, verbose=VERBOSE)
    # 4. Turn into a Dataset object
    dataset = DatasetDict(
        {
            s: Dataset.from_dict(chapters_data[s])
            for s in ["train", "validation", "test"]
        }
    )
    ## Encode columns
    dataset = dataset.class_encode_column("label")
    dataset = dataset.class_encode_column("author_gender")

    if not DO_NOT_TELL:
        do_not_tell_path = os.path.join(OUTPUT_DIR, "dev_secret_dataset")
        dataset.save_to_disk(do_not_tell_path)
        print(f"🤫 Secret dataset saved at: {do_not_tell_path}.")

    ## Keep the labels secret for test set!!
    dataset["test"] = dataset["test"].remove_columns(["label"])
    # Save to disk
    save_path = os.path.join(OUTPUT_DIR, "dataset")
    dataset.save_to_disk(save_path)
    print(f"🎈 Final dataset saved at: {save_path} ✅")

    return 


if __name__ == "__main__":
    main()
