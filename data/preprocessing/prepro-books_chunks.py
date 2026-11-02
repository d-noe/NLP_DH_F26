"""
This script filters book chunks from HuggingFace's dataset: Despina/project_gutenberg
    reducing the number of chunks from ~6M to ~20K
    it is implemented to have chunks balanced between selected genres (/ 'topics')
    and 'author_gender' has much as possible (but heavily leans towards 'male'-written works)
    It saves the filtered dataset into a DatasetDict object with train/validation/test splits.

SAVED OUTPUTS:
    - `retained_books.csv`:
        csv file with metadata related to the books retained in the output dataset
    - `sampled_chunks`: 
        Arrow-formated DatasetDict (train/validation/test splits) with columns:
            - book_id: book identifier
            - chunk_id: chunk identifier
            - text: actual book chunk
            - label: encoded book genre label
            - author_gender: encoded inferred (binary) author gender
    - `sampled_chunks.csv`:
        csv file that concatenates the splits of the dataset (same columns + 'split')
    - `sampled_chunks.zip`: (if `DO_ZIP` set to `True`):
        zipped version of `sampled_chuks` dataset

Note: to prevent important data leakage, there cannot be chunks from the same book in different data splits.

/!\ Disclaimer: part of this code was co-written with ChatGPT
"""
import os
import sys
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel

# ------------------------------------------------------------
# Configuration (hard-coded for now)
# ------------------------------------------------------------
DATASET_NAME = "Despina/project_gutenberg"  # e.g. "cnn_dailymail"
DATASET_SUBSET_1 = "fiction_books"
DATASET_SUBSET_2 = "fiction_books_in_chunks"

OUTPUT_DIR = "../literary_sft/"
DO_ZIP = False

SUBSET_SELECTED_TOPICS = [
    'detective and mystery stories',
    'science fiction',
    'adventure stories',
    "children's stories",
]

SEED = 42
# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def confirm_download():
    """Ask user for confirmation before downloading large datasets."""
    print(
        "⚠️  This script will download more than 3GB of data from the Hugging Face Hub.\n"
        "Make sure you have a stable internet connection and enough disk space.\n"
    )
    user_input = input("Do you want to continue? [y/n]: ").strip().lower()
    if user_input not in ("y", "yes"):
        print("Aborting...")
        sys.exit(0)

def zip_dir(src_dir):
    src_dir = Path(src_dir)
    zip_path = src_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in src_dir.rglob("*"):
            z.write(file, file.relative_to(src_dir))
    return zip_path

# ------------------------------------------------------------
# SELECTION & FILTERING
# ------------------------------------------------------------
def select_books_topics(
    dataset, # DatasetDict (only 'train')
    subset_selected_topics:list=SUBSET_SELECTED_TOPICS,
):
    df = dataset["train"].to_pandas()[[
        'book_id', 
        'title', 
        'author', 
        'author_gender', 
        'author_birth_year', 
        'author_death_year', 
        'pg_subjects', 
        'topics', 
    ]]
    # keep only rows with at least one topic in subset
    mask_subset = [
        np.any([selected_t in row_topics for selected_t in subset_selected_topics])
        for row_topics in df["topics"]
    ]
    df_subset = df[mask_subset]

    # keep only rows with non overlapping topics from subset
    mask_unique_selected = [
        np.sum([selected_t in t for selected_t in subset_selected_topics])==1 # keep only samples with only **one** topic from the selected subset
        for t in df_subset["topics"] 
    ]
    df_subset_unique = df_subset[mask_unique_selected]
    ## convert multi topics into single topic from subset
    df_subset_unique["topics"] = [
        t[np.argmax([_t in subset_selected_topics for _t in t])] # only 1 is True
        for t in df_subset_unique['topics']
    ]
    df_subset_unique = df_subset_unique.rename({"topics":"topic"}, axis=1)

    return df_subset_unique

def sample_books(
    books_df,
    target_per_topic = 250,
):
    def sample_topic(group):
        """Sample up to target_per_topic books per topic,
        balancing genders and avoiding author duplication when possible."""
        
        n_total = min(len(group), target_per_topic)
        n_female_target = n_total // 2
        n_other_target = n_total - n_female_target
        
        # Split by gender
        females = group[group['author_gender'].str.lower() == 'female']
        others  = group[group['author_gender'].str.lower() != 'female']
        
        def sample_gender_subset(subset, target):
            """Try to sample one book per author if possible; otherwise allow repeats."""
            # First, one per author
            unique_authors = subset.drop_duplicates(subset=['author'], keep='first')
            n_available = len(unique_authors)
            if n_available >= target:
                return unique_authors.sample(n=target, random_state=SEED)
            else:
                # Take all unique authors first
                sampled = unique_authors.copy()
                remaining_n = target - n_available
                
                # Sample extra books (even if from same author) to fill up
                remaining_pool = subset.loc[~subset['book_id'].isin(sampled['book_id'])]
                if len(remaining_pool) > 0:
                    extra = remaining_pool.sample(
                        n=min(remaining_n, len(remaining_pool)), random_state=SEED
                    )
                    sampled = pd.concat([sampled, extra])
                return sampled
        
        # Gender-balanced sampling
        sampled_females = sample_gender_subset(females, n_female_target)
        sampled_others  = sample_gender_subset(others, n_other_target)
        
        sampled = pd.concat([sampled_females, sampled_others])
        
        # If we still don't reach n_total, top up from remaining pool (any author/gender)
        if len(sampled) < n_total:
            remaining_pool = group.loc[~group['book_id'].isin(sampled['book_id'])]
            if len(remaining_pool) > 0:
                top_up = remaining_pool.sample(
                    n=min(len(remaining_pool), n_total - len(sampled)), random_state=SEED
                )
                sampled = pd.concat([sampled, top_up])
        
        return sampled

    # Apply to each topic
    sampled_df = books_df.groupby('topic', group_keys=False).apply(sample_topic)

    # Extract retained book IDs
    retained_book_ids = sampled_df['book_id'].tolist()

    # Optional summary
    print(f"Total retained books: {len(retained_book_ids)}")
    print(sampled_df['topic'].value_counts())

    return retained_book_ids

def sample_book_chunks(
    dataset,
    selected_ids,
    target_total = 20000,  # aim between 10k–20k
):
    # Filter chunk dataset based on retained book ids
    filtered_dataset = dataset["train"].filter(lambda ex: ex['book_id'] in selected_ids)

    # --- 1. Determine sampling target ---
    book_ids = filtered_dataset.unique('book_id')
    num_books = len(book_ids)
    chunks_per_book_target = max(1, target_total // num_books)

    print(f"Targeting about {chunks_per_book_target} chunks per book across {num_books} books.")

    # --- 2. Sample chunks per book (equal sampling) ---
    # Convert only book_id and indices to DataFrame for efficient grouping
    df_index = pd.DataFrame({
        "idx": range(len(filtered_dataset)),
        "book_id": filtered_dataset["book_id"],
    })

    # Group by book_id and sample
    def sample_indices_per_book(group):
        n_to_sample = min(len(group), chunks_per_book_target)
        return group.sample(n=n_to_sample, random_state=SEED)["idx"].tolist()

    sampled_indices = (
        df_index.groupby("book_id", group_keys=False)
        .apply(sample_indices_per_book)
        .explode()
        .astype(int)
        .tolist()
    )

    # Subset the dataset
    sampled_dataset = filtered_dataset.select(sampled_indices)
    print(f"Sampled {len(sampled_dataset)} chunks total.")

    return sampled_dataset

def split_sampled_chunks(
    dataset,
):
    # --- 1. Prepare book-level table ---
    # Extract per-book metadata
    book_df = pd.DataFrame({
        "book_id": dataset["book_id"],
        "topics": dataset["topics"],
        "author_gender": dataset["author_gender"],
    })

    # Remove potential duplicates (one row per book)
    book_df = book_df.drop_duplicates(subset=["book_id"]).reset_index(drop=True)

    # --- 2. Prepare stratification key (topic + gender) ---
    book_df["strat_key"] = book_df["topics"].astype(str) + "_" + book_df["author_gender"].astype(str)

    # Identify strata that are too small to stratify
    counts = Counter(book_df["strat_key"])
    too_small_keys = {k for k, v in counts.items() if v < 2}
    print(f"Found {len(too_small_keys)} small strata (<2 books). They will be grouped under 'other'.")

    book_df["strat_key_fixed"] = np.where(book_df["strat_key"].isin(too_small_keys),
                                        "other", book_df["strat_key"])

    # --- 3. Stratified split at the book level ---
    try:
        train_books, temp_books = train_test_split(
            book_df,
            test_size=0.2,
            stratify=book_df["strat_key_fixed"],
            random_state=SEED,
        )
    except ValueError:
        print("⚠️ Falling back to random split (too few strata).")
        train_books, temp_books = train_test_split(book_df, test_size=0.2, random_state=SEED)

    try:
        val_books, test_books = train_test_split(
            temp_books,
            test_size=0.5,
            stratify=temp_books["strat_key_fixed"],
            random_state=SEED,
        )
    except ValueError:
        print("⚠️ Falling back to random val/test split.")
        val_books, test_books = train_test_split(temp_books, test_size=0.5, random_state=SEED)

    # --- 4. Build lookup sets of book_ids per split ---
    train_book_ids = set(train_books["book_id"])
    val_book_ids = set(val_books["book_id"])
    test_book_ids = set(test_books["book_id"])

    # --- 5. Filter chunks by book_id ---
    def filter_by_book(example, allowed_ids):
        return example["book_id"] in allowed_ids

    train_dataset = dataset.filter(lambda e: e["book_id"] in train_book_ids)
    val_dataset = dataset.filter(lambda e: e["book_id"] in val_book_ids)
    test_dataset = dataset.filter(lambda e: e["book_id"] in test_book_ids)

    # --- 6. Build DatasetDict ---
    dataset_splits = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })

    # --- 7. Summary ---
    print(f"Books per split -> Train: {len(train_books)}, Val: {len(val_books)}, Test: {len(test_books)}")
    print(f"Chunks per split -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return dataset_splits

def clean_chunks_dataset(
    dataset_splits,
    books_df,
):
    # Associate with unique topic 
    dict_book_to_topic = {
        row['book_id']: row['topic']
        for _, row in books_df.iterrows()
    }
    def book_id_to_topic(example):
        example["topic"] = dict_book_to_topic[example['book_id']]
        return example

    # Tokenize the texts
    dataset_splits = dataset_splits.map(book_id_to_topic)

    # Remove | Rename some columns
    dataset_splits = dataset_splits.remove_columns(['title', 'author', 'author_birth_year', 'author_death_year', 'release_date', 'pg_subjects', 'topics'])
    
    dataset_splits = dataset_splits.rename_column("chunk", "text")
    dataset_splits = dataset_splits.rename_column("topic", "label")

    # Convert to ClassLabel
    # Suppose you have label strings under "topics"
    dataset_splits = dataset_splits.class_encode_column("label")
    dataset_splits = dataset_splits.class_encode_column("author_gender")


    return dataset_splits


# ------------------------------------------------------------
# Main execution flow
# ------------------------------------------------------------

def main():
    # 0. Confirm proceed
    confirm_download()
    # 1. Filter at the book level
    books_dataset = load_dataset(DATASET_NAME, DATASET_SUBSET_1)
    books_dataset = books_dataset.remove_columns("text") # we wont be using the 'text' column 
    df_subset_books = select_books_topics(books_dataset)

    # 2. Random sample from the filtered subset while trying to balance: 'topic' (book genre) and 'author_gender' (inferred, binary)
    retained_book_ids = sample_books(df_subset_books)

    ## Keep only retained book (after sampling)
    mask_retained = [_id in retained_book_ids for _id in selected_ids_df["book_id"]]
    retained_book_ids_df = df_subset_books[mask_retained]
    
    ## Save for metadata
    retained_book_ids_df.to_csv(os.path.join(OUTPUT_DIR, "retained_books.csv"), index=False)

    # 3. Filter books chunks 
    chunks_dataset = load_dataset(DATASET_NAME, DATASET_SUBSET_2)
    sampled_chunks_dataset = sample_book_chunks(
        chunks_dataset,
        selected_ids=retained_book_ids,
    )
    dataset_splits = split_sampled_chunks(
        sampled_chunks_dataset,
    )
    dataset_splits = clean_chunks_dataset(
        dataset_splits,
        books_df=retained_book_ids_df,
    )

    ## SAVE TO DISK!
    save_dataset_path = os.path.join(OUTPUT_DIR, 'sampled_chunks')
    dataset_splits.save_to_disk(save_dataset_path)
    print(f"✅ Final dataset saved at: {save_dataset_path} 🎈")

    ### Also save as concatenated csv
    splits = ["train", "validation", "test"]
    splits_dfs = []

    for s in splits:
        s_df = dataset_splits[s].to_pandas()
        s_df["split"] = s
        splits_dfs += [s_df]

    df = pd.concat(splits_dfs)
    csv_save_path = os.path.join(OUTPUT_DIR, "sampled_chunks.csv")
    df.to_csv(csv_save_path, index=False)

    ### Also save as zip
    if DO_ZIP:
        zip_dir(src_dir=os.path.join(OUTPUT_DIR, "sampled_chunks"))


if __name__ == "__main__":
    main()